
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class CommAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CommAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.n_agents = args.n_agents
        self.comm_step = args.comm_step

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.comm = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # inputs: [bz * n_agent, -1]
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h_rnn = self.rnn(x, h_in)

        # get neighbor to communicate
        neighbor = self._get_neighbor(inputs)
        n_neighbor = neighbor.sum(-1).unsqueeze(-1) # [-1, 1]
        neighbor = neighbor.reshape(-1, self.n_agents, self.n_agents) # [-1, n_agents, n_agents]

        # communicate k step
        h = th.tensor(h_rnn)
        c = th.zeros_like(h)
        for k in range(self.comm_step):
            h_comm = h.reshape(-1, self.n_agents, self.args.rnn_hidden_dim) # [-1, n_agents, n_feats]
            c = th.matmul(neighbor, h_comm)  # [-1, n_agents, n_feats]
            c = c.reshape(-1, self.args.rnn_hidden_dim)
            c_scalar = c / n_neighbor

            h = self.comm(h, c_scalar)

        q = self.fc2(h)
        return q, h_rnn

    def _get_neighbor(self, obs):
        """Treat agents that can be seen in the field of vision as neighbors
        """
        start_ind = self.args.move_feats_dim + self.args.enemy_feats_dim
        end_index = start_ind + self.args.ally_feats_dim
        stride = self.args.ally_feats_dim // (self.n_agents-1)

        with th.no_grad():
            # shape [bz * n_agents, n_agents-1]
            neightbor = obs[:, start_ind:end_index:stride]
            # shape [bz, n_agents, n_agents-1]
            neightbor_reshape = neightbor.reshape(-1, self.n_agents, self.n_agents-1)

            zeros = th.zeros((neightbor_reshape.size(0), 1))
            neightbors = []
            for i in range(self.n_agents):
                left, right = neightbor_reshape[:, i].split([i, self.n_agents - i - 1], dim=-1)

                neightbor = th.cat([left, zeros, right], dim=-1)
                neightbors.append(neightbor)

            neightbors = th.stack(neightbors, dim=0)
            neightbors = neightbors.reshape(self.n_agents, -1, self.n_agents)
            neightbors = neightbors.permute([1, 0, 2])
            neightbors = neightbors.reshape(-1, self.n_agents)
        return neightbors









