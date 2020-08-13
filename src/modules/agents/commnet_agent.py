
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

        neighbor = th.ones(inputs.size(0) // self.n_agents, self.n_agents, self.n_agents, device=inputs.device) / self.n_agents

        # communicate k step 
        h = th.tensor(h_rnn)
        c = th.zeros_like(h)
        for k in range(self.comm_step):
            h_comm = h.reshape(-1, self.n_agents, self.args.rnn_hidden_dim) # [-1, n_agents, n_feats]
            c = th.matmul(neighbor, h_comm)  # [-1, n_agents, n_feats]
            c = c.reshape(-1, self.args.rnn_hidden_dim)

            h = self.comm(h, c)

        q = self.fc2(h)
        return q, h_rnn

