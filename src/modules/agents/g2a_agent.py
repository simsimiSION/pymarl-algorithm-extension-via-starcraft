import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class G2AAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(G2AAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        self.hard_bi_gru = nn.GRU(args.rnn_hidden_dim*2, args.rnn_hidden_dim, bidirectional=True)
        self.hard_classifier = nn.Linear(args.rnn_hidden_dim*2, 2)

        self.q = nn.Linear(args.rnn_hidden_dim, args.n_attention, bias=False)
        self.k = nn.Linear(args.rnn_hidden_dim, args.n_attention, bias=False)

        self.fc2 = nn.Linear(args.rnn_hidden_dim * 2, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        bz = inputs.size(0)
        input_x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h_rnn = self.rnn(input_x, h_in) # [-1, n_feats]

        # ---------------------------------------------------
        #               hard attention setting
        # ---------------------------------------------------
        # set hard attention input vector
        select_ind = th.tensor([j for i in range(self.n_agents) for j in range(self.n_agents) if i != j])
        if self.args.use_cuda:
            select_ind = select_ind.cuda()
        select_ind = select_ind.reshape(self.n_agents, self.n_agents - 1).unsqueeze(0).unsqueeze(-1)

        input_hard = h_rnn.reshape(-1, self.n_agents, self.args.rnn_hidden_dim)
        input_hard = input_hard.unsqueeze(-2)                                 # [-1, n_agents, 1, n_feats]
        input_hard_repeat = input_hard.repeat(1, 1, self.n_agents, 1)         # [-1, n_agents, n_agent, n_feats]
        input_hard_repeat_t = input_hard_repeat.transpose(1, 2)               # [-1, n_agents, n_agent, n_feats]
        input_hard = th.cat([input_hard_repeat, input_hard_repeat_t], dim=-1) # [-1, n_agents, n_agent, n_feats*2]

        input_hard = input_hard.gather(2, select_ind.repeat(bz // self.n_agents, 1, 1, self.args.rnn_hidden_dim*2)) # [-1, n_agents, n_agent-1, n_feats*2]
        input_hard = input_hard.permute(2, 0, 1, 3).reshape(self.n_agents-1, -1, self.args.rnn_hidden_dim*2)

        h_gru = th.zeros(2*1, bz, self.args.rnn_hidden_dim)
        if self.args.use_cuda:
            h_gru = h_gru.cuda()
        if self.args.use_cuda:
            h_gru.cuda()

        h_hard, _ = self.hard_bi_gru(input_hard, h_gru)                       # [n_agent-1, batchsize * n_agents, n_feats*2]
        h_hard = h_hard.permute(1, 0, 2)                                      # [batchsize * n_agents, n_agent-1, n_feats*2]
        h_hard = h_hard.reshape(-1, self.args.rnn_hidden_dim*2)

        hard_weight = self.hard_classifier(h_hard)
        hard_weight = F.gumbel_softmax(hard_weight, tau=self.args.gumbel_softmax_tau)
        hard_weight = hard_weight[:, 1].reshape(-1, self.n_agents, self.n_agents-1, 1) # [-1, n_agents, n_agents-1, 1]

        # ---------------------------------------------------
        #               soft attention setting
        # ---------------------------------------------------
        q = self.q(h_rnn).reshape(-1, self.n_agents, 1, self.args.n_attention) # [-1, n_agents, 1, n_feats]
        k = self.k(h_rnn).reshape(-1, self.n_agents, 1, self.args.n_attention).repeat(1, 1, self.n_agents, 1)
        k = k.gather(2, select_ind.repeat(bz // self.n_agents, 1, 1, self.args.n_attention))     # [-1, n_agents, n_agents-1, n_feats]

        # hard attention first
        k = hard_weight * k                                                    # [-1, n_agents, n_agents-1, n_feats]
        k = k.permute(0, 1, 3, 2)                                              # [-1, n_agents, n_feats, n_agents-1]
        raw_soft_weight = th.matmul(q, k)                                      # [-1, n_agents, 1, n_agents-1]
        scalar_raw_soft_weight = raw_soft_weight / np.sqrt(self.args.n_attention)
        soft_weight = F.softmax(scalar_raw_soft_weight, dim=-1)
        soft_weight = soft_weight.reshape(-1, self.n_agents, self.n_agents-1, 1) # [-1, n_agents, n_agents-1, 1]

        x = h_rnn.reshape(-1, self.n_agents, self.args.rnn_hidden_dim).unsqueeze(-2)
        x = x.repeat(1, 1, self.n_agents, 1)                                   # [-1, n_agents, n_agents, n_feats]
        x = x.gather(2, select_ind.repeat(bz // self.n_agents, 1, 1, self.args.rnn_hidden_dim))  # [-1, n_agents, n_agents-1, n_feats]

        weight = (hard_weight * soft_weight).reshape(-1, self.n_agents, 1, self.n_agents-1) # [-1, n_agents, 1, n_agents-1]
        x = weight @ x                                                         # [-1, n_agents, 1, n_feats]

        x = th.cat([h_rnn, x.reshape(-1, self.args.rnn_hidden_dim)], dim=-1)   # [-1, n_feats]
        q = self.fc2(x)

        return q, h_rnn


