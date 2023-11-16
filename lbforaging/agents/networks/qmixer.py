import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QMixer(nn.Module):
    def __init__(self, 
        n_agents,
        state_shape,
        mixing_embed_dim = 32,
        hypernet_layers = 1,
        hypernet_embed = 64,
    ):
        super(QMixer, self).__init__()
        
        self.n_agents = n_agents
        self.state_dim = int(np.prod(state_shape))

        self.embed_dim = mixing_embed_dim

        if hypernet_layers == 1:
            # receives state (63) and outputs a vector of size 32*n_agents
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif hypernet_layers == 2:
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif hypernet_layers > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        # receives state (63) and outputs a vector of size 32
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        agent_qs = torch.stack(agent_qs, dim=0)
        # print("original agent qs", agent_qs)
        if (agent_qs.dim() == 2):
            # then we're in eval mode, so we need to add a batch dimension
            agent_qs = agent_qs.unsqueeze(0)
        # batch size
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        # agent_qs = agent_qs.view(-1, 1, self.n_agents)
        agent_qs = agent_qs.view(1, -1, self.n_agents)
        # First layer
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        # print("agent_qs", agent_qs)
        # print("agent_qs.shape", agent_qs.shape)
        # print("w1.shape", w1.shape)
        # print("b1.shape", b1.shape)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot