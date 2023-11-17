import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixNet(nn.Module):
    # QMixNet is the mixer network that takes the Q-values of all agents and the global state as input
    # and outputs the global Q-value for the joint actions of all agents
    # That is:
    # Input: (Q_a1, Q_a2, ..., Q_an, s_t)
    # Output: Q_tot (s_t)
    def __init__(self, 
        n_agents,
        state_shape,
        mixing_embed_dim = 32,
        hypernet_layers = 2,
        hypernet_embed = 64,
    ):
        super(QMixNet, self).__init__()
        
        # Here we just set the input dimensions, self-explanatory
        self.n_agents = n_agents
        self.state_dim = int(np.prod(state_shape))

        self.embed_dim = mixing_embed_dim

        if hypernet_layers == 1:
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
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.output = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        # "bs" is the batch size
        # Analogous to DQN, we pass batches of inputs to the network
        # for optimization, but we also need to be able to pass single
        # inputs to the if we are choosing actions
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.output(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot