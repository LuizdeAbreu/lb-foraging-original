import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, hidden_dim = 128):
        super(DQN, self).__init__()
        # Standard DQN architecture:
        # DQN takes the observations as input and outputs the Q-values for each action in the given state
        # 1 input layer, 1 hidden layer with 128 neurons and 1 output layer with n_actions neurons
        self.layer1 = nn.Linear(n_observations, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        # Called with either one element to determine next action, or a batch
        # during optimization. Returns Q-values for each action;
        # Q_a (o_t) where o_t is the observation at time t
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
