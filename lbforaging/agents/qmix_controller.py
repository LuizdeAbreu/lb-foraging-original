import random
import math
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from lbforaging.foraging.agent import Agent
from lbforaging.agents.networks.dqn import DQN
from lbforaging.foraging.environment import Action, ForagingEnv as Env
from lbforaging.agents.helpers import ReplayMemory, Transition
from lbforaging.agents.networks.qmixer import QMixer

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QMIX_Controller(Agent):
    name = "QMIX Controller"

    def __init__(self, player):
        super().__init__(player)

        self.policy_net = None
        self.target_net = None
        self.memory = None
        self.optimizer = None
        self.previous_obs = None
        self.previous_action = None
        self.steps_done = 0
        self.players = [player]

    def add_player(self, player):
        self.players.append(player)

    def init_from_env(self, env):
        if (len(env.players) != len(self.players)):
            raise ValueError("The number of players in the environment does not match the number of players in the controller.")
        n_observations = env.observation_space[0].shape[0]
        n_actions = env.action_space[0].n
        state_shape = n_observations*len(env.players)

        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.params = list(self.policy_net.parameters())
        self.mixer = QMixer(len(env.players), state_shape)
        self.params += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)
        self.optimizer = optim.AdamW(self.params, lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

        print("players", self.players)