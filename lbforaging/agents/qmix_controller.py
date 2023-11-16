import random
import math
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from lbforaging.foraging.agent import Agent
from lbforaging.agents.networks.dqn import DQN
from lbforaging.foraging.environment import Action, ForagingEnv as Env
from lbforaging.agents.helpers import ReplayMemory, Transition
from lbforaging.agents.networks.qmixer import QMixer
from lbforaging.agents.dqn_agent import DQNAgent
from lbforaging.agents.networks.dqn import DQN

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
        self.previous_state = None
        self.previous_actions = None
        self.steps_done = 0
        self.players = [player]
        self.agent_networks = []

    def add_player(self, player):
        self.players.append(player)

    def init_from_env(self, env):
        if (len(env.players) != len(self.players)):
            raise ValueError("The number of players in the environment does not match the number of players in the controller.")
        n_observations = env.observation_space[0].shape[0]
        n_actions = env.action_space[0].n
        state_shape = n_observations*len(env.players)
        
        self.params = []
        
        for _ in self.players:
            network = DQN(n_observations, n_actions).to(device)
            self.agent_networks.append(network)
            self.params += list(network.parameters())

        self.mixer = QMixer(len(env.players), state_shape)
        self.target_mixer = copy.deepcopy(self.mixer)

        self.params += list(self.mixer.parameters())
        self.optimizer = optim.AdamW(params=self.params, lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.last_target_update_episode = 0    
        self.last_target_update_step = 0

    def choose_action(self, state, first=False):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold and first != False:
            with torch.no_grad():
                q_values = []
                for i in range(len(self.players)):
                    agent_network = self.agent_networks[i]
                    agent_q_values = agent_network(torch.tensor(state[i], dtype=torch.float32, device=device).unsqueeze(0))
                    q_values.append(agent_q_values[0])
                
                state = np.array(state)
                result = self.mixer(q_values, torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                print("RESULT CHOOSE ACTION MIXER", result)
                return result.max(1)[1].view(1, 1)
        else:
            return [random.choice(Env.action_set) for _ in range(len(self.players))]

    def _step(self, state, reward, done):
        if (self.previous_state is None):
            self.previous_state = state
            self.previous_actions = self.choose_action(state, first=True)
            return 
        
        state = np.array(state)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        reward = torch.tensor([reward], dtype=torch.float32, device=device)

        if done:
            state = None

        self.memory.push(self.previous_state, self.previous_actions, state, reward)

        self.previous_state = state

        self.optimize_model()

        mixer_state_dict = self.mixer.state_dict()
        target_mixer_state_dict = self.target_mixer.state_dict()
        for key in mixer_state_dict.keys():
            target_mixer_state_dict[key] = TAU * mixer_state_dict[key] + (1 - TAU) * target_mixer_state_dict[key]
        self.target_mixer.load_state_dict(target_mixer_state_dict)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to self.policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
            
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.params, 100)
        self.optimizer.step()

    

    

