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
from lbforaging.agents.helpers import ReplayMemory, QMixTransition, Transition
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
        self.memory = None
        self.optimizer = None
        
        self.previous_states = None
        self.previous_actions = None
        self.previous_global_state = None

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

        self.mixer = QMixer(len(env.players), state_shape, n_actions)
        self.target_mixer = copy.deepcopy(self.mixer)

        self.params += list(self.mixer.parameters())
        self.optimizer = optim.AdamW(params=self.params, lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.last_target_update_episode = 0    
        self.last_target_update_step = 0

    def choose_action(self, states, first=False):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold and first != False:
            with torch.no_grad():
                actions = []
                for i in range(len(self.players)):
                    agent_network = self.agent_networks[i]
                    agent_q_values = agent_network(torch.tensor(states[i], dtype=torch.float32, device=device).unsqueeze(0))
                    actions.append(agent_q_values.max(1)[1].view(1, 1))
                return actions
        else:
            return [random.choice(Env.action_set) for _ in range(len(self.players))]

    def _step(self, states, rewards, done, info):
        if (self.previous_states is None):
            self.previous_states = states
            self.previous_global_state = info["global"]
            self.previous_actions = self.choose_action(states, first=True)
            return 
        
        states = np.array(states)
        states = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
        rewards = torch.tensor([rewards], dtype=torch.float32, device=device)


        global_state = info["global"] 

        all_done = all(flag == True for (flag) in done)
        if (all_done):
            states = None
            global_state = None  

        # self.memory.push(self.previous_states, self.previous_actions, states, rewards)
        self.memory.push_qmix(self.previous_states, self.previous_actions, states, rewards, self.previous_global_state, global_state)

        self.previous_states = states
        self.previous_global_state = global_state

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
        batch = QMixTransition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        print("batch", batch)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_states)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_states
                                                    if s is not None])
        state_batch = torch.cat(batch.states)
        action_batch = torch.cat(batch.actions)
        reward_batch = torch.cat(batch.rewards)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to self.mixer
        state_action_values = self.mixer(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_mixer(non_final_next_states).max(1)[0]
            
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

    

    

