import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from lbforaging.foraging.agent import Agent
from lbforaging.agents.dqn import DQN
from lbforaging.foraging.environment import ForagingEnv as env
from lbforaging.agents.helpers import ReplayMemory, Transition

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent(Agent):
    name = "Q Agent"

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)

        self.policy_net = None
        self.target_net = None
        self.memory = None
        self.optimizer = None
        self.previous_state = None
        self.previous_action = None

    def choose_action(self, obs):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(obs).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.choice(env.action_set)]], device=device, dtype=torch.long)

    def step(self, obs, reward, done):
        if (self.policy_net == None):
            n_observations = env.observation_space[0].shape[0]
            n_actions = env.action_space[0].n
            self.policy_net = DQN(n_observations, n_actions).to(device)
            self.target_net = DQN(n_observations, n_actions).to(device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
            self.memory = ReplayMemory(10000)
            self.previous_state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            self.previous_action = 0

        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        reward = torch.tensor([reward], dtype=torch.float32, device=device)

        if done:
            next_state = None
        else:
            next_state = obs

        self.memory.push(self.previous_state, torch.tensor([[self.previous_action]], device=device, dtype=torch.long), next_state, reward)

        self.previous_state = obs

        self.optimize_model()

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)     
        

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
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    