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

# Based on a simple DQN implementation from PyTorch, adapted to Agent class
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

global device

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent(Agent):
    name = "DQN Agent"

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)

        self.policy_net = None
        self.target_net = None
        self.memory = None
        self.optimizer = None
        self.previous_obs = None
        self.previous_action = None
        self.steps_done = 0

    def init_from_env(self, env):
        # Called after the environment is initialized
        # in order to use the environment's observation and action spaces
        # to initialize the neural networks
        n_observations = env.observation_space[0].shape[0]
        n_actions = env.action_space[0].n
        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.params = list(self.policy_net.parameters())
        self.optimizer = optim.AdamW(self.params, lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

    def get_qvalues(self, obs):
        # Returns the Q-values for a given observation
        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        return self.policy_net(obs)

    def choose_action(self, obs):
        # Use epsilon-greedy policy to choose an action
        # with the current policy network
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        result = None
        if sample > eps_threshold and self.policy_net is not None:
            with torch.no_grad():
                # q_values will be a tensor of shape (1, n_actions)
                q_values = self.get_qvalues(obs)
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                result = q_values.max(1)[1].view(1, 1)
                action = Action(result.item())
                self.previous_action = action
                result = action
        else:
            result = random.choice(Env.action_set)
        self.previous_action = result
        return result

    def _step(self, obs, reward, done, info, episode):
        if (self.previous_obs is None):
            # if this is the first step, we don't have a previous observation
            # so we just save the current observation and action and return
            self.previous_obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            self.previous_action = self.choose_action(obs)
            return
            
        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        reward = torch.tensor([reward], dtype=torch.float32, device=device)

        if done:
            next_obs = None
        else:
            next_obs = obs

        self.previous_action = int(self.previous_action)

        # Store the transition in memory
        self.memory.push(self.previous_obs, torch.tensor([[self.previous_action]], device=device, dtype=torch.long), next_obs, reward)

        self.previous_obs = obs

        # Perform one step of the optimization (on the target network)
        self.optimize_model()

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            # if we don't have enough transitions in memory to sample a full batch, we can't learn yet
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

        # Update the target network, copying all weights and biases in DQN
        # We do a soft update here, so we don't completely overwrite the target network
        # in fact we only update it by a small amount (TAU = 0.005 or 0.5%)
        # as opposed to the hard update which completely overwrites the target network every C steps
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)     

    