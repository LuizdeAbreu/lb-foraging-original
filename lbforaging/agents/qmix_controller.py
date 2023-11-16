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

        self.mixer = None
        self.target_mixer = None
        self.agent_networks = []
        self.target_agent_networks = []

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
            target_network = copy.deepcopy(network)
            self.agent_networks.append(network)
            self.target_agent_networks.append(target_network)
            self.params += list(network.parameters())

        self.mixer = QMixer(len(env.players), state_shape, n_actions)
        self.target_mixer = copy.deepcopy(self.mixer)

        self.params += list(self.mixer.parameters())
        self.optimizer = optim.AdamW(params=self.params, lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.last_target_update_episode = 0    
        self.last_target_update_step = 0
        self.last_saved_model_step = 0

    def choose_action(self, states, first=False):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold and first == False:
            with torch.no_grad():
                actions = []
                for i in range(len(self.players)):
                    agent_network = self.agent_networks[i]
                    state = torch.tensor(states[i], dtype=torch.float32, device=device).unsqueeze(0)
                    agent_q_values = agent_network(state)
                    result = agent_q_values.max(1)[1].view(1, 1)
                    action = Action(result.item())
                    actions.append(action)
                return actions
        else:
            return [random.choice(Env.action_set) for _ in range(len(self.players))]

    def _step(self, states, rewards, done, info):
        states = np.array(states)
        states = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
        if (self.previous_states is None):
            self.previous_states = states
            self.previous_global_state = info["global"]
            self.previous_actions = self.choose_action(states, True)
            return 

        rewards = torch.tensor([rewards], dtype=torch.float32, device=device)

        global_state = info["global"] 

        all_done = all(flag == True for (flag) in done)
        if (all_done):
            states = None
            global_state = None  

        self.previous_actions = [int(x) for x in self.previous_actions]
        self.previous_actions = torch.tensor(self.previous_actions, dtype=torch.float32, device=device)

        self.memory.push_qmix(self.previous_states, self.previous_actions, states, rewards, self.previous_global_state, global_state)

        self.previous_states = states
        self.previous_global_state = global_state

        self.optimize_model()

        mixer_state_dict = self.mixer.state_dict()
        target_mixer_state_dict = self.target_mixer.state_dict()
        for key in mixer_state_dict.keys():
            target_mixer_state_dict[key] = TAU * mixer_state_dict[key] + (1 - TAU) * target_mixer_state_dict[key]
        self.target_mixer.load_state_dict(target_mixer_state_dict)

        # we also need to update the agent networks
        for i in range(len(self.players)):
            agent_state_dict = self.agent_networks[i].state_dict()
            target_agent_state_dict = self.target_agent_networks[i].state_dict()
            for key in agent_state_dict.keys():
                target_agent_state_dict[key] = TAU * agent_state_dict[key] + (1 - TAU) * target_agent_state_dict[key]
            self.target_agent_networks[i].load_state_dict(target_agent_state_dict)

        if (self.steps_done - self.last_saved_model_step >= 1000):
            self.save_models()

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)

        batch = QMixTransition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_states)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_states
                                                    if s is not None])
        state_batch = torch.cat(batch.states)
        action_batch = torch.stack(batch.actions)
        reward_batch = torch.cat(batch.rewards)

        agent_state_action_values = []
        for i in range(len(self.players)):
            agent_network = self.agent_networks[i]
            agent_actions = action_batch[:,i].long()
            q_values = agent_network(state_batch[:,i,:])
            select_action_q_values = q_values.gather(1, agent_actions.unsqueeze(1))
            agent_state_action_values.append(select_action_q_values)

        # transform list to tensor
        agent_state_action_values = torch.stack(agent_state_action_values, dim=0)
        # should be (batch size, number of agents, q_values (one for each action))
        agent_state_action_values = agent_state_action_values.transpose(0,1)
        mixer_state_action_values = self.mixer(agent_state_action_values, state_batch).squeeze(1)
        
        target_agent_state_action_values = []
        for i in range(len(self.players)):
            target_agent_network = self.target_agent_networks[i]
            target_agent_actions = action_batch[:,i].long()
            target_q_values = target_agent_network(non_final_next_states[:,i,:])
            # get target actions only for non final states
            target_agent_actions = target_agent_actions[non_final_mask]
            target_select_action_q_values = target_q_values.gather(1, target_agent_actions.unsqueeze(1))
            target_agent_state_action_values.append(target_select_action_q_values)
        
        # transform list to tensor
        target_agent_state_action_values = torch.stack(target_agent_state_action_values, dim=0)
        target_agent_state_action_values = target_agent_state_action_values.transpose(0,1)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, 1, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_mixer(target_agent_state_action_values, non_final_next_states).max(1)[0]

        # for each reward batch, sum the rewards for each agent into one value
        sum_rewards_batch = torch.sum(reward_batch, dim=1).unsqueeze(1)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + sum_rewards_batch

        # Compute loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(mixer_state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.params, 100)
        self.optimizer.step()

    def save_models(self):
        torch.save(self.mixer.state_dict(), "saved/mixer.pt")
        torch.save(self.target_mixer.state_dict(), "saved/target_mixer.pt")
        for i in range(len(self.players)):
            torch.save(self.agent_networks[i].state_dict(), "saved/agent_network_" + str(i) + ".pt")
            torch.save(self.target_agent_networks[i].state_dict(), "saved/target_agent_network_" + str(i) + ".pt")

    def load_models(self):
        self.mixer.load_state_dict(torch.load("saved/mixer.pt"))
        self.target_mixer.load_state_dict(torch.load("saved/target_mixer.pt"))
        for i in range(len(self.players)):
            self.agent_networks[i].load_state_dict(torch.load("saved/agent_network_" + str(i) + ".pt"))
            self.target_agent_networks[i].load_state_dict(torch.load("saved/target_agent_network_" + str(i) + ".pt"))
        

    

    

