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
from lbforaging.agents.networks.qmixer import QMixNet
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
        # Add a player to the controller
        # since in the case of the QMixController we need to
        # handle multiple players
        self.players.append(player)

    def init_from_env(self, env):
        if (len(env.players) != len(self.players)):
            raise ValueError("The number of players in the environment does not match the number of players in the controller.")
        # Analogous to DQNAgent.init_from_env
        # but with extra steps
        n_observations = env.observation_space[0].shape[0]
        n_actions = env.action_space[0].n
        state_shape = n_observations*len(env.players)
        
        # Through QMIX, we want to optimize both the agent networks 
        # and the mixing network through the same process, so we 
        # need to keep track of the parameters of all the networks involved
        self.params = []
        
        # For each player, we create an agent network and a target agent network
        # and add their parameters to the list of parameters to be optimized
        for _ in self.players:
            network = DQN(n_observations, n_actions).to(device)
            target_network = copy.deepcopy(network)
            self.agent_networks.append(network)
            self.target_agent_networks.append(target_network)
            self.params += list(network.parameters())

        # Then we create the mixing network and the target mixing network
        self.mixer = QMixNet(len(env.players), state_shape, n_actions)
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        # Here we just initialize the optimizer and the memory as in DQNAgent
        self.optimizer = optim.AdamW(params=self.params, lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.last_target_update_episode = 0    
        self.last_target_update_step = 0
        self.last_saved_model_step = 0

    def choose_action(self, states, first=False):
        # Same as DQNAgent, 
        # but note that in this case we return a list of actions
        # (the joint action) by using each agent network
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        result = None
        if sample > eps_threshold and first == False:
            result = self.choose_optimal_action(states)
        else:
            result = [random.choice(Env.action_set) for _ in range(len(self.players))]
        self.previous_actions = result
        return result
        
    def choose_optimal_action(self, states):
        actions = []
        with torch.no_grad():
            for i in range(len(self.players)):
                agent_network = self.agent_networks[i]
                agent_network.eval()
                state = torch.tensor(states[i], dtype=torch.float32, device=device).unsqueeze(0)
                agent_q_values = agent_network(state)
                result = agent_q_values.max(1)[1].view(1, 1)
                action = Action(result.item())
                actions.append(action)
                agent_network.train()
        return actions


    def _step(self, states, rewards, done, info, episode):
        # Same as DQNAgent, but we're now receiving multiple instances
        # of each input, i.e, states is the observations of all players,
        # rewards is a list of each player's reward at that step, etc
        states = np.array(states)
        states = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
        if (self.previous_states is None):
            self.previous_states = states
            self.previous_global_state = info["global"]
            self.previous_actions = self.choose_action(states, True)
            self._set_all_to_train()
            return 

        rewards = torch.tensor([rewards], dtype=torch.float32, device=device)

        global_state = info["global"] 

        all_done = all(flag == True for (flag) in done)
        if (all_done):
            states = None
            global_state = None  

        self.previous_actions = [int(x) for x in self.previous_actions]
        self.previous_actions = torch.tensor(self.previous_actions, dtype=torch.float32, device=device)

        # Here we store a QMixTransition (see helper.py)
        # which is like a default Transition but with the addition
        # of the global state used by the mixing network
        self.memory.push_qmix(self.previous_states, self.previous_actions, states, rewards, self.previous_global_state, global_state)

        self.previous_states = states
        self.previous_global_state = global_state

        self.optimize_model()

        # steps_since_last_save = self.steps_done - self.last_saved_model_step
        # print("steps since last save: " + str(steps_since_last_save))
        # if (steps_since_last_save >= 1000):
        #     self.save_models()

    def optimize_model(self):
        # Again, analogous to DQNAgent, but with extra steps
        # since now we need to update both the agent networks
        # and the mixing network
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

        # For each agent, we get the q values of the actions taken
        # in each transition of the batch
        agent_state_action_values = []
        for i in range(len(self.players)):
            agent_network = self.agent_networks[i]
            agent_actions = action_batch[:,i].long()
            q_values = agent_network(state_batch[:,i,:])
            select_action_q_values = q_values.gather(1, agent_actions.unsqueeze(1))
            agent_state_action_values.append(select_action_q_values)

        agent_state_action_values = torch.stack(agent_state_action_values, dim=0)
        # should be (batch size, number of agents, q_values (one for each action))
        agent_state_action_values = agent_state_action_values.transpose(0,1)
        # Note that "mixer_q_values" is equal to Qtot(s_t, a_t) where s_t is the global state
        # and a_t is the joint action taken by the agents
        mixer_q_values = self.mixer(agent_state_action_values, state_batch).squeeze(1)

        # Recap:
        # Qtot (s_t, a_t) = retorna os valores esperados de um estado global e de uma joint action (uma ação pra cada agente)
        # Qa (o_t, u_t) = retorna o valor esperado da observação do agente e uma ação que ele tomou
        
        # Now we need to compute the target values
        target_agent_state_action_values = []
        for i in range(len(self.players)):
            target_agent_network = self.target_agent_networks[i]
            target_agent_actions = action_batch[:,i].long()
            target_q_values = target_agent_network(non_final_next_states[:,i,:])
            # get target actions only for non final states
            target_agent_actions = target_agent_actions[non_final_mask]
            target_select_action_q_values = target_q_values.gather(1, target_agent_actions.unsqueeze(1))
            target_agent_state_action_values.append(target_select_action_q_values)
        
        target_agent_state_action_values = torch.stack(target_agent_state_action_values, dim=0)
        target_agent_state_action_values = target_agent_state_action_values.transpose(0,1)

        target_mixer_q_values = torch.zeros(BATCH_SIZE, 1, device=device)
        with torch.no_grad():
            target_mixer_q_values[non_final_mask] = self.target_mixer(target_agent_state_action_values, non_final_next_states).max(1)[0]

        # for each reward batch, sum the rewards for each agent into one value
        sum_rewards_batch = torch.sum(reward_batch, dim=1).unsqueeze(1)
        # Compute the expected Q values (i.e. the values outputted by the target mixer)
        expected_state_action_values = (target_mixer_q_values * GAMMA) + sum_rewards_batch

        # Compute loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(mixer_q_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.params, 100)
        self.optimizer.step()

        # Now we update the target networks through soft updates;
        # First the mixing one
        mixer_state_dict = self.mixer.state_dict()
        target_mixer_state_dict = self.target_mixer.state_dict()
        for key in mixer_state_dict.keys():
            target_mixer_state_dict[key] = TAU * mixer_state_dict[key] + (1 - TAU) * target_mixer_state_dict[key]
        self.target_mixer.load_state_dict(target_mixer_state_dict)

        # We also need to update the agent networks
        for i in range(len(self.players)):
            agent_state_dict = self.agent_networks[i].state_dict()
            target_agent_state_dict = self.target_agent_networks[i].state_dict()
            for key in agent_state_dict.keys():
                target_agent_state_dict[key] = TAU * agent_state_dict[key] + (1 - TAU) * target_agent_state_dict[key]
            self.target_agent_networks[i].load_state_dict(target_agent_state_dict)

    def save(self, _, __, ___):
        # Helper function to save model after a number of episodes
        self.last_saved_model_step = self.steps_done

        torch.save(self.mixer.state_dict(), "saved/mixer.pt")
        torch.save(self.target_mixer.state_dict(), "saved/target_mixer.pt")
        for i in range(len(self.players)):
            torch.save(self.agent_networks[i].state_dict(), "saved/agent_network_" + str(i) + ".pt")
            torch.save(self.target_agent_networks[i].state_dict(), "saved/target_agent_network_" + str(i) + ".pt")

        torch.save({
            'optimizer_state_dict': self.optimizer.state_dict(),
            'memory': self.memory,
            'steps_done': self.steps_done,
            'last_saved_model_step': self.last_saved_model_step,
        }, "saved/other.pt")

    def load(self, _):
        # Helper function to load model to start training from a previous point
        self.mixer.load_state_dict(torch.load("saved/mixer.pt"))
        self.target_mixer.load_state_dict(torch.load("saved/target_mixer.pt"))
        
        self.mixer.train()
        self.target_mixer.train()

        for i in range(len(self.players)):
            self.agent_networks[i].load_state_dict(torch.load("saved/agent_network_" + str(i) + ".pt"))
            self.target_agent_networks[i].load_state_dict(torch.load("saved/target_agent_network_" + str(i) + ".pt"))

            self.agent_networks[i].train()
            self.target_agent_networks[i].train()

        checkpoint = torch.load("saved/other.pt")
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.memory = checkpoint['memory']
        self.steps_done = checkpoint['steps_done']
        self.last_saved_model_step = checkpoint['last_saved_model_step']
        
    def _set_all_to_eval(self):
        self.mixer.eval()
        self.target_mixer.eval()
        
        for i in range(len(self.players)):
            self.agent_networks[i].eval()
            self.target_agent_networks[i].eval()

    def _set_all_to_train(self):
        self.mixer.train()
        self.target_mixer.train()
        
        for i in range(len(self.players)):
            self.agent_networks[i].train()
            self.target_agent_networks[i].train()

    

    

