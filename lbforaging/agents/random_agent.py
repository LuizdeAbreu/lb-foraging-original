import random

from lbforaging.foraging.agent import Agent
from lbforaging.foraging.environment import ForagingEnv


class RandomAgent(Agent):
    name = "Random Agent"

    def _step(self, obs):
        return random.choice(ForagingEnv.action_set)
