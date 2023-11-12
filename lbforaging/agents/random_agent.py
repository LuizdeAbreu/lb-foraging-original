import random

from lbforaging.foraging.agent import Agent


class RandomAgent(Agent):
    name = "Random Agent"

    def _step(self, obs):
        return random.choice(obs.actions)
