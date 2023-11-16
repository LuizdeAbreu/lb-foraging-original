import random 
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

QMixTransition = namedtuple('QMixTransition',
                        ('states', 'actions', 'next_states', 'rewards', 'global_state', 'next_global_state'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def push_qmix(self, *args):
        """Save a transition for QMIX"""
        self.memory.append(QMixTransition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)