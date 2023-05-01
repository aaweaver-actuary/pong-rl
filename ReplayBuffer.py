"""
This file contains the ReplayBuffer class, which is used to store the experiences
of the agent. The agent will sample from this buffer to train the neural network.
"""


import random
from collections import deque, namedtuple

# need a transition class to store the experience
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        """push a transition into the buffer"""
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        """sample a batch of transitions"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)