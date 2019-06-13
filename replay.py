"""
Experience replay implementations - Includes multi-agent variant
"""
import random
from operator import itemgetter
from collections import namedtuple

import torch
import numpy as np

Transition = namedtuple(
    "Transition", ("states", "actions", "next_states", "rewards", "done")
)


class ReplayBuffer:
    def __init__(self, capacity, device="cpu"):
        self.capacity = int(capacity)
        self.memory = []
        self.position = 0
        self.device = device

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(
            *[torch.from_numpy(e).view(1, -1).to(self.device) for e in args]
        )

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.memory)


class MultiAgentReplayBuffer:
    def __init__(self, agents, capacity, device="cpu"):
        self.agents = agents
        self.sa_buffers = [ReplayBuffer(capacity, device) for _ in range(agents)]
        self.can_prioritize = False

    def __len__(self):
        return len(self.sa_buffers[0])

    def sample(self, batch_size, device=None):
        samples = np.random.randint(0, high=len(self), size=batch_size)
        agent_batches = []

        for buffer in self.sa_buffers:
            transitions = itemgetter(*samples)(buffer.memory)
            batch = Transition(*[torch.cat(e, dim=0) for e in zip(*transitions)])
            agent_batches.append(batch)

        return agent_batches

    def push(self, *args):
        for i, buffer in enumerate(self.sa_buffers):
            buffer.push(*[a[i] for a in args])
