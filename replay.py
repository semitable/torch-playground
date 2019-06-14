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
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.memory = None
        self.writes = 0

    def init_memory(self, transition):
        for t in transition:
            assert t.ndim == 1  # sanity check

        self.memory = Transition(
            *[np.zeros([self.capacity, t.size], dtype=t.dtype) for t in transition]
        )

    def push(self, *args):

        if not self.memory:
            self.init_memory(Transition(*args))

        position = (self.writes) % self.capacity
        for i, data in enumerate(args):
            self.memory[i][position, :] = data

        self.writes = self.writes + 1

    def sample(self, batch_size):
        raise NotImplementedError

    def __len__(self):
        return min(self.writes, self.capacity)


class MultiAgentReplayBuffer:
    def __init__(self, agents, capacity):
        self.agents = agents
        self.sa_buffers = [ReplayBuffer(capacity) for _ in range(agents)]
        self.can_prioritize = False

    def __len__(self):
        return len(self.sa_buffers[0])

    def sample(self, batch_size, device="cpu"):
        samples = np.random.randint(0, high=len(self), size=batch_size)
        agent_batches = []

        for buffer in self.sa_buffers:
            batch = Transition(
                *[
                    torch.from_numpy(np.take(d, samples, axis=0)).to(device)
                    for d in buffer.memory
                ]
            )
            agent_batches.append(batch)

        return agent_batches

    def push(self, *args):
        for i, buffer in enumerate(self.sa_buffers):
            buffer.push(*[a[i] for a in args])
