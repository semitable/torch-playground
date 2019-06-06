"""
Experience replay implementations - Includes multi-agent variants and prioritized experience replay
"""
import random
from operator import itemgetter
from collections import namedtuple

import torch
import numpy as np
from segment_tree import SumSegmentTree, MinSegmentTree


Transition = namedtuple(
    "Transition", ("states", "actions", "next_states", "rewards", "done")
)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.memory)


class MultiAgentReplayBuffer:
    def __init__(self, agents, capacity):
        self.agents = agents
        self.sa_buffers = [ReplayBuffer(capacity) for _ in range(agents)]
        self.can_prioritize = False

    def __len__(self):
        return len(self.sa_buffers[0])

    def sample(self, batch_size, device=None):
        samples = np.random.randint(0, high=len(self), size=batch_size)
        agent_batches = []

        for buffer in self.sa_buffers:
            transitions = itemgetter(*samples)(buffer.memory)
            if device:
                batch = Transition(
                    *[
                        torch.Tensor(e).view(batch_size, -1).to(device)
                        for e in zip(*transitions)
                    ]
                )
            else:
                batch = Transition(*zip(*transitions))
            agent_batches.append(batch)

        return agent_batches

    def push(self, *args):
        for i, buffer in enumerate(self.sa_buffers):
            buffer.push(*[a[i] for a in args])


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, alpha):
        """"
        Arguments:
            capacity {int} -- maximum capacity of the buffer
            alpha {float} -- how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super().__init__(capacity)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < capacity:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

        self._idxes = None

    def push(self, *args):
        """See ReplayBuffer.push"""
        idx = self.position
        super().push(*args)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self.memory) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self.memory)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self.memory)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)

        transitions = itemgetter(*idxes)(self.memory)
        batch = Transition(
            *[torch.Tensor(e).view(batch_size, -1) for e in zip(*transitions)]
        )

        self._idxes = idxes
        return batch
        # return batch, weights, idxes

    def update_priorities(self, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(self._idxes) == len(priorities)
        for idx, priority in zip(self._idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.memory)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


class PrioritizedMultiAgentReplayBuffer(MultiAgentReplayBuffer):
    def __init__(self, agents, capacity, alpha):
        super().__init__(agents, capacity)

        self.can_prioritize = True
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < capacity:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

        self._idxes = None

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, device=None):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        """
        # assert beta > 0

        idxes = self._sample_proportional(batch_size)

        # weights = []
        # p_min = self._it_min.min() / self._it_sum.sum()
        # max_weight = (p_min * len(self)) ** (-beta)

        # for idx in idxes:
        #     p_sample = self._it_sum[idx] / self._it_sum.sum()
        #     weight = (p_sample * len(self)) ** (-beta)
        #     weights.append(weight / max_weight)
        # weights = np.array(weights)

        agent_batches = []
        for buffer in self.sa_buffers:
            transitions = itemgetter(*idxes)(buffer.memory)
            if device:
                batch = Transition(
                    *[
                        torch.Tensor(e).view(batch_size, -1).to(device)
                        for e in zip(*transitions)
                    ]
                )
            else:
                batch = Transition(*zip(*transitions))
            agent_batches.append(batch)

        self._idxes = idxes
        return agent_batches

    def push(self, *args):
        idx = self.sa_buffers[0].position
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

        for i, buffer in enumerate(self.sa_buffers):
            buffer.push(*[a[i] for a in args])

    def update_priorities(self, priorities):
        assert len(self._idxes) == len(priorities)
        for idx, priority in zip(self._idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
