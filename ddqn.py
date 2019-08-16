#!python
# -*- coding: utf-8 -*-
"""
"""

import argparse
import copy
import pickle
import random
import time
from collections import namedtuple
from operator import itemgetter

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.functional import F

# import multiagent.scenarios as scenarios
# from multiagent.environment import MultiAgentEnv
from torch.optim import Adam

import lbforaging

MSELoss = torch.nn.MSELoss()
MAX_EPISODES = 100000


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

    def __len__(self):
        return len(self.sa_buffers[0])

    def sample(self, batch_size, tensorize=False):
        samples = np.random.randint(0, high=len(self), size=batch_size)
        agent_batches = []

        for buffer in self.sa_buffers:
            transitions = itemgetter(*samples)(buffer.memory)
            if tensorize:
                batch = Transition(
                    *[torch.Tensor(e).view(batch_size, -1) for e in zip(*transitions)]
                )
            else:
                batch = Transition(*zip(*transitions))
            agent_batches.append(batch)

        return agent_batches

    def push(self, *args):
        for i, buffer in enumerate(self.sa_buffers):
            buffer.push(*[a[i] for a in args])


def onehot_from_logits(logits, epsilon=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if epsilon == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(
        torch.eye(logits.shape[1])[
            [np.random.choice(range(logits.shape[1]), size=logits.shape[0])]
        ],
        requires_grad=False,
    )
    # chooses between best and random actions using epsilon greedy
    return torch.stack(
        [
            argmax_acs[i] if r > epsilon else rand_acs[i]
            for i, r in enumerate(torch.rand(logits.shape[0]))
        ]
    )


class DDQN:
    def __init__(self, params):
        state_sizes = params["state_space"]
        action_sizes = params["action_space"]
        agent_count = len(action_sizes)

        # self.discrete_actions = [True, True]

        self.gamma = params["discount"]
        self.update_freq = params["update_freq"]
        self.target_update_freq = params["target_update_freq"]
        self.batch_size = params["batch_size"]
        self.grad_clip = params["grad_clip"]
        self.epsilon_target = params["epsilon_target"]
        self.use_dropout = params["dropout"]
        self.epsilon_anneal = params["epsilon_anneal"]
        self.centralized = params["centralized_expl"]

        self.timesteps = 0

        self.policies = [
            FCNetwork(
                (s.shape[0], *params["network_size"], a.n), dropout=params["dropout"]
            )
            for s, a in zip(state_sizes, action_sizes)
        ]

        self.policy_targets = copy.deepcopy(self.policies)

        self.policy_optimizers = [
            Adam(x.parameters(), lr=params["lr"]) for x in self.policies
        ]

        self.agent_count = agent_count
        self.state_size = state_sizes
        self.action_size = action_sizes

        self.replay_buffer = MultiAgentReplayBuffer(agent_count, params["buffer_size"])

    def select_actions(self, states, explore=False):
        """

        :param states: List of the agent states
        :return: actions for each agent
        """

        actions = []
        if self.centralized:
            rand = np.random.uniform()
        for i, state in enumerate(states):
            if not self.centralized:
                rand = np.random.uniform()

            sb = torch.Tensor(state)
            if explore and rand < self.epsilon:
                action = onehot_from_logits(
                    self.policies[i](sb.unsqueeze(0)), epsilon=1.0
                )
            else:
                action = onehot_from_logits(self.policies[i](sb.unsqueeze(0)))
            actions.append(action.squeeze().detach())

        return actions

    def update(self, agent):
        """

        :param agent: The agent ID)
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        nbatch = self.replay_buffer.sample(self.batch_size, tensorize=True)
        batch = nbatch[agent]

        non_final_mask = torch.ByteTensor(tuple(map(lambda s: not s, batch.done)))

        non_final_next_states = torch.stack(
            [s for done, s in zip(batch.done, batch.next_states) if not done]
        )

        state_batch = batch.states
        action_batch = batch.actions
        reward_batch = batch.rewards

        state_action_values = (
            (self.policies[agent](state_batch) * action_batch).sum(dim=1).view(-1, 1)
        )

        next_state_values = torch.zeros(self.batch_size)
        best_actions = (
            self.policies[agent](non_final_next_states).argmax(1).unsqueeze(-1)
        )
        next_state_values[non_final_mask] = (
            self.policy_targets[agent](non_final_next_states)
            .gather(dim=1, index=best_actions)
            .squeeze()
            .detach()
        )
        targets = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, targets.unsqueeze(1))
        self.policy_optimizers[agent].zero_grad()
        loss.backward()
        for param in self.policies[agent].parameters():
            param.grad.data.clamp_(-self.grad_clip, self.grad_clip)
        self.policy_optimizers[agent].step()

    def update_epsilon(self):
        ratio = min(1, float(self.timesteps) / self.epsilon_anneal)
        self.epsilon = ratio * self.epsilon_target + (1 - ratio) * 1

    def set_mode(self, mode):
        for network in self.policies:
            if mode == "eval":
                network.eval()
            elif mode == "train":
                network.train()
            else:
                raise ValueError()

    def play_episode(self, env, evaluate=False, render=False):
        obs_n = env.reset()
        episode_rewards = np.zeros(self.agent_count)
        done = False
        if not evaluate:
            self.update_epsilon()

        while not done:
            # query for action from each agent's policy
            # act_n = [np.array([0, 0, 1]), env.action_space[1].sample()]
            actions = [
                a.detach().numpy()
                for a in self.select_actions(
                    obs_n, explore=(not evaluate and not self.use_dropout)
                )
            ]
            if render:
                print(actions)

            # step environment
            next_obs_n, reward_n, done_n, _ = env.step(
                [np.argmax(actions[0]), np.argmax(actions[1])]
            )
            done = np.all(done_n)

            episode_rewards += reward_n

            self.replay_buffer.push(obs_n, actions, next_obs_n, reward_n, done_n)

            # render all agent views
            if render:
                env.render()

            obs_n = next_obs_n

            if evaluate:
                continue

            self.timesteps += 1

            if self.timesteps % self.update_freq == 0:
                for agent in range(self.agent_count):
                    self.update(agent)

            if self.timesteps % self.target_update_freq == 0:
                for i in range(self.agent_count):
                    self.policy_targets[i].hard_update(self.policies[i])

        return episode_rewards


def movingaverage(values, window):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, "valid")
    return sma


def plot_rewards(rewards, eval_every):

    ma_length = 10

    plt.figure(1)
    plt.clf()
    # plt.yscale("symlog")
    plt.title("Training...")
    plt.xlabel("Timestep")
    plt.ylabel("Reward")

    x = np.arange(eval_every * len(np.sum(rewards, axis=1)), step=eval_every)
    y = np.sum(rewards, axis=1)
    yMA = movingaverage(y, ma_length)

    plt.plot(x, y, "x")
    if len(y) > ma_length:
        plt.plot(x[len(x) - len(yMA) :], yMA, color="r")
    plt.pause(0.001)  # pause a bit so that plots are updated


def evaluate(env, maddpg, rewards, eval_episodes, eval_every, render):
    maddpg.set_mode("eval")
    episode_rewards = np.zeros(maddpg.agent_count)
    for i in range(eval_episodes):
        episode_rewards += (
            maddpg.play_episode(env, evaluate=True, render=render) / eval_episodes
        )
    maddpg.set_mode("train")

    rewards = (
        np.vstack([rewards, episode_rewards])
        if rewards is not None
        else np.array([episode_rewards])
    )
    if render:
        plot_rewards(rewards, eval_every)
    return rewards


def main(**params):

    plt.ion()
    rewards = None
    torch.set_num_threads(1)
    seed = params["seed"]

    env = gym.make(params["gym_env"])

    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        env.seed(seed)

    # env = make_env("simple_speaker_listener", discrete_action=True)

    params.update({"state_space": env.observation_space})
    params.update({"action_space": env.action_space})

    # print(env.action_space)
    maddpg = DDQN(params)
    last_eval = 0
    saved_networks = []

    while maddpg.timesteps <= params["max_timesteps"]:
        _ = maddpg.play_episode(env, evaluate=False, render=False)
        if maddpg.timesteps - last_eval >= params["eval_every"]:
            last_eval = maddpg.timesteps
            rewards = evaluate(
                env,
                maddpg,
                rewards,
                params["eval_episodes"],
                params["eval_every"],
                render=params["render"],
            )
            saved_networks.append([a.state_dict() for a in maddpg.policies])
            print(maddpg.timesteps)

    return {
        "saved_networks": saved_networks,
        "freq": params["eval_every"],
        "reruns": params["eval_episodes"],
        "rewards": rewards,
    }


if __name__ == "__main__":
    # Example on how to initialize global locks for processes
    # and counters.

    parser = argparse.ArgumentParser()
    # parser.add_argument('--hidden-layers', type=int, nargs='+', default=[64, 64])
    parser.add_argument("--gym-env", default="Foraging-6x6-2p-1f-v0", type=str)
    parser.add_argument("--seed", default=None)
    parser.add_argument("--max-timesteps", default=1e7, type=float)
    parser.add_argument("--update-freq", type=float, default=4)
    parser.add_argument("--target-update-freq", type=float, default=10000)
    parser.add_argument("--network-size", default=(64, 64))
    parser.add_argument("--epsilon-target", type=float, default=0.1)
    parser.add_argument("--epsilon-anneal", type=float, default=1e7)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=10000)
    parser.add_argument("--buffer-size", type=int, default=1e8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--centralized-expl", action="store_true")
    parser.add_argument("--dropout", action="store_true")
    parser.add_argument("--render", action="store_true")

    args = parser.parse_args()

    data = main(**vars(args))
    pickle.dump(data, open("egreedy.local.p", "wb"))
