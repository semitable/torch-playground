import copy
import torch
from torch import nn
from torch.autograd import Variable
from torch.functional import F
import numpy as np

# import multiagent.scenarios as scenarios
# from multiagent.environment import MultiAgentEnv
from torch.optim import Adam
import argparse
import random
from collections import namedtuple
import gym
import pickle
import lbforaging
import time
from operator import itemgetter
import matplotlib.pyplot as plt

MSELoss = torch.nn.MSELoss()
MAX_EPISODES = 100000


class FCNetwork(nn.Module):
    def __init__(self, dims):
        """
        Creates a network using ReLUs between layers and no activation at the end
        :param dims: tuple in the form of (100, 100, ..., 5). for dim sizes
        """
        super().__init__()
        h_sizes = dims[:-1]
        out_size = dims[-1]

        # Hidden layers
        self.hidden = []
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))
            self.add_module("hidden_layer" + str(k), self.hidden[-1])

        # Output layer
        self.out = nn.Linear(h_sizes[-1], out_size)

    @staticmethod
    def calc_layer_size(size, extra):
        if type(size) is int:
            return size
        return extra["size"]

    def forward(self, x):
        # Feedforward
        for layer in self.hidden:
            x = F.relu(layer(x))
        output = self.out(x)
        return output

    def hard_update(self, source):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def soft_update(self, source, t):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)


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


def select_actions(states, actors):

    actions = []
    for i, state in enumerate(states):

        sb = torch.Tensor(state)
        action = onehot_from_logits(actors[i](sb.unsqueeze(0)), epsilon=0.1)
        actions.append(action.squeeze().detach())

    return actions


def play_episode(env, actors, render=False):
    obs_n = env.reset()
    episode_rewards = np.zeros(2)
    done = False

    while not done:
        # query for action from each agent's policy
        # act_n = [np.array([0, 0, 1]), env.action_space[1].sample()]
        actions = [a.detach().numpy() for a in select_actions(obs_n, actors)]

        # step environment
        next_obs_n, reward_n, done_n, _ = env.step(
            [np.argmax(actions[0]), np.argmax(actions[1])]
        )
        done = np.all(done_n)

        episode_rewards += reward_n

        # render all agent views
        if render:
            print(actions)
            env.render()
            time.sleep(1)

        obs_n = next_obs_n

    return episode_rewards


def main(**params):

    plt.ion()
    rewards = None
    torch.set_num_threads(1)
    seed = params["seed"]

    data = pickle.load(open("egreedy.local.p", "rb"))
    network_size = data["config"]["network_size"]

    env = gym.make(data["config"]["gym_env"])

    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        env.seed(seed)

    # env = make_env("simple_speaker_listener", discrete_action=True)

    params.update({"state_space": env.observation_space})
    params.update({"action_space": env.action_space})

    state_sizes = params["state_space"]
    action_sizes = params["action_space"]
    agent_count = len(action_sizes)

    actors = [
        FCNetwork((s.shape[0], *network_size, a.n))
        for s, a in zip(state_sizes, action_sizes)
    ]

    networks = data["saved_networks"][-9]
    print(networks)
    print(len(networks))
    print(state_sizes)
    for actor, network in zip(actors, networks):
        actor.load_state_dict(network)

    for e in range(10):
        play_episode(env, actors, True)


if __name__ == "__main__":
    # Example on how to initialize global locks for processes
    # and counters.

    parser = argparse.ArgumentParser()
    # parser.add_argument('--hidden-layers', type=int, nargs='+', default=[64, 64])
    # parser.add_argument("--gym-env", default="Foraging-6x6-2p-1f-v0", type=str)
    parser.add_argument("--seed", default=None)
    parser.add_argument("--max-timesteps", default=1e7, type=float)
    parser.add_argument("--update-freq", type=float, default=100)
    parser.add_argument("--target-tau", type=float, default=0.01)
    parser.add_argument("--epsilon-target", type=float, default=0.1)
    parser.add_argument("--epsilon-anneal", type=float, default=1e7)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--critic-lr", type=float, default=0.0001)
    parser.add_argument("--actor-lr", type=float, default=0.0001)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--buffer-size", type=int, default=1e8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--centralized-expl", action="store_true")
    parser.add_argument("--render", action="store_true")

    args = parser.parse_args()

    main(**vars(args))
