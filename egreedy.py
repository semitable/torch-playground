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

from replay import PrioritizedMultiAgentReplayBuffer, MultiAgentReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor, device="cpu"):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False).to(device)
    return -torch.log(-torch.log(U + eps) + eps)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(
        logits.shape, tens_type=type(logits.data), device=logits.device
    )
    return F.softmax(y / temperature, dim=1)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=100.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y


class MADDPG:
    def __init__(self, params):
        state_sizes = params["state_space"]
        action_sizes = params["action_space"]
        agent_count = len(action_sizes)

        # self.discrete_actions = [True, True]

        self.gamma = params["discount"]
        self.update_freq = params["update_freq"]
        self.batch_size = params["batch_size"]
        self.target_tau = params["target_tau"]
        self.grad_clip = params["grad_clip"]
        self.epsilon_target = params["epsilon_target"]
        self.epsilon_anneal = params["epsilon_anneal"]
        self.centralized = params["centralized_expl"]

        self.timesteps = 0

        self.actors = [
            FCNetwork((s.shape[0], *params["network_size"], a.n)).to(device)
            for s, a in zip(state_sizes, action_sizes)
        ]

        critic_input_size = sum([s.shape[0] for s in state_sizes]) + sum(
            [a.n for a in action_sizes]
        )
        self.critics = [
            FCNetwork((critic_input_size, *params["network_size"], 1)).to(device)
            for _ in range(agent_count)
        ]

        self.actor_targets = copy.deepcopy(self.actors)
        self.critic_targets = copy.deepcopy(self.critics)

        self.critic_optimizers = [
            Adam(x.parameters(), lr=params["critic_lr"]) for x in self.critics
        ]
        self.actor_optimizers = [
            Adam(x.parameters(), lr=params["actor_lr"]) for x in self.actors
        ]

        self.agent_count = agent_count
        self.state_size = state_sizes
        self.action_size = action_sizes

        if params["prioritized_replay"]:
            self.replay_buffer = PrioritizedMultiAgentReplayBuffer(
                agent_count, params["buffer_size"], alpha=0.6
            )
        else:
            self.replay_buffer = MultiAgentReplayBuffer(
                agent_count, params["buffer_size"]
            )

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

            sb = torch.Tensor(state).to(device)
            if explore and rand < self.epsilon:
                action = gumbel_softmax(torch.ones(6).unsqueeze(0), hard=True)
            else:
                action = onehot_from_logits(self.actors[i](sb.unsqueeze(0))).to("cpu")
            actions.append(action.squeeze().detach())

        return actions

    def update(self, agent):
        """

        :param agent: The agent ID)
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        nbatch = self.replay_buffer.sample(self.batch_size, device=device)
        jstates = torch.cat([batch.states for batch in nbatch], dim=1)
        jactions = torch.cat([batch.actions for batch in nbatch], dim=1)
        jnext_states = torch.cat([batch.next_states for batch in nbatch], dim=1)

        target_actions = torch.cat(
            [
                onehot_from_logits(tp(batch.next_states))
                for tp, batch in zip(self.actor_targets, nbatch)
            ],
            dim=1,
        ).detach()

        target_critic_in = torch.cat((jnext_states, target_actions), dim=1)

        target_value = (
            nbatch[agent].rewards
            + self.gamma
            * (1 - nbatch[agent].done)
            * self.critic_targets[agent](target_critic_in)
        ).detach()

        critic_in = torch.cat((jstates, jactions), dim=1)
        value = self.critics[agent](critic_in)

        self.critic_optimizers[agent].zero_grad()
        loss = MSELoss(value, target_value)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics[agent].parameters(), self.grad_clip)
        self.critic_optimizers[agent].step()

        if self.replay_buffer.can_prioritize:
            errors = torch.abs(value - target_value).data.numpy().squeeze() + 0.001
            self.replay_buffer.update_priorities(errors)

        ########## Update Actors ##################

        logits = self.actors[agent](nbatch[agent].states)
        jactions = torch.cat(
            [
                gumbel_softmax(logits, hard=True) if i == agent else batch.actions
                for i, batch in enumerate(nbatch)
            ],
            dim=1,
        )

        critic_in = torch.cat([jstates, jactions], dim=1)

        qvalue = self.critics[agent](critic_in)

        self.actor_optimizers[agent].zero_grad()
        loss = -torch.mean(qvalue) + 0.01 * (logits ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actors[agent].parameters(), self.grad_clip)
        self.actor_optimizers[agent].step()

    def update_epsilon(self):
        ratio = min(1, float(self.timesteps) / self.epsilon_anneal)
        self.epsilon = ratio * self.epsilon_target + (1 - ratio) * 1

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
                for a in self.select_actions(obs_n, explore=(not evaluate))
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

            if self.timesteps % self.update_freq != 0:
                continue

            for agent in range(self.agent_count):
                self.update(agent)
            for i in range(self.agent_count):
                self.actor_targets[i].soft_update(self.actors[i], self.target_tau)
                self.critic_targets[i].soft_update(self.critics[i], self.target_tau)

        return episode_rewards


def movingaverage(values, window):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, "valid")
    return sma


def plot_rewards(rewards, eval_every):

    ma_length = 50

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
    episode_rewards = np.zeros(maddpg.agent_count)
    for i in range(eval_episodes):
        episode_rewards += (
            maddpg.play_episode(env, evaluate=True, render=render) / eval_episodes
        )

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
    maddpg = MADDPG(params)
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
            saved_networks.append([a.state_dict() for a in maddpg.actors])
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
    parser.add_argument("--gym-env", default="Foraging-8x8-2p-2f-v0", type=str)
    parser.add_argument("--seed", default=None)
    parser.add_argument("--max-timesteps", default=1e7, type=float)
    parser.add_argument("--update-freq", type=float, default=100)
    parser.add_argument("--target-tau", type=float, default=0.01)
    parser.add_argument("--network-size", default=(64, 64))
    parser.add_argument("--epsilon-target", type=float, default=0.1)
    parser.add_argument("--epsilon-anneal", type=float, default=1e7)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--critic-lr", type=float, default=0.0001)
    parser.add_argument("--actor-lr", type=float, default=0.00001)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=10000)
    parser.add_argument("--buffer-size", type=int, default=1e7)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--centralized-expl", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--prioritized-replay", action="store_true")

    args = parser.parse_args()

    data = main(**vars(args))
    pickle.dump(data, open("egreedy.local.p", "wb"))
