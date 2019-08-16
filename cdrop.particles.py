import argparse
import copy
import pickle
import random

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.functional import F
from tqdm import tqdm

# import multiagent.scenarios as scenarios
# from multiagent.environment import MultiAgentEnv
from torch.optim import Adam

import lbforaging
from replay import MultiAgentReplayBuffer
from networks import FCNetwork

USE_GPU = False

device = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")
MSELoss = torch.nn.MSELoss()
MAX_EPISODES = 100000
PARTICLE_MAX_STEP = 25


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


def make_env(scenario_name, benchmark=False):
    """
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.
    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)
    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    """
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(
            world,
            scenario.reset_world,
            scenario.reward,
            scenario.observation,
            scenario.benchmark_data,
        )
    else:
        env = MultiAgentEnv(
            world, scenario.reset_world, scenario.reward, scenario.observation
        )
    return env


class MADDPG:
    def __init__(self, params):
        state_sizes = params["state_space"]
        action_sizes = params["action_space"]
        print(state_sizes)
        print(action_sizes)
        agent_count = len(action_sizes)

        # self.discrete_actions = [True, True]

        self.gamma = params["discount"]
        self.update_freq = params["update_freq"]
        self.batch_size = params["batch_size"]
        self.target_tau = params["target_tau"]
        self.grad_clip = params["grad_clip"]
        self.dropout = params["dropout"]

        self.timesteps = 0

        self.actors = [
            FCNetwork(
                (s.shape[0], *params["network_size"], a.n), dropout=self.dropout
            ).to(device)
            for s, a in zip(state_sizes, action_sizes)
        ]

        critic_input_size = sum([s.shape[0] for s in state_sizes]) + sum(
            [a.n for a in action_sizes]
        )
        self.critics = [
            FCNetwork(
                (critic_input_size, *params["network_size"], 1), dropout=False
            ).to(device)
            for _ in range(agent_count)
        ]
        self.ex_actors = [
            FCNetwork(
                (s.shape[0] + other_action.n, *params["network_size"], a.n),
                dropout=self.dropout,
            ).to(device)
            for s, a, other_action in zip(
                state_sizes,
                action_sizes,
                reversed(action_sizes),  # todo reversed only works for 2 agents
            )
        ]

        self.actor_targets = copy.deepcopy(self.actors)
        self.critic_targets = copy.deepcopy(self.critics)

        self.critic_optimizers = [
            Adam(x.parameters(), lr=params["critic_lr"]) for x in self.critics
        ]
        self.actor_optimizers = [
            Adam(x.parameters(), lr=params["actor_lr"]) for x in self.actors
        ]
        self.ex_actor_optimizers = [
            Adam(x.parameters(), lr=params["actor_lr"]) for x in self.ex_actors
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
        if not explore:
            actions = []
            for i, state in enumerate(states):
                sb = torch.Tensor(state).to(device)
                action = onehot_from_logits(self.actors[i](sb.unsqueeze(0))).to("cpu")
                actions.append(action.squeeze().detach())
            return actions

        actions = []
        sb = torch.Tensor(states[0]).to(device)
        action = onehot_from_logits(self.actors[0](sb.unsqueeze(0)))
        actions.append(action.to("cpu").squeeze().detach())
        sb = torch.Tensor(states[1]).to(device)
        ex_in = torch.cat([sb.unsqueeze(0), action], dim=1)
        action = onehot_from_logits(self.ex_actors[1](ex_in)).to("cpu")
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

        ######## Update Exploration Actors ###########

        ex_actor_in = torch.cat(
            [nbatch[agent].states, nbatch[not agent].actions], dim=1
        )
        logits = self.ex_actors[agent](ex_actor_in)
        jactions = torch.cat(
            [
                gumbel_softmax(logits, hard=True) if i == agent else batch.actions
                for i, batch in enumerate(nbatch)
            ],
            dim=1,
        )
        critic_in = torch.cat([jstates, jactions], dim=1)
        qvalue = self.critics[agent](critic_in)

        self.ex_actor_optimizers[agent].zero_grad()
        loss = -torch.mean(qvalue) + 0.01 * (logits ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actors[agent].parameters(), self.grad_clip)
        self.ex_actor_optimizers[agent].step()

    def play_episode(self, env, evaluate=False, render=False):
        obs_n = env.reset()
        episode_rewards = np.zeros(self.agent_count)
        done = False
        stp = 0

        while not done and stp < PARTICLE_MAX_STEP:
            stp += 1
            # query for action from each agent's policy
            # act_n = [np.array([0, 0, 1]), env.action_space[1].sample()]
            actions = [
                a.detach().numpy()
                for a in self.select_actions(obs_n, explore=(not evaluate))
            ]
            # if render:
            #     print(actions)

            # step environment
            next_obs_n, reward_n, done_n, _ = env.step(actions)
            done = np.all(done_n)

            episode_rewards += reward_n

            self.replay_buffer.push(
                [o.astype(np.float32) for o in obs_n],
                actions,
                [o.astype(np.float32) for o in next_obs_n],
                [np.array([r], dtype=np.float32) for r in reward_n],
                [np.array([d], dtype=np.float32) for d in done_n],
            )
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


def plot_rewards(r1, r2, eval_every):

    ma_length = 10

    plt.figure(1)
    plt.clf()
    # plt.yscale("symlog")
    plt.title("Training...")
    plt.xlabel("Timestep")
    plt.ylabel("Reward")

    x = np.arange(eval_every * len(np.sum(r1, axis=1)), step=eval_every)
    y = np.sum(r1, axis=1)
    yMA = movingaverage(y, ma_length)
    plt.plot(x, y, "x", label="Dropout Exploration", color="darkorange")

    if len(y) > ma_length:
        plt.plot(
            x[len(x) - len(yMA) :],
            yMA,
            color="darkorange",
            label="Dropout Exploration (MA10)",
        )

    x = np.arange(eval_every * len(np.sum(r2, axis=1)), step=eval_every)
    y = np.sum(r2, axis=1)
    yMA = movingaverage(y, ma_length)

    plt.plot(x, y, "x", label="Coordinated Dropout Exploration", color="b")
    if len(y) > ma_length:
        plt.plot(
            x[len(x) - len(yMA) :],
            yMA,
            color="b",
            label="Coordinated Dropout Exploration (MA10)",
        )

    plt.legend()
    plt.title("Foraging-8x8-2p-2f-v0")
    plt.savefig("cdrop.svg")
    plt.pause(0.001)  # pause a bit so that plots are updated


def evaluate(env, maddpg, rewards, eval_episodes, eval_every, render):
    episode_rewards1 = np.zeros(maddpg.agent_count)
    episode_rewards2 = np.zeros(maddpg.agent_count)
    r1, r2 = rewards
    for i in range(eval_episodes):
        episode_rewards1 += (
            maddpg.play_episode(env, evaluate=True, render=True) / eval_episodes
        )
    for i in range(eval_episodes):
        episode_rewards2 += (
            maddpg.play_episode(env, evaluate=False, render=False) / eval_episodes
        )

    r1 = (
        np.vstack([r1, episode_rewards1])
        if r1 is not None
        else np.array([episode_rewards1])
    )
    r2 = (
        np.vstack([r2, episode_rewards2])
        if r2 is not None
        else np.array([episode_rewards2])
    )
    if render:
        plot_rewards(r1, r2, eval_every)
    return r1, r2


def main(**params):

    plt.ion()
    rewards = None, None
    torch.set_num_threads(1)
    seed = params["seed"]

    # env = gym.make(params["gym_env"])
    env = make_env("simple_speaker_listener")

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

    with tqdm(
        total=params["max_timesteps"], smoothing=0.3, disable=not params["render"]
    ) as pbar:
        while maddpg.timesteps <= params["max_timesteps"]:
            _ = maddpg.play_episode(env, evaluate=False, render=False)
            # pbar.update(env.current_step)
            pbar.update(PARTICLE_MAX_STEP)
            if maddpg.timesteps - last_eval >= params["eval_every"]:
                pbar.set_description("Evaluating...")
                last_eval = maddpg.timesteps
                rewards = evaluate(
                    env,
                    maddpg,
                    rewards,
                    params["eval_episodes"],
                    params["eval_every"],
                    render=params["render"],
                )
                pbar.set_description("Training...")
                saved_networks.append([a.state_dict() for a in maddpg.actors])
                # print(maddpg.timesteps)

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
    parser.add_argument("--gym-env", default="Foraging-8x8-2p-2f-v0", type=str)

    # parser.add_argument(
    #     "--expl-method",
    #     choices=["egreedy", "coord_egreedy", "gumbel", "dropout"],
    #     default="egreedy",
    #     type=str,
    # )
    parser.add_argument("--seed", default=None)
    parser.add_argument("--max-timesteps", default=1e7, type=float)
    parser.add_argument("--update-freq", type=float, default=100)
    parser.add_argument("--target-tau", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--network-size", default=(64, 64))
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--critic-lr", type=float, default=0.001)
    parser.add_argument("--actor-lr", type=float, default=0.0001)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=10000)
    parser.add_argument("--buffer-size", type=int, default=1e8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--render", action="store_true")

    args = parser.parse_args()

    data = main(**vars(args))
    pickle.dump(data, open("egreedy.local.p", "wb"))
