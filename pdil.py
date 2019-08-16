import numpy as np
import gym
from enum import Enum
from collections import defaultdict
from gym.envs.registration import register
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import pandas as pd

MAX_EPISODES = 1
EPISODE_LENGTH = 50
LEARNING_RATE = 0.1
SEEDS = 10


class Action(Enum):
    COOPERATE = 0
    DEFECT = 1


class IteratedPrisonerEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, itermax):
        self.itermax = itermax
        self.cur_iter = 0

    def step(self, actions):
        actions = [Action(a) for a in actions]

        if actions[0] == Action.COOPERATE and actions[1] == Action.COOPERATE:
            rewards = (-1, -1)
        elif actions[0] == Action.DEFECT and actions[1] == Action.DEFECT:
            rewards = (-2, -2)
        elif actions[0] == Action.COOPERATE and actions[1] == Action.DEFECT:
            rewards = (-3, 0)
        elif actions[0] == Action.DEFECT and actions[1] == Action.COOPERATE:
            rewards = (0, -3)

        nobs = (np.array([0]), np.array([0]))
        dones = (self.cur_iter >= self.itermax, self.cur_iter >= self.itermax)
        ninfo = [{} for _ in nobs]

        self.cur_iter += 1

        return nobs, rewards, dones, ninfo

    def reset(self):
        self.cur_iter = 0
        return (np.array([0]), np.array([0]))


class QAgent:
    def __init__(self):
        self.q_table = defaultdict(lambda: {action: 0.0 for action in Action})
        self.lr = LEARNING_RATE
        self.discount = 0.7

        self.epsilon = None

    def step(self, obs, rand=None):
        if not rand:
            rand = np.random.uniform()
        obs = tuple(obs)
        if rand < self.epsilon:
            return np.random.randint(0, 2)

        max_val = max([self.q_table[obs][a] for a in Action])
        choices = [a for a in Action if np.isclose(self.q_table[obs][a], max_val)]
        return random.choice(choices).value

    def update(self, state, action, reward, next_state):
        state = tuple(state)
        next_state = tuple(next_state)
        action = Action(action)

        delta = self.lr * (
            reward
            + self.discount * max([self.q_table[next_state][a] for a in Action])
            - self.q_table[state][action]
        )
        a = self.q_table[state][action]
        self.q_table[state][action] += delta
        # print(a - self.q_table[state][action])
        # print(a)


def main(epsilon, centralized):
    env = gym.make("IteratedPrisonerEnv-v0")

    agents = QAgent(), QAgent()
    for a in agents:
        a.epsilon = epsilon
    for _ in range(MAX_EPISODES):
        done = False
        nobs = env.reset()

        while not done:
            if centralized:
                rand = np.random.uniform()
            else:
                rand = None
            actions = [a.step(obs, rand=rand) for a, obs in zip(agents, nobs)]
            idx = (Action(actions[0]), Action(actions[1]))
            # idx = (Action.COOPERATE, Action.COOPERATE)
            df.loc[epsilon, centralized][idx] += 1
            nnext_obs, nrewards, ndones, _ = env.step(actions)

            nobs = nnext_obs
            done = ndones[0] and ndones[1]
            # if done:
            #     continue

            for i, a in enumerate(agents):
                a.update(nobs[i], actions[i], nrewards[i], nnext_obs[i])

    return (
        max(Action, key=lambda x: agents[0].q_table[(0,)][x]),
        max(Action, key=lambda x: agents[1].q_table[(0,)][x]),
    )


if __name__ == "__main__":
    register(
        id="IteratedPrisonerEnv-v0",
        entry_point="pdil:IteratedPrisonerEnv",
        kwargs={"itermax": EPISODE_LENGTH},
    )

    index = pd.MultiIndex.from_product(
        [np.arange(0.1, 1, 0.1), [True, False], range(SEEDS)],
        names=["epsilon", "centralised", "seed"],
    )
    df = pd.DataFrame(
        index=index,
        columns=[
            (Action.COOPERATE, Action.COOPERATE),
            (Action.COOPERATE, Action.DEFECT),
            (Action.DEFECT, Action.COOPERATE),
            (Action.DEFECT, Action.DEFECT),
        ],
        dtype=int,
    )
    df[:] = 0
    print(df)
    # r = defaultdict(lambda: np.zeros(9))
    for e in tqdm(np.arange(0.1, 1, 0.1)):
        for c in [True, False]:
            for i in range(SEEDS):
                conv = main(e, c)
                # print(conv)
                # df.loc[(e, c, i)][conv] += 1
                # r[c][int(e * 10 - 1)] += 1
    # df = df.groupby(level=["epsilon", "centralised"]).agg(["sum"])
    # print(df)
    # df.to_pickle("pdil.pkl")
    # plt.plot(np.arange(0.1, 1.0, 0.1), r[True] * 100, label="Centralized e-greedy")
    # plt.plot(np.arange(0.1, 1.0, 0.1), r[False] * 100, label="Non Centralized e-greedy")
    # plt.title("Iterated Prisoner's Dilemma (50 steps)")
    # plt.xlabel("epsilon")
    # plt.ylabel("Converged to optimal (C/C) %")
    # plt.legend()
    # plt.savefig("pdil.svg")
    # plt.show()
