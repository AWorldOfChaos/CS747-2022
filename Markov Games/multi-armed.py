import numpy as np
import matplotlib.pyplot as plt
import argparse

class Bandit():
    def __init__(self) -> None:
        self.mean = np.random.rand()

    def pull_arm(self) -> int:
        return np.random.rand() < self.mean

    
class E1():
    def __init__(self, horizon) -> None:
        np.random.seed(100)
        self.num_bandits = 100

        self.bandits = []
        for i in range(self.num_bandits):
            a = Bandit()
            self.bandits.append(a)

        self.num_pulled = np.zeros(self.num_bandits)
        self.means = np.zeros(self.num_bandits)
        self.epsilon = 0.1
        self.horizon = horizon

        means = []
        for i in self.bandits:
            means.append(i.mean)
        self.maximum_mean = np.max(np.array(means))

    def choose_arm(self, time):
        if time < self.horizon*self.epsilon:
            return np.random.randint(self.num_bandits)
        else:
            return np.argmax(self.means)

    def update(self, time, reward, arm_index):
        self.means[arm_index] = (self.means[arm_index] * self.num_pulled[arm_index] + reward)/(self.num_pulled[arm_index] + 1)
        self.num_pulled[arm_index] += 1
    
    def compute_regret(self):
        total_reward = 0
        for time in range(self.horizon):
            arm_index = self.choose_arm(time)
            reward = self.bandits[arm_index].pull_arm()
            total_reward += reward
            self.update(time, reward, arm_index)

        return self.horizon*self.maximum_mean - total_reward

class UCB():
    def __init__(self, horizon) -> None:
        np.random.seed(100)
        self.num_bandits = 100

        self.bandits = []
        for i in range(self.num_bandits):
            a = Bandit()
            self.bandits.append(a)

        self.num_pulled = np.zeros(self.num_bandits)
        self.means = np.zeros(self.num_bandits)
        self.epsilon = 0.1
        self.horizon = horizon

        means = []
        for i in self.bandits:
            means.append(i.mean)
        self.maximum_mean = np.max(np.array(means))

    def choose_arm(self, time):
        return np.argmax(self.means + np.sqrt(2*np.log(time)/self.num_pulled+1))

    def update(self, time, reward, arm_index):
        self.means[arm_index] = (self.means[arm_index] * self.num_pulled[arm_index] + reward)/(self.num_pulled[arm_index] + 1)
        self.num_pulled[arm_index] += 1
    
    def compute_regret(self):
        total_reward = 0
        for time in range(self.horizon):
            arm_index = self.choose_arm(time)
            reward = self.bandits[arm_index].pull_arm()
            total_reward += reward
            self.update(time, reward, arm_index)

        return self.horizon*self.maximum_mean - total_reward


class Thompson():
    def __init__(self, horizon) -> None:
        np.random.seed(100)
        self.num_bandits = 100

        self.bandits = []
        for i in range(self.num_bandits):
            a = Bandit()
            self.bandits.append(a)

        self.num_pulled = np.zeros(self.num_bandits)
        self.successes = np.zeros(self.num_bandits)
        self.means = np.zeros(self.num_bandits)
        self.epsilon = 0.1
        self.horizon = horizon

        means = []
        for i in self.bandits:
            means.append(i.mean)
        self.maximum_mean = np.max(np.array(means))

    def choose_arm(self, time):
        x = []
        for i in range(self.num_bandits):
            x.append(np.random.beta(self.successes[i] + 1, self.num_pulled[i] - self.successes[i] + 1, 1))

        return np.argmax(np.array(x))

    def update(self, time, reward, arm_index):
        self.means[arm_index] = (self.means[arm_index] * self.num_pulled[arm_index] + reward)/(self.num_pulled[arm_index] + 1)
        self.num_pulled[arm_index] += 1
        if reward == 1:
            self.successes[arm_index] += 1
    
    def compute_regret(self):
        total_reward = 0
        for time in range(self.horizon):
            arm_index = self.choose_arm(time)
            reward = self.bandits[arm_index].pull_arm()
            total_reward += reward
            self.update(time, reward, arm_index)

        return self.horizon*self.maximum_mean - total_reward


class Finite():
    def __init__(self, horizon) -> None:
        np.random.seed(100)
        self.num_bandits = horizon

        self.bandits = []
        for i in range(self.num_bandits):
            a = Bandit()
            a.mean = i*1.0/self.num_bandits
            self.bandits.append(a)

        self.num_pulled = np.zeros(self.num_bandits)
        self.successes = np.zeros(self.num_bandits)
        self.means = np.zeros(self.num_bandits)
        self.epsilon = 0.1
        self.horizon = horizon
        self.last_chosen = 0

        means = []
        for i in self.bandits:
            means.append(i.mean)
        self.maximum_mean = np.max(np.array(means))

    def choose_arm(self, time):
        if self.means[self.last_chosen] <= 0.95:
            self.last_chosen = np.random.randint(self.num_bandits)
        return self.last_chosen

    def update(self, time, reward, arm_index):
        self.means[arm_index] = (self.means[arm_index] * self.num_pulled[arm_index] + reward)/(self.num_pulled[arm_index] + 1)
        self.num_pulled[arm_index] += 1
    
    def compute_regret(self):
        total_reward = 0
        for time in range(self.horizon):
            arm_index = self.choose_arm(time)
            reward = self.bandits[arm_index].pull_arm()
            total_reward += reward
            self.update(time, reward, arm_index)

        return self.horizon*self.maximum_mean - total_reward


parser = argparse.ArgumentParser()
parser.add_argument("--ucb", action='store_true')
parser.add_argument("--thompson", action='store_true')
parser.add_argument("--epsilon", action='store_true')
parser.add_argument("--finite", action='store_true')
args = parser.parse_args()

if args.epsilon:
    regrets = []
    horizon = []

    for i in range(200):
        print(i)
        horizon.append(i*100)
        e1 = E1(i*100)
        regrets.append(e1.compute_regret())

    plt.scatter(horizon, regrets)
    plt.show()

if args.ucb:
    regrets = []
    horizon = []

    for i in range(200):
        print(i)
        horizon.append(i*100)
        e1 = UCB(i*100)
        regrets.append(e1.compute_regret())

    plt.scatter(horizon, regrets)
    plt.show()

if args.thompson:
    regrets = []
    horizon = []

    for i in range(200):
        print(i)
        horizon.append(i*100)
        e1 = Thompson(i*100)
        regrets.append(e1.compute_regret())

    plt.scatter(horizon, regrets)
    plt.show()

if args.finite:
    regrets = []
    horizon = []

    for i in range(200):
        print(i)
        horizon.append(i*100+1)
        e1 = Finite(i*100+1)
        regrets.append(e1.compute_regret())

    plt.scatter(np.array(horizon), np.array(regrets)/np.array(horizon))
    plt.show()

