"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value


# START EDITING HERE
# You can use this space to define any helper functions that you need
def KL(x, y):
    if y>=1:
        return -1

    return (x+0.0001)*math.log((x+0.0001)/(y+0.0001)) + (1.0001-x)*math.log((1.0001-x)/(1.0001-y))

def find_bound(value, time, count):

    # Linear time algorithm = 758 seconds
    # Log time algorithm = 68 seconds

    target = (math.log(time) + 3*math.log(math.log(time)))/(0.001 + count)
    values = np.arange(value,1,0.01)
    if value > 0.99:
        return 1
    low = 0
    high = len(values) - 1
    mid = 0
 
    while low < high:
 
        mid = (high + low) // 2
 
        if KL(value, values[mid]) < target:
            low = mid + 1
        elif KL(value, values[mid]) > target:
            high = mid - 1
        else:
            return values[mid]
    
    return values[low]

    # for val in values:
    #     if KL(value, val) <= target and (KL(value, val+0.01) > target or KL(value,val+0.01) < 0):
    #         return val

    # return 1

# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.UCB_bounds = np.zeros(num_arms)
        self.time = 0
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        return np.argmax(self.UCB_bounds)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.time += 1
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value
        self.UCB_bounds = [i + math.sqrt(2*math.log(self.time)/ (0.001 + self.counts[index])) for index, i in enumerate(self.values)]
        # END EDITING HERE

class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.UCB_bounds = np.zeros(num_arms)
        self.time = 1
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        return np.argmax(self.UCB_bounds)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.time += 1
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value
        self.UCB_bounds = [find_bound(self.values[i], self.time, self.counts[i]) for i in range(self.num_arms)]
        # END EDITING HERE


class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.alphas = np.ones(num_arms)
        self.betas = np.ones(num_arms)
        # np.random.seed(0)

        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        # sampler_one = np.random.rand(self.num_arms)
        # sampler_two = np.random.rand(self.num_arms)
        samples = np.zeros(self.num_arms)

        for i in range(self.num_arms):
            samples[i] = np.random.beta(self.alphas[i], self.betas[i])

        # for index, sample in enumerate(sampler_one):
        #     samples[index] = (1 + self.alphas[index] + np.tan(180*(sample - 0.5)))/(0.001 + (1 + self.alphas[index] + np.tan(180*(sampler_two[index] - 0.5))) + (1 + self.betas[index] + np.tan(180*(sampler_two[index] - 0.5))))
        
        return np.argmax(samples)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        if reward:
            self.alphas[arm_index] += 1
        else:
            self.betas[arm_index] += 1
        # END EDITING HERE
