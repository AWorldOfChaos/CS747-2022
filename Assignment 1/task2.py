"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

You need to complete the following methods:
    - give_pull(self): This method is called when the algorithm needs to
        select the arms to pull for the next round. The method should return
        two arrays: the first array should contain the indices of the arms
        that need to be pulled, and the second array should contain how many
        times each arm needs to be pulled. For example, if the method returns
        ([0, 1], [2, 3]), then the first arm should be pulled 2 times, and the
        second arm should be pulled 3 times. Note that the sum of values in
        the second array should be equal to the batch size of the bandit.
    
    - get_reward(self, arm_rewards): This method is called just after the
        give_pull method. The method should update the algorithm's internal
        state based on the rewards that were received. arm_rewards is a dictionary
        from arm_indices to a list of rewards received. For example, if the
        give_pull method returned ([0, 1], [2, 3]), then arm_rewards will be
        {0: [r1, r2], 1: [r3, r4, r5]}. (r1 to r5 are each either 0 or 1.)
"""

import numpy as np

# START EDITING HERE
# You can use this space to define any helper functions that you need.
# END EDITING HERE

class AlgorithmBatched:
    def __init__(self, num_arms, horizon, batch_size):
        self.num_arms = num_arms
        self.horizon = horizon
        self.batch_size = batch_size
        assert self.horizon % self.batch_size == 0, "Horizon must be a multiple of batch size"
        # START EDITING HERE
        # Add any other variables you need here
        self.alphas = np.ones(num_arms)
        self.betas = np.ones(num_arms)
        # np.random.seed(0)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        samples = np.zeros(self.num_arms)
        indices = np.arange(0,self.num_arms,1)
        times = np.zeros(self.num_arms, dtype=int)

        for k in range(self.batch_size):
            for i in range(self.num_arms):
                samples[i] = np.random.beta(self.alphas[i], self.betas[i])
            
            a = np.argmax(samples)
            times[a] += 1
        
            
        # print(indices, times)
        return [indices, times]
        
        
        # return [0], [self.batch_size]
        # END EDITING HERE
    
    def get_reward(self, arm_rewards):
        # START EDITING HERE
        # print(arm_rewards)
        for x in arm_rewards:
            reward = arm_rewards[x]
            self.alphas[x] += len(reward[reward == 1])
            self.betas[x] += len(reward[reward == 0])
            
            
        pass
        # END EDITING HERE