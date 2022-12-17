import numpy as np

c = 100

class Agent():
    def __init__(self, numActions) -> None:
        self.numActions = numActions
        self.policy_map = np.ones(self.numActions)
        self.policy_map = self.policy_map/self.numActions
        self.reward = 0
        self.insight = np.zeros([self.numActions,self.numActions])

    def update_policy(self, reward, action, action_other, p, time):
        self.insight[action_other][action] = self.insight[action_other][action]*(c*time-1)/(c*time) + reward/(c*time)
        maxmin = np.min(self.insight*np.tile(p, (self.numActions, 1)).T, axis=0)
        best_val = np.max(maxmin)

        self.policy_map[maxmin - best_val > -0.001] += 1/(time)
        self.policy_map /= np.sum(self.policy_map)

    def play(self, time):
        decider = np.random.rand()
        if decider < 1/time:
            return np.random.randint(self.numActions)

        s = np.sum(self.policy_map)
        a = np.random.rand()
        
        for i in range(self.numActions):
            if a < np.sum(self.policy_map[:i+1]):
                return i
        


class Game():
    def __init__(self) -> None:
        np.random.seed(10)
        self.numActions = 3
        self.reward_scheme = np.random.randint(low = -10, high=10, size = [self.numActions, self.numActions])
        self.a1 = Agent(self.numActions)
        self.a2 = Agent(self.numActions)

    def playGame(self,time):
        move_one = self.a1.play(time)
        move_two = self.a2.play(time)
        self.a1.reward += self.reward_scheme[move_two][move_one]
        self.a2.reward += -1*self.reward_scheme[move_two][move_one]
        p1 = self.a1.policy_map
        p2 = self.a2.policy_map

        self.a1.update_policy(self.reward_scheme[move_two][move_one], move_one, move_two, p2, time)
        self.a2.update_policy(-1*self.reward_scheme[move_two][move_one], move_two, move_one, p1, time)

        return True

game = Game()

for i in range(1000000):
    game.playGame((i+1)/c)

    if i % 10000 == 0 and i != 0:
        print("Agent 1: ", game.a1.reward/i, "Agent 2: ", game.a2.reward/i)
    

print(game.a1.policy_map, game.a2.policy_map, game.a1.insight.T, game.a2.insight, game.reward_scheme, sep='\n')
