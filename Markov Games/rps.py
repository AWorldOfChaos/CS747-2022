import numpy as np

class Agent():
    def __init__(self) -> None:
        self.policy_map = np.array([1/2,1/3,1/6])
        self.reward = 0
        self.insight = np.zeros([3,3])

    def update_policy(self, reward, action, action_other, time):

        self.insight[action_other][action] = self.insight[action_other][action]*(time-1)/time + reward/time
        maxmin = np.min(self.insight, axis=0)
        best_val = np.max(maxmin)

        self.policy_map[maxmin - best_val > -0.001] += 1/time
        self.policy_map /= np.sum(self.policy_map)

    def play(self, time):
        decider = np.random.rand()
        if decider < 1/time:
            return np.random.randint(3)

        s = np.sum(self.policy_map)
        a = np.random.rand()
        
        if a < self.policy_map[0]:
            return 0
        elif a < self.policy_map[0] + self.policy_map[1]:
            return 1
        else:
            return 2


class Game():
    def __init__(self) -> None:
        self.reward_scheme = np.array([[0, 1, -1],[-1, 0, 1],[1, -1, 0]])
        self.a1 = Agent()
        self.a2 = Agent()

    def playGame(self,time):
        move_one = self.a1.play(time)
        move_two = self.a2.play(time)
        self.a1.reward += self.reward_scheme[move_two][move_one]
        self.a2.reward += self.reward_scheme[move_one][move_two]

        self.a1.update_policy(self.reward_scheme[move_two][move_one], move_one, move_two, time)
        self.a2.update_policy(self.reward_scheme[move_one][move_two], move_two, move_one, time)

        return True

game = Game()

for i in range(10000):
    game.playGame((i+1)/100)

    if i % 1000 == 0 and i != 0:
        print("Agent 1: ", game.a1.reward/i, "Agent 2: ", game.a2.reward/i)
    

print(game.a1.policy_map, "\n", game.a1.insight)
