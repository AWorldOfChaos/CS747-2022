import numpy as np
import argparse

class EncodedMdp:
    def __init__(self, mB, mR, q):
        self.maxBalls = mB
        self.maxRuns = mR
        self.q = q
        self.perBatterStates = (mB+1)*(mR+1)
        self.numStates = (mB+1)*(mR+1)*2
        self.numActions = 5
        self.mdpType = "episodic"
        self.discount = 1

        self.end = []
        for i in range(mB+1):
            self.end.append(i*(mR+1))
            self.end.append(i*(mR+1) + self.perBatterStates)
        for i in range(mR):
            self.end.append(i+1)
            self.end.append(i+1 + self.perBatterStates)
        
        self.transitions = []


    def makeTransitions(self, parameters):
        for i in range(self.maxBalls):
            for j in range(self.maxRuns):
                curr_state_num = (i+1)*(self.maxRuns+1) + j + 1
                counter = 0
                if i % 6 == 0:
                    counter += 1

                if j == 0:
                    for k in range(self.numActions):
                        self.transitions.append([curr_state_num, k, curr_state_num - 1 - j - (self.maxRuns+1), 0, parameters[k][0]])
                        self.transitions.append([curr_state_num, k, curr_state_num + self.perBatterStates*counter - (self.maxRuns+1), 0, parameters[k][1]])
                        self.transitions.append([curr_state_num, k, curr_state_num - 1 - j - (self.maxRuns+1), 1, 1 - parameters[k][0] - parameters[k][1]])

                        self.transitions.append([curr_state_num + self.perBatterStates, k, curr_state_num + self.perBatterStates - 1 - j - (self.maxRuns+1), 0, self.q])
                        self.transitions.append([curr_state_num + self.perBatterStates, k, curr_state_num + self.perBatterStates - 1 - j - (self.maxRuns+1), 1, (1 - self.q)/2])
                        self.transitions.append([curr_state_num + self.perBatterStates, k, curr_state_num + self.perBatterStates*(1-counter) - (self.maxRuns+1), 0, (1 - self.q)/2])

                elif j == 1:
                    for k in range(self.numActions):
                        self.transitions.append([curr_state_num, k, curr_state_num - 1 - j - (self.maxRuns+1), 0, parameters[k][0]])
                        self.transitions.append([curr_state_num, k, curr_state_num + self.perBatterStates*counter - (self.maxRuns+1), 0, parameters[k][1]])
                        self.transitions.append([curr_state_num, k, curr_state_num + self.perBatterStates*(1 - counter) - 1 - (self.maxRuns+1), 0, parameters[k][2]])
                        self.transitions.append([curr_state_num, k, curr_state_num - 1 - j - (self.maxRuns+1), 1, 1 - parameters[k][0] - parameters[k][1] - parameters[k][2]])

                        self.transitions.append([curr_state_num + self.perBatterStates, k, curr_state_num + self.perBatterStates - 1 - j - (self.maxRuns+1), 0, self.q])
                        self.transitions.append([curr_state_num + self.perBatterStates, k, curr_state_num + self.perBatterStates*(1-counter) - (self.maxRuns+1), 0, (1 - self.q)/2])
                        self.transitions.append([curr_state_num + self.perBatterStates, k, curr_state_num + self.perBatterStates*counter - 1 - (self.maxRuns+1), 0, (1 - self.q)/2])
                elif j == 2:
                    for k in range(self.numActions):
                        self.transitions.append([curr_state_num, k, curr_state_num - 1 - j - (self.maxRuns+1), 0, parameters[k][0]])
                        self.transitions.append([curr_state_num, k, curr_state_num + self.perBatterStates*counter - (self.maxRuns+1), 0, parameters[k][1]])
                        self.transitions.append([curr_state_num, k, curr_state_num + self.perBatterStates*(1 - counter) - 1 - (self.maxRuns+1), 0, parameters[k][2]])
                        self.transitions.append([curr_state_num, k, curr_state_num + self.perBatterStates*counter - 2 - (self.maxRuns+1), 0, parameters[k][3]])
                        self.transitions.append([curr_state_num, k, curr_state_num - 1 - j - (self.maxRuns+1), 1, 1 - parameters[k][0] - parameters[k][1] - parameters[k][2] - parameters[k][3]])

                        self.transitions.append([curr_state_num + self.perBatterStates, k, curr_state_num + self.perBatterStates - 1 - j - (self.maxRuns+1), 0, self.q])
                        self.transitions.append([curr_state_num + self.perBatterStates, k, curr_state_num + self.perBatterStates*(1-counter) - (self.maxRuns+1), 0, (1 - self.q)/2])
                        self.transitions.append([curr_state_num + self.perBatterStates, k, curr_state_num + self.perBatterStates*counter - 1 - (self.maxRuns+1), 0, (1 - self.q)/2])
                elif j == 3:
                    for k in range(self.numActions):
                        self.transitions.append([curr_state_num, k, curr_state_num - 1 - j - (self.maxRuns+1), 0, parameters[k][0]])
                        self.transitions.append([curr_state_num, k, curr_state_num + self.perBatterStates*counter - (self.maxRuns+1), 0, parameters[k][1]])
                        self.transitions.append([curr_state_num, k, curr_state_num + self.perBatterStates*(1 - counter) - 1 - (self.maxRuns+1), 0, parameters[k][2]])
                        self.transitions.append([curr_state_num, k, curr_state_num + self.perBatterStates*counter - 2 - (self.maxRuns+1), 0, parameters[k][3]])
                        self.transitions.append([curr_state_num, k, curr_state_num + self.perBatterStates*(1-counter) - 3 - (self.maxRuns+1), 0, parameters[k][4]])
                        self.transitions.append([curr_state_num, k, curr_state_num - 1 - j - (self.maxRuns+1), 1, 1 - parameters[k][0] - parameters[k][1] - parameters[k][2] - parameters[k][3] - parameters[k][4]])

                        self.transitions.append([curr_state_num + self.perBatterStates, k, curr_state_num + self.perBatterStates - 1 - j - (self.maxRuns+1), 0, self.q])
                        self.transitions.append([curr_state_num + self.perBatterStates, k, curr_state_num + self.perBatterStates*(1-counter) - (self.maxRuns+1), 0, (1 - self.q)/2])
                        self.transitions.append([curr_state_num + self.perBatterStates, k, curr_state_num + self.perBatterStates*counter - 1 - (self.maxRuns+1), 0, (1 - self.q)/2])
                elif j == 4 or j == 5:
                    for k in range(self.numActions):
                        self.transitions.append([curr_state_num, k, curr_state_num - 1 - j - (self.maxRuns+1), 0, parameters[k][0]])
                        self.transitions.append([curr_state_num, k, curr_state_num + self.perBatterStates*counter - (self.maxRuns+1), 0, parameters[k][1]])
                        self.transitions.append([curr_state_num, k, curr_state_num + self.perBatterStates*(1 - counter) - 1 - (self.maxRuns+1), 0, parameters[k][2]])
                        self.transitions.append([curr_state_num, k, curr_state_num + self.perBatterStates*counter - 2 - (self.maxRuns+1), 0, parameters[k][3]])
                        self.transitions.append([curr_state_num, k, curr_state_num + self.perBatterStates*(1-counter) - 3 - (self.maxRuns+1), 0, parameters[k][4]])
                        self.transitions.append([curr_state_num, k, curr_state_num + self.perBatterStates*counter - 4 - (self.maxRuns+1), 0, parameters[k][5]])
                        self.transitions.append([curr_state_num, k, curr_state_num - 1 - j - (self.maxRuns+1), 1, 1 - parameters[k][0] - parameters[k][1] - parameters[k][2] - parameters[k][3] - parameters[k][4] - parameters[k][5]])

                        self.transitions.append([curr_state_num + self.perBatterStates, k, curr_state_num + self.perBatterStates - 1 - j - (self.maxRuns+1), 0, self.q])
                        self.transitions.append([curr_state_num + self.perBatterStates, k, curr_state_num + self.perBatterStates*(1-counter) - (self.maxRuns+1), 0, (1 - self.q)/2])
                        self.transitions.append([curr_state_num + self.perBatterStates, k, curr_state_num + self.perBatterStates*counter - 1 - (self.maxRuns+1), 0, (1 - self.q)/2])
                else:
                    for k in range(self.numActions):
                        self.transitions.append([curr_state_num, k, curr_state_num - 1 - j - (self.maxRuns+1), 0, parameters[k][0]])
                        self.transitions.append([curr_state_num, k, curr_state_num + self.perBatterStates*counter - (self.maxRuns+1), 0, parameters[k][1]])
                        self.transitions.append([curr_state_num, k, curr_state_num + self.perBatterStates*(1 - counter) - 1 - (self.maxRuns+1), 0, parameters[k][2]])
                        self.transitions.append([curr_state_num, k, curr_state_num + self.perBatterStates*counter - 2 - (self.maxRuns+1), 0, parameters[k][3]])
                        self.transitions.append([curr_state_num, k, curr_state_num + self.perBatterStates*(1-counter) - 3 - (self.maxRuns+1), 0, parameters[k][4]])
                        self.transitions.append([curr_state_num, k, curr_state_num + self.perBatterStates*counter - 4 - (self.maxRuns+1), 0, parameters[k][5]])
                        self.transitions.append([curr_state_num, k, curr_state_num + self.perBatterStates*counter - 6 - (self.maxRuns+1), 0, parameters[k][6]])

                        self.transitions.append([curr_state_num + self.perBatterStates, k, curr_state_num + self.perBatterStates - 1 - j - (self.maxRuns+1), 0, self.q])
                        self.transitions.append([curr_state_num + self.perBatterStates, k, curr_state_num + self.perBatterStates*(1-counter) - (self.maxRuns+1), 0, (1 - self.q)/2])
                        self.transitions.append([curr_state_num + self.perBatterStates, k, curr_state_num + self.perBatterStates*counter - 1 - (self.maxRuns+1), 0, (1 - self.q)/2])
                

    def printDetails(self):
        print("numStates", self.numStates)
        print("numActions", self.numActions)

        print("end", end=" ")
        for e in self.end:
            print(e,end=" ")

        print("")

        for transition in self.transitions:
            print("transition", transition[0], transition[1], transition[2], transition[3], transition[4])

        print("mdptype", self.mdpType)

        print("discount", self.discount)

        


parser = argparse.ArgumentParser()
parser.add_argument("--states", type=str, required=True, help="Enter the path to the state file")
parser.add_argument("--parameters", type=str, required=True, help="Enter the path to the parameters")
parser.add_argument("--q", type=float, required=True, help="Enter q")
args = parser.parse_args()

parameters = np.zeros([5,7])
ind = -1

with open(args.parameters) as f:
    for line in f:
        line = line.rstrip()
        words = line.split()
        if words[0] == "action":
            continue
        else:
            ind += 1
            for i in range(7):
                parameters[ind][i] = float(words[i+1])

maxBalls = 0
maxRuns = 0

with open(args.states) as f:
    for line in f:
        line = line.rstrip()
        maxBalls = int(line[0:2])
        maxRuns = int(line[2:])
        break

mdp = EncodedMdp(maxBalls, maxRuns, args.q)

mdp.makeTransitions(parameters)

mdp.printDetails()
