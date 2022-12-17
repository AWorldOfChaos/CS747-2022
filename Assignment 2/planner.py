import os
import argparse
import numpy as np
from pulp import *

class MDP:
    def __init__(self, inputfile):
        self.numStates = 0
        self.numActions = 0
        self.mdptype = 0
        self.discount = 1
        self.state_matrix_reward = []
        self.state_matrix_prob = []
        self.end_states = []
        self.mdpPath = inputfile
        self.state_action_mapping = 0
        self.state_values = 0

    def updateStateMatrix(self, data):
        data[1] = float(data[1])
        data[2] = float(data[2])
        data[3] = float(data[3])
        data[4] = float(data[4])
        data[5] = float(data[5])
        self.state_matrix_reward[int(data[1])][int(data[2])][int(data[3])] = float(data[4])
        self.state_matrix_prob[int(data[1])][int(data[2])][int(data[3])] = float(data[5])

    def createStateMatrix(self):
        self.state_matrix_reward = np.zeros((self.numStates,self.numActions,self.numStates))
        self.state_matrix_prob =  np.zeros((self.numStates,self.numActions,self.numStates))
        self.state_action_mapping = np.zeros((self.numStates, self.numActions))
        self.state_values = np.max(self.state_action_mapping, axis=1)

    def evaluatePolicy(self, policyFile):
        policy = []
        rewards = np.array(self.state_matrix_reward)
        probs = np.array(self.state_matrix_prob)
        rtoll = 0
        atol = 1e-11

        with open(policyFile) as f:
            for line in f:
                line = line.rstrip()
                words = line.split()
                policy.append(int(words[0]))

        policy = np.array(policy)
        
        if len(policy) != self.numStates:
            policy = np.zeros(self.numStates, dtype=int)
            maxRuns = 0
            upd = 1
            with open(policyFile) as f:
                for line in f:
                    line = line.rstrip()
                    words = line.split()
                    if upd == 1:
                        maxRuns = int(words[0][2:])
                        upd = 0
                    i = int(words[0][0:2])
                    j = int(words[0][2:])
                    a = int(words[1])
                    if a == 4:
                        a = 3
                    elif a == 6:
                        a = 4
                    if i*(maxRuns+1) + j < self.numStates:
                        policy[i*(maxRuns+1) + j] = a

        while True:
            self.state_action_mapping = np.sum(probs*(rewards + self.discount*self.state_values), axis=2)
            new_state_values = np.array([self.state_action_mapping[i][policy[i]] for i in range(self.numStates)])

            if np.allclose(np.array(new_state_values), np.array(self.state_values), rtoll, atol):
                break
            else:
                self.state_values = new_state_values.copy()

        for i in range(self.numStates):
            print(str.format('{0:.6f}', self.state_values[i]),int(np.argmax(self.state_action_mapping,axis=1)[i]))

        pass

    def value_iteration(self):
        rewards = np.array(self.state_matrix_reward)
        probs = np.array(self.state_matrix_prob)
        rtoll = 0
        atol = 1e-11
        while True:
            self.state_action_mapping = np.sum(probs*(rewards + self.discount*self.state_values), axis=2)
            new_state_values = np.max(self.state_action_mapping, axis=1)

            if np.allclose(np.array(new_state_values), np.array(self.state_values), rtoll, atol):
                break
            else:
                self.state_values = new_state_values.copy()

        for i in range(self.numStates):
            print(str.format('{0:.6f}', self.state_values[i]),int(np.argmax(self.state_action_mapping,axis=1)[i]))

    def policy_iteration(self):
        rewards = np.array(self.state_matrix_reward)
        probs = np.array(self.state_matrix_prob)
        rtoll = 0
        atol = 1e-11
        curr_policy = np.zeros(self.numStates, dtype=int)
        new_policy = np.zeros(self.numStates, dtype=int)
        while True:
            curr_policy = new_policy.copy()
            while True:
                self.state_action_mapping = np.sum(probs*(rewards + self.discount*self.state_values), axis=2)
                new_state_values = np.array([self.state_action_mapping[i][curr_policy[i]] for i in range(self.numStates)])

                if np.allclose(np.array(new_state_values), np.array(self.state_values), rtoll, atol):
                    break
                else:
                    self.state_values = new_state_values.copy()

            new_policy = np.argmax(self.state_action_mapping,axis=1)
            if (new_policy==curr_policy).all():
                break

        for i in range(self.numStates):
            print(str.format('{0:.6f}', self.state_values[i]),int(np.argmax(self.state_action_mapping,axis=1)[i]))
        pass

    def linear_programming(self):
        problem = LpProblem('MDP', LpMaximize)

        # vars = [LpVariable("h") for i in range(self.numStates)]
        names = [str(i) for i in range(self.numStates)]

        vars = LpVariable.dicts("variable", names)
        
        #Objective Function
        problem += pulp.lpSum([-1*vars[name] for name in names]) , 'Objective Function'
        #Constraints
        for i, n in enumerate(names):
            for j in range(self.numActions):
                problem += pulp.lpSum([ self.state_matrix_prob[i,j,k]*(self.state_matrix_reward[i,j,k] + self.discount*vars[name]) for k, name in enumerate(names) ]) <= vars[n], "Must_bound_%s_%d" % (n,j)
        problem.solve(PULP_CBC_CMD(msg=0))

        self.state_values = np.array([float(value(vars[n])) for i,n in enumerate(names)])
        self.state_action_mapping = np.sum(self.state_matrix_prob*(self.state_matrix_reward + self.discount*self.state_values), axis=2)

        for i, name in enumerate(names):
            print(str.format('{0:.6f}', round(value(vars[name]), 6)), int(np.argmax(self.state_action_mapping,axis=1)[i]))
            
    def parseInput(self):
        with open(self.mdpPath) as f:
            for line in f:
                line = line.rstrip()
                words = line.split()
                if words[0] == "numStates":
                    self.numStates = int(words[1])
                elif words[0] == "numActions":
                    self.numActions = int(words[1])
                    self.createStateMatrix()
                elif words[0] == "transition":
                    self.updateStateMatrix(words)
                elif words[0] == "end":
                    if int(words[1]) == -1:
                        continue
                    else:
                        for i in range(len(words) - 1):
                            self.end_states.append(i+1)
                elif words[0] == "mdptype":
                    self.mdptype = words[1]
                else:
                    self.discount = float(words[1])


parser = argparse.ArgumentParser()
parser.add_argument("--mdp", type=str, required=True, help="Enter the path to the mdp file")
parser.add_argument("--algorithm", type=str, required=False, default="vi", help="Enter the path to the algorithm")
parser.add_argument("--policy", type=str, required=False, help="Enter the path to the policy")
args = parser.parse_args()

m = MDP(args.mdp)
m.parseInput()

if args.policy:
    m.evaluatePolicy(args.policy)
elif args.algorithm:
    if args.algorithm == "hpi":
        m.policy_iteration()
    elif args.algorithm == "lp":
        m.linear_programming()
    else:
        m.value_iteration()
else:
    m.value_iteration()