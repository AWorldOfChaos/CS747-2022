import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--states", type=str, required=True, help="Enter the path to the state file")
parser.add_argument("--value-policy", type=str, required=True, help="Enter the path to the solution")
args = parser.parse_args()

maxBalls = 0
maxRuns = 0

with open(args.states) as f:
    for line in f:
        line = line.rstrip()
        maxBalls = int(line[0:2])
        maxRuns = int(line[2:])
        break

act = np.zeros([maxBalls+1,maxRuns+1])
val = np.zeros([maxBalls+1,maxRuns+1])

with open(args.value_policy) as f:
    count = 0
    for line in f:
        line = line.rstrip()
        words = line.split()
        state_index = [count // (maxRuns + 1), count % (maxRuns + 1)]
        a = words[1]
        if a == "3":
            a = "4"
        elif a == "4":
            a = "6"
        act[state_index[0], state_index[1]] = a
        val[state_index[0], state_index[1]] = words[0]
        count += 1
        if count == (maxRuns+1)*(maxBalls+1):
            break

with open(args.states) as f:
    for line in f:
        line = line.rstrip()
        balls = int(line[0:2])
        runs = int(line[2:])

        print(line, int(act[balls,runs]), str.format('{0:.6f}', val[balls,runs]))
        # print(line, int(act[balls,runs]), val[balls,runs])
