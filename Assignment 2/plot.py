import matplotlib.pyplot as plt
import argparse

x1 = [i for i in range(100)]
y1a = []
y1b = []

x2 = [i+1 for i in range(20)]
y2a = []
y2b = []

x3 = [i+1 for i in range(15)]
y3a = []
y3b = []

parser = argparse.ArgumentParser()
parser.add_argument("--parameter", type=str, required=True, help="Enter the parameter to be plotted")
args = parser.parse_args()

with open("result1a.txt") as f:
    for line in f:
        line = line.rstrip()
        words = line.split()
        y1a.append(float(words[2]))

with open("result1b.txt") as f:
    for line in f:
        line = line.rstrip()
        words = line.split()
        y1b.append(float(words[2]))

with open("result2a.txt") as f:
    for line in f:
        line = line.rstrip()
        words = line.split()
        y2a.append(float(words[2]))

with open("result2b.txt") as f:
    for line in f:
        line = line.rstrip()
        words = line.split()
        y2b.append(float(words[2]))

with open("result3a.txt") as f:
    for line in f:
        line = line.rstrip()
        words = line.split()
        y3a.append(float(words[2]))

with open("result3b.txt") as f:
    for line in f:
        line = line.rstrip()
        words = line.split()
        y3b.append(float(words[2]))
  


def plot(x, ya, yb, lab, title, save):
    # Plotting both the curves simultaneously
    plt.plot(x, ya, color='r', label='opt')
    plt.plot(x, yb, color='g', label='ran')
    
    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel(lab)
    plt.ylabel("Probability")
    plt.title(title)
    
    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()
    
    # To load the display window
    plt.savefig(save)


if args.parameter == "q":
    plot(x1,y1a,y1b,"q-value","","q.png")
elif args.parameter == "Runs":
    plot(x2,y2a,y2b,"Runs","","Runs.png")
elif args.parameter == "Balls":
    plot(x3,y3a,y3b,"Balls","","Balls.png")