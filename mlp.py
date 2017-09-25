# Multilayer Perceptron algorithm
# implemented using backpropagation
# @author James Hurt, 2017
import numpy as np
import pandas as pd
import math


def init():
    input_file = pd.read_csv("data/cross_data.csv", header=None)
    train_data = input_file.iloc[:, :-1]
    labels = input_file.iloc[:, -1:]

    w1 = pd.read_csv("data/w1.csv", header=None)
    b1 = pd.read_csv("data/b1.csv", header=None)

    w2 = pd.read_csv("data/w2.csv", header=None)
    b2 = pd.read_csv("data/b2.csv", header=None)

    run(train_data, labels, w1, b1, w2, b2)


def fi(v):
    denom = 1 + math.e**(-1 * v)
    val = 1 / denom
    return val


def run(training_data, desired_output, w1, b1, w2, b2):
    training_data = training_data.values
    desired_output = desired_output.values
    w1 = w1.values
    w2 = w2.values
    b1 = b1.values
    b2 = b2.values
    for i, row in enumerate(training_data):
        for j, weights in enumerate(w1):
            v = np.dot(row, weights) + b1[j]
            output = fi(v)
            print(row)
            print(weights)
            print(b1[j])
            print(v)
            print(output)
            print("-------------------------")


if __name__ == "__main__":
    init()
