from matplotlib import pyplot as plt
import numpy as np
import os
import math
from mlp import show_to_layer


def graph_init(error_per_epoch, train_data, labels, w1, w2, b1, b2):
    """
    Allow user to see graphs
    """
    # the input
    num = 0
    # sentinal
    while num != 3:
        # get user input
        num = int(
            input("Would you like to view: \n1. Error Per Epoch\n2. Data\n3. Exit\n>"))
        if num == 1:
            # grph error per epoch
            graph_error_per_epoch(error_per_epoch)
        elif num == 2:
            # graph the data
            graph_data_with_solution(train_data, labels, w1, w2, b1, b2)


def graph_error_per_epoch(error_per_epoch):
    """
    Graph the error per epoch
    """
    # plot the graph
    plt.plot(range(len(error_per_epoch)), error_per_epoch)
    # set title
    plt.title("Error Per Epoch")
    # set labels
    plt.ylabel("Error")
    plt.xlabel("Number of Epochs")
    # show the graph
    plt.show()


def graph_data_with_solution(train_data, labels, w1, w2, b1, b2):
    """
    Graph the data
    """
    # find the largest x value and largest y value
    biggest_x = -1
    biggest_y = -1
    for x, y in train_data:
        if x > biggest_x:
            biggest_x = x
        if y > biggest_y:
            biggest_y = y
    # create arrays going from - biggest x/y to + biggest x/y with interval .1
    x_array = np.arange(-biggest_x, biggest_x, .1)
    y_array = np.arange(-biggest_y, biggest_y, .1)
    # iterate through the arrays
    for i in x_array:
        for j in x_array:
            # create a datapoint
            datapoint = np.array([i, j])
            # show to hidden layer
            hidden = show_to_layer(datapoint, w1, b1)
            # show to output layer
            output = show_to_layer(hidden, w2, b2)[0]
            # values aren't exactly 1 and 0, need to round
            output = np.around(output)
            # plot the point
            plt.scatter(i, j, c="red" if output ==
                        0 else "blue", alpha=.2)

    for (x, y), label in zip(train_data, labels):
        # plot the data, altering color based on label
        plt.scatter(x, y, c=("red" if label == 0 else "blue"),
                    alpha=1, marker="+")
    # show the graph
    plt.show()
