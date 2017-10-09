from matplotlib import pyplot as plt
import numpy as np
import os


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
    for (x, y), label in zip(train_data, labels):
        # plot the data, altering color based on label
        plt.scatter(x, y, c=("red" if label == 0 else "blue"))
    # show the graph
    plt.show()
