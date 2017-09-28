# Multilayer Perceptron algorithm
# implemented using backpropagation
# @author James Hurt, 2017
import numpy as np
import pandas as pd
import math
import argparse
import os


def init():
    """
    Initialize the program by getting all input data
    """
    # parse command line arguments
    num_dim, num_hidden, num_output, partA = getInputArgs()

    if not partA:
        # read in filenames
        input_filename = input("Enter train data filename: ")
        w1_filename = input(
            "Enter weights filename from input to first hidden layer: ")
        w2_filename = input("Enter weights from hidden layer to output: ")
        b1_filename = input(
            "Enter bias filename from input to first hidden layer: ")
        b2_filename = input(
            "Enter bias filename from hidden layer to output: ")
    else:
        input_filename = "data/cross_data.csv"
        w1_filename = "data/w1.csv"
        w2_filename = "data/w2.csv"
        b1_filename = "data/b1.csv"
        b2_filename = "data/b2.csv"

    # concat the current directory so we can use absolute paths
    input_filename = os.path.join(os.getcwd(), input_filename)
    w1_filename = os.path.join(os.getcwd(), w1_filename)
    w2_filename = os.path.join(os.getcwd(), w2_filename)
    b1_filename = os.path.join(os.getcwd(), b1_filename)
    b2_filename = os.path.join(os.getcwd(), b2_filename)

    # try to actually get the data
    try:
        input_file = pd.read_csv(input_filename, header=None)
        train_data = input_file.iloc[:, :-1]
        labels = input_file.iloc[:, -1:]
        w1 = pd.read_csv(w1_filename, header=None)
        b1 = pd.read_csv(b1_filename, header=None)

        w2 = pd.read_csv(w2_filename, header=None)
        b2 = pd.read_csv(b2_filename, header=None)
    except Exception as err:
        print("Error! {}\nUnable to parse arguments and get data!".format(err))
        return

    # convert everythine to NumPy arrays
    train_data = train_data.values
    labels = labels.values
    w1 = w1.values
    w2 = w2.values
    b1 = b1.values
    b2 = b2.values

    # run the algorithm
    if partA:  # part A only runs a single epoch
        w1, w2, b1, b2 = epoch(train_data, labels, w1, w2, b1, b2)
    else:
        # run the entire algorithm
        run(train_data, labels, w1, b1, w2, b2)


def getInputArgs():
    """
    Get all arguments from the command line
    """

    # create the argument parser
    parser = argparse.ArgumentParser()
    # the number of features / dimensions
    parser.add_argument('num_dim', help='The number of features')
    # the number of nuerons in hidden layer
    parser.add_argument(
        'num_hidden', help='Number of neurons in the hidden layer')
    # the number of output nodes
    parser.add_argument(
        'num_output', help='Number of neurons in the output layer')
    # default to part a
    parser.add_argument(
        '-a', '--partA', action="store_true", help="Run part A")
    # parse the arguments
    args = vars(parser.parse_args())
    # store the directories as variables
    num_dim, num_hidden, num_output = args["num_dim"], args["num_hidden"], args["num_output"]
    partA = True if args["partA"] else False
    # return
    return num_dim, num_hidden, num_output, partA


def fi(v):
    """
    Define the activation function and return fi of v
    """

    # define the sigmoid and return the value
    denom = 1 + math.e**(-1 * v)
    val = 1 / denom
    return val


def run(training_data, desired_output, w1, b1, w2, b2):
    """
    Run the algorithm with the given parameters
    """
    # TODO make this run for some condition like SSE
    for i in range(100):
        w1, w2, b1, b2 = epoch(training_data, desired_output, w1, w2, b1, b2)


def epoch(training_data, desired_output, w1, w2, b1, b2):
    """
    Run a single epoch throught network with given params
    """
    # run an epoch
    for i, datapoint in enumerate(training_data):
        # show to first hidden layer
        first_layer_output = show_to_layer(datapoint, w1, b1)
        # show to output layer
        output = show_to_layer(first_layer_output, w2, b2)[0]
        # backpropoate
        w1, w2, b1, b2 = backpropagate(
            output, desired_output[i], w1, w2, b1, b2)

    return w1, w2, b1, b2


def backpropagate(output, label, w1, w2, b1, b2):
    """
    Backpropagate the error
    """
    return w1, w2, b1, b2


def show_to_layer(inputs, weights, biases):
    """
    Take in the input, weights, and bias and show an input 
    to the layer specified by the weights and biases
    """

    # rename inputs
    w1 = weights
    training_data = inputs
    b1 = biases
    num_neurons = len(w1)
    # create the array to hold the output of this layer
    next_layer_input = np.empty(num_neurons)
    # go through each neuron in this layer
    for j, weights in enumerate(w1):
        # v = wTx + b
        v = np.dot(inputs, weights) + b1[j]
        # output is the activation function
        output = fi(v)
        # put in the array
        next_layer_input[j] = output
    # return the output of this layer
    return next_layer_input


if __name__ == "__main__":
    init()
