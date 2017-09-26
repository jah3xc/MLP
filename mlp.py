# Multilayer Perceptron algorithm
# implemented using backpropagation
# @author James Hurt, 2017
import numpy as np
import pandas as pd
import math


def init():
    """
    Initialize the program by getting all input data
    """
    # parse command line arguments
    num_dim, num_hidden, num_output = getInputArgs()

    # read in filenames
    input_filename = input("Enter train data filename: ")
    w1_filename = input(
        "Enter weights filename from input to first hidden layer: ")
    w2_filename = input("Enter weights from hidden layer to output: ")
    b1_filename = input(
        "Enter bias filename from input to first hidden layer: ")
    b2_filename = input("Enter bias filename from hidden layer to output: ")

    # try to actually get the data
    try:
        input_file = pd.read_csv(input_filename, header=None)
        train_data = input_file.iloc[:, :-1]
        labels = input_file.iloc[:, -1:]
        w1 = pd.read_csv(w1_filename, header=None)
        b1 = pd.read_csv(b1_filename, header=None)

        w2 = pd.read_csv(w2_filename, header=None)
        b2 = pd.read_csv(b2_filename, header=None)
    except:
        print("Unable to parse arguments and get data!")
        init()

    # run the algorithm
    run(train_data, labels, w1, b1, w2, b2)


def getInputArgs():
    """
    Get all arguments from the command line
    """
    ###########################
    # TEMP - NEED TO REPLACE WITH ACTUAL LOGIC
    ###############################
    num_dim = 2
    num_hidden = 10
    num_output = 1
    return num_dim, num_hidden, num_output


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

    # convert everythine to NumPy arrays
    training_data = training_data.values
    desired_output = desired_output.values
    w1 = w1.values
    w2 = w2.values
    b1 = b1.values
    b2 = b2.values

    # run an epoch
    for i, datapoint in enumerate(training_data):
        # show to first hidden layer
        first_layer_output = show_to_layer(datapoint, w1, b1)
        # show to output layer
        output = show_to_layer(first_layer_output, w2, b2)[0]
        # backpropoate
        backpropagate()


def backpropagate():
    """
    Backpropagate the error
    """
    pass


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
