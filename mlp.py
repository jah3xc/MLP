# Multilayer Perceptron algorithm
# implemented using backpropagation
# @author James Hurt, 2017
import numpy as np
import pandas as pd
import math
import argparse
import os
from pprint import pprint

# define constants
ALPHA = .7
BETA = .3
TERMINATION_THRESHOLD = .001


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
        exit(1)

    # convert everythine to NumPy arrays
    train_data = train_data.values
    labels = labels.values
    w1 = w1.values
    w2 = w2.values
    b1 = b1.values
    b2 = b2.values

    # run the algorithm
    if partA:  # part A only runs a single epoch
        w1_new, w2_new, b1_new, b2_new, avg_error_energy = epoch(
            train_data, labels, w1, w2, b1, b2)
        print_results(w1_new, w2_new, b1_new, b2_new, avg_error_energy)

        # run the entire algorithm for the next part of part A
        run(train_data, labels, w1, w2, b1, b2)
    else:
        # run the entire algorithm
        run(train_data, labels, w1, w2, b1, b2)


def print_results(w1, w2, b1, b2, avg_error_energy):
    """
    Prints the results given NumPy arrays
    """
    print("----------------------\n\t\tW1\n----------------------")
    pprint(np.around(w1, decimals=4).tolist())
    print("----------------------\n\t\tB1\n----------------------")
    pprint(np.around(w2, decimals=4).tolist(), width=1)
    print("----------------------\n\t\tW2\n----------------------")
    pprint(np.around(b1, decimals=4).tolist(), width=1)
    print("----------------------\n\t\tB2\n----------------------")
    pprint(np.around(b2, decimals=4).tolist(), width=1)
    print(
        "----------------------\n\t\tERROR\n----------------------\nAverage Error Energy: {:10.4f}".format(avg_error_energy[0]))


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


def run(training_data, desired_output, w1, w2, b1, b2):
    """
    Run the algorithm with the given parameters
    """
    i = 0
    while True:
        # run an epoch
        w1, w2, b1, b2, avg_error = epoch(
            training_data, desired_output, w1, w2, b1, b2)

        if avg_error < TERMINATION_THRESHOLD:
            break
        i += 1
        print("Epoch Number: {}\t Avg Error: {}".format(i, avg_error))

    print_results(w1, w2, b1, b2, avg_error)


def epoch(training_data, desired_output, w1, w2, b1, b2):
    """
    Run a single epoch throught network with given params
    """

    avg_error = 0
    previous_w1, previous_w2, previous_b1, previous_b2 = w1, w2, b1, b2
    # run an epoch
    for i, datapoint in enumerate(training_data):
        # show to first hidden layer
        first_layer_output = show_to_layer(datapoint, w1, b1)
        # show to output layer
        output = show_to_layer(first_layer_output, w2, b2)
        # error
        er = (desired_output[i] - output)**2
        avg_error += er
        # backpropoate
        next_w1, next_w2, next_b1, next_b2 = backpropagate(datapoint,
                                                           output, first_layer_output, desired_output[i], w1, w2, b1, b2, previous_w1, previous_w2)

        previous_w1, previous_w2, previous_b1, previous_b2 = w1, w2, b1, b2
        w1, w2, b1, b2 = next_w1, next_w2, next_b1, next_b2

    avg_error = avg_error / (2 * len(training_data))
    return w1, w2, b1, b2, avg_error


def backpropagate(datapoint, output, output_layer1, label, w1, w2, b1, b2, previous_w1, previous_w2):
    """
    Backpropagate the error
    w(k+1) = w(k) + B(w(k) - w(k-1)) + A(delta)(output)
    """

    # output layer
    w2_new = np.empty([len(w2), len(w2[0])])
    for i, (neuron, prev_neuron) in enumerate(zip(w2, previous_w2)):
        # holder variable
        new_nueron_w = np.empty(len(neuron))
        # calc the delta
        delta = calc_delta_output_layer(
            output_layer1, neuron, output, label, b2[i])
        # get the learning term
        learn_term = ALPHA * delta * output_layer1[i]

        # adjust the bias
        b2[i] = adjust_b(b2[i], delta)
        for j, (weight, prev_weight) in enumerate(zip(neuron, prev_neuron)):
            # calc momentum term
            momentum_term = BETA * (weight - prev_weight)
            #  calculate the difference
            diff = momentum_term + learn_term
            # calculate the new weight
            new_w = weight + diff
            # store this result
            new_nueron_w[j] = new_w
        # store the result
        w2_new[i] = new_nueron_w
    # set this to w2
    w2 = w2_new

    # hidden layer
    w1_new = np.empty([len(w1), len(w1[0])])
    for i, (neuron, prev_neuron) in enumerate(zip(w1, previous_w1)):
        # holder variable
        new_nueron_w = np.empty(len(neuron))
        # calc the delta
        delta = calc_delta_hidden_layer(
            datapoint, neuron, b1[i], w2, b2, output, label, output_layer1)
        # get the learning term
# --------------> TODO check if this is right
        learn_term = ALPHA * delta * output_layer1[i]

        # adjust the bias
        b1[i] = adjust_b(b1[i], delta)
        for j, (weight, prev_weight) in enumerate(zip(neuron, prev_neuron)):
            # calc momentum term
            momentum_term = BETA * (weight - prev_weight)
            #  calculate the difference
            diff = momentum_term + learn_term
            # calculate the new weight
            new_w = weight + diff
            # store this result
            new_nueron_w[j] = new_w
        # store the result
        w1_new[i] = new_nueron_w
    # set this to w2
    w1 = w1_new

    return w1, w2, b1, b2


def calc_average_error_energy(output, label):
    er = 0
    for o, l in zip(output, label):
        nueron_error = label - output
        er += nueron_error**2
    return er / 2


def calc_delta_hidden_layer(datapoint, weight, bias, w2, b2, output, label, output_layer1):
    """
    Calculate the delta for a hidden node
    delta = fi_prime(v) * summation(delta(h+1)w(h+1))
    """
    # calc fi prime of v
    fiP = fi_prime(calc_v(datapoint, weight, bias))
    # need to find delta of all forward connected neurons
    summation = 0
    for i, neuron in enumerate(w2):
        # calc the delta
        delta = calc_delta_output_layer(
            output_layer1, neuron, output, label, b2[i])

        summation = summation + (weight[i] * delta)

    return fiP * summation


def adjust_b(bias, delta):
    """
    Adjust the bias
    bias += ALPHA * 1 * delta
    """
    return bias + (ALPHA * delta)


def calc_delta_output_layer(datapoint, weight, output, label, bias):
    """
    Calculate the delta for the output layer
    delta = err * fi_prime(v)
    """
    # get the components of the delta at the output layer
    err = label - output
    v = calc_v(datapoint, weight, bias)
    prime = fi_prime(v)

    return prime * err


def fi_prime(v):
    """
    Return the value of the activation function's derivative
    """

    # calc fi
    f = fi(v)
    # fi * 1 - fi
    return f * (1 - f)


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
        v = calc_v(inputs, weights, b1[j])
        # output is the activation function
        output = fi(v)
        # put in the array
        next_layer_input[j] = output
    # return the output of this layer
    return next_layer_input


def calc_v(inputs, weights, bias):
    """
    Calculate v, the input vector
    v = wTp + b
    """
    return np.dot(inputs, weights) + bias


if __name__ == "__main__":
    init()
