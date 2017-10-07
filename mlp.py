# Multilayer Perceptron algorithm
# implemented using backpropagation
# @author James Hurt, 2017
import pandas as pd
import argparse
import os
from pprint import pprint
from random import random
import math
# my files
from graph import *
from calculations import *
from constants import *


def init():
    """
    Initialize the program by getting all input data
    """
    # parse command line arguments
    num_dim, num_hidden, num_output, partA, partB = getInputArgs()

    if partA:
        input_filename = "data/cross_data.csv"
        w1_filename = "data/w1.csv"
        w2_filename = "data/w2.csv"
        b1_filename = "data/b1.csv"
        b2_filename = "data/b2.csv"

    elif partB:
        print("In works!")
        exit(1)
    else:
        # read in filenames
        input_filename = input("Enter train data filename: ")
        w1_filename = input(
            "Enter weights filename from input to first hidden layer: ")
        w2_filename = input("Enter weights from hidden layer to output: ")
        b1_filename = input(
            "Enter bias filename from input to first hidden layer: ")
        b2_filename = input(
            "Enter bias filename from hidden layer to output: ")

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
    b1 = b1.values.flatten()
    b2 = b2.values.flatten()

    # run the algorithm
    if partA:  # part A only runs a single epoch
        w1_new, w2_new, b1_new, b2_new, avg_error_energy = epoch(
            train_data, labels, w1, w2, b1, b2)
        print_results(w1_new, w2_new, b1_new, b2_new, avg_error_energy)

        input("Press [Enter] to continue...")

        # run the entire algorithm for the next part of part A
        error_per_epoch = run(train_data, labels, w1, w2, b1, b2)

        graph_error_per_epoch(error_per_epoch)
        graph_data_with_solution(train_data, labels, w1, w2, b1, b2)

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
        "----------------------\n\t\tERROR\n----------------------\nAverage Error Energy: {:10.4f}".format(
            avg_error_energy[0]))


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
    # part B
    parser.add_argument(
        '-b', '--partB', action="store_true", help="Run part B")
    # parse the arguments
    args = vars(parser.parse_args())
    # store the directories as variables
    num_dim, num_hidden, num_output = args["num_dim"], args["num_hidden"], args["num_output"]
    partA = True if args["partA"] else False

    partB = True if args["partB"] else False
    # return
    return num_dim, num_hidden, num_output, partA, partB


def run(training_data, desired_output, w1, w2, b1, b2):
    """
    Run the algorithm with the given parameters
    """
    error = []
    i = 0
    prev_error = 1
    while True:
        # randomize data
        training_data, desired_output = randomize_data(training_data, desired_output)
        # run an epoch
        w1, w2, b1, b2, avg_error = epoch(
            training_data, desired_output, w1, w2, b1, b2)

        if avg_error < TERMINATION_THRESHOLD:
            break
        i += 1
        avg_error = avg_error[0]
        error.append(avg_error)
        diff = prev_error - avg_error
        diff = diff / prev_error * 100
        diff = diff if diff >= 0 else -diff
        print("Epoch Number: {:6d} \t{:>15} {:.7f}\tPercent Change: {:.4f}".format(
            i, "Average Error: ", avg_error, diff

        ))
        prev_error = avg_error

    print_results(w1, w2, b1, b2, avg_error)
    return error

def randomize_data(a, b):
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b
    

def epoch(training_data, desired_output, w1, w2, b1, b2):
    """
    Run a single epoch throught network with given params
    """

    avg_error = 0.
    previous_w1, previous_w2, previous_b1, previous_b2 = w1, w2, b1, b2
    # run an epoch
    for i, datapoint in enumerate(training_data):
        # show to first hidden layer
        first_layer_output = show_to_layer(datapoint, w1, b1)
        # show to output layer
        output = show_to_layer(first_layer_output, w2, b2)
        # error
        er = (desired_output[i] - output) ** 2
        avg_error += er
        # backpropoate
        next_w1, next_w2, next_b1, next_b2 = backpropagate(datapoint,
                                                           output, first_layer_output, desired_output[i], w1, w2, b1,
                                                           b2, previous_w1, previous_w2)

        previous_w1, previous_w2, previous_b1, previous_b2 = w1, w2, b1, b2
        w1, w2, b1, b2 = next_w1, next_w2, next_b1, next_b2

    avg_error = avg_error / (2. * len(training_data))
    return w1, w2, b1, b2, avg_error


def backpropagate(datapoint, output, output_layer1, label, w1, w2, b1, b2, previous_w1, previous_w2):
    """
    Backpropagate the error
    w(k+1) = w(k) + B(w(k) - w(k-1)) + A(delta)(output)
    """

    # output layer
    output_deltas = np.empty(len(w2))
    w2_new = np.empty([len(w2), len(w2[0])])
    for i, (neuron, prev_neuron) in enumerate(zip(w2, previous_w2)):
        # holder variable
        new_nueron_w = np.empty(len(neuron))
        # calc the delta
        v = calc_v(output_layer1, neuron, b2[i])
        prime = fi_prime(v)
        e = label[i] - output[i]
        delta = prime * e
        output_deltas[i] = delta
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
        # calc the delta = fiprime * summation(delta_output * w)
        v = calc_v(datapoint, neuron, b1[i])
        prime = fi_prime(v)
        summation = 0
        for j, d in enumerate(output_deltas):
            summation += d * w2[j][i]
        delta = prime * summation
        # adjust the bias
        b1[i] = adjust_b(b1[i], delta)
        for j, (weight, prev_weight) in enumerate(zip(neuron, prev_neuron)):
            # calc momentum term
            momentum_term = BETA * (weight - prev_weight)
            # calc learn term
            learn_term = ALPHA * delta * datapoint[j]
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


if __name__ == "__main__":
    init()
