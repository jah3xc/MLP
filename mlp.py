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
    # can't execute both parts A and B
    if partA and partB:
        print("Error! You cannot run both part A and B!")
        exit(1)
    # if the user wishes to do part A
    if partA:
        # set the filenames
        input_filename = "data/cross_data.csv"
        w1_filename = "data/w1.csv"
        w2_filename = "data/w2.csv"
        b1_filename = "data/b1.csv"
        b2_filename = "data/b2.csv"
        # set the sizes of the layers
        num_dim, num_hidden, num_output = 2, 10, 1
        # read the input file
        input_file = pd.read_csv(input_filename, header=None)
        # extract train data and labels from the input data
        train_data = input_file.iloc[:, :-1]
        labels = input_file.iloc[:, -1:]
    # if we want part B
    elif partB:
        # read the data file
        all_data = pd.read_csv(
            "data/Two_Class_FourDGaussians500.txt", sep="  ", header=None, engine='python')
        # set the dims of the network
        num_dim, num_hidden, num_output = 4, 3, 2
        # extract the training and validation data
        train_and_validation = all_data.iloc[:, :-1]
        # extract labels
        labels = all_data.iloc[:, -1:]

        # need to separate the train and validation data

        # get points 100-500 and points 600-1000 as training data
        train_data = pd.concat([
            train_and_validation.iloc[100:500, :], train_and_validation.iloc[600:, :]])
        labels = pd.concat([labels.iloc[100:500, :], labels.iloc[600:, :]])
        # get points 0-100 and 500-600 (the first 100 of each set) as validation/test data
        validation_data = pd.concat([
            train_and_validation.iloc[:100, :], train_and_validation.iloc[500:600, :]])
        validation_labels = pd.concat([
            labels.iloc[:100, :], labels.iloc[500:600, :]])
        # convert the labels to numpy arrays of 2 dims instead of a single number
        validation_labels = label_2D(validation_labels)
        # convert validation data to numpy
        validation_data = validation_data.values
        # set filenames for w1, w2, b1, b2
        w1_filename = "data/partB_w1.csv"
        w2_filename = "data/partB_w2.csv"
        b1_filename = "data/partB_b1.csv"
        b2_filename = "data/partB_b2.csv"
    else:
        # if we don't want part a or part b, then we need the filenames

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
        # read in CSV files
        input_file = pd.read_csv(input_filename, header=None)
        # get the train data
        train_data = input_file.iloc[:, :-1]
        # get the labels
        labels = input_file.iloc[:, -1:]

    # try to actually get the data
    try:
        w1 = pd.read_csv(w1_filename, header=None)
        b1 = pd.read_csv(b1_filename, header=None)

        w2 = pd.read_csv(w2_filename, header=None)
        b2 = pd.read_csv(b2_filename, header=None)
    except Exception as err:
        # if we error - tell the user and exit
        print("Error! {}\nUnable to parse arguments and get data!".format(err))
        exit(1)

    # convert everythine to NumPy arrays
    train_data = train_data.values
    labels = labels.values if not partB else label_2D(labels)
    w1 = w1.values
    w2 = w2.values
    # flatten into simgle dim because b1 and b2 should always be single dim
    b1 = b1.values.flatten()
    b2 = b2.values.flatten()

    # run the algorithm
    if partA:  # part A only runs a single epoch
        # run the alg
        w1_new, w2_new, b1_new, b2_new, avg_error_energy = epoch(
            train_data, labels, w1, w2, b1, b2)
        # print the values
        print_results(w1_new, w2_new, b1_new, b2_new, avg_error_energy)
        # wait for user input
        input("\n\nPress [Enter] to train network...\n")

        # run the entire algorithm for the next part of part A
        w1, w2, b1, b2, error_per_epoch = run(
            train_data, labels, w1, w2, b1, b2)
        # allow user to see data
        graph_init(error_per_epoch, train_data, labels, w1, w2, b1, b2)
    elif partB:
        # train the network
        w1, w2, b1, b2, error = run(train_data, labels, w1, w2, b1, b2)
        # wait for user to ok testing
        input("\n\nPress [Enter] to test network...\n")

        print("Validating Data...")
        # test the data
        accuracy = validate(validation_data, validation_labels, w1, w2, b1, b2)
        # display the accuracy
        print("Accuracy:  {}".format(accuracy))
    else:
        # run the entire algorithm
        run(train_data, labels, w1, w2, b1, b2)


def validate(validation_data, validation_labels, w1, w2, b1, b2):
    """
    Validate with validation data
    """
    # number correct
    num_correct = 0
    # iterate through validation data and show it to the network
    for i, datapoint in enumerate(validation_data):
        # show to hidden layer
        hidden = show_to_layer(datapoint, w1, b1)
        # show to output layer
        output = show_to_layer(hidden, w2, b2)
        # values aren't exactly 1 and 0, need to round
        output = np.around(output)
        # if we get it right, then inc num_correct
        if np.array_equal(validation_labels[i], output):
            num_correct += 1
    # accuracy is the num correct over the total
    a = float(num_correct) / float(len(validation_data))
    return a


def label_2D(labels):
    """
    Take in labels of 1 and 2 and turn them into a 2D vector of (1,0) and (0,1)
    """
    # turn labels into a numpy array of 1 dimension
    labels = labels.values.flatten()
    # create an empty array
    new_labels = np.empty([len(labels), 2])
    # iterate through labels
    for i, l in enumerate(labels):
        # make this value [0,1] if the current label is 0, otherwise make this label [1,0]
        new_labels[i] = np.array(
            [0, 1]) if l == 0 else np.array([1, 0])
    # return 2D labels
    return new_labels


def print_results(w1, w2, b1, b2, avg_error_energy):
    """
    Prints the results given NumPy arrays
    """
    print("----------------------\n\t\tW1\n----------------------")
    pprint(np.around(w1, decimals=4).tolist())
    print("----------------------\n\t\tW2\n----------------------")
    pprint(np.around(w2, decimals=4).tolist(), width=1)
    print("----------------------\n\t\tB1\n----------------------")
    pprint(np.around(b1, decimals=4).tolist(), width=1)
    print("----------------------\n\t\tB2\n----------------------")
    pprint(np.around(b2, decimals=4).tolist(), width=1)
    print(
        "----------------------\n\t\tERROR\n----------------------\nAverage Error Energy: {:10.4f}".format(
            avg_error_energy))


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
    # check for part A
    partA = True if args["partA"] else False
    # check for part B
    partB = True if args["partB"] else False
    # return
    return num_dim, num_hidden, num_output, partA, partB


def run(training_data, desired_output, w1, w2, b1, b2):
    """
    Run the algorithm with the given parameters
    """
    # the errors of each epoch
    error = []
    # iteration number
    i = 0
    # previous error to calc next error
    prev_error = 1
    # run forever!
    while True:
        # randomize data
        training_data, desired_output = randomize_data(
            training_data, desired_output)
        # run an epoch
        w1, w2, b1, b2, avg_error = epoch(
            training_data, desired_output, w1, w2, b1, b2)
        # check termination condition
        if avg_error < TERMINATION_THRESHOLD:
            break
        # incrememnt iterate counter
        i += 1
        # add this epochs error to the list
        error.append(avg_error)
        # calc the change in diff
        diff = prev_error - avg_error
        # calc the percent error
        diff = diff / prev_error * 100
        # absolute value
        diff = diff if diff >= 0 else -diff
        # print the iteration num, the percent change, and the average error
        print("Epoch Number: {:6d} \t{:>15} {:.7f}\tPercent Change: {:.4f}".format(
            i, "Average Error: ", avg_error, diff

        ))
        # move the avg to prev
        prev_error = avg_error
    # once we've terminated, print the results
    print_results(w1, w2, b1, b2, avg_error)
    # return
    return w1, w2, b1, b2, error


def randomize_data(a, b):
    """
    Randomize a, b so that the order shown to the 
    network is random
    """
    # create empty arrays
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    # get a permutation of array a
    permutation = np.random.permutation(len(a))
    # go throgh the permutation and switch values in both labels and data
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    # return the shuffled arrays
    return shuffled_a, shuffled_b


def epoch(training_data, desired_output, w1, w2, b1, b2):
    """
    Run a single epoch throught network with given params
    """
    # initiate the avg error
    avg_error = 0.
    # assign prev to current for first iteration
    previous_w1, previous_w2, previous_b1, previous_b2 = w1, w2, b1, b2
    # run an epoch
    for i, datapoint in enumerate(training_data):
        # show to first hidden layer
        first_layer_output = show_to_layer(datapoint, w1, b1)
        # show to output layer
        output = show_to_layer(first_layer_output, w2, b2)
        # error
        er = calc_error(output, desired_output[i])
        # add this error to the total error
        avg_error += er
        # backpropoate
        next_w1, next_w2, next_b1, next_b2 = backpropagate(datapoint,
                                                           output, first_layer_output, desired_output[i], w1, w2, b1,
                                                           b2, previous_w1, previous_w2)
        # set prev to current
        previous_w1, previous_w2, previous_b1, previous_b2 = w1, w2, b1, b2
        # set current to next
        w1, w2, b1, b2 = next_w1, next_w2, next_b1, next_b2
    # avg error = 1/2K * total error
    avg_error = avg_error / (2. * len(training_data))
    # return
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
        # calc the delta: delta = e * fiprime(v)
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
        # iterate through the deltas of the output layer
        for j, d in enumerate(output_deltas):
            # multiply the delta * the weight connecting to that neuron
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
    # return new parameters
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
        output = 1 / (1 + math.e**(-v))
        # put in the array
        next_layer_input[j] = output
    # return the output of this layer
    return next_layer_input


if __name__ == "__main__":
    init()
