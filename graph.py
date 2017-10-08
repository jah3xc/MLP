from matplotlib import pyplot as plt
import numpy as np
from mlp import show_to_layer
from multiprocessing import Pool
import os


def graph_init(error_per_epoch, train_data, labels, w1, w2, b1, b2):
    num = 0
    while num != 3:
        num = int(input("Would you like to view: \n1. Error Per Epoch\n2. Data\n3. Exit\n>"))
        if num == 1:
            graph_error_per_epoch(error_per_epoch)
        elif num == 2: 
            graph_data_with_solution(train_data, labels, w1, w2, b1, b2)

def graph_error_per_epoch(error_per_epoch):
    plt.plot(range(len(error_per_epoch)), error_per_epoch)
    plt.title("Error Per Epoch")
    plt.ylabel("Error")
    plt.xlabel("Number of Epochs")
    plt.show()


def graph_data_with_solution(train_data, labels, w1, w2, b1, b2):
    fig = plt.gcf()
    size = fig.get_size_inches()*fig.dpi
    width, height = int(size[0]), int(size[1])
    with Pool(processes=int(os.cpu_count() -1 )) as pool:
        for i in range(width):
            for j in range(height):
                print("Processing {},{}".format(i,j))
                datapoint = np.array([i,j])
                first_layer_output = show_to_layer(datapoint, w1, b1)
                # show to output layer
                output = round(show_to_layer(first_layer_output, w2, b2)[0])
                plt.scatter(i,j,c="yellow" if output == 0 else "green")
        pool.close()
        pool.join()

    for (x, y), label in zip(train_data, labels):
        plt.scatter(x, y, c=("red" if label == 0 else "blue"))
    plt.show()
