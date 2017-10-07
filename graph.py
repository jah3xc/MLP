from matplotlib import pyplot as plt

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
    for (x, y), label in zip(train_data, labels):
        plt.scatter(x, y, c=("red" if label == 0 else "blue"))

    plt.show()
