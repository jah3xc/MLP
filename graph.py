from matplotlib import pyplot as plt


def graph_error_per_epoch(error_per_epoch):
    plt.plot(range(len(error_per_epoch)), error_per_epoch)
    plt.show()


def graph_data_with_solution(train_data, labels, w1, w2, b1, b2):
    for (x, y), label in zip(train_data, labels):
        plt.scatter(x, y, c=("red" if label == 0 else "blue"))

    plt.show()
