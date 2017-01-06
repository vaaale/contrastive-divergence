import matplotlib.pyplot as plt


def display(pos, neg):
    n = 10
    for i in range(n):
        x = pos[i]
        y = neg[i]

        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display noisy
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(y)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
