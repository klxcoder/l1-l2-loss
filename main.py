import matplotlib.pyplot as plt
import numpy as np

def get_data():
    np.random.seed(0)
    x = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 2 * x.squeeze() + np.random.normal(0, 1, x.shape[0])
    y[30] = 30  # Add an outlier
    return x, y

def main():
    x, y = get_data()
    plt.scatter(x, y)
    plt.show()

if __name__ == "__main__":
    main()