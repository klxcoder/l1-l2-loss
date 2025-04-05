import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

def get_data():
    np.random.seed(0)
    x: NDArray[np.float64] = np.array(np.arange(10), dtype="float64")
    y: NDArray[np.float64] = 2 * x + np.random.normal(0, 1, x.shape[0])
    y[2] = 100  # Add an outlier
    return x, y

def get_l1_line(x: NDArray[np.float64], y: NDArray[np.float64]):
    w = 2
    b = 1
    return w, b

def get_l2_line(x: NDArray[np.float64], y: NDArray[np.float64]):
    w = 2
    b = 2
    return w, b

def main():
    x, y = get_data()
    w1, b1 = get_l1_line(x, y)
    w2, b2 = get_l2_line(x, y)

    # Create subplots
    _, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].scatter(x, y, color='blue', label='Data points')
    axes[0, 0].plot(x, w1*x + b1, color='red', label='L1 predicted line')
    axes[0, 0].plot(x, w2*x + b2, color='green', label='L2 predicted line')
    axes[0, 0].set_title('Data and predicted lines')
    axes[0, 0].legend()

    plt.show()

if __name__ == "__main__":
    main()