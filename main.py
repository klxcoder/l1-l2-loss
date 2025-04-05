import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing import Callable

def get_data():
    np.random.seed(0)
    x: NDArray[np.float64] = np.array(np.arange(10), dtype="float64")
    y: NDArray[np.float64] = 2 * x + np.random.normal(0, 1, x.shape[0])
    y[2] = 100  # Add an outlier
    return x, y

def get_l1_loss(y: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    loss: float = 0
    for i in range(len(y)):
        loss += np.abs(y[i] - y_pred[i])
    return loss/len(y)

def get_l2_loss(y: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    loss: float = 0
    for i in range(len(y)):
        loss += (y[i] - y_pred[i]) ** 2
    return loss/len(y)

def get_derivatives(
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        w: float,
        b: float,
        loss_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], float],
    ):
    small = 0.01
    loss = loss_fn(y, w*x + b)
    dloss_dw = (loss_fn(y, (w+small)*x + b) - loss) / small
    dloss_db = (loss_fn(y, w*x + (b+small)) - loss) / small
    return dloss_dw, dloss_db, loss

def get_best_fit_line(
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        loss_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], float],
    ):
    w: float = 0
    b: float = 0
    times: int = 1000
    alpha: float = 0.01
    x_loss: list[float] = []
    y_loss: list[float] = []
    for _ in range(times):
        dloss_dw, dloss_db, loss = get_derivatives(x, y, w, b, loss_fn)
        x_loss.append(len(x_loss))
        y_loss.append(loss)
        w -= alpha * dloss_dw
        b -= alpha * dloss_db
    return w, b, x_loss, y_loss

def main():
    x, y = get_data()

    w1, b1, x1_loss, y1_loss = get_best_fit_line(x, y, get_l1_loss)
    y1_pred = w1*x + b1

    w2, b2, x2_loss, y2_loss = get_best_fit_line(x, y, get_l2_loss)
    y2_pred = w2*x + b2

    # Create subplots
    _, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].scatter(x, y, color='blue', label='Data points')
    axes[0].plot(x, y1_pred, color='red', label='L1 predicted line')
    axes[0].plot(x, y2_pred, color='green', label='L2 predicted line')
    axes[0].set_title('Data and predicted lines')
    axes[0].legend()

    axes[1].plot(x1_loss, y1_loss, color='red', label='L1 loss')
    axes[1].plot(x2_loss, y2_loss, color='green', label='L2 loss')
    axes[1].set_title('Losses')
    axes[1].legend()

    plt.show()

if __name__ == "__main__":
    main()