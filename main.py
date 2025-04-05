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

def get_loss(
        y: NDArray[np.float64],
        y_pred: NDArray[np.float64],
        l: int,
    ) -> float:
    """
    Calculate the loss between the true values and predicted values.

    Parameters:
    y (NDArray[np.float64]): True values.
    y_pred (NDArray[np.float64]): Predicted values.
    l (int): Order of the loss function (1 for L1, 2 for L2).

    Returns:
    float: Calculated loss.
    """
    y_delta: NDArray[np.float64] = y - y_pred

    if l == 1:
        return np.sum(np.abs(y_delta)) / len(y)
    if l == 2:
        return np.sum(y_delta * y_delta) / len(y)
    return 0

def get_l1_loss(
        y: NDArray[np.float64],
        y_pred: NDArray[np.float64],
    ) -> float:
    return get_loss(y, y_pred, 1)

def get_l2_loss(
        y: NDArray[np.float64],
        y_pred: NDArray[np.float64],
    ) -> float:
    return get_loss(y, y_pred, 2)

def get_derivatives(
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        w: float,
        b: float,
        loss_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], float],
    ):
    small: float = 0.01
    loss: float = loss_fn(y, w*x + b)
    dloss_dw: float = (loss_fn(y, (w+small)*x + b) - loss) / small
    dloss_db: float = (loss_fn(y, w*x + (b+small)) - loss) / small
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