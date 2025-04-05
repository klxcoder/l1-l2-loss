import matplotlib.pyplot as plt
import numpy as np

def get_data():
    np.random.seed(0)
    x = np.arange(10)
    y = 2 * x + np.random.normal(0, 1, x.shape[0])
    y_pred_l1 = 2*x + 0.5
    y_pred_l2 = 2*x + 1
    y[2] = 100  # Add an outlier
    return x, y, y_pred_l1, y_pred_l2

def main():
    x, y, y_pred_l1, y_pred_l2 = get_data()

    # Create subplots
    _, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].scatter(x, y, color='blue', label='Data points')
    axes[0, 0].plot(x, y_pred_l1, color='red', label='L1 predicted line')
    axes[0, 0].plot(x, y_pred_l2, color='green', label='L2 predicted line')
    axes[0, 0].set_title('Data and predicted lines')
    axes[0, 0].legend()

    plt.show()

if __name__ == "__main__":
    main()