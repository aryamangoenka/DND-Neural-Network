from tensorflow.keras.datasets import mnist
import numpy as np

def load_mnist_dataset():
    """
    Load and preprocess the MNIST dataset.

    Returns:
        Tuple: (x_train, y_train), (x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    y_train = np.eye(10)[y_train]  # One-hot encode labels
    y_test = np.eye(10)[y_test]
    return (x_train, y_train), (x_test, y_test)


if __name__=="__main__":
    (x_train, y_train), (x_test, y_test) = load_mnist_dataset()
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")