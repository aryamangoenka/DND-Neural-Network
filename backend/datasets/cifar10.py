from tensorflow.keras.datasets import cifar10
import numpy as np

def load_cifar10_dataset():
    """
    Load and preprocess the CIFAR-10 dataset.

    Returns:
        Tuple: (x_train, y_train), (x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = np.eye(10)[y_train.flatten()]  # One-hot encode labels
    y_test = np.eye(10)[y_test.flatten()]
    return (x_train, y_train), (x_test, y_test)


if __name__=="__main__":
    (x_train, y_train), (x_test, y_test) = load_cifar10_dataset()
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
