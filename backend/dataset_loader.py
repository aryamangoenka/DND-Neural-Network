from datasets.iris import load_iris_dataset
from datasets.mnist import load_mnist_dataset
from datasets.cifar10 import load_cifar10_dataset
from datasets.california import load_california_housing_dataset
from datasets.breastcancer import load_breast_cancer_dataset
import json
DATASET_LOADERS = {
    "Iris": load_iris_dataset,
    "MNIST": load_mnist_dataset,
    "CIFAR-10": load_cifar10_dataset,
    "California Housing": load_california_housing_dataset,
    "Breast Cancer": load_breast_cancer_dataset
}

MODEL_ARCHITECTURE_FILE = "saved_model.json"
# Save model architecture to a file
with open(MODEL_ARCHITECTURE_FILE, "r") as f:
    model=json.load(f)
dataset_name=model.get("dataset")

def load_dataset(dataset_name):
    """
    Load the dataset dynamically based on the dataset name.

    Args:
        dataset_name (str): Name of the dataset to load.

    Returns:
        Tuple: (x_train, y_train), (x_test, y_test)
    """


    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")
    return DATASET_LOADERS[dataset_name]()

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_dataset(dataset_name)
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
