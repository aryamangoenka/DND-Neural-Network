from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def load_iris_dataset():
    """
    Load and preprocess the Iris dataset.

    Returns:
        Tuple: (x_train, y_train), (x_test, y_test)
    """
    data = load_iris()
    X, y = data.data, data.target
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y.reshape(-1, 1))  # One-hot encode labels
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return (x_train, y_train), (x_test, y_test)

if __name__=="__main__":
    (x_train, y_train), (x_test, y_test) = load_iris_dataset()
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
