from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_california_housing_dataset():
    """
    Load and preprocess the California Housing dataset.

    Returns:
        Tuple: (x_train, y_train), (x_test, y_test)
    """
    data = fetch_california_housing()
    X, y = data.data, data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)  # Normalize features
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return (x_train, y_train), (x_test, y_test)

if __name__=="__main__":
    (x_train, y_train), (x_test, y_test) = load_iris_dataset()
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
