from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

def load_california_housing_dataset():
    """
    Load and preprocess the California Housing dataset.

    Returns:
        Tuple: (x_train, y_train), (x_test, y_test)
    """
    # Load dataset
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Convert to TensorFlow tensors
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train.reshape(-1, 1), dtype=tf.float32)  # ✅ Reshaped
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test.reshape(-1, 1), dtype=tf.float32)    # ✅ Reshaped

    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_california_housing_dataset()
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
