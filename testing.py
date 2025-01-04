import ssl
import certifi

# Set TensorFlow's default SSL context to use certifi's CA certificates
ssl._create_default_https_context = ssl.create_default_context
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("MNIST dataset loaded successfully!")
