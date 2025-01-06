from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import ssl
import certifi
import json
import os
from flask_socketio import SocketIO, emit
import time  # For simulating training updates

app = Flask(__name__)
CORS(app,resources={r"/ws/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")
app.config["SECRET_KEY"] = "your-secret-key"


# Set global SSL context using certifi's CA certificates
ssl._create_default_https_context = ssl.create_default_context

MODEL_ARCHITECTURE_FILE = "saved_model.json"

@app.route("/")
def home():
    return jsonify({"message": "Hello, Flask!"})

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "running", "message": "Flask backend is operational!"})

@app.route("/save_model", methods=["POST"])
def save_model():
    """
    Save the model architecture received from the frontend.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON payload received"}), 400

        # Save model architecture to a file
        with open(MODEL_ARCHITECTURE_FILE, "w") as f:
            json.dump(data, f)

        return jsonify({"message": "Model architecture saved successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@socketio.on("connect", namespace="/ws/train")
def handle_connect():
    """Handle WebSocket connection."""
    print("Client connected to /ws/train")
    emit("message", {"type": "info", "data": "Connected to WebSocket!"})

@socketio.on("disconnect", namespace="/ws/train")
def handle_disconnect():
    """Handle WebSocket disconnection."""
    print("Client disconnected from /ws/train")

@socketio.on("start_training", namespace="/ws/train")
def start_training(data):
    """
    Handle the WebSocket event to start training and send real-time updates.
    """
    try:
        # Ensure model architecture exists
        if not os.path.exists(MODEL_ARCHITECTURE_FILE):
            emit("training_error", {"error": "Model architecture not found. Please save it first."})
            return

        # Load the saved model architecture
        with open(MODEL_ARCHITECTURE_FILE, "r") as f:
            model_architecture = json.load(f)

        # Extract dataset and training configuration
        dataset = model_architecture.get("dataset")
        if not dataset:
            emit("training_error", {"error": "Dataset information missing in model architecture"})
            return

        training_config = data  # Training config comes from WebSocket payload

        # Load and preprocess the dataset
        (x_train, y_train), (x_test, y_test) = load_dataset(dataset)

        # Build the model
        model = build_model_from_architecture(model_architecture)

        # Compile the model
        model.compile(
            optimizer=training_config["optimizer"].lower(),
            loss=training_config["lossFunction"].lower(),
            metrics=["accuracy"]
        )

        # Emit a message that training is starting
        emit("training_start", {"message": "Training has started!"})

        # Simulate model training by epoch
        epochs = training_config["epochs"]
        for epoch in range(1, epochs + 1):
            time.sleep(1)  # Simulate training time per epoch
            history = model.fit(
                x_train,
                y_train,
                validation_data=(x_test, y_test),
                batch_size=training_config["batchSize"],
                epochs=1,
                verbose=0
            )
            # Emit training progress
            emit("training_progress", {
                "epoch": epoch,
                "loss": float(history.history["loss"][-1]),
                "accuracy": float(history.history["accuracy"][-1]),
                "val_loss": float(history.history["val_loss"][-1]),
                "val_accuracy": float(history.history["val_accuracy"][-1])
            })

        # Emit final training results
        emit("training_complete", {
            "message": "Training completed successfully!",
            "final_loss": float(history.history["loss"][-1]),
            "final_accuracy": float(history.history["accuracy"][-1])
        })

    except Exception as e:
        emit("training_error", {"error": str(e)})

def load_dataset(dataset_name):
    """
    Loads and preprocesses the dataset based on the name.

    Args:
        dataset_name (str): Name of the dataset to load.

    Returns:
        Tuple: (x_train, y_train), (x_test, y_test)
    """
    if dataset_name == "MNIST":
        # Load MNIST dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
        y_train = np.eye(10)[y_train]  # One-hot encode labels
        y_test = np.eye(10)[y_test]

    elif dataset_name == "Iris":
        # Load Iris dataset
        data = load_iris()
        X, y = data.data, data.target
        enc = OneHotEncoder(sparse_output=False)
        y = enc.fit_transform(y.reshape(-1, 1))  # One-hot encode labels
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    elif dataset_name == "CIFAR-10":
        # Load CIFAR-10 dataset
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        y_train = np.eye(10)[y_train.flatten()]  # One-hot encode labels
        y_test = np.eye(10)[y_test.flatten()]

    elif dataset_name == "California housing":
        # Load California Housing dataset
        data = fetch_california_housing()
        X, y = data.data, data.target
        scaler = StandardScaler()
        X = scaler.fit_transform(X)  # Normalize features
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")

    return (x_train, y_train), (x_test, y_test)

def build_model_from_architecture(architecture):
    """
    Build a Keras model based on the architecture provided.

    Args:
        architecture (dict): The model architecture containing nodes and edges.

    Returns:
        keras.Model: A compiled Keras model.
    """
    nodes = architecture["nodes"]
    edges = architecture["edges"]

    # Validate input and output layers
    input_layer = next((node for node in nodes if node["type"] == "input"), None)
    output_layer = next((node for node in nodes if node["type"] == "output"), None)

    if not input_layer or not output_layer:
        raise ValueError("Model must have both an input and an output layer.")

    # Start building the model
    model = Sequential()

    # Add layers based on the nodes
    for node in nodes:
        layer_type = node["type"]
        layer_data = node["data"]

        if layer_type == "dense":
            model.add(Dense(
                units=layer_data["neurons"],
                activation=layer_data["activation"].lower()
            ))
        elif layer_type == "convolution":
            model.add(Conv2D(
                filters=layer_data["filters"],
                kernel_size=tuple(layer_data["kernelSize"]),
                strides=tuple(layer_data["stride"]),
                activation=layer_data["activation"].lower()
            ))
        elif layer_type == "maxpooling":
            model.add(MaxPooling2D(
                pool_size=tuple(layer_data["poolSize"]),
                strides=tuple(layer_data["stride"])
            ))
        elif layer_type == "flatten":
            model.add(Flatten())
        elif layer_type == "dropout":
            model.add(Dropout(rate=layer_data["rate"]))
        elif layer_type == "batchnormalization":
            model.add(BatchNormalization(
                momentum=layer_data["momentum"],
                epsilon=layer_data["epsilon"]
            ))
        elif layer_type == "input":
            model.add(Dense(units=layer_data.get("neurons", 1), input_shape=(layer_data.get("shape", 1),)))

    return model

@app.route("/export", methods=["GET"])
def export_model():
    return {"message": "Model exported successfully!"}

app.debug=True

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000,debug=True)
