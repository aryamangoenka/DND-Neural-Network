<<<<<<< HEAD
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from tensorflow.keras.datasets import mnist, cifar10
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import ssl
import certifi

app = Flask(__name__)
CORS(app)

# Set global SSL context using certifi's CA certificates
ssl._create_default_https_context = ssl.create_default_context

=======
from flask import Flask,jsonify,request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  
>>>>>>> 68e4c30d619c0bf796ec773f5a0a10b50fa30d92

@app.route("/")
def home():
    return jsonify({"message": "Hello, Flask!"})

<<<<<<< HEAD

=======
>>>>>>> 68e4c30d619c0bf796ec773f5a0a10b50fa30d92
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "running", "message": "Flask backend is operational!"})


<<<<<<< HEAD
=======
#Dataset Upload Endpoint
>>>>>>> 68e4c30d619c0bf796ec773f5a0a10b50fa30d92
@app.route("/upload", methods=["POST"])
def upload_dataset():
    if "file" not in request.files:
        return {"error": "No file uploaded"}, 400

    file = request.files["file"]
    file.save(f"./uploads/{file.filename}")
    return {"message": f"{file.filename} uploaded successfully!"}, 200

<<<<<<< HEAD

=======
#Model Configuration Endpoint
>>>>>>> 68e4c30d619c0bf796ec773f5a0a10b50fa30d92
@app.route("/configure", methods=["POST"])
def configure_model():
    config = request.json  # JSON payload from frontend
    return {"message": "Model configured", "config": config}, 200

<<<<<<< HEAD

=======
#Training Endpoint
>>>>>>> 68e4c30d619c0bf796ec773f5a0a10b50fa30d92
@app.route("/train", methods=["POST"])
def train_model():
    try:
        # Debugging: Log the incoming request
        print("Raw Request Data:", request.data)
        print("Headers:", request.headers)

        # Ensure the Content-Type is application/json
        if not request.is_json:
            return jsonify({"error": "Invalid Content-Type. Expected application/json"}), 415

        # Parse JSON payload
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON payload received"}), 400

<<<<<<< HEAD
        # Extract nodes, edges, and dataset
        nodes = data.get("nodes")
        edges = data.get("edges")
        dataset = data.get("dataset")
=======
        # Extract nodes and edges
        nodes = data.get("nodes")
        edges = data.get("edges")
>>>>>>> 68e4c30d619c0bf796ec773f5a0a10b50fa30d92

        # Validate input structure
        if not nodes or not edges:
            return jsonify({"error": "Nodes and edges are required"}), 400

<<<<<<< HEAD
        # Validate dataset
        if not dataset:
            return jsonify({"error": "Dataset is required"}), 400

        # Debugging: Log received data
        print("Received Nodes:", nodes)
        print("Received Edges:", edges)
        print("Selected Dataset:", dataset)

        # Load and preprocess the dataset
        (x_train, y_train), (x_test, y_test) = load_dataset(dataset)

        # Log dataset shapes for debugging
        print(f"Dataset '{dataset}' loaded successfully!")
        print(f"Training data shape: {x_train.shape}, {y_train.shape}")
        print(f"Test data shape: {x_test.shape}, {y_test.shape}")

        # Simulate training logic
        print(f"Simulating training on dataset: {dataset}")

        # Respond with success
        return jsonify({
            "message": "Training logic not implemented yet. Dataset loaded successfully.",
            "status": "success",
            "dataset": dataset,
            "train_data_shape": x_train.shape,
            "test_data_shape": x_test.shape
=======
        # Simulate training logic
        print("Received Nodes:", nodes)
        print("Received Edges:", edges)

        # Respond with success
        return jsonify({
            "message": "Training started successfully!",
            "status": "success",
            "nodes_count": len(nodes),
            "edges_count": len(edges)
>>>>>>> 68e4c30d619c0bf796ec773f5a0a10b50fa30d92
        }), 200

    except Exception as e:
        # Handle unexpected errors
        return jsonify({"error": str(e)}), 500


<<<<<<< HEAD
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


=======
#Model Export Endpoint
>>>>>>> 68e4c30d619c0bf796ec773f5a0a10b50fa30d92
@app.route("/export", methods=["GET"])
def export_model():
    return {"message": "Model exported successfully!"}

<<<<<<< HEAD

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
=======
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

>>>>>>> 68e4c30d619c0bf796ec773f5a0a10b50fa30d92
