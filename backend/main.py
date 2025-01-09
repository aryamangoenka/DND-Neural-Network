from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import json
import os
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization, Input
from dataset_loader import load_dataset  # Import the load_dataset function
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
import numpy as np
# Dictionary to track stop flags for each client
stop_flags = {}

class RealTimeUpdateCallback(Callback):
    def __init__(self, socketio, client_id, total_epochs):
        self.socketio = socketio
        self.client_id = client_id
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        """Emit training progress after each epoch."""
        if logs is not None:
            self.socketio.emit("training_progress", {
                "epoch": epoch + 1,
                "total_epochs": self.total_epochs,
                "loss": logs.get("loss"),
                "accuracy": logs.get("accuracy"),
                "val_loss": logs.get("val_loss"),
                "val_accuracy": logs.get("val_accuracy"),
            }, to=self.client_id)
        if stop_flags.get(self.client_id):
            raise KeyboardInterrupt("Training stopped by the client disconnecting.")

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
app.config["SECRET_KEY"] = "your-secret-key"

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

@socketio.on("connect")
def handle_connect():
    """Handle WebSocket connection."""
    """Handle WebSocket connection."""
    client_id = request.sid  # Unique client ID
    stop_flags[client_id] = False  # Initialize stop flag for the client
  
    print("Client connected to WebSocket")
    emit("message", {"type": "info", "data": "Connected to WebSocket!"})

@socketio.on("disconnect")
def handle_disconnect():
    """Handle WebSocket disconnection."""
    """Handle WebSocket connection."""
    client_id = request.sid  # Unique client ID
    stop_flags[client_id] = True  # Initialize stop flag for the client
  
    
    print("Client disconnected from WebSocket")


@socketio.on("start_training")
def start_training(data):
    """
    Handle the WebSocket event to start training and send real-time updates.
    """
    LOSS_FUNCTION_MAPPING = {
        "Categorical Cross-Entropy": "categorical_crossentropy",
        "Binary Cross-Entropy": "binary_crossentropy",
        "Mean Squared Error": "mse",
        "Mean Absolute Error": "mae",
        "Huber Loss": "huber"
    }
    client_id = request.sid  # Unique client ID
    
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
        print(training_config)

        # Map loss function
        loss_function = LOSS_FUNCTION_MAPPING.get(training_config["lossFunction"])
        if not loss_function:
            emit("training_error", {"error": f"Invalid loss function: {training_config['lossFunction']}"})
            return

        # Load and preprocess the dataset using load_dataset
        (x_train, y_train), (x_test, y_test) = load_dataset(dataset)

        # Build the model
        model = build_model_from_architecture(model_architecture, x_train.shape[1:],dataset)

        # Compile the model
        model.compile(
            optimizer=training_config["optimizer"].lower(),
            loss=loss_function,
            metrics=["accuracy"]
        )

        # Emit a message that training is starting
        emit("training_start", {"message": "Training has started!"})
        # Define total epochs and stage size
        total_epochs = training_config["epochs"]
        if dataset in ["Iris", "Breast Cancer"]:
            stage_size = 5
        elif dataset in ["MNIST"]:
            stage_size = 3
        elif dataset == "CIFAR-10":
            stage_size = 1
        else:
            stage_size = 2  # Default



        # Train the model in stages
        for stage_start in range(0, total_epochs, stage_size):
            if stop_flags.get(client_id):
                print(f"Training stopped for client {client_id}.")
                emit("training_stopped", {"message": "Training was stopped by the client."})
                return

            
            # Calculate the number of epochs for this stage
            current_stage_size = min(stage_size, total_epochs - stage_start)

            # Define the callback for real-time updates
            callback = RealTimeUpdateCallback(socketio,client_id, total_epochs)

            # Train the model for the current stage
            history = model.fit(
                x_train,
                y_train,
                validation_data=(x_test, y_test),
                batch_size=training_config["batchSize"],
                epochs=stage_start + current_stage_size,
                initial_epoch=stage_start,
                verbose=0,  # Suppress built-in progress bar
                callbacks=[callback]  # Add the custom callback for real-time updates
            )

            # Emit staged progress
            for epoch_offset in range(current_stage_size):
                epoch = stage_start + epoch_offset + 1
                print(f"About to emit progress for epoch {epoch}", flush=True)
                emit("training_progress_stage", {
                    "epoch": epoch,
                    "total_epochs": total_epochs,
                    "loss": history.history["loss"][epoch_offset],
                    "accuracy": history.history["accuracy"][epoch_offset],
                    "val_loss": history.history["val_loss"][epoch_offset],
                    "val_accuracy": history.history["val_accuracy"][epoch_offset],
                })
                print(f"Emit completed for epoch {epoch}", flush=True)
    
                socketio.sleep(0)

        final_metrics={}

        # Additional metrics for classification datasets
        if dataset in ["Iris", "MNIST", "CIFAR-10", "Breast Cancer"]:
            predictions = model.predict(x_test)
            y_pred = np.argmax(predictions, axis=1)
            y_true = np.argmax(y_test, axis=1)
            conf_matrix = confusion_matrix(y_true, y_pred).tolist()  # Convert to list for JSON serialization
            final_metrics["confusion_matrix"] = conf_matrix
            print(final_metrics)

        elif dataset == "California Housing":
            predictions = model.predict(x_test).flatten()
            residuals = (y_test - predictions).tolist()  # Calculate residuals

            # Add metrics for California Housing
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)

            final_metrics["rmse"] = rmse
            final_metrics["r2"] = r2
            final_metrics["predicted_vs_actual"] = {
                "predicted": predictions.tolist(),  # Predicted values
                "actual": y_test.tolist()  # Actual values
            }
            final_metrics["residuals"] = residuals


        # Emit final training results
        print("Payload emitted to frontend:", {"message": "Training completed successfully!", "metrics": final_metrics})

        emit("training_complete", {
            "message": "Training completed successfully!",
            "metrics": final_metrics,
            "loss_over_time": history.history["loss"],
            "val_loss_over_time": history.history.get("val_loss", [])
        })
    
    except Exception as e:
        emit("training_error", {"error": str(e)})

def build_model_from_architecture(architecture, input_shape,dataset_name):
    """
    Build a Keras model based on the architecture provided.

    Args:
        architecture (dict): The model architecture containing nodes and edges.
        input_shape (tuple): Shape of the input data.

    Returns:
        keras.Model: A compiled Keras model.
    """
    nodes = architecture["nodes"]
    edges = architecture["edges"]

    # Validate input and output layers
    input_layer = next((node for node in nodes if node["type"] == "input"), None)
    output_layer = next((node for node in nodes if node["type"] == "output"), None)
    print(output_layer)
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
                activation=layer_data["activation"].lower(),
                input_shape=input_shape if len(model.layers) == 0 else None
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
            model.add(Input(shape=input_shape))
        

    # Configure the output layer dynamically based on the dataset
    output_units = determine_output_units(dataset_name)
    activation=output_layer["data"]["activation"].lower()  
    if activation=="none":
        activation=None
    model.add(Dense(
        units=output_units,  # Dynamically determine number of units
        activation=activation# Use user-defined activation
    ))
    print(model.summary())
     
    return model
    
def determine_output_units(dataset_name):
    """
    Determine the number of units for the output layer based on the dataset.

    Args:
        dataset_name (str): The name of the dataset (e.g., 'Iris', 'MNIST', 'CIFAR-10', 'California Housing', 'Breast Cancer').

    Returns:
        int: The number of units for the output layer.
    """
    if dataset_name == "Iris":
        return 3  # 3 classes
    elif dataset_name == "MNIST":
        return 10  # 10 digits
    elif dataset_name == "CIFAR-10":
        return 10  # 10 classes
    elif dataset_name == "California Housing":
        return 1  # Regression
    elif dataset_name == "Breast Cancer":
        return 1  # Binary classification
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Only 'Iris', 'MNIST', 'CIFAR-10', 'California Housing', and 'Breast Cancer' are supported.")
@app.route("/export", methods=["GET"])
def export_model():
    return {"message": "Model exported successfully!"}


if __name__ == "__main__":
    import eventlet
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
