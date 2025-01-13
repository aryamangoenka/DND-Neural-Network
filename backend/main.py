from flask import Flask, jsonify, request, send_file
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
import tensorflow as tf
from tensorflow.keras.backend import clear_session

# Dictionary to track stop flags for each client
stop_flags = {}
tf.config.run_functions_eagerly(False)
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
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=300, ping_interval=25)
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



        # Use tf.data.Dataset for efficient batching
        batch_size = training_config["batchSize"]
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size, drop_remainder=False)
        val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size, drop_remainder=False)

        # Clear previous TensorFlow session to reset the model
        clear_session()

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
                train_dataset,
                validation_data=val_dataset,
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
            if dataset == "Breast Cancer":
                # Binary classification: Apply threshold for class prediction
                y_pred = (predictions > 0.5).astype(int) # Convert to 0 or 1
                y_true = y_test  # Ensure y_test is also flat
            else:
                # Multi-class classification
                y_pred = np.argmax(predictions, axis=1)
                y_true = np.argmax(y_test, axis=1)
            conf_matrix = confusion_matrix(y_true, y_pred).tolist()  # Convert to list for JSON serialization
            final_metrics["confusion_matrix"] = conf_matrix
            

        elif dataset == "California Housing":
            # Make predictions
            predictions = model.predict(x_test)

            # Ensure predictions and y_test are NumPy arrays
            predictions = predictions if isinstance(predictions, np.ndarray) else predictions.numpy()
            y_test = y_test if isinstance(y_test, np.ndarray) else y_test.numpy()

            # Calculate residuals
            residuals = (y_test - predictions).tolist()  # ✅ Removed .numpy()

            # Compute Regression Metrics
            rmse = np.sqrt(mean_squared_error(y_test, predictions))  # ✅ NumPy arrays used
            r2 = r2_score(y_test, predictions)                      # ✅ NumPy arrays used

            # Save metrics
            final_metrics["rmse"] = rmse
            final_metrics["r2"] = r2

            # Save predicted vs actual for visualization
            final_metrics["predicted_vs_actual"] = {
                "predicted": predictions.tolist(),  # ✅ Removed redundant .numpy()
                "actual": y_test.tolist()           # ✅ Removed redundant .numpy()
            }

            # Save residuals plot data (Predicted vs Residuals)
            final_metrics["residuals_plot"] = {
                "predictedValues": predictions.tolist(),  # ✅ Predicted on X-axis
                "residuals": residuals              # ✅ Residuals on Y-axis
            }
            #print(final_metrics["predicted_vs_actual"])
            print(final_metrics["predicted_vs_actual"]["predicted"])

        # Emit final training results
        #print("Payload emitted to frontend:", {"message": "Training completed successfully!", "metrics": final_metrics})

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

# Existing imports and app setup remain unchanged...

EXPORT_FOLDER = "exports"
os.makedirs(EXPORT_FOLDER, exist_ok=True)  # Ensure the folder exists

@app.route("/export/<format>", methods=["GET"])
def export_model(format):
    """
    Export the trained model in the specified format.
    Supported formats: py, ipynb, savedmodel, hdf5
    """
    try:
        # Load the trained model (assuming it's saved in memory or disk)
        model_path = "trained_model.h5"  # Assuming the model is saved here for simplicity
        model = tf.keras.models.load_model(model_path)

        # Export according to the requested format
        if format == "py":
            file_path = os.path.join(EXPORT_FOLDER, "trained_model.py")
            with open(file_path, "w") as f:
                f.write(generate_python_script())
            return send_file(file_path, as_attachment=True)

        elif format == "ipynb":
            file_path = os.path.join(EXPORT_FOLDER, "trained_model.ipynb")
            with open(file_path, "w") as f:
                f.write(generate_notebook())
            return send_file(file_path, as_attachment=True)

        elif format == "savedmodel":
            saved_model_dir = os.path.join(EXPORT_FOLDER, "saved_model")
            model.save(saved_model_dir, save_format="tf")
            return jsonify({"message": "Model exported as TensorFlow SavedModel."})

        elif format == "hdf5":
            file_path = os.path.join(EXPORT_FOLDER, "trained_model.h5")
            model.save(file_path, save_format="h5")
            return send_file(file_path, as_attachment=True)

        else:
            return jsonify({"error": f"Unsupported format: {format}"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def generate_python_script():
    """
    Generate Python script code for model architecture and training.
    """
    return """
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the model architecture
model = Sequential([
    Dense(128, activation='relu', input_shape=(input_shape_here,)),
    Dense(64, activation='relu'),
    Dense(output_units, activation='softmax')  # Adjust based on the dataset
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Save the model
model.save('trained_model.h5')
    """


def generate_notebook():
    """
    Generate Jupyter Notebook content for model training.
    """
    notebook_content = {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "import tensorflow as tf\n",
                    "from tensorflow.keras.models import Sequential\n",
                    "from tensorflow.keras.layers import Dense\n\n",
                    "# Define the model\n",
                    "model = Sequential([\n",
                    "    Dense(128, activation='relu', input_shape=(input_shape_here,)),\n",
                    "    Dense(64, activation='relu'),\n",
                    "    Dense(output_units, activation='softmax')\n",
                    "])\n\n",
                    "model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])\n",
                    "model.fit(x_train, y_train, epochs=10, batch_size=32)\n",
                    "model.save('trained_model.h5')\n"
                ],
                "execution_count": None,
                "outputs": []
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 2
    }
    return json.dumps(notebook_content)



if __name__ == "__main__":
    import eventlet
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
