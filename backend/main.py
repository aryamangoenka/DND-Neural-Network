from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import json
import os
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization, Input, MultiHeadAttention, Lambda, Reshape
from dataset_loader import load_dataset  # Import the load_dataset function
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import clear_session
import shutil
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

#new work starts here
# Dictionary to track stop flags for each client
stop_flags = {}
tf.config.run_functions_eagerly(False)
# 📂 Path to store the trained model
EXPORT_FOLDER = "exports"
os.makedirs(EXPORT_FOLDER, exist_ok=True)  # Ensure the folder exists
TRAINED_MODEL_PATH = os.path.join(EXPORT_FOLDER, "trained_model.keras")
# Store the latest training configuration globally
latest_training_config = {}
x_train_shape=()

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
    global latest_training_config
    global x_train_shape
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
        latest_training_config = data
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
        model = build_model_from_architecture(model_architecture, x_train.shape[1:], dataset)
        x_train_shape=x_train.shape[1:]
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
            predictions = model.predict(x_test)

            # Ensure predictions and y_test are NumPy arrays
            predictions = predictions if isinstance(predictions, np.ndarray) else predictions.numpy()
            y_test = y_test if isinstance(y_test, np.ndarray) else y_test.numpy()

            # Calculate residuals
            residuals = (y_test - predictions).tolist()

            # Compute Regression Metrics
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)

            # Save metrics
            final_metrics["rmse"] = rmse
            final_metrics["r2"] = r2

            # Send raw data for residual plot
            final_metrics["residuals_plot"] = {
                "predicted_values": predictions.tolist(),
                "residuals": residuals
            }

            print("✅ Residual plot data sent to frontend.")


            # ------------------ Multicollinearity Heatmap ------------------

            # Convert feature set to DataFrame for easy correlation calculation
            feature_names = [
                "MedInc", "HouseAge", "AveRooms", "AveBedrms",
                "Population", "AveOccup", "Latitude", "Longitude"
            ]  # Replace with actual feature names if different
            # Convert Tensor to NumPy array if needed
            x_test_np = x_test if isinstance(x_test, np.ndarray) else x_test.numpy()

            x_test_df = pd.DataFrame(x_test_np, columns=feature_names)

            # Compute Pearson Correlation Matrix
            correlation_matrix = x_test_df.corr().values  # Get correlation values

            # Generate heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                correlation_matrix,
                annot=True,
                cmap="coolwarm",
                center=0,
                linewidths=0.5,
                fmt=".2f",
                annot_kws={"size": 10}
            )
            plt.title("Multicollinearity Heatmap")

            # Convert heatmap to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            heatmap_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()
            plt.close()

            # Add heatmap to metrics
            final_metrics["multicollinearity_heatmap"] = heatmap_base64
            print("✅ Multicollinearity heatmap generated.")





        # Emit final training results
        #print("Payload emitted to frontend:", {"message": "Training completed successfully!", "metrics": final_metrics})
        
        emit("training_complete", {
            "message": "Training completed successfully!",
            "metrics": final_metrics,
            "loss_over_time": history.history["loss"],
            "val_loss_over_time": history.history.get("val_loss", []),
            "success":True
        })
        
        # Save the model with custom_objects for Lambda layers
        try:
            # First try saving with include_optimizer=False which often helps with Lambda layers
            model.save(TRAINED_MODEL_PATH, include_optimizer=False)
        except Exception as save_error:
            print(f"Warning: Could not save model with standard method: {str(save_error)}")
            
            # Alternative approach: Save only the weights
            weights_path = os.path.join(EXPORT_FOLDER, "model_weights.h5")
            model.save_weights(weights_path)
            print(f"Saved model weights to {weights_path}")
            
            # Save the architecture separately
            with open(os.path.join(EXPORT_FOLDER, "model_architecture.json"), "w") as f:
                f.write(model.to_json())
            print("Saved model architecture separately")
            
    except Exception as e:
        emit("training_error", {"error": str(e)})

def build_model_from_architecture(architecture, input_shape, dataset_name):
    """
    Build a Keras model based on the architecture provided.

    Args:
        architecture (dict): The model architecture containing nodes and edges.
        input_shape (tuple): Shape of the input data.
        dataset_name (str): Name of the dataset.

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
    model.add(Input(shape=input_shape))  # Always start with an input layer

    # Add layers based on the nodes and edges
    # First, create a dictionary of nodes by ID for easy lookup
    nodes_by_id = {node["id"]: node for node in nodes}
    
    # Create a dictionary to track which nodes have been processed
    processed_nodes = {input_layer["id"]: True}
    
    # Start with the input layer and follow the edges
    current_layer_ids = [input_layer["id"]]
    
    while current_layer_ids:
        next_layer_ids = []
        
        for layer_id in current_layer_ids:
            # Find all edges that start from this layer
            outgoing_edges = [edge for edge in edges if edge["source"] == layer_id]
            
            for edge in outgoing_edges:
                target_id = edge["target"]
                
                # Skip if already processed
                if target_id in processed_nodes:
                    continue
                
                target_node = nodes_by_id[target_id]
                layer_type = target_node["type"]
                layer_data = target_node["data"]
                
                # Add the appropriate layer based on type
                if layer_type == "dense":
                    model.add(Dense(
                        units=layer_data["neurons"],
                        activation=None if layer_data["activation"].lower() == "none" else layer_data["activation"].lower()
                    ))
                elif layer_type == "convolution":
                    model.add(Conv2D(
                        filters=layer_data["filters"],
                        kernel_size=tuple(layer_data["kernelSize"]),
                        strides=tuple(layer_data["stride"]),
                        activation=None if layer_data["activation"].lower() == "none" else layer_data["activation"].lower()
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
                elif layer_type == "attention":
                    # Get the attention parameters
                    num_heads = layer_data.get("heads", 8)
                    key_dim = layer_data.get("keyDim", 64)
                    dropout_rate = layer_data.get("dropout", 0.0)
                    
                    # For MNIST data, we need to reshape the input for attention
                    if dataset_name == "MNIST" or dataset_name == "CIFAR-10":
                        # For image data, we need to reshape before applying attention
                        # First, add a reshape layer to convert 2D image data to sequence data
                        model.add(tf.keras.layers.Reshape((-1, input_shape[0])))  # Reshape to (sequence_length, features)
                        
                        # Use our custom attention layer instead of Lambda
                        model.add(CustomAttentionLayer(
                            num_heads=num_heads,
                            key_dim=key_dim,
                            dropout=dropout_rate,
                            name=f"attention_{target_id}"
                        ))
                        
                        # Reshape back to original shape for subsequent layers if needed
                        model.add(tf.keras.layers.Reshape(input_shape))
                    else:
                        # For non-image data, apply attention directly using our custom attention layer
                        model.add(CustomAttentionLayer(
                            num_heads=num_heads,
                            key_dim=key_dim,
                            dropout=dropout_rate,
                            name=f"attention_{target_id}"
                        ))
                elif layer_type == "output":
                    # Configure the output layer dynamically based on the dataset
                    output_units = determine_output_units(dataset_name)
                    activation = layer_data["activation"].lower()
                    if activation == "none":
                        activation = None
                    model.add(Dense(
                        units=output_units,
                        activation=activation
                    ))
                
                # Mark this node as processed
                processed_nodes[target_id] = True
                
                # Add to the next layer IDs to process
                if layer_type != "output":  # Don't process beyond output layer
                    next_layer_ids.append(target_id)
        
        # Update current layer IDs for the next iteration
        current_layer_ids = next_layer_ids
    
    print(model.summary())
    print(model.layers)
    
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

# Define custom attention function for Lambda layer
@tf.keras.utils.register_keras_serializable(package='custom_layers')
def apply_attention(x, num_heads=8, key_dim=64, dropout=0.0):
    attention_layer = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout
    )
    return attention_layer(x, x)

# Create a custom attention layer class instead of using Lambda
@tf.keras.utils.register_keras_serializable(package='custom_layers')
class CustomAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads=8, key_dim=64, dropout=0.0, **kwargs):
        super(CustomAttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout = dropout
        self.attention = None  # Will be initialized in build()
        
    def build(self, input_shape):
        self.attention = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout
        )
        super(CustomAttentionLayer, self).build(input_shape)
        
    def call(self, inputs, training=None):
        return self.attention(inputs, inputs, training=training)
        
    def get_config(self):
        config = super(CustomAttentionLayer, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'dropout': self.dropout
        })
        return config

# Register the custom function for Lambda layer serialization
tf.keras.utils.register_keras_serializable(package='custom_layers')(apply_attention)

# Enable unsafe deserialization for Lambda layers
tf.keras.config.enable_unsafe_deserialization()

# ✅ Load the trained Keras model if it exists
keras_model = None
if os.path.exists(TRAINED_MODEL_PATH):
    try:
        keras_model = tf.keras.models.load_model(TRAINED_MODEL_PATH, compile=False)
        print("✅ Successfully loaded the trained model")
    except Exception as e:
        print(f"⚠️ Could not load the trained model: {str(e)}")
        print("This is normal if no model has been trained yet.")

import torch
import torch.nn as nn
import torch.optim as optim



@app.route("/export/<format>", methods=["GET"])
def export_model(format):
    """
    Export the trained model in the specified format.
    Supported formats: py, ipynb, savedmodel, hdf5
    """
    try:
        # ✅ Load the latest trained model, not the dummy one
        if not os.path.exists(TRAINED_MODEL_PATH):
            return jsonify({"error": "No trained model found. Please train the model first."}), 400

        # Enable unsafe deserialization for Lambda layers
        tf.keras.config.enable_unsafe_deserialization()
        
        model = tf.keras.models.load_model(TRAINED_MODEL_PATH, compile=False)
        
        # Get the dataset name from the latest training config
        dataset_name = latest_training_config.get("dataset", "")

        # Export according to the requested format
        if format == "py":
            file_path = os.path.join(EXPORT_FOLDER, "trained_model.py")
            with open(file_path, "w") as f:
                f.write(generate_python_script(model, latest_training_config, x_train_shape))
            return send_file(file_path, as_attachment=True)

        elif format == "ipynb":
            file_path = os.path.join(EXPORT_FOLDER, "trained_model.ipynb")
            with open(file_path, "w") as f:
                f.write(generate_notebook(model, latest_training_config, x_train_shape))
            return send_file(file_path, as_attachment=True)

        elif format == "savedmodel":
            # ✅ Save the model in TensorFlow 2 SavedModel format
            saved_model_dir = os.path.join(EXPORT_FOLDER, "saved_model_tf2")
            
            try:
                # First try direct save
                model.save(saved_model_dir, save_format='tf', include_optimizer=False)
            except Exception as save_error:
                print(f"Warning: Could not save model directly: {str(save_error)}")
                
                # Alternative approach: Rebuild the model from architecture and copy weights
                try:
                    # Get the model architecture from the saved file
                    with open(MODEL_ARCHITECTURE_FILE, "r") as f:
                        model_architecture = json.load(f)
                    
                    # Build a fresh model from the architecture
                    fresh_model = build_model_from_architecture(
                        model_architecture, 
                        x_train_shape, 
                        dataset_name
                    )
                    
                    # Copy weights from the trained model to the fresh model
                    fresh_model.set_weights(model.get_weights())
                    
                    # Save the fresh model
                    fresh_model.save(saved_model_dir, save_format='tf', include_optimizer=False)
                except Exception as rebuild_error:
                    return jsonify({"error": f"Could not save model: {str(rebuild_error)}"}), 500

            # Zip the SavedModel directory
            zip_path = shutil.make_archive(saved_model_dir, 'zip', saved_model_dir)

            # 📤 Send the zipped SavedModel
            return send_file(
                zip_path,
                as_attachment=True,
                mimetype='application/zip'
            )

        elif format == "keras":
            file_path = os.path.join(EXPORT_FOLDER, "trained_model.keras")
            
            try:
                # First try direct save
                model.save(file_path, include_optimizer=False)
            except Exception as save_error:
                print(f"Warning: Could not save model directly: {str(save_error)}")
                
                # Alternative approach: Rebuild the model from architecture and copy weights
                try:
                    # Get the model architecture from the saved file
                    with open(MODEL_ARCHITECTURE_FILE, "r") as f:
                        model_architecture = json.load(f)
                    
                    # Build a fresh model from the architecture
                    fresh_model = build_model_from_architecture(
                        model_architecture, 
                        x_train_shape, 
                        dataset_name
                    )
                    
                    # Copy weights from the trained model to the fresh model
                    fresh_model.set_weights(model.get_weights())
                    
                    # Save the fresh model
                    fresh_model.save(file_path, include_optimizer=False)
                except Exception as rebuild_error:
                    return jsonify({"error": f"Could not save model: {str(rebuild_error)}"}), 500
                
            return send_file(file_path, as_attachment=True)
        
        elif format == "pytorch":
            # 🔥 Convert Keras model to PyTorch manually
            pytorch_script = generate_pytorch_script(model, latest_training_config, x_train_shape)

            # ✅ Save to a file
            file_path = os.path.join(EXPORT_FOLDER, "trained_model_pytorch.py")
            with open(file_path, "w") as f:
                f.write(pytorch_script)

            return send_file(file_path, as_attachment=True)
            

        else:
            return jsonify({"error": f"Unsupported format: {format}"}), 400

    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Export error: {str(e)}\n{traceback_str}")
        return jsonify({"error": str(e), "traceback": traceback_str}), 500

def generate_python_script(model, training_config, x_train_shape):
    """Generate a Python script that recreates the model architecture and training process."""
    
    # Get dataset name from training config
    dataset_name = training_config.get("dataset", "Unknown")
    
    # Determine input shape based on dataset
    input_shape_str = str(x_train_shape[1:]) if len(x_train_shape) > 1 else "(None,)"
    
    # Extract model parameters
    epochs = training_config.get("epochs", 10)
    batch_size = training_config.get("batch_size", 32)
    validation_split = training_config.get("validation_split", 0.2)
    
    # Start building the script
    script = [
        "import tensorflow as tf",
        "import numpy as np",
        "from tensorflow.keras.models import Sequential",
        "from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Input, MultiHeadAttention, Lambda, Reshape",
        "",
        "# Enable unsafe deserialization for Lambda layers",
        "tf.keras.config.enable_unsafe_deserialization()",
        "",
        "# Define custom attention function for Lambda layer",
        "@tf.keras.utils.register_keras_serializable(package='custom_layers')",
        "def apply_attention(x, num_heads=8, key_dim=64, dropout=0.0):",
        "    attention_layer = MultiHeadAttention(",
        "        num_heads=num_heads,",
        "        key_dim=key_dim,",
        "        dropout=dropout",
        "    )",
        "    return attention_layer(x, x)",
        "",
        "# Define custom attention layer class",
        "@tf.keras.utils.register_keras_serializable(package='custom_layers')",
        "class CustomAttentionLayer(tf.keras.layers.Layer):",
        "    def __init__(self, num_heads=8, key_dim=64, dropout=0.0, **kwargs):",
        "        super(CustomAttentionLayer, self).__init__(**kwargs)",
        "        self.num_heads = num_heads",
        "        self.key_dim = key_dim",
        "        self.dropout = dropout",
        "        self.attention = None  # Will be initialized in build()",
        "        ",
        "    def build(self, input_shape):",
        "        self.attention = MultiHeadAttention(",
        "            num_heads=self.num_heads,",
        "            key_dim=self.key_dim,",
        "            dropout=self.dropout",
        "        )",
        "        super(CustomAttentionLayer, self).build(input_shape)",
        "        ",
        "    def call(self, inputs, training=None):",
        "        return self.attention(inputs, inputs, training=training)",
        "        ",
        "    def get_config(self):",
        "        config = super(CustomAttentionLayer, self).get_config()",
        "        config.update({",
        "            'num_heads': self.num_heads,",
        "            'key_dim': self.key_dim,",
        "            'dropout': self.dropout",
        "        })",
        "        return config",
        "",
        "# Load and preprocess the dataset"
    ]
    
    # Add dataset-specific code
    if dataset_name == "MNIST":
        script.extend([
            "# Load MNIST dataset",
            "(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()",
            "# Normalize pixel values to be between 0 and 1",
            "x_train, x_test = x_train / 255.0, x_test / 255.0",
            "# Reshape for CNN input (28, 28, 1)",
            "x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)",
            "x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)",
            "# Convert class vectors to binary class matrices (one-hot encoding)",
            "y_train = tf.keras.utils.to_categorical(y_train, 10)",
            "y_test = tf.keras.utils.to_categorical(y_test, 10)"
        ])
        input_shape_str = "(28, 28, 1)"
    elif dataset_name == "CIFAR-10":
        script.extend([
            "# Load CIFAR-10 dataset",
            "(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()",
            "# Normalize pixel values to be between 0 and 1",
            "x_train, x_test = x_train / 255.0, x_test / 255.0",
            "# Convert class vectors to binary class matrices (one-hot encoding)",
            "y_train = tf.keras.utils.to_categorical(y_train, 10)",
            "y_test = tf.keras.utils.to_categorical(y_test, 10)"
        ])
        input_shape_str = "(32, 32, 3)"
    elif dataset_name == "Iris":
        script.extend([
            "# Load Iris dataset",
            "from sklearn.datasets import load_iris",
            "from sklearn.model_selection import train_test_split",
            "from sklearn.preprocessing import StandardScaler, OneHotEncoder",
            "",
            "iris = load_iris()",
            "X = iris.data",
            "y = iris.target.reshape(-1, 1)",
            "",
            "# Scale features",
            "scaler = StandardScaler()",
            "X_scaled = scaler.fit_transform(X)",
            "",
            "# One-hot encode the labels",
            "encoder = OneHotEncoder(sparse=False)",
            "y_encoded = encoder.fit_transform(y)",
            "",
            "# Split the data",
            "x_train, x_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)"
        ])
        input_shape_str = "(4,)"
    elif dataset_name == "Breast Cancer":
        script.extend([
            "# Load Breast Cancer dataset",
            "from sklearn.datasets import load_breast_cancer",
            "from sklearn.model_selection import train_test_split",
            "from sklearn.preprocessing import StandardScaler",
            "",
            "cancer = load_breast_cancer()",
            "X = cancer.data",
            "y = cancer.target",
            "",
            "# Scale features",
            "scaler = StandardScaler()",
            "X_scaled = scaler.fit_transform(X)",
            "",
            "# Split the data",
            "x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)"
        ])
        input_shape_str = "(30,)"
    elif dataset_name == "California Housing":
        script.extend([
            "# Load California Housing dataset",
            "from sklearn.datasets import fetch_california_housing",
            "from sklearn.model_selection import train_test_split",
            "from sklearn.preprocessing import StandardScaler",
            "",
            "housing = fetch_california_housing()",
            "X = housing.data",
            "y = housing.target",
            "",
            "# Scale features",
            "scaler = StandardScaler()",
            "X_scaled = scaler.fit_transform(X)",
            "",
            "# Split the data",
            "x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)"
        ])
        input_shape_str = "(8,)"
    else:
        script.extend([
            "# Replace with your own dataset loading code",
            "# x_train, y_train = ...",
            "# x_test, y_test = ..."
        ])
    
    script.append("")
    
    # Add model definition
    script.extend([
        "# Define the model",
        f"model = Sequential([",
        f"    Input(shape={input_shape_str}),"
    ])
    
    # Extract model architecture from the model object
    for layer in model.layers[1:]:  # Skip the input layer
        if isinstance(layer, tf.keras.layers.Dense):
            activation = f", activation='{layer.activation.__name__}'" if layer.activation.__name__ != "linear" else ""
            script.append(f"    Dense({layer.units}{activation}),")
        elif isinstance(layer, tf.keras.layers.Conv2D):
            activation = f", activation='{layer.activation.__name__}'" if layer.activation.__name__ != "linear" else ""
            script.append(f"    Conv2D({layer.filters}, {layer.kernel_size}, strides={layer.strides}, padding='{layer.padding}'{activation}),")
        elif isinstance(layer, tf.keras.layers.MaxPooling2D):
            script.append(f"    MaxPooling2D(pool_size={layer.pool_size}, strides={layer.strides}, padding='{layer.padding}'),")
        elif isinstance(layer, tf.keras.layers.Flatten):
            script.append(f"    Flatten(),")
        elif isinstance(layer, tf.keras.layers.Dropout):
            script.append(f"    Dropout({layer.rate}),")
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            script.append(f"    BatchNormalization(momentum={layer.momentum}, epsilon={layer.epsilon}),")
        elif isinstance(layer, tf.keras.layers.Lambda):
            if "attention" in layer.name:
                # Extract parameters from the layer name if available
                parts = layer.name.split('_')
                num_heads = 8
                key_dim = 64
                dropout_rate = 0.0
                
                for part in parts:
                    if part.startswith('heads'):
                        try:
                            num_heads = int(part.replace('heads', ''))
                        except:
                            pass
                    elif part.startswith('key'):
                        try:
                            key_dim = int(part.replace('key', ''))
                        except:
                            pass
                    elif part.startswith('drop'):
                        try:
                            dropout_rate = float(part.replace('drop', ''))
                        except:
                            pass
                
                script.append(f"    CustomAttentionLayer(num_heads={num_heads}, key_dim={key_dim}, dropout={dropout_rate}, name='{layer.name}'),")
            else:
                script.append(f"    Lambda(lambda x: x),  # Custom Lambda layer")
        elif isinstance(layer, CustomAttentionLayer):
            script.append(f"    CustomAttentionLayer(num_heads={layer.num_heads}, key_dim={layer.key_dim}, dropout={layer.dropout}, name='{layer.name}'),")
        elif isinstance(layer, tf.keras.layers.Reshape):
            script.append(f"    Reshape({layer.target_shape}),")
    
    script.append("])")
    script.append("")
    
    # Add compilation and training
    loss_function = training_config.get("loss", "categorical_crossentropy")
    optimizer = training_config.get("optimizer", "adam")
    
    script.extend([
        "# Compile the model",
        f"model.compile(optimizer='{optimizer}', loss='{loss_function}', metrics=['accuracy'])",
        "",
        "# Display model summary",
        "model.summary()",
        "",
        "# Train the model",
        f"history = model.fit(x_train, y_train, epochs={epochs}, batch_size={batch_size}, validation_split={validation_split})",
        "",
        "# Evaluate the model",
        "loss, accuracy = model.evaluate(x_test, y_test)",
        "print(f\"Test Loss: {loss}\")",
        "print(f\"Test Accuracy: {accuracy}\")",
        "",
        "# Save the model",
        "model.save('trained_model.keras')",
        "",
        "# Optional: Plot training history",
        "import matplotlib.pyplot as plt",
        "",
        "plt.figure(figsize=(12, 4))",
        "plt.subplot(1, 2, 1)",
        "plt.plot(history.history['loss'], label='Training Loss')",
        "plt.plot(history.history['val_loss'], label='Validation Loss')",
        "plt.xlabel('Epoch')",
        "plt.ylabel('Loss')",
        "plt.legend()",
        "",
        "plt.subplot(1, 2, 2)",
        "plt.plot(history.history['accuracy'], label='Training Accuracy')",
        "plt.plot(history.history['val_accuracy'], label='Validation Accuracy')",
        "plt.xlabel('Epoch')",
        "plt.ylabel('Accuracy')",
        "plt.legend()",
        "plt.tight_layout()",
        "plt.show()"
    ])
    
    return "\n".join(script)






def generate_notebook(model, training_config, x_train_shape):
    LOSS_FUNCTION_MAPPING = {
        "Categorical Cross-Entropy": "categorical_crossentropy",
        "Binary Cross-Entropy": "binary_crossentropy",
        "Mean Squared Error": "mse",
        "Mean Absolute Error": "mae",
        "Huber Loss": "huber"
    }
    optimizer = training_config.get("optimizer", "adam").lower()
    loss_function_init = training_config.get("lossFunction", "Categorical Cross-Entropy")
    loss_function=LOSS_FUNCTION_MAPPING.get(loss_function_init)
    batch_size = training_config.get("batchSize", 32)
    epochs = training_config.get("epochs", 10)
    dataset_name = training_config.get("dataset", "").lower()

    validation_split = 0.2 if dataset_name in ["iris", "breast cancer", "california housing"] else 0.1

# Dataset-specific preprocessing code
    preprocessing_code = ""

    if dataset_name == "iris":
        preprocessing_code = """
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load Iris dataset
data = load_iris()
X, y = data.data, data.target

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""
    elif dataset_name == "breast cancer":
        preprocessing_code = """
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Breast Cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
"""
    elif dataset_name == "california housing":
        preprocessing_code = """
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load California Housing dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""
    elif dataset_name == "mnist":
        preprocessing_code = """
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
"""
    elif dataset_name == "cifar-10":
        preprocessing_code = """
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
"""




        # Generate model layers code
    layers_code = ""
    layers_code+=f"Input(shape={x_train_shape}),\n"
    for i, layer in enumerate(model.layers):
        config = layer.get_config()
        if isinstance(layer, Dense):
            activation = config['activation'] if config['activation'] else 'None'
            layers_code += f"Dense({config['units']}, activation='{activation}'),\n"
        elif isinstance(layer, Conv2D):
            activation = config['activation'] if config['activation'] else 'None'
            layers_code += f"Conv2D({config['filters']}, {config['kernel_size']}, activation='{activation}'),\n"
        elif isinstance(layer, Flatten):
            layers_code += "Flatten(),\n"
        elif isinstance(layer, Dropout):
            layers_code += f"Dropout({config['rate']}),\n"
        elif isinstance(layer, MaxPooling2D):
            layers_code += f"MaxPooling2D(pool_size={config['pool_size']}),\n"
        elif isinstance(layer, BatchNormalization):
            layers_code += "BatchNormalization(),\n"
        elif isinstance(layer, MultiHeadAttention) or (hasattr(layer, 'layer') and isinstance(layer.layer, MultiHeadAttention)):
            # Handle MultiHeadAttention layer
            if hasattr(layer, 'layer') and isinstance(layer.layer, MultiHeadAttention):
                # If it's wrapped in a Lambda layer
                attention_config = layer.layer.get_config()
            else:
                attention_config = config
            
            num_heads = attention_config.get('num_heads', 8)
            key_dim = attention_config.get('key_dim', 64)
            dropout = attention_config.get('dropout', 0.0)
            
            # For MNIST or CIFAR-10, we need special handling with reshaping
            if dataset_name.lower() in ["mnist", "cifar-10"]:
                layers_code += f"# Reshape for attention with image data\n"
                layers_code += f"Reshape((-1, {x_train_shape[0]})),  # Reshape to sequence\n"
                layers_code += f"CustomAttentionLayer(num_heads={num_heads}, key_dim={key_dim}, dropout={dropout})),\n"
                layers_code += f"Reshape({x_train_shape}),  # Reshape back to original\n"
            else:
                layers_code += f"# Attention mechanism\n"
                layers_code += f"CustomAttentionLayer(num_heads={num_heads}, key_dim={key_dim}, dropout={dropout})),\n"

    # Return the complete script
    return f"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Input, MultiHeadAttention, Lambda, Reshape

# Define custom attention function for Lambda layer
@tf.keras.utils.register_keras_serializable(package='custom_layers')
def apply_attention(x, num_heads=8, key_dim=64, dropout=0.0):
    attention_layer = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout
    )
    return attention_layer(x, x)

# Define custom attention layer class
@tf.keras.utils.register_keras_serializable(package='custom_layers')
class CustomAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads=8, key_dim=64, dropout=0.0, **kwargs):
        super(CustomAttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout = dropout
        self.attention = None  # Will be initialized in build()
        
    def build(self, input_shape):
        self.attention = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout
        )
        super(CustomAttentionLayer, self).build(input_shape)
        
    def call(self, inputs, training=None):
        return self.attention(inputs, inputs, training=training)
        
    def get_config(self):
        config = super(CustomAttentionLayer, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'dropout': self.dropout
        })
        return config

# Dataset preprocessing
{preprocessing_code}

# Define the model
model = Sequential([
    {layers_code.strip()}
])

model.compile(optimizer='{optimizer}', loss='{loss_function}', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs={epochs}, batch_size={batch_size}, validation_split={validation_split})

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {{loss}}")
print(f"Test Accuracy: {{accuracy}}")

# Save the model
model.save('trained_model.keras')
"""






def generate_pytorch_script(model, training_config, x_train_shape):
    """
    Generates a PyTorch script equivalent to the trained Keras model.
    """
    LOSS_FUNCTION_MAPPING = {
        "Categorical Cross-Entropy": "nn.CrossEntropyLoss()",
        "Binary Cross-Entropy": "nn.BCELoss()",
        "Mean Squared Error": "nn.MSELoss()",
        "Mean Absolute Error": "nn.L1Loss()",
        "Huber Loss": "nn.SmoothL1Loss()"
    }
    
    optimizer = training_config.get("optimizer", "adam").lower()
    loss_function_init = training_config.get("lossFunction", "Categorical Cross-Entropy")
    loss_function = LOSS_FUNCTION_MAPPING.get(loss_function_init)
    batch_size = training_config.get("batchSize", 32)
    epochs = training_config.get("epochs", 10)
    dataset_name = training_config.get("dataset", "").lower()

    pytorch_code = f"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
"""

    if dataset_name == "iris":
        pytorch_code += f"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load Iris dataset
data = load_iris()
X, y = data.data, data.target

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# Convert to tensors
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)
"""

    elif dataset_name == "breast cancer":
        pytorch_code += f"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to tensors
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
"""

    elif dataset_name == "california housing":
        pytorch_code += f"""
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to tensors
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
"""

    elif dataset_name == "mnist":
        pytorch_code += f"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformations
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size={batch_size}, shuffle=False)
"""

    elif dataset_name == "cifar-10":
        pytorch_code += f"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformations
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size={batch_size}, shuffle=False)
"""
    # Wrap tabular datasets in DataLoader
    if dataset_name in ["iris", "breast cancer", "california housing"]:
        pytorch_code+=f'''
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size={batch_size}, shuffle=False)
'''
    pytorch_code += "\n\n# Define the PyTorch model\n"
    pytorch_code += "class NeuralNetwork(nn.Module):\n"
    pytorch_code += "    def __init__(self):\n"
    pytorch_code += "        super(NeuralNetwork, self).__init__()\n"

    previous_layer_output = x_train_shape  # Start with input shape
    is_flattened = False  # Track if we applied a Flatten layer

    for i, layer in enumerate(model.layers):
        config = layer.get_config()
        
        if isinstance(layer, Conv2D):
            in_channels = previous_layer_output[0] if i > 0 else 1  # Assume grayscale unless changed
            out_channels = config["filters"]
            kernel_size = tuple(config["kernel_size"])

            pytorch_code += f"        self.conv{i} = nn.Conv2d({in_channels}, {out_channels}, {kernel_size})\n"
            previous_layer_output = (out_channels, previous_layer_output[1], previous_layer_output[2])  # Update channels

        elif isinstance(layer, MaxPooling2D):
            pytorch_code += f"        self.pool{i} = nn.MaxPool2d(kernel_size={config['pool_size']})\n"

        elif isinstance(layer, Flatten):
            pytorch_code += f"        self.flatten = nn.Flatten()\n"
            is_flattened = True

        elif isinstance(layer, Dense):
            # Use the correct input features for the first dense layer
            in_features = previous_layer_output[0] if previous_layer_output else x_train_shape[0]

            # Define Dense (Fully Connected) Layer
            pytorch_code += f"        self.fc{i} = nn.Linear({in_features}, {config['units']})\n"

            # Define Activation Function (Only if explicitly specified)
            activation_fn = config['activation'].lower() if config['activation'] else 'none'
            if activation_fn == "relu":
                pytorch_code += f"        self.act{i} = nn.ReLU()\n"
            elif activation_fn == "sigmoid":
                pytorch_code += f"        self.act{i} = nn.Sigmoid()\n"
            elif activation_fn == "tanh":
                pytorch_code += f"        self.act{i} = nn.Tanh()\n"
            elif activation_fn == "softmax":
                pytorch_code += f"        self.act{i} = nn.Softmax(dim=1)\n"

            # Update previous layer output size
            previous_layer_output = (config['units'],)


        elif isinstance(layer, Dropout):
            pytorch_code += f"        self.drop{i} = nn.Dropout({config['rate']})\n"

        elif isinstance(layer, BatchNormalization):
            if len(previous_layer_output) == 1:
                pytorch_code += f"        self.bn{i} = nn.BatchNorm1d({previous_layer_output[0]})\n"
            else:
                pytorch_code += f"        self.bn{i} = nn.BatchNorm2d({previous_layer_output[0]})\n"
                
        elif isinstance(layer, MultiHeadAttention) or (hasattr(layer, 'layer') and isinstance(layer.layer, MultiHeadAttention)):
            # Handle MultiHeadAttention layer
            if hasattr(layer, 'layer') and isinstance(layer.layer, MultiHeadAttention):
                # If it's wrapped in a Lambda layer
                attention_config = layer.layer.get_config()
            else:
                attention_config = config
            
            num_heads = attention_config.get('num_heads', 8)
            key_dim = attention_config.get('key_dim', 64)
            dropout = attention_config.get('dropout', 0.0)
            
            # PyTorch's MultiheadAttention has a different interface
            pytorch_code += f"        # Attention mechanism\n"
            
            # For image data, we need special handling
            if dataset_name.lower() in ["mnist", "cifar-10"]:
                pytorch_code += f"        # Handle attention for image data\n"
                pytorch_code += f"        batch_size = x.size(0)\n"
                pytorch_code += f"        x_reshaped = x.view(batch_size, -1, {x_train_shape[0]})\n"
                pytorch_code += f"        x_t = x_reshaped.transpose(0, 1)  # PyTorch expects seq_len, batch, features\n"
                pytorch_code += f"        attn_output, _ = self.attention{i}(x_t, x_t, x_t)\n"
                pytorch_code += f"        x = attn_output.transpose(0, 1).reshape(batch_size, {x_train_shape[0]}, {x_train_shape[1]}, {x_train_shape[2]})\n"
            else:
                pytorch_code += f"        # Prepare input for attention (assuming batch_first=False)\n"
                pytorch_code += f"        x_t = x.transpose(0, 1)  # PyTorch expects seq_len, batch, features\n"
                pytorch_code += f"        attn_output, _ = self.attention{i}(x_t, x_t, x_t)\n"
                pytorch_code += f"        x = attn_output.transpose(0, 1)  # Back to batch, seq_len, features\n"

    pytorch_code += "    def forward(self, x):\n"
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Dense):
            pytorch_code += f"        x = self.fc{i}(x)\n"
            if hasattr(layer, 'activation') and layer.activation is not None:
                pytorch_code += f"        x = self.act{i}(x)\n"
        elif isinstance(layer, Conv2D):
            pytorch_code += f"        x = self.conv{i}(x)\n"
        elif isinstance(layer, Flatten):
            pytorch_code += f"        x = self.flatten(x)\n"
        elif isinstance(layer, Dropout):
            pytorch_code += f"        x = self.drop{i}(x)\n"
        elif isinstance(layer, MaxPooling2D):
            pytorch_code += f"        x = self.pool{i}(x)\n"
        elif isinstance(layer, BatchNormalization):
            pytorch_code += f"        x = self.bn{i}(x)\n"
        elif isinstance(layer, MultiHeadAttention) or (hasattr(layer, 'layer') and isinstance(layer.layer, MultiHeadAttention)):
            # For PyTorch's MultiheadAttention, we need to handle the input differently
            pytorch_code += f"        # Prepare input for attention (assuming batch_first=False)\n"
            pytorch_code += f"        x_t = x.transpose(0, 1)  # PyTorch expects seq_len, batch, features\n"
            pytorch_code += f"        attn_output, _ = self.attention{i}(x_t, x_t, x_t)\n"
            pytorch_code += f"        x = attn_output.transpose(0, 1)  # Back to batch, seq_len, features\n"
    
    pytorch_code += "        return x\n"

    pytorch_code += "\n# Initialize the model\n"
    pytorch_code += "model = NeuralNetwork()\n"

    pytorch_code += f"""
# Define loss function and optimizer
loss_function = {loss_function}
optimizer = optim.{optimizer.capitalize()}(model.parameters(), lr=0.001)

# Training loop
for epoch in range({epochs}):
    for batch in train_loader:
        x_batch, y_batch = batch
        optimizer.zero_grad()
        output = model(x_batch)
        loss = loss_function(output, y_batch)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {{epoch+1}} - Loss: {{loss.item()}}")

# Save model
torch.save(model.state_dict(), "trained_model.pth")


"""

    return pytorch_code




@app.route("/api/clear_model", methods=["POST"])
def clear_saved_model():
    """
    Clear the saved_model.json file when requested by the frontend.
    """
    try:
        if os.path.exists(MODEL_ARCHITECTURE_FILE):
            with open(MODEL_ARCHITECTURE_FILE, "w") as file:
                file.write('{}')  # Overwrite with empty JSON
            print("🧹 saved_model.json has been cleared by frontend request.")
            return jsonify({"status": "success", "message": "Model cleared successfully!"}), 200
        else:
            return jsonify({"status": "error", "message": "Model file not found."}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    import eventlet
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
