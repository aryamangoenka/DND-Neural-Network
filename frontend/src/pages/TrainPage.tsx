import { useState, useEffect, useRef } from "react";
import { io, Socket } from "socket.io-client"; // Import the Socket.IO client
import "../styles/components/TrainPage.scss";
import { useDataset } from "../context/DatasetContext";

const TrainPage = (): JSX.Element => {
  const { dataset } = useDataset();
  const [lossFunction, setLossFunction] = useState("Categorical Cross-Entropy");
  const [optimizer, setOptimizer] = useState("Adam");
  const [learningRate, setLearningRate] = useState(0.001);
  const [batchSize, setBatchSize] = useState(32);
  const [epochs, setEpochs] = useState(10);
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState<string>(""); // Store training logs/progress
  const [liveMetrics, setLiveMetrics] = useState({
    epoch: 0,
    loss: 0,
    accuracy: 0,
    val_loss: 0,
    val_accuracy: 0,
  });
  const socketRef = useRef<Socket | null>(null); // Use Socket.IO client instance

  const handleTrain = (): void => {
    if (!dataset) {
      alert("No dataset selected! Please select a dataset before training.");
      return;
    }

    setIsTraining(true);

    const payload = {
      dataset,
      lossFunction,
      optimizer,
      learningRate,
      batchSize,
      epochs,
    };

    console.log(payload);

    // Send training configuration to the server
    socketRef.current?.emit("start_training", payload); // Emit 'start_training' event to the server
  };

  // Handle WebSocket connection setup
  useEffect(() => {
    // Initialize Socket.IO client
    const socket = io("http://127.0.0.1:5000"); // Connect to the backend server
    socketRef.current = socket;

    socket.on("connect", () => {
      console.log("Socket.IO connected:", socket.id);
    });

    // Listen for training progress updates
    socket.on("training_progress", (data) => {
      console.log("Training progress received:", data);
      setProgress((prev) => `${prev}\nEpoch ${data.epoch}: ${data.progress}`);
      setLiveMetrics(data);
    });

    // Listen for training start event
    socket.on("training_start", (data) => {
      console.log("Training started:", data.message);
      setProgress((prev) => `${prev}\n${data.message}`);
    });

    // Listen for training complete event
    socket.on("training_complete", (data) => {
      console.log("Training complete:", data.message);
      setProgress((prev) => `${prev}\n${data.message}`);
      setIsTraining(false);
    });

    // Listen for training errors
    socket.on("training_error", (data) => {
      console.error("Training error:", data.error);
      setProgress((prev) => `${prev}\nError: ${data.error}`);
      setIsTraining(false);
    });

    // Cleanup on component unmount
    return () => {
      socket.disconnect();
    };
  }, []);

  return (
    <div className="train-page">
      <div className="left-sidebar">
        <h2>Configuration</h2>

        <div className="config-section">
          <label>Dataset:</label>
          <p>{dataset || "No dataset selected"}</p>
        </div>

        <div className="config-section">
          <h3>Loss Function</h3>
          <select
            value={lossFunction}
            onChange={(e) => setLossFunction(e.target.value)}
          >
            <option value="Categorical Cross-Entropy">
              Categorical Cross-Entropy
            </option>
            <option value="Binary Cross-Entropy">Binary Cross-Entropy</option>
            <option value="Mean Squared Error">Mean Squared Error</option>
            <option value="Mean Absolute Error">Mean Absolute Error</option>
            <option value="Huber Loss">Huber Loss</option>
          </select>
        </div>

        <div className="config-section">
          <h3>Optimizer</h3>
          <select
            value={optimizer}
            onChange={(e) => setOptimizer(e.target.value)}
          >
            <option value="Adam">Adam</option>
            <option value="SGD">SGD</option>
            <option value="RMSprop">RMSprop</option>
            <option value="Adagrad">Adagrad</option>
          </select>
        </div>

        <div className="config-section">
          <h3>Hyperparameters</h3>
          <label>Learning Rate:</label>
          <input
            type="number"
            value={learningRate}
            onChange={(e) => setLearningRate(+e.target.value)}
            step="0.0001"
          />

          <label>Batch Size:</label>
          <input
            type="number"
            value={batchSize}
            onChange={(e) => setBatchSize(+e.target.value)}
            min="1"
          />

          <label>Epochs:</label>
          <input
            type="number"
            value={epochs}
            onChange={(e) => setEpochs(+e.target.value)}
            min="1"
          />
        </div>
      </div>

      <div className="main-content">
        <h2 style={{ textAlign: "center" }}>Training Visualization</h2>
        <div className="chart-section">
          <p>Loss and Accuracy graphs will appear here.</p>
          {/* Placeholder for chart visualization */}
        </div>
        <div className="log-section">
          <h3>Logs</h3>
          <textarea
            value={progress}
            readOnly
            style={{
              width: "100%",
              height: "200px",
              resize: "none",
              backgroundColor: "#f7f7f7",
            }}
          ></textarea>
        </div>
      </div>

      <div className="right-sidebar">
        <h2>Training Status</h2>
        {isTraining ? (
          <>
            <p>Epoch: {liveMetrics.epoch}</p>
            <p>Loss: {liveMetrics.loss.toFixed(4)}</p>
            <p>Accuracy: {liveMetrics.accuracy.toFixed(4)}</p>
            <p>Validation Loss: {liveMetrics.val_loss.toFixed(4)}</p>
            <p>Validation Accuracy: {liveMetrics.val_accuracy.toFixed(4)}</p>
          </>
        ) : (
          <p>Not training.</p>
        )}
        <div className="button-container">
          <button
            className="start-button"
            onClick={handleTrain}
            disabled={isTraining}
          >
            {isTraining ? "Training..." : "Start Training"}
          </button>
          <button
            className="stop-button"
            onClick={() => {
              setIsTraining(false);
              socketRef.current?.disconnect(); // Disconnect the Socket.IO connection
            }}
            disabled={!isTraining}
          >
            Stop Training
          </button>
        </div>
      </div>
    </div>
  );
};

export default TrainPage;
