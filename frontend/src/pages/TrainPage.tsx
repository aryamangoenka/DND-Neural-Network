import { useState, useEffect, useRef } from "react";
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
  const socketRef = useRef<WebSocket | null>(null);

  const handleTrain = async (): Promise<void> => {
    if (!dataset) {
      alert("No dataset selected! Please select a dataset before training.");
      return;
    }

    setIsTraining(true);

    try {
      const payload = {
        dataset,
        lossFunction,
        optimizer,
        learningRate,
        batchSize,
        epochs,
      };

      console.log(payload);

      // Open WebSocket connection
      socketRef.current = new WebSocket("ws://127.0.0.1:5000/ws/train");

      // Send training configuration to WebSocket on connection
      socketRef.current.onopen = () => {
        console.log("WebSocket connected");
        socketRef.current?.send(JSON.stringify(payload));
      };

      // Listen for messages from the WebSocket
      socketRef.current.onmessage = (event) => {
        const message = JSON.parse(event.data);
        console.log("WebSocket message received:", message);

        // Update training progress or metrics
        if (message.type === "progress") {
          setProgress((prev) => `${prev}\n${message.data}`);
        } else if (message.type === "metrics") {
          setLiveMetrics(message.data);
        }
      };

      // Handle WebSocket errors
      socketRef.current.onerror = (error) => {
        console.error("WebSocket error:", error);
        alert("An error occurred with the WebSocket connection.");
        setIsTraining(false);
      };

      // Close WebSocket connection on training completion
      socketRef.current.onclose = () => {
        console.log("WebSocket connection closed");
        setIsTraining(false);
      };
    } catch (error) {
      alert(`An error occurred: ${error}`);
      setIsTraining(false);
    }
  };

  // Cleanup WebSocket on component unmount
  useEffect(() => {
    return () => {
      socketRef.current?.close();
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
              socketRef.current?.close(); // Close WebSocket connection
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
