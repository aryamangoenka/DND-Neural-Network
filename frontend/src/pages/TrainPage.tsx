import { useEffect, useRef, useState } from "react";
import { io, Socket } from "socket.io-client";
import { Line } from "react-chartjs-2"; // Import Line chart from react-chartjs-2
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js"; // Import Chart.js core components
import "../styles/components/TrainPage.scss";
import { useDataset } from "../context/DatasetContext";
import { useTrainPageContext } from "../context/TrainPageContext";

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const TrainPage = (): JSX.Element => {
  const { dataset } = useDataset();
  const {
    lossFunction,
    setLossFunction,
    optimizer,
    setOptimizer,
    learningRate,
    setLearningRate,
    batchSize,
    setBatchSize,
    epochs,
    setEpochs,
    isTraining,
    setIsTraining,
    progress,
    setProgress,
    liveMetrics,
    setLiveMetrics,
  } = useTrainPageContext();

  const socketRef = useRef<Socket | null>(null);
  const [trainingProgress, setTrainingProgress] = useState(0); // Progress percentage
  const [lossData, setLossData] = useState<number[]>([]); // Loss data for graph
  const [accuracyData, setAccuracyData] = useState<number[]>([]); // Accuracy data for graph
  const [labels, setLabels] = useState<string[]>([]); // Epoch labels for graph
  const [valLossData, setValLossData] = useState<number[]>([]); // Validation Loss data for graph
  const [valAccuracyData, setValAccuracyData] = useState<number[]>([]); // Validation Accuracy data for graph
  const [confusionMatrix, setConfusionMatrix] = useState<number[][] | null>(
    null
  );
  const [rmse, setRmse] = useState<number | null>(null);
  const [r2, setR2] = useState<number | null>(null);

  const handleTrain = (): void => {
    if (!dataset) {
      alert("No dataset selected! Please select a dataset before training.");
      return;
    }

    // Reset states to refresh the page
    setIsTraining(true);
    setTrainingProgress(0); // Reset progress bar
    setLossData([]); // Clear previous loss data
    setAccuracyData([]); // Clear previous accuracy data
    setLabels([]); // Clear epoch labels
    setProgress(""); // Clear logs
    setLiveMetrics({
      epoch: 0,
      loss: 0,
      accuracy: 0,
      val_loss: 0,
      val_accuracy: 0,
    }); // Reset live metrics
    setConfusionMatrix(null);
    setRmse(null);
    setR2(null);

    const payload = {
      dataset,
      lossFunction,
      optimizer,
      learningRate,
      batchSize,
      epochs,
    };

    console.log("Training payload sent:", payload);

    // Emit 'start_training' event to the backend
    socketRef.current?.emit("start_training", payload);
    socketRef.current?.connect();
  };

  useEffect(() => {
    // Initialize Socket.IO client
    const socket = io("http://127.0.0.1:5000");
    socketRef.current = socket;

    socket.on("connect", () => {
      console.log("Socket.IO connected:", socket.id);
    });

    // Listen for training start
    socket.on("training_start", (data) => {
      console.log("Training started:", data.message);
      setProgress((prev) => `${prev}\n${data.message}`);
    });

    // Listen for real-time training progress
    socket.on("training_progress", (data) => {
      console.log("Training progress received:", data);

      // Calculate progress percentage
      const percentage = Math.round((data.epoch / data.total_epochs) * 100);
      setTrainingProgress(percentage); // Update progress bar

      // Update metrics
      setProgress(
        (prev) =>
          `${prev}\nEpoch ${data.epoch}: Loss=${data.loss.toFixed(
            4
          )}, Accuracy=${data.accuracy.toFixed(
            4
          )}, Val_Loss=${data.val_loss.toFixed(
            4
          )}, Val_Accuracy=${data.val_accuracy.toFixed(4)}`
      );

      setLiveMetrics(data); // Update live metrics

      // Update graphs
      setLossData((prev) => [...prev, data.loss]); // Append loss
      setAccuracyData((prev) => [...prev, data.accuracy]); // Append accuracy
      setValLossData((prev) => [...prev, data.val_loss]); // Append val_loss
      setValAccuracyData((prev) => [...prev, data.val_accuracy]); // Append val_accuracy
      setLabels((prev) => [...prev, `Epoch ${data.epoch}`]); // Append epoch label
    });

    // Listen for training complete
    socket.on("training_complete", (data) => {
      console.log("Training complete:", data.message);
      setProgress((prev) => `${prev}\n${data.message}`);
      setIsTraining(false);
      setTrainingProgress(100); // Set progress to 100%
      if (data.metrics && data.metrics.confusion_matrix) {
        console.log("Confusion Matrix:", data.metrics.confusion_matrix);
        setConfusionMatrix(data.metrics.confusion_matrix);
      } else {
        console.log("No Confusion Matrix in metrics");
      }
      if (data.rmse) setRmse(data.rmse);
      if (data.r2) setR2(data.r2);
    });

    // Listen for training error
    socket.on("training_error", (data) => {
      console.error("Training error:", data.error);
      setProgress((prev) => `${prev}\nError: ${data.error}`);
      setIsTraining(false);
    });

    // Cleanup Socket.IO connection on component unmount
    return () => {
      socket.disconnect();
    };
  }, []);

  const lossChartData = {
    labels,
    datasets: [
      {
        label: "Loss",
        data: lossData,
        borderColor: "rgba(255, 99, 132, 1)",
        backgroundColor: "rgba(255, 99, 132, 0.2)",
        fill: true,
        tension: 0.4,
      },
      {
        label: "Validation Loss",
        data: valLossData,
        borderColor: "rgba(54, 162, 235, 1)",
        backgroundColor: "rgba(54, 162, 235, 0.2)",
        fill: true,
        tension: 0.4,
      },
    ],
  };

  const accuracyChartData = {
    labels,
    datasets: [
      {
        label: "Accuracy",
        data: accuracyData,
        borderColor: "rgba(54, 162, 235, 1)",
        backgroundColor: "rgba(54, 162, 235, 0.2)",
        fill: true,
        tension: 0.4,
      },
      {
        label: "Validation Accuracy",
        data: valAccuracyData,
        borderColor: "rgba(153, 102, 255, 1)",
        backgroundColor: "rgba(153, 102, 255, 0.2)",
        fill: true,
        tension: 0.4,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: { display: true },
      tooltip: { enabled: true },
    },
  };
  const getClassLabels = (datasetName: String): string[] => {
    switch (datasetName) {
      case "Iris":
        return ["Setosa", "Versicolor", "Virginica"];
      case "MNIST":
        return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
      case "CIFAR-10":
        return [
          "Airplane",
          "Car",
          "Bird",
          "Cat",
          "Deer",
          "Dog",
          "Frog",
          "Horse",
          "Ship",
          "Truck"
        ];
      case "Breast Cancer":
        return ["Benign", "Malignant"];
      default:
        return [];
    }
  };
  
  const renderConfusionMatrix = (): JSX.Element | null => {
    // Use a default value if `dataset` is null
    const dataset: string = useDataset()?.dataset || "Unknown";
  
    const classLabels = getClassLabels(dataset);
  
    if (!confusionMatrix || confusionMatrix.length === 0) {
      return <p>Confusion matrix is not available or invalid.</p>;
    }
  
    return (
      <div className="confusion-matrix-container">
        <table className="confusion-matrix">
          <thead>
            <tr>
              <th></th>
              {classLabels.map((label, index) => (
                <th key={index}>{label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {confusionMatrix.map((row, rowIndex) => (
              <tr key={rowIndex}>
                <th>{classLabels[rowIndex]}</th>
                {row.map((value, colIndex) => (
                  <td key={colIndex}>{value}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };
  
  

  return (
    <div className="train-page">
      <div className="left-sidebar">
        <h2>Configuration</h2>
        {/* Configuration Section */}
        {/* Add your dataset, loss function, optimizer, hyperparameter inputs */}
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
        <div className="progress-bar">
          <div
            className="progress-bar-fill"
            style={{ width: `${trainingProgress}%` }}
          >
            {trainingProgress.toFixed(2)}%
          </div>
        </div>

        <div className="charts-container">
          {/* Loss Graph */}
          <div className="chart-wrapper">
            <h3>Loss Over Time</h3>
            <Line data={lossChartData} options={chartOptions} />
          </div>

          {/* Accuracy Graph */}
          <div className="chart-wrapper">
            <h3>Accuracy Over Time</h3>
            <Line data={accuracyChartData} options={chartOptions} />
          </div>
        </div>
        {/* Dataset-Specific Visualizations */}
        <div className="dataset-visualizations">
          <h3>Additional Metrics</h3>
          {dataset &&
            ["Iris", "MNIST", "CIFAR-10", "Breast Cancer"].includes(
              dataset
            ) && (
              <div>
                <h3>Confusion Matrix</h3>
                {confusionMatrix ? (
                  <div className="confusion-matrix">
                    {renderConfusionMatrix()}
                  </div>
                ) : (
                  <p>Confusion matrix is not available yet.</p>
                )}
              </div>
            )}

          {dataset === "California Housing" && (
            <div>
              <h3>Regression Metrics</h3>
              <p>RMSE: {rmse?.toFixed(4) || "N/A"}</p>
              <p>R²: {r2?.toFixed(4) || "N/A"}</p>
            </div>
          )}
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
              socketRef.current?.disconnect();
            }}
            disabled={!isTraining}
          >
            Stop Training
          </button>
          <p>Batch Progress: {trainingProgress.toFixed(2)}%</p>
          <p>Loss: {liveMetrics.loss?.toFixed(6) || "N/A"}</p>
          <p>Accuracy: {liveMetrics.accuracy?.toFixed(6) || "N/A"}</p>
        </div>
      </div>
    </div>
  );
};

export default TrainPage;
