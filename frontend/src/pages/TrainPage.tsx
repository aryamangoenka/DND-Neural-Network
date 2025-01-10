import { useEffect, useRef, useState } from "react";
import { io, Socket } from "socket.io-client";
import { Line } from "react-chartjs-2"; // Import Line chart from react-chartjs-2

import { FixedSizeGrid } from "react-window"; // Import React-Window for grids
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
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
  Legend,
  Filler
);

const TrainPage = (): JSX.Element => {
  const dataset: string = useDataset()?.dataset || "Unknown";
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
  const [predicted, setPredicted] = useState<number[]>([]); // Predicted values
  const [actual, setActual] = useState<number[]>([]); // Actual values
  const [residuals, setResiduals] = useState<number[]>([]); // Residual values

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
    // Listen for staged progress
    socket.on("training_progress_stage", (data) => {
      console.log("Staged training progress:", data);

      // Emit a summary for the stage
      setProgress(
        (prev) =>
          `${prev}\n[Stage Progress]: Completed up to Epoch ${
            data.epoch
          }. Stage Loss: ${data.loss.toFixed(
            4
          )}, Stage Accuracy: ${data.accuracy.toFixed(
            4
          )}, Validation Loss: ${data.val_loss.toFixed(
            4
          )}, Validation Accuracy: ${data.val_accuracy.toFixed(4)}`
      );
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
      if (data.metrics.rmse) setRmse(data.metrics.rmse);
      if (data.metrics.r2) setR2(data.metrics.r2);

      if (data.metrics.predicted_vs_actual) {
        setPredicted(data.metrics.predicted_vs_actual.predicted);
        setActual(data.metrics.predicted_vs_actual.actual);
      }

      if (data.metrics.residuals) {
        setResiduals(data.metrics.residuals);
      }
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
          "Truck",
        ];
      case "Breast Cancer":
        return ["Benign", "Malignant"];
      default:
        return [];
    }
  };

  const ConfusionMatrix = ({
    matrix,
    labels,
  }: {
    matrix: number[][];
    labels: string[];
  }) => {
    const containerRef = useRef<HTMLDivElement>(null);

    // Dynamically calculate dimensions based on dataset size
    const [dimensions, setDimensions] = useState({ width: 500, height: 500 });

    useEffect(() => {
      const updateDimensions = () => {
        if (containerRef.current) {
          const { clientWidth, clientHeight } = containerRef.current;

          // Adjust size dynamically for datasets with many classes (e.g., CIFAR-10, MNIST)
          const dynamicSize = Math.max(400, labels.length * 70);

          setDimensions({
            width: Math.min(clientWidth, dynamicSize),
            height: Math.min(clientHeight, dynamicSize),
          });
        }
      };

      updateDimensions();
      window.addEventListener("resize", updateDimensions);
      return () => window.removeEventListener("resize", updateDimensions);
    }, [labels.length]);

    const { width, height } = dimensions;

    const columnWidth = width / (matrix[0].length + 1); // +1 for label column
    const rowHeight = height / (matrix.length + 1); // +1 for label row

    // Determine background color intensity based on value
    const getCellColor = (value: number, maxValue: number) => {
      const opacity = maxValue > 0 ? value / maxValue : 0;
      return `rgba(76, 175, 80, ${opacity})`; // Green color gradient
    };

    const maxValue = Math.max(...matrix.flat());

    return (
      <div
        className="confusion-matrix-container"
        ref={containerRef}
        style={{
          width: "100%",
          maxWidth: "800px",
          margin: "20px auto",
          overflow: "auto",
          padding: "10px",
          backgroundColor: "#ffffff",
          border: "2px solid #ccc",
          borderRadius: "10px",
          boxShadow: "0px 4px 8px rgba(0, 0, 0, 0.2)",
        }}
      >
        <FixedSizeGrid
          columnCount={matrix[0].length + 1}
          rowCount={matrix.length + 1}
          columnWidth={columnWidth}
          rowHeight={rowHeight}
          height={height}
          width={width}
        >
          {({ columnIndex, rowIndex, style }) => {
            if (rowIndex === 0 && columnIndex === 0) {
              return <div style={style}></div>; // Top-left corner
            }

            if (rowIndex === 0) {
              // Column headers
              return (
                <div
                  style={{
                    ...style,
                    fontWeight: "bold",
                    textAlign: "center",
                    backgroundColor: "#4ade80",
                    color: "#ffffff",
                    border: "1px solid #ddd",
                  }}
                >
                  {labels[columnIndex - 1]}
                </div>
              );
            }

            if (columnIndex === 0) {
              // Row headers
              return (
                <div
                  style={{
                    ...style,
                    fontWeight: "bold",
                    textAlign: "center",
                    backgroundColor: "#4ade80",
                    color: "#ffffff",
                    border: "1px solid #ddd",
                    padding: "10px",
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                  }}
                >
                  {labels[rowIndex - 1]}
                </div>
              );
            }

            // Data cells with dynamic color scaling
            const value = matrix[rowIndex - 1][columnIndex - 1];

            return (
              <div
                style={{
                  ...style,
                  textAlign: "center",
                  fontWeight: "bold",
                  color: value > 0.5 * maxValue ? "#fff" : "#333",
                  backgroundColor: getCellColor(value, maxValue),
                  border: "1px solid #ddd",
                }}
              >
                {value}
              </div>
            );
          }}
        </FixedSizeGrid>
      </div>
    );
  };

  
  // Predicted vs. Actual Chart Data
  const predictedVsActualChartData = {
    datasets: [
      {
        label: "Predicted vs Actual",
        data: actual.map((value, index) => ({
          x: value, // Correct X-axis with actual values
          y: predicted[index], // Correct Y-axis with predicted values
        })),
        borderColor: "rgba(75, 192, 192, 1)",
        backgroundColor: "rgba(75, 192, 192, 0.2)",
        pointRadius: 5,
        pointHoverRadius: 7,
        showLine: false, // Keep it as a scatter plot
      },
    ],
  };

  const residualsChartData = {
    datasets: [
      {
        label: "Residuals",
        data: actual.map((value, index) => ({
          x: value, // Actual value on X-axis
          y: residuals[index], // Residual value on Y-axis
        })),
        borderColor: "rgba(255, 159, 64, 1)",
        backgroundColor: "rgba(255, 159, 64, 0.2)",
        pointRadius: 5,
        pointHoverRadius: 7,
        showLine: false, // Keep it as a scatter plot
      },
    ],
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
          {/* Visualizations for Non-Cali Datasets */}
          {dataset !== "California Housing" && (
            <div className="non-cali-visualizations">
              {/* Loss and Accuracy Graphs */}
              {/* Loss and Accuracy Graphs */}
              <div className="charts-row">
                <div className="chart-small">
                  <h3>Loss Over Time</h3>
                  <Line data={lossChartData} options={chartOptions} />
                </div>
                <div className="chart-small">
                  <h3>Accuracy Over Time</h3>
                  <Line data={accuracyChartData} options={chartOptions} />
                </div>
              </div>

              {/* Confusion Matrix */}

              <h3>Confusion Matrix</h3>
              {confusionMatrix ? (
                <ConfusionMatrix
                  matrix={confusionMatrix}
                  labels={getClassLabels(dataset)}
                />
              ) : (
                <p>Confusion matrix is not available yet.</p>
              )}
            </div>
          )}
        </div>

        {/* Dataset-Specific Visualizations */}
        <div className="dataset-visualizations">
          {dataset === "California Housing" && (
            <div className="cali-visualizations">
              {/* Smaller Loss Graph on Top */}
              <div className="loss-chart-wrapper">
                <h3>Loss Over Time</h3>
                <Line data={lossChartData} options={chartOptions} />
              </div>

              {/* Predicted vs. Actual and Residuals Plots Side by Side */}
              <div className="regression-charts-container">
                <div className="chart-wrapper">
                  <h4>Predicted vs Actual</h4>
                  <Line
                    data={predictedVsActualChartData}
                    options={{
                      responsive: true,
                      plugins: {
                        legend: { display: true },
                        tooltip: { enabled: true },
                      },
                      scales: {
                        x: { title: { display: true, text: "Actual Values" } },
                        y: {
                          title: { display: true, text: "Predicted Values" },
                        },
                      },
                    }}
                  />
                </div>
                <div className="chart-wrapper">
                  <h4>Residuals</h4>
                  <Line
                    data={residualsChartData}
                    options={{
                      responsive: true,
                      plugins: {
                        legend: { display: true },
                        tooltip: { enabled: true },
                      },
                      scales: {
                        x: { title: { display: true, text: "Actual Values" } },
                        y: { title: { display: true, text: "Residuals" } },
                      },
                    }}
                  />
                </div>
              </div>
            </div>
          )}
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

          <p>Batch Progress: {trainingProgress.toFixed(2)}%</p>
          <p>Loss: {liveMetrics.loss?.toFixed(6) || "N/A"}</p>
          <p>Accuracy: {liveMetrics.accuracy?.toFixed(6) || "N/A"}</p>
        </div>
      </div>
    </div>
  );
};

export default TrainPage;
