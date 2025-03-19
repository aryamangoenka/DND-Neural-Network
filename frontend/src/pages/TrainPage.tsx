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
    setModelTrained,
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

  const [residuals, setResiduals] = useState<number[]>([]); // Residual values
  const [predictedValues, setPredictedValues] = useState<number[]>([]); // Residual values

  const [residualsChartData, setResidualsChartData] = useState<any>(null);
  const [chartKey, setChartKey] = useState(0);

  const [heatmapImage, setHeatmapImage] = useState<string | null>(null);

  const handleTrain = (): void => {
    if (!dataset) {
      alert("No dataset selected! Please select a dataset before training.");
      return;
    }

    // Reset states to refresh the page
    setIsTraining(true);
    setTrainingProgress(0); // Reset progress bar
    setLossData([]); // Clear previous loss data
    setValAccuracyData([]); // Clear previous accuracy data
    setValLossData([]); // Clear previous val loss data
    setAccuracyData([]);
    setChartKey((prev) => prev + 1); // Clear previous val accuracy data

    setPredictedValues([]);
    setHeatmapImage("");
    setResidualsChartData({ datasets: [] });
    setResiduals([]);
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
      dataset, // Assuming this is correct
      lossFunction: lossFunction, // Ensure this matches backend keys exactly
      optimizer: optimizer.toLowerCase(), // Optimizer should be lowercase (e.g., "adam")
      learningRate: parseFloat(learningRate.toString()), // Convert to float
      batchSize: parseInt(batchSize.toString(), 10), // Convert to integer
      epochs: parseInt(epochs.toString(), 10), // Convert to integer
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
      setLossData((prev) => [...prev, parseFloat(data.loss)]); // Append loss
      setAccuracyData((prev) => [...prev, parseFloat(data.accuracy)]); // Append accuracy
      setValLossData((prev) => [...prev, parseFloat(data.val_loss)]); // Append val_loss
      setValAccuracyData((prev) => [...prev, parseFloat(data.val_accuracy)]); // Append val_accuracy
      setLabels((prev) => [...prev, `Epoch ${parseFloat(data.epoch)}`]); // Append epoch label
    });

    // Listen for training complete
    socket.on("training_complete", (data) => {
      setModelTrained(true);
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

      // ✅ Corrected residuals plot data extraction
      if (data.metrics.residuals_plot) {
        console.log(data.metrics.residuals_plot);

        // Ensure data is not undefined
        const predictedData =
          data.metrics.residuals_plot.predicted_values || [];
        const residualsData = data.metrics.residuals_plot.residuals || [];

        setPredictedValues(predictedData);
        setResiduals(residualsData);
      }

      // Check and set the multicollinearity heatmap data
      if (data.metrics?.multicollinearity_heatmap) {
        setHeatmapImage(
          `data:image/png;base64,${data.metrics.multicollinearity_heatmap}`
        );
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
        borderColor: "#FF7043",
        backgroundColor: "rgba(255, 99, 132, 0.2)",
        fill: true,
        tension: 0.4,
      },
      {
        label: "Validation Loss",
        data: valLossData,
        borderColor: "#29B6F6",
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
          backgroundColor: "#F0F4F8",
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
                    color: "#e9eade",
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
                    color: "#e9eade",
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

  // Corrected Residuals Plot
  useEffect(() => {
    if (
      predictedValues &&
      residuals &&
      predictedValues.length > 0 &&
      residuals.length > 0
    ) {
      const plotData = predictedValues.map((pred, index) => ({
        x: pred, // Predicted values on X-axis
        y: residuals[index], // Residuals on Y-axis
      }));

      setResidualsChartData({
        datasets: [
          {
            label: "Residuals Plot",
            data: plotData,
            borderColor: "rgba(255, 159, 64, 1)", // Orange color
            backgroundColor: "rgba(255, 159, 64, 0.2)", // Light orange
            pointRadius: 5,
            pointHoverRadius: 7,
            showLine: false, // Scatter plot style
          },
        ],
      });
    }
  }, [predictedValues, residuals]);

  return (
    <div className="train-page">
      <div className="left-sidebar">
        <h2>Configuration</h2>
        {/* Configuration Section */}
        <div className="config-section">
          <label>
            <i className="fas fa-database"></i> Dataset
          </label>
          <div className="selected-dataset">
            {dataset || "No dataset selected"}
          </div>
        </div>

        <div className="config-section">
          <h3>
            <i className="fas fa-chart-line"></i> Loss Function
          </h3>
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
          <h3>
            <i className="fas fa-cogs"></i> Optimizer
          </h3>
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

        <div className="config-section-1">
          <h3>Hyperparameters</h3>

          <label>
            <i className="fas fa-tachometer-alt"></i> Learning Rate:
          </label>
          <input
            type="number"
            value={learningRate}
            onChange={(e) => setLearningRate(+e.target.value)}
            step="0.0001"
          />

          <label>
            <i className="fas fa-layer-group"></i> Batch Size:
          </label>
          <input
            type="number"
            value={batchSize}
            onChange={(e) => setBatchSize(+e.target.value)}
            min="1"
          />

          <label>
            <i className="fas fa-history"></i> Epochs:
          </label>
          <input
            type="number"
            value={epochs}
            onChange={(e) => setEpochs(+e.target.value)}
            min="1"
          />
        </div>
      </div>

      <div className="main-content">
        <div className="training-header">
          <h2>
            <i className="fas fa-brain"></i> Model Training
            {isTraining && (
              <span className="training-badge">Training in progress</span>
            )}
          </h2>
        </div>

        <div className="progress-section">
          <div className="progress-info">
            <span>Training Progress:</span>
            <span>{trainingProgress.toFixed(2)}%</span>
          </div>
          <div className="progress-bar">
            <div
              className="progress-bar-fill text-light"
              style={{ width: `${trainingProgress}%` }}
            >
              {trainingProgress > 5 ? `${trainingProgress.toFixed(0)}%` : ""}
            </div>
          </div>
        </div>

        <div className="charts-container">
          {/* Visualizations for Non-Cali Datasets */}
          {dataset !== "California Housing" && (
            <div className="non-cali-visualizations">
              {/* Loss and Accuracy Graphs */}
              <div className="charts-row">
                <div className="chart-small">
                  <h3>
                    <i className="fas fa-chart-line"></i> Loss Over Time
                  </h3>
                  <Line data={lossChartData} options={chartOptions} />
                </div>
                <div className="chart-small">
                  <h3>
                    <i className="fas fa-bullseye"></i> Accuracy Over Time
                  </h3>
                  <Line data={accuracyChartData} options={chartOptions} />
                </div>
              </div>

              {/* Confusion Matrix */}
              <div className="matrix-container">
                <h3>
                  <i className="fas fa-th"></i> Confusion Matrix
                </h3>
                {confusionMatrix ? (
                  <ConfusionMatrix
                    matrix={confusionMatrix}
                    labels={getClassLabels(dataset)}
                  />
                ) : (
                  <div className="empty-state">
                    <i className="fas fa-chart-pie fa-3x"></i>
                    <p>
                      Confusion matrix will appear here once training has
                      started
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Dataset-Specific Visualizations */}
        <div className="dataset-visualizations">
          {dataset === "California Housing" && (
            <div className="cali-visualizations">
              <div className="charts-row">
                {/* Smaller Loss Graph on Top */}
                <div className="loss-chart-wrapper">
                  <div className="chart-small">
                    <h3>
                      <i className="fas fa-chart-line"></i> Loss Over Time
                    </h3>
                    <Line data={lossChartData} options={chartOptions} />
                  </div>
                </div>
                <div className="residuals-chart-wrapper">
                  <h4>
                    <i className="fas fa-chart-bar"></i> Residuals Plot
                  </h4>
                  {residualsChartData ? (
                    <Line
                      key={`residuals-${chartKey}`}
                      data={residualsChartData}
                      options={{
                        responsive: true,
                        plugins: {
                          legend: { display: true },
                          tooltip: { enabled: true },
                        },
                        scales: {
                          x: {
                            type: "linear",
                            title: {
                              display: true,
                              text: "Predicted Values", // X-axis
                            },
                            grid: { display: true },
                          },
                          y: {
                            title: {
                              display: true,
                              text: "Residuals (Actual - Predicted)", // Y-axis
                            },
                            grid: { display: true },
                          },
                        },
                      }}
                    />
                  ) : (
                    <div className="empty-state">
                      <i className="fas fa-chart-bar fa-3x"></i>
                      <p>
                        Residuals Plot will appear here once training has
                        started
                      </p>
                    </div>
                  )}
                </div>
              </div>

              <div className="regression-charts-container">
                <div className="chart-wrapper">
                  <h4>
                    <i className="fas fa-th"></i> Multicollinearity Heatmap
                  </h4>
                  <div>
                    {heatmapImage ? (
                      <img
                        src={heatmapImage}
                        alt="Multicollinearity Heatmap"
                        style={{ width: "100%", maxWidth: "800px" }}
                      />
                    ) : (
                      <div className="empty-state">
                        <i className="fas fa-th fa-3x"></i>
                        <p>
                          Heatmap will appear here once training has started
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="right-sidebar">
        <h2>Training Control</h2>
        <div className="button-container">
          <button
            className={`button ${isTraining ? "training" : ""}`}
            onClick={handleTrain}
            disabled={isTraining}
          >
            {isTraining ? (
              <>
                <i className="fas fa-spinner fa-spin"></i> Training...
              </>
            ) : (
              <>
                <i className="fas fa-play"></i> Start Training
              </>
            )}
          </button>

          <div className="metrics-container">
            <h3>
              <i className="fas fa-chart-bar"></i> Live Metrics
            </h3>

            <p>
              Batch Progress: <span>{trainingProgress.toFixed(2)}%</span>
            </p>
            <p>
              Loss: <span>{liveMetrics.loss?.toFixed(6) || "N/A"}</span>
            </p>
            <p>
              Validation Loss:{" "}
              <span>{liveMetrics.val_loss?.toFixed(6) || "N/A"}</span>
            </p>

            {dataset !== "California Housing" && (
              <>
                <p>
                  Accuracy:{" "}
                  <span>{liveMetrics.accuracy?.toFixed(6) || "N/A"}</span>
                </p>
                <p>
                  Validation Accuracy:{" "}
                  <span>{liveMetrics.val_accuracy?.toFixed(6) || "N/A"}</span>
                </p>
              </>
            )}

            {dataset === "California Housing" && (
              <>
                <p>
                  RMSE: <span>{rmse !== null ? rmse.toFixed(4) : "N/A"}</span>
                </p>
                <p>
                  R² Score: <span>{r2 !== null ? r2.toFixed(4) : "N/A"}</span>
                </p>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainPage;
