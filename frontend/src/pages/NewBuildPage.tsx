import React, { useState, useRef, useEffect } from "react";
import "../styles/components/NewBuildPage.scss";
import { useNewBuildPageContext } from "../context/NewBuildPageContext";
import ReactFlow, {
  addEdge,
  Background,
  Controls,
  applyNodeChanges,
  applyEdgeChanges,
  Node,
  Connection,
  NodeChange,
  EdgeChange,
  OnNodesChange,
  OnEdgesChange,
} from "reactflow";
import "reactflow/dist/style.css";
import { io, Socket } from "socket.io-client";
import { Line } from "react-chartjs-2";
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
} from "chart.js";
import {
  DenseNode,
  ConvolutionNode,
  MaxPoolingNode,
  FlattenNode,
  DropoutNode,
  BatchNormalizationNode,
  InputNode,
  OutputNode,
  AttentionNode,
} from "../components/CustomNodes";
import NavBar from "../components/NavBar";
import axios from "axios";

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

// Define the node types for ReactFlow
const nodeTypes = {
  input: InputNode,
  output: OutputNode,
  dense: DenseNode,
  convolution: ConvolutionNode,
  maxpooling: MaxPoolingNode,
  flatten: FlattenNode,
  dropout: DropoutNode,
  batchnormalization: BatchNormalizationNode,
  attention: AttentionNode,
};

// Define the sidebar navigation options
type SidebarOption =
  | "layers"
  | "templates"
  | "settings"
  | "hyperparameters"
  | "training"
  | "model_config"
  | "layer_params";

// Add these near the top of the file with other type definitions
type ValidationErrors = string[];

// Update the TrainingProgress type to match the context
type TrainingProgress = {
  currentEpoch: number;
  totalEpochs: number;
  accuracy: number;
  loss: number;
  valAccuracy: number;
  valLoss: number;
};

// Define a custom event for saving the model
const triggerSaveModel = () => {
  const event = new CustomEvent("saveModel");
  window.dispatchEvent(event);
};

const NewBuildPage: React.FC = () => {
  const {
    nodes,
    setNodes,
    edges,
    setEdges,
    selectedDataset,
    setSelectedDataset,
    isTraining,
    setIsTraining,
    trainingProgress,
    setTrainingProgress,
    trainingConfig,
    setTrainingConfig,
  } = useNewBuildPageContext();

  // Socket.io reference
  const socketRef = useRef<Socket | null>(null);

  // State to track which sidebar option is active
  const [activeSidebarOption, setActiveSidebarOption] =
    useState<SidebarOption>("layers");

  // State to track the selected node for parameter editing
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);

  // State to track the selected visualization
  const [selectedVisualization, setSelectedVisualization] =
    useState<string>("accuracy");

  // Add these state variables with the other useState declarations
  const [validationErrors, setValidationErrors] = useState<ValidationErrors>(
    []
  );
  const [showValidationErrors, setShowValidationErrors] =
    useState<boolean>(false);

  // Chart data state variables
  const [labels, setLabels] = useState<string[]>([]);
  const [lossData, setLossData] = useState<number[]>([]);
  const [accuracyData, setAccuracyData] = useState<number[]>([]);
  const [valLossData, setValLossData] = useState<number[]>([]);
  const [valAccuracyData, setValAccuracyData] = useState<number[]>([]);

  // Confusion matrix state
  const [confusionMatrix, setConfusionMatrix] = useState<number[][]>([]);
  const [chartKey, setChartKey] = useState<number>(0); // For forcing chart re-renders

  // California Housing specific visualizations
  const [heatmapImage, setHeatmapImage] = useState<string | null>(null);
  const [residuals, setResiduals] = useState<number[]>([]);
  const [predictedValues, setPredictedValues] = useState<number[]>([]);
  const [rmse, setRmse] = useState<number | null>(null);
  const [r2, setR2] = useState<number | null>(null);

  // Chart data objects
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
    maintainAspectRatio: false,
    plugins: {
      legend: { display: true },
      tooltip: { enabled: true },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  // State to track if the model has been saved
  const [isModelSaved, setIsModelSaved] = useState<boolean>(false);

  // Add the export status message state inside the component
  const [exportStatusMessage, setExportStatusMessage] = useState<string>("");

  // Reset isModelSaved when nodes or edges change
  useEffect(() => {
    setIsModelSaved(false);
  }, [nodes, edges]);

  // Function to get visualization options based on dataset
  const getVisualizationOptions = () => {
    // Default options available for all datasets
    const defaultOptions = [
      { value: "accuracy", label: "Accuracy" },
      { value: "loss", label: "Loss" },
    ];

    // Dataset-specific options that are currently supported
    const datasetOptions: Record<string, { value: string; label: string }[]> = {
      MNIST: [{ value: "confusion_matrix", label: "Confusion Matrix" }],
      "CIFAR-10": [{ value: "confusion_matrix", label: "Confusion Matrix" }],
      Iris: [{ value: "confusion_matrix", label: "Confusion Matrix" }],
      "Breast Cancer": [
        { value: "confusion_matrix", label: "Confusion Matrix" },
      ],
      "California Housing": [
        { value: "heatmap", label: "Multicollinearity Heatmap" },
      ],
    };

    // Combine default options with dataset-specific options
    return selectedDataset
      ? [...defaultOptions, ...(datasetOptions[selectedDataset] || [])]
      : defaultOptions;
  };

  // Update the socket.io event handling for training progress
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
      // Reset chart data when training starts
      setLabels([]);
      setLossData([]);
      setAccuracyData([]);
      setValLossData([]);
      setValAccuracyData([]);
      setConfusionMatrix([]);

      // Reset training progress
      setTrainingProgress({
        currentEpoch: 0,
        totalEpochs: parseInt(trainingConfig.epochs.toString()),
        accuracy: 0,
        loss: 0,
        valAccuracy: 0,
        valLoss: 0,
      });
    });

    // Listen for real-time training progress
    socket.on("training_progress", (data) => {
      console.log("Training progress received:", data);

      // Parse numeric values to ensure they're numbers, not strings
      const accuracy =
        typeof data.accuracy === "string"
          ? parseFloat(data.accuracy)
          : data.accuracy;
      const loss =
        typeof data.loss === "string" ? parseFloat(data.loss) : data.loss;
      const valAccuracy =
        typeof data.val_accuracy === "string"
          ? parseFloat(data.val_accuracy)
          : data.val_accuracy;
      const valLoss =
        typeof data.val_loss === "string"
          ? parseFloat(data.val_loss)
          : data.val_loss;

      // Update training progress state with properly parsed values
      setTrainingProgress({
        currentEpoch: data.epoch,
        totalEpochs: data.total_epochs,
        accuracy: accuracy,
        loss: loss,
        valAccuracy: valAccuracy,
        valLoss: valLoss,
      });

      // Update chart data with properly parsed values
      setLabels((prev) => [...prev, `Epoch ${data.epoch}`]);
      setLossData((prev) => [...prev, loss]);
      setAccuracyData((prev) => [...prev, accuracy]);
      setValLossData((prev) => [...prev, valLoss]);
      setValAccuracyData((prev) => [...prev, valAccuracy]);

      // Force chart re-render
      setChartKey((prev) => prev + 1);
    });

    // Listen for training complete
    socket.on("training_complete", (data) => {
      console.log("Training complete:", data.message);
      setIsTraining(false);

      // Check for confusion matrix data
      if (data.metrics && data.metrics.confusion_matrix) {
        console.log(
          "Confusion Matrix received:",
          data.metrics.confusion_matrix
        );
        setConfusionMatrix(data.metrics.confusion_matrix);
        setChartKey((prev) => prev + 1); // Force chart re-render
      } else {
        console.log("No Confusion Matrix in metrics");
      }

      // California Housing specific metrics
      if (selectedDataset === "California Housing") {
        // Set RMSE and R2 score if available
        if (data.metrics?.rmse) setRmse(data.metrics.rmse);
        if (data.metrics?.r2) setR2(data.metrics.r2);

        // Set residuals and predicted values for residual plot
        if (data.metrics?.residuals_plot) {
          console.log(
            "Residuals plot data received:",
            data.metrics.residuals_plot
          );
          setPredictedValues(
            data.metrics.residuals_plot.predicted_values || []
          );
          setResiduals(data.metrics.residuals_plot.residuals || []);
        }

        // Set heatmap image if available
        if (data.metrics?.multicollinearity_heatmap) {
          console.log("Multicollinearity heatmap received");
          setHeatmapImage(
            `data:image/png;base64,${data.metrics.multicollinearity_heatmap}`
          );
        }
      }
    });

    // Listen for training error
    socket.on("training_error", (data) => {
      console.error("Training error:", data.error);
      setIsTraining(false);
      alert(`Training error: ${data.error}`);
    });

    // Cleanup Socket.IO connection on component unmount
    return () => {
      socket.disconnect();
    };
  }, []);

  // Add an event listener for the save model event
  useEffect(() => {
    const saveModelHandler = () => {
      handleSaveModel();
    };

    window.addEventListener("saveModel", saveModelHandler);

    // Cleanup function
    return () => {
      window.removeEventListener("saveModel", saveModelHandler);
    };
  }, []);

  // Function to normalize dataset name for the backend
  const normalizeDatasetName = (datasetName: string): string => {
    // Convert dataset names to lowercase and replace spaces with underscores
    const normalized = datasetName.toLowerCase().replace(/\s+/g, "_");

    // Map specific dataset names to their expected backend format
    const datasetMapping: Record<string, string> = {
      mnist: "MNIST",
      "cifar-10": "CIFAR-10",
      iris: "Iris",
      breast_cancer: "Breast Cancer",
      california_housing: "California Housing",
    };

    return datasetMapping[normalized] || normalized;
  };

  // Helper function to get class labels for different datasets
  const getClassLabels = (datasetName: string): string[] => {
    switch (datasetName) {
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
      case "Iris":
        return ["Setosa", "Versicolor", "Virginica"];
      case "Breast Cancer":
        return ["Benign", "Malignant"];
      default:
        return [];
    }
  };

  // Function to handle saving the model
  const handleSaveModel = async (): Promise<void> => {
    // Check if dataset is selected
    if (!selectedDataset) {
      alert("No dataset selected! Please select a dataset before saving.");
      return;
    }

    // Validate the model architecture
    const errors = validateLayerParameters();
    if (errors.length > 0) {
      setValidationErrors(errors);
      displayValidationErrors(errors);
      return;
    }

    try {
      // Show saving indicator
      const saveButton = document.querySelector(
        ".save-button"
      ) as HTMLButtonElement;
      if (saveButton) {
        saveButton.innerHTML =
          '<i class="fas fa-spinner fa-spin"></i> Saving...';
        saveButton.disabled = true;
      }

      // Prepare model architecture
      const modelArchitecture = {
        nodes: nodes.map((node) => ({
          id: node.id,
          type: node.type,
          position: node.position,
          data: node.data,
        })),
        edges: edges.map((edge) => ({
          id: edge.id,
          source: edge.source,
          target: edge.target,
        })),
        dataset: normalizeDatasetName(selectedDataset),
      };

      // Make a POST request to the backend
      const response = await fetch("http://127.0.0.1:5000/save_model", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(modelArchitecture),
      });

      // Reset button state
      if (saveButton) {
        saveButton.innerHTML = '<i class="fas fa-save"></i> Save Model';
        saveButton.disabled = false;
      }

      if (!response.ok) {
        const errorData = await response.json();
        console.error("Backend Error:", errorData);

        // Display a more detailed error message
        const errorMessage =
          errorData.error || "Unknown error occurred while saving the model";
        alert(`Saving failed: ${errorMessage}`);
        return;
      }

      const data = await response.json();
      console.log("Response from Backend:", data);

      // Set model as saved if successful
      if (
        data.message &&
        (data.message.includes("saved successfully") ||
          data.message.includes("Model architecture saved"))
      ) {
        setIsModelSaved(true);

        // Display success message
        alert(`Model saved successfully! ${data.message}`);
      } else {
        // Display the exact message from the backend
        alert(data.message || "Model saved successfully!");
      }
    } catch (error) {
      console.error("Error saving model:", error);

      // Reset button state
      const saveButton = document.querySelector(
        ".save-button"
      ) as HTMLButtonElement;
      if (saveButton) {
        saveButton.innerHTML = '<i class="fas fa-save"></i> Save Model';
        saveButton.disabled = false;
      }

      // Display a more detailed error message
      alert(
        `An error occurred while saving the model: ${
          error instanceof Error ? error.message : String(error)
        }`
      );
    }
  };

  // Start training the model
  const handleStartTraining = async () => {
    // Validate the model architecture
    const errors = validateLayerParameters();
    if (errors.length > 0) {
      setValidationErrors(errors);
      displayValidationErrors(errors);
      return;
    }

    if (!selectedDataset) {
      alert("No dataset selected! Please select a dataset before training.");
      return;
    }

    // Set training state
    setIsTraining(true);

    // Reset training progress and chart data
    setTrainingProgress({
      currentEpoch: 0,
      totalEpochs: parseInt(trainingConfig.epochs.toString()),
      accuracy: 0,
      loss: 0,
      valAccuracy: 0,
      valLoss: 0,
    });

    // Reset chart data
    setLabels([]);
    setLossData([]);
    setAccuracyData([]);
    setValLossData([]);
    setValAccuracyData([]);

    // Reset confusion matrix
    setConfusionMatrix([]);

    // Reset California Housing specific visualizations
    if (selectedDataset === "California Housing") {
      setHeatmapImage(null);
      setResiduals([]);
      setPredictedValues([]);
      setRmse(null);
      setR2(null);
    }

    // Prepare training payload with proper type conversions
    const trainingPayload = {
      dataset: normalizeDatasetName(selectedDataset),
      lossFunction: trainingConfig.lossFunction,
      optimizer: trainingConfig.optimizer.toLowerCase(),
      batchSize: parseInt(trainingConfig.batchSize.toString(), 10),
      epochs: parseInt(trainingConfig.epochs.toString(), 10),
      learningRate: parseFloat(trainingConfig.learningRate.toString()),
    };

    console.log("Training payload sent:", trainingPayload);

    // Emit the start_training event
    if (socketRef.current) {
      socketRef.current.emit("start_training", trainingPayload);
      console.log("Start training event emitted");
    } else {
      console.error("Socket connection not established");
      alert(
        "Socket connection not established. Please refresh the page and try again."
      );
      setIsTraining(false);
      return;
    }

    // Show success message
    console.log("Training started successfully!");
  };

  // Function to handle stopping the training
  const handleStopTraining = (): void => {
    if (socketRef.current) {
      socketRef.current.emit("stop_training");
      console.log("Stop training signal sent");
      setIsTraining(false);
      alert("Training stopped by user");
    }
  };

  // Function to add a new layer
  const addLayer = (type: string): void => {
    // Get count of existing layers of this type to create a unique name
    const existingLayersOfType = nodes.filter(
      (node) => node.type?.toLowerCase() === type.toLowerCase()
    ).length;

    const layerNumber = existingLayersOfType + 1;

    // Create a formatted layer name
    const getDefaultLayerName = (type: string): string => {
      switch (type.toLowerCase()) {
        case "dense":
          return `Dense Layer ${layerNumber}`;
        case "convolution":
          return `Conv2D Layer ${layerNumber}`;
        case "maxpooling":
          return `MaxPool2D Layer ${layerNumber}`;
        case "flatten":
          return `Flatten Layer ${layerNumber}`;
        case "dropout":
          return `Dropout Layer ${layerNumber}`;
        case "batchnormalization":
          return `BatchNorm Layer ${layerNumber}`;
        case "attention":
          return `Attention Layer ${layerNumber}`;
        case "output":
          return `Output Layer ${layerNumber}`;
        default:
          return `${type} Layer ${layerNumber}`;
      }
    };

    const defaultParams: Record<string, any> = {
      dense: { neurons: 64, activation: "None" },
      convolution: { filters: 32, kernelSize: [3, 3], stride: [1, 1] },
      maxpooling: { poolSize: [2, 2], stride: [2, 2], padding: "none" },
      flatten: {},
      dropout: { rate: 0.2 },
      batchnormalization: { momentum: 0.99, epsilon: 0.001 },
      attention: { heads: 8, keyDim: 64, dropout: 0.0 },
      output: { activation: "None" },
    };

    const newNode: Node = {
      id: `${type}-${Date.now()}`,
      data: {
        label: getDefaultLayerName(type),
        ...defaultParams[type.toLowerCase()],
      },
      position: { x: Math.random() * 600, y: Math.random() * 400 },
      type: type.toLowerCase(),
    };

    setNodes((nds) => [...nds, newNode]);
  };

  // Function to load a template
  const loadTemplate = (templateKey: string): void => {
    const templates: Record<string, Node[]> = {
      "Simple Feedforward": [
        {
          id: "Input-1",
          data: { label: "Input Layer" },
          position: { x: 100, y: 100 },
          type: "input",
        },
        {
          id: "Dense-1",
          data: {
            label: "Hidden Dense Layer",
            neurons: 64,
            activation: "ReLU",
          },
          position: { x: 300, y: 200 },
          type: "dense",
        },
        {
          id: "Output-1",
          data: { label: "Output Layer", activation: "Softmax" },
          position: { x: 500, y: 300 },
          type: "output",
        },
      ],
      CNN: [
        {
          id: "Input-1",
          data: { label: "Input Layer" },
          position: { x: 100, y: 100 },
          type: "input",
        },
        {
          id: "Conv-1",
          data: {
            label: "Convolution Layer",
            filters: 32,
            kernelSize: [3, 3],
            stride: [1, 1],
            activation: "ReLU",
          },
          position: { x: 300, y: 200 },
          type: "convolution",
        },
        {
          id: "MaxPool-1",
          data: {
            label: "MaxPooling Layer",
            poolSize: [2, 2],
            stride: [2, 2],
            padding: "valid",
          },
          position: { x: 500, y: 300 },
          type: "maxpooling",
        },
        {
          id: "Flatten-1",
          data: { label: "Flatten Layer" },
          position: { x: 700, y: 400 },
          type: "flatten",
        },
        {
          id: "Dense-1",
          data: { label: "Dense Layer", neurons: 64, activation: "ReLU" },
          position: { x: 900, y: 500 },
          type: "dense",
        },
        {
          id: "Output-1",
          data: { label: "Output Layer", activation: "Softmax" },
          position: { x: 1100, y: 600 },
          type: "output",
        },
      ],
      Transformer: [
        {
          id: "Input-1",
          data: { label: "Input Layer" },
          position: { x: 100, y: 100 },
          type: "input",
        },
        {
          id: "BatchNorm-1",
          data: {
            label: "Layer Normalization",
            momentum: 0.99,
            epsilon: 0.001,
          },
          position: { x: 300, y: 100 },
          type: "batchnormalization",
        },
        {
          id: "Attention-1",
          data: {
            label: "Multi-Head Attention",
            heads: 8,
            keyDim: 64,
            dropout: 0.1,
          },
          position: { x: 500, y: 100 },
          type: "attention",
        },
        {
          id: "Dense-1",
          data: {
            label: "Feed-Forward Network",
            neurons: 256,
            activation: "ReLU",
          },
          position: { x: 700, y: 100 },
          type: "dense",
        },
        {
          id: "Output-1",
          data: { label: "Output Layer", activation: "Softmax" },
          position: { x: 900, y: 100 },
          type: "output",
        },
      ],
      Regression: [
        {
          id: "Input-1",
          data: { label: "Input Layer" },
          position: { x: 100, y: 100 },
          type: "input",
        },
        {
          id: "Dense-1",
          data: { label: "Dense Layer", neurons: 64, activation: "ReLU" },
          position: { x: 300, y: 200 },
          type: "dense",
        },
        {
          id: "Dense-2",
          data: { label: "Dense Layer", neurons: 32, activation: "ReLU" },
          position: { x: 500, y: 300 },
          type: "dense",
        },
        {
          id: "Output-1",
          data: { label: "Output Layer", activation: "None" },
          position: { x: 700, y: 400 },
          type: "output",
        },
      ],
      Blank: [
        {
          id: "Input-1",
          data: { label: "Input Layer" },
          position: { x: 100, y: 100 },
          type: "input",
        },
        {
          id: "Output-1",
          data: { label: "Output Layer" },
          position: { x: 700, y: 300 },
          type: "output",
        },
      ],
    };

    setNodes(templates[templateKey]);
    setEdges([]);
  };

  // Function to update node parameters
  const updateParameter = (param: string, value: any): void => {
    if (!selectedNode) return;

    setNodes((nds) =>
      nds.map((node) =>
        node.id === selectedNode.id
          ? { ...node, data: { ...node.data, [param]: value } }
          : node
      )
    );

    setSelectedNode((prevNode) =>
      prevNode
        ? { ...prevNode, data: { ...prevNode.data, [param]: value } }
        : null
    );
  };

  // ReactFlow event handlers
  const onNodesChange: OnNodesChange = (changes: NodeChange[]) => {
    setNodes((nds) => applyNodeChanges(changes, nds));
  };

  const onEdgesChange: OnEdgesChange = (changes: EdgeChange[]) => {
    setEdges((eds) => applyEdgeChanges(changes, eds));
  };

  const onConnect = (connection: Connection): void => {
    setEdges((eds) => addEdge(connection, eds));
  };

  const onNodeClick = (_: React.MouseEvent, node: Node): void => {
    setSelectedNode(node);
    setActiveSidebarOption("layer_params");
  };

  // Validate layer parameters before saving or training
  const validateLayerParameters = (): ValidationErrors => {
    const errors: ValidationErrors = [];

    // Check if dataset is selected
    if (!selectedDataset) {
      errors.push("Please select a dataset before saving or training");
      return errors;
    }

    // Check if there are at least two nodes (input and output)
    if (nodes.length < 2) {
      errors.push("Model must have at least input and output layers");
      return errors;
    }

    // Find input and output layers
    const inputLayer = nodes.find((node) => node.type === "input");
    const outputLayer = nodes.find((node) => node.type === "output");

    // Check if input layer exists
    if (!inputLayer) {
      errors.push("Model must have an input layer");
    } else {
      // Check if input layer is connected
      const isInputConnected = edges.some(
        (edge) => edge.source === inputLayer.id
      );
      if (!isInputConnected) {
        errors.push(
          `Input layer ${inputLayer.id} must have at least one outgoing connection`
        );
      }
    }

    // Check if output layer exists
    if (!outputLayer) {
      errors.push("Model must have an output layer");
    } else {
      // Check if output layer is connected
      const isOutputConnected = edges.some(
        (edge) => edge.target === outputLayer.id
      );
      if (!isOutputConnected) {
        errors.push(
          `Output layer ${outputLayer.id} must have at least one incoming connection`
        );
      }
    }

    // Check if all nodes are connected
    const connectedNodeIds = new Set<string>();

    // Add source and target nodes from edges
    edges.forEach((edge) => {
      connectedNodeIds.add(edge.source);
      connectedNodeIds.add(edge.target);
    });

    // Find disconnected nodes
    const disconnectedNodes = nodes.filter(
      (node) => !connectedNodeIds.has(node.id)
    );
    if (disconnectedNodes.length > 0) {
      errors.push(
        `The following layers are disconnected: ${disconnectedNodes
          .map((node) => node.id)
          .join(", ")}`
      );
    }

    // Validate parameters for each layer type
    nodes.forEach((node) => {
      switch (node.type) {
        case "dense":
          if (!Number.isInteger(node.data.neurons) || node.data.neurons <= 0) {
            errors.push(
              `Dense layer ${node.id}: Number of neurons must be a positive integer`
            );
          }
          if (
            ![
              "None",
              "ReLU",
              "Sigmoid",
              "Softmax",
              "Tanh",
              "Leaky ReLU",
            ].includes(node.data.activation)
          ) {
            errors.push(`Dense layer ${node.id}: Invalid activation function`);
          }
          break;
        case "convolution":
          if (!Number.isInteger(node.data.filters) || node.data.filters <= 0) {
            errors.push(
              `Convolution layer ${node.id}: Number of filters must be a positive integer`
            );
          }
          if (
            !Array.isArray(node.data.kernelSize) ||
            node.data.kernelSize.length !== 2 ||
            node.data.kernelSize.some(
              (size: number) => !Number.isInteger(size) || size <= 0
            )
          ) {
            errors.push(
              `Convolution layer ${node.id}: Kernel size must be an array of two positive integers`
            );
          }
          if (
            !Array.isArray(node.data.stride) ||
            node.data.stride.length !== 2 ||
            node.data.stride.some(
              (size: number) => !Number.isInteger(size) || size <= 0
            )
          ) {
            errors.push(
              `Convolution layer ${node.id}: Stride must be an array of two positive integers`
            );
          }
          if (
            !["None", "ReLU", "Sigmoid", "Tanh", "Leaky ReLU"].includes(
              node.data.activation
            )
          ) {
            errors.push(
              `Convolution layer ${node.id}: Invalid activation function`
            );
          }
          break;
        case "maxpooling":
          if (
            !Array.isArray(node.data.poolSize) ||
            node.data.poolSize.length !== 2 ||
            node.data.poolSize.some(
              (size: number) => !Number.isInteger(size) || size <= 0
            )
          ) {
            errors.push(
              `MaxPooling layer ${node.id}: Pool size must be an array of two positive integers`
            );
          }
          if (
            !Array.isArray(node.data.stride) ||
            node.data.stride.length !== 2 ||
            node.data.stride.some(
              (size: number) => !Number.isInteger(size) || size <= 0
            )
          ) {
            errors.push(
              `MaxPooling layer ${node.id}: Stride must be an array of two positive integers`
            );
          }
          if (
            !["valid", "same", "none"].includes(
              node.data.padding?.toLowerCase()
            )
          ) {
            errors.push(
              `MaxPooling layer ${node.id}: Padding must be 'valid', 'same', or 'none'`
            );
          }
          break;
        case "dropout":
          if (
            typeof node.data.rate !== "number" ||
            node.data.rate < 0 ||
            node.data.rate >= 1
          ) {
            errors.push(
              `Dropout layer ${node.id}: Rate must be a number between 0 and 1`
            );
          }
          break;
        case "batchnormalization":
          if (
            typeof node.data.momentum !== "number" ||
            node.data.momentum <= 0 ||
            node.data.momentum >= 1
          ) {
            errors.push(
              `BatchNormalization layer ${node.id}: Momentum must be a number between 0 and 1`
            );
          }
          if (typeof node.data.epsilon !== "number" || node.data.epsilon <= 0) {
            errors.push(
              `BatchNormalization layer ${node.id}: Epsilon must be a positive number`
            );
          }
          break;
        case "attention":
          if (!Number.isInteger(node.data.heads) || node.data.heads <= 0) {
            errors.push(
              `Attention layer ${node.id}: Number of heads must be a positive integer`
            );
          }
          if (!Number.isInteger(node.data.keyDim) || node.data.keyDim <= 0) {
            errors.push(
              `Attention layer ${node.id}: Key dimension must be a positive integer`
            );
          }
          if (
            typeof node.data.dropout !== "number" ||
            node.data.dropout < 0 ||
            node.data.dropout >= 1
          ) {
            errors.push(
              `Attention layer ${node.id}: Dropout rate must be a number between 0 and 1`
            );
          }
          break;
        case "output":
          if (!["None", "Sigmoid", "Softmax"].includes(node.data.activation)) {
            errors.push(
              `Output layer ${node.id}: Invalid activation function. Must be None, Sigmoid, or Softmax`
            );
          }
          break;
      }
    });

    // Dataset-specific validations
    if (selectedDataset) {
      // MNIST and CIFAR-10 specific validations
      if (selectedDataset === "MNIST" || selectedDataset === "CIFAR-10") {
        // Check if there's a convolutional layer for image datasets
        const hasConvolutionLayer = nodes.some(
          (node) => node.type === "convolution"
        );

        if (!hasConvolutionLayer) {
          errors.push(
            `${selectedDataset} dataset works best with at least one convolutional layer`
          );
        }

        // Check if convolutional or maxpooling layers are followed by a flatten layer
        const convOrPoolNodes = nodes.filter(
          (node) => node.type === "convolution" || node.type === "maxpooling"
        );

        for (const node of convOrPoolNodes) {
          // Find all edges from this node
          const outgoingEdges = edges.filter((edge) => edge.source === node.id);

          // Check if any of these edges connect to a flatten layer
          const connectsToFlatten = outgoingEdges.some((edge) => {
            const targetNode = nodes.find((n) => n.id === edge.target);
            return targetNode && targetNode.type === "flatten";
          });

          if (!connectsToFlatten) {
            errors.push(
              `${node.type} layer ${node.id} should eventually connect to a Flatten layer`
            );
          }
        }

        // Output layer must use Softmax for classification
        if (outputLayer && outputLayer.data.activation !== "Softmax") {
          errors.push(
            `Output layer must use Softmax activation for ${selectedDataset} dataset`
          );
        }
      }

      // Iris dataset specific validations
      else if (selectedDataset === "Iris") {
        // Check for incompatible layers
        const incompatibleLayers = nodes.filter((node) =>
          ["convolution", "maxpooling", "attention"].includes(node.type || "")
        );

        if (incompatibleLayers.length > 0) {
          errors.push(
            `Iris dataset is not compatible with: ${incompatibleLayers
              .map((n) => n.type)
              .join(", ")} layers`
          );
        }

        // Output layer must use Softmax for multi-class classification
        if (outputLayer && outputLayer.data.activation !== "Softmax") {
          errors.push(
            `Output layer must use Softmax activation for Iris dataset`
          );
        }
      }

      // Breast Cancer dataset specific validations
      else if (selectedDataset === "Breast Cancer") {
        // Check for incompatible layers
        const incompatibleLayers = nodes.filter((node) =>
          ["convolution", "maxpooling", "attention"].includes(node.type || "")
        );

        if (incompatibleLayers.length > 0) {
          errors.push(
            `Breast Cancer dataset is not compatible with: ${incompatibleLayers
              .map((n) => n.type)
              .join(", ")} layers`
          );
        }

        // Output layer must use Sigmoid for binary classification
        if (outputLayer && outputLayer.data.activation !== "Sigmoid") {
          errors.push(
            `Output layer must use Sigmoid activation for Breast Cancer dataset`
          );
        }
      }

      // California Housing dataset specific validations
      else if (selectedDataset === "California Housing") {
        // Check for incompatible layers
        const incompatibleLayers = nodes.filter((node) =>
          ["convolution", "maxpooling", "attention"].includes(node.type || "")
        );

        if (incompatibleLayers.length > 0) {
          errors.push(
            `California Housing dataset is not compatible with: ${incompatibleLayers
              .map((n) => n.type)
              .join(", ")} layers`
          );
        }

        // Output layer should have no activation (linear) for regression
        if (outputLayer && outputLayer.data.activation !== "None") {
          errors.push(
            `Output layer should have no activation (linear) for California Housing dataset`
          );
        }
      }
    }

    return errors;
  };

  // Display validation errors
  const displayValidationErrors = (errors: ValidationErrors) => {
    setValidationErrors(errors);
    setShowValidationErrors(true);

    // Create a formatted error message for the alert
    const errorMessage =
      errors.length > 0
        ? `Validation failed with the following errors:\n\n${errors.join("\n")}`
        : "Validation failed!";

    // Show alert with all errors
    alert(errorMessage);

    // Don't automatically hide validation errors - let the user close them manually
    // This gives them time to read and fix the issues
  };

  // Render the content based on the active sidebar option
  const renderSidebarContent = () => {
    switch (activeSidebarOption) {
      case "layers":
        return (
          <div className="sidebar-content-section">
            <h3>Available Layers</h3>
            <div className="layer-list">
              {[
                "Dense",
                "Convolution",
                "MaxPooling",
                "Flatten",
                "Dropout",
                "BatchNormalization",
                "Attention",
              ].map((layerType) => (
                <div key={layerType} className="layer-item">
                  <span>{layerType}</span>
                  <button
                    className="add-button"
                    onClick={() => addLayer(layerType)}
                  >
                    +
                  </button>
                </div>
              ))}
            </div>
          </div>
        );
      case "layer_params":
        return (
          <div className="sidebar-content-section">
            <h3>Layer Parameters</h3>
            {!selectedNode ? (
              <div className="no-layer-selected">
                <p>
                  <i className="fas fa-info-circle"></i> Select a layer from the
                  canvas to view and edit its parameters.
                </p>
              </div>
            ) : (
              <>
                <div className="layer-params-list">
                  {nodes.map((node) => (
                    <div
                      key={node.id}
                      className={`layer-params-item ${
                        selectedNode?.id === node.id ? "selected" : ""
                      }`}
                      onClick={() => {
                        setSelectedNode(node);
                      }}
                    >
                      <div className="layer-params-header">
                        <span className="layer-params-type">{node.type}</span>
                      </div>
                      <div className="layer-params-details">
                        {node.type === "dense" && (
                          <>
                            <div className="param-item">
                              <span className="param-label">Neurons:</span>
                              <span className="param-value">
                                {node.data.neurons}
                              </span>
                            </div>
                            <div className="param-item">
                              <span className="param-label">Activation:</span>
                              <span className="param-value">
                                {node.data.activation}
                              </span>
                            </div>
                          </>
                        )}
                        {node.type === "convolution" && (
                          <>
                            <div className="param-item">
                              <span className="param-label">Filters:</span>
                              <span className="param-value">
                                {node.data.filters}
                              </span>
                            </div>
                            <div className="param-item">
                              <span className="param-label">Kernel:</span>
                              <span className="param-value">
                                {node.data.kernelSize?.join("×")}
                              </span>
                            </div>
                          </>
                        )}
                        {node.type === "maxpooling" && (
                          <>
                            <div className="param-item">
                              <span className="param-label">Pool Size:</span>
                              <span className="param-value">
                                {node.data.poolSize?.join("×")}
                              </span>
                            </div>
                            <div className="param-item">
                              <span className="param-label">Stride:</span>
                              <span className="param-value">
                                {node.data.stride?.join("×")}
                              </span>
                            </div>
                          </>
                        )}
                        {node.type === "dropout" && (
                          <div className="param-item">
                            <span className="param-label">Rate:</span>
                            <span className="param-value">
                              {node.data.rate}
                            </span>
                          </div>
                        )}
                        {node.type === "batchnormalization" && (
                          <>
                            <div className="param-item">
                              <span className="param-label">Momentum:</span>
                              <span className="param-value">
                                {node.data.momentum}
                              </span>
                            </div>
                            <div className="param-item">
                              <span className="param-label">Epsilon:</span>
                              <span className="param-value">
                                {node.data.epsilon}
                              </span>
                            </div>
                          </>
                        )}
                        {node.type === "attention" && (
                          <>
                            <div className="param-item">
                              <span className="param-label">Heads:</span>
                              <span className="param-value">
                                {node.data.heads}
                              </span>
                            </div>
                            <div className="param-item">
                              <span className="param-label">Key Dim:</span>
                              <span className="param-value">
                                {node.data.keyDim}
                              </span>
                            </div>
                          </>
                        )}
                        {node.type === "output" && (
                          <div className="param-item">
                            <span className="param-label">Activation:</span>
                            <span className="param-value">
                              {node.data.activation}
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
                {renderNodeParameters()}
              </>
            )}
          </div>
        );
      case "templates":
        return (
          <div className="sidebar-content-section">
            <h3>Model Templates</h3>
            <div className="template-list">
              {[
                "Simple Feedforward",
                "CNN",
                "Transformer",
                "Regression",
                "Blank",
              ].map((template) => (
                <div key={template} className="template-item">
                  <span>{template}</span>
                  <button
                    className="load-button"
                    onClick={() => loadTemplate(template)}
                  >
                    Load
                  </button>
                </div>
              ))}
            </div>
          </div>
        );
      case "hyperparameters":
        return (
          <div className="sidebar-content-section">
            <h3>Hyperparameters</h3>
            <div className="settings-list">
              <div className="setting-item">
                <label>Batch Size</label>
                <input
                  type="number"
                  min="1"
                  value={trainingConfig.batchSize}
                  onChange={(e) =>
                    setTrainingConfig({
                      ...trainingConfig,
                      batchSize: parseInt(e.target.value) || 32,
                    })
                  }
                />
              </div>
              <div className="setting-item">
                <label>Learning Rate</label>
                <input
                  type="number"
                  min="0.0001"
                  max="1"
                  step="0.0001"
                  value={trainingConfig.learningRate}
                  onChange={(e) =>
                    setTrainingConfig({
                      ...trainingConfig,
                      learningRate: parseFloat(e.target.value) || 0.001,
                    })
                  }
                />
              </div>
              <div className="setting-item">
                <label>Epochs</label>
                <input
                  type="number"
                  min="1"
                  value={trainingConfig.epochs}
                  onChange={(e) =>
                    setTrainingConfig({
                      ...trainingConfig,
                      epochs: parseInt(e.target.value) || 10,
                    })
                  }
                />
              </div>
            </div>
          </div>
        );
      case "training":
        return (
          <div className="sidebar-content-section">
            <h3>Training Options</h3>
            <div className="settings-list">
              <div className="setting-item">
                <label>Optimizer</label>
                <select
                  value={trainingConfig.optimizer}
                  onChange={(e) =>
                    setTrainingConfig({
                      ...trainingConfig,
                      optimizer: e.target.value,
                    })
                  }
                >
                  <option value="Adam">Adam</option>
                  <option value="SGD">SGD</option>
                  <option value="RMSprop">RMSprop</option>
                  <option value="Adagrad">Adagrad</option>
                  <option value="Adadelta">Adadelta</option>
                </select>
              </div>
              <div className="setting-item">
                <label>Loss Function</label>
                <select
                  value={trainingConfig.lossFunction}
                  onChange={(e) =>
                    setTrainingConfig({
                      ...trainingConfig,
                      lossFunction: e.target.value,
                    })
                  }
                >
                  <option value="Categorical Cross-Entropy">
                    Categorical Cross-Entropy
                  </option>
                  <option value="Binary Cross-Entropy">
                    Binary Cross-Entropy
                  </option>
                  <option value="Mean Squared Error">Mean Squared Error</option>
                  <option value="Mean Absolute Error">
                    Mean Absolute Error
                  </option>
                  <option value="Sparse Categorical Cross-Entropy">
                    Sparse Categorical Cross-Entropy
                  </option>
                </select>
              </div>
              <div className="setting-item">
                <label>Validation Split</label>
                <input
                  type="number"
                  min="0"
                  max="0.5"
                  step="0.05"
                  value={trainingConfig.validationSplit || 0.2}
                  onChange={(e) =>
                    setTrainingConfig({
                      ...trainingConfig,
                      validationSplit: parseFloat(e.target.value) || 0.2,
                    })
                  }
                />
              </div>
            </div>

            {/* Training Status Section - Keep in left panel */}
            <div className="training-status mt-4">
              <h3>Training Status</h3>
              <div className="status-item">
                <span className="status-label">Status:</span>
                <span
                  className={`status-value ${isTraining ? "training" : ""}`}
                >
                  {isTraining ? "Training..." : "Not Training"}
                </span>
              </div>

              {/* Always show the progress bar, but keep it empty when not training */}
              <div className="progress-bar-container">
                <div
                  className="progress-bar"
                  style={{
                    width: isTraining
                      ? `${
                          (trainingProgress.currentEpoch /
                            trainingProgress.totalEpochs) *
                          100
                        }%`
                      : "0%",
                  }}
                ></div>
              </div>

              {/* Always show metrics */}
              <div className="status-item">
                <span className="status-label">Epoch:</span>
                <span className="status-value">
                  {trainingProgress.currentEpoch > 0
                    ? `${trainingProgress.currentEpoch}/${trainingProgress.totalEpochs}`
                    : "N/A"}
                </span>
              </div>
              <div className="status-item">
                <span className="status-label">Loss:</span>
                <span className="status-value">
                  {trainingProgress.loss > 0
                    ? trainingProgress.loss.toFixed(4)
                    : "N/A"}
                </span>
              </div>
              <div className="status-item">
                <span className="status-label">Accuracy:</span>
                <span className="status-value">
                  {trainingProgress.accuracy > 0
                    ? trainingProgress.accuracy.toFixed(4)
                    : "N/A"}
                </span>
              </div>

              {/* Always show validation metrics if they exist */}
              <div className="status-item">
                <span className="status-label">Val Loss:</span>
                <span className="status-value">
                  {trainingProgress.valLoss > 0
                    ? trainingProgress.valLoss.toFixed(4)
                    : "N/A"}
                </span>
              </div>
              <div className="status-item">
                <span className="status-label">Val Accuracy:</span>
                <span className="status-value">
                  {trainingProgress.valAccuracy > 0
                    ? trainingProgress.valAccuracy.toFixed(4)
                    : "N/A"}
                </span>
              </div>

              {/* California Housing specific metrics */}
              {selectedDataset === "California Housing" && (
                <>
                  <div className="status-item">
                    <span className="status-label">RMSE:</span>
                    <span className="status-value">
                      {rmse !== null ? rmse.toFixed(4) : "N/A"}
                    </span>
                  </div>
                  <div className="status-item">
                    <span className="status-label">R² Score:</span>
                    <span className="status-value">
                      {r2 !== null ? r2.toFixed(4) : "N/A"}
                    </span>
                  </div>
                </>
              )}

              {/* Add training controls when not training */}
              {!isTraining && <div className="training-controls mt-3"></div>}

              {/* Add stop button when training */}
              {isTraining && (
                <div className="training-controls mt-3">
                  <button
                    className="stop-button-sidebar"
                    onClick={handleStopTraining}
                  >
                    <i className="fas fa-stop-circle"></i> Stop Training
                  </button>
                </div>
              )}
            </div>
          </div>
        );
      case "settings":
        return (
          <div className="sidebar-content-section">
            <h3>Dataset</h3>
            <div className="settings-list">
              <div className="setting-item">
                <label>Dataset</label>
                <select
                  value={selectedDataset}
                  onChange={(e) => setSelectedDataset(e.target.value)}
                >
                  <option value="">Select Dataset</option>
                  <option value="MNIST">MNIST</option>
                  <option value="CIFAR-10">CIFAR-10</option>
                  <option value="Iris">Iris</option>
                  <option value="Breast Cancer">Breast Cancer</option>
                  <option value="California Housing">California Housing</option>
                </select>
              </div>
              {/* More settings will be added later */}
            </div>
          </div>
        );
      case "model_config":
        return (
          <div className="sidebar-content-section">
            <h3>Model Configuration</h3>
            <div className="info-item">
              <span className="info-label">Dataset:</span>
              <span className="info-value">{selectedDataset || "None"}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Optimizer:</span>
              <span className="info-value">{trainingConfig.optimizer}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Loss Function:</span>
              <span className="info-value">{trainingConfig.lossFunction}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Batch Size:</span>
              <span className="info-value">{trainingConfig.batchSize}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Learning Rate:</span>
              <span className="info-value">{trainingConfig.learningRate}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Epochs:</span>
              <span className="info-value">{trainingConfig.epochs}</span>
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  // Render the node parameters panel
  const renderNodeParameters = () => {
    if (!selectedNode) return null;

    return (
      <div className="node-parameters">
        <h3>Layer Parameters</h3>
        <div className="parameter-list">
          {/* Common parameters for all nodes */}
          <label>Layer Name:</label>
          <div className="name-input-container">
            <input
              type="text"
              value={selectedNode.data.label || ""}
              onChange={(e) => updateParameter("label", e.target.value)}
              placeholder="Enter custom layer name"
              className="layer-name-input"
            />
            <small className="name-hint">
              This name will appear on the layer in the canvas
            </small>
          </div>

          {/* Dense Layer */}
          {selectedNode.type === "dense" && (
            <>
              <label>Neurons:</label>
              <input
                type="number"
                min="1"
                value={selectedNode.data.neurons || ""}
                onChange={(e) => updateParameter("neurons", +e.target.value)}
              />

              <label>Activation:</label>
              <select
                value={selectedNode.data.activation || "None"}
                onChange={(e) => updateParameter("activation", e.target.value)}
              >
                <option value="None">None</option>
                <option value="ReLU">ReLU</option>
                <option value="Sigmoid">Sigmoid</option>
                <option value="Tanh">Tanh</option>
                <option value="Softmax">Softmax</option>
                <option value="Leaky ReLU">Leaky ReLU</option>
              </select>
            </>
          )}

          {/* Dropout Layer */}
          {selectedNode.type === "dropout" && (
            <>
              <label>Rate:</label>
              <input
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={selectedNode.data.rate || 0.2}
                onChange={(e) =>
                  updateParameter("rate", parseFloat(e.target.value) || 0)
                }
              />
            </>
          )}

          {/* Convolutional Layer */}
          {selectedNode.type === "convolution" && (
            <>
              <label>Filters:</label>
              <input
                type="number"
                min="1"
                value={selectedNode.data.filters || ""}
                onChange={(e) => updateParameter("filters", +e.target.value)}
              />

              <label>Kernel Size:</label>
              <div className="input-group">
                <input
                  type="number"
                  min="1"
                  value={selectedNode.data.kernelSize?.[0] || ""}
                  onChange={(e) =>
                    updateParameter("kernelSize", [
                      parseInt(e.target.value) || 1,
                      selectedNode.data.kernelSize?.[1] || 1,
                    ])
                  }
                  style={{ width: "60px" }}
                />
                <span>x</span>
                <input
                  type="number"
                  min="1"
                  value={selectedNode.data.kernelSize?.[1] || ""}
                  onChange={(e) =>
                    updateParameter("kernelSize", [
                      selectedNode.data.kernelSize?.[0] || 1,
                      parseInt(e.target.value) || 1,
                    ])
                  }
                  style={{ width: "60px" }}
                />
              </div>

              <label>Stride:</label>
              <div className="input-group">
                <input
                  type="number"
                  min="1"
                  value={selectedNode.data.stride?.[0] || ""}
                  onChange={(e) =>
                    updateParameter("stride", [
                      parseInt(e.target.value) || 1,
                      selectedNode.data.stride?.[1] || 1,
                    ])
                  }
                  style={{ width: "60px" }}
                />
                <span>x</span>
                <input
                  type="number"
                  min="1"
                  value={selectedNode.data.stride?.[1] || ""}
                  onChange={(e) =>
                    updateParameter("stride", [
                      selectedNode.data.stride?.[0] || 1,
                      parseInt(e.target.value) || 1,
                    ])
                  }
                  style={{ width: "60px" }}
                />
              </div>

              <label>Activation:</label>
              <select
                value={selectedNode.data.activation || "None"}
                onChange={(e) => updateParameter("activation", e.target.value)}
              >
                <option value="None">None</option>
                <option value="ReLU">ReLU</option>
                <option value="Sigmoid">Sigmoid</option>
                <option value="Tanh">Tanh</option>
                <option value="Leaky ReLU">Leaky ReLU</option>
              </select>
            </>
          )}

          {/* MaxPooling Layer */}
          {selectedNode.type === "maxpooling" && (
            <>
              <label>Pool Size:</label>
              <div className="input-group">
                <input
                  type="number"
                  min="1"
                  value={selectedNode.data.poolSize?.[0] || ""}
                  onChange={(e) =>
                    updateParameter("poolSize", [
                      parseInt(e.target.value) || 1,
                      selectedNode.data.poolSize?.[1] || 1,
                    ])
                  }
                  style={{ width: "60px" }}
                />
                <span>x</span>
                <input
                  type="number"
                  min="1"
                  value={selectedNode.data.poolSize?.[1] || ""}
                  onChange={(e) =>
                    updateParameter("poolSize", [
                      selectedNode.data.poolSize?.[0] || 1,
                      parseInt(e.target.value) || 1,
                    ])
                  }
                  style={{ width: "60px" }}
                />
              </div>

              <label>Stride:</label>
              <div className="input-group">
                <input
                  type="number"
                  min="1"
                  value={selectedNode.data.stride?.[0] || ""}
                  onChange={(e) =>
                    updateParameter("stride", [
                      parseInt(e.target.value) || 1,
                      selectedNode.data.stride?.[1] || 1,
                    ])
                  }
                  style={{ width: "60px" }}
                />
                <span>x</span>
                <input
                  type="number"
                  min="1"
                  value={selectedNode.data.stride?.[1] || ""}
                  onChange={(e) =>
                    updateParameter("stride", [
                      selectedNode.data.stride?.[0] || 1,
                      parseInt(e.target.value) || 1,
                    ])
                  }
                  style={{ width: "60px" }}
                />
              </div>

              <label>Padding:</label>
              <select
                value={selectedNode.data.padding || "none"}
                onChange={(e) => updateParameter("padding", e.target.value)}
              >
                <option value="none">None</option>
                <option value="valid">Valid</option>
                <option value="same">Same</option>
              </select>
            </>
          )}

          {/* Batch Normalization Layer */}
          {selectedNode.type === "batchnormalization" && (
            <>
              <label>Momentum:</label>
              <input
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={selectedNode.data.momentum || 0.99}
                onChange={(e) =>
                  updateParameter("momentum", parseFloat(e.target.value))
                }
              />

              <label>Epsilon:</label>
              <input
                type="number"
                min="0.00001"
                step="0.00001"
                value={selectedNode.data.epsilon || 0.001}
                onChange={(e) =>
                  updateParameter("epsilon", parseFloat(e.target.value))
                }
              />
            </>
          )}

          {/* Attention Layer */}
          {selectedNode.type === "attention" && (
            <>
              <label>Number of Heads:</label>
              <input
                type="number"
                min="1"
                value={selectedNode.data.heads || 8}
                onChange={(e) =>
                  updateParameter("heads", parseInt(e.target.value))
                }
              />

              <label>Key Dimension:</label>
              <input
                type="number"
                min="1"
                value={selectedNode.data.keyDim || 64}
                onChange={(e) =>
                  updateParameter("keyDim", parseInt(e.target.value))
                }
              />

              <label>Dropout Rate:</label>
              <input
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={selectedNode.data.dropout || 0.0}
                onChange={(e) =>
                  updateParameter("dropout", parseFloat(e.target.value))
                }
              />
            </>
          )}

          {/* Output Layer */}
          {selectedNode.type === "output" && (
            <>
              <label>Activation:</label>
              <select
                value={selectedNode.data.activation || "None"}
                onChange={(e) => updateParameter("activation", e.target.value)}
              >
                <option value="None">None</option>
                <option value="Sigmoid">Sigmoid</option>
                <option value="Softmax">Softmax</option>
              </select>
            </>
          )}
        </div>
      </div>
    );
  };

  // Render visualization based on selected option
  const renderVisualization = () => {
    if (!selectedDataset) {
      return (
        <div className="visualization-placeholder">
          <p>Select a dataset to view visualizations</p>
        </div>
      );
    }

    switch (selectedVisualization) {
      case "accuracy":
        return (
          <div className="visualization-container">
            <h4>Accuracy Over Time</h4>
            <div className="chart-container">
              {isTraining || labels.length > 0 ? (
                <Line
                  key={`accuracy-chart-${chartKey}`}
                  data={accuracyChartData}
                  options={chartOptions}
                />
              ) : (
                <div className="empty-chart">
                  <p>Start training to see accuracy metrics</p>
                </div>
              )}
            </div>
          </div>
        );
      case "loss":
        return (
          <div className="visualization-container">
            <h4>Loss Over Time</h4>
            <div className="chart-container">
              {isTraining || labels.length > 0 ? (
                <Line
                  key={`loss-chart-${chartKey}`}
                  data={lossChartData}
                  options={chartOptions}
                />
              ) : (
                <div className="empty-chart">
                  <p>Start training to see loss metrics</p>
                </div>
              )}
            </div>
          </div>
        );
      case "confusion_matrix":
        return (
          <div className="visualization-container">
            <h4>Confusion Matrix</h4>
            {confusionMatrix.length > 0 ? (
              <>
                <p className="matrix-info">
                  Showing confusion matrix for {selectedDataset} dataset
                </p>
                <ConfusionMatrix
                  matrix={confusionMatrix}
                  labels={getClassLabels(selectedDataset)}
                />
              </>
            ) : (
              <div className="placeholder-visualization">
                <p>
                  Confusion matrix visualization will be shown here after
                  training
                </p>
                <div className="matrix-placeholder"></div>
              </div>
            )}
          </div>
        );
      case "heatmap":
        return (
          <div className="visualization-container">
            <h4>Multicollinearity Heatmap</h4>
            <div
              className="chart-container"
              style={{ height: "auto", minHeight: "280px" }}
            >
              {heatmapImage ? (
                <div className="heatmap-container">
                  <img
                    src={heatmapImage}
                    alt="Multicollinearity Heatmap"
                    style={{
                      width: "100%",
                      maxWidth: "800px",
                      margin: "0 auto",
                      display: "block",
                    }}
                  />
                  {rmse !== null && r2 !== null && (
                    <div className="regression-metrics">
                      <div className="metric-item">
                        <span className="metric-label">RMSE:</span>
                        <span className="metric-value">{rmse.toFixed(4)}</span>
                      </div>
                      <div className="metric-item">
                        <span className="metric-label">R² Score:</span>
                        <span className="metric-value">{r2.toFixed(4)}</span>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="empty-chart">
                  <p>Train the model to generate a multicollinearity heatmap</p>
                </div>
              )}
            </div>
          </div>
        );
      case "class_distribution":
        return (
          <div className="visualization-container">
            <h4>Class Distribution</h4>
            <div className="placeholder-visualization">
              <p>Class distribution visualization will be shown here</p>
              <div className="distribution-placeholder"></div>
            </div>
          </div>
        );
      case "feature_importance":
        return (
          <div className="visualization-container">
            <h4>Feature Importance</h4>
            <div className="placeholder-visualization">
              <p>
                Feature importance visualization will be shown here after
                training
              </p>
              <div className="feature-importance-placeholder"></div>
            </div>
          </div>
        );
      case "scatter_plot":
        return (
          <div className="visualization-container">
            <h4>Scatter Plot</h4>
            <div className="placeholder-visualization">
              <p>Scatter plot visualization will be shown here</p>
              <div className="scatter-placeholder"></div>
            </div>
          </div>
        );
      case "pca":
        return (
          <div className="visualization-container">
            <h4>PCA Visualization</h4>
            <div className="placeholder-visualization">
              <p>PCA visualization will be shown here after training</p>
              <div className="pca-placeholder"></div>
            </div>
          </div>
        );
      case "roc_curve":
        return (
          <div className="visualization-container">
            <h4>ROC Curve</h4>
            <div className="placeholder-visualization">
              <p>ROC curve will be shown here after training</p>
              <div className="roc-placeholder"></div>
            </div>
          </div>
        );
      case "residual_plot":
        return (
          <div className="visualization-container">
            <h4>Residual Plot</h4>
            <div className="placeholder-visualization">
              <p>Residual plot will be shown here after training</p>
              <div className="residual-placeholder"></div>
            </div>
          </div>
        );
      case "feature_maps":
        return (
          <div className="visualization-container">
            <h4>Feature Maps</h4>
            <div className="placeholder-visualization">
              <p>Feature maps will be shown here after training</p>
              <div className="feature-maps-placeholder"></div>
            </div>
          </div>
        );
      default:
        return (
          <div className="visualization-placeholder">
            <p>Select a visualization type</p>
          </div>
        );
    }
  };

  // Confusion Matrix Component
  const ConfusionMatrix = ({
    matrix,
    labels,
  }: {
    matrix: number[][];
    labels: string[];
  }) => {
    const maxValue = Math.max(...matrix.flat());

    return (
      <div className="confusion-matrix-container">
        <table className="confusion-matrix-table">
          <thead>
            <tr>
              <th style={{ width: "40px" }}></th>
              {labels.map((label, index) => (
                <th
                  key={`header-${index}`}
                  className="matrix-header"
                  style={{ width: `${100 / (labels.length + 1)}%` }}
                >
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, rowIndex) => (
              <tr key={`row-${rowIndex}`}>
                <td className="matrix-header">{labels[rowIndex]}</td>
                {row.map((value, colIndex) => {
                  // Calculate opacity based on value
                  const opacity = maxValue > 0 ? value / maxValue : 0;
                  return (
                    <td
                      key={`cell-${rowIndex}-${colIndex}`}
                      className="matrix-cell"
                      style={{
                        backgroundColor: `rgba(76, 175, 80, ${opacity})`,
                        color: opacity > 0.5 ? "#fff" : "#000",
                      }}
                    >
                      {value}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  // Add the handleExport function before the return statement
  const handleExport = async (format: string) => {
    if (!selectedDataset) {
      alert("⚠️ Please select a dataset before exporting.");
      return;
    }

    // Check if model is saved
    if (!isModelSaved) {
      const saveFirst = window.confirm(
        "The model needs to be saved before exporting. Save now?"
      );
      if (saveFirst) {
        await handleSaveModel();
        if (!isModelSaved) return; // If saving failed, stop export
      } else {
        return; // User chose not to save, stop export
      }
    }

    setExportStatusMessage(`Exporting model as ${format.toUpperCase()}...`);

    try {
      const response = await axios.get(
        `http://127.0.0.1:5000/export/${format}`,
        {
          responseType: "blob", // Important for file download
        }
      );

      // Set correct file extension for each format
      let fileExtension;
      if (format === "savedmodel") {
        fileExtension = "zip";
      } else if (format === "pytorch") {
        fileExtension = "py"; // PyTorch model extension
      } else if (format === "py") {
        fileExtension = "py"; // Python script extension
      } else if (format === "ipynb") {
        fileExtension = "ipynb"; // Jupyter notebook extension
      } else {
        fileExtension = format;
      }

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", `trained_model.${fileExtension}`);
      document.body.appendChild(link);
      link.click();
      link.remove();

      setExportStatusMessage(
        `Model exported successfully as ${format.toUpperCase()}!`
      );

      // Clear status message after 5 seconds
      setTimeout(() => {
        setExportStatusMessage("");
      }, 5000);
    } catch (error) {
      console.error("Export failed:", error);
      setExportStatusMessage(
        `Failed to export model as ${format.toUpperCase()}.`
      );

      // Clear error message after 5 seconds
      setTimeout(() => {
        setExportStatusMessage("");
      }, 5000);
    }
  };

  return (
    <div className="new-build-page">
      <div className="new-build-container">
        {/* Validation Error Messages */}
        {showValidationErrors && validationErrors.length > 0 && (
          <div className="validation-errors-container">
            <div className="validation-errors-header">
              <i className="fas fa-exclamation-triangle"></i>
              <span>Validation Errors</span>
              <button
                className="close-errors-btn"
                onClick={() => setShowValidationErrors(false)}
              >
                <i className="fas fa-times"></i>
              </button>
            </div>
            <ul className="validation-errors-list">
              {validationErrors.map((error, index) => (
                <li key={index} className="validation-error-item">
                  <i className="fas fa-times-circle"></i>
                  <span>{error}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="left-panel">
          <div className="sidebar-nav">
            <div
              className={`sidebar-nav-item ${
                activeSidebarOption === "layers" ? "active" : ""
              }`}
              onClick={() => setActiveSidebarOption("layers")}
            >
              <i className="fas fa-layer-group"></i>
              <span>Layers</span>
            </div>
            <div
              className={`sidebar-nav-item ${
                activeSidebarOption === "templates" ? "active" : ""
              }`}
              onClick={() => setActiveSidebarOption("templates")}
            >
              <i className="fas fa-file-code"></i>
              <span>Templates</span>
            </div>
            <div
              className={`sidebar-nav-item ${
                activeSidebarOption === "layer_params" ? "active" : ""
              }`}
              onClick={() => setActiveSidebarOption("layer_params")}
            >
              <i className="fas fa-sliders-h"></i>
              <span>Layer Params</span>
            </div>
            <div
              className={`sidebar-nav-item ${
                activeSidebarOption === "hyperparameters" ? "active" : ""
              }`}
              onClick={() => setActiveSidebarOption("hyperparameters")}
            >
              <i className="fas fa-cogs"></i>
              <span>Hyperparams</span>
            </div>
            <div
              className={`sidebar-nav-item ${
                activeSidebarOption === "training" ? "active" : ""
              }`}
              onClick={() => setActiveSidebarOption("training")}
            >
              <i className="fas fa-play"></i>
              <span>Training</span>
            </div>
            <div
              className={`sidebar-nav-item ${
                activeSidebarOption === "model_config" ? "active" : ""
              }`}
              onClick={() => setActiveSidebarOption("model_config")}
            >
              <i className="fas fa-project-diagram"></i>
              <span>Model Config</span>
            </div>
            <div
              className={`sidebar-nav-item ${
                activeSidebarOption === "settings" ? "active" : ""
              }`}
              onClick={() => setActiveSidebarOption("settings")}
            >
              <i className="fas fa-cog"></i>
              <span>Settings</span>
            </div>
          </div>
          <div className="sidebar-content">{renderSidebarContent()}</div>
        </div>

        <div className="center-panel">
          <div className="canvas-container">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onNodeClick={onNodeClick}
              nodeTypes={nodeTypes}
              fitView
              style={{ width: "100%", height: "100%" }}
            >
              <Background />
              <Controls />
            </ReactFlow>
          </div>
        </div>

        <div className="right-panel">
          <div className="training-container">
            <div className="training-info">
              {/* Any training info that should remain in the right panel */}
            </div>

            {/* Add save and train buttons next to each other */}
            <div className="action-buttons-container">
              <button
                className="save-button"
                onClick={handleSaveModel}
                disabled={isTraining}
              >
                <i className="fas fa-save"></i> Save Model
              </button>
              <button
                className="train-button"
                disabled={!selectedDataset || isTraining}
                onClick={handleStartTraining}
              >
                {isTraining ? (
                  <>
                    <span className="spinner"></span> Training...
                  </>
                ) : (
                  "Train Model"
                )}
              </button>
            </div>

            {/* Stop training button (only shown when training) */}
            {isTraining && (
              <button className="stop-button" onClick={handleStopTraining}>
                <i className="fas fa-stop-circle"></i> Stop Training
              </button>
            )}
          </div>

          <div className="visualization-section">
            <h3>Visualizations</h3>
            <div className="visualization-controls">
              <label htmlFor="visualization-select">
                Select Visualization:
              </label>
              <select
                id="visualization-select"
                value={selectedVisualization}
                onChange={(e) => setSelectedVisualization(e.target.value)}
                disabled={!selectedDataset}
              >
                {getVisualizationOptions().map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
            {renderVisualization()}
          </div>

          <div className="export-container">
            <h3>Export Model</h3>
            <div className="export-options">
              <button
                className="export-button python"
                disabled={isTraining}
                onClick={() => handleExport("py")}
                title="Export model as a Python script with TensorFlow implementation"
              >
                <i className="fab fa-python"></i> Python
              </button>
              <button
                className="export-button keras"
                disabled={isTraining}
                onClick={() => handleExport("keras")}
                title="Export as Keras .h5 model file"
              >
                <i className="fas fa-cube"></i> Keras
              </button>
              <button
                className="export-button pytorch"
                disabled={isTraining}
                onClick={() => handleExport("pytorch")}
                title="Export model converted to PyTorch format"
              >
                <i className="fas fa-fire"></i> PyTorch
              </button>
              <button
                className="export-button savedmodel"
                disabled={isTraining}
                onClick={() => handleExport("savedmodel")}
                title="Export as TensorFlow SavedModel format"
              >
                <i className="fas fa-save"></i> SavedModel
              </button>
              <button
                className="export-button notebook"
                disabled={isTraining}
                onClick={() => handleExport("ipynb")}
                title="Export as Jupyter Notebook with model code and visualization"
              >
                <i className="fas fa-book"></i> Notebook
              </button>
            </div>

            {exportStatusMessage && (
              <div
                className={`export-status-message ${
                  exportStatusMessage.includes("Failed") ? "error" : "success"
                }`}
                role="alert"
              >
                {exportStatusMessage}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default NewBuildPage;
