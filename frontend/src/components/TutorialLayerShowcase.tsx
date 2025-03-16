import React from "react";
import "../styles/components/TutorialLayerShowcase.scss";

interface LayerInfo {
  type: string;
  name: string;
  description: string;
  color: string;
  properties: { label: string; value: string }[];
}

const TutorialLayerShowcase: React.FC = () => {
  // Sample data for each layer type
  const layers: LayerInfo[] = [
    {
      type: "input",
      name: "Input Layer",
      description:
        "The entry point for data into your neural network. Defines the shape of your input data.",
      color: "#00BCD4", // Cyan
      properties: [],
    },
    {
      type: "dense",
      name: "Dense Layer",
      description:
        "Fully connected layer where each neuron is connected to every neuron in the previous layer. Used for learning complex patterns.",
      color: "#4CAF50", // Green
      properties: [
        { label: "Neurons", value: "128" },
        { label: "Activation", value: "relu" },
      ],
    },
    {
      type: "convolution",
      name: "Convolution Layer",
      description:
        "Applies filters to detect spatial patterns in input data. Essential for image processing tasks.",
      color: "#2196F3", // Blue
      properties: [
        { label: "Filters", value: "32" },
        { label: "Kernel Size", value: "3x3" },
        { label: "Stride", value: "1x1" },
        { label: "Activation", value: "relu" },
      ],
    },
    {
      type: "maxpooling",
      name: "MaxPooling Layer",
      description:
        "Reduces the spatial dimensions of the data by taking the maximum value in each window. Helps with feature extraction and reduces computation.",
      color: "#9C27B0", // Purple
      properties: [
        { label: "Pool Size", value: "2x2" },
        { label: "Stride", value: "2x2" },
        { label: "Padding", value: "valid" },
      ],
    },
    {
      type: "flatten",
      name: "Flatten Layer",
      description:
        "Converts multi-dimensional input into a 1D vector. Often used between convolutional and dense layers.",
      color: "#FF9800", // Orange
      properties: [],
    },
    {
      type: "dropout",
      name: "Dropout Layer",
      description:
        "Randomly sets a fraction of inputs to zero during training to prevent overfitting.",
      color: "#F44336", // Red
      properties: [{ label: "Rate", value: "0.25" }],
    },
    {
      type: "batchnormalization",
      name: "Batch Normalization Layer",
      description:
        "Normalizes the activations of the previous layer for each batch. Accelerates training and adds stability.",
      color: "#188cc6", // Gray-Blue
      properties: [
        { label: "Momentum", value: "0.99" },
        { label: "Epsilon", value: "0.001" },
      ],
    },
    {
      type: "attention",
      name: "Attention Layer",
      description:
        "Allows the model to focus on relevant parts of the input. Important for sequence data and natural language processing.",
      color: "#673AB7", // Deep Purple
      properties: [
        { label: "Heads", value: "8" },
        { label: "Key Dim", value: "64" },
        { label: "Dropout", value: "0.1" },
      ],
    },
    {
      type: "output",
      name: "Output Layer",
      description:
        "The final layer that produces the model's predictions. Activation function depends on the task (e.g., softmax for classification).",
      color: "#FF5722", // Deep Orange
      properties: [{ label: "Activation", value: "softmax" }],
    },
  ];

  // Custom node component that mimics the look of the actual nodes
  const LayerNode: React.FC<{
    name: string;
    color: string;
    properties: { label: string; value: string }[];
  }> = ({ name, color, properties }) => {
    return (
      <div
        className="tutorial-node"
        style={{
          backgroundColor: color,
          borderColor: adjustColor(color, -20),
        }}
      >
        <h4>{name}</h4>
        {properties.map((prop, index) => (
          <p key={index}>
            {prop.label}: {prop.value}
          </p>
        ))}
      </div>
    );
  };

  // Helper function to darken/lighten a color
  const adjustColor = (color: string, amount: number): string => {
    return color;
  };

  return (
    <div className="tutorial-layer-showcase">
      <h3>Available Layer Types</h3>
      <div className="layer-showcase-grid">
        {layers.map((layer, index) => (
          <div key={index} className="layer-showcase-item">
            <div className="layer-component">
              <LayerNode
                name={layer.name}
                color={layer.color}
                properties={layer.properties}
              />
            </div>
            <div className="layer-info">
              <h4>{layer.name}</h4>
              <p>{layer.description}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default TutorialLayerShowcase;
