import React, { createContext, useContext, useState } from "react";
import { Node, Edge } from "reactflow";

// Define the structure of data for each Node
interface NodeData {
  label: string;
  [key: string]: any; // Allows for additional dynamic properties
}

// Define the structure of validation errors
type ValidationErrors = string[];

// Context Type Definition
interface NewBuildPageContextType {
  nodes: Node<NodeData>[]; // Nodes with custom data type
  setNodes: React.Dispatch<React.SetStateAction<Node<NodeData>[]>>;
  edges: Edge[];
  setEdges: React.Dispatch<React.SetStateAction<Edge[]>>;
  validationErrors: ValidationErrors;
  setValidationErrors: React.Dispatch<React.SetStateAction<ValidationErrors>>;
  selectedDataset: string;
  setSelectedDataset: React.Dispatch<React.SetStateAction<string>>;
  trainingConfig: TrainingConfig;
  setTrainingConfig: React.Dispatch<React.SetStateAction<TrainingConfig>>;
  isTraining: boolean;
  setIsTraining: React.Dispatch<React.SetStateAction<boolean>>;
  trainingProgress: TrainingProgress;
  setTrainingProgress: React.Dispatch<React.SetStateAction<TrainingProgress>>;
}

// Training configuration interface
interface TrainingConfig {
  epochs: number;
  batchSize: number;
  optimizer: string;
  lossFunction: string;
  learningRate: number;
  validationSplit: number;
}

// Training progress interface
interface TrainingProgress {
  currentEpoch: number;
  totalEpochs: number;
  loss: number;
  accuracy: number;
  valLoss: number;
  valAccuracy: number;
}

// Default blank template
const blankTemplateNodes: Node<NodeData>[] = [
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
];
const blankTemplateEdges: Edge[] = [];

// Default training configuration
const defaultTrainingConfig: TrainingConfig = {
  epochs: 10,
  batchSize: 32,
  optimizer: "Adam",
  lossFunction: "Categorical Cross-Entropy",
  learningRate: 0.001,
  validationSplit: 0.2,
};

// Default training progress
const defaultTrainingProgress: TrainingProgress = {
  currentEpoch: 0,
  totalEpochs: 0,
  loss: 0,
  accuracy: 0,
  valLoss: 0,
  valAccuracy: 0,
};

// Create the context
const NewBuildPageContext = createContext<NewBuildPageContextType | undefined>(
  undefined
);

// Provider component
export const NewBuildPageProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [nodes, setNodes] = useState<Node<NodeData>[]>(blankTemplateNodes);
  const [edges, setEdges] = useState<Edge[]>(blankTemplateEdges);
  const [validationErrors, setValidationErrors] = useState<ValidationErrors>(
    []
  );
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig>(
    defaultTrainingConfig
  );
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainingProgress, setTrainingProgress] = useState<TrainingProgress>(
    defaultTrainingProgress
  );

  const value = {
    nodes,
    setNodes,
    edges,
    setEdges,
    validationErrors,
    setValidationErrors,
    selectedDataset,
    setSelectedDataset,
    trainingConfig,
    setTrainingConfig,
    isTraining,
    setIsTraining,
    trainingProgress,
    setTrainingProgress,
  };

  return (
    <NewBuildPageContext.Provider value={value}>
      {children}
    </NewBuildPageContext.Provider>
  );
};

// Custom hook to use the context
export const useNewBuildPageContext = () => {
  const context = useContext(NewBuildPageContext);
  if (context === undefined) {
    throw new Error(
      "useNewBuildPageContext must be used within a NewBuildPageProvider"
    );
  }
  return context;
};
