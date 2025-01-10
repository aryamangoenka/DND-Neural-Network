import React, { createContext, useContext, useState } from "react";

interface LiveMetrics {
  epoch: number;
  loss: number;
  accuracy: number;
  val_loss: number;
  val_accuracy: number;
}

interface TrainPageContextProps {
  lossFunction: string;
  setLossFunction: React.Dispatch<React.SetStateAction<string>>;
  optimizer: string;
  setOptimizer: React.Dispatch<React.SetStateAction<string>>;
  learningRate: number;
  setLearningRate: React.Dispatch<React.SetStateAction<number>>;
  batchSize: number;
  setBatchSize: React.Dispatch<React.SetStateAction<number>>;
  epochs: number;
  setEpochs: React.Dispatch<React.SetStateAction<number>>;
  isTraining: boolean;
  setIsTraining: React.Dispatch<React.SetStateAction<boolean>>;
  progress: string;
  setProgress: React.Dispatch<React.SetStateAction<string>>;
  liveMetrics: LiveMetrics;
  setLiveMetrics: React.Dispatch<React.SetStateAction<LiveMetrics>>;
}

const TrainPageContext = createContext<TrainPageContextProps | undefined>(
  undefined
);

export const TrainPageProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [lossFunction, setLossFunction] = useState("Categorical Cross-Entropy");
  const [optimizer, setOptimizer] = useState("Adam");
  const [learningRate, setLearningRate] = useState<number>(0.001);
  const [batchSize, setBatchSize] = useState<number>(32);
  const [epochs, setEpochs] = useState<number>(10);
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState<string>("");
  const [liveMetrics, setLiveMetrics] = useState<LiveMetrics>({
    epoch: 0,
    loss: 0,
    accuracy: 0,
    val_loss: 0,
    val_accuracy: 0,
  });

  return (
    <TrainPageContext.Provider
      value={{
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
      }}
    >
      {children}
    </TrainPageContext.Provider>
  );
};

export const useTrainPageContext = (): TrainPageContextProps => {
  const context = useContext(TrainPageContext);
  if (!context) {
    throw new Error(
      "useTrainPageContext must be used within a TrainPageProvider"
    );
  }
  return context;
};
