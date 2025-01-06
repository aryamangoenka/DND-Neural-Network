import React, { createContext, useState, useContext, ReactNode } from "react";

interface TrainingContextType {
  isTrainingAllowed: boolean;
  setIsTrainingAllowed: (value: boolean) => void;
}

const TrainingContext = createContext<TrainingContextType | undefined>(undefined);

export const TrainingProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [isTrainingAllowed, setIsTrainingAllowed] = useState(false);

  return (
    <TrainingContext.Provider value={{ isTrainingAllowed, setIsTrainingAllowed }}>
      {children}
    </TrainingContext.Provider>
  );
};

export const useTrainingContext = (): TrainingContextType => {
  const context = useContext(TrainingContext);
  if (!context) {
    throw new Error("useTrainingContext must be used within a TrainingProvider");
  }
  return context;
};
