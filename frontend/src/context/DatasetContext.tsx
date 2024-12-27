import React, { createContext, useState, useContext } from "react";

// Define the context type
interface DatasetContextType {
    dataset: string | null;
    setDataset: (dataset: string | null) => void;
}

// Create the context
const DatasetContext = createContext<DatasetContextType | undefined>(undefined);

// Define the provider props
interface DatasetProviderProps {
    children: React.ReactNode; // Ensure children prop is recognized
}

// Provider component
export const DatasetProvider: React.FC<DatasetProviderProps> = ({ children }) => {
    const [dataset, setDataset] = useState<string | null>(null);

    return (
        <DatasetContext.Provider value={{ dataset, setDataset }}>
            {children}
        </DatasetContext.Provider>
    );
};

// Hook for using the context
export const useDataset = (): DatasetContextType => {
    const context = useContext(DatasetContext);
    if (!context) {
        throw new Error("useDataset must be used within a DatasetProvider");
    }
    return context;
};
