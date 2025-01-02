import React, { createContext, useContext, useState } from "react";

// Updated Layer interface to include x and y properties
interface Layer {
    id: string;
    type: string;
    color: string;
    x: number; // X-coordinate for position
    y: number; // Y-coordinate for position
    
}

// BuildPageContextType uses the updated Layer interface
interface BuildPageContextType {
    layers: Layer[];
    setLayers: React.Dispatch<React.SetStateAction<Layer[]>>;
}

const BuildPageContext = createContext<BuildPageContextType | undefined>(undefined);

export const BuildPageProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    // Initialize layers with x and y properties
    const [layers, setLayers] = useState<Layer[]>([
        { id: "Input-1", type: "Input", color: "#cccccc", x: 100, y: 100 },
        { id: "Output-1", type: "Output", color: "#cccccc", x: 700, y: 300 }
    ]);

    return (
        <BuildPageContext.Provider value={{ layers, setLayers }}>
            {children}
        </BuildPageContext.Provider>
    );
};

export const useBuildPageContext = () => {
    const context = useContext(BuildPageContext);
    if (!context) {
        throw new Error("useBuildPageContext must be used within a BuildPageProvider");
    }
    return context;
};
