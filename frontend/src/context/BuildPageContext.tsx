import React, { createContext, useContext, useState } from "react";

interface Layer {
    id: string;
    type: string;
    color: string;
}

interface BuildPageContextType {
    layers: Layer[];
    setLayers: React.Dispatch<React.SetStateAction<Layer[]>>;
}

const BuildPageContext = createContext<BuildPageContextType | undefined>(undefined);

export const BuildPageProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [layers, setLayers] = useState<Layer[]>([]);

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
