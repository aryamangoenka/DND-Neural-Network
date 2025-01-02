import React, { createContext, useContext, useState } from "react";
import { Node, Edge } from "reactflow";

interface BuildPageContextType {
  nodes: Node[];
  setNodes: React.Dispatch<React.SetStateAction<Node[]>>;
  edges: Edge[];
  setEdges: React.Dispatch<React.SetStateAction<Edge[]>>;
}

const BuildPageContext = createContext<BuildPageContextType | undefined>(undefined);

export const BuildPageProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [nodes, setNodes] = useState<Node[]>([
    { id: "Input-1", data: { label: "Input Layer" }, position: { x: 100, y: 100 }, type: "input" },
    { id: "Output-1", data: { label: "Output Layer" }, position: { x: 700, y: 300 }, type: "output" },
  ]);

  const [edges, setEdges] = useState<Edge[]>([]);

  return (
    <BuildPageContext.Provider value={{ nodes, setNodes, edges, setEdges }}>
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
