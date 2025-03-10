import React, { createContext, useContext, useState, useEffect } from "react";
import { Node, Edge } from "reactflow";
// Define the structure of data for each Node
interface NodeData {
  label: string;
  [key: string]: any; // Allows for additional dynamic properties
}

// Define the structure of validation errors
type ValidationErrors = Record<string, Record<string, string>>;




// Context Type Definition
interface BuildPageContextType {
  nodes: Node<NodeData>[]; // Nodes with custom data type
  setNodes: React.Dispatch<React.SetStateAction<Node<NodeData>[]>>;
  edges: Edge[];
  setEdges: React.Dispatch<React.SetStateAction<Edge[]>>;
  validationErrors: Record<string, Record<string, string>>;
  setValidationErrors: React.Dispatch<React.SetStateAction<ValidationErrors>>;
  
}

const BuildPageContext = createContext<BuildPageContextType | undefined>(
  undefined
);

// **Blank Template Nodes and Edges**: Default layout
const blankTemplateNodes: Node<NodeData>[] = [
  { id: "Input-1", data: { label: "Input Layer" }, position: { x: 100, y: 100 }, type: "input" },
  { id: "Output-1", data: { label: "Output Layer" }, position: { x: 700, y: 300 }, type: "output" },
];
const blankTemplateEdges: Edge[] = [];

// BuildPageProvider Component
export const BuildPageProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [nodes, setNodes] = useState<Node<NodeData>[]>(blankTemplateNodes);
  const [edges, setEdges] = useState<Edge[]>(blankTemplateEdges);
  const [validationErrors, setValidationErrors] = useState<ValidationErrors>(
    {}
  );
  

 

  // Reset to Blank Template on Mount (ignores localStorage)
  useEffect(() => {
    setNodes(blankTemplateNodes);
    setEdges(blankTemplateEdges);
  }, []); // Runs once on mount

  return (
    <BuildPageContext.Provider
      value={{
        nodes,
        setNodes,
        edges,
        setEdges,
        validationErrors,
        setValidationErrors,
        
      }}
    >
      {children}
    </BuildPageContext.Provider>
  );
};

// Custom Hook to Access BuildPageContext
export const useBuildPageContext = () => {
  const context = useContext(BuildPageContext);
  if (!context) {
    throw new Error(
      "useBuildPageContext must be used within a BuildPageProvider"
    );
  }
  return context;
};
