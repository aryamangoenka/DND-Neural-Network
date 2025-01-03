import { useState } from "react";
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
import "../styles/components/BuildPage.scss";
import { useBuildPageContext } from "../context/BuildPageContext";

const BuildPage = (): JSX.Element => {
  const { nodes, setNodes, edges, setEdges, validationErrors, setValidationErrors } = useBuildPageContext();
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);

  const addLayer = (type: string): void => {
    const defaultParams: Record<string, any> = {
      dense: { neurons: 64, activation: "None" },
      convolution: { filters: 32, kernelSize: [3, 3], stride: [1, 1] },
      dropout: { rate: 0.2 },
      activation: { activation: "ReLU" },
      batchnormalization: { momentum: 0.99, epsilon: 0.001 },
      output: { activation: "Softmax" },
    };

    const newNode: Node = {
      id: `${type}-${Date.now()}`,
      data: { label: `${type} Layer`, ...defaultParams[type] },
      position: { x: Math.random() * 600, y: Math.random() * 400 },
      type: type.toLowerCase(),
    };

    setNodes((nds) => [...nds, newNode]);
  };

  const onNodesChange: OnNodesChange = (changes: NodeChange[]) => {
    setNodes((nds) => applyNodeChanges(changes, nds));
  };

  const onEdgesChange: OnEdgesChange = (changes: EdgeChange[]) => {
    setEdges((eds) => applyEdgeChanges(changes, eds));
  };

  const onConnect = (connection: Connection): void => {
    setEdges((eds) => addEdge(connection, eds));
  };

  const loadTemplate = (templateKey: string): void => {
    const templates: Record<string, Node[]> = {
      SimpleFeedforward: [
        {
          id: "Input-1",
          data: { label: "Input Layer" },
          position: { x: 100, y: 100 },
          type: "input",
        },
        {
          id: "Dense-1",
          data: { label: "Dense Layer", activation: "None" },
          position: { x: 300, y: 200 },
          type: "dense",
        },
        {
          id: "Output-1",
          data: { label: "Output Layer" },
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
          data: { label: "Convolution Layer", activation: "None" },
          position: { x: 300, y: 200 },
          type: "convolution",
        },
        {
          id: "MaxPool-1",
          data: { label: "MaxPooling Layer" },
          position: { x: 500, y: 300 },
          type: "maxpooling",
        },
        {
          id: "Output-1",
          data: { label: "Output Layer" },
          position: { x: 700, y: 400 },
          type: "output",
        },
      ],
      FullyConnectedRegression: [
        {
          id: "Input-1",
          data: { label: "Input Layer" },
          position: { x: 100, y: 100 },
          type: "input",
        },
        {
          id: "Dense-1",
          data: { label: "Dense Layer", activation: "None" },
          position: { x: 300, y: 200 },
          type: "dense",
        },
        {
          id: "Dense-2",
          data: { label: "Dense Layer", activation: "None" },
          position: { x: 500, y: 300 },
          type: "dense",
        },
        {
          id: "Output-1",
          data: { label: "Output Layer" },
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

  const validateLayerParameters = (): Record<string, Record<string, string>> => {
    const errors: Record<string, Record<string, string>> = {};

    nodes.forEach((node) => {
      const { type, data } = node;
      const nodeErrors: Record<string, string> = {};

      if (type === "dense") {
        if (!Number.isInteger(data.neurons) || data.neurons <= 0) {
          nodeErrors.neurons = "Number of neurons must be a positive integer.";
        }
      }

      if (type === "convolution") {
        if (!Number.isInteger(data.filters) || data.filters <= 0) {
          nodeErrors.filters = "Filters must be a positive integer.";
        }
        if (!Array.isArray(data.kernelSize) || data.kernelSize.some((v: number) => v <= 0)) {
          nodeErrors.kernelSize = "Kernel size must be positive integers.";
        }
        if (!Array.isArray(data.stride) || data.stride.some((v: number) => v <= 0)) {
          nodeErrors.stride = "Stride must be positive integers.";
        }
      }

      if (type === "maxpooling") {
        if (!Array.isArray(data.poolSize) || data.poolSize.some((v: number) => v <= 0)) {
          nodeErrors.poolSize = "Pool size must be positive integers.";
        }
        if (!Array.isArray(data.stride) || data.stride.some((v: number) => v <= 0)) {
          nodeErrors.stride = "Stride must be positive integers.";
        }
      }

      if (Object.keys(nodeErrors).length > 0) {
        errors[node.id] = nodeErrors;
      }
    });

    return errors;
  };

  const handleTrain = async (): Promise<void> => {
    const errors = validateLayerParameters();
  
    if (Object.keys(errors).length > 0) {
      setValidationErrors(errors);
      alert("Validation failed! Please fix the errors.");
      return;
    }
  
    try {
      // Serialize the nodes and edges
      const payload = {
        nodes,
        edges,
      };
  
      // Make a POST request to the backend
      const response = await fetch("http://127.0.0.1:5000/train", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });
  
      if (!response.ok) {
        const errorData = await response.json();
        alert(`Training failed: ${errorData.error || "Unknown error"}`);
        return;
      }
  
      const data = await response.json();
      alert(data.message || "Training started successfully!");
      setValidationErrors({});
    } catch (error) {
      alert(`An error occurred: ${error}`);
    }
  };
  
  return (
    <div className="build-page">
      <div className="left-sidebar">
        <h2>Layers</h2>
        {[
          "Dense",
          "Convolution",
          "MaxPooling",
          "Flatten",
          "Dropout",
          "Batch\nNormalization",
        ].map((type) => (
          <div key={type} className="layer-card">
            <span>{type}</span>
            <button onClick={() => addLayer(type)}>Add</button>
          </div>
        ))}
      </div>
  
      <div className="canvas">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={(_, node) => setSelectedNode(node)}
          fitView
        >
          <Background />
          <Controls />
        </ReactFlow>
      </div>
  
      <div className="right-sidebar">
        <h2>Templates</h2>
        <button onClick={() => loadTemplate("SimpleFeedforward")}>
          Simple Feedforward
        </button>
        <button onClick={() => loadTemplate("CNN")}>CNN</button>
        <button onClick={() => loadTemplate("FullyConnectedRegression")}>
          Fully Connected Regression
        </button>
        <button onClick={() => loadTemplate("Blank")}>Blank Template</button>
  
        <div className="parameters-section">
          <h2>Parameters</h2>
  
          {!selectedNode ? (
            <p>Select a layer to see parameters</p>
          ) : (
            <>
              <p>Layer Type: {selectedNode.type}</p>
  
              {/* Validation Errors */}
              {validationErrors[selectedNode.id] && (
                <div className="validation-errors">
                  {(Object.values(validationErrors[selectedNode.id] || {}) as string[]).map(
                    (error, index) => (
                      <p key={index} style={{ color: "red" }}>
                        {error}
                      </p>
                    )
                  )}
                </div>
              )}
  
              {/* Dense Layer */}
              {selectedNode.type === "dense" && (
                <>
                  <label>Neurons:</label>
                  <input
                    type="number"
                    value={selectedNode.data.neurons || ""}
                    onChange={(e) => updateParameter("neurons", +e.target.value)}
                  />
                  <label>Activation Function:</label>
                  <select
                    value={selectedNode.data.activation || "None"}
                    onChange={(e) =>
                      updateParameter("activation", e.target.value)
                    }
                  >
                    <option value="None">None</option>
                    <option value="ReLU">ReLU</option>
                    <option value="Sigmoid">Sigmoid</option>
                    <option value="Softmax">Softmax</option>
                    <option value="Tanh">Tanh</option>
                    <option value="Leaky ReLU">Leaky ReLU</option>
                  </select>
                </>
              )}
  
              {/* Convolutional Layer */}
              {selectedNode.type === "convolution" && (
                <>
                  {/* Filters */}
                  <label>Filters:</label>
                  <input
                    type="number"
                    min="1"
                    value={selectedNode.data.filters || ""}
                    onChange={(e) =>
                      updateParameter("filters", +e.target.value)
                    }
                  />
  
                  {/* Kernel Size */}
                  <label>Kernel Size:</label>
                  <div
                    style={{
                      display: "flex",
                      gap: "10px",
                      alignItems: "center",
                    }}
                  >
                    <input
                      type="number"
                      min="1"
                      value={selectedNode.data.kernelSize?.[0] || ""}
                      onChange={(e) =>
                        updateParameter("kernelSize", [
                          parseInt(e.target.value) || 1, // Height
                          selectedNode.data.kernelSize?.[1] || 1, // Current Width
                        ])
                      }
                      placeholder=""
                      style={{ width: "60px" }}
                    />
                    <span>x</span>
                    <input
                      type="number"
                      min="1"
                      value={selectedNode.data.kernelSize?.[1] || ""}
                      onChange={(e) =>
                        updateParameter("kernelSize", [
                          selectedNode.data.kernelSize?.[0] || 1, // Current Height
                          parseInt(e.target.value) || 1, // Width
                        ])
                      }
                      placeholder=""
                      style={{ width: "60px" }}
                    />
                  </div>
  
                  {/* Stride */}
                  <label>Stride:</label>
                  <div
                    style={{
                      display: "flex",
                      gap: "10px",
                      alignItems: "center",
                    }}
                  >
                    <input
                      type="number"
                      min="1"
                      value={selectedNode.data.stride?.[0] || ""}
                      onChange={(e) =>
                        updateParameter("stride", [
                          parseInt(e.target.value) || 1, // Height
                          selectedNode.data.stride?.[1] || 1, // Current Width
                        ])
                      }
                      placeholder=""
                      style={{ width: "60px" }}
                    />
                    <span>x</span>
                    <input
                      type="number"
                      min="1"
                      value={selectedNode.data.stride?.[1] || ""}
                      onChange={(e) =>
                        updateParameter("stride", [
                          selectedNode.data.stride?.[0] || 1, // Current Height
                          parseInt(e.target.value) || 1, // Width
                        ])
                      }
                      placeholder=""
                      style={{ width: "60px" }}
                    />
                  </div>
                </>
              )}
  
              {/* MaxPooling Layer */}
              {selectedNode.type === "maxpooling" && (
                <>
                  {/* Pool Size */}
                  <label>Pool Size:</label>
                  <div
                    style={{
                      display: "flex",
                      gap: "10px",
                      alignItems: "center",
                    }}
                  >
                    <input
                      type="number"
                      min="1"
                      value={selectedNode.data.poolSize?.[0] || ""}
                      onChange={(e) =>
                        updateParameter("poolSize", [
                          parseInt(e.target.value) || 1, // Height
                          selectedNode.data.poolSize?.[1] || 1, // Current Width
                        ])
                      }
                      placeholder=""
                      style={{ width: "60px" }}
                    />
                    <span>x</span>
                    <input
                      type="number"
                      min="1"
                      value={selectedNode.data.poolSize?.[1] || ""}
                      onChange={(e) =>
                        updateParameter("poolSize", [
                          selectedNode.data.poolSize?.[0] || 1, // Current Height
                          parseInt(e.target.value) || 1, // Width
                        ])
                      }
                      placeholder=""
                      style={{ width: "60px" }}
                    />
                  </div>
  
                  {/* Stride */}
                  <label>Stride:</label>
                  <div
                    style={{
                      display: "flex",
                      gap: "10px",
                      alignItems: "center",
                    }}
                  >
                    <input
                      type="number"
                      min="1"
                      value={selectedNode.data.stride?.[0] || ""}
                      onChange={(e) =>
                        updateParameter("stride", [
                          parseInt(e.target.value) || 1, // Height
                          selectedNode.data.stride?.[1] || 1, // Current Width
                        ])
                      }
                      placeholder=""
                      style={{ width: "60px" }}
                    />
                    <span>x</span>
                    <input
                      type="number"
                      min="1"
                      value={selectedNode.data.stride?.[1] || ""}
                      onChange={(e) =>
                        updateParameter("stride", [
                          selectedNode.data.stride?.[0] || 1, // Current Height
                          parseInt(e.target.value) || 1, // Width
                        ])
                      }
                      placeholder=""
                      style={{ width: "60px" }}
                    />
                  </div>
  
                  {/* Padding */}
                  <label>Padding:</label>
                  <select
                    value={selectedNode.data.padding || "valid"}
                    onChange={(e) => updateParameter("padding", e.target.value)}
                  >
                    <option value="valid">Valid</option>
                    <option value="same">Same</option>
                  </select>
                </>
              )}
            </>
          )}
        </div>
  
        {/* Train Button */}
        <button className="train-button" onClick={handleTrain}>
          Train
        </button>
      </div>
    </div>
  );
  
};

export default BuildPage;
