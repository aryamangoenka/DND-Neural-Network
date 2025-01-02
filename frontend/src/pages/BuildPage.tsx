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
  const { nodes, setNodes, edges, setEdges } = useBuildPageContext();
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);

  const addLayer = (type: string): void => {
    const defaultParams: Record<string, any> = {
      dense: { neurons: 64, activation: "None" },
      convolution: { filters: 32, kernelSize: "3x3", stride: 1 },
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

  const onNodeClick = (_: any, node: Node): void => {
    setSelectedNode(node);
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
          onNodeClick={onNodeClick}
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
                  <label>Filters:</label>
                  <input
                    type="number"
                    value={selectedNode.data.filters || ""}
                    onChange={(e) => updateParameter("filters", +e.target.value)}
                  />
                  <label>Kernel Size:</label>
                  <input
                    type="text"
                    value={selectedNode.data.kernelSize || ""}
                    onChange={(e) =>
                      updateParameter("kernelSize", e.target.value)
                    }
                  />
                  <label>Stride:</label>
                  <input
                    type="number"
                    value={selectedNode.data.stride || ""}
                    onChange={(e) => updateParameter("stride", +e.target.value)}
                  />
                </>
              )}
              {/* MaxPooling Layer */}
              {selectedNode.type === "maxpooling" && (
                <>
                  <label>Pool Size:</label>
                  <input
                    type="text"
                    value={selectedNode.data.poolSize || ""}
                    onChange={(e) => updateParameter("poolSize", e.target.value)}
                  />
                  <label>Stride:</label>
                  <input
                    type="number"
                    value={selectedNode.data.stride || ""}
                    onChange={(e) => updateParameter("stride", +e.target.value)}
                  />
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

              {/* Dropout Layer */}
              {selectedNode.type === "dropout" && (
                <>
                  <label>Dropout Rate:</label>
                  <input
                    type="number"
                    value={selectedNode.data.rate || ""}
                    step="0.1"
                    min="0"
                    max="1"
                    onChange={(e) => updateParameter("rate", +e.target.value)}
                  />
                </>
              )}

              {/* BatchNormalization Layer */}
              {selectedNode.type === "batch\nnormalization" && (
                <>
                  <label>Momentum:</label>
                  <input
                    type="number"
                    step="0.01"
                    value={selectedNode.data.momentum || 0.99}
                    onChange={(e) => updateParameter("momentum", +e.target.value)}
                  />

                  <label>Epsilon:</label>
                  <input
                    type="number"
                    step="0.0001"
                    value={selectedNode.data.epsilon || 0.001}
                    onChange={(e) => updateParameter("epsilon", +e.target.value)}
                  />
                </>
              )}

              {/* Activation Layer */}
              {selectedNode.type === "activation" && (
                <>
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

              {/* Output Layer */}
              {selectedNode.type === "output" && (
                <>
                  <label>Activation Function:</label>
                  <select
                    value={selectedNode.data.activation || "None"}
                    onChange={(e) =>
                      updateParameter("activation", e.target.value)
                    }
                  >
                    <option value="None">None</option>
                    <option value="Softmax">Softmax</option>
                    <option value="Sigmoid">Sigmoid</option>
                  </select>
                </>
              )}
            </>
          )}
        </div>

        
      </div>
    </div>
  );
};

export default BuildPage;
