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
import { useDataset } from "../context/DatasetContext";
import {
  DenseNode,
  ConvolutionNode,
  MaxPoolingNode,
  FlattenNode,
  DropoutNode,
  BatchNormalizationNode,
  InputNode,
  OutputNode,
} from "../components/CustomNodes";

const nodeTypes = {
  input: InputNode,
  output: OutputNode,
  dense: DenseNode,
  convolution: ConvolutionNode,
  maxpooling: MaxPoolingNode,
  flatten: FlattenNode,
  dropout: DropoutNode,
  batchnormalization: BatchNormalizationNode,
};

const BuildPage = (): JSX.Element => {
  const {
    nodes,
    setNodes,
    edges,
    setEdges,
    validationErrors,
    setValidationErrors,
  } = useBuildPageContext();
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const { dataset } = useDataset();
  const [isTraining, setIsTraining] = useState(false);

  const addLayer = (type: string): void => {
    const defaultParams: Record<string, any> = {
      dense: { neurons: 64, activation: "None" },
      convolution: { filters: 32, kernelSize: [3, 3], stride: [1, 1] },
      maxpooling: { poolSize: [2, 2], stride: [2, 2], padding: "none" },
      dropout: { rate: 0.2 },
      activation: { activation: "ReLU" },
      batchnormalization: { momentum: 0.99, epsilon: 0.001 },
      output: { activation: "None" },
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

  const validateLayerParameters = (): Record<
    string,
    Record<string, string>
  > => {
    const errors: Record<string, Record<string, string>> = {};

    nodes.forEach((node) => {
      const { type, data } = node;
      const nodeErrors: Record<string, string> = {};

      // Ensure at least one input layer exists
      const inputNodes = nodes.filter((node) => node.type === "input");
      if (inputNodes.length === 0) {
        errors["global"] = { inputLayer: "An input layer is required." };
      } else {
        // Ensure all input nodes are connected
        inputNodes.forEach((inputNode) => {
          const isConnected = edges.some(
            (edge) => edge.source === inputNode.id
          );
          if (!isConnected) {
            errors[inputNode.id] = {
              connection:
                "Input layer must have at least one outgoing connection.",
            };
          }
        });
      }
      // Ensure at least one output layer exists
      const outputNodes = nodes.filter((node) => node.type === "output");
      if (outputNodes.length === 0) {
        errors["global"] = {
          ...(errors["global"] || {}),
          outputLayer: "An output layer is required.",
        };
      } else {
        // Ensure all output nodes are connected
        outputNodes.forEach((outputNode) => {
          const isConnected = edges.some(
            (edge) => edge.target === outputNode.id
          );
          if (!isConnected) {
            errors[outputNode.id] = {
              connection:
                "Output layer must have at least one incoming connection.",
            };
          }
        });
      }

      // Ensure no disconnected nodes exist
      const allConnectedNodeIds = new Set(
        edges.flatMap((edge) => [edge.source, edge.target])
      );
      nodes.forEach((node) => {
        if (!allConnectedNodeIds.has(node.id)) {
          errors[node.id] = {
            ...(errors[node.id] || {}),
            connection: "This layer is not connected to any other layer.",
          };
        }
      });

      // Dense Layer Validation
      if (type === "dense") {
        if (!Number.isInteger(data.neurons) || data.neurons <= 0) {
          nodeErrors.neurons = "Number of neurons must be a positive integer.";
        }
        if (
          ![
            "None",
            "ReLU",
            "Sigmoid",
            "Softmax",
            "Tanh",
            "Leaky ReLU",
          ].includes(data.activation)
        ) {
          nodeErrors.activation =
            "Invalid activation function for dense layer.";
        }
      }
      // Convolutional Layer Validation
      if (type === "convolution") {
        if (!Number.isInteger(data.filters) || data.filters <= 0) {
          nodeErrors.filters = "Number of filters must be a positive integer.";
        }
        if (
          !Array.isArray(data.kernelSize) ||
          data.kernelSize.length !== 2 ||
          data.kernelSize.some((v: number) => v <= 0)
        ) {
          nodeErrors.kernelSize =
            "Kernel size must be an array of two positive integers.";
        }
        if (
          !Array.isArray(data.stride) ||
          data.stride.length !== 2 ||
          data.stride.some((v: number) => v <= 0)
        ) {
          nodeErrors.stride =
            "Stride must be an array of two positive integers.";
        }
        if (
          ![
            "None",
            "ReLU",
            "Sigmoid",
            "Softmax",
            "Tanh",
            "Leaky ReLU",
          ].includes(data.activation)
        ) {
          nodeErrors.activation =
            "Invalid activation function for convolutional layer.";
        }
      }

      // MaxPooling Layer Validation
      if (type === "maxpooling") {
        if (
          !Array.isArray(data.poolSize) ||
          data.poolSize.length !== 2 ||
          data.poolSize.some((v: number) => v <= 0)
        ) {
          nodeErrors.poolSize =
            "Pool size must be an array of two positive integers.";
        }
        if (
          !Array.isArray(data.stride) ||
          data.stride.length !== 2 ||
          data.stride.some((v: number) => v <= 0)
        ) {
          nodeErrors.stride =
            "Stride must be an array of two positive integers.";
        }
        if (!["valid", "same"].includes(data.padding)) {
          nodeErrors.padding = "Padding must be 'valid' or 'same'.";
        }
      }

      // Dropout Layer Validation
      if (type === "dropout") {
        if (typeof data.rate !== "number" || data.rate < 0 || data.rate > 1) {
          nodeErrors.rate = "Dropout rate must be a number between 0 and 1.";
        }
      }

      // Batch Normalization Layer Validation
      if (type === "batchnormalization") {
        if (
          typeof data.momentum !== "number" ||
          data.momentum <= 0 ||
          data.momentum >= 1
        ) {
          nodeErrors.momentum = "Momentum must be a number between 0 and 1.";
        }
        if (typeof data.epsilon !== "number" || data.epsilon <= 0) {
          nodeErrors.epsilon = "Epsilon must be a positive number.";
        }
      }

      // Output Layer Validation
      if (type === "output") {
        if (!["None", "Sigmoid", "Softmax"].includes(data.activation)) {
          nodeErrors.activation =
            "Invalid activation function for output layer.";
        }
      }
      // Iris-specific validations
      if (dataset === "Iris") {
        // Output Layer Validation
        if (type === "output") {
          if (data.activation !== "Softmax") {
            nodeErrors.activation =
              "Output layer must use Softmax activation for Iris dataset.";
          }
        }

        // Ensure no incompatible layers for Iris
        if ([type && "convolution", "maxpooling", "flatten"].includes(type)) {
          nodeErrors.type = `Layer type '${type}' is not compatible with the Iris dataset.`;
        }
      }
      if (dataset === "MNIST" || dataset === "CIFAR-10") {
        // Ensure no incompatible layers for MNIST
        const hasConvolutionLayer = nodes.some(
          (node) => node.type === "convolution"
        );

        if (type && ["dense"].includes(type) && !hasConvolutionLayer) {
          nodeErrors.type = `Layer type '${type}' should not be used alone for MNIST dataset. Add a convolutional.`;
        }

        // Ensure MaxPooling and Convolution layers are followed by valid layers
        if (type === "convolution" || type === "maxpooling") {
          const isFlattenConnected = edges.some((edge) =>
            nodes.some(
              (targetNode) =>
                edge.source === node.id && targetNode.type === "flatten"
            )
          );
          if (!isFlattenConnected) {
            nodeErrors.connection = `convulation/maxpooling layer must connect to a Flatten layer.`;
          }
        }

        // Output Layer Activation Check
        if (type === "output" && data.activation !== "Softmax") {
          nodeErrors.activation =
            "Output layer must use Softmax activation for MNIST dataset.";
        }
      }

      // California Housing Dataset Specific Validations
      if (dataset === "California Housing") {
        // Ensure no incompatible layers for California Housing
        if (type && ["convolution", "maxpooling", "flatten"].includes(type)) {
          nodeErrors.type = `Layer type '${type}' is not compatible with the California Housing dataset.`;
        }

        // Output Layer Activation Check
        if (type === "output" && data.activation !== "None") {
          nodeErrors.activation =
            "Output layer must have no activation (linear) for California Housing dataset.";
        }
      }

      // Breast Cancer Dataset Specific Validations
      if (dataset === "Breast Cancer") {
        // Ensure no incompatible layers for Breast Cancer
        if (type && ["convolution", "maxpooling", "flatten"].includes(type)) {
          nodeErrors.type = `Layer type '${type}' is not compatible with the Breast Cancer dataset.`;
        }

        // Output Layer Activation Check
        if (type === "output" && data.activation !== "Sigmoid") {
          nodeErrors.activation =
            "Output layer must use Sigmoid activation for Breast Cancer dataset.";
        }
      }

      if (Object.keys(nodeErrors).length > 0) {
        errors[node.id] = nodeErrors;
      }
    });

    return errors;
  };
  const serializePayload = (): { nodes: any[]; edges: any[] } => {
    // Extract relevant fields for nodes
    const serializedNodes = nodes.map(({ id, type, data }) => ({
      id,
      type,
      data,
    }));

    // Extract relevant fields for edges
    const serializedEdges = edges.map(({ source, target }) => ({
      source,
      target,
    }));

    return { nodes: serializedNodes, edges: serializedEdges };
  };

  const handleSaveConfig = async (): Promise<void> => {
    const errors = validateLayerParameters();

    if (!dataset) {
      alert("No dataset selected! Please select a dataset before training.");
      return;
    }

    if (Object.keys(errors).length > 0) {
      setValidationErrors(errors);
      alert("Validation failed! Please fix the errors.");
      return;
    }
    setIsTraining(true);

    try {
      // Serialize the payload
      const payload = {
        ...serializePayload(),
        dataset, // Add the selected dataset to the payload
      };
      console.log(payload);
      // Make a POST request to the backend
      const response = await fetch("http://127.0.0.1:5000/save_model", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.error("Backend Error:", errorData);
        alert(`Training failed: ${errorData.error || "Unknown error"}`);
        return;
      }

      const data = await response.json();
      console.log("Response from Backend:", data);
      alert(data.message || "Saving started successfully!");
      setValidationErrors({});
    } catch (error) {
      alert(`An error occurred: ${error}`);
    } finally {
      setIsTraining(false);
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
          "BatchNormalization",
        ].map((type) => (
          <div key={type} className="layer-card">
            <span>{type}</span>
            <button onClick={() => addLayer(type)}>Add</button>
          </div>
        ))}

        {/* Templates Section */}
        <div className="templates-section">
          <h2>Templates</h2>
          <button onClick={() => loadTemplate("SimpleFeedforward")}>
            Simple Feedforward
          </button>
          <button onClick={() => loadTemplate("CNN")}>CNN</button>
          <button onClick={() => loadTemplate("FullyConnectedRegression")}>
            Fully Connected Regression
          </button>
          <button onClick={() => loadTemplate("Blank")}>Blank Template</button>
        </div>
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
          nodeTypes={nodeTypes} // Registered here
        >
          <Background />
          <Controls />
        </ReactFlow>
      </div>

      <div className="right-sidebar">
        <p className="text-center">
          Dataset: <strong>{dataset || "No dataset selected"}</strong>
        </p>
        <div className="train-section">
          <button
            className="train-button"
            onClick={handleSaveConfig}
            disabled={isTraining} // Disable button during training
          >
            {isTraining ? (
              <span className="spinner"></span> // Spinner while training
            ) : (
              "Save Model"
            )}
          </button>
          {isTraining && <p>Training in progress. Please wait...</p>}
        </div>

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
                  {(
                    Object.values(
                      validationErrors[selectedNode.id] || {}
                    ) as string[]
                  ).map((error, index) => (
                    <p key={index} style={{ color: "red" }}>
                      {error}
                    </p>
                  ))}
                </div>
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
                    <option value="Sigmoid">Sigmoid</option>
                    <option value="Softmax">Softmax</option>
                  </select>
                </>
              )}

              {/* Dense Layer */}
              {selectedNode.type === "dense" && (
                <>
                  <label>Neurons:</label>
                  <input
                    type="number"
                    value={selectedNode.data.neurons || ""}
                    onChange={(e) =>
                      updateParameter("neurons", +e.target.value)
                    }
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
              {selectedNode.type === "dropout" && (
                <>
                  <label>Dropout Rate:</label>
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.01"
                    value={selectedNode.data.rate || 0.2} // Default rate is 0.2
                    onChange={
                      (e) =>
                        updateParameter("rate", parseFloat(e.target.value) || 0) // Parse the value as a float
                    }
                  />
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
                    value={selectedNode.data.padding || "none"}
                    onChange={(e) => updateParameter("padding", e.target.value)}
                  >
                    <option value="none">None</option>
                    <option value="valid">Valid</option>
                    <option value="same">Same</option>
                  </select>
                </>
              )}
              {/* Batch Normalization Layer */}
              {selectedNode.type === "batchnormalization" && (
                <>
                  <label>Momentum:</label>
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.01"
                    value={selectedNode.data.momentum || 0.99}
                    onChange={(e) =>
                      updateParameter("momentum", parseFloat(e.target.value))
                    }
                  />

                  <label>Epsilon:</label>
                  <input
                    type="number"
                    min="0.00001"
                    step="0.00001"
                    value={selectedNode.data.epsilon || 0.001}
                    onChange={(e) =>
                      updateParameter("epsilon", parseFloat(e.target.value))
                    }
                  />
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
