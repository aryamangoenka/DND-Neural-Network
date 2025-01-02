import { useCallback, useState } from "react";
import ReactFlow, {
  addEdge,
  Background,
  Controls,
  useNodesState,
  useEdgesState,
  Node,
  Edge,
  Connection,
} from "reactflow";
import "reactflow/dist/style.css";
import "../styles/components/BuildPage.scss";

const initialNodes: Node[] = [
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
];
const initialEdges: Edge[] = [];

const BuildPage = (): JSX.Element => {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);

  const addLayer = (type: string): void => {
    const newNode: Node = {
      id: `${type}-${Date.now()}`,
      data: { label: `${type} Layer`, activation: "None" },
      position: { x: Math.random() * 600, y: Math.random() * 400 },
      type: type.toLowerCase(),
    };
    setNodes((nds) => [...nds, newNode]);
  };

  const onConnect = useCallback(
    (connection: Connection): void => {
      setEdges((eds) => addEdge(connection, eds));
    },
    [setEdges]
  );

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
    };
    setNodes(templates[templateKey]);
    setEdges([]);
  };

  const onNodeClick = (_: any, node: Node): void => {
    setSelectedNode(node);
  };

  const updateActivation = (activation: string): void => {
    if (!selectedNode) return;

    setNodes((nds) =>
      nds.map((node) =>
        node.id === selectedNode.id
          ? {
              ...node,
              data: {
                ...node.data,
                activation: activation,
                label:
                  activation === "None"
                    ? node.data.label.split(" ")[0]
                    : `${node.data.label.split(" ")[0]} (${activation})`,
              },
            }
          : node
      )
    );

    setSelectedNode((prevNode) =>
      prevNode
        ? {
            ...prevNode,
            data: {
              ...prevNode.data,
              activation: activation,
            },
          }
        : null
    );
  };

  return (
    <div className="build-page">
      <div className="left-sidebar">
        <h2>Layers</h2>
        {["Dense", "Convolution", "MaxPooling", "Flatten", "Dropout"].map(
          (type) => (
            <div key={type} className="layer-card">
              <span>{type}</span>
              <button onClick={() => addLayer(type)}>Add</button>
            </div>
          )
        )}

        <h2>Activation Layers</h2>
        <ul>
          <li>ReLU</li>
          <li>Sigmoid</li>
          <li>Softmax</li>
          <li>Tanh</li>
          <li>Leaky ReLU</li>
        </ul>
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

        {selectedNode && (
          <div className="parameters-section">
            <h2>Parameters</h2>
            <p>Layer Type: {selectedNode.type || "Unknown"}</p>
            {["dense", "convolution"].includes(selectedNode.type ?? "") && (
              <>
                <label>Activation:</label>
                <select
                  value={selectedNode.data.activation || "None"}
                  onChange={(e) => updateActivation(e.target.value)}
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
          </div>
        )}
        <h2>Train</h2>
        <button className="train-button">Start Training</button>
      </div>
    </div>
  );
};

export default BuildPage;
