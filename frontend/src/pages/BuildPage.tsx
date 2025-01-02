import { useCallback } from "react";
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
  { id: "Input-1", data: { label: "Input Layer" }, position: { x: 100, y: 100 }, type: "input" },
  { id: "Output-1", data: { label: "Output Layer" }, position: { x: 700, y: 300 }, type: "output" },
];

const initialEdges: Edge[] = [];

const BuildPage = (): JSX.Element => {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Add a new layer (node)
  const addLayer = (type: string): void => {
    const newNode: Node = {
      id: `${type}-${Date.now()}`,
      data: { label: `${type} Layer` },
      position: { x: Math.random() * 600, y: Math.random() * 400 },
    };
    setNodes((nds) => [...nds, newNode]);
  };

  // Handle connections (edges) between nodes
  const onConnect = useCallback(
    (connection: Connection): void => {
      setEdges((eds) => addEdge(connection, eds));
    },
    [setEdges]
  );

  // Predefined templates
  const loadTemplate = (templateKey: string): void => {
    const templates: Record<string, Node[]> = {
      SimpleFeedforward: [
        { id: "Input-1", data: { label: "Input Layer" }, position: { x: 100, y: 100 }, type: "input" },
        { id: "Dense-1", data: { label: "Dense Layer" }, position: { x: 300, y: 200 } },
        { id: "Output-1", data: { label: "Output Layer" }, position: { x: 500, y: 300 }, type: "output" },
      ],
      CNN: [
        { id: "Input-1", data: { label: "Input Layer" }, position: { x: 100, y: 100 }, type: "input" },
        { id: "Conv-1", data: { label: "Convolution Layer" }, position: { x: 300, y: 200 } },
        { id: "MaxPool-1", data: { label: "MaxPooling Layer" }, position: { x: 500, y: 300 } },
        { id: "Output-1", data: { label: "Output Layer" }, position: { x: 700, y: 400 }, type: "output" },
      ],
      FullyConnectedRegression: [
        { id: "Input-1", data: { label: "Input Layer" }, position: { x: 100, y: 100 }, type: "input" },
        { id: "Dense-1", data: { label: "Dense Layer" }, position: { x: 300, y: 200 } },
        { id: "Dense-2", data: { label: "Dense Layer" }, position: { x: 500, y: 300 } },
        { id: "Output-1", data: { label: "Output Layer" }, position: { x: 700, y: 400 }, type: "output" },
      ],
    };
    setNodes(templates[templateKey]);
    setEdges([]); // Reset edges
  };

  return (
    <div className="build-page">
      {/* Left Sidebar */}
      <div className="left-sidebar">
        <h2>Layers</h2>
        {["Dense", "Convolution", "MaxPooling", "Flatten", "Dropout"].map((type) => (
          <div key={type} className="layer-card">
            <span>{type}</span>
            <button onClick={() => addLayer(type)}>Add</button>
          </div>
        ))}

        <h2>Activation Layers</h2>
        <ul>
          <li>ReLU</li>
          <li>Sigmoid</li>
          <li>Softmax</li>
          <li>Tanh</li>
        </ul>
      </div>

      {/* Canvas */}
      <div className="canvas">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          fitView
        >
          <Background />
          <Controls />
        </ReactFlow>
      </div>

      {/* Right Sidebar */}
      <div className="right-sidebar">
        <h2>Templates</h2>
        <button onClick={() => loadTemplate("SimpleFeedforward")}>Simple Feedforward</button>
        <button onClick={() => loadTemplate("CNN")}>CNN</button>
        <button onClick={() => loadTemplate("FullyConnectedRegression")}>
          Fully Connected Regression
        </button>

        <h2>Train</h2>
        <button className="train-button">Start Training</button>
      </div>
    </div>
  );
};

export default BuildPage;
