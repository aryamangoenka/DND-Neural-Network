import React from "react";
import {
  DndContext,
  closestCenter,
  useSensor,
  useSensors,
  PointerSensor,
  DragEndEvent,
} from "@dnd-kit/core";
import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import "../styles/components/BuildPage.scss";
import { useBuildPageContext } from "../context/BuildPageContext";
import { useDataset } from "../context/DatasetContext";

type Layer = {
  id: string;
  type: string;
  color: string;
  x: number;
  y: number;
  
};

const BuildPage: React.FC = () => {
  const { layers, setLayers } = useBuildPageContext() as {
    layers: Layer[];
    setLayers: React.Dispatch<React.SetStateAction<Layer[]>>;
  };

  const sensors = useSensors(useSensor(PointerSensor));
  const { dataset } = useDataset();

  const availableLayers = [
    { type: "Dense", color: "#ffffff" },
    { type: "Convolution", color: "#ffffff" },
    { type: "MaxPooling", color: "#ffffff" },
    { type: "Batch Normalization", color: "#ffffff" },
    { type: "Flatten", color: "#ffffff" },
    { type: "Dropout", color: "#ffffff" },
  ];

  const activationLayers = [
    { type: "RelU", color: "#ffffff" },
    { type: "Sigmoid", color: "#ffffff" },
    { type: "Softmax", color: "#ffffff" },
    { type: "Leaky ReLU", color: "#ffffff" },
    { type: "Tanh", color: "#ffffff" },
  ];

  const templates: Record<string, Layer[]> = {
    SimpleFeedforward: [
      { id: "Input-1", type: "Input", color: "#cccccc", x: 100, y: 100 },
      { id: "Dense-1", type: "Dense", color: "#ffffff", x: 300, y: 200 },
      { id: "Output-1", type: "Output", color: "#eeeeee", x: 500, y: 300 },
    ],
    CNN: [
      { id: "Input-1", type: "Input", color: "#cccccc", x: 100, y: 100 },
      { id: "Conv-1", type: "Convolution", color: "#ffffff", x: 300, y: 200 },
      { id: "MaxPool-1", type: "MaxPooling", color: "#eeeeee", x: 500, y: 300 },
      { id: "Output-1", type: "Output", color: "#dddddd", x: 700, y: 400 },
    ],
    BinaryClassification: [
      { id: "Input-1", type: "Input", color: "#cccccc", x: 100, y: 100 },
      { id: "Dense-1", type: "Dense", color: "#ffffff", x: 300, y: 200 },
      { id: "Dense-2", type: "Dense", color: "#dddddd", x: 500, y: 300 },
      { id: "Output-1", type: "Output", color: "#eeeeee", x: 700, y: 400 },
    ],
  };

  const loadTemplate = (templateKey: string) => {
    const selectedTemplate = templates[templateKey];
    setLayers(selectedTemplate); // Update layers with the selected template
  };

  const addLayer = (layer: { type: string; color: string }) => {
    setLayers((prevLayers) => [
      ...prevLayers,
      {
        id: `${layer.type}-${Date.now()}`,
        type: layer.type,
        color: layer.color,
        x: 100, // Default position
        y: 100, // Default position
      },
    ]);
  };

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, delta } = event;

    setLayers((prevLayers) =>
      prevLayers.map((layer) =>
        layer.id === active.id
          ? {
              ...layer,
              x: layer.x + delta.x, // Update X-coordinate
              y: layer.y + delta.y, // Update Y-coordinate
            }
          : layer
      )
    );
  };

  return (
    <div className="build-page">
      {/* Left Sidebar */}
      <div className="left-sidebar">
        <p className="text-center ">
          <h3>
            {""} <strong>{dataset || "No dataset selected"}</strong>
          </h3>
        </p>

        <h2 className="text-white">Layers</h2>
        {availableLayers.map((layer) => (
          <button
            key={layer.type}
            style={{ backgroundColor: layer.color }}
            onClick={() => addLayer(layer)}
          >
            {layer.type}
          </button>
        ))}
        <h2>Activation Layers</h2>
        {activationLayers.map((layer) => (
          <button
            key={layer.type}
            style={{
              backgroundColor: layer.color,
            }}
            onClick={() => addLayer(layer)}
          >
            {layer.type}
          </button>
        ))}
      </div>

      {/* Drag-and-Drop Canvas */}
      <div className="canvas">
        <DndContext
          sensors={sensors}
          collisionDetection={closestCenter}
          onDragEnd={handleDragEnd}
        >
          {layers.map((layer) => (
            <DraggableLayer
              key={layer.id}
              id={layer.id}
              type={layer.type}
              color={layer.color}
              x={layer.x}
              y={layer.y}
            />
          ))}
        </DndContext>
      </div>

      {/* Right Sidebar */}
      <div className="right-sidebar">
        <div className="templates-section">
          <h2>Templates</h2>
          {Object.keys(templates).map((key) => (
            <button key={key} onClick={() => loadTemplate(key)}>
              {key.replace(/([A-Z])/g, " $1").trim()}
            </button>
          ))}
        </div>
        <button className="train-button">Train</button>
      </div>
    </div>
  );
};

interface DraggableLayerProps {
  id: string;
  type: string;
  color: string;
  x: number;
  y: number;
}

const DraggableLayer: React.FC<DraggableLayerProps> = ({
  id,
  type,
  color,
  x,
  y,
}) => {
  const { attributes, listeners, setNodeRef, transform, transition } =
    useSortable({ id });

  const style: React.CSSProperties = {
    position: "absolute",
    left: x + (transform?.x || 0),
    top: y + (transform?.y || 0),
    transform: CSS.Transform.toString(transform) || "translate(0, 0)",
    transition,
    backgroundColor: color,
    width: "120px",
    height: "50px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontWeight: "bold",
    borderRadius: "5px",
    boxShadow: "0 2px 4px rgba(0, 0, 0, 0.2)",
    cursor: "grab",
  };

  return (
    <div
      ref={setNodeRef}
      style={style}
      {...attributes}
      {...listeners}
      className="layer"
    >
      {type}
    </div>
  );
};

export default BuildPage;
