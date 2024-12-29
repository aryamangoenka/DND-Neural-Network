import React from "react";
import {
    DndContext,
    closestCenter,
    useSensor,
    useSensors,
    PointerSensor,
} from "@dnd-kit/core";
import {
    SortableContext,
    verticalListSortingStrategy,
    useSortable,
    arrayMove,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import "../styles/components/BuildPage.scss";
import { useBuildPageContext } from "../context/BuildPageContext";

const BuildPage: React.FC = () => {
    const { layers, setLayers } = useBuildPageContext();
    const sensors = useSensors(useSensor(PointerSensor));

    const availableLayers = [
        { type: "Dense", color: "#4CAF50" },
        { type: "Convolution", color: "#FF5722" },
        { type: "MaxPooling", color: "#03A9F4" },
        { type: "Activation", color: "#9C27B0" },
    ];

    const addLayer = (layer: { type: string; color: string }) => {
        setLayers([
            ...layers,
            { id: `${layer.type}-${Date.now()}`, type: layer.type, color: layer.color },
        ]);
    };

    const handleDragEnd = (event: any) => {
        const { active, over } = event;
        if (!over || active.id === over.id) return;

        setLayers((prevLayers) => {
            const oldIndex = prevLayers.findIndex((item) => item.id === active.id);
            const newIndex = prevLayers.findIndex((item) => item.id === over.id);
            return arrayMove(prevLayers, oldIndex, newIndex);
        });
    };

    return (
        <div className="build-page">
            {/* Left Sidebar */}
            <div className="left-sidebar">
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
            </div>

            {/* Drag-and-Drop Canvas */}
            <div className="canvas">
                <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
                    <SortableContext
                        items={layers.map((layer) => layer.id)}
                        strategy={verticalListSortingStrategy}
                    >
                        <svg className="connections" xmlns="http://www.w3.org/2000/svg">
                            {layers.slice(1).map((_, index) => (
                                <line
                                    key={index}
                                    x1={50}
                                    y1={index * 100 + 75}
                                    x2={50}
                                    y2={(index + 1) * 100 + 75}
                                    stroke="black"
                                    strokeWidth={2}
                                />
                            ))}
                        </svg>
                        {layers.map((layer) => (
                            <DraggableLayer key={layer.id} id={layer.id} type={layer.type} color={layer.color} />
                        ))}
                    </SortableContext>
                </DndContext>
            </div>

            {/* Right Sidebar */}
            <div className="right-sidebar">
                <button className="train-button">Train</button>
            </div>
        </div>
    );
};

interface DraggableLayerProps {
    id: string;
    type: string;
    color: string;
}

const DraggableLayer: React.FC<DraggableLayerProps> = ({ id, type, color }) => {
    const { attributes, listeners, setNodeRef, transform, transition } = useSortable({ id });

    const style = {
        transform: CSS.Transform.toString(transform),
        transition,
        backgroundColor: color,
    };

    return (
        <div ref={setNodeRef} style={style} {...attributes} {...listeners} className="layer">
            {type}
        </div>
    );
};

export default BuildPage;

       
