import { Handle, Position } from "reactflow";
import "../styles/components/CustomNodes.scss";

// Dense Node
export const DenseNode = ({ data }: { data: any }) => (
  <div className="custom-node dense-node">
    <Handle type="target" position={Position.Left} style={{ background: "#555" }} />
    <h4>Dense Layer</h4>
    <p>Neurons: {data.neurons}</p>
    <p>Activation: {data.activation}</p>
    <Handle type="source" position={Position.Right} style={{ background: "#555" }} />
  </div>
);

// Convolutional Node
export const ConvolutionNode = ({ data }: { data: any }) => (
  <div className="custom-node convolution-node">
    <Handle type="target" position={Position.Left} style={{ background: "#555" }} />
    <h4>Convolution Layer</h4>
    <p>Filters: {data.filters}</p>
    <p>Kernel Size: {data.kernelSize?.join("x")}</p>
    <p>Stride: {data.stride?.join("x")}</p>
    <p>Activation: {data.activation}</p>
    <Handle type="source" position={Position.Right} style={{ background: "#555" }} />
  </div>
);

// MaxPooling Node
export const MaxPoolingNode = ({ data }: { data: any }) => (
  <div className="custom-node maxpooling-node">
    <Handle type="target" position={Position.Left} style={{ background: "#555" }} />
    <h4>MaxPooling Layer</h4>
    <p>Pool Size: {data.poolSize?.join("x")}</p>
    <p>Stride: {data.stride?.join("x")}</p>
    <p>Padding: {data.padding}</p>
    <Handle type="source" position={Position.Right} style={{ background: "#555" }} />
  </div>
);

// Flatten Node
export const FlattenNode = () => (
  <div className="custom-node flatten-node">
    <Handle type="target" position={Position.Left} style={{ background: "#555" }} />
    <h4>Flatten Layer</h4>
    <Handle type="source" position={Position.Right} style={{ background: "#555" }} />
  </div>
);

// Dropout Node
export const DropoutNode = ({ data }: { data: any }) => (
  <div className="custom-node dropout-node">
    <Handle type="target" position={Position.Left} style={{ background: "#555" }} />
    <h4>Dropout Layer</h4>
    <p>Rate: {data.rate}</p>
    <Handle type="source" position={Position.Right} style={{ background: "#555" }} />
  </div>
);

// Batch Normalization Node
export const BatchNormalizationNode = ({ data }: { data: any }) => (
  <div className="custom-node batchnormalization-node">
    <Handle type="target" position={Position.Left} style={{ background: "#555" }} />
    <h4>Batch Normalization Layer</h4>
    <p>Momentum: {data.momentum}</p>
    <p>Epsilon: {data.epsilon}</p>
    <Handle type="source" position={Position.Right} style={{ background: "#555" }} />
  </div>
);

// Input Node
export const InputNode = ({ data }: { data: any }) => (
  <div className="custom-node input-node">
    <h4>Input Layer</h4>
    <p>{data.label}</p>
    <Handle type="source" position={Position.Right} style={{ background: "#555" }} />
  </div>
);

// Output Node
export const OutputNode = ({ data }: { data: any }) => (
  <div className="custom-node output-node">
    <Handle type="target" position={Position.Left} style={{ background: "#555" }} />
    <h4>Output Layer</h4>
    <p>Activation: {data.activation}</p>
    
  </div>
);