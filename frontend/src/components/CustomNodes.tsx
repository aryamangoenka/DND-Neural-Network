import "../styles/components/CustomNodes.scss";


// Dense Node
export const DenseNode = ({ data }: { data: any }) => (
  <div className="custom-node dense-node">
    <h4>Dense Layer</h4>
    <p>Neurons: {data.neurons}</p>
    <p>Activation: {data.activation}</p>
  </div>
);

// Convolutional Node
export const ConvolutionNode = ({ data }: { data: any }) => (
  <div className="custom-node convolution-node">
    <h4>Convolution Layer</h4>
    <p>Filters: {data.filters}</p>
    <p>Kernel Size: {data.kernelSize?.join("x")}</p>
    <p>Stride: {data.stride?.join("x")}</p>
    <p>Activation: {data.activation}</p>
  </div>
);

// Dropout Node
export const DropoutNode = ({ data }: { data: any }) => (
  <div className="custom-node dropout-node">
    <h4>Dropout Layer</h4>
    <p>Rate: {data.rate}</p>
  </div>
);

// Activation Node
export const ActivationNode = ({ data }: { data: any }) => (
  <div className="custom-node activation-node">
    <h4>Activation Layer</h4>
    <p>Activation: {data.activation}</p>
  </div>
);

// Batch Normalization Node
export const BatchNormalizationNode = ({ data }: { data: any }) => (
  <div className="custom-node batch-normalization-node">
    <h4>Batch Normalization Layer</h4>
    <p>Momentum: {data.momentum}</p>
    <p>Epsilon: {data.epsilon}</p>
  </div>
);

// Output Node
export const OutputNode = ({ data }: { data: any }) => (
  <div className="custom-node output-node">
    <h4>Output Layer</h4>
    <p>Activation: {data.activation}</p>
  </div>
);
