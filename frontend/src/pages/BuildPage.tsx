import React from "react";
import { useDataset } from "../context/DatasetContext";

const BuildPage: React.FC = () => {
    const { dataset } = useDataset(); // Access the dataset from React Context

    return (
        <div className="container-fluid bg-light m-0 p-0" style={{ minHeight: "100vh" }}>
            <h1 className="text-center">Build Neural Network</h1>
            <p className="text-center">
                Selected Dataset:{" "}
                <strong>{dataset || "No dataset selected"}</strong>
            </p>
        </div>
    );
};

export default BuildPage;
