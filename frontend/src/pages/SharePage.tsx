import React from "react";
import { useDataset } from "../context/DatasetContext";

const ExportPage: React.FC = () => {
    const { dataset } = useDataset(); // Access the dataset from React Context

    return (
        <div className="container-fluid" >
            <h1 className="text-center">Share it!</h1>
            <p className="text-center">
                Selected Dataset:{" "}
                <strong>{dataset || "No dataset selected"}</strong>
            </p>
        </div>
    );
};

export default ExportPage;
