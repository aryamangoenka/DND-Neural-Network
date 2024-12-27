import React from "react";
import { useLocation } from "react-router-dom";

const BuildPage: React.FC = () => {
    const location = useLocation();
    const dataset = location.state?.dataset || "No dataset selected";

    return (
        <div className="container mt-5">
            <h1 className="text-center">Build Neural Network</h1>
            <p className="text-center">Selected Dataset: <strong>{dataset}</strong></p>
        </div>
    );
};

export default BuildPage;
