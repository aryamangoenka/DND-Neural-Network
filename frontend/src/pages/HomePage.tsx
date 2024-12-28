import React from "react";
import { useNavigate } from "react-router-dom";
import { useDataset } from "../context/DatasetContext"; // Import the context hook
import "../styles/HomePage.css";

const HomePage: React.FC = () => {
    const navigate = useNavigate();
    const { setDataset } = useDataset(); // Access the setDataset function

    // Predefined datasets
    const datasets = [
        { name: "Iris", description: "Iris dataset for classification tasks." },
        { name: "MNIST", description: "MNIST dataset for image classification." },
        { name: "Titanic", description: "Titanic dataset for binary classification." },
        { name: "Wine Quality", description: "Wine Quality dataset for regression tasks." },
    ];

    // Handle dataset selection
    const handleDatasetSelect = (dataset: string) => {
        setDataset(dataset); // Set the dataset in the context
        navigate("/build"); // Navigate to the Build Page
    };

    return (
        
        <div className="container-fluid bg-secondary" style={{minHeight:"100vh"}} >
            <h1 className="text-center mb-4">Choose a Dataset</h1>
            <p className="text-center text-muted mb-5">
                Select one of the predefined datasets to build your neural network.
            </p>
            <div className="row">
                {datasets.map((dataset) => (
                    <div className="col-md-6 col-lg-3 mb-4" key={dataset.name}>
                        <div
                            className="card shadow-sm rounded hover-card"
                            style={{ cursor: "pointer" }}
                            onClick={() => handleDatasetSelect(dataset.name)}
                        >
                            <div className="card-body text-center">
                                <h5 className="card-title">{dataset.name}</h5>
                                <p className="card-text text-muted">{dataset.description}</p>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>

    );
};

export default HomePage;
