import React, { useState } from "react";
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

    const [selectedDataset, setSelectedDataset] = useState<string>("");

    // Handle dataset selection
    const handleDatasetSelect = () => {
        if (selectedDataset) {
            setDataset(selectedDataset); // Set the dataset in the context
            navigate("/build"); // Navigate to the Build Page
        } else {
            alert("Please select a dataset to proceed!");
        }
    };

    return (
        <div className="container-fluid bg-dark d-flex align-items-center justify-content-center" style={{ minHeight: "100vh" }}>
            <div className="text-center text-white">
                <h1 className="mb-4">Build.Train.Share</h1>
                <p className="text-center text-white mb-5">
                    An Interactive DND Neural Network builder
                </p>

                
                {/* Dropdown Menu */}
                <div className="mb-4">
                    <select
                        className="form-select form-select-lg"
                        value={selectedDataset}
                        onChange={(e) => setSelectedDataset(e.target.value)}
                        style={{ maxWidth: "400px", margin: "0 auto" }}
                    >
                        <option value="">Select Dataset</option>
                        {datasets.map((dataset) => (
                            <option key={dataset.name} value={dataset.name}>
                                {dataset.name} - {dataset.description}
                            </option>
                        ))}
                    </select>
                </div>
                
                {/* Submit Button */}
                <button
                    className="btn btn-secondary btn-lg"
                    onClick={handleDatasetSelect}
                    style={{ marginTop: "20px" }}
                >
                    Proceed
                </button>
            </div>
        </div>
    );
};

export default HomePage;
