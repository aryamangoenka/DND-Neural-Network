import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useDataset } from "../context/DatasetContext"; // Import the context hook
import "../styles/components/HomePage.scss"; // Import the updated SCSS

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
        <div className="container-fluid">
            <div className="text-center">
                <h1 className="home-title">Build.Train.Share</h1>
                <p className="home-subtitle">
                    An Interactive DND Neural Network Builder
                </p>
                {/* Dropdown Menu */}
                <div className="dropdown-wrapper">
                    <select
                        className="form-select"
                        value={selectedDataset}
                        onChange={(e) => setSelectedDataset(e.target.value)}
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
                    className="btn btn-secondary btn-lg proceed-button"
                    onClick={handleDatasetSelect}
                >
                    Proceed
                </button>
            </div>
        </div>
    );
};

export default HomePage;
