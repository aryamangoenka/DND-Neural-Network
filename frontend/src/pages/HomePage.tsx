import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useDataset } from "../context/DatasetContext"; // Custom context hook
import "../styles/components/HomePage.scss"; // SCSS styling for HomePage


// Define props for HomePage (empty for now, but extensible)
interface HomePageProps {}

const HomePage = ({}: HomePageProps) => {
    const navigate = useNavigate(); // For navigation
    const { setDataset } = useDataset(); // Access global dataset state management

    const datasets = [
        { name: "Iris", description: "Multi-class classification" },
        { name: "MNIST", description: "Image classification" },
        { name: "California Housing", description: "Regression tasks" },
        { name: "CIFAR-10", description: "Image classification" },
        { name: "Breast Cancer", description: "Binary classification" }
    ];
    

    // State to track selected dataset
    const [selectedDataset, setSelectedDataset] = useState<string>("");

    // Handle dataset selection and navigation
    const handleDatasetSelect = () => {
        if (selectedDataset) {
            setDataset(selectedDataset); // Set the selected dataset globally
            navigate("/build"); // Navigate to the Build Page
        } else {
            alert("Please select a dataset to proceed!"); // Show alert if no dataset is selected
        }
    };

    return (
        <div className="container-fluid">
            <div className="hero-section">
                <h1 className="home-title">🚀 Build. Train. Share.</h1>
                <p className="home-subtitle">An Interactive Drag-and-Drop Neural Network Builder</p>

                <div className="dropdown-wrapper">
                    <select
                        className="form-select"
                        value={selectedDataset}
                        onChange={(e) => setSelectedDataset(e.target.value)}
                    >
                        <option value="">📂 Select Dataset</option>
                        {datasets.map((dataset) => (
                            <option key={dataset.name} value={dataset.name}>
                                {dataset.name} - {dataset.description}
                            </option>
                        ))}
                    </select>
                </div>

                <button
                    className="btn btn-secondary btn-lg proceed-button"
                    onClick={handleDatasetSelect}
                >
                    🚀 Proceed
                </button>
            </div>
        </div>
    );
};

export default HomePage;
