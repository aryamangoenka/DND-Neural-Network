import { useState } from "react";
import axios from "axios";
import "../styles/components/SharePage.scss";
import { useDataset } from "../context/DatasetContext";
import 'bootstrap/dist/css/bootstrap.min.css';
import { useTrainPageContext } from "../context/TrainPageContext";

const ExportPage = (): JSX.Element => {
  const dataset: string = useDataset()?.dataset || "Unknown";
  const [statusMessage, setStatusMessage] = useState<string>("");
  
  const { modelTrained } = useTrainPageContext();

  // Generic function to handle downloads
  const handleExport = async (format: string) => {
    if (!modelTrained) {
      alert("⚠️ Please train the model before exporting.");
      return;  // Stop export if model isn't trained
    }
    
    setStatusMessage(`Exporting model as ${format.toUpperCase()}...`);

    try {
      const response = await axios.get(`http://127.0.0.1:5000/export/${format}`, {
        responseType: "blob", // Important for file download
      });
      // 📝 If exporting as savedmodel, download as a .zip file
      const fileExtension = format === "savedmodel" ? "zip" : format;

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", `trained_model.${fileExtension}`);
      document.body.appendChild(link);
      link.click();
      link.remove();

      setStatusMessage(`Model exported successfully as ${format.toUpperCase()}!`);
    } catch (error) {
      console.error("Export failed:", error);
      setStatusMessage(`Failed to export model as ${format.toUpperCase()}.`);
    }

    
  };

  return (
    <div className="container text-center mt-5">
      <h1 className="mb-4">📤 Export Your Trained Model</h1>
      <p className="text-muted">
        <strong>Selected Dataset:</strong> {dataset}
      </p>

      <div className="d-flex justify-content-center gap-3 mt-4">
        <button
          className="btn btn-outline-primary"
          onClick={() => handleExport("py")}
          
        >
          📄 Export as Python Script
        </button>

        <button
          className="btn btn-outline-success"
          onClick={() => handleExport("ipynb")}
          
        >
          📓 Export as Jupyter Notebook
        </button>

        <button
          className="btn btn-outline-warning"
          onClick={() => handleExport("savedmodel")}
          
        >
          🔄 Export as SavedModel
        </button>

        <button
          className="btn btn-outline-danger"
          onClick={() => handleExport("keras")}
          
        >
          🔄 Export as keras
        </button>
      </div>

      {statusMessage && (
        <div className={`alert ${statusMessage.includes('Failed') ? 'alert-danger' : 'alert-success'} mt-4`} role="alert">
          {statusMessage}
        </div>
      )}
    </div>
  );
};

export default ExportPage;