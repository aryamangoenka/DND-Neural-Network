import { useState } from "react";
import axios from "axios";
import "../styles/components/SharePage.scss";
import { useDataset } from "../context/DatasetContext";
import 'bootstrap/dist/css/bootstrap.min.css';


const ExportPage = (): JSX.Element => {
  const dataset: string = useDataset()?.dataset || "Unknown";
  const [statusMessage, setStatusMessage] = useState<string>("");
  const [isDownloading, setIsDownloading] = useState<boolean>(false);

  // Generic function to handle downloads
  const handleExport = async (format: string) => {
    setIsDownloading(true);
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

    setIsDownloading(false);
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
          disabled={isDownloading}
        >
          📄 Export as Python Script
        </button>

        <button
          className="btn btn-outline-success"
          onClick={() => handleExport("ipynb")}
          disabled={isDownloading}
        >
          📓 Export as Jupyter Notebook
        </button>

        <button
          className="btn btn-outline-warning"
          onClick={() => handleExport("savedmodel")}
          disabled={isDownloading}
        >
          🔄 Export as SavedModel
        </button>

        <button
          className="btn btn-outline-success"
          onClick={() => handleExport("keras")}
          disabled={isDownloading}
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