import { useState } from "react";
import axios from "axios";
import "../styles/components/DeployPage.scss";

import "bootstrap/dist/css/bootstrap.min.css";
import { useTrainPageContext } from "../context/TrainPageContext";

const DeployPage = (): JSX.Element => {
  const [statusMessage, setStatusMessage] = useState<string>("");

  const { modelTrained } = useTrainPageContext();

  // Generic function to handle downloads
  const handleExport = async (format: string) => {
    if (!modelTrained) {
      alert("⚠️ Please train the model before exporting.");
      return; // Stop export if model isn't trained
    }

    setStatusMessage(`Exporting model as ${format.toUpperCase()}...`);

    try {
      const response = await axios.get(
        `http://127.0.0.1:5000/export/${format}`,
        {
          responseType: "blob", // Important for file download
        }
      );
      // 📝 If exporting as savedmodel, download as a .zip file
      // 📝 Set correct file extension for each format
      let fileExtension;
      if (format === "savedmodel") {
        fileExtension = "zip";
      } else if (format === "pytorch") {
        fileExtension = "py"; // ✅ PyTorch model extension
      } else {
        fileExtension = format;
      }

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", `trained_model.${fileExtension}`);
      document.body.appendChild(link);
      link.click();
      link.remove();

      setStatusMessage(
        `Model exported successfully as ${format.toUpperCase()}!`
      );
    } catch (error) {
      console.error("Export failed:", error);
      setStatusMessage(`Failed to export model as ${format.toUpperCase()}.`);
    }
  };

  return (
    <div className="container fluid">
      <div className="hero-section">
        <h1 className="home-title"> Ready to Launch?</h1>
        <h1 className="home-subtitle">
          📤 Turn your hard work into deployable models!
        </h1>

        <div className="d-flex justify-content-center gap-3 mt-4">
          <button
            className="btn btn-primary"
            onClick={() => handleExport("py")}
          >
            📄 Export as Python Script
          </button>

          <button
            className="btn btn-success"
            onClick={() => handleExport("ipynb")}
          >
            📓 Export as Jupyter Notebook
          </button>

          <button
            className="btn btn-warning"
            onClick={() => handleExport("savedmodel")}
          >
            🔄 Export as SavedModel
          </button>

          <button
            className="btn btn-danger"
            onClick={() => handleExport("keras")}
          >
            🔄 Export as keras
          </button>

          <button
            className="btn btn-info"
            onClick={() => handleExport("pytorch")}
          >
            🔄 Export as Pytorch
          </button>
        </div>

        {statusMessage && (
          <div
            className={`alert ${
              statusMessage.includes("Failed")
                ? "alert-danger"
                : "alert-success"
            } mt-4`}
            role="alert"
          >
            {statusMessage}
          </div>
        )}
      </div>
    </div>
  );
};

export default DeployPage;
