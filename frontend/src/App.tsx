import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import HomePage from "./pages/HomePage";
import BuildPage from "./pages/BuildPage";
import TrainPage from "./pages/TrainPage";
import DeployPage from "./pages/DeployPage";
import NavBar from "./components/NavBar";
import axios from "axios";
import { useEffect } from "react";

// Functional component to handle the main application content and routing
const AppContent: React.FC = () => {
    useEffect(() => {
        // Send a POST request to clear the saved model data when the app loads
        axios.post("http://localhost:5000/api/clear_model")
            .then(response => {
                console.log("🧹", response.data.message);  // Logs success message
            })
            .catch(error => {
                console.error("❌ Error clearing model:", error);  // Logs any errors
            });
    }, []);  // Empty dependency array ensures this effect runs only once when the component mounts
    
    return (
        <div className="container-fluid m-0 p-0" style={{ minHeight: "100vh" }}>
            {/* Define routes for different pages in the application */}
            <Routes>
                <Route path="/" element={<HomePage />} />       {/* Home Page Route */}
                <Route path="/build" element={<BuildPage />} />  {/* Build Page Route */}
                <Route path="/train" element={<TrainPage />} />  {/* Train Page Route */}
                <Route path="/deploy" element={<DeployPage />} />{/* Deploy Page Route */}
            </Routes>
        </div>
    );
};

// Main App component that wraps the application with routing and navigation bar
const App: React.FC = () => {
    return (
        <BrowserRouter>
            {/* NavBar component is displayed across all pages */}
            <NavBar />

            {/* Render the main application content */}
            <AppContent />
        </BrowserRouter>
    );
};

// Export the App component as the default export of this module
export default App;
