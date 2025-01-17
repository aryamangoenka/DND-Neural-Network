import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import HomePage from "./pages/HomePage";
import BuildPage from "./pages/BuildPage";
import TrainPage from "./pages/TrainPage";
import DeployPage from "./pages/DeployPage";
import NavBar from "./components/NavBar";
import axios from "axios";
import { useEffect } from "react";
const AppContent: React.FC = () => {
    useEffect(() => {
        // Send a POST request to clear saved_model.json on app load
        axios.post("http://localhost:5000/api/clear_model")
            .then(response => {
                console.log("🧹", response.data.message);  // Logs success message
            })
            .catch(error => {
                console.error("❌ Error clearing model:", error);
            });
    }, []);  // Runs once on app load
    
    return (
        
        <div className="container-fluid m-0 p-0" style={{minHeight: "100vh"}}>
            
            

            {/* Routes */}
            <Routes>
                <Route path="/" element={<HomePage />} />
                <Route path="/build" element={<BuildPage />} />
                <Route path="/train" element={<TrainPage />} />
                <Route path="/deploy" element={<DeployPage />} />
            </Routes>
        </div>
    );
};

const App: React.FC = () => {
    return (
        <BrowserRouter>
        { <NavBar />}
            <AppContent />
        </BrowserRouter>
    );
};

export default App;
