import { useState, useEffect } from "react";
import axios from "axios";
import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import API_BASE_URL from "./utils/apiConfig";
import HomePage from "./pages/HomePage";
import BuildPage from "./pages/BuildPage";
import NavBar from "./components/NavBar";
import "./App.css";

const AppContent: React.FC = () => {
    const [backendStatus, setBackendStatus] = useState("");

    // Fetch backend status on component mount
    useEffect(() => {
        const fetchBackendStatus = async () => {
            try {
                const response = await axios.get(`${API_BASE_URL}/api/health`);
                setBackendStatus(response.data.message); // Set the message from the backend
            } catch (error) {
                console.error("Error connecting to the backend:", error);
                setBackendStatus("Error connecting to backend");
            }
        };

        fetchBackendStatus();
    }, []);

    // Check if the current route is the Home Page
    const location = useLocation();
    const isHomePage = location.pathname === "/";

    return (
        <div className="container mt-3">
            {/* Backend Status */}
            <div className="alert alert-info text-center">
                Backend Status: {backendStatus || "Checking..."}
            </div>

            {/* Conditionally Render NavBar */}
            {!isHomePage && <NavBar />}

            {/* Routes */}
            <Routes>
                <Route path="/" element={<HomePage />} />
                <Route path="/build" element={<BuildPage />} />
                <Route path="/train" element={<div>Training Page</div>} />
                <Route path="/export" element={<div>Export Page</div>} />
            </Routes>
        </div>
    );
};

const App: React.FC = () => {
    return (
        <BrowserRouter>
            <AppContent />
        </BrowserRouter>
    );
};

export default App;
