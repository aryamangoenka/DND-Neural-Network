import React from "react";
import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import HomePage from "./pages/HomePage";
import BuildPage from "./pages/BuildPage";
import TrainPage from "./pages/TrainPage";
import ExportPage from "./pages/ExportPage";
import NavBar from "./components/NavBar";
import "./App.css";

const AppContent: React.FC = () => {
    const location = useLocation();

    // Check if the current route is the Home Page
    const isHomePage = location.pathname === "/";

    return (
        <div className="container mt-3">
            {/* Conditionally Render Navigation Bar */}
            {!isHomePage && <NavBar />}

            {/* Routes */}
            <Routes>
                <Route path="/" element={<HomePage />} />
                <Route path="/build" element={<BuildPage />} />
                <Route path="/train" element={<TrainPage />} />
                <Route path="/export" element={<ExportPage />} />
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
