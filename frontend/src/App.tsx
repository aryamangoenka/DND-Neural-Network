import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import HomePage from "./pages/HomePage";
import BuildPage from "./pages/BuildPage";
import TrainPage from "./pages/TrainPage";
import ExportPage from "./pages/ExportPage";
import NavBar from "./components/NavBar";


const AppContent: React.FC = () => {
    

    // Check if the current route is the Home Page
   

    return (
        <div className="container-fluid mt-3 m-0 p-0" style={{minHeight: "100vh"}}>
            {/* Conditionally Render Navigation Bar */}
            { <NavBar />}

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
