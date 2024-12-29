import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import HomePage from "./pages/HomePage";
import BuildPage from "./pages/BuildPage";
import TrainPage from "./pages/TrainPage";
import SharePage from "./pages/SharePage";
import NavBar from "./components/NavBar";


const AppContent: React.FC = () => {
    
    return (
        
        <div className="container-fluid m-0 p-0" style={{minHeight: "100vh"}}>
            
            

            {/* Routes */}
            <Routes>
                <Route path="/" element={<HomePage />} />
                <Route path="/build" element={<BuildPage />} />
                <Route path="/train" element={<TrainPage />} />
                <Route path="/share" element={<SharePage />} />
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
