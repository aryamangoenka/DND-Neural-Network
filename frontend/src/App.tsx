import React, { useEffect } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import NewBuildPage from "./pages/NewBuildPage";
import TutorialPage from "./pages/TutorialPage";
import NavBar from "./components/NavBar";
import axios from "axios";

// AppContent component that uses the context
const AppContent: React.FC = () => {
  useEffect(() => {
    // Send a POST request to clear the saved model data when the app loads
    axios
      .post("http://localhost:5000/api/clear_model")
      .then((response) => {
        console.log("🧹", response.data.message); // Logs success message
      })
      .catch((error) => {
        console.error("❌ Error clearing model:", error); // Logs any errors
      });
  }, []); // Empty dependency array ensures this effect runs only once when the component mounts

  return (
    <>
      {/* NavBar component is displayed across all pages */}
      <NavBar />

      {/* Main application content */}
      <div className="container-fluid m-0 p-0" style={{ minHeight: "100vh" }}>
        {/* Define routes for different pages in the application */}
        <Routes>
          <Route path="/newbuild" element={<NewBuildPage />} />
          <Route path="/tutorial" element={<TutorialPage />} />
          <Route path="*" element={<Navigate to="/newbuild" replace />} />
        </Routes>
      </div>
    </>
  );
};

// Main App component that wraps the application with routing
const App: React.FC = () => {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  );
};

// Export the App component as the default export of this module
export default App;
