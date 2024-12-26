import { useState, useEffect } from "react";
import axios from "axios";
import reactLogo from "./assets/react.svg";
import viteLogo from "/vite.svg";
import "./App.css";

function App() {
  // Existing state for your counter
  const [count, setCount] = useState(0);

  // New state for backend status
  const [backendStatus, setBackendStatus] = useState("");

  // Fetch backend status on component mount
  useEffect(() => {
    const fetchBackendStatus = async () => {
      try {
        const response = await axios.get(
          `${import.meta.env.VITE_BACKEND_URL}/api/health`
        );
        setBackendStatus(response.data.message); // Set the message from the backend
      } catch (error) {
        console.error("Error connecting to the backend:", error);
        setBackendStatus("Error connecting to backend");
      }
    };

    fetchBackendStatus();
  }, []); // Empty dependency array means this runs only once when the component mounts

  return (
    <>
      <div>
        <a href="https://vite.dev" target="_blank">
          <img src={viteLogo} className="logo" alt="Vite logo" />
        </a>
        <a href="https://react.dev" target="_blank">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
      </div>
      <h1>Vite + React</h1>
      <div className="card">
        <button onClick={() => setCount((count) => count + 1)}>
          count is {count}
        </button>
        <p>
          Edit <code>src/App.tsx</code> and save to test HMR
        </p>
      </div>
      <p className="read-the-docs">
        Click on the Vite and React logos to learn more
      </p>
      {/* New section to display the backend status */}
      <div className="backend-status">
        <h2>Backend Status</h2>
        <p>{backendStatus || "Loading backend status..."}</p>
      </div>
    </>
  );
}

export default App;
