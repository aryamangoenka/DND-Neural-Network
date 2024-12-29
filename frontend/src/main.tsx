import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import "./styles/GlobalStyles.scss";


import App from './App.tsx'
import "bootstrap/dist/css/bootstrap.min.css";
import { DatasetProvider } from "./context/DatasetContext";

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <DatasetProvider>
            <App />
    </DatasetProvider>
  </StrictMode>,
)


