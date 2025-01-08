import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import "./styles/GlobalStyles.scss";


import App from './App.tsx'
import "bootstrap/dist/css/bootstrap.min.css";
import { DatasetProvider } from "./context/DatasetContext";
import { BuildPageProvider } from './context/BuildPageContext.tsx';
import { TrainPageProvider } from './context/TrainPageContext.tsx';
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <DatasetProvider>
      <BuildPageProvider>
        <TrainPageProvider>
          <App />
          </TrainPageProvider>
          </BuildPageProvider>
    </DatasetProvider>
      
  </StrictMode>
)


