import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import "./styles/GlobalStyles.scss";


<<<<<<< HEAD

=======
>>>>>>> 68e4c30d619c0bf796ec773f5a0a10b50fa30d92
import App from './App.tsx'
import "bootstrap/dist/css/bootstrap.min.css";
import { DatasetProvider } from "./context/DatasetContext";
import { BuildPageProvider } from './context/BuildPageContext.tsx';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <DatasetProvider>
      <BuildPageProvider>
            <App />
            </BuildPageProvider>
    </DatasetProvider>
      
  </StrictMode>
)


