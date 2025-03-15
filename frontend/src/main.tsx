import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./styles/GlobalStyles.scss";

import App from "./App.tsx";
import "bootstrap/dist/css/bootstrap.min.css";
import "@fortawesome/fontawesome-free/css/all.min.css";
import { DatasetProvider } from "./context/DatasetContext";
import { BuildPageProvider } from "./context/BuildPageContext.tsx";
import { TrainPageProvider } from "./context/TrainPageContext.tsx";
import { NewBuildPageProvider } from "./context/NewBuildPageContext.tsx";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <DatasetProvider>
      <BuildPageProvider>
        <TrainPageProvider>
          <NewBuildPageProvider>
            <App />
          </NewBuildPageProvider>
        </TrainPageProvider>
      </BuildPageProvider>
    </DatasetProvider>
  </StrictMode>
);
