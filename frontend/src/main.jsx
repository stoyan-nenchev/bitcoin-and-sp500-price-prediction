import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import GlobalStyle from "./globalStyles.js";
import { BrowserRouter } from "react-router-dom";

createRoot(document.getElementById("root")).render(
    <BrowserRouter>
        <GlobalStyle />
        <App />
    </BrowserRouter>
);
