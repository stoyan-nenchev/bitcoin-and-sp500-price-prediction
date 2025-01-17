import Header from "./components/Header";
import { Routes, Route } from "react-router";
import Bitcoin from "./components/Bitcoin";
import Spx from "./components/Spx";

function App() {
    return (
        <>
            <Header />
            <Routes>
                <Route path="/" element={<Spx />} />
                <Route path="/bitcoin" element={<Bitcoin />} />
            </Routes>
        </>
    );
}

export default App;
