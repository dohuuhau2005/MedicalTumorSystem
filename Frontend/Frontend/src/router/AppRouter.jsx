import { Routes, Route } from "react-router-dom";

import Home from "../pages/Home";
import UploadPage from "../pages/UploadPage";
// import Result from "../pages/Result";
// import About from "../pages/About";

export default function AppRouter() {

    return (

        <Routes>

            <Route path="/" element={<Home />} />

            <Route path="/upload" element={<UploadPage />} />

            {/* <Route path="/result" element={<Result />} />

            <Route path="/about" element={<About />} /> */}

        </Routes>

    );

}