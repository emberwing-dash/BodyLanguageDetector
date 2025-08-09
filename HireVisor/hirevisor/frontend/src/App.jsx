import React, { useState, useEffect } from "react";
import { Routes, Route } from "react-router-dom";

import HomePage from "./pages/HomePage";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import Dashboard from "./pages/Dashboard";
import InterviewSetup from "./pages/InterviewSetup";
import Report from "./pages/Report";

export default function App() {
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    document.body.classList.toggle("dark-mode", darkMode);
  }, [darkMode]);

  return (
    <Routes>
      <Route path="/" element={<HomePage darkMode={darkMode} setDarkMode={setDarkMode} />} />
      <Route path="/login" element={<Login darkMode={darkMode} setDarkMode={setDarkMode} />} />
      <Route path="/signup" element={<Signup darkMode={darkMode} setDarkMode={setDarkMode} />} />
      <Route path="/dashboard" element={<Dashboard darkMode={darkMode} setDarkMode={setDarkMode} />} />
      <Route path="/interview" element={<InterviewSetup darkMode={darkMode} setDarkMode={setDarkMode} />} />
      <Route path="/report" element={<Report darkMode={darkMode} setDarkMode={setDarkMode} />} />
    </Routes>
  );
}
