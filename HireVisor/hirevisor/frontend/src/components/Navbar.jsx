import React from "react";
import { Link } from "react-router-dom";

export default function Navbar({ darkMode, setDarkMode }) {
  return (
    <nav className={`navbar ${!darkMode ? "light-mode" : ""}`}>
      <div><Link to="/">HireVisor</Link></div>
      <div>
        <Link to="/dashboard">Dashboard</Link>
        <Link to="/interview">Interview</Link>
        <Link to="/report">Report</Link>
        <Link to="/login">Login</Link>
        <button onClick={() => setDarkMode(!darkMode)} style={{ marginLeft: "1rem" }}>
          {darkMode ? "☀️" : "🌙"}
        </button>
      </div>
    </nav>
  );
}
