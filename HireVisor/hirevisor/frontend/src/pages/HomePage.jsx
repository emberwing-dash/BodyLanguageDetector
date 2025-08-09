import React from "react";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";

export default function HomePage({ darkMode, setDarkMode }) {
  return (
    <div className="page-container">
      <Navbar darkMode={darkMode} setDarkMode={setDarkMode} />
      <main>
        <h1>Welcome to HireVisor</h1>
        <p>Your AI-powered interview and candidate management platform.</p>
        <div style={{ marginTop: "2rem" }}>
          <a href="/login"><button className="primary-btn">Login</button></a>
          <a href="/signup" style={{ marginLeft: "1rem" }}>
            <button className="secondary-btn">Sign Up</button>
          </a>
        </div>
      </main>
      <Footer darkMode={darkMode} />
    </div>
  );
}
