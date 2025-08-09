import React from "react";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";

export default function LandingPage({ darkMode, setDarkMode }) {
  return (
    <div className="page-container">
      <Navbar darkMode={darkMode} setDarkMode={setDarkMode} />
      <main>
        <h1>About HireVisor</h1>
        <p>Streamline hiring with smart interview tools, automated reports, and real-time candidate analysis.</p>
      </main>
      <Footer darkMode={darkMode} />
    </div>
  );
}
