import React from "react";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";

export default function Report({ darkMode, setDarkMode }) {
  return (
    <div className="page-container">
      <Navbar darkMode={darkMode} setDarkMode={setDarkMode} />
      <main>
        <h1>Reports</h1>
        <p>Access detailed candidate reports and insights here.</p>
      </main>
      <Footer darkMode={darkMode} />
    </div>
  );
}
