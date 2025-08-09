import React from "react";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";

export default function CandidateInterview({ darkMode, setDarkMode }) {
  return (
    <div className="page-container">
      <Navbar darkMode={darkMode} setDarkMode={setDarkMode} />
      <main>
        <h1>Candidate Interview</h1>
        <p>Interview is in progress. Please answer the questions displayed here.</p>
      </main>
      <Footer darkMode={darkMode} />
    </div>
  );
}
