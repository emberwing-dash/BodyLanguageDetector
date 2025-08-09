import React, { useState } from "react";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";

export default function InterviewSetup({ darkMode, setDarkMode }) {
  const [topic, setTopic] = useState("");
  const [duration, setDuration] = useState(30);
  const [difficulty, setDifficulty] = useState("Easy");

  return (
    <div className="page-container">
      <Navbar darkMode={darkMode} setDarkMode={setDarkMode} />
      <main>
        <h1>Setup Interview</h1>
        <form style={{ width: "100%", maxWidth: "500px" }}>
          <input 
            type="text" 
            placeholder="Interview Topic" 
            value={topic} 
            onChange={(e) => setTopic(e.target.value)} 
          />
          <input 
            type="number" 
            placeholder="Duration (minutes)" 
            value={duration}
            onChange={(e) => setDuration(e.target.value)}
          />
          <select value={difficulty} onChange={(e) => setDifficulty(e.target.value)}>
            <option>Easy</option>
            <option>Medium</option>
            <option>Hard</option>
          </select>
          <button className="primary-btn" style={{ width: "100%" }}>Start Interview</button>
        </form>
      </main>
      <Footer darkMode={darkMode} />
    </div>
  );
}
