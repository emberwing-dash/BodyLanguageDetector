import React, { useState } from "react";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";

export default function Signup({ darkMode, setDarkMode }) {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  return (
    <div className="page-container">
      <Navbar darkMode={darkMode} setDarkMode={setDarkMode} />
      <main>
        <h1>Sign Up</h1>
        <form style={{ width: "100%", maxWidth: "450px" }}>
          <input type="text" placeholder="Full Name" value={name} onChange={(e) => setName(e.target.value)} />
          <input type="email" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} />
          <input type="password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} />
          <button className="primary-btn" style={{ width: "100%" }}>Create Account</button>
        </form>
        <p style={{ marginTop: "1rem" }}>Already have an account? <a href="/login">Login</a></p>
      </main>
      <Footer darkMode={darkMode} />
    </div>
  );
}
