import React, { useState } from "react";
 

const HomePage = () => {
  // Button hover state
  const [btnHover, setBtnHover] = useState(false);

  // Reusable section styles
  const sectionStyle = (bgColor) => ({
    padding: "60px 20px",
    backgroundColor: bgColor,
    textAlign: "center",
  });

  const headingStyle = {
    fontSize: "2.5rem",
    fontWeight: "bold",
    marginBottom: "20px",
  };

  const paragraphStyle = {
    fontSize: "1.1rem",
    maxWidth: "800px",
    margin: "0 auto 20px",
    lineHeight: "1.6",
  };

  const buttonStyle = {
    padding: "12px 30px",
    fontSize: "1rem",
    border: "none",
    borderRadius: "5px",
    backgroundColor: btnHover ? "#0056b3" : "#007BFF",
    color: "#fff",
    cursor: "pointer",
    transition: "background-color 0.3s ease",
  };

  const threeColGrid = {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
    gap: "20px",
    maxWidth: "1000px",
    margin: "0 auto",
  };

  const cardStyle = {
    backgroundColor: "#fff",
    padding: "20px",
    borderRadius: "8px",
    boxShadow: "0 4px 8px rgba(0,0,0,0.1)",
    textAlign: "left",
  };

  return (
    <div>

        
      Hero Section
      <section style={sectionStyle("#f8f9fa")}>
        <h1 style={headingStyle}>AI-Based human interview</h1>
        <p style={paragraphStyle}>
          Real-time analysis of micro-expressions, eye contact, and gestures to
          give candidates instant, unbiased feedback during online interviews.
        </p>
        <button
          style={buttonStyle}
          onMouseOver={() => setBtnHover(true)}
          onMouseOut={() => setBtnHover(false)}
        >
          Start Demo
        </button>
      </section>

      {/* Features Section */}
      <section style={sectionStyle("#e9ecef")}>
        <h2 style={headingStyle}>How It Works</h2>
        <div style={threeColGrid}>
          <div style={cardStyle}>
            <h3>Step 1: Understand the Words</h3>
            <p>Analyzes the transcript to understand candidate responses.</p>
          </div>
          <div style={cardStyle}>
            <h3>Step 2: Listen to the Voice</h3>
            <p>
              Detects tone, pitch, and pace to capture hidden emotions in
              speech.
            </p>
          </div>
          <div style={cardStyle}>
            <h3>Step 3: Watch the Video</h3>
            <p>
              Reads facial expressions and gestures for non-verbal cues.
            </p>
          </div>
          <div style={cardStyle}>
            <h3>Step 4: Combine Everything</h3>
            <p>
              Integrates speech, tone, and body language to predict emotions.
            </p>
          </div>
        </div>
      </section>

      {/* Why This Matters */}
      <section style={sectionStyle("#f8f9fa")}>
        <h2 style={headingStyle}>Why This Matters</h2>
        <div style={threeColGrid}>
          <div style={cardStyle}>
            <h3>What We Observe</h3>
            <p>Words, voice tone, facial expressions, and gestures.</p>
          </div>
          <div style={cardStyle}>
            <h3>What the Model Does</h3>
            <p>
              Reads, listens, and watches to detect emotions in real-time.
            </p>
          </div>
          <div style={cardStyle}>
            <h3>Why It Helps</h3>
            <p>
              Provides consistent, unbiased insights for better hiring
              decisions.
            </p>
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section style={sectionStyle("#e9ecef")}>
        <h2 style={headingStyle}>FAQ</h2>
        <div style={{ maxWidth: "800px", margin: "0 auto", textAlign: "left" }}>
          <h4>1. How do you ensure privacy?</h4>
          <p>
            All data is processed securely and never stored without consent.
          </p>
          <h4>2. How accurate is the analysis?</h4>
          <p>
            The AI model is trained on diverse datasets for high reliability.
          </p>
          <h4>3. Can bias affect results?</h4>
          <p>
            The model is designed to minimize bias by focusing on objective
            behavioral cues.
          </p>
        </div>
      </section>

      {/* Testimonials Section */}
      <section style={sectionStyle("#f8f9fa")}>
        <h2 style={headingStyle}>What People Say</h2>
        <div style={threeColGrid}>
          <div style={cardStyle}>
            <p>
              "This tool gave me confidence in my interviews and helped me
              improve my delivery."
            </p>
            <strong>- Candidate</strong>
          </div>
          <div style={cardStyle}>
            <p>
              "It’s like having a body language coach for every candidate —
              unbiased and precise."
            </p>
            <strong>- Recruiter</strong>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer
        style={{
          backgroundColor: "#343a40",
          color: "#fff",
          padding: "20px",
          textAlign: "center",
        }}
      >
        <p>© 2025 AI Body Language Analyzer | All Rights Reserved</p>
        <p>
          <a
            href="#"
            style={{ color: "#fff", textDecoration: "none", margin: "0 10px" }}
          >
            GitHub
          </a>
          |
          <a
            href="#"
            style={{ color: "#fff", textDecoration: "none", margin: "0 10px" }}
          >
            Contact
          </a>
        </p>
      </footer>
    </div>
  );
};

export default HomePage;
