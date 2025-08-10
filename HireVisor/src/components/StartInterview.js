import React, { useState, useRef, useEffect } from "react";

/**
 * StartInterview.js
 * HierVisor - AI-Based Body Language Analyzer (Demo / Mock)
 *
 * Inline styles only. No external libraries.
 */

export default function StartInterview() {
  // media
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  // UI & responsive
  const [isRunning, setIsRunning] = useState(false);
  const [isMobile, setIsMobile] = useState(
    typeof window !== "undefined" ? window.innerWidth <= 900 : false
  );

  // Hover states for buttons
  const [hoverStart, setHoverStart] = useState(false);
  const [hoverStop, setHoverStop] = useState(false);
  const [hoverSave, setHoverSave] = useState(false);

  // Simulated analysis state
  const [emotion, setEmotion] = useState({
    Happiness: 0.6,
    Nervous: 0.2,
    Neutral: 0.2,
  });
  const [eyeContact, setEyeContact] = useState(78); // percent
  const [voiceTone, setVoiceTone] = useState("Confident"); // Confident/Hesitant/Neutral
  const [gestures, setGestures] = useState({
    fidgeting: false,
    handMovement: true,
  });
  const [confidenceScore, setConfidenceScore] = useState(72); // 0-100

  const [feedback, setFeedback] = useState("Waiting for analysis...");
  const feedbackPool = [
    "Good eye contact during answers.",
    "Voice sounds confident — keep the pace steady.",
    "Slight fidgeting detected — try resting hands.",
    "Smiled at key moments — positive signal.",
    "Tone fluctuates — take a breath to steady voice.",
    "Excellent posture and steady eye contact.",
  ];

  // intervals refs for cleanup
  const simIntervalRef = useRef(null);
  const feedbackIntervalRef = useRef(null);

  // Responsive handler
  useEffect(() => {
    function onResize() {
      setIsMobile(window.innerWidth <= 900);
    }
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // Start webcam & simulated analysis
  const startInterview = async () => {
    if (isRunning) return;
    try {
      const constraints = { video: { width: 1280, height: 720 }, audio: true };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play().catch(() => {});
      }
      setIsRunning(true);

      // kick off simulated AI updates if no backend
      startSimulation();
      startFeedbackRotation();
    } catch (err) {
      console.error("Error accessing media devices:", err);
      alert("Unable to access camera/microphone. Please grant permissions and try again.");
    }
  };

  // Stop webcam & simulation
  const stopInterview = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      try {
        videoRef.current.pause();
        videoRef.current.srcObject = null;
      } catch (e) {}
    }
    setIsRunning(false);
    stopSimulation();
    stopFeedbackRotation();
    setFeedback("Analysis stopped.");
  };

  // Save report JSON (mock)
  const saveReport = () => {
    const report = {
      timestamp: new Date().toISOString(),
      emotion,
      eyeContact,
      voiceTone,
      gestures,
      confidenceScore,
      feedback,
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `hiervisor_report_${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  // Simulation: random-ish updates to mimic real model outputs
  const startSimulation = () => {
    stopSimulation(); // ensure no duplicate intervals
    simIntervalRef.current = setInterval(() => {
      // random small variations
      const noise = () => (Math.random() - 0.5) * 0.06;

      let h = Math.max(0, Math.min(1, emotion.Happiness + noise()));
      let n = Math.max(0, Math.min(1, emotion.Nervous + noise()));
      let neutral = Math.max(0, Math.min(1, 1 - (h + n)));
      // re-normalize sum to 1
      const sum = h + n + neutral || 1;
      h /= sum;
      n /= sum;
      neutral /= sum;

      setEmotion({ Happiness: h, Nervous: n, Neutral: neutral });

      // eye contact drifts slightly
      setEyeContact((prev) => Math.max(10, Math.min(99, Math.round(prev + (Math.random() - 0.5) * 6))));

      // voice tone toggles between options occasionally
      const toneRand = Math.random();
      setVoiceTone(toneRand > 0.85 ? "Hesitant" : toneRand < 0.15 ? "Confident" : "Neutral");

      // gestures booleans via probability influenced by nervousness
      setGestures({
        fidgeting: Math.random() < 0.25 + n * 0.5, // more nervous increases fidget probability
        handMovement: Math.random() < 0.6,
      });

      // confidence score inversely linked to nervousness and eye contact
      setConfidenceScore((prev) => {
        const base = Math.round((h * 60 + (eyeContact / 100) * 30 + (1 - n) * 10) + (Math.random() - 0.5) * 8);
        return Math.max(0, Math.min(100, base));
      });
    }, 900); // ~1Hz updates
  };

  const stopSimulation = () => {
    if (simIntervalRef.current) {
      clearInterval(simIntervalRef.current);
      simIntervalRef.current = null;
    }
  };

  const startFeedbackRotation = () => {
    stopFeedbackRotation();
    feedbackIntervalRef.current = setInterval(() => {
      const idx = Math.floor(Math.random() * feedbackPool.length);
      setFeedback(feedbackPool[idx]);
    }, 3500);
  };

  const stopFeedbackRotation = () => {
    if (feedbackIntervalRef.current) {
      clearInterval(feedbackIntervalRef.current);
      feedbackIntervalRef.current = null;
    }
  };

  // cleanup on unmount
  useEffect(() => {
    return () => {
      stopInterview();
    };
    // eslint-disable-next-line
  }, []);

  // ----- Inline styles -----
  const pageStyle = {
    fontFamily: "'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial",
    background: "#f5f7fb",
    minHeight: "100vh",
    padding: isMobile ? "80px 12px 40px" : "100px 40px 40px",
    boxSizing: "border-box",
  };

  const headerStyle = {
    maxWidth: 1100,
    margin: "0 auto 24px",
    textAlign: "center",
  };

  const titleStyle = {
    fontSize: isMobile ? "1.6rem" : "2rem",
    fontWeight: 700,
    margin: 0,
    color: "#0f1724",
  };

  const subtitleStyle = {
    marginTop: 10,
    color: "#334155",
    fontSize: isMobile ? "0.95rem" : "1.05rem",
  };

  const layoutStyle = {
    maxWidth: 1100,
    margin: "20px auto 0",
    display: "flex",
    gap: 20,
    flexDirection: isMobile ? "column" : "row",
    alignItems: "flex-start",
    justifyContent: "center",
  };

  // Left column (video)
  const leftStyle = {
    flex: isMobile ? "unset" : "1 1 60%",
    background: "#fff",
    borderRadius: 12,
    padding: 16,
    boxShadow: "0 8px 20px rgba(15,23,42,0.06)",
    minHeight: 360,
    display: "flex",
    flexDirection: "column",
    gap: 12,
  };

  const videoWrapperStyle = {
    flex: 1,
    minHeight: 220,
    borderRadius: 8,
    overflow: "hidden",
    background: "#000",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  };

  const videoStyle = {
    width: "100%",
    height: "100%",
    objectFit: "cover",
    display: "block",
  };

  const controlRowStyle = {
    display: "flex",
    gap: 12,
    alignItems: "center",
    marginTop: 6,
    flexWrap: "wrap",
  };

  const buttonBase = {
    padding: "10px 16px",
    borderRadius: 8,
    border: "none",
    cursor: "pointer",
    fontWeight: 600,
    boxShadow: "0 6px 12px rgba(2,6,23,0.06)",
  };

  const startButtonStyle = {
    ...buttonBase,
    background: hoverStart ? "#0ea5a4" : "#06b6d4",
    color: "#fff",
  };

  const stopButtonStyle = {
    ...buttonBase,
    background: hoverStop ? "#ef4444" : "#f43f5e",
    color: "#fff",
  };

  const saveButtonStyle = {
    ...buttonBase,
    background: hoverSave ? "#374151" : "#111827",
    color: "#fff",
  };

  // Right column (analysis)
  const rightStyle = {
    flex: isMobile ? "unset" : "1 1 40%",
    background: "#fff",
    borderRadius: 12,
    padding: 16,
    boxShadow: "0 8px 20px rgba(15,23,42,0.06)",
    minHeight: 360,
    display: "flex",
    flexDirection: "column",
    gap: 12,
  };

  const panelTitleStyle = {
    margin: 0,
    fontWeight: 700,
    color: "#0f1724",
    fontSize: "1.05rem",
  };

  const statRowStyle = {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: 10,
    marginTop: 8,
  };

  const smallLabelStyle = { color: "#64748b", fontSize: "0.9rem" };

  // emotion bars container
  const emotionBarsContainer = {
    display: "flex",
    flexDirection: "column",
    gap: 8,
    marginTop: 8,
  };

  const barOuter = {
    background: "#f1f5f9",
    borderRadius: 8,
    height: 14,
    overflow: "hidden",
  };

  const barInner = (pct, color) => ({
    width: `${Math.round(pct * 100)}%`,
    height: "100%",
    background: color,
    transition: "width 0.6s ease",
  });

  // simple donut (emotion breakdown) via SVG: draw Happiness as slice
  const donutSize = 90;
  const donutRadius = 36;
  const circumference = 2 * Math.PI * donutRadius;
  const happinessPct = emotion.Happiness;
  const nervousPct = emotion.Nervous;
  const neutralPct = emotion.Neutral;
  const dashH = circumference * happinessPct;
  const dashN = circumference * nervousPct;
  // We'll show happiness slice, then nervous slice by overlaying strokes.

  // confidence bar
  const confidenceOuter = { background: "#eef2ff", borderRadius: 8, height: 18, overflow: "hidden" };
  const confidenceInner = (v) => ({
    width: `${Math.round(v)}%`,
    height: "100%",
    background: `linear-gradient(90deg,#10b981,#06b6d4)`,
    transition: "width 0.6s ease",
  });

  // ---------------------------
  // JSX
  // ---------------------------
  return (
    <div style={pageStyle}>
      <header style={headerStyle}>
        <h1 style={titleStyle}>HierVisor — AI Interview Analyzer</h1>
        <p style={subtitleStyle}>
          Your camera and microphone will be used to analyze your expressions, tone, and gestures in real-time.
        </p>
      </header>

      <main style={layoutStyle}>
        {/* LEFT: Camera & Controls */}
        <section style={leftStyle}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <h3 style={{ margin: 0, fontSize: "1.05rem", color: "#0f1724" }}>Live Camera</h3>
            <div style={{ color: "#64748b", fontSize: "0.9rem" }}>{isRunning ? "Analyzing..." : "Idle"}</div>
          </div>

          <div style={videoWrapperStyle}>
            {!isRunning && (
              <div style={{ color: "#94a3b8", textAlign: "center" }}>
                <div style={{ fontSize: 28 }}>📷</div>
                <div style={{ marginTop: 8 }}>Camera preview will appear here</div>
              </div>
            )}

            <video
              ref={videoRef}
              style={videoStyle}
              playsInline
              muted
              autoPlay
            />
          </div>

          <div style={controlRowStyle}>
            <button
              style={startButtonStyle}
              onClick={startInterview}
              onMouseOver={() => setHoverStart(true)}
              onMouseOut={() => setHoverStart(false)}
              aria-pressed={isRunning}
            >
              Start Interview
            </button>

            <button
              style={stopButtonStyle}
              onClick={stopInterview}
              onMouseOver={() => setHoverStop(true)}
              onMouseOut={() => setHoverStop(false)}
            >
              Stop Interview
            </button>

            <button
              style={saveButtonStyle}
              onClick={saveReport}
              onMouseOver={() => setHoverSave(true)}
              onMouseOut={() => setHoverSave(false)}
            >
              Save Report
            </button>

            <div style={{ marginLeft: "auto", color: "#475569", fontSize: "0.9rem" }}>
              Tip: Allow camera & mic for best demo experience
            </div>
          </div>
        </section>

        {/* RIGHT: Analysis Panel */}
        <aside style={rightStyle}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <h3 style={panelTitleStyle}>Real-time Analysis</h3>
            <div style={{ color: "#94a3b8", fontSize: "0.85rem" }}>Updated live</div>
          </div>

          {/* Emotion Donut + Bars */}
          <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
            <div style={{ width: 110, textAlign: "center" }}>
              <svg width={donutSize} height={donutSize} viewBox={`0 0 ${donutSize} ${donutSize}`} style={{ transform: "rotate(-90deg)" }}>
                {/* background ring */}
                <circle cx={donutSize/2} cy={donutSize/2} r={donutRadius} fill="transparent" stroke="#eef2ff" strokeWidth="12" />
                {/* happiness slice */}
                <circle
                  cx={donutSize/2}
                  cy={donutSize/2}
                  r={donutRadius}
                  fill="transparent"
                  stroke="#10b981"
                  strokeWidth="12"
                  strokeDasharray={`${dashH} ${circumference - dashH}`}
                  strokeLinecap="round"
                />
                {/* nervous slice (drawn on top) */}
                <circle
                  cx={donutSize/2}
                  cy={donutSize/2}
                  r={donutRadius}
                  fill="transparent"
                  stroke="#f97316"
                  strokeWidth="12"
                  strokeDasharray={`${dashN} ${circumference - dashN}`}
                  strokeDashoffset={-dashH}
                  strokeLinecap="butt"
                />
                {/* center label */}
                <g style={{ transform: "rotate(90deg)", transformOrigin: "center" }}>
                  {/* rotate back for readable text */}
                </g>
              </svg>
              <div style={{ marginTop: 8, color: "#0f1724", fontWeight: 700 }}>
                {Math.round(emotion.Happiness * 100)}% Positive
              </div>
              <div style={{ color: "#64748b", fontSize: "0.85rem" }}>Emotion balance</div>
            </div>

            <div style={{ flex: 1 }}>
              <div style={emotionBarsContainer}>
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <div style={{ fontWeight: 600, color: "#0f1724" }}>Happiness</div>
                    <div style={{ color: "#64748b" }}>{Math.round(emotion.Happiness * 100)}%</div>
                  </div>
                  <div style={barOuter}><div style={barInner(emotion.Happiness, "#10b981")}></div></div>
                </div>

                <div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <div style={{ fontWeight: 600, color: "#0f1724" }}>Nervous</div>
                    <div style={{ color: "#64748b" }}>{Math.round(emotion.Nervous * 100)}%</div>
                  </div>
                  <div style={barOuter}><div style={barInner(emotion.Nervous, "#f97316")}></div></div>
                </div>

                <div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <div style={{ fontWeight: 600, color: "#0f1724" }}>Neutral</div>
                    <div style={{ color: "#64748b" }}>{Math.round(emotion.Neutral * 100)}%</div>
                  </div>
                  <div style={barOuter}><div style={barInner(emotion.Neutral, "#94a3b8")}></div></div>
                </div>
              </div>
            </div>
          </div>

          {/* Eye contact & voice */}
          <div style={{ marginTop: 12 }}>
            <div style={statRowStyle}>
              <div>
                <div style={{ ...smallLabelStyle }}>Eye Contact</div>
                <div style={{ fontWeight: 700, color: "#0f1724" }}>{eyeContact}%</div>
              </div>
              <div>
                <div style={{ ...smallLabelStyle }}>Voice Tone</div>
                <div style={{ fontWeight: 700, color: "#0f1724" }}>{voiceTone}</div>
              </div>
              <div>
                <div style={{ ...smallLabelStyle }}>Gestures</div>
                <div style={{ fontWeight: 700, color: "#0f1724" }}>
                  {gestures.fidgeting ? "Fidgeting" : "Stable"} • {gestures.handMovement ? "Hands moving" : "Hands still"}
                </div>
              </div>
            </div>
          </div>

          {/* Confidence bar */}
          <div style={{ marginTop: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div style={{ fontWeight: 700, color: "#0f1724" }}>Confidence Score</div>
              <div style={{ color: "#64748b" }}>{confidenceScore}%</div>
            </div>
            <div style={{ marginTop: 8, ...confidenceOuter }}>
              <div style={confidenceInner(confidenceScore)} />
            </div>
          </div>

          {/* Feedback summary */}
          <div style={{ marginTop: 12 }}>
            <div style={{ fontWeight: 700, color: "#0f1724" }}>Feedback</div>
            <div style={{ marginTop: 6, color: "#334155" }}>{feedback}</div>
          </div>
        </aside>
      </main>

      {/* small footer */}
      <footer style={{ maxWidth: 1100, margin: "32px auto 12px", textAlign: "center", color: "#64748b" }}>
        <div>HierVisor • AI Body Language Analyzer — Demo Mode</div>
        <div style={{ marginTop: 6, fontSize: "0.9rem" }}>© {new Date().getFullYear()} HierVisor</div>
      </footer>
    </div>
  );
}
