import React from "react";

export default function LiveFeedback({ feedback }) {
  const style = { padding: 12, marginTop: 12, border: "1px solid #e6e6e6", width: 480 };
  if (!feedback) return <div style={style}>No feedback yet</div>;
  return (
    <div style={style}>
      <h4>Live Feedback</h4>
      <div><b>Emotion:</b> {feedback.emotion}</div>
      <div><b>Eye contact:</b> {feedback.eye_contact}</div>
      <div><b>Confidence:</b> {Math.round((feedback.confidence||0)*100)}%</div>
      <div style={{ marginTop: 8 }}><b>Tip:</b> {feedback.tip}</div>
    </div>
  );
}
