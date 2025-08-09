import React from "react";

export default function ReportCard({ report }) {
  return (
    <div style={{ border: "1px solid #ddd", padding: 12, margin: 8 }}>
      <h4>{report?.candidate || "Candidate"}</h4>
      <div>Avg confidence: {report?.avg_confidence ?? "-"}</div>
    </div>
  );
}
