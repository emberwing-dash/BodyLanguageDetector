import React from "react";

export default function Header() {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
      <img src="/logo192.png" alt="logo" style={{ height: 40 }} />
      <div>
        <strong>HireVisor</strong><br />
        <small>Decode Emotions. Enhance Hiring.</small>
      </div>
    </div>
  );
}
