import React, { useState } from "react";

const Navbar = () => {
  const [menuOpen, setMenuOpen] = useState(false);

  const navStyle = {
    position: "fixed",
    top: 0,
    left: 0,
    width: "100%",
    backgroundColor: "#343a40",
    color: "#fff",
    padding: "10px 20px",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    zIndex: 1000,
  };

  const linkContainerStyle = {
    display: "flex",
    gap: "20px",
  };

  const linkStyle = {
    color: "#fff",
    textDecoration: "none",
    fontSize: "16px",
    transition: "color 0.3s",
  };

  const hamburgerStyle = {
    display: "none",
    flexDirection: "column",
    cursor: "pointer",
    gap: "4px",
  };

  const barStyle = {
    width: "25px",
    height: "3px",
    backgroundColor: "#fff",
    transition: "0.3s",
  };

  const mobileMenuStyle = {
    display: "flex",
    flexDirection: "column",
    position: "absolute",
    top: "50px",
    right: "20px",
    backgroundColor: "#343a40",
    padding: "10px",
    borderRadius: "5px",
    gap: "10px",
  };

  const handleHover = (e, isHover) => {
    e.target.style.color = isHover ? "#00d1ff" : "#fff";
  };

  return (
    <nav style={navStyle}>
      {/* Logo / Project Name */}
      <div style={{ fontSize: "20px", fontWeight: "bold" }}>HierVisor</div>

      {/* Desktop Links */}
      <div
        style={{
          ...linkContainerStyle,
          display: window.innerWidth > 768 ? "flex" : "none",
        }}
      >
        {["Home", "Features", "FAQ", "Testimonials", "Contact"].map((item) => (
          <a
            key={item}
            href={`#${item.toLowerCase()}`}
            style={linkStyle}
            onMouseOver={(e) => handleHover(e, true)}
            onMouseOut={(e) => handleHover(e, false)}
          >
            {item}
          </a>
        ))}
      </div>

      {/* Mobile Hamburger */}
      <div
        style={{
          ...hamburgerStyle,
          display: window.innerWidth <= 768 ? "flex" : "none",
        }}
        onClick={() => setMenuOpen(!menuOpen)}
      >
        <div style={barStyle}></div>
        <div style={barStyle}></div>
        <div style={barStyle}></div>
      </div>

      {/* Mobile Menu */}
      {menuOpen && window.innerWidth <= 768 && (
        <div style={mobileMenuStyle}>
          {["Home", "Features", "FAQ", "Testimonials", "Contact"].map((item) => (
            <a
              key={item}
              href={`#${item.toLowerCase()}`}
              style={linkStyle}
              onMouseOver={(e) => handleHover(e, true)}
              onMouseOut={(e) => handleHover(e, false)}
            >
              {item}
            </a>
          ))}
        </div>
      )}
    </nav>
  );
};

export default Navbar;
