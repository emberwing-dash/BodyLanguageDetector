// src/components/ThreeDMarquee.js
import React, { useMemo } from "react";
import { motion } from "framer-motion";

/**
 * Named export ThreeDMarquee
 * Simple, inline-style version (no Tailwind) that mimics the vertical animated columns.
 * Make sure framer-motion is installed: npm install framer-motion
 */
export const ThreeDMarquee = ({ images = [], className = "", cols = 4, onImageClick }) => {
  // duplicate images for a continuous feel
  const duplicated = useMemo(() => [...images, ...images], [images]);

  const groupSize = Math.max(1, Math.ceil(duplicated.length / cols));
  const groups = Array.from({ length: cols }, (_, i) =>
    duplicated.slice(i * groupSize, (i + 1) * groupSize)
  );

  const handleImageClick = (img, idx) => {
    if (onImageClick) return onImageClick(img, idx);
    if (img.href) window.open(img.href, img.target || "_self");
  };

  // inline styles
  const container = {
    width: "100%",
    height: "600px",
    maxHeight: "600px",
    overflow: "hidden",
    borderRadius: 16,
    background: "transparent",
    perspective: "1200px",
  };

  const rotated = {
    width: "100%",
    height: "100%",
    transform: "rotateX(55deg) rotateZ(45deg)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  };

  const grid = {
    display: "grid",
    gridTemplateColumns: `repeat(${cols}, 1fr)`,
    gap: 16,
    width: "100%",
    maxWidth: 1200,
    transformOrigin: "center",
    padding: 20,
    boxSizing: "border-box",
  };

  const column = {
    display: "flex",
    flexDirection: "column",
    gap: 18,
    alignItems: "center",
    position: "relative",
  };

  const imgStyle = {
    width: "100%",
    maxWidth: 220,
    height: 160,
    objectFit: "cover",
    borderRadius: 12,
    boxShadow: "0 10px 30px rgba(2,6,23,0.12)",
    cursor: "pointer",
  };

  return (
    <section style={container} className={className}>
      <div style={rotated}>
        <div style={{ width: "100%", transform: "scale(0.95)" }}>
          <div style={grid}>
            {groups.map((groupImgs, colIdx) => {
              const even = colIdx % 2 === 0;
              return (
                <motion.div
                  key={`col-${colIdx}`}
                  style={column}
                  animate={{ y: even ? [0, 100, 0] : [0, -100, 0] }}
                  transition={{ duration: even ? 10 : 15, repeat: Infinity, ease: "linear" }}
                >
                  {groupImgs.map((img, imgIdx) => {
                    const globalIndex = colIdx * groupSize + imgIdx;
                    const clickable = Boolean(img && (img.href || onImageClick));
                    return (
                      <div key={`img-${colIdx}-${imgIdx}`} style={{ width: "100%", display: "flex", justifyContent: "center" }}>
                        <motion.img
                          whileHover={{ y: -8, scale: 1.02 }}
                          transition={{ duration: 0.25 }}
                          src={img.src}
                          alt={img.alt || `marquee-${globalIndex}`}
                          style={{ ...imgStyle, cursor: clickable ? "pointer" : "default" }}
                          onClick={() => (clickable ? handleImageClick(img, globalIndex) : null)}
                        />
                      </div>
                    );
                  })}
                </motion.div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
};
