import React, { useEffect, useRef } from "react";

export default function VideoCapture({ onFrame }) {
  const ref = useRef(null);
  useEffect(() => {
    async function start() {
      const s = await navigator.mediaDevices.getUserMedia({ video: true });
      ref.current.srcObject = s;
      ref.current.play();
      const id = setInterval(() => {
        const canvas = document.createElement("canvas");
        canvas.width = 320; canvas.height = 240;
        canvas.getContext("2d").drawImage(ref.current, 0, 0, canvas.width, canvas.height);
        onFrame(canvas.toDataURL("image/jpeg", 0.6));
      }, 800);
      return () => { clearInterval(id); s.getTracks().forEach(t => t.stop()); };
    }
    start();
  }, [onFrame]);
  return <video ref={ref} style={{ width: 320 }} />;
}
