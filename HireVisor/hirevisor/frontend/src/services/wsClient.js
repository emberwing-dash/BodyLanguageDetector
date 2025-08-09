const wsClient = {
    connect(path = "/ws/interview/demo") {
      const base = (window.location.hostname === "localhost") ? "ws://localhost:8000" : `wss://${window.location.hostname}`;
      const ws = new WebSocket(base + path);
      ws.onopen = () => console.log("ws open");
      ws.onclose = () => console.log("ws closed");
      ws.onerror = (e) => console.error("ws error", e);
      return ws;
    }
  };
  
  export default wsClient;
  