import axios from "axios";
const base = process.env.REACT_APP_API_BASE || "http://localhost:8000";
export default {
  getReports: () => axios.get(`${base}/reports`).then(r => r.data),
  // add more REST calls as needed
};
