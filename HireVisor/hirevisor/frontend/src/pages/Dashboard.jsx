import React from "react";
import { Link } from "react-router-dom";
import { Pie, Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
} from "chart.js";

ChartJS.register(
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement
);

const Dashboard = ({ userType = "individual", userName = "User" }) => {
  // Mock Data - Replace with API later
  const stats = [
    { title: "Interviews This Month", value: 12 },
    { title: "Avg Confidence", value: "78%" },
    { title: "Top Emotion", value: "Confident" },
    { title: "AI Accuracy", value: "92%" },
  ];

  const recentInterviews = [
    { name: "Amit Sharma", date: "2025-08-08", role: "Software Engineer", confidence: "82%", emotion: "Calm" },
    { name: "Priya Patel", date: "2025-08-07", role: "Data Analyst", confidence: "75%", emotion: "Confident" },
    { name: "John Doe", date: "2025-08-06", role: "Product Manager", confidence: "80%", emotion: "Neutral" },
  ];

  const pieData = {
    labels: ["Confident", "Calm", "Nervous", "Neutral"],
    datasets: [
      {
        label: "Emotion Distribution",
        data: [45, 25, 15, 15],
        backgroundColor: ["#4CAF50", "#2196F3", "#FF5722", "#FFC107"],
      },
    ],
  };

  const lineData = {
    labels: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    datasets: [
      {
        label: "Confidence Trend",
        data: [70, 72, 75, 80, 78, 85, 82],
        borderColor: "#4CAF50",
        tension: 0.3,
        fill: false,
      },
    ],
  };

  return (
    <div className="min-h-screen bg-white dark:bg-black text-gray-900 dark:text-white px-6 py-8 font-primarylw">
      {/* Greeting */}
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold">Welcome back, {userName} 👋</h1>
        <p className="text-lg text-gray-600 dark:text-gray-400 mt-2">
          Your latest AI-powered interview insights are ready.
        </p>
      </div>

      {/* Quick Actions */}
      <div className="flex flex-wrap justify-center gap-4 mb-10">
        <Link to="/interview" className="px-6 py-3 bg-primarylw-2 text-white rounded-lg shadow hover:scale-105 transition">
          🎤 Start Interview
        </Link>
        <Link to="/upload" className="px-6 py-3 bg-gray-200 dark:bg-gray-800 rounded-lg shadow hover:scale-105 transition">
          ⬆️ Upload Recording
        </Link>
        <Link to="/reports" className="px-6 py-3 border border-primarylw-2 text-primarylw-2 rounded-lg shadow hover:scale-105 transition">
          📊 View Reports
        </Link>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
        {stats.map((stat, idx) => (
          <div key={idx} className="bg-gray-100 dark:bg-gray-900 p-6 rounded-lg shadow text-center hover:shadow-lg transition">
            <h2 className="text-2xl font-bold">{stat.value}</h2>
            <p className="text-gray-600 dark:text-gray-400">{stat.title}</p>
          </div>
        ))}
      </div>

      {/* Recent Interviews */}
      <div className="bg-gray-100 dark:bg-gray-900 p-6 rounded-lg shadow mb-10">
        <h2 className="text-2xl font-bold mb-4">📅 Recent Interviews</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-gray-300 dark:border-gray-700">
                <th className="py-2 px-4">Candidate</th>
                <th className="py-2 px-4">Date</th>
                <th className="py-2 px-4">Role</th>
                <th className="py-2 px-4">Confidence</th>
                <th className="py-2 px-4">Emotion</th>
                <th className="py-2 px-4">Action</th>
              </tr>
            </thead>
            <tbody>
              {recentInterviews.map((item, idx) => (
                <tr key={idx} className="hover:bg-gray-200 dark:hover:bg-gray-800">
                  <td className="py-2 px-4">{item.name}</td>
                  <td className="py-2 px-4">{item.date}</td>
                  <td className="py-2 px-4">{item.role}</td>
                  <td className="py-2 px-4">{item.confidence}</td>
                  <td className="py-2 px-4">{item.emotion}</td>
                  <td className="py-2 px-4">
                    <Link to="/reports" className="text-primarylw-2 hover:underline">View</Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-100 dark:bg-gray-900 p-6 rounded-lg shadow">
          <h2 className="text-xl font-bold mb-4">Emotion Distribution</h2>
          <Pie data={pieData} />
        </div>
        <div className="bg-gray-100 dark:bg-gray-900 p-6 rounded-lg shadow">
          <h2 className="text-xl font-bold mb-4">Confidence Trend</h2>
          <Line data={lineData} />
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
