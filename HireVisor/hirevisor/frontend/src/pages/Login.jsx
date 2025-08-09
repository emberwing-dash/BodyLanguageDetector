// src/pages/AuthPage.js
import React, { useState, useEffect } from "react";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";

export default function AuthPage() {
  const [darkMode, setDarkMode] = useState(false);
  const [isSignUp, setIsSignUp] = useState(false);

  useEffect(() => {
    document.body.classList.toggle("dark", darkMode);
  }, [darkMode]);

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-[#0f2027] via-[#203a43] to-[#2c5364] text-white">
      <Navbar darkMode={darkMode} setDarkMode={setDarkMode} />

      <main className="flex flex-1 items-center justify-center px-4 sm:px-6 lg:px-8">
        <div className="max-w-md w-full space-y-8 bg-white/10 backdrop-blur-lg p-8 rounded-2xl shadow-lg">
          <h2 className="text-center text-3xl font-extrabold">
            {isSignUp ? "Create your account" : "Sign in to your account"}
          </h2>
          <p className="mt-2 text-center text-sm text-gray-300">
            {isSignUp ? "Already have an account?" : "Don't have an account?"}{" "}
            <button
              className="text-blue-400 hover:underline"
              onClick={() => setIsSignUp(!isSignUp)}
            >
              {isSignUp ? "Sign In" : "Sign Up"}
            </button>
          </p>

          <form className="mt-8 space-y-6">
            {isSignUp && (
              <div>
                <label className="block text-sm font-medium">Full Name</label>
                <input
                  type="text"
                  required
                  className="mt-1 w-full px-4 py-3 rounded-lg bg-white/20 text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-400"
                  placeholder="Enter your name"
                />
              </div>
            )}

            <div>
              <label className="block text-sm font-medium">Email address</label>
              <input
                type="email"
                required
                className="mt-1 w-full px-4 py-3 rounded-lg bg-white/20 text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-400"
                placeholder="Enter your email"
              />
            </div>

            <div>
              <label className="block text-sm font-medium">Password</label>
              <input
                type="password"
                required
                className="mt-1 w-full px-4 py-3 rounded-lg bg-white/20 text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-400"
                placeholder="Enter your password"
              />
            </div>

            <button
              type="submit"
              className="w-full py-3 rounded-lg bg-blue-500 hover:bg-blue-600 transition-all text-lg font-semibold"
            >
              {isSignUp ? "Create Account" : "Sign In"}
            </button>
          </form>

          <div className="mt-6">
            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-gray-400"></div>
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="px-2 bg-transparent">Or continue with</span>
              </div>
            </div>

            <div className="mt-6 grid grid-cols-3 gap-4">
              <SocialButton provider="GitHub" />
              <SocialButton provider="Google" />
              <SocialButton provider="Microsoft" />
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}

function SocialButton({ provider }) {
  const logos = {
    GitHub: (
      <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 32 32">
        <path
          fillRule="evenodd"
          d="M16 4C9.37 4 4 9.37 4 16c0 5.3 3.44 9.8 8.21 11.39..."
        />
      </svg>
    ),
    Google: (
      <svg className="w-6 h-6" viewBox="0 0 48 48">
        <path fill="#fbc02d" d="M43.61,20.08H42V20H24v8h11.3..." />
      </svg>
    ),
    Microsoft: (
      <svg className="w-6 h-6" viewBox="0 0 48 48">
        <path fill="#0078d4" d="M42,37c0,2.76-2.24,5-5,5H11c-2.76..." />
      </svg>
    ),
  };

  return (
    <button className="flex items-center justify-center px-4 py-3 rounded-lg bg-white/20 hover:bg-white/30 transition">
      {logos[provider]}
    </button>
  );
}
