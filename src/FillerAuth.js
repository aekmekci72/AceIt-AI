import React, { useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';

const LandingPage = () => {
  const featuresRef = useRef(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('opacity-100', 'translate-y-0');
          entry.target.classList.remove('opacity-0', 'translate-y-10');
        } else {
          entry.target.classList.remove('opacity-100', 'translate-y-0');
          entry.target.classList.add('opacity-0', 'translate-y-10');
        }
      },
      { threshold: 0.1 }
    );

    if (featuresRef.current) {
      observer.observe(featuresRef.current);
    }

    return () => observer.disconnect();
  }, []);

  return (
    <div className="bg-gray-900 text-gray-100 font-sans">
      <div className="min-h-screen flex flex-col items-center justify-center relative">
  <div className="text-center px-4">
    <img 
      src="./logo (2).png" 
      alt="AceIt Logo" 
      className="w-64 h-64 mx-auto mb-2 filter cyan-filter brightness-105 contrast-110"
    />
    <p className="text-xl md:text-2xl mb-8 text-gray-300">Welcome to AceIt AI, your smart study companion.</p>
    <div className="flex justify-center space-x-4">
  
            <Link 
              to="/login" 
              className="bg-cyan-600 hover:bg-cyan-700 text-white font-medium py-2 px-4 rounded-md transition duration-300 ease-in-out transform hover:scale-105"
            >
              Login
            </Link>
            <Link 
              to="/signup" 
              className="bg-cyan-600 hover:bg-cyan-700 text-white font-medium py-2 px-4 rounded-md transition duration-300 ease-in-out transform hover:scale-105"
            >
              Sign Up
            </Link>
          </div>
        </div>
      </div>

      <div ref={featuresRef} className="py-16 transition-all duration-1000 ease-out transform opacity-0 translate-y-10">
        <div className="flex flex-wrap justify-center">
          <h2 className="text-4xl md:text-5xl font-bold mb-12 text-cyan-300">Key Features</h2>
        </div>
        <div className="container mx-auto px-4 flex flex-wrap justify-center gap-8">
          {[
            { title: "Personalized Study Plans", description: "Get customized study plans tailored to your learning needs and goals." },
            { title: "Interactive Flashcards", description: "Create and review flashcards to reinforce your knowledge effectively." },
            { title: "Progress Tracking", description: "Monitor your progress and stay on top of your study schedule." }
          ].map((feature, index) => (
            <div key={index} className="bg-gray-800 p-6 rounded-lg shadow-lg hover:shadow-cyan-500/50 transition duration-300 w-72 h-48">
              <h3 className="text-lg font-semibold mb-2 text-cyan-400">{feature.title}</h3>
              <p className="text-gray-400 text-sm">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default LandingPage;
