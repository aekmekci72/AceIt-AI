import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Navbar from './Navbar';
function Home() {
  const [numQuestions, setNumQuestions] = useState(5);
  const navigate = useNavigate();

  useEffect(() => {
    fetch('https://quizme-j6kd.onrender.com/increment_counter')
      .then(response => response.text())
      .then(data => {
        // Handle the response if needed
      })
      .catch(error => {
        console.error('Error fetching visit count:', error);
      });
  }, []);

  useEffect(() => {
    const aboutSection = document.querySelector('.about-section');
    let lastScrollTop = 0;

    const handleScroll = () => {
      const sectionTop = aboutSection.getBoundingClientRect().top;
      const windowHeight = window.innerHeight;
      const currentScrollTop = window.scrollY;

      if (currentScrollTop > lastScrollTop) {
        if (sectionTop < windowHeight - 100) {
          aboutSection.classList.add('opacity-100');
          aboutSection.classList.remove('opacity-0');
        }
      } else {
        if (sectionTop > windowHeight - 100) {
          aboutSection.classList.add('opacity-0');
          aboutSection.classList.remove('opacity-100');
        }
      }
      lastScrollTop = currentScrollTop <= 0 ? 0 : currentScrollTop;
    };

    window.addEventListener('scroll', handleScroll);

    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  const getRandomQuestions = (questions, num) => {
    const shuffled = questions.sort(() => 0.5 - Math.random());
    return shuffled.slice(0, num);
  };

  const handleStartSession = () => {
    const num = Math.max(numQuestions, 5);
    fetch('http://localhost:5000/get_questions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        collection_id: localStorage.getItem('currentCollection'),
        section_id: localStorage.getItem('currentSection'),
        num: num,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data && Array.isArray(data) && data.length > 0) {
          const questionsToUse = getRandomQuestions(data, 5);
          navigate('/quizsession', {
            state: {
              questions: questionsToUse,
              collection_id: localStorage.getItem('currentCollection'),
              section_id: localStorage.getItem('currentSection'),
            },
          });
        } else {
          alert('No questions found.');
        }
      })
      .catch((error) => {
        console.error('Error:', error);
        alert('Failed to fetch questions.');
      });
  };

  return (
    <div className="bg-gray-900 text-gray-100 font-sans min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-grow flex flex-col items-center justify-center px-4 py-8">
        <header className="text-center mb-12">
          <h1 className="text-7xl md:text-8xl font-bold mb-6 text-cyan-400">QuizMe</h1>
          <p className="text-2xl text-gray-300">Your AI companion for test readiness and confidence</p>
        </header>
        <div className="flex justify-center w-full">
          <div className="max-w-md w-full bg-gray-800 rounded-lg shadow-lg p-8">
            <div className="space-y-6">
              <div>
                <label htmlFor="numQuestions" className="block text-sm font-medium text-gray-300 mb-2">
                  Number of Questions
                </label>
                <input
                  type="number"
                  id="numQuestions"
                  className="w-full bg-gray-700 text-white border border-gray-600 rounded-lg py-2 px-3 focus:outline-none focus:border-cyan-400"
                  min="1"
                  value={numQuestions}
                  onChange={(e) => setNumQuestions(e.target.value)}
                />
              </div>
              <button
                className="w-full bg-cyan-600 hover:bg-cyan-700 text-white font-medium py-3 px-4 rounded-lg transition duration-300 ease-in-out transform hover:scale-105"
                onClick={handleStartSession}
              >
                Start Test Prep Session
              </button>
            </div>
          </div>
        </div>
      </main>
      <div className="about-section opacity-0 transition-opacity duration-500 ease-in-out bg-gray-800 py-16">
        <div className="container mx-auto px-4">
          <h2 className="text-4xl font-bold mb-8 text-cyan-400">About QuizMe</h2>
          <p className="text-lg text-gray-300 mb-6">
            QuizMe is your intelligent study companion, designed to enhance your learning experience and boost your confidence for upcoming tests and exams.
          </p>
          <p className="text-lg text-gray-300">
            With our AI-powered question generation and adaptive learning algorithms, QuizMe tailors each session to your unique needs, helping you focus on areas that need improvement while reinforcing your strengths.
          </p>
        </div>
      </div>
    </div>
  );
}

export default Home;