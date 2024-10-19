import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import Navbar from './Navbar';
function QuestionsPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { questions: initialQuestions, collection_id, section_id } = location.state;
  const [questions, setQuestions] = useState(initialQuestions);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [userAnswer, setUserAnswer] = useState('');
  const [feedback, setFeedback] = useState('');
  const [showConfetti, setShowConfetti] = useState(false);
  const [loading, setLoading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmitAnswer = () => {
    const currentQuestion = questions[currentQuestionIndex];
    setLoading(true);
    setIsSubmitting(true);

    fetch('http://localhost:5000/get_feedback', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ userAnswer, question: currentQuestion, collection_id, section_id }),
    })
      .then((response) => response.json())
      .then((data) => {
        setLoading(false);
        setIsSubmitting(false);
        if (data) {
          setFeedback(data);
          setUserAnswer('');
        } else {
          setFeedback('No feedback received.');
          setUserAnswer('');
        }
      })
      .catch((error) => {
        console.error('Error:', error);
        setLoading(false);
        setFeedback('Failed to get feedback.');
      });
  };

  const handleMarkCorrect = () => {
    const remainingQuestions = questions.filter((_, index) => index !== currentQuestionIndex);
    setQuestions(remainingQuestions);
    if (remainingQuestions.length === 0) {
      setShowConfetti(true);
    }
    setFeedback('');
  };

  const handleMarkIncorrect = () => {
    const currentQuestion = questions[currentQuestionIndex];
    const updatedQuestions = [...questions.slice(1), currentQuestion];
    setQuestions(updatedQuestions);
    setFeedback('');
  };

  const handleRedoQuestions = () => {
    setQuestions(initialQuestions);
    setShowConfetti(false);
  };

  const handleGoHome = () => {
    navigate('/chapter');
  };

  return (
    <div className="bg-gray-900 text-gray-100 font-sans min-h-screen">
    <Navbar />
    <div className="flex items-center justify-center py-12 px-4">
      <div className="w-full max-w-3xl">
        {questions.length > 0 && (
          <div className="bg-gray-800 rounded-lg shadow-lg p-6 mb-8">
            <p className="text-xl text-cyan-400 mb-4">{questions[currentQuestionIndex]}</p>
            {!feedback && (
              <>
                <textarea
                  className="w-full bg-gray-700 text-white border border-gray-600 rounded-lg py-2 px-3 mb-4 focus:outline-none focus:border-cyan-400 h-32"
                  placeholder="Type your answer here..."
                  value={userAnswer}
                  onChange={(e) => setUserAnswer(e.target.value)}
                  disabled={isSubmitting}
                />
                <button 
                  className="w-full bg-cyan-600 hover:bg-cyan-700 text-white font-medium py-3 px-4 rounded-lg transition duration-300 ease-in-out transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
                  onClick={handleSubmitAnswer} 
                  disabled={isSubmitting}
                >
                  Submit Answer
                </button>
              </>
            )}
          </div>
        )}
        {loading && (
          <div className="bg-gray-800 rounded-lg shadow-lg p-6 mb-8">
            <div className="flex flex-col items-center justify-center">
              <div className="spinner border-t-4 border-cyan-400 border-solid rounded-full w-12 h-12 mb-4 animate-spin"></div>
              <p className="text-cyan-400">Generating feedback...</p>
            </div>
          </div>
        )}
        {feedback && !loading && (
          <div className="bg-gray-800 rounded-lg shadow-lg p-6 mb-8 flex flex-col h-96">
             <div className="flex-grow overflow-y-auto mb-4">
            <p className="text-lg text-gray-300 mb-4">{feedback}</p>
            <div className="flex justify-between">
            <button 
              className="bg-cyan-600 hover:bg-cyan-700 text-white font-medium py-2 px-4 rounded-lg transition duration-300 ease-in-out transform hover:scale-105"
              onClick={handleMarkCorrect}
            >
              Next
            </button>
            <button 
              className="bg-cyan-600 hover:bg-cyan-700 text-white font-medium py-2 px-4 rounded-lg transition duration-300 ease-in-out transform hover:scale-105"
              onClick={handleMarkIncorrect}
            >
              Review Later
            </button>
            </div>
          </div>
          </div>
        )}
        {showConfetti && (
          <div className="bg-gray-800 rounded-lg shadow-lg p-6 text-center">
            <h2 className="text-3xl font-bold text-cyan-400 mb-4">Great Job!</h2>
            <p className="text-xl text-gray-300 mb-6">All questions completed.</p>
            <div className="flex justify-center space-x-4">
              <button 
                className="bg-cyan-600 hover:bg-cyan-700 text-white font-medium py-2 px-4 rounded-lg transition duration-300 ease-in-out transform hover:scale-105"
                onClick={handleGoHome}
              >
                Back
              </button>
              <button 
                className="bg-cyan-600 hover:bg-cyan-700 text-white font-medium py-2 px-4 rounded-lg transition duration-300 ease-in-out transform hover:scale-105"
                onClick={handleRedoQuestions}
              >
                Review Questions
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
    </div>
  );
}

export default QuestionsPage;