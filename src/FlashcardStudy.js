import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Navbar from './Navbar';

const FlashcardStudy = () => {
  const [flashcards, setFlashcards] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isFlipped, setIsFlipped] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    fetchFlashcards();
  }, []);

  const fetchFlashcards = async () => {
    try {
      const response = await fetch(`http://localhost:5000/get_flashcards?collection_id=${localStorage.getItem('currentCollection')}&section_id=${localStorage.getItem('currentSection')}`);
      if (!response.ok) {
        throw new Error('Failed to fetch flashcards');
      }
      const data = await response.json();
      setFlashcards(data.flashcards);
    } catch (error) {
      console.error('Error fetching flashcards:', error.message);
    }
  };

  const handleNext = () => {
    if (currentIndex < flashcards.length - 1) {
      setCurrentIndex(currentIndex + 1);
      setIsFlipped(false);
    } else {
      handleGoBack();
    }
  };

  const handlePrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
      setIsFlipped(false);
    }
  };

  const toggleFlip = () => {
    setIsFlipped(!isFlipped);
  };

  const handleGoBack = () => {
    navigate('/flashcards');
  };

  if (flashcards.length === 0) {
    return <div className="text-center text-gray-300 text-xl mt-10">Loading flashcards...</div>;
  }

  const isLastCard = currentIndex === flashcards.length - 1;

  return (
    <div className="bg-gray-900 text-gray-100 font-sans min-h-screen overflow-y-auto">
      <Navbar />
      <div className="container mx-auto px-4 py-8 flex flex-col items-center">
        <h1 className="text-6xl md:text-7xl font-bold mb-12 text-cyan-400 text-center">Study Flashcards</h1>
        
        <div 
          onClick={toggleFlip}
          className="w-full max-w-md h-64 perspective-1000 cursor-pointer mb-8"
        >
          <div className={`relative w-full h-full transition-transform duration-600 transform ${isFlipped ? 'rotate-y-180' : ''}`} style={{ transformStyle: 'preserve-3d' }}>
            <div className="absolute w-full h-full backface-hidden bg-gray-800 rounded-lg shadow-lg hover:shadow-cyan-500/50 flex items-center justify-center p-6">
              <h3 className="text-2xl font-semibold text-cyan-400">{flashcards[currentIndex].question}</h3>
            </div>
            <div className="absolute w-full h-full backface-hidden bg-gray-800 rounded-lg shadow-lg hover:shadow-cyan-500/50 flex items-center justify-center p-6 rotate-y-180">
              <p className="text-xl text-gray-300">{flashcards[currentIndex].answer}</p>
            </div>
          </div>
        </div>

        <div className="flex justify-center space-x-4">
          {currentIndex > 0 && (
            <button
              onClick={handlePrevious}
              className="bg-cyan-600 hover:bg-cyan-700 text-white font-medium py-3 px-6 rounded-lg transition duration-300 ease-in-out transform hover:scale-105"
            >
              Previous
            </button>
          )}
          <button
            onClick={handleNext}
            className="bg-cyan-600 hover:bg-cyan-700 text-white font-medium py-3 px-6 rounded-lg transition duration-300 ease-in-out transform hover:scale-105"
          >
            {isLastCard ? 'Back to Flashcards' : 'Next'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default FlashcardStudy;