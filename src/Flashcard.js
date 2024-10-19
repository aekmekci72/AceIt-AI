

import React, { useState, useEffect } from 'react';
import Navbar from './Navbar';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const FlashcardApp = () => {
  const [flashcards, setFlashcards] = useState([]);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [suggestedFlashcards, setSuggestedFlashcards] = useState([]);
  const [showModal, setShowModal] = useState(false);
  const [flippedCards, setFlippedCards] = useState([]);
  const [startTime, setStartTime] = useState(null);
  const [endTime, setEndTime] = useState(null);
  const [totalTimeSpent, setTotalTimeSpent] = useState(0);
  const chapterId = localStorage.getItem('currentSection');
  const collName = localStorage.getItem('collectionName');
  const chapterName = localStorage.getItem('currentSectionName');
  const collectionId = localStorage.getItem('currentCollection');
  const [suggestedFlippedCards, setSuggestedFlippedCards] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    let startTime = null;
    let endTime = null;
    async function fetchFlashcards() {
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
    }

    const suggestFlashcards = async () => {
      try {
          const response = await fetch('http://localhost:5000/suggestflashcards', {
              method: 'POST',
              headers: {
                  'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                  collectionId: localStorage.getItem('currentCollection'),
                  sectionId: localStorage.getItem('currentSection'),
              }),
          });
          if (!response.ok) {
              throw new Error('Failed to suggest flashcards');
          }
          const data = await response.json();
          
          // Filter out flashcards that are already in the user's flashcards based on question
          const existingQuestions = flashcards.map(flashcard => flashcard.question);
          const filteredSuggestions = data.response.filter(suggestion => 
              !existingQuestions.includes(suggestion.question)
          );
  
          // Take only the first three elements of the filtered array
          setSuggestedFlashcards(filteredSuggestions);
      } catch (error) {
          console.error('Error suggesting flashcards:', error.message);
      }
  };
  
    const handleBeforeUnload = () => {
      endTime = new Date().getTime();
      if (startTime && endTime) {
        const timeSpent = endTime - startTime;
        setTotalTimeSpent(timeSpent); // Update state for display or debugging purposes
        try {
          axios.post('http://localhost:5000/time_spent', {
            collection_id: collectionId,
            section_id: chapterId,
            total_time_spent: timeSpent,
          });
        } catch (error) {
          console.error('Error updating time spent:', error);
        }
      }
    };
    
  
    const handleUnload = () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      window.removeEventListener('unload', handleUnload);
      endTime = new Date().getTime();
      if (startTime && endTime) {
        const timeSpent = endTime - startTime;
        try {
          axios.post('http://localhost:5000/time_spent', {
            collection_id: collectionId,
            section_id: chapterId,
            total_time_spent: timeSpent,
          });
        } catch (error) {
          console.error('Error updating time spent:', error);
        }
      }
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    window.addEventListener('unload', handleUnload);
  
    startTime = new Date().getTime();
    
    
    fetchFlashcards();
    suggestFlashcards();
    return () => {
      handleBeforeUnload();
    };
  }, [chapterId, collectionId]);

  const addFlashcard = async () => {
    if (question.trim() === '' || answer.trim() === '') {
      alert('Please enter both question and answer.');
      return;
    }

    setFlashcards([...flashcards, { question, answer }]);
    setQuestion('');
    setAnswer('');

    try {
      const response = await fetch('http://localhost:5000/addflashcard', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          collectionId: localStorage.getItem('currentCollection'),
          sectionId: localStorage.getItem('currentSection'),
          question: question,
          answer: answer,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to add flashcard');
      }

      setShowModal(false);
      console.log('Flashcard added successfully!');
    } catch (error) {
      console.error('Error adding flashcard:', error.message);
    }
  };

  const addSuggestedFlashcard = async (question, answer, index) => {
    try {
      const response = await fetch('http://localhost:5000/addflashcard', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          collectionId: localStorage.getItem('currentCollection'),
          sectionId: localStorage.getItem('currentSection'),
          question: question,
          answer: answer,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to add flashcard');
      }

      const updatedSuggestions = [...suggestedFlashcards];
      updatedSuggestions.splice(index, 1);
      setSuggestedFlashcards(updatedSuggestions);

      console.log('Flashcard added successfully!');
    } catch (error) {
      console.error('Error adding flashcard:', error.message);
    }
  };

  const toggleFlip = (index, isSuggested = false) => {
    if (isSuggested) {
      setSuggestedFlippedCards((prevFlippedCards) => {
        const newFlippedCards = [...prevFlippedCards];
        newFlippedCards[index] = !newFlippedCards[index];
        return newFlippedCards;
      });
    } else {
      setFlippedCards((prevFlippedCards) => {
        const newFlippedCards = [...prevFlippedCards];
        newFlippedCards[index] = !newFlippedCards[index];
        return newFlippedCards;
      });
    }
  };
  const handleStudyFlashcards = () => {
    navigate('/flashcardstudy');
  };


  return (
    <div className="bg-gray-900 text-gray-100 font-sans min-h-screen overflow-y-auto">
      <Navbar />
      <div className="container mx-auto px-4 py-8 flex flex-col items-center">
      <h1 className="text-6xl md:text-7xl font-bold mb-12 text-cyan-400 text-center">Flashcards</h1>
        
        <div className="w-full max-w-md mb-8 space-y-4">
          <button
            onClick={() => setShowModal(true)}
            className="w-full bg-cyan-600 hover:bg-cyan-700 text-white font-medium py-3 px-4 rounded-lg transition duration-300 ease-in-out transform hover:scale-105"
          >
            Add Flashcard
          </button>
          <button
            onClick={handleStudyFlashcards}
            className="w-full bg-cyan-600 hover:bg-cyan-700 text-white font-medium py-3 px-4 rounded-lg transition duration-300 ease-in-out transform hover:scale-105"
          >
            Study Flashcards
          </button>
        </div>

        <div className="w-full">
          <h2 className="text-3xl font-bold mb-6 text-cyan-300">Your Flashcards</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {flashcards.map((flashcard, index) => (
              <div 
                key={index} 
                className="bg-gray-800 rounded-lg shadow-lg hover:shadow-cyan-500/50 transition duration-300 cursor-pointer h-48 perspective-1000"
                onClick={() => toggleFlip(index)}
              >
                <div className={`relative w-full h-full transition-transform duration-600 transform ${flippedCards[index] ? 'rotate-y-180' : ''}`} style={{ transformStyle: 'preserve-3d' }}>
                  <div className="absolute w-full h-full backface-hidden flex items-center justify-center p-4">
                    <h3 className="text-lg font-semibold text-cyan-400">{flashcard.question}</h3>
                  </div>
                  <div className="absolute w-full h-full backface-hidden flex items-center justify-center p-4 rotate-y-180">
                    <p className="text-gray-300">{flashcard.answer}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="w-full mt-12 mb-16">
          <h2 className="text-3xl font-bold mb-6 text-cyan-300">Suggested Flashcards</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 pb-24">            {suggestedFlashcards.map((flashcard, index) => (
              <div 
                key={index} 
                className="relative bg-gray-800 rounded-lg shadow-lg hover:shadow-cyan-500/50 transition duration-300 cursor-pointer h-56 perspective-1000"
                onClick={() => toggleFlip(index, true)}
              >
                <div className={`relative w-full h-full transition-transform duration-600 transform ${suggestedFlippedCards[index] ? 'rotate-y-180' : ''}`} style={{ transformStyle: 'preserve-3d' }}>
                  <div className="absolute w-full h-full backface-hidden flex items-center justify-center p-4">
                    <h3 className="text-lg font-semibold text-cyan-400">{flashcard.question}</h3>
                  </div>
                  <div className="absolute w-full h-full backface-hidden flex items-center justify-center p-4 rotate-y-180">
                    <p className="text-gray-300">{flashcard.answer}</p>
                  </div>
                </div>
                <button
                  className="absolute bottom-2 right-2 bg-cyan-600 hover:bg-cyan-700 text-white font-medium py-1 px-2 rounded-lg transition duration-300 ease-in-out transform hover:scale-105 z-10"
                  onClick={(e) => {
                    e.stopPropagation();
                    addSuggestedFlashcard(flashcard.question, flashcard.answer, index);
                  }}
                >
                  Add
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>

      {showModal && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center">
          <div className="bg-gray-800 p-6 rounded-lg shadow-lg w-11/12 sm:w-3/4 lg:w-1/2 xl:w-1/3">
            <h2 className="text-2xl font-semibold text-gray-200 mb-4">Add a New Flashcard</h2>
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Question"
              className="w-full bg-gray-700 text-gray-200 border border-gray-600 rounded-lg py-2 px-3 mb-4 focus:outline-none focus:border-cyan-400"
            />
            <textarea
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
              placeholder="Answer"
              className="w-full bg-gray-700 text-gray-200 border border-gray-600 rounded-lg py-2 px-3 mb-4 focus:outline-none focus:border-cyan-400 h-24"
            />
            <div className="flex justify-end">
              <button
                onClick={() => setShowModal(false)}
                className="bg-red-600 hover:bg-red-700 text-white font-medium py-2 px-4 rounded-lg mr-2 transition duration-300 ease-in-out transform hover:scale-105"
              >
                Cancel
              </button>
              <button
                onClick={addFlashcard}
                className="bg-cyan-600 hover:bg-cyan-700 text-white font-medium py-2 px-4 rounded-lg transition duration-300 ease-in-out transform hover:scale-105"
              >
                Add Flashcard
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FlashcardApp;