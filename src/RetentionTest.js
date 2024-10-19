import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faMicrophone, faKeyboard, faEdit, faStop , faSpinner } from '@fortawesome/free-solid-svg-icons';
import Navbar from './Navbar';

const RetentionTest = () => {
  const collectionId = localStorage.getItem('currentCollection');
  const sectionId = localStorage.getItem('currentSection');
  const [notes, setNotes] = useState([]);
  const [selectedNotes, setSelectedNotes] = useState([]);
  const [inputMethod, setInputMethod] = useState('type');
  const [userInput, setUserInput] = useState('');
  const [feedback, setFeedback] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [currentTranscript, setCurrentTranscript] = useState('');
  const [editingIndex, setEditingIndex] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const recognitionRef = useRef(null);
  const formatFeedback = (text) => {
    // Split by section numbers
    const sections = text.split(/\n\s*\d+\.\s/).filter(Boolean);
    
    // Prepare the feedback with bolded section headers
    return sections.map((section, index) => {
      const firstLineBreak = section.indexOf('\n');
      const sectionHeader = section.slice(0, firstLineBreak); // First line as header
      const sectionContent = section.slice(firstLineBreak + 1); // Remaining content

      return (
        <div key={index} className="mb-6">
          {/* Bolded section header */}
          <h3 className="font-bold text-lg mb-2 text-cyan-400">
            {`${index + 1}. ${sectionHeader}`}
          </h3>
          {/* Section content */}
          <p className="text-gray-300 text-lg whitespace-pre-line">{sectionContent.trim()}</p>
        </div>
      );
    });
  };
  useEffect(() => {
    const fetchNotes = async () => {
      try {
        const response = await axios.get('http://localhost:5000/get_notes', {
          params: { collection_id: collectionId, section_id: sectionId },
        });
        setNotes(response.data.notes);
      } catch (error) {
        console.error('Error fetching notes:', error);
      }
    };

    fetchNotes();

    if ('webkitSpeechRecognition' in window) {
      recognitionRef.current = new window.webkitSpeechRecognition();
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;

      recognitionRef.current.onresult = (event) => {
        let finalTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript + ' ';
          }
        }

        if (finalTranscript) {
          setCurrentTranscript(prev => prev + finalTranscript);
        }
      };

      recognitionRef.current.onerror = (event) => {
        console.error('Speech recognition error', event.error);
      };
    } else {
      console.log('Web Speech API is not supported in this browser.');
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, [collectionId, sectionId]);

  const handleNoteSelection = (noteId) => {
    setSelectedNotes(prevState =>
      prevState.includes(noteId)
        ? prevState.filter(id => id !== noteId)
        : [...prevState, noteId]
    );
  };

  const handleInputMethodChange = (method) => {
    setInputMethod(method);
  };

  const handleStartRecording = () => {
    setIsRecording(true);
    if (recognitionRef.current) {
      recognitionRef.current.start();
    }
  };

  const handleStopRecording = () => {
    setIsRecording(false);
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
  };

  const handleEditTranscript = () => {
    setEditingIndex(0);
  };

  const handleStopEditing = () => {
    setEditingIndex(null);
  };

  const handleSubmit = async () => {
    const finalUserInput = inputMethod === 'speak' ? currentTranscript : userInput;
    setIsLoading(true);

    try {
      const response = await axios.post('http://localhost:5000/retention', {
        section_id: sectionId,
        collection_id: collectionId,
        notes_selected: selectedNotes.map(note => note).join('\n'),        user_text: finalUserInput,
      });
      setFeedback(response.data);
    } catch (error) {
      console.error('Error submitting retention test:', error);
      setFeedback('An error occurred while processing your submission.');
    } finally {
      setIsLoading(false);

    }
  };

  return (
    <div className="bg-gray-900 text-gray-100 font-sans min-h-screen overflow-y-auto">
  <Navbar />
  <div className="container mx-auto px-4 py-8 max-w-3xl h-screen flex flex-col ">
    <div className="space-y-12 flex-grow pb-32"> 
      <h1 className="text-5xl md:text-6xl font-bold mb-12 text-cyan-400 text-center">Retention Test</h1>

          <section>
            <h2 className="text-3xl font-semibold mb-6 text-cyan-300">Select Notes to Test</h2>
            <div className="space-y-4 max-h-60 overflow-y-auto pr-2">
              {notes.map(note => (
                <div key={note.id} className="bg-gray-800 p-4 rounded-lg shadow-md">
                  <label className="flex items-center space-x-3 text-gray-300">
                    <input
                      type="checkbox"
                      checked={selectedNotes.includes(note.notes)}
                      onChange={() => handleNoteSelection(note.notes)}
                      className="form-checkbox h-6 w-6 text-cyan-600"
                    />
                    <span className="text-lg">{note.tldr}</span>
                  </label>
                </div>
              ))}
            </div>
          </section>

          {/* Input method selection */}
          <section>
            <h2 className="text-3xl font-semibold mb-6 text-cyan-300">Input Method</h2>
            <div className="space-y-4">
              <button
                onClick={() => handleInputMethodChange('type')}
                className={`w-full flex items-center justify-center p-4 rounded-lg shadow-md transition duration-300 ${
                  inputMethod === 'type' ? 'bg-cyan-600 text-white' : 'bg-gray-700 text-gray-300'
                }`}
              >
                <FontAwesomeIcon icon={faKeyboard} className="mr-3" />
                Type
              </button>
              <button
                onClick={() => handleInputMethodChange('speak')}
                className={`w-full flex items-center justify-center p-4 rounded-lg shadow-md transition duration-300 ${
                  inputMethod === 'speak' ? 'bg-cyan-600 text-white' : 'bg-gray-700 text-gray-300'
                }`}
              >
                <FontAwesomeIcon icon={faMicrophone} className="mr-3" />
                Speak
              </button>
            </div>
          </section>

          {/* User response section */}
          <section>
            <h2 className="text-3xl font-semibold mb-6 text-cyan-300">Your Response</h2>
            {inputMethod === 'type' ? (
              <textarea
                value={userInput}
                onChange={(e) => setUserInput(e.target.value)}
                className="w-full h-48 p-4 bg-gray-800 text-white rounded-lg border border-gray-700 focus:border-cyan-500 focus:ring focus:ring-cyan-500 focus:ring-opacity-50 text-lg"
                placeholder="Type your response here..."
              />
            ) : (
              <div className="bg-gray-800 p-4 rounded-lg">
                <div className="flex justify-between mb-4">
                  <button
                    onClick={isRecording ? handleStopRecording : handleStartRecording}
                    className={`${isRecording ? 'bg-red-600 hover:bg-red-700' : 'bg-cyan-600 hover:bg-cyan-700'} text-white font-medium py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out`}
                  >
                    {isRecording ? 'Stop Recording' : 'Start Recording'}
                  </button>
                  {currentTranscript && (
                    <button
                      onClick={editingIndex === 0 ? handleStopEditing : handleEditTranscript}
                      className="bg-cyan-600 hover:bg-cyan-700 text-white font-medium py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out"
                    >
                      <FontAwesomeIcon icon={editingIndex === 0 ? faStop : faEdit} />
                      {editingIndex === 0 ? ' Stop Editing' : ' Edit'}
                    </button>
                  )}
                </div>
                {editingIndex === 0 ? (
                  <textarea
                    value={currentTranscript}
                    onChange={(e) => setCurrentTranscript(e.target.value)}
                    className="w-full h-48 p-4 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-cyan-500 focus:ring focus:ring-cyan-500 focus:ring-opacity-50 text-lg"
                  />
                ) : (
                  <p className="text-lg">{currentTranscript}</p>
                )}
              </div>
            )}
          </section>
          <section className="mt-8">
            <div className="space-y-4">
              <div className="bg-gray-800 p-4 rounded-lg">
                <button
                  onClick={handleSubmit}
                  disabled={isLoading}
                  className="w-full bg-cyan-600 hover:bg-cyan-700 text-white font-medium py-4 px-6 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105 text-lg disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isLoading ? (
                    <>
                      <FontAwesomeIcon icon={faSpinner} spin className="mr-2" />
                      Submitting...
                    </>
                  ) : (
                    'Submit'
                  )}
                </button>
              </div>

              {feedback && (
            <section className="mt-12 p-6 bg-gray-800 rounded-lg shadow-md">
              <h2 className="text-3xl font-semibold mb-4 text-cyan-300">Feedback</h2>
              {/* <p className="text-lg">{feedback}</p> */}
              {formatFeedback(feedback)}
            </section>
          )}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};

export default RetentionTest;