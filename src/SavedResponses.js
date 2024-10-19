import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Navbar from './Navbar';
import LoadingSpinner from './LoadingSpinner';

const SavedResponsesPage = () => {
  const chapterId = localStorage.getItem('currentSection');
  const collectionId = localStorage.getItem('currentCollection');
  const [savedResponses, setSavedResponses] = useState([]);
  const [modalIsOpen, setModalIsOpen] = useState(false);
  const [currentResponse, setCurrentResponse] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchSavedResponses = async () => {
      try {
        const response = await axios.get(`http://localhost:5000/get_saved_responses?collection_id=${collectionId}&section_id=${chapterId}`);
        setSavedResponses(response.data.notes);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching saved responses:', error);
        setLoading(false);
      }
    };

    fetchSavedResponses();
  }, [chapterId, collectionId]);

  const openModal = (response) => {
    setCurrentResponse(response);
    setModalIsOpen(true);
  };

  const closeModal = () => {
    setModalIsOpen(false);
    setCurrentResponse(null);
  };

  const deleteResponse = async (id) => {
    try {
      await axios.delete(`http://localhost:5000/delete_response`, {
        params: { collection_id: collectionId, section_id: chapterId, response_id: id }
      });
      setSavedResponses(savedResponses.filter(response => response.id !== id));
    } catch (error) {
      console.error('Error deleting response:', error);
    }
  };

  const addToNotesAndDelete = async (response) => {
    try {
      await axios.post('http://localhost:5000/add_response_to_notes', {
        collection_id: collectionId,
        section_id: chapterId,
        response_id: response.id
      });
      setSavedResponses(savedResponses.filter(res => res.id !== response.id));
      closeModal();
    } catch (error) {
      console.error('Error adding to notes and deleting response:', error);
    }
  };

  return (
    <div className="bg-gray-900 text-gray-100 font-sans min-h-screen">
      <Navbar />
      <div className="container mx-auto px-4 py-8 flex flex-col items-center">
        <h1 className="text-6xl md:text-7xl font-bold mb-12 text-cyan-400 text-center">Saved Responses</h1>

        {loading ? (
          <div className="flex justify-center items-center h-64">
            <LoadingSpinner />
          </div>
        ) : (
          <div className="w-full">
            {savedResponses.length === 0 ? (
              <p className="text-center text-xl text-gray-400">No saved responses found.</p>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                {savedResponses.map((savedResponse) => (
                  <div
                    key={savedResponse.id}
                    className="bg-gray-800 p-4 rounded-lg shadow-lg hover:shadow-cyan-500/50 transition duration-300 cursor-pointer"
                  >
                    <h3 className="text-lg font-semibold mb-2 text-cyan-400">{savedResponse.question || 'Untitled'}</h3>
                    <p className="text-gray-400 text-sm mb-4">{savedResponse.tldr || 'No summary available'}</p>
                    <div className="flex justify-between">
                      <button
                        onClick={() => openModal(savedResponse)}
                        className="px-3 py-1 bg-cyan-600 text-white rounded hover:bg-cyan-700 transition duration-300"
                      >
                        View
                      </button>
                      <button
                        onClick={() => deleteResponse(savedResponse.id)}
                        className="px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700 transition duration-300"
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {modalIsOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center">
          <div className="bg-gray-800 p-6 rounded-lg shadow-xl w-full max-w-2xl">
            <h2 className="text-2xl font-bold mb-4 text-cyan-400">{currentResponse.question || 'Untitled'}</h2>
            <div className="mb-4 max-h-96 overflow-y-auto text-gray-300">
              <p>{currentResponse.data}</p>
            </div>
            <div className="flex justify-end">
              <button
                onClick={() => addToNotesAndDelete(currentResponse)}
                className="mr-2 px-4 py-2 bg-cyan-600 text-white rounded hover:bg-cyan-700 transition duration-300"
              >
                Add to Notes
              </button>
              <button
                onClick={closeModal}
                className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition duration-300"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SavedResponsesPage;