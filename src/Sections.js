import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faTrash, faPlus } from '@fortawesome/free-solid-svg-icons';
import Navbar from './Navbar';
import { useNavigate } from 'react-router-dom';
import LoadingSpinner from './LoadingSpinner';

const Sections = () => {
  const collectionId = localStorage.getItem('currentCollection');
  const [sections, setSections] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [newSectionName, setNewSectionName] = useState('');
  const [newReviewName, setNewReviewName] = useState('');
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isReviewModalOpen, setIsReviewModalOpen] = useState(false);
  const [draggedSectionId, setDraggedSectionId] = useState(null);
  const [isDraggingOverTrash, setIsDraggingOverTrash] = useState(false);
  const collectionName = localStorage.getItem('collectionName');
  const [selectedSections, setSelectedSections] = useState([]);
  const navigate = useNavigate();
  const [suggestedSections, setSuggestedSections] = useState([]);

  useEffect(() => {
    fetchSections();
    fetchSuggestions();
  }, [collectionId]);

  const fetchSections = async () => {
    try {
      if (!collectionId) {
        throw new Error('Collection ID not found in local storage');
      }
      const response = await axios.get(`http://localhost:5000/get_sections?collection_id=${collectionId}`);
      const sortedSections = response.data.sections.sort((a, b) => 
        a.section_name.localeCompare(b.section_name, undefined, { sensitivity: 'base' })
      );

      setSections(sortedSections);
      setLoading(false);
    } catch (error) {
      setError(error.message);
      setLoading(false);
    }
  };

  const fetchSuggestions = async () => {
    try {
      const response = await axios.get(`http://localhost:5000/suggest_sections?collection_id=${collectionId}`);
      setSuggestedSections(response.data.suggested_sections || []);
    } catch (error) {
      console.error(error);
    }
  };

  const handleCreateSection = async () => {
    try {
      await axios.post('http://localhost:5000/create_section', {
        collection_id: collectionId,
        section_name: newSectionName,
      });
      setNewSectionName('');
      setIsModalOpen(false);
      fetchSections();
    } catch (error) {
      setError(error.message);
    }
  };

  const handleSectionClick = (section) => {
    localStorage.setItem('currentSection', section.id);
    localStorage.setItem('currentSectionName', section.section_name);
    navigate('/chapter');
  };

  const handleDeleteSection = async (sectionId) => {
    const confirmDelete = window.confirm('Are you sure you want to delete this section?');
    if (!confirmDelete) return;

    try {
      console.log('Deleting section:', sectionId);

      await axios.delete('http://localhost:5000/delete_section', {
        data: { collection_id: collectionId, section_id: sectionId }
      });
      fetchSections();
    } catch (error) {
      console.error(error);
      setError(error.message);
    }
  };

  const handleDragStart = (e, sectionId) => {
    e.stopPropagation();
    setDraggedSectionId(sectionId);
    e.dataTransfer.setData('text/plain', sectionId.toString());
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDraggingOverTrash(true);
  };

  const handleDragLeave = () => {
    setIsDraggingOverTrash(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const sectionId = e.dataTransfer.getData('text');
    if (sectionId) {
      handleDeleteSection(sectionId);
    }
    setDraggedSectionId(null);
    setIsDraggingOverTrash(false);
  };

  const handleSubmitReview = async () => {
    try {
      await axios.post('http://localhost:5000/create_review', {
        collection_id: collectionId,
        section_ids: selectedSections,
        name: newReviewName,
        username: localStorage.getItem('userName')
      });
      setIsReviewModalOpen(false);
      setSelectedSections([]);
      setNewReviewName('');
      fetchSections();
    } catch (error) {
      setError(error.message);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-screen bg-gray-900">
        <LoadingSpinner />
      </div>
    );
  }

  return (
    <div className="bg-gray-900 min-h-screen text-gray-100 flex flex-col">
    <Navbar />
    <div className="flex-grow overflow-y-auto pb-20">
      <div className="container mx-auto px-4 py-8 flex flex-col items-center space-y-8 ">
        <h2 className="text-4xl md:text-5xl font-bold text-cyan-400">Sections in {collectionName}</h2>
        <div className="w-full max-w-6xl">
          <div className="flex flex-wrap justify-center gap-6">
            <button
              onClick={() => setIsModalOpen(true)}
              className="bg-cyan-600 hover:bg-cyan-700 text-white font-medium p-4 rounded-lg transition duration-300 ease-in-out transform hover:scale-105 flex items-center justify-center h-32 w-48"
            >
              <FontAwesomeIcon icon={faPlus} className="mr-2" />
              New Section
            </button>
            <button
              onClick={() => setIsReviewModalOpen(true)}
              className="bg-cyan-600 hover:bg-cyan-700 text-white font-medium p-4 rounded-lg transition duration-300 ease-in-out transform hover:scale-105 flex items-center justify-center h-32 w-48"
            >
              <FontAwesomeIcon icon={faPlus} className="mr-2" />
              New Review
            </button>
            {sections.map(section => (
              <div 
                key={section.id}
                draggable
                onDragStart={(e) => handleDragStart(e, section.id)}
                onClick={() => handleSectionClick(section)}
                className="bg-gray-800 p-4 rounded-lg shadow-lg hover:shadow-cyan-500/50 transition duration-300 cursor-pointer h-32 w-48 flex items-center justify-center"
              >
                <h3 className="text-lg font-semibold text-cyan-400 text-center">{section.section_name || 'Untitled'}</h3>
              </div>
            ))}
          </div>
        </div>

        {suggestedSections.length > 0 && (
          <div className="w-full max-w-6xl mt-8 pb-40">
            <h3 className="text-2xl font-bold text-cyan-300 mb-4 justify-center flex flex-wrap gap-6">Suggested Learning Paths</h3>
            <div className="flex flex-wrap justify-center gap-6">
              {suggestedSections.map((section, index) => (
                <div
                  key={index}
                  className="bg-gray-800 p-4 rounded-lg shadow-lg hover:shadow-cyan-500/50 transition duration-300 cursor-pointer h-32 w-48 flex items-center justify-center"
                  onClick={() => {
                    setNewSectionName(section);
                    setIsModalOpen(true);
                  }}
                >
                  <h3 className="text-lg font-semibold text-cyan-400 text-center">{section}</h3>
                </div>
              ))}
            </div>
          </div>
        )}

        <div
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onDragLeave={handleDragLeave}
          className={`fixed bottom-10 left-1/2 transform -translate-x-1/2 text-6xl cursor-pointer transition-colors duration-300 ${
            isDraggingOverTrash ? 'text-red-600' : 'text-red-400'
          }`}
        >
          <FontAwesomeIcon icon={faTrash} />
        </div>
      </div>
      </div>

      {isModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center">
          <div className="bg-gray-800 p-6 rounded-lg shadow-xl w-full max-w-md">
            <h2 className="text-2xl font-bold mb-4 text-cyan-400">Create New Section</h2>
            <input
              type="text"
              placeholder="Section Name"
              value={newSectionName}
              onChange={(e) => setNewSectionName(e.target.value)}
              className="w-full p-2 mb-4 bg-gray-700 text-white rounded border border-gray-600 focus:border-cyan-500 focus:ring focus:ring-cyan-500 focus:ring-opacity-50"
            />
            {error && <p className="text-red-500 mb-4">{error}</p>}
            <div className="flex justify-end">
              <button
                onClick={() => setIsModalOpen(false)}
                className="mr-2 px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition duration-300"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateSection}
                className="px-4 py-2 bg-cyan-600 text-white rounded hover:bg-cyan-700 transition duration-300"
              >
                Create Section
              </button>
            </div>
          </div>
        </div>
      )}

      {isReviewModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center">
          <div className="bg-gray-800 p-6 rounded-lg shadow-xl w-full max-w-md">
            <h2 className="text-2xl font-bold mb-4 text-cyan-400">Create Review</h2>
            <input
              type="text"
              placeholder="Review Name"
              value={newReviewName}
              onChange={(e) => setNewReviewName(e.target.value)}
              className="w-full p-2 mb-4 bg-gray-700 text-white rounded border border-gray-600 focus:border-cyan-500 focus:ring focus:ring-cyan-500 focus:ring-opacity-50"
            />
            <div className="max-h-60 overflow-y-auto mb-4">
              {sections.map(section => (
                <div key={section.id} className="flex items-center mb-2">
                  <input
                    type="checkbox"
                    id={`section-${section.id}`}
                    checked={selectedSections.includes(section.id)}
                    onChange={() => setSelectedSections(prev => 
                      prev.includes(section.id) 
                        ? prev.filter(id => id !== section.id)
                        : [...prev, section.id]
                    )}
                    className="mr-2"
                  />
                  <label htmlFor={`section-${section.id}`} className="text-gray-300">{section.section_name}</label>
                </div>
              ))}
            </div>
            <div className="flex justify-end">
              <button
                onClick={() => setIsReviewModalOpen(false)}
                className="mr-2 px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition duration-300"
              >
                Cancel
              </button>
              <button
                onClick={handleSubmitReview}
                className="px-4 py-2 bg-cyan-600 text-white rounded hover:bg-cyan-700 transition duration-300"
              >
                Create Review
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Sections;