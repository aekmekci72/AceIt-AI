import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faTrash, faEdit, faPlusCircle, faRobot, faSpinner, faTimes } from '@fortawesome/free-solid-svg-icons';
import Navbar from './Navbar';

const StudyTimelines = () => {
  const [existingPlans, setExistingPlans] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [showAiModal, setShowAiModal] = useState(false);
  const [showManualModal, setShowManualModal] = useState(false);
  const [testDate, setTestDate] = useState('');
  const [newPlan, setNewPlan] = useState({ name: '', concepts: [], testDate: '' });
  const username = localStorage.getItem('userName');
  const [collections, setCollections] = useState([]);
const [sections, setSections] = useState([]);
const [selectedCollection, setSelectedCollection] = useState('');
const [selectedSection, setSelectedSection] = useState('');
  useEffect(() => {
    fetchExistingPlans();
  }, []);
  useEffect(() => {
    fetchCollections();
  }, []);
  
  const fetchCollections = async () => {
    try {
      const username = localStorage.getItem('userName');
      if (!username) {
        throw new Error('Username not found in local storage');
      }
      const response = await axios.get(`http://localhost:5000/get_my_collections?username=${username}`);
      const sortedCollections = response.data.collections.sort((a, b) => 
        a.title.localeCompare(b.title, undefined, { sensitivity: 'base' })
      );
      setCollections(sortedCollections);
    } catch (error) {
      console.error(error);
      setError(error.message);
    }
  };
  const fetchSections = async (collectionId) => {
    try {
      const response = await axios.get(`http://localhost:5000/get_sections?collection_id=${collectionId}`);
      const sortedSections = response.data.sections.sort((a, b) => 
        a.section_name.localeCompare(b.section_name, undefined, { sensitivity: 'base' })
      );
      setSections(sortedSections);
    } catch (error) {
      console.error(error);
      setError(error.message);
    }
  };

  const fetchExistingPlans = async () => {
    const username = localStorage.getItem('userName');
  
    if (username) {
      try {
        const response = await axios.get(`http://localhost:5000/study-plans?username=${encodeURIComponent(username)}`);
        if (response.data.success && Array.isArray(response.data.studyPlans)) {
          setExistingPlans(response.data.studyPlans);
        } else {
          setError('Invalid data format received from server.');
        }
      } catch (error) {
        console.error('Error fetching existing plans:', error);
        setError('Failed to fetch existing plans. Please try again.');
      }
    } else {
      setError('Username not found. Please log in again.');
    }
  };

  const handleDeletePlan = async (index) => {
    const username = localStorage.getItem('userName');
    const planToDelete = index;
    try {
        await axios({
          method: 'delete',
          url: 'http://localhost:5000/study-plans',
          data: { username, index },
          headers: { 'Content-Type': 'application/json' }
        });
        fetchExistingPlans();
    
    } catch (error) {
      console.error('Error deleting plan:', error);
      setError('Failed to delete plan. Please try again.');
    }
  };

  const handleEditPlan = (plan) => {
    // Implement edit functionality
    console.log('Edit plan:', plan);
  };
  const handleCollectionChange = (e) => {
    const collectionId = e.target.value;
    setSelectedCollection(collectionId);
    setSelectedSection('');
    fetchSections(collectionId);
  };
  
  const handleSectionChange = (e) => {
    setSelectedSection(e.target.value);
  };

  const handleAiSuggestion = async () => {
    setIsLoading(true);
    try {
      const response = await axios.post('http://localhost:5000/ai-study-suggestion', {
        testDate: testDate,
        collection_id: selectedCollection,
        section_id: selectedSection,
      });
      
      // Convert the object to an array
      const conceptsArray = Object.values(response.data).map(item => ({
        name: item.topic,
        finishDate: item.date
      }));
  
      setNewPlan({
        concepts: conceptsArray,
        testDate: testDate
      });
      setShowAiModal(false);
      setShowManualModal(true);
    } catch (error) {
      console.error('Error getting AI suggestion:', error);
      setError('Failed to get AI suggestion. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCreatePlan = async () => {
    setIsLoading(true);
    try {
      const planToSubmit = {
        ...newPlan,
        collectionId: selectedCollection,
        username: username,
        sectionId: selectedSection,
        concepts: newPlan.concepts.map(concept => ({
          name: concept.name,
          finishDate: concept.finishDate
        }))
      };
      await axios.post('http://localhost:5000/study-plans', planToSubmit);
      fetchExistingPlans();
      setNewPlan({concepts: [], testDate: '' });
      setSelectedCollection('');
      setSelectedSection('');
      setShowManualModal(false);
    } catch (error) {
      console.error('Error creating plan:', error);
      setError('Failed to create plan. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleAddConcept = () => {
    setNewPlan(prevPlan => ({
      ...prevPlan,
      concepts: [...prevPlan.concepts, { name: '', finishDate: '' }],
    }));
  };

  const handleConceptChange = (index, field, value) => {
    setNewPlan(prevPlan => {
      const updatedConcepts = prevPlan.concepts.map((concept, i) =>
        i === index ? { ...concept, [field]: value } : concept
      );
      return { ...prevPlan, concepts: updatedConcepts };
    });
  };

  const Modal = ({ show, onClose, title, children }) => {
    if (!show) return null;

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center">
        <div className="bg-gray-800 p-6 rounded-lg shadow-lg w-full max-w-md">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-xl font-semibold text-cyan-300">{title}</h3>
            <button onClick={onClose} className="text-gray-400 hover:text-white">
              <FontAwesomeIcon icon={faTimes} />
            </button>
          </div>
          {children}
        </div>
      </div>
    );
  };

  return (
    <div className="bg-gray-900 text-gray-100 font-sans min-h-screen">
      <Navbar />
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        <h1 className="text-5xl md:text-6xl font-bold mb-12 text-cyan-400 text-center">Study Timelines</h1>

        {/* Existing Study Plans Section */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-6 text-cyan-300">Existing Study Plans</h2>
          <div className="space-y-4">
            {existingPlans.map((plan, index) => (
              <div key={index} className="bg-gray-800 p-4 rounded-lg shadow-md">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-xl font-semibold">Study Plan</h3>
                  <div>
                    <button
                      onClick={() => handleEditPlan(plan)}
                      className="text-cyan-400 hover:text-cyan-300 mr-2"
                    >
                      <FontAwesomeIcon icon={faEdit} />
                    </button>
                    <button
                      onClick={() => handleDeletePlan(index)}
                      className="text-red-400 hover:text-red-300"
                    >
                      <FontAwesomeIcon icon={faTrash} />
                    </button>
                  </div>
                </div>
                <ul className="list-disc list-inside space-y-2">
                    {plan.concepts.map((concept, index) => (
                        <li key={index}>
                        {concept.name} - Finish by: {new Date(concept.finishDate).toLocaleDateString()}
                        </li>
                    ))}
                    </ul>
                <p className="mt-4 text-cyan-300">Test Date: {plan.testDate}</p>
              </div>
            ))}
          </div>
        </section>

        {/* New Study Timeline Section */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-6 text-cyan-300">New Study Timeline</h2>
          <div className="space-y-4">
            <button
              onClick={() => setShowAiModal(true)}
              className="w-full bg-cyan-600 hover:bg-cyan-700 text-white font-medium py-3 px-4 rounded-lg shadow-md transition duration-300 ease-in-out"
            >
              <FontAwesomeIcon icon={faRobot} className="mr-2" />
              Get AI Suggestion
            </button>
            <button
              onClick={() => setShowManualModal(true)}
              className="w-full bg-cyan-600 hover:bg-cyan-700 text-white font-medium py-3 px-4 rounded-lg shadow-md transition duration-300 ease-in-out"
            >
              <FontAwesomeIcon icon={faPlusCircle} className="mr-2" />
              Create Manual Plan
            </button>
          </div>
        </section>

        {/* AI Suggestion Modal */}
        <Modal show={showAiModal} onClose={() => setShowAiModal(false)} title="AI Study Suggestion">
  <div className="space-y-4">
    <select
      value={selectedCollection}
      onChange={handleCollectionChange}
      className="w-full p-2 bg-gray-700 text-white rounded-md"
    >
      <option value="">Select a Collection</option>
      {collections.map((collection) => (
        <option key={collection.id} value={collection.id}>
          {collection.title}
        </option>
      ))}
    </select>

    {selectedCollection && (
      <select
        value={selectedSection}
        onChange={handleSectionChange}
        className="w-full p-2 bg-gray-700 text-white rounded-md"
      >
        <option value="">Select a Section</option>
        {sections.map((section) => (
          <option key={section.id} value={section.id}>
            {section.section_name}
          </option>
        ))}
      </select>
    )}
            <input
              type="date"
              value={testDate}
              onChange={(e) => setTestDate(e.target.value)}
              className="w-full p-2 bg-gray-700 text-white rounded-md"
            />
            <button
              onClick={handleAiSuggestion}
              disabled={isLoading}
              className="w-full bg-cyan-600 hover:bg-cyan-700 text-white font-medium py-2 px-4 rounded-md shadow-md transition duration-300 ease-in-out"
            >
              {isLoading ? <FontAwesomeIcon icon={faSpinner} spin /> : 'Generate Suggestion'}
            </button>
          </div>
        </Modal>

        {/* Manual Creation Modal */}
        <Modal show={showManualModal} onClose={() => setShowManualModal(false)} title="Create Study Plan">
  <div className="space-y-4 max-h-[70vh] overflow-y-auto pr-2">
    <select
      value={selectedCollection}
      onChange={handleCollectionChange}
      className="w-full p-2 bg-gray-700 text-white rounded-md"
    >
      <option value="">Select a Collection</option>
      {collections.map((collection) => (
        <option key={collection.id} value={collection.id}>
          {collection.title}
        </option>
      ))}
    </select>

    {selectedCollection && (
      <select
        value={selectedSection}
        onChange={handleSectionChange}
        className="w-full p-2 bg-gray-700 text-white rounded-md"
      >
        <option value="">Select a Section</option>
        {sections.map((section) => (
          <option key={section.id} value={section.id}>
            {section.section_name}
          </option>
        ))}
      </select>
    )}
        
            <input
              type="date"
              value={newPlan.testDate}
              onChange={(e) => setNewPlan({ ...newPlan, testDate: e.target.value })}
              className="w-full p-2 bg-gray-700 text-white rounded-md"
            />
            {newPlan.concepts.map((concept, index) => (
                <div key={index} className="flex space-x-2">
                    <input
                    type="text"
                    value={concept.name}
                    onChange={(e) => handleConceptChange(index, 'name', e.target.value)}
                    placeholder="Concept Name"
                    className="flex-grow p-2 bg-gray-700 text-white rounded-md"
                    />
                    <input
                    type="date"
                    value={concept.finishDate || ''}
                    onChange={(e) => handleConceptChange(index, 'finishDate', e.target.value)}
                    min={new Date().toISOString().split('T')[0]}
                    max={newPlan.testDate}
                    className="w-40 p-2 bg-gray-700 text-white rounded-md"
                    />
                </div>
                ))}
            <button
              onClick={handleAddConcept}
              className="w-full bg-cyan-600 hover:bg-cyan-700 text-white font-medium py-2 px-4 rounded-md shadow-md transition duration-300 ease-in-out"
            >
              <FontAwesomeIcon icon={faPlusCircle} className="mr-2" />
              Add Concept
            </button>
            <button
              onClick={handleCreatePlan}
              disabled={isLoading}
              className="w-full bg-cyan-600 hover:bg-cyan-700 text-white font-medium py-2 px-4 rounded-md shadow-md transition duration-300 ease-in-out"
            >
              {isLoading ? <FontAwesomeIcon icon={faSpinner} spin /> : 'Create Plan'}
            </button>
          </div>
        </Modal>

        {error && <p className="text-red-500 mt-4">{error}</p>}
      </div>
    </div>
  );
};

export default StudyTimelines;