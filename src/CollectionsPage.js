import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faTrash, faPlus } from '@fortawesome/free-solid-svg-icons';
import Navbar from './Navbar';
import { useAuth } from './AuthContext';
import { useNavigate } from 'react-router-dom';
import LoadingSpinner from './LoadingSpinner';

const Collections = () => {
  const { auth } = useAuth();
  const [collections, setCollections] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [newCollectionName, setNewCollectionName] = useState('');
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [draggedCollectionId, setDraggedCollectionId] = useState(null);
  const [isDraggingOverTrash, setIsDraggingOverTrash] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
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
        setLoading(false);
      } catch (error) {
        console.error(error);
        setError(error.message);
        setLoading(false);
      }
    };

    fetchCollections();
  }, []);

  const handleCreateCollection = async () => {
    if (!newCollectionName.trim()) {
      setError('Collection name cannot be empty.');
      return;
    }

    if (collections.some(collection => collection.title === newCollectionName)) {
      setError('A collection with this name already exists.');
      return;
    }

    try {
      const username = localStorage.getItem('userName');
      if (!username) {
        throw new Error('Username not found in local storage');
      }

      await axios.post('http://localhost:5000/create_collection', {
        collection_name: newCollectionName,
        username: username
      });

      const response = await axios.get(`http://localhost:5000/get_my_collections?username=${username}`);
      setCollections(response.data.collections);

      setNewCollectionName('');
      setIsModalOpen(false);
      setError(null);
    } catch (error) {
      console.error(error);
      setError(error.message);
    }
  };

  const handleDeleteCollection = async (collectionId) => {
    const confirmDelete = window.confirm('Are you sure you want to delete this collection?');
    if (!confirmDelete) return;

    try {
      const username = localStorage.getItem('userName');
      console.log('Deleting collection:', collectionId);
      await axios.delete('http://localhost:5000/delete_collection', {
        data: { collection_id: collectionId }
      });

      const response = await axios.get(`http://localhost:5000/get_my_collections?username=${username}`);
      setCollections(response.data.collections);
      setError(null);
    } catch (error) {
      console.error(error);
      setError(error.message);
    }
  };

  const handleOpenCollection = (collectionId, collectionName) => {
    localStorage.setItem('currentCollection', collectionId);
    localStorage.setItem('collectionName', collectionName);
    navigate('/sections');
  };

  const handleDragStart = (e, collectionId) => {
    e.stopPropagation();
    setDraggedCollectionId(collectionId);
    e.dataTransfer.setData('text/plain', collectionId.toString());
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
    const collectionId = e.dataTransfer.getData('text');
    if (collectionId) {
      handleDeleteCollection(collectionId);
    }
    setDraggedCollectionId(null);
    setIsDraggingOverTrash(false);
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-screen bg-gray-900">
        <LoadingSpinner />
      </div>
    );
  }

  return (
    <div className="bg-gray-900 min-h-screen text-gray-100">
      <Navbar />
      <div className="container mx-auto px-4 py-8 flex flex-col items-center space-y-8">
        <h2 className="text-4xl md:text-5xl font-bold text-cyan-400">Your Collections</h2>
        
        <div className="w-full max-w-6xl">
          <div className="flex flex-wrap justify-center gap-6">
            <button
              onClick={() => setIsModalOpen(true)}
              className="bg-cyan-600 hover:bg-cyan-700 text-white font-medium p-4 rounded-lg transition duration-300 ease-in-out transform hover:scale-105 flex items-center justify-center h-32 w-48"
            >
              <FontAwesomeIcon icon={faPlus} className="mr-2" />
              New Collection
            </button>
            {collections.map(collection => (
              <div 
                key={collection.id}
                draggable
                onDragStart={(e) => handleDragStart(e, collection.id)}
                onClick={() => handleOpenCollection(collection.id, collection.title)}
                className="bg-gray-800 p-4 rounded-lg shadow-lg hover:shadow-cyan-500/50 transition duration-300 cursor-pointer h-32 w-48 flex items-center justify-center"
              >
                <h3 className="text-lg font-semibold text-cyan-400 text-center">{collection.title || 'Untitled'}</h3>
              </div>
            ))}
          </div>
        </div>

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


      {isModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center">
          <div className="bg-gray-800 p-6 rounded-lg shadow-xl w-full max-w-md">
            <h2 className="text-2xl font-bold mb-4 text-cyan-400">Create New Collection</h2>
            <input
              type="text"
              placeholder="Collection Name"
              value={newCollectionName}
              onChange={(e) => setNewCollectionName(e.target.value)}
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
                onClick={handleCreateCollection}
                className="px-4 py-2 bg-cyan-600 text-white rounded hover:bg-cyan-700 transition duration-300"
              >
                Create Collection
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Collections;