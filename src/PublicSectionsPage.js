import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import Navbar from './Navbar';
import LoadingSpinner from './LoadingSpinner';
const HomePage = () => {
  const navigate = useNavigate();
  const [searchResults, setSearchResults] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedItem, setSelectedItem] = useState(null);
  const [sectionNotes, setSectionNotes] = useState([]);
  const [existingCollections, setExistingCollections] = useState([]);
  const [addToExisting, setAddToExisting] = useState(true);
  const [selectedCollectionId, setSelectedCollectionId] = useState('');
  const [newCollectionName, setNewCollectionName] = useState('');
  const [showModal, setShowModal] = useState(false);
  const [loading, setLoading] = useState(true);
  const [friendRequestSent, setFriendRequestSent] = useState(false);

  const modalRef = useRef(null);
  const username = localStorage.getItem('userName');
  const st = localStorage.getItem('searchTerm');
  const sc = localStorage.getItem('searchCategory');

  const handleKeyPress = (event) => {
    if (event.key === 'Enter') {
      localStorage.setItem('searchTerm', searchTerm);
      navigate(`/publicsections`);
    }
  };

  useEffect(() => {
    const fetchSearchResults = async () => {
      console.log(localStorage.getItem('userName'));
      try {
        const response = await axios.get(`http://localhost:5000/search_public_sections?search_term=${st}&search_category=${sc}&name=${username}`);
        if (sc === 'users') {
          setSearchResults(response.data.users || []);
        } else {
          setSearchResults(response.data.results || []);
        }
        setLoading(false);
      } catch (error) {
        console.error('Error fetching search results:', error);
        setLoading(false);
      }
    };

    fetchSearchResults();
  }, [st, sc, username]);

  useEffect(() => {
    const fetchExistingCollections = async () => {
      try {
        const response = await axios.get(`http://localhost:5000/get_my_collections?username=${username}`);
        setExistingCollections(response.data.collections);
      } catch (error) {
        console.error('Error fetching existing collections:', error);
      }
    };

    if (sc === 'publicsets') {
      fetchExistingCollections();
    }
  }, [username, sc]);

  useEffect(() => {
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleClickOutside = (event) => {
    if (modalRef.current && !modalRef.current.contains(event.target)) {
      setShowModal(false);
    }
  };

  const handleItemClick = async (item) => {
    setSelectedItem(item);
    if (sc === 'publicsets') {
      try {
        const response = await axios.get('http://localhost:5000/get_notes', {
          params: { collection_id: item.collection_id, section_id: item.id }
        });
        setSectionNotes(response.data.notes);
      } catch (error) {
        console.error('Error fetching notes:', error);
      }
    } else if (sc === 'users') {
      setFriendRequestSent(item.requests && item.requests.includes(username));
    }
    setShowModal(true);
  };

  const handleClone = async (sectionId) => {
    try {
      const payload = addToExisting
        ? { sectionId, addToExisting: true, collectionId: selectedCollectionId }
        : { sectionId, addToExisting: false, collectionName: newCollectionName, username };
      
      await axios.post('http://localhost:5000/clone_section', payload);
      navigate('/mygallery');
    } catch (error) {
      console.error('Error cloning section:', error);
    }
  };

  const handleSendFriendRequest = async () => {
    try {
      await axios.post('http://localhost:5000/send_friend_request', {
        from: username,
        to: selectedItem.username
      });
      setFriendRequestSent(true);
    } catch (error) {
      console.error('Error sending friend request:', error);
    }
  };

  return (
    <div className="bg-gray-900 text-gray-100 font-sans min-h-screen">
      <Navbar />
      <div className="container mx-auto px-4 py-8 flex flex-col items-center">
        <h2 className="text-4xl md:text-5xl font-bold mb-12 text-cyan-400 text-center">Search Results</h2>

        {loading ? (
          <LoadingSpinner />
        ) : (
          <div className="w-full max-w-6xl">
            {searchResults.length === 0 ? (
              <p className="text-center text-xl text-gray-400">No results found...</p>
            ) : (
              <div className="flex flex-wrap justify-center gap-6 ">
                {searchResults.map(item => (
                  <div
                    key={item.id}
                    onClick={() => handleItemClick(item)}
                    className="bg-gray-800 p-4 rounded-lg shadow-lg hover:shadow-cyan-500/50 transition duration-300 cursor-pointer h-32 w-48 flex items-center justify-center"
                  >
                    <h3 className="text-lg font-semibold text-cyan-400 text-center">
                      {sc === 'publicsets' ? (item.title || 'Untitled') : item.username}
                    </h3>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {showModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center">
          <div ref={modalRef} className="bg-gray-800 p-6 rounded-lg shadow-xl w-full max-w-md">
            {sc === 'publicsets' ? (
              <>
                <h2 className="text-2xl font-bold mb-4 text-cyan-400">Section Notes</h2>
                <div className="mb-4 max-h-60 overflow-y-auto">
                  {sectionNotes.map((note, index) => (
                    <div key={index} className="mb-2 p-2 bg-gray-700 rounded">
                      <p className="text-sm text-gray-300">{note.tldr}</p>
                    </div>
                  ))}
                </div>
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    <input
                      type="radio"
                      checked={addToExisting}
                      onChange={() => setAddToExisting(true)}
                      className="mr-2"
                    />
                    Add to existing collection
                  </label>
                  {addToExisting && (
                    <select
                      value={selectedCollectionId}
                      onChange={(e) => setSelectedCollectionId(e.target.value)}
                      className="w-full p-2 bg-gray-700 text-white rounded border border-gray-600"
                    >
                      <option value="">Select a collection</option>
                      {existingCollections.map((collection) => (
                        <option key={collection.id} value={collection.id}>
                          {collection.title}
                        </option>
                      ))}
                    </select>
                  )}
                </div>
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    <input
                      type="radio"
                      checked={!addToExisting}
                      onChange={() => setAddToExisting(false)}
                      className="mr-2"
                    />
                    Create new collection
                  </label>
                  {!addToExisting && (
                    <input
                      type="text"
                      value={newCollectionName}
                      onChange={(e) => setNewCollectionName(e.target.value)}
                      placeholder="New collection name"
                      className="w-full p-2 bg-gray-700 text-white rounded border border-gray-600"
                    />
                  )}
                </div>
                <div className="flex justify-end">
                  <button
                    onClick={() => setShowModal(false)}
                    className="mr-2 px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition duration-300"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={() => handleClone(selectedItem.id)}
                    className="px-4 py-2 bg-cyan-600 text-white rounded hover:bg-cyan-700 transition duration-300"
                  >
                    Clone Section
                  </button>
                </div>
              </>
            ) : (
              <>
              <h2 className="text-2xl font-bold mb-4 text-cyan-400">{selectedItem.username}</h2>
                <div className="flex justify-center">
                {selectedItem.friends && 
                selectedItem.friends.includes(localStorage.getItem('userName')) ? (
                  <p className="text-green-400">Already friends</p>
                ) : selectedItem.friend_requests && 
                  selectedItem.friend_requests.includes(localStorage.getItem('userName')) ? (
                  <p className="text-gray-300">Friend request pending</p>
                ) : friendRequestSent ? (
                  <p className="text-gray-300">Friend request sent</p>
                ) : (
                  <button
                    onClick={handleSendFriendRequest}
                    className="px-4 py-2 bg-cyan-600 text-white rounded hover:bg-cyan-700 transition duration-300"
                  >
                    Send Friend Request
                  </button>
                )}
              </div>
              <div className="mt-4 flex justify-end">
                <button
                  onClick={() => setShowModal(false)}
                  className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition duration-300"
                >
                  Close
                </button>
              </div>
            </>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default HomePage;