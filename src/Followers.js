import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCheck, faTimes, faUser } from '@fortawesome/free-solid-svg-icons';
import Navbar from './Navbar';
import { useNavigate } from 'react-router-dom';

const FriendsPage = () => {

  const navigate = useNavigate();
  const [followRequests, setFollowRequests] = useState([]);
  const [friends, setFriends] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedFriend, setSelectedFriend] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [friendSections, setFriendSections] = useState([]);
  const [selectedSection, setSelectedSection] = useState(null);
  const [sectionNotes, setSectionNotes] = useState([]);
  const [addToExisting, setAddToExisting] = useState(true);
  const [selectedCollectionId, setSelectedCollectionId] = useState('');
  const [newCollectionName, setNewCollectionName] = useState('');
  const [existingCollections, setExistingCollections] = useState([]);
  const modalRef = useRef();

  useEffect(() => {
    const fetchData = async () => {
      try {
        await Promise.all([fetchFollowRequests(), fetchFriends(), fetchExistingCollections()]);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching data:', error);
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const fetchFollowRequests = async () => {
    const username = localStorage.getItem('userName');
    const response = await axios.get(`http://localhost:5000/get_follow_requests?username=${username}`);
    setFollowRequests(response.data.follow_requests);
  };

  const fetchFriends = async () => {
    const username = localStorage.getItem('userName');
    const response = await axios.get(`http://localhost:5000/get_friends?username=${username}`);
    setFriends(response.data.friends);
  };

  const fetchExistingCollections = async () => {
    try {
      const response = await axios.get(`http://localhost:5000/get_my_collections?username=${localStorage.getItem('userName')}`);
      setExistingCollections(response.data.collections);
    } catch (error) {
      console.error('Error fetching existing collections:', error);
    }
  }



  const handleAccept = async (requester) => {
    const username = localStorage.getItem('userName');
    await axios.post('http://localhost:5000/accept_friend_request', {
      user: username,
      requester: requester
    });
    await Promise.all([fetchFollowRequests(), fetchFriends()]);
  };

  const handleDecline = async (requester) => {
    const username = localStorage.getItem('userName');
    await axios.post('http://localhost:5000/decline_friend_request', {
      user: username,
      requester: requester
    });
    await fetchFollowRequests();
  };

  const handleFriendClick = async (friendId) => {
    try {
      const response = await axios.get(`http://localhost:5000/friend-public-sections/${friendId}`);
      setSelectedFriend(friendId);
      setFriendSections(response.data.sections);
      setShowModal(true);
    } catch (error) {
      console.error('Error fetching friend sections:', error);
    }
  };

  const handleSectionClick = async (section) => {
    setSelectedSection(section);
    try {
      const response = await axios.get('http://localhost:5000/get_notes', {
        params: { collection_id: section.collection_id, section_id: section.id }
      });
      setSectionNotes(response.data.notes);
    } catch (error) {
      console.error('Error fetching notes:', error);
    }
  };

  const handleClone = async (sectionId) => {
    try {
      const payload = addToExisting
        ? { sectionId, addToExisting: true, collectionId: selectedCollectionId }
        : { sectionId, addToExisting: false, collectionName: newCollectionName, username:localStorage.getItem('userName') };
      
      await axios.post('http://localhost:5000/clone_section', payload);
      navigate('/mygallery');
    } catch (error) {
      console.error('Error cloning section:', error);
    }
  };

  if (loading) {
    return (
      <div className="bg-gray-900 text-gray-100 font-sans min-h-screen overflow-y-auto">
        <Navbar />
        <div className="flex justify-center items-center h-screen">
          <div className="animate-spin rounded-full h-32 w-32 border-t-2 border-b-2 border-cyan-400"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 text-gray-100 font-sans min-h-screen overflow-y-auto">
      <Navbar />
      <div className="container mx-auto px-4 py-8 flex flex-col items-center">
        <h1 className="text-6xl md:text-7xl font-bold mb-12 text-cyan-400 text-center">Friends & Requests</h1>
        
        {/* Follow Requests Section */}
        <div className="w-full max-w-4xl mb-12">
          <h2 className="text-3xl font-bold mb-6 text-cyan-300">Follow Requests</h2>
          {followRequests.length > 0 ? (
            <div className="space-y-4">
              {followRequests.map((request) => (
                <div key={request} className="bg-gray-800 rounded-lg p-4 shadow-lg flex justify-between items-center">
                  <p className="text-lg font-medium">{request}</p>
                  <div className="space-x-2">
                    <button
                      onClick={() => handleAccept(request)}
                      className="bg-green-600 hover:bg-green-700 text-white px-3 py-1 rounded transition duration-300"
                    >
                      <FontAwesomeIcon icon={faCheck} className="mr-1" /> Accept
                    </button>
                    <button
                      onClick={() => handleDecline(request)}
                      className="bg-red-600 hover:bg-red-700 text-white px-3 py-1 rounded transition duration-300"
                    >
                      <FontAwesomeIcon icon={faTimes} className="mr-1" /> Decline
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-400">No pending follow requests.</p>
          )}
        </div>

        {/* Friends Section */}
        <div className="w-full max-w-4xl">
          <h2 className="text-3xl font-bold mb-6 text-cyan-300">Your Friends</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {friends.map((friend) => (
              <div
                key={friend}
                onClick={() => handleFriendClick(friend)}
                className="bg-gray-800 rounded-lg p-4 shadow-lg hover:shadow-cyan-500/50 transition duration-300 cursor-pointer flex flex-col items-center"
              >
                <FontAwesomeIcon icon={faUser} className="text-cyan-400 text-4xl mb-2" />
                <p className="text-lg font-medium">{friend}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Modal */}
      {showModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center">
          <div ref={modalRef} className="bg-gray-800 p-6 rounded-lg shadow-xl w-full max-w-md">
            <h2 className="text-2xl font-bold mb-4 text-cyan-400">{selectedFriend}'s Public Sections</h2>
            <div className="mb-4 max-h-60 overflow-y-auto">
              {friendSections.map((section) => (
                <div
                  key={section.id}
                  onClick={() => handleSectionClick(section)}
                  className="mb-2 p-2 bg-gray-700 rounded cursor-pointer hover:bg-gray-600"
                >
                  <p className="text-sm text-gray-300">{section.title}</p>
                </div>
              ))}
            </div>
            {selectedSection && (
  <>
    <h3 className="text-xl font-bold mb-2 text-cyan-300">Section Notes</h3>
    <div className="mb-4 max-h-40 overflow-y-auto">
      {sectionNotes.length > 0 ? (
        sectionNotes.map((note, index) => (
          <div key={index} className="mb-2 p-2 bg-gray-700 rounded">
            <p className="text-sm text-gray-300">{note.tldr}</p>
          </div>
        ))
      ) : (
        <p className="text-gray-400">No notes available.</p> // Message for no notes
      )}
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
        onClick={() => handleClone(selectedSection.id)}
        className="px-4 py-2 bg-cyan-600 text-white rounded hover:bg-cyan-700 transition duration-300"
      >
        Clone Section
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

export default FriendsPage;