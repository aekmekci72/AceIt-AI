import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import Navbar from './Navbar';
import LoadingSpinner from './LoadingSpinner';

const HomePage = () => {
  const navigate = useNavigate();
  const [recentSections, setRecentSections] = useState([]);
  const [loading, setLoading] = useState(true);
  const username = localStorage.getItem('userName');

  useEffect(() => {
    const fetchData = async () => {
      try {
        const recentResponse = await axios.get('http://localhost:5000/get_my_sections_recent', {
          params: { username: username }
        });
        const sortedSections = recentResponse.data.collections.sort((a, b) => 
          a.title.localeCompare(b.title, undefined, { sensitivity: 'base' })
        );
        setRecentSections(sortedSections);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching data:', error);
        setLoading(false);
      }
    };

    fetchData();
  }, [username]);

  const handleSectionClick = (sectionId, collId, title, colName) => {
    localStorage.setItem('chapterId', sectionId);
    localStorage.setItem('collectionId', collId);
    localStorage.setItem('collectionName', colName);
    localStorage.setItem('currentSectionName', title);
    navigate(`/chapter`);
  };

  const handleCreateCollection = () => {
    navigate('/mygallery'); 
  };

  return (
    <div className="bg-gray-900 text-gray-100 font-sans min-h-screen flex flex-col">
      <Navbar />
      <div className="flex-1 overflow-y-auto">
        <div className="container mx-auto px-4 py-8 flex flex-col items-center">
          <h1 className="text-6xl md:text-7xl font-bold mb-6 text-cyan-400 text-center">AceIt</h1>
          <p className="text-xl text-gray-300 mb-12 text-center">
            Welcome to AceIt AI, your learning companion.
          </p>
          {loading ? (
            <LoadingSpinner />
          ) : (
            <div className="w-full max-w-6xl">
              <h2 className="text-3xl font-bold mb-6 text-cyan-300 text-center">Recently Viewed</h2>
              <div className="flex flex-wrap justify-center gap-6 pb-24">
                {recentSections.length > 0 ? (
                  recentSections.map(section => (
                    <div 
                      key={section.id} 
                      onClick={() => handleSectionClick(section.id, section.collection_id, section.title, section.collName)}
                      className="bg-gray-800 p-4 rounded-lg shadow-lg hover:shadow-cyan-500/50 transition duration-300 cursor-pointer h-32 w-48 flex flex-col justify-center"
                    >
                      <h3 className="text-lg font-semibold mb-2 text-cyan-400 text-left">{section.title || 'Untitled'}</h3>
                      <p className="text-gray-400 text-sm text-left">{section.collName || 'No collection'}</p>
                    </div>
                  ))
                ) : (
                  <div className="flex flex-col items-center">
                    <p className="text-xl text-gray-400 mb-6">No recently viewed sections found.</p>
                    <button
                      onClick={handleCreateCollection}
                      className="bg-cyan-600 hover:bg-cyan-700 text-white font-bold py-2 px-4 rounded transition duration-300"
                    >
                      Create Your First Collection
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default HomePage;