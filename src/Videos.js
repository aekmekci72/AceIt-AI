import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Modal from 'react-modal';
import './styles/stuff.css';
import Navbar from './Navbar';

Modal.setAppElement('#root');

const VideoPage = () => {
  const collectionId = localStorage.getItem('currentCollection');
  const sectionId = localStorage.getItem('currentSection');
  const [videoPaths, setVideoPaths] = useState([]);
  const [loading, setLoading] = useState(false);
  const [notes, setNotes] = useState([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedNotes, setSelectedNotes] = useState([]);
  const [askSpecificResponse, setAskSpecificResponse] = useState(null);
  const [startTime, setStartTime] = useState(null);
  const [endTime, setEndTime] = useState(null);
  const [totalTimeSpent, setTotalTimeSpent] = useState(0);
  
  useEffect(() => {
    let startTime = null;
    let endTime = null;
  
    const fetchVideoPaths = async () => {
      try {
        const response = await axios.get('http://localhost:5000/get_video_paths', {
          params: {
            collection_id: collectionId,
            section_id: sectionId,
          },
        });
        setVideoPaths(response.data.videoPaths);
      } catch (error) {
        console.error('Error fetching video paths:', error);
      }
    };

    const fetchNotes = async () => {
      try {
        const response = await axios.get('http://localhost:5000/get_notes', {
          params: {
            collection_id: collectionId,
            section_id: sectionId,
          },
        });
        setNotes(response.data.notes);
      } catch (error) {
        console.error('Error fetching notes:', error);
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
            section_id: sectionId,
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
            section_id: sectionId,
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
    

    fetchVideoPaths();
    fetchNotes();
    return () => {
      handleBeforeUnload();
    };
  }, [collectionId, sectionId]);

  const handleGenerateVideo = async () => {
    setLoading(true);
    try {
      const concatenatedNotes = selectedNotes.join(' ');
      const generateVideoResponse = await axios.post('http://localhost:5000/generate_video_from_notes', {
        notes: concatenatedNotes,
        collection_id: collectionId,
        section_id: sectionId,
      });
      
      if (generateVideoResponse.video_path) {
        setVideoPaths(prevPaths => [...prevPaths, { path: generateVideoResponse.video_path }]);
      }
    } catch (error) {
      console.error('Error generating video:', error);
    } finally {
      setLoading(false);
      setIsModalOpen(false);
    }
  };

  const handleNoteSelection = (noteId) => {
    setSelectedNotes(prevState =>
      prevState.includes(noteId)
        ? prevState.filter(id => id !== noteId)
        : [...prevState, noteId]
    );
  };

  return (
    <div className="bg-gray-900 text-gray-100 font-sans min-h-screen overflow-y-auto">
      <Navbar />
      <div className="container mx-auto px-4 py-8 flex flex-col items-center">
        <h1 className="text-6xl md:text-7xl font-bold mb-12 text-cyan-400 text-center">Video Page</h1>
        
        <div className="w-full max-w-md mb-8">
          <button 
            onClick={() => setIsModalOpen(true)} 
            disabled={loading}
            className="w-full bg-cyan-600 hover:bg-cyan-700 text-white font-medium py-3 px-4 rounded-lg transition duration-300 ease-in-out transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Generating...' : 'Generate New Video'}
          </button>
        </div>

        <div className="w-full grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {videoPaths.map((video, index) => (
            <div 
              key={index} 
              onClick={() => window.location.href = `http://localhost:3000/videoplayer/${video.video_path}`}
              className="bg-gray-800 rounded-lg shadow-lg hover:shadow-cyan-500/50 transition duration-300 cursor-pointer overflow-hidden"
            >
              <video preload="metadata" className="w-full h-48 object-cover">
                <source src={`http://localhost:5000/videos/${video.video_path}`} type="video/mp4" />
                Your browser does not support the video tag.
              </video>
              <div className="p-4">
              </div>
            </div>
          ))}
        </div>

        <Modal
          isOpen={isModalOpen}
          onRequestClose={() => setIsModalOpen(false)}
          contentLabel="Select Notes"
          className="bg-gray-800 rounded-lg p-8 max-w-md mx-auto mt-20"
          overlayClassName="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center"
        >
          <h2 className="text-2xl font-bold mb-6 text-cyan-400">Select Notes</h2>
          <div className="mb-6 max-h-60 overflow-y-auto">
            {notes.map(note => (
              <div key={note.id} className="mb-2">
                <label className="flex items-center space-x-2 text-gray-300">
                  <input
                    type="checkbox"
                    checked={selectedNotes.includes(note.notes)}
                    onChange={() => handleNoteSelection(note.notes)}
                    className="form-checkbox h-5 w-5 text-cyan-600"
                  />
                  <span>{note.tldr}</span>
                </label>
              </div>
            ))}
          </div>
          <div className="flex justify-end space-x-4">
            <button 
              onClick={() => setIsModalOpen(false)}
              className="bg-red-600 hover:bg-red-700 text-white font-medium py-2 px-4 rounded-lg transition duration-300 ease-in-out transform hover:scale-105"
            >
              Close
            </button>
            <button 
              onClick={handleGenerateVideo} 
              disabled={loading}
              className="bg-cyan-600 hover:bg-cyan-700 text-white font-medium py-2 px-4 rounded-lg transition duration-300 ease-in-out transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Generating...' : 'Generate Video'}
            </button>
          </div>
        </Modal>
      </div>
    </div>
  );
};

export default VideoPage;