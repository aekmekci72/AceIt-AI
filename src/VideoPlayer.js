import React, { useState, useRef, useEffect } from 'react';
import { useParams } from 'react-router-dom';

const VideoPlayer = () => {
  const { videoPath } = useParams();
  const videoRef = useRef(null);
  const [captions, setCaptions] = useState('');
  const [captionsEnabled, setCaptionsEnabled] = useState(false);
  const recognitionRef = useRef(null);

  useEffect(() => {
    if ('webkitSpeechRecognition' in window) {
      const SpeechRecognition = window.webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;

      recognitionRef.current.onresult = (event) => {
        let interimTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            setCaptions((prev) => prev + transcript + ' ');
          } else {
            interimTranscript += transcript;
          }
        }
        document.getElementById('caption-display').textContent = captions + interimTranscript;
      };
    } else {
      console.log('Speech recognition not supported');
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, []);

  const toggleCaptions = () => {
    if (captionsEnabled) {
      recognitionRef.current.stop();
    } else {
      recognitionRef.current.start();
    }
    setCaptionsEnabled(!captionsEnabled);
  };

  return (
    <div className="bg-gray-900 text-gray-100 font-sans min-h-screen p-4">
      <div className="max-w-4xl mx-auto">
        <div className="relative">
          <video 
            ref={videoRef}
            controls 
            className="w-full rounded-lg shadow-lg"
          >
            <source src={`http://localhost:5000/videos/${videoPath}`} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
          {captionsEnabled && (
            <div 
              id="caption-display"
              className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-50 text-white p-2 text-center"
            ></div>
          )}
        </div>
        <button 
          onClick={toggleCaptions}
          className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition duration-300"
        >
          {captionsEnabled ? 'Disable Captions' : 'Enable Captions'}
        </button>
      </div>
    </div>
  );
};

export default VideoPlayer;