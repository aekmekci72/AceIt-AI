import React from 'react';
import CreateCollection from './CreateCollection';
import CollectionGallery from './CollectionGallery';
import ScanNotes from './ScanNotes';
import axios from 'axios';
import { useState } from 'react';

const Upload = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [recognizedText, setRecognizedText] = useState('');
  const [selectedURL, setSelectedURL] = useState(null);
  const [recognizedVid, setRecognizedVid] = useState('');

  const handleImageChange = (event) => {
    setSelectedImage(event.target.files[0]);
  };
  const handleURLChange = (event) => {
    setSelectedURL(event.target.value);
  };
  


  const handleUploadImage = async () => {
    const formData = new FormData();
    formData.append('image', selectedImage);

    try {
      const response = await axios.post('http://localhost:5000/recognize', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setRecognizedText(response.data.text);
    } catch (error) {
      console.error('Error:', error);
    }
  };
  const handleUploadVideo = async () => {
    const formData = new FormData();
    formData.append('url', selectedURL);
  
    try {
      const response = await axios.post('http://localhost:5000/get_transcript', formData);
      setRecognizedVid(response.data.response);
    } catch (error) {
      console.error('Error:', error);
    }
  };
  

  return (
    <div>
      <input type="file" accept="image/*" onChange={handleImageChange} />
      <button onClick={handleUploadImage}>Upload Image</button>
      {recognizedText && <p>Recognized Text: {recognizedText}</p>}
        <input type="text"  onChange={handleURLChange}/>
        <button onClick={handleUploadVideo}>Upload YouTube Video</button>
        {recognizedVid && <p>Recognized Vid: {recognizedVid}</p>}

    </div>
  );
};

export default Upload;
