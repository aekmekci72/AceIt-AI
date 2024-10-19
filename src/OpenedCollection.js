import React, { useState } from 'react';
import axios from 'axios';
const OpenedCollection = () => {
  const collectionName = localStorage.getItem('collectionName');

  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');

  const handleSubmit = async () => {
    try {
      const res = await axios.post('http://localhost:5000/answer_question', { username: localStorage.getItem('username'),  class: localStorage.getItem('currentCollection'), data: prompt });
      setResponse(res.data.response);
    } catch (error) {
      console.error('Error:', error);
    }
  };
  const handleUploadDataPage = () => {
    window.location.href = "/uploaddata";
  };

  return (
    <div>
      <h2>{collectionName}</h2>
      <div>
      <button onClick={() => handleUploadDataPage()}>Upload Data</button>
      <h1>Ask A Question (this is based on user notes from the collection)</h1>
      <div>
        <textarea
          rows="4"
          cols="50"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter your prompt here..."
        />
      </div>
      <div>
        <button onClick={handleSubmit}>Submit</button>
      </div>
      {response && (
        <div>
          <h2>Response:</h2>
          <p>{response}</p>
        </div>
      )}
    </div>
    </div>
  );
};

export default OpenedCollection;
