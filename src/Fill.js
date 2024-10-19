import React, { useState } from 'react';

const FillerAuth = () => {
  const [username, setUsername] = useState('');

  const handleLogin = () => {
    if (username.trim() !== '') {
      localStorage.setItem('username', username);
      window.location.href = '/mygallery'; // Redirect to the "/question" page
    } else {
      alert('Please enter a username');
    }
  };

  return (
    <div>
      <h2>Login/Signup</h2>
      <input
        type="text"
        placeholder="Enter your username"
        value={username}
        onChange={(e) => setUsername(e.target.value)}
      />
      <br />
      <button onClick={handleLogin}>Login/Signup</button>
    </div>
  );
};

export default FillerAuth;