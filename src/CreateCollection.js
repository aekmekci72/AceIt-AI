import React, { useState } from 'react';
import axios from 'axios';

const CreateCollection = () => {
  const [collectionName, setCollectionName] = useState('');

  const handleCreateCollection = async () => {
    if (collectionName.trim() !== '') {
      try {
        await axios.post('http://localhost:5000/create_collection', { collection_name: collectionName });
        console.log('Collection created successfully!');
        setCollectionName('');
      } catch (error) {
        console.error('Error creating collection:', error);
      }
    }
  };

  return (
    <div>
      <input
        type="text"
        value={collectionName}
        onChange={(e) => setCollectionName(e.target.value)}
        placeholder="Enter collection name"
      />
      <button onClick={handleCreateCollection}>Create Collection</button>
    </div>
  );
};
