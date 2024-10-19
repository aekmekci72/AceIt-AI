// import React, { useState } from 'react';
// import axios from 'axios';

// const App = () => {
//   const [selectedImage, setSelectedImage] = useState(null);
//   const [recognizedText, setRecognizedText] = useState('');

//   const handleImageChange = (event) => {
//     setSelectedImage(event.target.files[0]);
//   };

//   const handleUploadImage = async () => {
//     const formData = new FormData();
//     formData.append('image', selectedImage);

//     try {
//       const response = await axios.post('http://localhost:5000/recognize', formData, {
//         headers: {
//           'Content-Type': 'multipart/form-data'
//         }
//       });
//       setRecognizedText(response.data.text);
//     } catch (error) {
//       console.error('Error:', error);
//     }
//   };

//   return (
//     <div>
//       <input type="file" accept="image/*" onChange={handleImageChange} />
//       <button onClick={handleUploadImage}>Upload Image</button>
//       {recognizedText && <p>Recognized Text: {recognizedText}</p>}
//     </div>
//   );
// };

// export default App;

// import React, { useState, useEffect } from 'react';
// import axios from 'axios';

// const ScanNotes = () => {
//   const [selectedImage, setSelectedImage] = useState(null);
//   const [recognizedText, setRecognizedText] = useState('');
//   const [collections, setCollections] = useState([]);
//   const [selectedCollection, setSelectedCollection] = useState('');

//   useEffect(() => {
//     const fetchCollections = async () => {
//       try {
//         const response = await axios.get('http://localhost:5000/get_collections');
//         setCollections(response.data.collections);
//       } catch (error) {
//         console.error('Error fetching collections:', error);
//       }
//     };

//     fetchCollections();
//   }, []);

//   const handleImageChange = (event) => {
//     setSelectedImage(event.target.files[0]);
//   };

//   const handleUploadImage = async () => {
//     if (!selectedImage || !selectedCollection) {
//       alert('Please select an image and a collection.');
//       return;
//     }

//     const formData = new FormData();
//     formData.append('image', selectedImage);
//     formData.append('collection', selectedCollection);

//     try {
//       const response = await axios.post('http://localhost:5000/recognize', formData, {
//         headers: {
//           'Content-Type': 'multipart/form-data'
//         }
//       });
//       setRecognizedText(response.data.text);
//     } catch (error) {
//       console.error('Error:', error);
//     }
//   };

//   return (
//     <div>
//       <input type="file" accept="image/*" onChange={handleImageChange} />
//       <select value={selectedCollection} onChange={(e) => setSelectedCollection(e.target.value)}>
//         <option value="">Select Collection</option>
//         {collections.map((collection, index) => (
//           <option key={index} value={collection}>{collection}</option>
//         ))}
//       </select>
//       <button onClick={handleUploadImage}>Upload Image</button>
//       {recognizedText && <p>Recognized Text: {recognizedText}</p>}
//     </div>
//   );
// };

// export default ScanNotes;

import React, { useState, useEffect } from 'react';
import axios from 'axios';

const ScanNotes = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [recognizedText, setRecognizedText] = useState('');
  const [collections, setCollections] = useState([]);
  const [selectedCollection, setSelectedCollection] = useState('');

  useEffect(() => {
    const fetchCollections = async () => {
      try {
        const response = await axios.get('http://localhost:5000/get_collections');
        setCollections(response.data.collections);
      } catch (error) {
        console.error('Error fetching collections:', error);
      }
    };

    fetchCollections();
  }, []);

  const handleImageChange = (event) => {
    setSelectedImage(event.target.files[0]);
  };

  const handleUploadImage = async () => {
    if (!selectedImage || !selectedCollection) {
      alert('Please select an image and a collection.');
      return;
    }

    const formData = new FormData();
    formData.append('image', selectedImage);
    formData.append('collection', selectedCollection);

    try {
      const response = await axios.post('http://localhost:5000/recognize', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setRecognizedText(response.data.text);
      console.log(response.data.text);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div>
      <input type="file" accept="image/*" onChange={handleImageChange} />
      <select value={selectedCollection} onChange={(e) => setSelectedCollection(e.target.value)}>
        <option value="">Select Collection</option>
        {collections.map((collection, index) => (
          <option key={index} value={collection}>{collection}</option>
        ))}
      </select>
      <button onClick={handleUploadImage}>Upload Image</button>
      {recognizedText && <p>Recognized Text: {recognizedText}</p>}
    </div>
  );
};

export default ScanNotes;
