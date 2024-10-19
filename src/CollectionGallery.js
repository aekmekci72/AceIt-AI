import React, { useState, useEffect } from 'react';
import axios from 'axios';

const CollectionGallery = () => {
  const [collections, setCollections] = useState([]);

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

  return (
    <div>
      <h2>Collection Gallery</h2>
      <ul>
        {collections.map((collection, index) => (
          <li key={index}>{collection}</li>
        ))}
      </ul>
    </div>
  );
};

export default CollectionGallery;
// import React, { useState, useEffect } from 'react';
// import axios from 'axios';

// const CollectionGallery = () => {
//   const [collections, setCollections] = useState([]);

//   useEffect(() => {
//     const fetchCollections = async () => {
//       try {
//         const response = await axios.get('http://127.0.0.1:5000/get_my_collections');
//         console.log(response.data);
//         setCollections(response.data.collections);
//       } catch (error) {
//         console.error('Error fetching collections:', error);
//       }
//     };

//     fetchCollections();
//   }, []);

//   return (
//     <div>
//       <h2>Collection Gallery</h2>
//       <ul>
//         {collections.map((collection, index) => (
//           <li key={index}>{collection}</li>
//         ))}
//       </ul>
//     </div>
//   );
// };


// export default CollectionGallery;
