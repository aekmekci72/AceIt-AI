import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faTrash, faEdit } from '@fortawesome/free-solid-svg-icons';
import Navbar from './Navbar';
import LoadingSpinner from './LoadingSpinner';

const Upload = ({ onUploadSuccess }) => {
  const [rawText, setRawText] = useState('');
  const [response, setResponse] = useState('');

  const handleTextChange = (event) => {
    setRawText(event.target.value);
  };

  const handleSubmitText = async () => {
    try {
      const formData = new FormData();
      formData.append('raw_text', rawText);
      formData.append('collection_id', localStorage.getItem('currentCollection'));
      formData.append('section_id', localStorage.getItem('currentSection'));

      const response = await axios.post('http://localhost:5000/process_text', formData);
      setResponse(response.data.response);
      onUploadSuccess();
    } catch (error) {
      console.error('Error uploading text:', error);
    }
  };

  return (
    <div className="space-y-4">
      <textarea
        rows="4"
        value={rawText}
        onChange={handleTextChange}
        placeholder="Enter your text here..."
        className="w-full p-2 bg-gray-700 text-white rounded border border-gray-600 focus:border-cyan-500 focus:ring focus:ring-cyan-500 focus:ring-opacity-50"
      />
      <button 
        onClick={handleSubmitText} 
        className="w-full bg-cyan-600 hover:bg-cyan-700 text-white font-medium py-2 px-4 rounded transition duration-300"
      >
        Upload Text
      </button>
      {response && <p className="text-gray-300">{response}</p>}
    </div>
  );
};

const UploadVideo = ({ onUploadSuccess }) => {
  const [video, setVideo] = useState('');
  const [response, setResponse] = useState('');

  const handleTextChange = (event) => {
    setVideo(event.target.value);
  };

  const handleSubmitText = async () => {
    try {
      const formData = new FormData();
      formData.append('url', video);
      formData.append('collection_id', localStorage.getItem('currentCollection'));
      formData.append('section_id', localStorage.getItem('currentSection'));

      const response = await axios.post('http://localhost:5000/get_transcript', formData);
      setResponse(response.data.response);
      onUploadSuccess();
    } catch (error) {
      console.error('Error uploading text:', error);
    }
  };

  return (
    <div className="space-y-4">
      <input
        type="text"
        value={video}
        onChange={handleTextChange}
        placeholder="Enter URL here..."
        className="w-full p-2 bg-gray-700 text-white rounded border border-gray-600 focus:border-cyan-500 focus:ring focus:ring-cyan-500 focus:ring-opacity-50"
      />
      <button 
        onClick={handleSubmitText} 
        className="w-full bg-cyan-600 hover:bg-cyan-700 text-white font-medium py-2 px-4 rounded transition duration-300"
      >
        Upload URL
      </button>
      {response && <p className="text-gray-300">{response}</p>}
    </div>
  );
};

const UploadLink = ({ onUploadSuccess }) => {
  const [link, setLink] = useState('');
  const [response, setResponse] = useState('');

  const handleLinkChange = (event) => {
    setLink(event.target.value);
  };

  const handleUploadLink = async () => {
    try {
      const formData = new FormData();
      formData.append('link', link);
      formData.append('collection_id', localStorage.getItem('currentCollection'));
      formData.append('section_id', localStorage.getItem('currentSection'));

      const response = await axios.post('http://localhost:5000/process_link', formData);
      setResponse(response.data.response);
      onUploadSuccess();
    } catch (error) {
      console.error('Error uploading link:', error);
    }
  };

  return (
    <div className="space-y-4">
      <input
        type="text"
        value={link}
        onChange={handleLinkChange}
        placeholder="Enter website link here..."
        className="w-full p-2 bg-gray-700 text-white rounded border border-gray-600 focus:border-cyan-500 focus:ring focus:ring-cyan-500 focus:ring-opacity-50"
      />
      <button 
        onClick={handleUploadLink} 
        className="w-full bg-cyan-600 hover:bg-cyan-700 text-white font-medium py-2 px-4 rounded transition duration-300"
      >
        Upload Link
      </button>
      {response && <p className="text-gray-300">{response}</p>}
    </div>
  );
};

const UploadImage = ({ onUploadSuccess }) => {
  const [image, setImage] = useState('');
  const [response, setResponse] = useState('');

  const handleImageChange = (event) => {
    setImage(event.target.files[0]);
  };

  const handleUploadImage = async () => {
    try {
      const formData = new FormData();
      formData.append('image', image);
      formData.append('collection_id', localStorage.getItem('currentCollection'));
      formData.append('section_id', localStorage.getItem('currentSection'));

      const response = await axios.post('http://localhost:5000/recognize', formData);
      setResponse(response.data.response);
      onUploadSuccess();
    } catch (error) {
      console.error('Error uploading image:', error);
    }
  };

  return (
    <div className="space-y-4">
      <input 
        type="file" 
        onChange={handleImageChange} 
        className="w-full p-2 bg-gray-700 text-white rounded border border-gray-600 focus:border-cyan-500 focus:ring focus:ring-cyan-500 focus:ring-opacity-50"
      />
      <button 
        onClick={handleUploadImage} 
        className="w-full bg-cyan-600 hover:bg-cyan-700 text-white font-medium py-2 px-4 rounded transition duration-300"
      >
        Upload Image
      </button>
      {response && <p className="text-gray-300">{response}</p>}
    </div>
  );
};

const UploadPDF = ({ onUploadSuccess }) => {
  const [pdfFile, setPdfFile] = useState(null);
  const [response, setResponse] = useState('');

  const handleFileChange = (event) => {
    setPdfFile(event.target.files[0]);
  };

  const handleSubmitPDF = async () => {
    if (!pdfFile) {
      alert("Please select a PDF file to upload");
      return;
    }

    try {
      const formData = new FormData();
      formData.append('pdf_file', pdfFile);
      formData.append('collection_id', localStorage.getItem('currentCollection'));
      formData.append('section_id', localStorage.getItem('currentSection'));

      const response = await axios.post('http://localhost:5000/process_pdf', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResponse(response.data.response);
      onUploadSuccess();
    } catch (error) {
      console.error('Error uploading PDF:', error);
    }
  };

  return (
    <div className="space-y-4">
      <input 
        type="file" 
        accept="application/pdf" 
        onChange={handleFileChange} 
        className="w-full p-2 bg-gray-700 text-white rounded border border-gray-600 focus:border-cyan-500 focus:ring focus:ring-cyan-500 focus:ring-opacity-50"
      />
      <button 
        onClick={handleSubmitPDF} 
        className="w-full bg-cyan-600 hover:bg-cyan-700 text-white font-medium py-2 px-4 rounded transition duration-300"
      >
        Upload PDF
      </button>
      {response && <p className="text-gray-300">{response}</p>}
    </div>
  );
}
const ChapterPage = () => {
  const navigate = useNavigate();
  const chapterId = localStorage.getItem('currentSection');
  const collName = localStorage.getItem('collectionName');
  const chapterName = localStorage.getItem('currentSectionName');
  const collectionId = localStorage.getItem('currentCollection');
  
  const [sources, setSources] = useState([]);
  const [selectedSource, setSelectedSource] = useState([]);
  const [recognizedText, setRecognizedText] = useState('');
  const [recognizedVid, setRecognizedVid] = useState('');
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');
  const [isPublic, setIsPublic] = useState(true);
  const [noteIdBeingEdited, setNoteIdBeingEdited] = useState(null);
  const [isEditing, setIsEditing] = useState(false);
  const [editedNotes, setEditedNotes] = useState('');
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [uploadType, setUploadType] = useState(null);
  const [responseSaved, setResponseSaved] = useState(false);
  const [notes, setNotes] = useState([]);
  const [selectedSourceNotes, setSelectedSourceNotes] = useState('');
  const [flashcardSaved, setFlashcardSaved] = useState(false);
  const [loading, setLoading] = useState(false);
  const [worksheets, setWorksheets] = useState([]);
  const [uploadedWorksheet, setUploadedWorksheet] = useState(null);
  const [uploadWorksheetModalOpen, setUploadWorksheetModalOpen] = useState(false);
  const [startTime, setStartTime] = useState(null);
  const [endTime, setEndTime] = useState(null);
  const [totalTimeSpent, setTotalTimeSpent] = useState(0);
  const [componentInitialized, setComponentInitialized] = useState(false);
  
  useEffect(() => {
    let startTime = null;
    let endTime = null;
    // const timeout = setTimeout(() => {
    //   setComponentInitialized(true);
    // }, 2000);
    const fetchNotes = async () => {
      try {
        console.log(collectionId);
          console.log(chapterId);
          
        const response = await axios.get('http://localhost:5000/get_notes', {
          
          params: {
            collection_id: collectionId,
            section_id: chapterId,
          },
        });
        setNotes(response.data.notes);
      } catch (error) {
        console.error('Error:', error);
      }
    };
    const updateAccessTime = async () => {
      try {
        await axios.post('http://localhost:5000/update_access_time', {
          collection_id: collectionId,
          section_id: chapterId,
        });
      } catch (error) {
        console.error('Error updating access time:', error);
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
            section_id: chapterId,
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
            section_id: chapterId,
            total_time_spent: timeSpent,
          });
        } catch (error) {
          console.error('Error updating time spent:', error);
        }
      }
    };
    fetchNotes();
    window.addEventListener('beforeunload', handleBeforeUnload);
    window.addEventListener('unload', handleUnload);
  
    startTime = new Date().getTime();
    updateAccessTime();
  
    return () => {
      handleBeforeUnload();
      // clearTimeout(timeout);
    };
    
  }, [chapterId, collectionId]);
  
  const handleSourceChange = (source) => {
    setSelectedSource((prevState) =>
      prevState.includes(source) ? prevState.filter((s) => s !== source) : [...prevState, source]
    );
  };
  const handleUploadWorksheet = async () => {
    try {
      const formData = new FormData();
      formData.append('worksheet', uploadedWorksheet);
      formData.append('collection_id',collectionId);
      formData.append('section_id', chapterId);

      const response = await axios.post('http://localhost:5000/upload_worksheet', formData);
      console.log(response);
      setResponse(response.data.response);
      closeWorksheetModal();
    } catch (error) {
      console.error('Error uploading image:', error);
    }
  };


  const openUploadModal = () => {
    setUploadModalOpen(true);
  };

  const closeUploadModal = () => {
    setUploadModalOpen(false);
  };
  const openWorksheetModal = () => {
    setUploadWorksheetModalOpen(true);
  };

  const closeWorksheetModal = () => {
    setUploadWorksheetModalOpen(false);
  };

  const handleUpload = async (event) => {
    const formData = new FormData();
    if (uploadType === 'image') {
      formData.append('image', event.target.files[0]);
    } else if (uploadType === 'video') {
      formData.append('url', event.target.value);
    } else if (uploadType === 'text') {
      formData.append('raw_text', event.target.value);
    } else if (uploadType === 'link') {
      formData.append('link', event.target.value);
    } else if (uploadType === 'pdf') {
      formData.append('pdf_file', event.target.files[0]);
    }
    formData.append('collection_id', collectionId);
    formData.append('section_id', chapterId);

    try {
      let response;
      if (uploadType === 'image') {
        response = await axios.post('http://localhost:5000/recognize', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
      } else if (uploadType === 'video') {
        response = await axios.post('http://localhost:5000/get_transcript', formData);
      } else if (uploadType === 'text') {
        response = await axios.post('http://localhost:5000/process_text', formData);
      } else if (uploadType === 'link') {
        response = await axios.post('http://localhost:5000/process_link', formData);
      } else if (uploadType === 'pdf') {
        response = await axios.post('http://localhost:5000/process_pdf', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
      }
      const updatedNotesResponse = await axios.get('http://localhost:5000/get_notes', {
        params: {
          collection_id: collectionId,
          section_id: chapterId,
        },
      });
      
      setNotes(updatedNotesResponse.data.notes);
      closeUploadModal();
    } catch (error) {
      console.error(`Error uploading ${uploadType}:`, error);
    }
  };

  const handleUploadType = (type) => {
    setUploadType(type);
  };

  const formatBingResponse = (response) => {
    const paragraphs = response.split('\n');
    
    const formattedParagraphs = paragraphs.map((paragraph, index) => {
      const boldRegex = /\*\*(.*?)\*\*/g;
      let formattedParagraph = paragraph.replace(boldRegex, '<strong>$1</strong>');
  
      formattedParagraph = formattedParagraph.replace(/\[(.*?)\]\((.*?)\)/g, '\n<a href="$2">$1</a>');
  
      return <p key={index} dangerouslySetInnerHTML={{ __html: formattedParagraph }} />;
    });
  
    return formattedParagraphs;
  };
  

  const handleSubmitQuestion = async () => {
    try {
      setLoading(true);
      // const res = await axios.post('http://localhost:5000/answer_question', {
      //   username: localStorage.getItem('username'),
      //   class: localStorage.getItem('currentCollection'),
      //   data: prompt,
      // });
      // const data = {
      //   collection_id: collectionId,
      //   section_id: chapterId,
      //   question: prompt
      // };

      // const config = {
      //   headers: {
      //     'Content-Type': 'application/json',
      //   }
      // };

      const res = await axios.post('http://localhost:5000/conduct_conversation', 
        {
          collection_id: collectionId,
          section_id: chapterId,
          question: prompt,
        }, 
        {
            headers: {
                'Content-Type': 'application/json'
            }
        }
      );
      setResponse(res.data.response);
    } catch (error) {
      console.error('Error submitting question:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveResponse = async () => {
    try {
      await axios.post('http://localhost:5000/save_response', {
        question: prompt,
        response: response,
        collection_id: collectionId,
        section_id: chapterId,
      });
      setResponseSaved(true);
    } catch (error) {
      console.error('Error saving response:', error);
    }
  };

  const toggleVisibility = async () => {
    const newVisibility = isPublic ? 'private' : 'public';
    setIsPublic(!isPublic);

    try {
      const response = await axios.post('http://localhost:5000/section_visibility', {
        collection_id: collectionId,
        section_id: chapterId,
        visibility: newVisibility,
      });
      console.log('Visibility updated:', response.data);
    } catch (error) {
      console.error('Error updating visibility:', error);
    }
  };

  const addToFlashcards = async () => {
    try {
      const r = await axios.post('http://localhost:5000/add_res_to_flashcards', {
        question: prompt, // Use the user's question as the flashcard question
        answer: response, // Use the AI's response as the flashcard answer
        collection_id: collectionId,
        section_id: chapterId,
      });
      setFlashcardSaved(true);

      console.log('Flashcard added:', r.data);
    } catch (error) {
      console.error('Error adding to flashcards:', error);
    }
  };
  const handleEditNote = (note) => {
    setIsEditing(true);
    setEditedNotes(note.notes);
    setNoteIdBeingEdited(note.id);
  };

  const handleSubmitEdit = async () => {

    try {
      await axios.post('http://localhost:5000/edit_note', {
        collection_id: collectionId,
        section_id: chapterId,
        note_id: noteIdBeingEdited,
        new_notes: editedNotes,
      });

      const updatedNotesResponse = await axios.get('http://localhost:5000/get_notes', {
        params: {
          collection_id: collectionId,
          section_id: chapterId,
        },
      });
      setNotes(updatedNotesResponse.data.notes);
      setIsEditing(false);
      setEditedNotes('');
      setNoteIdBeingEdited(null);
    } catch (error) {
      console.error('Error editing note:', error);
    }
  };

  const handleDeleteNote = async (noteId) => {
    try {
      await axios.delete('http://localhost:5000/delete_note', {
        params: {
          collection_id: collectionId,
          section_id: chapterId,
          note_id: noteId,
        },
      });
      const updatedNotesResponse = await axios.get('http://localhost:5000/get_notes', {
        params: {
          collection_id: collectionId,
          section_id: chapterId,
        },
      });
      setNotes(updatedNotesResponse.data.notes);
    } catch (error) {
      console.error('Error deleting note:', error);
    }
  };
  const handleWorksheetChange = (event) => {
    setUploadedWorksheet(event.target.files[0]);
  };

  const handleDeleteWorksheet = async (noteId) => {
    try {
      await axios.delete('http://localhost:5000/delete_worksheet', {
        params: {
          collection_id: collectionId,
          section_id: chapterId,
          worksheet_id: noteId,
        },
      });
      const updatedNotesResponse = await axios.get('http://localhost:5000/get_worksheets', {
        params: {
          collection_id: collectionId,
          section_id: chapterId,
        },
      });
      setWorksheets(updatedNotesResponse.data.worksheets);
    } catch (error) {
      console.error('Error deleting note:', error);
    }
  };
  const handleViewSource = (note) => {
    setSelectedSourceNotes(note.notes);
    setNoteIdBeingEdited(note.id);
  };

  return (
    <div className="bg-gray-900 min-h-screen text-gray-100 overflow-hidden">
      <Navbar />
      <div className="flex h-[calc(100vh-64px)]"> {/* Adjust 64px to match your Navbar height */}
        <div className="w-1/5 fixed top-16 bottom-0 left-0 overflow-y-auto bg-gray-800 p-4">
          <h2 className="text-2xl font-bold mb-4 text-cyan-400">{collName} - {chapterName}</h2>
          <div className="mb-4">
            <button onClick={openUploadModal} className="w-full bg-cyan-600 hover:bg-cyan-700 text-white font-medium py-2 px-4 rounded transition duration-300">Upload Source</button>
          </div>
          <div className="mt-4">
            <h3 className="text-xl font-semibold mb-2 text-cyan-300">Notes</h3>
            {notes.map((note) => (
              <div key={note.id} className="mb-2 p-2 bg-gray-700 rounded">
                <p className="text-sm">{note.tldr}</p>
                <button onClick={() => handleViewSource(note)} className="text-cyan-400 hover:text-cyan-300 text-sm mr-2">View Source</button>
                <FontAwesomeIcon icon={faTrash} className="text-red-400 hover:text-red-300 cursor-pointer" onClick={() => handleDeleteNote(note.id)} />
              </div>
            ))}
          </div>
          <div className="flex mt-4">
            <button
              onClick={toggleVisibility}
              className={`flex-1 py-2 px-4 rounded-l ${isPublic ? 'bg-cyan-600 text-white' : 'bg-gray-700 text-gray-300'}`}
            >
              Public
            </button>
            <button
              onClick={toggleVisibility}
              className={`flex-1 py-2 px-4 rounded-r ${!isPublic ? 'bg-cyan-600 text-white' : 'bg-gray-700 text-gray-300'}`}
            >
              Private
            </button>
          </div>
        </div>
        
        <div className="flex-1 ml-[20%] p-4 overflow-y-auto">
          <div className="flex mb-4 bg-gray-800 rounded-lg overflow-hidden">
            {['Saved Responses', 'Flashcards', 'Videos', 'Retention Test', 'Quiz Me'].map((tab) => (
              <button
                key={tab}
                onClick={() => navigate(`/${tab.toLowerCase().replace(' ', '')}`)}
                className="flex-1 py-2 px-4 bg-gray-700 hover:bg-gray-600 text-gray-300 hover:text-white transition duration-300"
              >
                {tab}
              </button>
            ))}
          </div>
          <div className="bg-gray-800 p-4 rounded-lg">
            <h3 className="text-xl font-semibold mb-2 text-cyan-300">Ask A Question</h3>
            <textarea
              rows="4"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter your prompt here..."
              className="w-full p-2 mb-4 bg-gray-700 text-white rounded border border-gray-600 focus:border-cyan-500 focus:ring focus:ring-cyan-500 focus:ring-opacity-50"
            />
            <button
              onClick={handleSubmitQuestion}
              disabled={loading || responseSaved}
              className={`py-2 px-4 rounded ${loading || responseSaved ? 'bg-gray-600 cursor-not-allowed' : 'bg-cyan-600 hover:bg-cyan-700'} text-white transition duration-300`}
            >
              {loading ? 'Generating...' : 'Submit'}
            </button>
            {response && (
              <div className="mt-4">
                <h2 className="text-lg font-semibold mb-2">Response:</h2>
                <p className="mb-4">{formatBingResponse(response)}</p>
                <button
                  onClick={handleSaveResponse}
                  disabled={responseSaved}
                  className={`mr-2 py-2 px-4 rounded ${responseSaved ? 'bg-gray-600 cursor-not-allowed' : 'bg-cyan-600 hover:bg-cyan-700'} text-white transition duration-300`}
                >
                  {responseSaved ? 'Saved!' : 'Save Response'}
                </button>
                <button
                  onClick={addToFlashcards}
                  disabled={flashcardSaved}
                  className={`py-2 px-4 rounded ${flashcardSaved ? 'bg-gray-600 cursor-not-allowed' : 'bg-cyan-600 hover:bg-cyan-700'} text-white transition duration-300`}
                >
                  {flashcardSaved ? 'Saved!' : 'Add to Flashcards'}
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {selectedSourceNotes && (
        <div
          className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center"
          onClick={() => setSelectedSourceNotes('')}
        >
          <div
            className="bg-gray-800 text-white p-6 rounded-lg max-w-2xl max-h-[80vh] overflow-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="text-2xl font-bold mb-4 text-gray-200 text-center">Full Contents</h2>
            <FontAwesomeIcon
              icon={faEdit}
              className="text-cyan-400 hover:text-cyan-500 cursor-pointer float-right"
              onClick={() => handleEditNote({ id: noteIdBeingEdited, notes: selectedSourceNotes })}
            />
            {isEditing ? (
              <div>
                <textarea
                  value={editedNotes}
                  onChange={(e) => setEditedNotes(e.target.value)}
                  className="w-full h-48 p-2 bg-gray-700 border border-gray-600 rounded text-white placeholder-gray-400"
                  placeholder="Edit notes here..."
                />
                <button
                  onClick={handleSubmitEdit}
                  className="mt-2 bg-cyan-500 hover:bg-cyan-600 text-white font-bold py-2 px-4 rounded"
                >
                  Submit
                </button>
              </div>
            ) : (
              <p className="text-gray-300">{selectedSourceNotes}</p>
            )}
          </div>
        </div>
      )} 

      {uploadModalOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center"
          onClick={closeUploadModal}
        >
          <div
            className="bg-gray-800 text-white p-6 rounded-lg"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="text-2xl font-bold mb-4 text-gray-200">Upload Source</h2>
            <div className="flex flex-wrap gap-2 mb-4">
              {['Image', 'Video', 'Text', 'Link', 'PDF'].map((type) => (
                <button
                  key={type}
                  onClick={() => handleUploadType(type.toLowerCase())}
                  className="bg-cyan-500 hover:bg-cyan-600 text-white font-bold py-2 px-4 rounded"
                >
                  Upload {type}
                </button>
              ))}
            </div>
            {uploadType && (
              <div className="mt-4">
                {uploadType === 'text' && <Upload onUploadSuccess={closeUploadModal} />}
                {uploadType === 'link' && <UploadLink onUploadSuccess={closeUploadModal} />}
                {uploadType === 'pdf' && <UploadPDF onUploadSuccess={closeUploadModal} />}
                {uploadType === 'image' && <UploadImage onUploadSuccess={closeUploadModal} />}
                {uploadType === 'video' && <UploadVideo onUploadSuccess={closeUploadModal} />}
              </div>
            )}
          </div>
        </div>
      )}


    
    </div>
  );
};


export default ChapterPage;
