import React, { useState } from 'react';
import axios from 'axios';

const App = () => {
    const [uploadedFiles, setUploadedFiles] = useState([]);
    const [dbInitialized, setDbInitialized] = useState(false);
    const [qaInitialized, setQaInitialized] = useState(false);
    const [conversationResponse, setConversationResponse] = useState('');
    const [sourceDocuments, setSourceDocuments] = useState([]);

    const handleFileUpload = async (event) => {
        const formData = new FormData();
        for (const file of event.target.files) {
            formData.append('file', file);
        }
        try {
            const response = await axios.post('http://localhost:5000/upload', formData);
            setUploadedFiles(response.data.uploaded_files);
        } catch (error) {
            console.error('Error uploading files: ', error);
        }
    };

    const initializeDatabase = async () => {
        try {
            const response = await axios.post('http://localhost:5000/initialize_db');
            setDbInitialized(true);
            console.log(response.data.status);
        } catch (error) {
            console.error('Error initializing database: ', error);
        }
    };

    const initializeQAChain = async () => {
        try {
            const response = await axios.post('http://localhost:5000/initialize_qa');
            setQaInitialized(true);
            console.log(response.data.status);
        } catch (error) {
            console.error('Error initializing QA chain: ', error);
        }
    };

    const handleConversation = async (message) => {
        try {
            const response = await axios.post('http://localhost:5000/conversation', { message });
            setConversationResponse(response.data.response);
            setSourceDocuments(response.data.source_documents);
        } catch (error) {
            console.error('Error during conversation: ', error);
        }
    };

    return (
        <div>
            {/* File Upload Section */}
            <input type="file" multiple onChange={handleFileUpload} />
            {uploadedFiles.length > 0 && (
                <div>
                    <h2>Uploaded Files:</h2>
                    <ul>
                        {uploadedFiles.map((file, index) => (
                            <li key={index}>{file}</li>
                        ))}
                    </ul>
                    <button onClick={initializeDatabase}>Generate Vector Database</button>
                </div>
            )}

            {/* QA Chain Initialization Section */}
            {dbInitialized && (
                <div>
                    <h2>Vector Database Initialized</h2>
                    <button onClick={initializeQAChain}>Initialize QA Chain</button>
                </div>
            )}

            {/* Conversation Section */}
            {qaInitialized && (
                <div>
                    <h2>Conversation</h2>
                    <input type="text" placeholder="Type your message" />
                    <button onClick={() => handleConversation('Sample message')}>
                        Submit
                    </button>
                    <h3>Response:</h3>
                    <p>{conversationResponse}</p>
                    <h3>Source Documents:</h3>
                    <ul>
                        {sourceDocuments.map((doc, index) => (
                            <li key={index}>
                                <strong>Page {doc.page}:</strong> {doc.content}
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
};

export default App;
