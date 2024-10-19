from flask import Flask, jsonify, request
import firebase_admin
from firebase_admin import credentials, firestore
from flask_cors import CORS
import logging
import requests
import re

# Initialize Flask app
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)

# Initialize Firebase Admin SDK
cred = credentials.Certificate("serviceKey.json")
firebase_admin.initialize_app(cred)

# Initialize Firestore client
db = firestore.client()

@app.route('/test_get_my_collections')
def test_get_my_collections():
    try:
        username = request.args.get('username')
        if not username:
            return jsonify({'error': 'Username not provided'})

        collections = []
        collection_docs = db.collection('collections').where('username', '==', username).stream()
        for doc in collection_docs:
            title = doc.to_dict().get('data', {}).get('title', '')
            collections.append({'id': doc.id, 'title': title})

        return jsonify({'collections': collections})
    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
