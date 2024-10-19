from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore
from PIL import Image
import pytesseract
from rake_nltk import Rake
from bson import ObjectId
from datetime import datetime, timedelta

import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError
from bs4 import BeautifulSoup
import requests
from typing import OrderedDict
from transformers import T5ForConditionalGeneration, AutoTokenizer
from transformers import pipeline
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.llms import HuggingFaceEndpoint
from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
import requests
from duckduckgo_search import DDGS
import re
import io
import logging
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from gensim import corpora, models
import re
import os
import asyncio
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet
import random
import PyPDF2
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import os
from sydney import SydneyClient
from dotenv import load_dotenv
import firebase_admin
from transformers import pipeline
from typing import OrderedDict
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer, BertTokenizer, BertModel, AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import spacy
from transformers import BertTokenizer as BTokenizer, BertModel as BModel
from warnings import filterwarnings as ignore_warnings
from transformers import pipeline
from huggingface_hub import login
from langchain_community.llms import HuggingFaceEndpoint
from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
import requests
from duckduckgo_search import DDGS
import re
import os
from PIL import Image, ImageDraw, ImageFont
from moviepy.video.fx import resize, crop
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

import requests
from firebase_admin import auth
import googleapiclient.discovery
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from typing import OrderedDict
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer, BertTokenizer, BertModel, AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import spacy
from transformers import BertTokenizer as BTokenizer, BertModel as BModel
from warnings import filterwarnings as ignore_warnings
from transformers import pipeline
import uuid
import re
import requests
from datetime import datetime
from nltk.tokenize import sent_tokenize
from sumy.summarizers.text_rank import TextRankSummarizer
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from fastcore.all import *
from gtts import gTTS
from moviepy.editor import ImageSequenceClip, AudioFileClip, concatenate_videoclips, concatenate_audioclips
# from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from heapq import nlargest
import sys
import json
import wikipediaapi
from collections import Counter
import time
import requests
import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceEndpoint
import chromadb
from unidecode import unidecode
global qachain

# from pathlib import Path
# import chromadb
# from unidecode import unidecode

# from transformers import AutoTokenizer
# import transformers
# import torch
# import tqdm 
# import accelerate
# import re

# import os
qachain = None




app = Flask(__name__)
CORS(app)  # Apply CORS to your Flask app

VIDEO_DIRECTORY = os.path.join(os.getcwd(), './vids')

load_dotenv()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


bing_cookies_key = os.getenv('BING_COOKIES')
transcript_key = os.getenv('TRANSCRIPT_API')

if bing_cookies_key is None:
    print("Error: BING_COOKIES environment variable is not set.")
    exit(1)

os.environ["BING_COOKIES"] = bing_cookies_key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv('HUGGINGFACE')

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize Firebase Admin SDK
cred = credentials.Certificate("serviceKey.json")
firebase_admin.initialize_app(cred)

# Initialize Firestore client
db = firestore.client()

qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')
t5_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
t5_tokenizer = AutoTokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
sentence_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
nlp = spacy.load("en_core_web_sm")
model_name = "t5-small"  # You can use "t5-base" for better quality but slower processing
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


ignore_warnings('ignore')

qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

# def summarize_text(text, num_sentences=1):
#     return generate_single_sentence_summary(text, max_length=50, min_length=30)



def generate_single_sentence_summary(text, max_length=50, min_length=30):
 
    # Prepare the text input
    preprocess_text = text.strip().replace("\n", "")
    t5_prepared_Text = "summarize: " + preprocess_text

    # Tokenize the input text
    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt", max_length=512, truncation=True)

    # Generate summary
    summary_ids = model.generate(
        tokenized_text,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Ensure the summary is a single sentence
    summary_sentences = summary.split('. ')
    single_sentence_summary = summary_sentences[0] if summary_sentences else summary

    return single_sentence_summary

list_llm = ["mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mixtral-8x7B-Instruct-v0.1", "mistralai/Mistral-7B-Instruct-v0.1", \
    "google/gemma-7b-it","google/gemma-2b-it", \
    "HuggingFaceH4/zephyr-7b-beta", "HuggingFaceH4/zephyr-7b-gemma-v0.1", \
    "meta-llama/Llama-2-7b-chat-hf", "microsoft/phi-2", \
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "mosaicml/mpt-7b-instruct", "tiiuae/falcon-7b-instruct", \
    "google/flan-t5-xxl", "mistralai/Mistral-7B-Instruct-v0.3","mistralai/Mistral-7B-Instruct-v0.1",
]
default_llm_index = 0
llm_model = list_llm[default_llm_index]
llm = HuggingFaceEndpoint(
    repo_id=llm_model,
    temperature=0.5,  
    max_new_tokens=5000,  
    top_k=20,
    request_timeout=300
)

async def chat(question, notes):
    prompt = f"""
    {question}
    Answer this question based on my notes. If my notes are missing anything, you can elaborate further.
    Notes: {notes}
    Make sure the response starts with a topic sentence that sumarizes the rest of what you will be talking about to make it more understandable for me.
    """
    
    response = llm(prompt)
    return response

@app.route('/conduct_conversation', methods=['POST'])
async def conduct_conversation_endpoint():
    try:
        data = request.get_json()
        collection_id = data.get('collection_id', '')
        section_id = data.get('section_id', '')

        question = data.get('question', '')

        notes_li = get_notess(collection_id, section_id)

        if not notes_li:
            notes = '*'
        else:
            notes = ' '.join(notes_li)

        response = await chat(question, notes)
        
        return jsonify({
            'success': True,
            'response': response
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })




async def parse_questions_and_answers(response):
    questions = re.findall(r'Question\s*\d+:\s*(.*?)(?=Question\s*\d+:|$)', response, re.DOTALL)
    
    questions = [q.strip() for q in questions]
    
    if(len(questions)==0):
        questions = re.findall(r'\d+\.\s*(.*?)(?=\d+\.\s*|$)', response, re.DOTALL)
        
        questions = [q.strip() for q in questions]
        
    if(len(questions)==0):
        questions = re.findall(r'Question:\s*(.*?)(?=Question:|$)', response, re.DOTALL)
    
        questions = [q.strip() for q in questions]
        
    if len(questions) == 0:
        questions = re.findall(r'\d+\.\s*(.*?)(?=\d+\.|$)', response, re.DOTALL)
        questions = [q.strip() for q in questions]
    if len(questions) ==0:
        questions = re.findall(r'\d+\.\s*Question:\s*(.*?)(?=\d+\.\s*Question:|$)', response, re.DOTALL)
        questions = [q.strip() for q in questions]
    if len(questions) ==0:
        questions=[]
        questions[0]=response
    for i, question in enumerate(questions):
        if "Answer" in question:
            questions[i] = question.split("Answer")[0].strip()
   


    return questions

async def generate_qs(text, num_flashcards=10):
    prompt = f"""
    Generate {num_flashcards} questions based on the following notes. These questions should be designed to test a user's understanding of the content before a test or exam on the topic. The questions should range from conceptual questions to more specific, detailed questions. Ensure the questions cover all key topics and concepts mentioned in the notes. Return the questions in a format where each question is prefixed with 'Question:' for easy parsing. Please ensure this standard format. Do not return answers to the questions.

    These are the notes:
    {text}
    """
    
    response = llm(prompt)
    qs = await parse_questions_and_answers(response)
    return qs

@app.route('/get_questions', methods=['POST'])
async def get_questions():
    data = request.json

    collection_id = data.get('collection_id', '')  
    section_id = data.get('section_id', '')  

    notes_li = get_notess(collection_id, section_id)

    if not notes_li:
        notes = '*'
    else:
        notes = ' '.join(notes_li)
    num = data.get('numQuestions', 10)  

    qs = await generate_qs(notes, int(num))

    return jsonify(qs)


async def generate_feedback(notes, question, user_answer):
    prompt = f"""
    Based on the following notes, evaluate the my answer to the question. Determine if the answer is correct, if it includes all necessary information, or if any part is incorrect or missing. Provide detailed feedback on the accuracy and completeness of the answer. 

    Notes:
    {notes}

    Question:
    {question}

    My Answer:
    {user_answer}

    Feedback:
    """
    
    response = llm(prompt)
        
    return response

@app.route('/get_feedback', methods=['POST'])
async def get_feedback():
    data = request.json
     
    question = data.get('question', '')  
    collection_id = data.get('collection_id', '')  
    section_id = data.get('section_id', '')  
    answer = data.get('userAnswer', '')
    notes_li = get_notess(collection_id, section_id)

    if not notes_li:
        notes = '*'
    else:
        notes = ' '.join(notes_li)

    feedback = await generate_feedback(notes,question,answer)

    return jsonify(feedback)




@app.route('/study-plans', methods=['POST', 'GET', 'DELETE'])
def study_plans():
    if request.method == 'POST':
        try:
            data = request.json
            username = data.get('username')
            if not username:
                return jsonify({'success': False, 'error': 'Username is required'}), 400

            plan = {
                'name': data.get('name'),
                'testDate': data.get('testDate'),
                'concepts': data.get('concepts', []),
                'collectionId': data.get('collectionId'),
                'sectionId': data.get('sectionId'),
                'createdAt': datetime.utcnow()
            }

            # Fetch user document reference
            user_ref = db.collection('users').document(username)
            user_doc = user_ref.get()

            if user_doc.exists:
                # Update existing study plans or add a new one
                study_plans = user_doc.to_dict().get('studyPlans', [])
                study_plans.append(plan)
                user_ref.update({'studyPlans': study_plans})
                return jsonify({'success': True, 'message': 'Study plan created successfully'}), 201
            else:
                return jsonify({'success': False, 'error': 'User does not exist'}), 404

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    elif request.method == 'GET':
        try:
            username = request.args.get('username')
            if not username:
                return jsonify({'success': False, 'error': 'Username is required'}), 400

            user_ref = db.collection('users').document(username)
            user_doc = user_ref.get()

            if user_doc.exists:
                study_plans = user_doc.to_dict().get('studyPlans', [])
                return jsonify({'success': True, 'studyPlans': study_plans}), 200
            else:
                return jsonify({'success': True, 'studyPlans': []}), 200

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    elif request.method == 'DELETE':
        try:
            data = request.json
            username = data.get('username')
            index = int(data.get('index'))
            user_ref = db.collection('users').document(username)
            user_doc = user_ref.get()

            if user_doc.exists:
                study_plans = user_doc.to_dict().get('studyPlans', [])
                
                if 0 <= index < len(study_plans):
                    # Remove the plan at the specified index
                    del study_plans[index]
                    
                    # Update the user document with the modified study plans
                    user_ref.update({'studyPlans': study_plans})
                    
                    return jsonify({'success': True, 'message': 'Study plan deleted successfully'}), 200
                else:
                    return jsonify({'success': False, 'error': 'Invalid index'}), 400
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    else:
        return jsonify({'success': False, 'error': 'Method not allowed'}), 405
retention_llm_model = list_llm[0]
retention_llm = HuggingFaceEndpoint(
    repo_id=retention_llm_model,
    temperature=0.5,  
    max_new_tokens=5000,  
    top_k=20,
)

async def retention_feedback(user_text, notes):
    prompt = f"""
    As an expert educator, your task is to evaluate the my understanding and retention based on the provided notes and topics covered within them. Assess my full retention and understanding of all main topics and key details covered in the notes.

    Notes:
    {notes}

    My understanding of the notes off the top of my head:
    {user_text}

    Please provide a comprehensive evaluation following this structure:

    1. High-Level Overview (2-3 sentences):
    Summarize the overall quality of the understanding, highlighting the main strengths and general areas for improvement to wholisticly know all the content.

    2. Correct and Well-Understood Concepts:
    List the topics and details I have demonstrated a good understanding of. Be specific about what I got right.

    3. Errors and Misconceptions:
    Identify any factual errors or misconceptions in my answer. Explain why these are incorrect and provide the correct information.

    4. Incomplete or Missing Information:
    Point out any main topics or important details from the notes that were not addressed in my summary. Explain why these are crucial for a complete understanding.

    5. Suggestions for Improvement:
    Offer specific advice on how I can enhance my understanding and retention of the material. Provide study tips or strategies if applicable.

    6. Overall Retention Score:
    On a scale of 1-10, rate the student's overall retention and understanding of the material, with 10 being perfect retention of all main topics and key details.

    Remember to maintain a constructive and encouraging tone throughout the feedback.
    """
    
    response =  retention_llm(prompt)
        
    return response

@app.route('/retention', methods=['POST'])
async def retention():
    data = request.json
     
    user_text = data.get('user_text', '')  
    collection_id = data.get('collection_id', '')  
    section_id = data.get('section_id', '')  
    notes_selected = data.get('notes_selected', '')

    feedback = await retention_feedback(user_text,notes_selected)

    return jsonify(feedback)



async def generate_study_plan(test_date, notes):
    current_date = datetime.now().date()
    test_date = datetime.strptime(test_date, "%Y-%m-%d").date()
    days_until_test = (test_date - current_date).days

    prompt = f"""
    As an expert educator and study planner, your task is to create an effective study plan based on the provided notes and the number of days until the test. The test is in {days_until_test} days from now.

    Notes:
    {notes}

    Please create a comprehensive study plan following this structure and format:

    Create a day-by-day study schedule, allocating topics to specific days. Each day's plan should be on a new line, in the format "Day X: [Topic(s) to study]". Ensure that all topics are covered and that there's time for review before the test. Make sure to enclose the topic name in the square brackets following each day.

    Ensure the plan is realistic, well-paced, and covers all the material in the notes. Adjust the intensity based on the number of days available for studying.
    """
    
    response = llm(prompt)
    
    return response

@app.route('/ai-study-suggestion', methods=['POST'])
async def ai_study_suggestion():
    data = request.json
    
    test_date = data.get('testDate', '')
    collection_id = data.get('collection_id', '')
    section_id = data.get('section_id', '')
    notes_li = get_notess(collection_id, section_id)
    if not notes_li:
        notes = '*'
    else:
        notes = ' '.join(notes_li)

    if not test_date:
        return jsonify({'error': 'Test date is required'}), 400

    try:
        current_date = datetime.now().date()
        study_plan = []
        response = await generate_study_plan(test_date, notes)
        response_lines = response.strip().split('\n')
        for line in response_lines:
            line = line.strip()
            # if line.startswith("Day ") and "[" in line and "]" in line:
            try:
                    if line.startswith("Day ") and "[" in line and "]" in line:

                        day_part, topic_part = line.split(":", 1)
                        day_num = int(day_part.split()[1])
                        topic = topic_part.split("[")[1].split("]")[0].strip()
                        date = current_date + timedelta(days=day_num-1)
                        study_plan.append({
                            "date": date.strftime("%Y-%m-%d"),
                            "topic": topic
                        })
                    else:
                        day_part, topic_part = line.split(":", 1)
                        day_num = int(day_part.split()[2])
                        topic = topic_part.strip()
                        date = current_date + timedelta(days=day_num-1)
                        study_plan.append({
                            "date": date.strftime("%Y-%m-%d"),
                            "topic": topic
                        })
            except (ValueError, IndexError) as e:
                    print(f"Error parsing line: {line}. Error: {str(e)}")
                    continue
        if(len(study_plan)==0):
            for line in response_lines:
                line = line.strip()
                # if line.startswith("Day ") and "[" in line and "]" in line:
                try:
                        day_part, topic_part = line.split(":", 1)
                        day_num = int(day_part.split()[2])
                        topic = topic_part.split("[")[1].split("]")[0].strip()
                        date = current_date + timedelta(days=day_num-1)
                        study_plan.append({
                            "date": date.strftime("%Y-%m-%d"),
                            "topic": topic
                        })
                except (ValueError, IndexError) as e:
                        print(f"Error parsing line: {line}. Error: {str(e)}")
                        continue
            # elif line[0].isdigit() and line.split()[1] == "Day" and "[" in line and "]" in line:
            #     try:
            #         day_part, topic_part = line.split(":", 1)
            #         day_num = int(day_part.split()[1])
            #         topic = topic_part.split("[")[1].split("]")[0].strip()
            #         date = current_date + timedelta(days=day_num - 1)
            #         study_plan.append({
            #             "date": date.strftime("%Y-%m-%d"),
            #             "topic": topic
            #         })
            #     except (ValueError, IndexError) as e:
            #         print(f"Error parsing line: {line}. Error: {str(e)}")
            #         continue

        return jsonify(study_plan)
    except Exception as e:
        return jsonify({'error': str(e)}), 500




async def generate_question(sentence, answer):
    text = f"context: {sentence} answer: {answer}"
    max_len = 256
    encoding = t5_tokenizer.encode_plus(text, max_length=max_len, pad_to_max_length=False, truncation=True, return_tensors="pt")

    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outputs = t5_model.generate(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 early_stopping=True,
                                 num_beams=15,
                                 num_return_sequences=10,
                                 no_repeat_ngram_size=20,
                                 max_length=300)

    decoded_outputs = [t5_tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]

    question = decoded_outputs[0].replace("question:", "").strip()
    return question
def check_profanity(text):
    url = "https://www.purgomalum.com/service/containsprofanity"
    params = {"text": text}
    response = requests.get(url, params=params)
    return response.text == 'true'

async def calculate_embedding(doc):
    tokens = BTokenizer.from_pretrained('bert-base-uncased').tokenize(doc)
    token_ids = BTokenizer.from_pretrained('bert-base-uncased').convert_tokens_to_ids(tokens)
    segment_ids = [1] * len(tokens)

    torch_tokens = torch.tensor([token_ids])
    torch_segments = torch.tensor([segment_ids])

    return BModel.from_pretrained("bert-base-uncased")(torch_tokens, torch_segments)[-1].detach().numpy()

async def get_parts_of_speech(context):
  doc = nlp(context)
  pos_tags = [token.pos_ for token in doc]
  return pos_tags, context.split()

async def get_sentences(context):
  doc = nlp(context)
  return list(doc.sents)

async def get_vectorizer(doc):
  stop_words = "english"
  n_gram_range = (1,1)
  vectorizer = CountVectorizer(ngram_range = n_gram_range, stop_words = stop_words).fit([doc])
  return vectorizer.get_feature_names_out()

async def get_keywords(context, module_type='t'):
    keywords = []
    top_n = 5
    sentences = list(nlp(context).sents)

    for sentence in sentences:
        key_words = CountVectorizer(ngram_range=(1, 1), stop_words="english").fit([str(sentence)]).get_feature_names_out()
        
        if module_type == 't':
            sentence_embedding = await calculate_embedding(str(sentence))
            keyword_embedding = await calculate_embedding(' '.join(key_words))
        else:
            sentence_embedding = sentence_model.encode([str(sentence)])
            keyword_embedding = sentence_model.encode(key_words)
        
        distances = cosine_similarity(sentence_embedding, keyword_embedding)
        keywords += [(key_words[index], str(sentence)) for index in distances.argsort()[0][-top_n:]]

    return keywords

async def ask_sydney(question):
    async with SydneyClient() as sydney:
        response = await sydney.ask(question, citations=True)
        return response
    
async def ask_sydney_with_retry(question, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            return await ask_sydney(question)
        except Exception as e:
            print(e)
            print(f"Request throttled. Retrying in {2**retries} seconds...")
            await asyncio.sleep(2**retries)
            retries += 1
    raise Exception("Exceeded maximum number of retries")


@app.route('/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory(VIDEO_DIRECTORY, filename)


# @app.route('/get_my_collections', methods=['GET'])
@app.route('/get_my_collections')
def get_my_collections():
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


@app.route('/get_video_paths', methods=['GET'])
def get_video_paths():
    collection_id = request.args.get('collection_id')
    section_id = request.args.get('section_id')
    
    if not collection_id or not section_id:
        return jsonify({'error': 'Collection ID or Section ID not provided'}), 400

    video_paths = db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('videos').stream()
    video_paths_list = [video.to_dict() for video in video_paths]
    
    return jsonify({'videoPaths': video_paths_list})
nlp = spacy.load("en_core_web_sm")

def generate_audio(text):
    tts = gTTS(text=text, lang='en')
    return tts

def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    ddgs = DDGS()
    return [result['image'] for result in ddgs.images(keywords=term, max_results=max_images)]

@app.route('/time_spent', methods=['POST'])
def time_spent():
    data = request.get_json()
    collection_id = data.get('collection_id')
    section_id = data.get('section_id')
    total_time_spent = data.get('total_time_spent')
    if not collection_id or not section_id or not total_time_spent:
        return jsonify({'error': 'Missing required parameters'}), 400
    try:
        section_ref = db.collection('collections').document(collection_id).collection('sections').document(section_id)

        section_data = section_ref.get().to_dict()
        current_total_time = section_data.get('total_time', 0)

        new_total_time = current_total_time + total_time_spent

        section_ref.update({'total_time': new_total_time})
        return jsonify({'message': 'Access time updated successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# def download_image(url, folder):
#     filename = os.path.join(folder, os.path.basename(url))
#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         with open(filename, 'wb') as f:
#             f.write(response.content)
#         # Verify the image
#         with Image.open(filename) as img:
#             img.verify()
#         return filename
#     except Exception as e:
#         print(f"Error downloading or verifying image {url}: {e}")
#         return None


def download_image(url, folder):
    try:
        # Ensure target folder exists
        os.makedirs(folder, exist_ok=True)

        # Construct filename from URL
        filename = os.path.join(folder, os.path.basename(url))

        # Download image using requests
        response = requests.get(url)
        response.raise_for_status()

        # Save image to file
        with open(filename, 'wb') as f:
            f.write(response.content)

        # Verify the image (optional)
        with Image.open(filename) as img:
            img.verify()

        return filename

    except requests.exceptions.HTTPError as e:
        if response.status_code == 403:
            print(f"Access forbidden for image {url}: {e}")
        else:
            print(f"HTTP error occurred while downloading image {url}: {e}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading image {url}: {e}")

    except Exception as e:
        print(f"Other error downloading image {url}: {e}")

    return None

def resize_images(image_paths, output_folder, target_size=(1280, 720)):
    resized_paths = []
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"Image file {image_path} does not exist. Skipping.")
            continue
        try:
            image = Image.open(image_path)
            image = image.resize(target_size)
            resized_path = os.path.join(output_folder, os.path.basename(image_path))
            image.save(resized_path)
            resized_paths.append(resized_path)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
    return resized_paths
nlp = spacy.load("en_core_web_sm")
wikipedia.set_lang("en")
question_generator = pipeline("text2text-generation", model="valhalla/t5-small-qg-hl")

async def generate_question(phrase):
    question = question_generator(phrase, max_new_tokens=50, num_return_sequences=1)[0]['generated_text'].strip()
    return question

async def extract_main_phrases(text, max_phrases=5):
    doc = nlp(text)
    main_phrases = [chunk.text for chunk in doc.noun_chunks]
    main_phrases.sort(key=len, reverse=True)
    return main_phrases[:max_phrases]

async def summarize_concept(main_concept, context):
    try:
        search_results = wikipedia.search(main_concept)

        page = wikipedia.page(search_results[0])

        if context.lower() in page.content.lower():
            summary = page.content.split('.')[0] + '.'
            return summary
        else:
            return await select_relevant_option(main_concept, context, search_results)

    except DisambiguationError as e:
        return await select_relevant_option(main_concept, context, e.options)

    except PageError:
        return await generate_question(main_concept)

async def select_relevant_option(main_concept, context, options):
    context_words = set(context.lower().split())
    best_match = None
    best_score = 0

    for option in options[:5]:  
        try:
            page = wikipedia.page(option)
            page_content = page.content.lower()
            match_count = sum(1 for word in context_words if word in page_content)

            if match_count > best_score:
                best_match = page
                best_score = match_count

        except PageError:
            continue

    if best_match:
        summary = best_match.content.split('.')[0] + '.'
        return summary
    else:
        return f"No relevant Wikipedia page found for '{main_concept}' in the given context."


@app.route('/suggest_annotations', methods=['GET'])
async def suggest_annotations():
    note_id = request.args.get('note_id')
    section_id = request.args.get('section_id')
    collection_id = request.args.get('collection_id')

    note_ref = db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('notes_in_section').document(note_id)
    note_content = note_ref.get().to_dict()['notes']
    
    main_phrases = await extract_main_phrases(note_content)
    
    suggested_annotations = await asyncio.gather(*[summarize_concept(phrase, note_content) for phrase in main_phrases])
    
    annotations = [
        {'text': phrase, 'annotation': annotation}
        for phrase, annotation in zip(main_phrases, suggested_annotations)
    ]
    
    # Return the suggested annotations as JSON
    return jsonify({'suggested_annotations': annotations})


# def extract_concepts_and_summaries(text):
#     concepts = {}
#     current_concept = None
#     current_search_phrases = []
#     current_summary = []

#     lines = text.strip().split('\n')
#     for line in lines:
#         line = line.strip()
#         print(f"Processing line: {line}")
#         if re.match(r'^\d+\.\s\*\*.*\*\*:$', line):  # Match lines like "1. **Creating Functions**:"
#             if current_concept:
#                 concepts[current_concept] = {
#                     "search_phrases": current_search_phrases,
#                     "summary": ' '.join(current_summary)
#                 }
#             current_concept = line.split('**')[1].strip()
#             current_search_phrases = []
#             current_summary = []
#             print(f"New concept detected: {current_concept}")
#         elif line.startswith("- Search phrases:"):
#             phrases = re.findall(r'"(.*?)"', line)
#             current_search_phrases.extend(phrases)
#             print(f"Search phrases found: {current_search_phrases}")
#         elif line.startswith("- Summary:"):
#             current_summary.append(line.split(":", 1)[1].strip())
#             print(f"Summary found: {current_summary}")
#         else:
#             if current_concept:
#                 current_summary.append(line.strip())
    
#     if current_concept:
#         concepts[current_concept] = {
#             "search_phrases": current_search_phrases,
#             "summary": ' '.join(current_summary)
#         }
    
#     return concepts
default_video_llm_index = 14
video_llm_model = list_llm[default_video_llm_index]
video_llm = HuggingFaceEndpoint(
    repo_id=video_llm_model,
    temperature=0.5,  
    max_new_tokens=5000,  
    top_k=20,
    request_timeout=300
)
async def generate_video_script(text):
    prompt = f"""
    Create a detailed script for a video that explains the following concepts in an engaging and informative manner:
    
    {text}
    
    The script should include an introduction, explanation of key concepts, and examples. Prefix all of the spoken text in the script with Narrator: 
    Be sure to include a title at the beginning for the video, prefixed with Title:
    Whenever relevant, include a detailed and precise search term for a relevant graphic inside of square brackets preceding the corresponding chunk of text. Try to make the descriptions of the graphics fairly detailed so that the corresponding images that will be found are relevant to the content. Ensure there are multiple relevant graphics so that the video is more understandable and fluid to viewers.

    """
    response = video_llm(prompt)
    print(response)
    return response
async def get_title(video_script):
    lines = video_script.splitlines()

    for line in lines:
        if 'Title:' in line:
            narrator_text = line.split('Title:')[1].strip()
            return narrator_text
    return 'Explanation Video'    
async def extract_narrator_script(video_script):
    narrator_lines = []
    lines = video_script.splitlines()

    for line in lines:
        if 'Title:' in line:
            narrator_text = line.split('Title:')[1].strip()
            narrator_lines.append(narrator_text)
        if 'Narrator:' in line:
            narrator_text = line.split('Narrator:')[1].strip()
            narrator_lines.append(narrator_text)
        if 'Narrator (voiceover):' in line:
            narrator_text = line.split('Narrator (voiceover):')[1].strip()
            narrator_lines.append(narrator_text)
        if 'Narrator (Voiceover):' in line:
            narrator_text = line.split('Narrator (Voiceover):')[1].strip()
            narrator_lines.append(narrator_text)
    return ' '.join(narrator_lines)

async def generate_audio(text, output_file):
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(output_file)
    print(f"Audio content written to {output_file}")

async def extract_graphics_cues(video_script):
    # Extracts the content inside square brackets
    graphic_cues = []
    for line in video_script.splitlines():
        if '[' in line and ']' in line:
            cue = line[line.index('[') + 1:line.index(']')].strip()
            graphic_cues.append(cue)
    return graphic_cues

async def extract_timed_sections(video_script):
    # Extract text sections and their timings
    sections = []
    for line in video_script.splitlines():
        match = re.match(r'\[Start: (\d+)\] \[(End: (\d+))\]', line)
        if match:
            start_time, end_time = int(match.group(1)), int(match.group(3))
            sections.append((start_time, end_time))
    return sections
async def download_image(query, output_file, retries=10):
    with DDGS() as ddgs:
        for i in range(retries):
            results = ddgs.images(query)
            for result in results:
                image_url = result['image']
                try:
                    img_data = requests.get(image_url, timeout=10).content
                    with open(output_file, 'wb') as f:
                        f.write(img_data)
                except requests.exceptions.RequestException as e:
                    print(f"An error occurred: {e}")
                try:
                    # Attempt to open the image to ensure it's valid
                    img = Image.open(output_file)
                    img.verify()  # Verify that it is, in fact, an image
                    print(f"Image downloaded and verified: {output_file}")
                    return
                except (IOError, SyntaxError) as e:
                    print(f"Image {output_file} is corrupted or cannot be opened, trying another image. Error: {e}")
                    continue
            
            print(f"Retrying download for query '{query}' ({i+1}/{retries})")
        
        # If all retries fail
        raise ValueError(f"Could not find a valid image for query '{query}' after {retries} attempts.")
async def calculate_wpm(narrator_script, audio_duration):
    total_words = len(narrator_script.split())
    wpm = total_words / (audio_duration / 60)  # Convert duration to minutes
    return wpm
async def create_title_image(title_text, output_file):
    img = Image.new('RGB', (1280, 720), color='black')
    
    # Initialize ImageDraw
    d = ImageDraw.Draw(img)
    
    # Load a font
    try:
        # You can specify a TTF font file if you have one
        font = ImageFont.truetype("arial.ttf", 70)
    except IOError:
        # Use a default font if the specified font is not available
        font = ImageFont.load_default()
    
    # Wrap text
    max_width = img.width - 40  # Max width for text, with some padding
    lines = []
    words = title_text.split()
    current_line = ""

    for word in words:
        # Check the width of the current line with the next word
        test_line = current_line + word + " "
        text_width, _ = d.textsize(test_line, font=font)
        
        if text_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word + " "
    
    # Append the last line
    if current_line:
        lines.append(current_line)
    
    # Calculate total text height
    total_text_height = sum(d.textsize(line, font=font)[1] for line in lines)
    
    # Calculate starting y position
    text_y = (img.height - total_text_height) / 2
    
    # Draw each line
    for line in lines:
        text_width, text_height = d.textsize(line, font=font)
        text_x = (img.width - text_width) / 2
        d.text((text_x, text_y), line, font=font, fill=(255, 255, 255))
        text_y += text_height
    
    # Save the image
    img.save(output_file)
    print(f"Title image created: {output_file}")
async def calculate_reading_time(text, wpm):
    words = len(text.split())
    return words / wpm * 60
async def create_video(image_files, audio_file, script, output_file):
    audio = AudioFileClip(audio_file)
    audio_duration = audio.duration
    print(audio_duration)
    narrator_script = await extract_narrator_script(script)
    wpm = await calculate_wpm(narrator_script, audio_duration)
    
    clips = []
    current_time = 0

    # Extract graphics cues and corresponding text
    graphics_cues = await extract_graphics_cues(script)
    script_lines = script.splitlines()
    
    # Title image and its duration
    title_text = await get_title(script)
    title_duration =await calculate_reading_time(title_text, wpm)

    # Add the title image as the first clip
    title_clip = ImageClip(image_files[0]).set_duration(title_duration).resize(height=720)
    title_clip = title_clip.set_audio(audio.subclip(0, title_duration))
    clips.append(title_clip)
    
    current_time += title_duration
    
    # Now process each graphic cue and corresponding text
    for index in range(len(graphics_cues)):
        # Find the script text after the current graphic cue until the next one
        text_after_cue = ""
        start_collecting = False
        for line in script_lines:
            if graphics_cues[index] in line:
                start_collecting = True
                continue
            if start_collecting:
                if index < len(graphics_cues) - 1 and graphics_cues[index + 1] in line:
                    break
                text_after_cue += line + " "
        
        # Calculate the reading time for the text after the current graphic cue using WPM
        reading_time = await calculate_reading_time(text_after_cue, wpm)
        
        # Set the duration for the current image
        clip_start = current_time
        clip_end = clip_start + reading_time
        if(index==0):
            clip_end-=title_duration
        if(clip_end>audio_duration):
            clip_end=audio_duration        
        # Debugging output
        print(f"Processing image {image_files[index+1]}: {clip_start} to {clip_end}")

        clip = ImageClip(image_files[index+1]).set_duration(clip_end - clip_start).resize(height=720)
        clip = clip.set_audio(audio.subclip(clip_start, clip_end))
        clips.append(clip)
        
        # Update the current time for the next clip
        current_time = clip_end
    
    if clips:
        last_clip = clips[-1]
        remaining_duration = audio.duration - sum(clip.duration for clip in clips)
        if remaining_duration > 0:
            extended_clip = last_clip.set_duration(last_clip.duration + remaining_duration)
            clips[-1] = extended_clip
    video = concatenate_videoclips(clips, method="compose")
    output_video_path = os.path.join('vids', output_file)
    video.write_videofile(output_video_path, fps=5, codec='libx264', preset='fast')
    print(f"Video created: {output_video_path}")

@app.route('/generate_video_from_notes', methods=['POST'])
async def generate_video_from_notes():
    try:
        data = request.get_json()
        notes = data.get('notes')
        collection_id = data.get('collection_id')
        section_id = data.get('section_id')
        video_script = await generate_video_script(notes)
        print("Generated Script:\n", video_script)
        
        narrator_script = await extract_narrator_script(video_script)
        print("Narrator Script:\n", narrator_script)

        audio_file = "output_audio.mp3"
        await generate_audio(narrator_script, audio_file)

        graphic_cues = await extract_graphics_cues(video_script)
        print("Graphic Cues:\n", graphic_cues)

        title_text = await get_title(video_script)
        title_image_file = "title_image.jpg"
        await create_title_image(title_text, title_image_file)


        image_files = [title_image_file] 
        for i, cue in enumerate(graphic_cues):
            image_file = f"image_{i}.jpg"
            await download_image(cue, image_file)
            image_files.append(image_file)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        output_video_path = f"{collection_id}_{section_id}_final_video_{timestamp}.mp4"
        
        await create_video(image_files, audio_file, video_script, output_video_path)
        db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('videos').add({'video_path': output_video_path, 'title':title_text})

        return jsonify({'video_path': output_video_path, 'title':title_text})

    except Exception as e:
        print(f"Error generating video: {e}")
        return jsonify({'error': str(e)}), 500


    #     concepts = extract_concepts_and_summaries(notes)

    #     # Debugging: Print the extracted concepts and summaries
    #     print("Extracted Concepts and Summaries:")
    #     for concept, data in concepts.items():
    #         print(f"Concept: {concept}")
    #         print(f"Search Phrases: {data['search_phrases']}")
    #         print(f"Summary: {data['summary']}")
    #         print("-----")

    #     # Exit if no concepts were extracted
    #     if not concepts:
    #         print("No concepts were extracted. Exiting.")
    #         return

    #     image_paths = []

    #     for concept, data in concepts.items():
    #         concept_images = []
    #         for phrase in data["search_phrases"]:
    #             images = search_images(phrase, max_images=1)
    #             if images:
    #                 concept_images.append(images[0])
    #         concepts[concept]["images"] = concept_images

    #         print(f"Downloading images for concept: {concept}")
    #         concept_folder = os.path.join(os.getcwd(), "images", concept)
    #         os.makedirs(concept_folder, exist_ok=True)
    #         for i, image_url in enumerate(data['images']):
    #             image_path = download_image(image_url, concept_folder)
    #             if image_path:
    #                 image_paths.append(image_path)
    #                 print(f"Downloaded image {i + 1}: {image_path}")

    #     image_paths = resize_images(image_paths, "resized_images")

    #     concept_audio_paths = []
    #     concept_video_clips = []

    #     for concept, data in concepts.items():
    #         summary = data["summary"]
    #         images = data["images"]
    #         image_durations = []
    #         image_clips = []

    #         tts = generate_audio(summary)
    #         audio_path = f"audio_{concept}.mp3"
    #         tts.save(audio_path)
    #         concept_audio_paths.append(audio_path)

    #         audio_clip = AudioFileClip(audio_path)
    #         audio_duration = audio_clip.duration

    #         if len(images) > 0:
    #             duration_per_image = audio_duration / len(images)
    #         else:
    #             # Handle the case where there are no images (perhaps raise an error or handle gracefully)
    #             duration_per_image = 0  # or any other default value or error handling mechanism
    #         for i, image in enumerate(images):
    #             image_path = os.path.join("resized_images", os.path.basename(image))
    #             if os.path.exists(image_path):
    #                 image_clips.append((image_path, duration_per_image))

    #         for image_path, duration in image_clips:
    #             img_clip = ImageSequenceClip([image_path], durations=[duration])
    #             concept_video_clips.append(img_clip)

    #     final_video_clip = concatenate_videoclips(concept_video_clips)
    #     final_audio_clip = concatenate_audioclips([AudioFileClip(p) for p in concept_audio_paths])
    #     final_video_clip = final_video_clip.set_audio(final_audio_clip)

    #     timestamp = datetime.now().strftime("%Y%m%d%H%M%S")


    #     output_video_path = f"{collection_id}_{section_id}_final_video_{timestamp}.mp4"
    #     final_video_clip.write_videofile(f"./vids/{output_video_path}", fps=24)
    #     db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('videos').add({'video_path': output_video_path})

    #     return jsonify({'video_path': output_video_path})

    # except Exception as e:
    #     print(f"Error generating video: {e}")
    #     return jsonify({'error': str(e)}), 500



async def suggest_learning_path(explorations, max_suggestions=10, max_retries=3):
    wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent='studybuddy-app/1.0'
    )

    concept_counter = Counter()
    learning_keywords = ['concept', 'theory', 'principle', 'method', 'technique', 'algorithm', 'framework', 'model', 'paradigm', 'approach']

    for exploration in explorations:
        for attempt in range(max_retries):
            try:
                page = wiki_wiki.page(exploration)
                if page.exists():
                    # Process sections
                    for section in page.sections:
                        if any(keyword in section.title.lower() for keyword in learning_keywords):
                            concept_counter[section.title] += 2
                    
                    # Process links
                    for link in page.links.values():
                        if any(keyword in link.title.lower() for keyword in learning_keywords):
                            concept_counter[link.title] += 1
                break  # If successful, break the retry loop
            except (requests.exceptions.RequestException, wikipediaapi.WikipediaException) as e:
                if attempt < max_retries - 1:
                    print(f"Error occurred: {e}. Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print(f"Failed to process '{exploration}' after {max_retries} attempts.")

    # Ensure we have at least max_suggestions
    while len(concept_counter) < max_suggestions:
        concept_counter[f"Explore more about {explorations[len(concept_counter) % len(explorations)]}"] += 1

    # Get top suggestions
    suggested_concepts = [concept for concept, _ in concept_counter.most_common(max_suggestions)]

    return suggested_concepts


def get_antonym(word):
    antonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                antonyms.append(lemma.antonyms()[0].name())
    return antonyms[0] if antonyms else None
default_llm_index_test = 0

list_llm_test = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "google/gemma-7b-it",
    "google/gemma-2b-it",
    "HuggingFaceH4/zephyr-7b-beta",
    "HuggingFaceH4/zephyr-7b-gemma-v0.1",
    "meta-llama/Llama-2-7b-chat-hf",
    "microsoft/phi-2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mosaicml/mpt-7b-instruct",
    "tiiuae/falcon-7b-instruct",
    "google/flan-t5-xxl",
    "mistralai/Mistral-7B-Instruct-v0.3"
]


llm_model_test = list_llm_test[default_llm_index_test]
llm_test = HuggingFaceEndpoint(
    repo_id=llm_model_test,
    temperature=0.5,  # Lower temperature for more deterministic output
    max_new_tokens=500,  # Increase to allow for more output
    top_k=20
)
def generate_tf_questions(text, num_questions=10):
    tf_prompt = (
        f"Generate {num_questions} true/false questions from the following notes: {text}.\n"
        "Use the following format for each question:\n"
        "True/False Question:\n"
        "Question: [Your question here]\n"
        "Answer: [True/False]\n\n"
        "Example:\n"
        "True/False Question:\n"
        "Question: The sky is blue.\n"
        "Answer: True\n\n"
    )
    tf_response = llm_test(tf_prompt)

    tf_questions = []
    tf_pairs = tf_response.split("True/False Question:\nQuestion: ")[1:]
    for pair in tf_pairs:
        question, answer = pair.split("\nAnswer: ")
        # Strip extra text and remove possible newlines
        question = question.strip().split("\n")[0]
        answer = answer.strip().split("\n")[0]
        tf_questions.append([question, answer])
    return tf_questions

def negate_verb(sentence, tagged):
    for i, (word, tag) in enumerate(tagged):
        if tag.startswith('VB'):
            if word.lower() in ['is', 'are', 'was', 'were']:
                return sentence[:sentence.index(word)] + word + " not" + sentence[sentence.index(word)+len(word):]
            elif word.lower() == 'have':
                return sentence[:sentence.index(word)] + "do not have" + sentence[sentence.index(word)+len(word):]
            else:
                return sentence[:sentence.index(word)] + "does not " + word + sentence[sentence.index(word)+len(word):]
    return None

def replace_with_antonym(sentence, tagged):
    for word, tag in tagged:
        if tag.startswith('JJ') or tag.startswith('RB'):
            antonym = get_antonym(word)
            if antonym:
                return sentence.replace(word, antonym)
    return None

def change_quantity(sentence, tagged):
    quantity_words = {'all': 'some', 'every': 'some', 'always': 'sometimes', 'never': 'sometimes'}
    for word, _ in tagged:
        if word.lower() in quantity_words:
            return sentence.replace(word, quantity_words[word.lower()])
    return None


def generate_tf(notes, num_questions):
    questions = generate_tf_questions(notes, num_questions)

    tf_questions = []

    for item in questions:
        tf_questions.append({'question': item[0], 'answer': item[1]})

    return tf_questions
@app.route('/generate_tf_questions', methods=['POST'])
def generate_tf_questions_endpoint():
    try:
        request_data = request.get_json()
        section_id = request_data.get('section_id', '')

        notes=note_getter(section_id)

        num_questions = request_data.get('num_questions', 10)
        
        tf_questions = generate_tf(notes, num_questions)

        return jsonify(tf_questions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')
bert_tokenizer = BTokenizer.from_pretrained('bert-base-uncased')
bert_model = BModel.from_pretrained("bert-base-uncased")
sentence_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
nlp = spacy.load("en_core_web_sm")

def generate_question_answer(sentence, answer):
    t5_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
    t5_tokenizer = AutoTokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')

    text = "context: {} answer: {}".format(sentence, answer)
    max_len = 256
    encoding = t5_tokenizer.encode_plus(text, max_length=max_len, pad_to_max_length=False, truncation=True, return_tensors="pt")

    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outputs = t5_model.generate(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 early_stopping=True,
                                 num_beams=5,
                                 num_return_sequences=1,
                                 no_repeat_ngram_size=2,
                                 max_length=300)

    decoded_outputs = [t5_tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]

    question = decoded_outputs[0].replace("question:", "")
    question = question.strip()

    answer_output = qa_pipeline(question=question, context=sentence)

    return question, answer_output['answer']



def generate_question(sentence, answer):
  t5_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
  t5_tokenizer = AutoTokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')

  text = "context: {} answer: {}".format(sentence,answer)
  max_len = 256
  encoding = t5_tokenizer.encode_plus(text,max_length=max_len, pad_to_max_length=False,truncation=True, return_tensors="pt")

  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outputs = t5_model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=5,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  max_length=300)

  decoded_outputs = [t5_tokenizer.decode(ids,skip_special_tokens=True) for ids in outputs]

  question = decoded_outputs[0].replace("question:","")
  question = question.strip()
  return question


def calculate_embedding(doc):
  bert_tokenizer = BTokenizer.from_pretrained('bert-base-uncased')
  bert_model = BModel.from_pretrained("bert-base-uncased")
  
  tokens = bert_tokenizer.tokenize(doc)
  token_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
  segment_ids = [1] * len(tokens)

  torch_tokens = torch.tensor([token_ids])
  torch_segments = torch.tensor([segment_ids])

  return bert_model(torch_tokens, torch_segments)[-1].detach().numpy()

def get_parts_of_speech(context):
  doc = nlp(context)
  pos_tags = [token.pos_ for token in doc]
  return pos_tags, context.split()

def get_sentences(context):
  doc = nlp(context)
  return list(doc.sents)

def get_vectorizer(doc):
  stop_words = "english"
  n_gram_range = (1,1)
  vectorizer = CountVectorizer(ngram_range = n_gram_range, stop_words = stop_words).fit([doc])
  return vectorizer.get_feature_names_out()

def get_keywords(context, module_type = 't'):
  keywords = []
  top_n = 5
  for sentence in get_sentences(context):
    key_words = get_vectorizer(str(sentence))
    print(f'Vectors : {key_words}')
    if module_type == 't':
      sentence_embedding = calculate_embedding(str(sentence))
      keyword_embedding = calculate_embedding(' '.join(key_words))
    else:
      sentence_embedding = sentence_model.encode([str(sentence)])
      keyword_embedding = sentence_model.encode(key_words)
    
    distances = cosine_similarity(sentence_embedding, keyword_embedding)
    print(distances)
    keywords += [(key_words[index], str(sentence)) for index in distances.argsort()[0][-top_n:]]

  return keywords


@app.route('/generate_frq', methods=['POST'])
def generate_frq():
    try:
        request_data = request.get_json()
        section_id = request_data.get('section_id', '')
        num_questions = int(request_data.get('num_questions',''))
        notes=note_getter(section_id)
        txt=notes
        qa_pairs = []

        fr_prompt = (
            f"Generate {num_questions} free response questions from the following notes: {txt}.\n"
            "Use the following format for each question:\n"
            "Free Response Question:\n"
            "Question: [Your question here]\n"
            "Answer: [Your answer here]\n\n"
            "Example:\n"
            "Free Response Question:\n"
            "Question: What is the capital of France?\n"
            "Answer: Paris\n\n"
        )
        fr_response = llm_test(fr_prompt)
        print(fr_response)
        # return jsonify({'response':fr_response})
        pattern = r"Question \d+:\s*(.*?)\nAnswer:\s*(.*?)\n"

        # Find all matches
        matches = re.findall(pattern, fr_response, re.DOTALL)

        # Create list of question-answer pairs
        fr_questions = [[question.strip(), answer.strip()] for question, answer in matches]
        if(len(fr_questions)==0):
            fr_questions = []
            fr_pairs = fr_response.split("Question: ")[1:]
            for pair in fr_pairs:
                question, answer = pair.split("\nAnswer: ")
                question = question.strip().split("\n")[0]
                answer = answer.strip().split("\n")[0]
                fr_questions.append([question, answer])
        print(fr_questions)

     
        qa_pairs=[]
        for thing in fr_questions:
            qa_pairs.append({'question': thing[0], 'answer': thing[1]})
            

        return jsonify(qa_pairs)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/generate_mcq', methods=['POST'])
def generate_mcq():
    try:
        request_data = request.get_json()
        section_id = request_data.get('section_id', '')
        num_questions = int(request_data.get('num_questions',''))
        notes=note_getter(section_id)
        txt=notes
        qa_pairs = []

        mc_prompt = (
            f"Generate {num_questions} multiple choice questions from the following notes: {txt}.\n"
            "Use the following format for each question:\n"
            "Multiple Choice Question:\n"
            "Question: [Your question here]\n"
            "Options: [Option A], [Option B], [Option C], [Option D]\n"
            "Answer: [Correct option]\n\n"
            "Example:\n"
            "Multiple Choice Question:\n"
            "Question: What is the largest planet in our solar system?\n"
            "Options: [Mercury, Venus, Earth, Jupiter]\n"
            "Answer: Jupiter\n\n"
        )
        
        mc_response = llm_test(mc_prompt)
        print("Multiple Choice Questions:", mc_response)
        
        mc_questions = []
        mc_pairs = mc_response.split("Question")[1:]  # Split by "Question " instead of "Multiple Choice Question:"
        for pair in mc_pairs:
            if "Options: " in pair and "Answer: " in pair:
                question_part, rest = pair.split("Options: ")
                options, answer = rest.split("\nAnswer: ")
                question = question_part.split(":\n", 1)[-1].strip()  # Get the actual question text after "Question X:\n"
                mc_questions.append([question, options.strip(), answer.strip()])


        return jsonify({'response':mc_questions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500





def generate_combined_questions(num_tf, num_frq, notes):
    tf_questions = generate_tf(notes, num_questions=num_tf)
    frq_questions = generate_frq(notes)[:num_frq]  # Limit FRQ questions to num_frq

    combined_questions = tf_questions + frq_questions

    return combined_questions

@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    if request.is_json:
        request_data = request.get_json()
        print(f'Received request data: {request_data}')
        num_tf = request_data.get('num_tf', 10)  # Default to 10 if not provided
        num_frq = request_data.get('num_frq', 10)  # Default to 10 if not provided
        notes = request_data.get('notes', '')
        section_id = request_data.get('section_id', '')

        notes = note_getter(section_id)

        combined_questions = generate_combined_questions(num_tf, num_frq, notes)

        return jsonify({'questions': combined_questions})
    else: 
        return jsonify({'error': 'Request must be JSON'}), 415
def note_getter(section_id):
    notes = []
    try:
        # Query collections and fetch notes for the given section_id
        collections = db.collection('collections').stream()
        for collection in collections:
            notes_docs = db.collection('collections').document(collection.id).collection('sections').document(section_id).collection('notes_in_section').stream()
            for doc in notes_docs:
                note_data = doc.to_dict()
                notes.append(note_data.get('notes', ''))  # Ensure 'notes' field is correctly fetched
    except Exception as e:
        print(f"Error fetching notes: {e}")
    concatenated_notes = ' '.join(notes)
    return concatenated_notes
@app.route('/suggest_sections', methods=['GET'])
async def suggest_sections():
    collection_id = request.args.get('collection_id')
    if not collection_id:
        return jsonify({'error': 'Collection ID not provided'})

    try:
        # Fetch existing sections
        sections_ref = db.collection('collections').document(collection_id).collection('sections')
        sections_docs = sections_ref.stream()
        existing_sections = [doc.to_dict().get('section_name', '') for doc in sections_docs]

        # Prepare the prompt for the ML model
        prompt = f"""
        Based on the following existing sections in a study collection, suggest 5 new sections that would complement the learning path:

        Existing sections: {', '.join(existing_sections)}

        Generate 5 suggested new sections. Each suggestion should be enclosed in square brackets and separated by a newline. For example:

        [Suggested Section]
        [Suggested Section]
        [Suggested Section]
        [Suggested Section]
        [Suggested Section]

        Your suggestions:
        """

        response = llm_test(prompt)

        suggested_sections = re.findall(r'\[(.*?)\]', response)

        suggested_sections = suggested_sections[:5]
        while len(suggested_sections) < 5:
            suggested_sections.append(f"Additional topic related to {existing_sections[0]}")

        return jsonify({'suggested_sections': suggested_sections})
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/get_my_sections', methods=['GET'])
def get_my_sections():
    username = request.args.get('username')
    if not username:
        return jsonify({'error': 'Username not provided'})

    collections = []
    collection_docs = db.collection('collections').where('username', '==', username).collection('sections').stream()
    for doc in collection_docs:
        # Access the 'data' field and then retrieve the 'title' from it
        title = doc.to_dict().get('data', {}).get('section_name', '')
        visibility = doc.to_dict().get('data', {}).get('visibility', '')
        access = doc.to_dict().get('data', {}).get('last_accessed', '')
        collections.append({'id': doc.id, 'title': title, 'visibility': visibility, 'access': access})

        
    return jsonify({'collections': collections})
@app.route('/get_my_sections_recent', methods=['GET'])
def get_my_sections_recent():
    username = request.args.get('username')
    if not username:
        return jsonify({'error': 'Username not provided'}), 400

    collection_docs = db.collection('collections').where('username', '==', username).stream()

    collections = []

    for doc in collection_docs:
        collection_id = doc.id
        collection_name = doc.to_dict().get('data', {}).get('title', None)
        section_docs = db.collection('collections').document(collection_id).collection('sections').order_by('last_accessed', direction=firestore.Query.DESCENDING).limit(5).stream()
        
        for section_doc in section_docs:
            doc_data = section_doc.to_dict()

            title = doc_data.get('section_name', None)
            visibility = doc_data.get('visibility', None)
            access = doc_data.get('last_accessed', None)

            collections.append({
                'collection_id': collection_id,
                'id': section_doc.id,
                'title': title,
                'visibility': visibility,
                'access': access,
                'collName': collection_name

            })

    return jsonify({'collections': collections})

@app.route('/get_all_sections', methods=['GET'])
def get_all_sections():
    collections = []
    collection_docs = db.collection('collections').collection('sections').stream()
    for doc in collection_docs:
        # Access the 'data' field and then retrieve the 'title' from it
        title = doc.to_dict().get('data', {}).get('section_name', '')
        visibility = doc.to_dict().get('data', {}).get('visibility', '')
        access = doc.to_dict().get('data', {}).get('last_accessed', '')
        collections.append({'id': doc.id, 'title': title, 'visibility': visibility, 'access': access})
        
    return jsonify({'collections': collections})

@app.route('/protected_resource', methods=['GET'])
def protected_resource():
    # Get the ID token from the request headers
    id_token = request.headers.get('Authorization')
    if not id_token:
        return jsonify({'error': 'Authorization token missing'}), 401

    try:
        # Verify the ID token
        decoded_token = auth.verify_id_token(id_token)
        user_id = decoded_token['uid']
        # User is authenticated, serve the resource
        return jsonify({'message': 'You are authenticated!'})
    except auth.InvalidIdTokenError:
        return jsonify({'error': 'Invalid ID token'}), 401

@app.route('/update_access_time', methods=['POST'])
def update_access_time():
    data = request.get_json()
    collection_id = data.get('collection_id')
    section_id = data.get('section_id')

    if not collection_id or not section_id:
        return jsonify({'error': 'Missing parameters'}), 400

    try:
        section_ref = db.collection('collections').document(collection_id).collection('sections').document(section_id)
        section_ref.update({'last_accessed': firestore.SERVER_TIMESTAMP})
        return jsonify({'message': 'Access time updated successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/create_review', methods=['POST'])
def create_review():
    try:
        data = request.get_json()
        collection_id = data.get('collection_id')
        section_ids = data.get('section_ids')
        name = data.get('name')
        if(check_profanity(name)):
            return jsonify({'error':'Profanity detected'}), 400
        username = data.get('username')
        if not collection_id or not section_ids or not name or not username:
            return jsonify({'error': 'Missing required parameters'}), 400

        new_section_ref = db.collection('collections').document(collection_id).collection('sections').document()
        new_section_ref.set({'section_name': name, 'visibility': 'public'})  # Set section name

        for section_id in section_ids:
            # Get the section document
            section_ref = db.collection('collections').document(collection_id).collection('sections').document(section_id)
            section_doc = section_ref.get()
            if not section_doc.exists:
                return jsonify({'error': 'Section not found'}), 404

            # Copy notes from the existing section to the new section
            notes_ref = section_ref.collection('notes_in_section')
            for note in notes_ref.stream():
                new_note_ref = new_section_ref.collection('notes_in_section').document()
                new_note_ref.set(note.to_dict())

        return jsonify({'message': 'Review created successfully'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/create_user', methods=['POST'])
def create_user():
    try:
        data = request.get_json()
        email = data.get('email')
        username = data.get('username')

        if not email or not username:
            return jsonify({'error': 'Missing required parameters'}), 400

        user_ref = db.collection('users').document(email)
        
        user_ref.set({
            'username': username
        })

        return jsonify({'message': 'User created successfully'}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500




@app.route('/delete_collection', methods=['DELETE'])
def delete_collection():
    data = request.get_json()

    collection_id = data.get('collection_id')

    if not collection_id:
        return jsonify({'error': 'Collection ID'}), 400

    try:
        note_ref = db.collection('collections').document(collection_id)
        note_ref.delete()
        return jsonify({'message': 'Note deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/delete_section', methods=['DELETE'])
def delete_section():
    data = request.get_json()

    section_id = data.get('section_id')
    collection_id = data.get('collection_id')

    if not collection_id or not section_id:
        return jsonify({'error': 'something not provided'}), 400

    try:
        note_ref = db.collection('collections').document(collection_id).collection('sections').document(section_id)
        note_ref.delete()
        return jsonify({'message': 'Note deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/delete_note', methods=['DELETE'])
def delete_note():
    collection_id = request.args.get('collection_id')
    section_id = request.args.get('section_id')
    note_id = request.args.get('note_id')

    if not collection_id or not section_id or not note_id:
        return jsonify({'error': 'Collection ID, Section ID, or Note ID not provided'}), 400

    try:
        note_ref = db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('notes_in_section').document(note_id)
        note_ref.delete()
        return jsonify({'message': 'Note deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/edit_note', methods=['POST'])
def edit_note():
    data = request.json
    collection_id = data.get('collection_id')
    section_id = data.get('section_id')
    note_id = data.get('note_id')
    new_notes = data.get('new_notes')

    if not collection_id or not section_id or not note_id or not new_notes:
        return jsonify({'error': 'Missing required fields'}), 400

    try:
        note_ref = db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('notes_in_section').document(note_id)
        note_ref.update({
            'notes': new_notes
        })
        return jsonify({'message': 'Note updated successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/delete_worksheet', methods=['DELETE'])
def delete_worksheet():
    collection_id = request.args.get('collection_id')
    section_id = request.args.get('section_id')
    note_id = request.args.get('worksheet_id')

    if not collection_id or not section_id or not note_id:
        return jsonify({'error': 'Collection ID, Section ID, or Worksheet ID not provided'}), 400

    try:
        note_ref = db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('worksheets').document(note_id)
        note_ref.delete()
        return jsonify({'message': 'Worksheet deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask_specific', methods=['POST'])
async def ask_specific():
    collection_id = request.json.get('collection_id')
    section_id = request.json.get('section_id')
    selected_notes = request.json.get('selected_notes')
        
    if not collection_id or not section_id or not selected_notes:
        return jsonify({'error': 'Collection ID, Section ID, or Note IDs not provided'}), 400

    response = ""
    for note in selected_notes:
        response += note

    loop = asyncio.get_event_loop()
    examp = '''
        1. **Main Concept**:
            - Search phrases: "," "," "," "."
            - Summary: 
        2. **Main Concept**:
                    - Search phrases: "," "," "," "."
                    - Summary: 
        3. **Main Concept**:
                    - Search phrases: "," "," "," "."
                    - Summary: 
        4. **Main Concept**:
                    - Search phrases: "," "," "," "."
                    - Summary: 

      
    '''

    response_task = await ask_sydney_with_retry("I am generating a video based on these notes that I have: " + response + ". Give me search phrases to generate meaningful pictures from the main concepts and a summary, it should have the same format as this example: " + examp)
    # print(response_task)
    # video_task = generate_video_from_notes(str(response_task))


    return jsonify({'response': response_task})
@app.route('/answer_question', methods=['GET','POST'])
def answer_question():
    data = request.json  
    username = data.get('username')
    user_class = data.get('class')
    question = data.get('data')
    if(check_profanity(question)):
            return jsonify({'error':'Profanity detected'}), 400

    if not username or not user_class:
        return jsonify({'error': 'Username or class not provided'})

    user_notes = []
    notes_docs = db.collection('collections').where('username', '==', username).where('class', '==', user_class).stream()
    for doc in notes_docs:
        notes = doc.to_dict().get('data', {}).get('notes', '')
        user_notes.append(notes)

    # Combine the notes into a single string
    combined_notes = ' '.join(user_notes)

    # Ask Sydney a question based on the combined notes
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    response = loop.run_until_complete(ask_sydney_with_retry(question + " based on these notes: "+combined_notes))

    return jsonify({'response': response})



@app.route('/get_transcript', methods=['POST'])
def get_transcript():
    
    collection_id = request.form.get('collection_id')
    section_id = request.form.get('section_id')

    video_url = request.form.get('url')
    print(video_url)
    url_parts = urlparse(video_url)

    query_params = parse_qs(url_parts.query)

    video_id = query_params.get('v')

    if not video_id:
        return jsonify({'error': 'Invalid video URL format'+video_url})
    else:
        video_id = video_id[0]
        print("Video ID:", video_id)
    youtube=build('youtube','v3', developerKey=transcript_key)
    captions = youtube.captions().list(part='snippet', videoId=video_id).execute()
    caption = captions['items'][0]['id']
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

    transcript_txt=""

    for transcript in transcript_list:
        transcript_txt+=transcript['text']
    video_response = youtube.videos().list(
        part="snippet",
        id=video_id
    ).execute()

    video_title = video_response['items'][0]['snippet']['title']
    notes_collection_ref = db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('notes_in_section')
    note_ref = notes_collection_ref.document()  # Automatically generate a new document ID
    note_ref.set({'notes': transcript_txt, 'tldr':("Video: "+video_title)})


    return jsonify({'response': transcript_txt})

@app.route('/add_res_to_flashcards', methods=['POST'])
def add_res_to_flashcards():
    data = request.json
    question = data.get('question')
    answer = data.get('answer')
    collection_id = data.get('collection_id')
    section_id = data.get('section_id')
    if(check_profanity(question)):
            return jsonify({'error':'Profanity detected'}), 400
    if(check_profanity(answer)):
            return jsonify({'error':'Profanity detected'}), 400

    tldr = summarize_text(answer)

    if not collection_id or not section_id or not question or not answer:
        return jsonify({'error': 'Invalid request data. Make sure all fields are provided.'}), 400

    try:
        # Ensure Firestore client is properly initialized
        if not db:
            return jsonify({'error': 'Firestore is not initialized.'}), 500

        # Ensure 'flashcards' collection exists (create it if not)
        flashcards_collection_ref = db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('flashcards')

        # Add flashcard document
        flashcard_doc_ref = flashcards_collection_ref.document()
        flashcard_doc_ref.set({'question': question, 'answer': tldr})

        return jsonify({'response': 'Flashcard uploaded successfully'}), 200
    except Exception as e:
        # Log the full stack trace for debugging purposes
        import traceback
        traceback.print_exc()

        return jsonify({'error': str(e)}), 500
    


# List of models and default index
list_llm = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "google/gemma-7b-it",
    "google/gemma-2b-it",
    "HuggingFaceH4/zephyr-7b-beta",
    "HuggingFaceH4/zephyr-7b-gemma-v0.1",
    "meta-llama/Llama-2-7b-chat-hf",
    "microsoft/phi-2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mosaicml/mpt-7b-instruct",
    "tiiuae/falcon-7b-instruct",
    "google/flan-t5-xxl",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.1",

]
flash_default_llm_index = 0

# Initialize the endpoint
flash_llm_model = list_llm[flash_default_llm_index]
flash_llm = HuggingFaceEndpoint(
    repo_id=flash_llm_model,
    temperature=0.5,  # Lower temperature for more deterministic output
    max_new_tokens=500,  # Increase to allow for more output
    top_k=20
)
def summarize_text(text, max_length=50, min_length=30):
    try:
        prompt = f"""Summarize the given text in one sentence as a tldr. Make sure this sentence encapsulates the main ideas of what is contained in the notes and is concise but clear.
        Please enclose the summary tldr sentence in square brackets, so I can parse it easily. Make sure your summary does not exceed one sentence and has 75 or less characters.
        This is the text: {text}. 
        Your summary: []
        """
        response = flash_llm(prompt)
        
        match = re.search(r'\[(.*?)\]', response)
        
        if match:
            summary = match.group(1)  
            return summary
        else:
            return response      
    except:
        return "tldr"
async def generate_flashcards(text, num_flashcards=10):
    prompt = (
        f"Generate flashcards from the following notes. "
        "Each flashcard should have a clear question and answer. "
        "Format each flashcard as follows:\n"
        "Question: [Your question here]\n"
        "Answer: [Your answer here]\n\n"
        "Make sure to include both the question and answer in your response for each flashcard and please prefix each question and answer by \'Question\' and \'Answer\'. \n"
        "The questions should cover a large range of the topics mentioned in the notes as they are meant to be comprehensive and ensure understanding of ALL the content. Include enough flashcards to cover all the content necessary.\n"
        f"Notes:\n{text}\n\n"
    )
     
    # Generate the output using the endpoint
    response =  flash_llm(prompt)
    
    # Debugging: Print the raw response
    print("Raw response:", response)
    
    # Process the response directly
    flashcards = []
    lines = response.strip().split('\n')
    
    # Iterate over lines to split into questions and answers
    i = 0
    while i < len(lines):
        # Check if the line starts with "Question" and there is a subsequent "Answer" line
        if lines[i].startswith("Question"):
            # Extract the question number
            question_number = lines[i].replace("Question", "").strip()
            # i += 1  # Move to the next line which might be the actual question text
            
            # Collect the question text (handle potential empty line)
            question = ""
            while i < len(lines) and not lines[i].startswith("Answer"):
                question += lines[i].replace("Question:", "").strip() + " "
                i += 1
            
            # Check if the next line is an answer
            if i < len(lines) and lines[i].startswith("Answer"):
                answer = lines[i].replace("Answer: ", "").strip()
                flashcards.append({
                    "question": question.strip(),
                    "answer": answer
                })
                i += 1  # Move past the answer line
        else:
            i += 1  # Skip any line that doesn't match the expected format
 
    if(len(flashcards)==0):
        lines = response.strip().split('\n')
        
        # Iterate over lines to split into questions and answers
        i = 0
        while i < len(lines):
            if lines[i].startswith("Question") and i + 1 < len(lines) and lines[i + 1].startswith("Answer"):
                question = lines[i].replace("Question: ", "").strip()
                answer = lines[i + 1].replace("Answer: ", "").strip()
                flashcards.append({
                    "question": question,
                    "answer": answer
                })
                i += 2  # Move to the next pair
            else:
                i += 1  # Skip any line that doesn't match the expected format


    return flashcards

 

@app.route('/suggestflashcards', methods=['POST'])
async def suggestflashcards():
    request_data = request.get_json()
    collection_id = request_data.get('collectionId')
    section_id = request_data.get('sectionId')
    notes_docs = db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('notes_in_section').stream()
    all_text = ''
    for doc in notes_docs:
        note_data = doc.to_dict().get('notes', '')
        all_text += note_data + ' '
    flashcards = await generate_flashcards(all_text)
   
    return jsonify({'response': flashcards}), 200


    

@app.route('/addflashcard', methods=['POST'])
def addflashcard():
    request_data = request.get_json()

    collection_id = request_data.get('collectionId')
    section_id = request_data.get('sectionId')
    question = request_data.get('question')
    answer = request_data.get('answer')
    if(check_profanity(question)):
            return jsonify({'error':'Profanity detected'}), 400
    if(check_profanity(answer)):
            return jsonify({'error':'Profanity detected'}), 400
    if not collection_id or not section_id or not question or not answer:
        return jsonify({'error': 'Invalid request data. Make sure all fields are provided.'}), 400

    try:
        # Ensure Firestore client is properly initialized
        if not db:
            return jsonify({'error': 'Firestore is not initialized.'}), 500

        # Ensure 'flashcards' collection exists (create it if not)
        flashcards_collection_ref = db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('flashcards')

        # Add flashcard document
        flashcard_doc_ref = flashcards_collection_ref.document()
        flashcard_doc_ref.set({'question': question, 'answer': answer})

        return jsonify({'response': 'Flashcard uploaded successfully'}), 200
    except Exception as e:
        # Log the full stack trace for debugging purposes
        import traceback
        traceback.print_exc()

        return jsonify({'error': str(e)}), 500



@app.route('/process_text', methods=['POST'])
def process_text():
    collection_id = request.form.get('collection_id')
    section_id = request.form.get('section_id')
    raw_text = request.form.get('raw_text')
    # if(check_profanity(raw_text)):
    #         return jsonify({'error':'Profanity detected'}), 400
    tldr = summarize_text(raw_text)


    try:
        notes_collection_ref = db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('notes_in_section')
        note_ref = notes_collection_ref.document()
        note_ref.set({'notes': raw_text, 'tldr': f"Raw Text: {tldr}"})
        
        return jsonify({'response': 'Raw text uploaded successfully'})
    except Exception as e:
        return jsonify({'error': str(e)})
    
# @app.route('/process_pdf', methods=['POST'])
# def process_pdf():
#     collection_id = request.form.get('collection_id')
#     section_id = request.form.get('section_id')
#     pdf_file = request.files.get('pdf_file')

#     if not pdf_file:
#         return jsonify({'error': 'No PDF file uploaded'}), 400

#     try:
#         # Read the PDF file
#         pdf_reader = PyPDF2.PdfFileReader(pdf_file)
#         text_content = ''
        
#         for page_num in range(pdf_reader.numPages):
#             page = pdf_reader.getPage(page_num)
#             text_content += page.extract_text()

#         # Save the text content to Firestore
#         notes_collection_ref = db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('notes_in_section')
#         note_ref = notes_collection_ref.document()
#         note_ref.set({'notes': text_content, 'tldr': "Text from PDF Uploaded"})
        
#         return jsonify({'response': 'PDF text uploaded successfully'})
#     except Exception as e:
#         return jsonify({'error': str(e)})

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    collection_id = request.form.get('collection_id')
    section_id = request.form.get('section_id')
    pdf_file = request.files.get('pdf_file')


    if not pdf_file:
        return jsonify({'error': 'No PDF file uploaded'}), 400

    try:
        # Read the PDF file
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text_content = ''
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text_content += page.extract_text()
            

        if(check_profanity(text_content)):
            return jsonify({'error':'Profanity detected'}), 400
        tldr = summarize_text(text_content)

        
        # Save the text content to Firestore
        notes_collection_ref = db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('notes_in_section')
        note_ref = notes_collection_ref.document()
        note_ref.set({'notes': text_content, 'tldr': f"PDF: {tldr}"})
        
        return jsonify({'response': 'PDF text uploaded successfully'}), 200
    except PyPDF2.utils.PdfReadError:
        return jsonify({'error': 'Failed to read PDF file'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/process_link', methods=['POST'])
def process_link():
    collection_id = request.form.get('collection_id')
    section_id = request.form.get('section_id')
    link = request.form.get('link')

    try:
        # Fetch the webpage content
        response = requests.get(link)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text content from the webpage
        text_content = ' '.join([p.text for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])])

        tldr = summarize_text(text_content)

        # Save the text content to Firestore
        notes_collection_ref = db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('notes_in_section')
        note_ref = notes_collection_ref.document()
        note_ref.set({'notes': text_content, 'tldr': f"Text from Link: {tldr}"})
        
        return jsonify({'response': 'Text and headings uploaded successfully'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_sections', methods=['GET'])
def get_sections():
    collection_id = request.args.get('collection_id')
    if not collection_id:
        return jsonify({'error': 'Collection ID not provided'})

    sections = []
    sections_docs = db.collection('collections').document(collection_id).collection('sections').stream()
    for doc in sections_docs:
        sections.append({'id': doc.id, 'section_name': doc.to_dict().get('section_name', '')})

    return jsonify({'sections': sections})


@app.route('/get_chapters', methods=['GET'])
def get_chapters():
    collection_id = request.args.get('collection_id')
    if not collection_id:
        return jsonify({'error': 'Collection ID not provided'})

    sections = []
    sections_docs = db.collection('chapters').document(collection_id).collection('colection_id').stream()
    for doc in sections_docs:
        sections.append({'id': doc.id, 'section_name': doc.to_dict().get('collection_id', '')})

    return jsonify({'chapters': sections})

@app.route('/create_section', methods=['POST'])
def create_section():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'})

    collection_id = data.get('collection_id')
    section_name = data.get('section_name')
    notes = data.get('notes', '')
    if(check_profanity(section_name)):
            return jsonify({'error':'Profanity detected'}), 400

    if not collection_id or not section_name:
        return jsonify({'error': 'Collection ID or Section name not provided'})

    collection_ref = db.collection('collections').document(collection_id)
    sections_ref = collection_ref.collection('sections').document()
    sections_ref.set({
        'section_name': section_name,
        'notes': notes,
        'visibility': 'public',
        'total_time': 0
    })

    return jsonify({'message': 'Section created successfully'})


@app.route('/ask_sydney', methods=['POST'])
def ask_sydney_route():
    data = request.get_json()
    if 'prompt' not in data:
        return jsonify({'error': 'Prompt not provided'})

    prompt = data['prompt']
    if(check_profanity(prompt)):
            return jsonify({'error':'Profanity detected'}), 400

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    response = loop.run_until_complete(ask_sydney_with_retry(prompt))
    response = jsonify({'response': response})
    response.headers.add('Access-Control-Allow-Origin', '*')  # Adjust the origin as needed
    return response
def get_notess(collection_id, section_id):
    notes_ref = db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('notes_in_section').stream()
    notes = [note.to_dict().get('notes', '') for note in notes_ref]
    return notes


@app.route('/get_flashcards', methods=['GET'])
def get_flashcards():
    section_id = request.args.get('section_id')
    collection_id = request.args.get('collection_id')

    if not section_id or not collection_id:
        return jsonify({'error': 'Section ID not provided'}), 400

    try:
        notes = []
        notes_docs = db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('flashcards').stream()
        for doc in notes_docs:
            note_data = doc.to_dict()
            note_data['id'] = doc.id
            notes.append(note_data)
        return jsonify({'flashcards': notes}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/top-sections', methods=['GET'])
def get_top_sections():
    username = request.args.get('username')

    try:
        sections = {}
        # Query sections created by the logged-in user
        collections_ref = db.collection('collections').where('username', '==', username)
        
        # Iterate over each document in the collections_ref query results
        for collection_doc in collections_ref.stream():
            collection_id = collection_doc.id
            sections_ref = db.collection('collections').document(collection_id).collection('sections').stream()
            
            for doc in sections_ref:
                section_data = doc.to_dict()
                section_name = section_data.get('section_name', '')
                section_time = section_data.get('total_time', 0)
                
                if section_name in sections:
                    sections[section_name]['total_time'] += section_time
                else:
                    sections[section_name] = {
                        'section_name': section_name,
                        'total_time': section_time
                    }

        # Sort the sections by total_time in descending order
        top_sections = sorted(sections.values(), key=lambda x: x['total_time'], reverse=True)[:5]

        return jsonify({'top_sections': top_sections}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/top-collections', methods=['GET'])
def get_top_collections():
    username = request.args.get('username')

    try:
        collections = {}
        # Query collections created by the logged-in user
        collections_ref = db.collection('collections').where('username', '==', username)
        
        # Iterate over each document in the collections_ref query results
        for collection_doc in collections_ref.stream():
            collection_id = collection_doc.id
            collection_data = collection_doc.to_dict()
            collection_name = collection_data.get('data', {}).get('title', '')
            
            # Query sections for the current collection
            sections_ref = db.collection('collections').document(collection_id).collection('sections').stream()
            
            # Calculate the total time for the current collection
            total_time = 0
            for section_doc in sections_ref:
                section_data = section_doc.to_dict()
                total_time += section_data.get('total_time', 0)
            
            if collection_name in collections:
                collections[collection_name]['total_time'] += total_time
            else:
                collections[collection_name] = {
                    'collection_name': collection_name,
                    'total_time': total_time
                }

        top_collections = sorted(collections.values(), key=lambda x: x['total_time'], reverse=True)[:5]

        return jsonify({'top_collections': top_collections}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_flashcards_frq', methods=['POST'])
def get_flashcards_frq():
    data = request.get_json()
    section_id = data.get('section_id')
    collection_id = data.get('collection_id')
    num_questions = data.get('num_questions')
    if not section_id:
        return jsonify({'error': 'Section ID not provided'}), 400

    if not collection_id:
        return jsonify({'error': 'Collection ID not provided'}), 400
    
    num_questions = int(num_questions)

    try:
        notes = []
        notes_docs = db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('flashcards').stream()
        for doc in notes_docs:
            note_data = doc.to_dict()
            note_data['id'] = doc.id
            notes.append(note_data)

        if len(notes) == 0:
            return jsonify({'error': 'No flashcards found'}), 404

        flashcards = [{'question': note['question'], 'answer': note['answer']} for note in notes]

        if num_questions > len(flashcards):
            num_questions = len(flashcards)

        random_flashcards = random.sample(flashcards, num_questions)

        return jsonify({'flashcards': random_flashcards}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/get_notes_ann', methods=['GET'])
def get_notes_ann():
    section_id = request.args.get('section_id')

    if not section_id:
        return jsonify({'error': 'Section ID not provided'}), 400

    try:
        notes = []
        collections = db.collection('collections').stream()
        for collection in collections:
            notes_docs = db.collection('collections').document(collection.id).collection('sections').document(section_id).collection('notes_in_section').stream()
            for doc in notes_docs:
                note_data = doc.to_dict()
                note = {
                    'id': doc.id,
                    'title': note_data['tldr'],
                    'notes': note_data['notes']
                }
                notes.append(note)
        return jsonify({'notes': notes}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_notes', methods=['GET'])
def get_notes():
    section_id = request.args.get('section_id')

    if not section_id:
        return jsonify({'error': 'Section ID not provided'}), 400

    try:
        notes = []
        collections = db.collection('collections').stream()
        for collection in collections:
            notes_docs = db.collection('collections').document(collection.id).collection('sections').document(section_id).collection('notes_in_section').stream()
            for doc in notes_docs:
                note_data = doc.to_dict()
                note_data['id'] = doc.id
                notes.append(note_data)
        return jsonify({'notes': notes}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
similarity_pipeline = pipeline("feature-extraction", model=model, tokenizer=tokenizer)
nlp = spacy.load('en_core_web_sm')
rake = Rake()

def extract_entities_and_phrases(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    rake.extract_keywords_from_text(text)
    key_phrases = rake.get_ranked_phrases()[:5]  # Get top 5 key phrases
    return entities + key_phrases

def perform_topic_modeling(notes):
    # Preprocess text
    texts = [[word for word in re.split('\W+', note.lower()) if word not in nlp.Defaults.stop_words]
             for note in notes]
    # Create a dictionary representation of the documents
    dictionary = corpora.Dictionary(texts)
    # Create a corpus
    corpus = [dictionary.doc2bow(text) for text in texts]
    # Apply LDA model
    lda = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)
    return lda.show_topics(formatted=False)

@app.route('/compare_and_fetch_concepts', methods=['POST'])
def compare_and_fetch_concepts():
    data = request.get_json()
    notes = data.get('notes', [])
    user_input = data.get('userInput', '')
    
    if not notes or not user_input:
        return jsonify({'error': 'Notes or user input is missing'}), 400

    user_embedding = similarity_pipeline(user_input)
    if isinstance(user_embedding, list):
        user_embedding = torch.tensor(user_embedding).mean(dim=0).unsqueeze(0)  # Ensure 2D shape
    else:
        user_embedding = user_embedding.mean(dim=0).unsqueeze(0)  # Ensure 2D shape

    results = []
    for note_text in notes:
        note_embedding = similarity_pipeline(note_text)
        if isinstance(note_embedding, list):
            note_embedding = torch.tensor(note_embedding).mean(dim=0).unsqueeze(0)  # Ensure 2D shape
        else:
            note_embedding = note_embedding.mean(dim=0).unsqueeze(0)  # Ensure 2D shape
        similarity = cosine_similarity(user_embedding.detach().numpy(), note_embedding.detach().numpy()).item()  # Convert to scalar
        results.append({'note_text': note_text, 'similarity': similarity})

    results.sort(key=lambda x: x['similarity'], reverse=True)

    concepts_mentioned = [note for note in results if note['similarity'] > 0.5]
    concepts_missed = [note for note in results if note['similarity'] <= 0.5]

    # Extract entities and phrases from the notes
    detailed_concepts_mentioned = []
    for concept in concepts_mentioned:
        extracted_concepts = extract_entities_and_phrases(concept['note_text'])
        detailed_concepts_mentioned.append({
            'note_text': concept['note_text'],
            'similarity': concept['similarity'],
            'detailed_concepts': extracted_concepts
        })
    
    # Perform topic modeling on the notes
    topics = perform_topic_modeling([note['note_text'] for note in notes])

    return jsonify({
        'concepts_mentioned': detailed_concepts_mentioned,
        'concepts_missed': concepts_missed,
        'topics': topics
    })

@app.route('/create_collection', methods=['POST'])
def create_collection():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'})

    collection_name = data.get('collection_name')
    notes = data.get('notes', '')
    username = data.get('username')
    if(check_profanity(collection_name)):
            return jsonify({'error':'Profanity detected'}), 400

    if not collection_name:
        return jsonify({'error': 'Collection name not provided'})

    if not username:
        return jsonify({'error': 'Username not provided'})

    # Store the collection in Firestore
    doc_ref = db.collection('collections').document()
    doc_ref.set({
        'username': username,
        'collectionIdentification': doc_ref.id,  # Using Firestore auto-generated ID
        'data': {
            'title': collection_name,
            'text': notes
        }
    })

    return jsonify({'message': 'Collection created successfully'})

@app.route('/get_collections', methods=['GET'])
def get_collections():
    collections = []
    collection_docs = db.collection('collections').stream()
    for doc in collection_docs:
        collection_data = doc.to_dict()
        collection_data['id'] = doc.id
        collections.append(collection_data)

    return jsonify({'collections': collections})



@app.route('/get_annotations', methods=['GET'])
def get_annotations():
    section_id = request.args.get('section_id')
    note_id = request.args.get('note_id')

    if not section_id:
        return jsonify({'error': 'Section ID not provided'}), 400

    try:
        notes = []
        collections = db.collection('collections').stream()
        for collection in collections:
            notes_docs = db.collection('collections').document(collection.id).collection('sections').document(section_id).collection('notes_in_section').document(note_id).collection('annotations').stream()
            for doc in notes_docs:
                note_data = doc.to_dict()
                note_data['id'] = doc.id
                notes.append(note_data)
        return jsonify({'annotations': notes}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/upload_annotation', methods=['POST'])
def upload_annotation():
    
    data = request.json

    collection_id = data.get('collection_id')
    section_id = data.get('section_id')
    note_id = data.get('note_id')
    annotation = data.get('annotation')
    start_ind = data.get('start')
    end_ind = data.get('end')

    if not collection_id or not section_id or not note_id:
        return jsonify({'error': 'Collection ID or Section ID or Note ID not provided'}), 400

    try:
        notes_collection_ref = db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('notes_in_section').document(note_id).collection('annotations')
        note_ref = notes_collection_ref.document()  

        note_ref.set({'annotation': annotation , 'start_ind': start_ind, 'end_ind':end_ind})

        return jsonify({ 'message': 'Text recognized and stored successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete_annotation', methods=['DELETE'])
def delete_annotation():
    collection_id = request.args.get('collection_id')
    section_id = request.args.get('section_id')
    note_id = request.args.get('note_id')
    annotation_id = request.args.get('annotation_id')

    if not collection_id or not section_id or not note_id or not annotation_id:
        return jsonify({'error': 'Collection ID, Section ID, Note ID or Annotation ID not provided'}), 400

    try:
        notes_collection_ref = db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('notes_in_section').document(note_id).collection('annotations')
        annotation_ref = notes_collection_ref.document(annotation_id)

        annotation_ref.delete()

        return jsonify({'message': 'Annotation deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/edit_annotation', methods=['PUT'])
def edit_annotation():
    data = request.json

    collection_id = data.get('collection_id')
    section_id = data.get('section_id')
    note_id = data.get('note_id')
    annotation_id = data.get('annotation_id')
    new_annotation = data.get('new_annotation')

    if not collection_id or not section_id or not note_id or not annotation_id or not new_annotation:
        return jsonify({'error': 'Collection ID, Section ID, Note ID, Annotation ID or new annotation not provided'}), 400

    try:
        notes_collection_ref = db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('notes_in_section').document(note_id).collection('annotations')
        annotation_ref = notes_collection_ref.document(annotation_id)

        annotation_ref.update({
            'annotation': new_annotation
        })

        return jsonify({'message': 'Annotation updated successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500





@app.route('/get_worksheets', methods=['GET'])
def get_worksheets():
    section_id = request.args.get('section_id')

    if not section_id:
        return jsonify({'error': 'Section ID not provided'}), 400

    try:
        notes = []
        collections = db.collection('collections').stream()
        for collection in collections:
            notes_docs = db.collection('collections').document(collection.id).collection('sections').document(section_id).collection('worksheets').stream()
            for doc in notes_docs:
                note_data = doc.to_dict()
                note_data['id'] = doc.id
                notes.append(note_data)
        return jsonify({'worksheets': notes}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/upload_worksheet', methods=['POST'])
def upload_worksheet():
    if 'worksheet' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    collection_id = request.form.get('collection_id')
    section_id = request.form.get('section_id')

    if not collection_id or not section_id:
        return jsonify({'error': 'Collection ID or Section ID not provided'}), 400

    image_file = request.files['worksheet']
    image_stream = io.BytesIO(image_file.read())
    image = Image.open(image_stream)

    recognized_text = pytesseract.image_to_string(image)
    # if(check_profanity(recognized_text)):
    #         return jsonify({'error':'Profanity detected'}), 400
    try:
        notes_collection_ref = db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('worksheets')
        note_ref = notes_collection_ref.document()  
        tldr = summarize_text(recognized_text)

        note_ref.set({'worksheet': recognized_text, 'tldr': tldr})

        return jsonify({'text': recognized_text, 'tldr': tldr, 'message': 'Text recognized and stored successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/recognize', methods=['POST'])
def recognize_handwriting():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    collection_id = request.form.get('collection_id')
    section_id = request.form.get('section_id')

    if not collection_id or not section_id:
        return jsonify({'error': 'Collection ID or Section ID not provided'}), 400

    image_file = request.files['image']
    image_stream = io.BytesIO(image_file.read())
    image = Image.open(image_stream)

    recognized_text = pytesseract.image_to_string(image)
    # if(check_profanity(recognized_text)):
    #         return jsonify({'error':'Profanity detected'}), 400
    
    try:
        notes_collection_ref = db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('notes_in_section')
        note_ref = notes_collection_ref.document()  
        tldr = summarize_text(recognized_text)


        note_ref.set({'notes': recognized_text, 'tldr':("Notes: "+tldr)})

        return jsonify({'text': recognized_text, 'tldr':tldr, 'message': 'Text recognized and stored successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search_public_sections', methods=['GET'])
async def search_public_sections():
    search_term = request.args.get('search_term')
    search_category = request.args.get('search_category')
    name = request.args.get('name')

    if check_profanity(search_term):
        return jsonify({'error': 'Profanity detected'}), 400
    
    if not search_term:
        return jsonify({'error': 'Search term not provided'}), 400

    if search_category == 'publicsets':
        return await search_public_sections(search_term, name)
    elif search_category == 'users':
        return await search_users(search_term)
    else:
        return jsonify({'error': 'Invalid search category'}), 400

async def search_public_sections(search_term, name):
    public_sections = []
    
    section_docs = db.collection('collections').where('username', '!=', name).stream()
    for doc in section_docs:
        section_ref = doc.reference.collection('sections').where('visibility', '==', 'public').stream()
        for section_doc in section_ref:
            section_data = section_doc.to_dict()
            title = section_data.get('section_name', '')
            if search_term.lower() in title.lower():
                public_sections.append({'id': section_doc.id, 'title': title})

    return jsonify({'sections': public_sections})

async def search_users(search_term):
    users = []
    
    user_docs = db.collection('users').stream()
    for user_doc in user_docs:
        user_data = user_doc.to_dict()
        username = user_data.get('username', '')
        friend_requests = user_data.get('friend_requests', [])
        friends = user_data.get('friends',[])
        if search_term.lower() in username.lower():
            users.append({
                'id': user_doc.id, 
                'username': username, 
                'friend_requests': friend_requests,
                'email': user_doc.id,
                'friends':friends
            })

    return jsonify({'users': users})


@app.route('/friend-public-sections/<friend_id>', methods=['GET'])
def get_friend_public_sections(friend_id):
    try:
        public_sections = []
        
        # Get all collections for the friend
        friend_collections = db.collection('collections').where('username', '==', friend_id).stream()
        
        for collection in friend_collections:
            # For each collection, get all public sections
            section_ref = collection.reference.collection('sections').where('visibility', '==', 'public').stream()
            for section_doc in section_ref:
                section_data = section_doc.to_dict()
                title = section_data.get('section_name', '')
                public_sections.append({
                    'id': section_doc.id,
                    'title': title
                })

        return jsonify({'sections': public_sections})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/send_friend_request', methods=['POST'])
def send_friend_request():
    data = request.json
    from_user = data.get('from')
    to_user = data.get('to')

    if not from_user or not to_user:
        return jsonify({'error': 'Both "from" and "to" users are required'}), 400

    try:


        users_ref = db.collection('users')
        query = users_ref.where('username', '==', to_user).limit(1)
        user_docs = query.get()

        if not user_docs:
            return jsonify({'error': 'Recipient user not found'}), 404

        user_doc = user_docs[0]
        user_data = user_doc.to_dict()
        friend_requests = user_data.get('friend_requests', [])

        # Check if the request already exists
        if from_user in friend_requests:
            return jsonify({'message': 'Friend request already sent'}), 400

        # Add the friend request
        friend_requests.append(from_user)
        user_doc.reference.update({'friend_requests': friend_requests})

        return jsonify({'message': 'Friend request sent successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/get_follow_requests', methods=['GET'])
def get_follow_requests():
    username = request.args.get('username')
    
    if not username:
        return jsonify({'error': 'Username is required'}), 400

    try:
        user_doc = db.collection('users').document(username).get()

        if not user_doc.exists:
            return jsonify({'error': 'User not found'}), 404

        user_data = user_doc.to_dict()
        friend_requests = user_data.get('friend_requests', [])

        return jsonify({'follow_requests': friend_requests}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/accept_friend_request', methods=['POST'])
def accept_friend_request():
    data = request.json
    user = data.get('user')
    requester = data.get('requester')

    if not user or not requester:
        return jsonify({'error': 'Both user and requester are required'}), 400

    try:
        user_doc = db.collection('users').document(user).get()
        requester_doc = db.collection('users').document(requester).get()

        if not user_doc.exists or not requester_doc.exists:
            return jsonify({'error': 'User or requester not found'}), 404

        # Update user's data
        user_data = user_doc.to_dict()
        friend_requests = user_data.get('friend_requests', [])
        friends = user_data.get('friends', [])

        if requester not in friend_requests:
            return jsonify({'error': 'Friend request not found'}), 404

        friend_requests.remove(requester)
        friends.append(requester)

        db.collection('users').document(user).update({
            'friend_requests': friend_requests,
            'friends': friends
        })

        # Update requester's data
        requester_data = requester_doc.to_dict()
        requester_friends = requester_data.get('friends', [])
        requester_friends.append(user)

        db.collection('users').document(requester).update({
            'friends': requester_friends
        })

        return jsonify({'message': 'Friend request accepted successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/decline_friend_request', methods=['POST'])
def decline_friend_request():
    data = request.json
    user = data.get('user')
    requester = data.get('requester')

    if not user or not requester:
        return jsonify({'error': 'Both user and requester are required'}), 400

    try:
        user_doc = db.collection('users').document(user).get()

        if not user_doc.exists:
            return jsonify({'error': 'User not found'}), 404

        user_data = user_doc.to_dict()
        friend_requests = user_data.get('friend_requests', [])

        if requester not in friend_requests:
            return jsonify({'error': 'Friend request not found'}), 404

        friend_requests.remove(requester)

        db.collection('users').document(user).update({
            'friend_requests': friend_requests
        })

        return jsonify({'message': 'Friend request declined successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/get_friends', methods=['GET'])
def get_friends():
    username = request.args.get('username')
    
    if not username:
        return jsonify({'error': 'Username is required'}), 400

    try:
        user_doc = db.collection('users').document(username).get()

        if not user_doc.exists:
            return jsonify({'error': 'User not found'}), 404

        user_data = user_doc.to_dict()
        friends = user_data.get('friends', [])

        return jsonify({'friends': friends}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save_test_score', methods=['POST'])
def save_test_score():
    data = request.get_json()
   
    collection_id = data.get('collection_id')
    section_id = data.get('section_id')
    percentage = data.get('percentage')

    if not collection_id or not section_id or not percentage:
        return jsonify({'error': 'Collection ID or Section id or percentage not provided'})
    scores = db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('test_scores').document()
    scores.set({
        'score': percentage
    })
    return jsonify({'success':'success'}), 200
@app.route('/most_challenging', methods=['GET'])
def get_most_challenging_sections():
    username = request.args.get('username')

    try:
        sections = {}
        # Query sections created by the logged-in user
        collections_ref = db.collection('collections').where('username', '==', username)

        # Iterate over each document in the collections_ref query results
        for collection_doc in collections_ref.stream():
            collection_id = collection_doc.id
            sections_ref = db.collection('collections').document(collection_id).collection('sections').stream()

            for doc in sections_ref:
                section_data = doc.to_dict()
                section_name = section_data.get('section_name', '')
                section_id = doc.id

                # Calculate the average test score for the section
                test_scores_ref = db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('test_scores')
                test_scores = [float(score.to_dict().get('score', 0)) for score in test_scores_ref.stream()]

                if not test_scores:
                    continue  # Skip sections with no test scores

                avg_score = sum(test_scores) / len(test_scores)

                if section_name in sections:
                    current_avg = sections[section_name]['avg_score']
                    num_tests = sections[section_name]['num_tests']
                    new_avg = (current_avg * num_tests + avg_score) / (num_tests + 1)
                    sections[section_name]['avg_score'] = new_avg
                    sections[section_name]['num_tests'] += 1
                else:
                    sections[section_name] = {
                        'section_name': section_name,
                        'avg_score': avg_score,
                        'num_tests': 1
                    }

        # Sort the sections by average score in ascending order (lowest to highest)
        most_challenging_sections = sorted(sections.values(), key=lambda x: x['avg_score'])[:5]

        return jsonify({'most_challenging_sections': most_challenging_sections}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save_response', methods=['POST'])
def save_response():
    data = request.get_json()
   
    collection_id = data.get('collection_id')
    section_id = data.get('section_id')
    response = data.get('response')
    question = data.get('question')

    if not collection_id or not section_id:
        return jsonify({'error': 'Collection ID or Section id not provided'})

    tldr = response.split('.')[0] + '.'


    responses = db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('saved_responses').document()
    responses.set({
        'data': response,
        'tldr': tldr,
        'question':question
    })
    return jsonify({'success':'success'}), 200

@app.route('/get_saved_responses', methods=['GET'])
def get_saved_responses():
    collection_id = request.args.get('collection_id')
    section_id = request.args.get('section_id')

    if not collection_id or not section_id:
        return jsonify({'error': 'Collection ID or Section ID not provided'}), 400

    try:
        notes = []
        notes_docs = db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('saved_responses').stream()
        for doc in notes_docs:
            note_data = doc.to_dict()
            note_data['id'] = doc.id
            notes.append(note_data)
        return jsonify({'notes': notes}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/delete_response', methods=['DELETE'])
def delete_response():
    collection_id = request.args.get('collection_id')
    section_id = request.args.get('section_id')
    response_id = request.args.get('response_id')

    if not collection_id or not section_id or not response_id:
        return jsonify({'error': 'Collection ID, Section ID, or Response ID not provided'}), 400

    try:
        db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('saved_responses').document(response_id).delete()
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/create_section_from_recommendation', methods=['POST'])
def create_section_from_recommendation():
    data = request.get_json()
    print(data)
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    collection_id = data.get('collection_id')
    section_data = data.get('section_data')

    if not section_data:
        return jsonify({'error': 'Section data not provided'}), 400

    try:
        if collection_id:
            # Add to an existing collection
            collection_ref = db.collection('collections').document(collection_id)
            if not collection_ref.get().exists:
                return jsonify({'error': 'Collection not found'}), 404
        else:
            # Create a new collection
            username = data.get('username')
            if not username:
                return jsonify({'error': 'Username not provided'}), 400

            collection_name = data.get('collection_name')
            if not collection_name:
                return jsonify({'error': 'Collection name not provided'}), 400

            collection_ref = db.collection('collections').document()
            collection_ref.set({
                'collectionIdentification': collection_ref.id,
                'username': username,
                'data': {'title': collection_name}
            })

        # Create the new section
        new_section_ref = collection_ref.collection('sections').document()
        new_section_ref.set({
            'section_name': section_data['topicName'],
            'visibility':'public'
        })
        link=section_data['sources'][0]['url']
        response = requests.get(link)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text content from the webpage
        text_content = ' '.join([p.text for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])])

        tldr = summarize_text(text_content)

        for source in section_data['sources']:
            new_note_ref = new_section_ref.collection('notes_in_section').document()
            new_note_ref.set({
                'notes': f"{text_content}",
                'tldr': tldr
            })

        return jsonify({'message': 'Section created successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

async def get_recommendations(username, recent_sections):
    async with SydneyClient() as sydney:

        # Construct the question string
        question = f'''Based on the user's most recently studied sections, here are their recent areas of focus: {recent_sections}. Can you find and provide 3 specific advanced topics, supplementary materials (books, articles, videos, and online resources including real-world applications) that they should explore further? Answer in this format:
        
        Recommended Topic 1:
        Topic Name:
        Topic Description:
        Sources:

        Recommended Topic 2:
        Topic Name:
        Topic Description:
        Sources:

        Recommended Topic 3:
        Topic Name:
        Topic Description:
        Sources:

        '''

        # Ask the question to the language model
        response = await sydney.ask(question, citations=True)
        return response


@app.route('/recommendations', methods=['GET'])
def get_recommendations_endpoint():
    username = request.args.get('username')
    recent_sections = request.args.getlist('recentSections')

    if not username:
        return jsonify({'error': 'Username not provided'}), 400

    recommendations = asyncio.run(get_recommendations(username, recent_sections))
    
    return jsonify({'recommendations': recommendations})

@app.route('/process_recommendations', methods=['GET'])
def process_recommendations():
    recommendations_text = request.args.get('recs')
    recommendations = []
    print("Received recommendations text:")
    print(recommendations_text)

    # Adjusted regex pattern to match the given text structure
    pattern = re.compile(
        r'\*\*(.*?)\*\*:\s*'                  # Topic Name
        r'- \*\*Topic\s*Description\*\*:\s*(.*?)'  # Topic Description
        r'- \*\*Sources\*\*:\s*'              # Sources label
        r'((?:- \[.*?\]\(.*?\)\s*)+)',        # One or more sources
        re.DOTALL
    )
    pattern1 = re.compile(
        r'\*\*Topic\s*Name\*\*:\s*(.*?)\n'           # Topic Name
        r'\*\*Topic\s*Description\*\*:\s*(.*?)\n'    # Topic Description
        r'\*\*Sources\*\*:\s*'                       # Sources label
        r'((?:- \[.*?\]\(.*?\)\n)+)',                # One or more sources
        re.DOTALL
    )
    pattern2 = re.compile(
        r'\d+\.\s\*\*(.*?)\*\*:\s*'           # Topic Title
        r'- \*\*Topic Name\*\*:\s*(.*?)\n'     # Topic Name
        r'- \*\*Topic Description\*\*:\s*(.*?)\n'  # Topic Description
        r'- \*\*Sources\*\*:\s*'               # Sources label
        r'((?:- \[.*?\]\(.*?\)\n)+)'           # One or more sources
    )
    pattern3 = re.compile(
        r'\*\*(.*?)\*\*:\s*'                  # Topic Title
        r'- \*\*Topic\s*Name\*\*:\s*(.*?)\n'  # Topic Name
        r'- \*\*Topic\s*Description\*\*:\s*(.*?)\n'  # Topic Description
        r'- \*\*Sources\*\*:\s*'               # Sources label
        r'((?:- \[.*?\]\(.*?\)\n)+)'           # One or more sources
    )
    pattern4 = re.compile(
        r'\d+\.\s\*\*(.*?)\*\*:\s*'                  # Topic Title
        r'- \*\*Topic\s*Name\*\*:\s*(.*?)\s*'         # Topic Name
        r'- \*\*Topic\s*Description\*\*:\s*(.*?)\s*'  # Topic Description
        r'- \*\*Sources\*\*:\s*'                      # Sources label
        r'((?:- \[.*?\]\(.*?\)\s*)+)',                # One or more sources
        re.DOTALL
    )

    matches = pattern.findall(recommendations_text)
    print("Regex matches:")
    print(matches)
    for match in matches:
        print("Processing match:")
        print(match)
        topic_name = match[0].strip()
        topic_description = match[1].strip()
        sources = re.findall(r'- \[(.*?)\]\((.*?)\)', match[2].strip())

        sources_list = [{'title': source[0], 'url': source[1]} for source in sources]

        recommendations.append({
            'topicName': topic_name,
            'topicDescription': topic_description,
            'sources': sources_list
        })
    if(len(recommendations)==0):
        matches = pattern1.findall(recommendations_text)
        print("Regex matches:")
        print(matches)
        for match in matches:
            print("Processing match:")
            print(match)
            topic_name = match[0].strip()
            topic_description = match[1].strip()
            sources = re.findall(r'- \[(.*?)\]\((.*?)\)\n', match[2].strip())

            sources_list = [{'title': source[0], 'url': source[1]} for source in sources]

            recommendations.append({
                'topicName': topic_name,
                'topicDescription': topic_description,
                'sources': sources_list
            })
    if(len(recommendations)==0):
        matches = pattern2.findall(recommendations_text)
        print("Regex matches:")
        print(matches)
        for match in matches:
            print("Processing match:")
            print(match)
            topic_title = match[0].strip()
            topic_name = match[1].strip()
            topic_description = match[2].strip()
            sources = re.findall(r'- \[(.*?)\]\((.*?)\)\n', match[3].strip())

            sources_list = [{'title': source[0], 'url': source[1]} for source in sources]

            recommendations.append({
                'topicTitle': topic_title,
                'topicName': topic_name,
                'topicDescription': topic_description,
                'sources': sources_list
            })
    if(len(recommendations)==0):
        matches = pattern3.findall(recommendations_text)
        print("Regex matches:")
        print(matches)
        for match in matches:
            print("Processing match:")
            print(match)
            topic_title = match[0].strip()
            topic_name = match[1].strip()
            topic_description = match[2].strip()
            sources = re.findall(r'- \[(.*?)\]\((.*?)\)\n', match[3].strip())

            sources_list = [{'title': source[0], 'url': source[1]} for source in sources]

            recommendations.append({
                'topicTitle': topic_title,
                'topicName': topic_name,
                'topicDescription': topic_description,
                'sources': sources_list
            })
    if(len(recommendations)==0):
        matches = pattern4.findall(recommendations_text)
        print("Regex matches:")
        print(matches)
        for match in matches:
            print("Processing match:")
            print(match)
            topic_title = match[0].strip()
            topic_name = match[1].strip()
            topic_description = match[2].strip()
            sources = re.findall(r'- \[(.*?)\]\((.*?)\)\s*', match[3].strip())

            sources_list = [{'title': source[0], 'url': source[1]} for source in sources]

            recommendations.append({
                'topicTitle': topic_title,
                'topicName': topic_name,
                'topicDescription': topic_description,
                'sources': sources_list
            })

    print("Processed recommendations:")
    print(recommendations)
    return jsonify({'recommendations': recommendations})



@app.route('/add_response_to_notes', methods=['POST'])
def add_to_notes():
    data = request.get_json()
    collection_id = data.get('collection_id')
    section_id = data.get('section_id')
    response_id = data.get('response_id')

    if not collection_id or not section_id or not response_id:
        return jsonify({'error': 'Collection ID, Section ID, or Response ID not provided'}), 400

    try:
        response_ref = db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('saved_responses').document(response_id)
        response_doc = response_ref.get()

        if response_doc.exists:
            response_data = response_doc.to_dict()
           

            notes_ref = db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('notes_in_section').document()
            notes_ref.set({
                'notes': response_data['data'],
                'tldr': f"AI Response: {response_data['data'].split('.')[0] + '.'}"
            })

            response_ref.delete()

            return jsonify({'success': True}), 200
        else:
            return jsonify({'error': 'Response not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/get_all_public_sections', methods=['GET'])
def get_all_public_sections():
    public_sections = []

    try:
        section_docs = db.collection('collections').stream()
        for doc in section_docs:
            section_ref = doc.reference.collection('sections').where('visibility', '==', 'public').stream()
            for section_doc in section_ref:
                section_data = section_doc.to_dict()
                title = section_data.get('section_name', '')
                public_sections.append({'id': section_doc.id, 'title': title})
    except Exception as e:
        print("Error fetching public sections:", e)
        return jsonify({'error': 'Failed to fetch public sections'}), 500

    return jsonify({'sections': public_sections})

# @app.route('/generate_questions', methods=['POST'])
# async def generate_questions():
#     data = request.get_json()
#     if not data:
#         return jsonify({'error': 'No data provided'}), 400

#     collection_id = data.get('collection_id')
#     section_id = data.get('section_id')
#     num_questions = data.get('num_questions')

#     if not num_questions:
#         return jsonify({'error': 'Number of questions not provided'}), 400

#     if not collection_id or not section_id:
#         return jsonify({'error': 'Collection ID or Section ID not provided'}), 400

#     try:
#         notes_docs = db.collection('collections').document(collection_id).collection('sections').document(section_id).collection('notes_in_section').stream()
#         all_text = ''
#         for doc in notes_docs:
#             note_data = doc.to_dict().get('notes', '')
#             all_text += note_data + ' '

#         if not all_text:
#             return jsonify({'error': 'No notes found in the specified section'}), 404

#         async with SydneyClient() as sydney:
#             question = f'''
#             Task: Generate 10 practice test questions based on the following notes. The question should cover the main topics. Ensure the question is of difficulty difficulty and is question_type.

#             Notes:
#             {all_text}

#             Question Format: Multiple Choice
#             Question: [Your question here]
#             Options: [Option 1, Option 2, Option 3, Option 4]
#             Answer: [Correct answer]
#             '''
#             response = await sydney.ask(question, citations=True)
#             return jsonify({'response': response}), 200

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

@app.route('/parse_questions', methods=['POST'])
async def parse_questions():
    # data = request.get_json()
    # if not data:
    #     return jsonify({'error': 'No data provided'}), 400

    # response = data.get('response')
    # if not response:
    #     return jsonify({'error': 'No response provided'}), 400
    # try:
            # question_pattern = re.compile(r"\d+\.\s\*\*Question\*\*:\s(.*?)\n((?:\s+-\s.*?\n)*?)\s+-\s\*\*Answer\*\*:\s(.*?)\n", re.DOTALL)
            # question_pattern1 = re.compile(r"(\d+)\. \*\*Question\*\*: (.+?)\n((?:\s+- [A-D]\) .+?\n)+)\n\[\d+\]\s+\*\*Answer\*\*: ([A-D]\)) (.+?)\n", re.MULTILINE)
            # question_pattern2 = r'\d+\.\s+\*\*Question\*\*: (.+?)\n\s+-\s+A\)\s(.+?)\n\s+-\s+B\)\s(.+?)\n\s+-\s+C\)\s(.+?)\n\s+-\s+D\)\s(.+?)\n\s+\*\*Answer\*\*: (.+?)\n\n'
            # question_pattern3 = re.compile(r'\d+\.\s+\*\*Question\*\*: (.*?)\n((?:\s+- [A-D]\) .+?\n)+)\*\*Answer\*\*: ([A-D]\)) (.+?)\n', re.MULTILINE | re.DOTALL)
            # question_pattern4 = re.compile(r'\d+\.\s+(.*?)\n((?:\s+- [A-D]\. .+?\n)+)\*\*Answer: ([A-D])\. (.+?)\*\*', re.DOTALL)
            # question_pattern5 = re.compile(r'\d+\.\s+(.*?)\n((?:\s+- [A-D]\) .+?\n)+)\*\*Answer: ([A-D])\) (.+?)\*\*', re.DOTALL)

            # question_pattern = re.compile(r"(\d+)\.\s*\*\*Question\*\*:\s*(.*?)\n\s*-\s*A\)\s*(.*?)\n\s*-\s*B\)\s*(.*?)\n\s*-\s*C\)\s*(.*?)\n\s*-\s*D\)\s*(.*?)\n\s*\*\*Answer\*\*:\s*(.*?)\n")

            # matches = question_pattern.findall(response)
            
            # # Log the intermediate matches
            # print(f"Matches: {matches}")
        try:
            qna_pairs = []
            # for match in matches:
            #     question_number = match[0]
            #     question_text = match[1].strip()
            #     options = [
            #         {"option": "A", "text": match[2].strip()},
            #         {"option": "B", "text": match[3].strip()},
            #         {"option": "C", "text": match[4].strip()},
            #         {"option": "D", "text": match[5].strip()}
            #     ]
            #     answer = match[6].strip()

            #     # Log each question and its options
            #     print(f"Question {question_number}: {question_text}")
            #     print(f"Options: {options}")
            #     print(f"Answer: {answer}")

            #     qna_pairs.append({
            #         "question": question_text,
            #         "options": options,
            #         "answer": answer
            #     })
            # if len(qna_pairs) == 0:
            qna_pairs.append({
                    "question": "What is recursion in programming?",
                    "options": [
                        ("A", "A method to iterate over data."),
                        ("B", "A way to solve problems by breaking them down into smaller instances."),
                        ("C", "A type of loop in programming."),
                        ("D", "A sorting algorithm.")
                    ],
                    "answer": ("B", "A way to solve problems by breaking them down into smaller instances.")
                    }, {
                    "question": "What is the base case in recursion?",
                    "options": [
                        ("A", "The case where the function stops calling itself."),
                        ("B", "The smallest instance of the problem that can be solved directly."),
                        ("C", "A recursive function."),
                        ("D", "An iterative solution.")
                    ],
                    "answer": ("B", "The smallest instance of the problem that can be solved directly.")
                },
                {
                    "question": "What happens if a recursive function does not have a base case?",
                    "options": [
                        ("A", "The function will cause a stack overflow."),
                        ("B", "The function will run forever."),
                        ("C", "The function will return incorrect results."),
                        ("D", "All of the above.")
                    ],
                    "answer": ("D", "All of the above.")
                },
                {
                    "question": "Which of the following data structures uses recursion inherently?",
                    "options": [
                        ("A", "Array"),
                        ("B", "Linked List"),
                        ("C", "Queue"),
                        ("D", "Binary Tree")
                    ],
                    "answer": ("D", "Binary Tree")
                },
                {
                    "question": "What is the time complexity of recursive Fibonacci function?",
                    "options": [
                        ("A", "O(1)"),
                        ("B", "O(n)"),
                        ("C", "O(2^n)"),
                        ("D", "O(log n)")
                    ],
                    "answer": ("C", "O(2^n)")
                },
                {
                    "question": "Which of the following problems can be solved using recursion?",
                    "options": [
                        ("A", "Finding factorial of a number"),
                        ("B", "Finding shortest path in a graph"),
                        ("C", "Sorting an array"),
                        ("D", "All of the above")
                    ],
                    "answer": ("D", "All of the above")
                },
                {
                    "question": "What is tail recursion?",
                    "options": [
                        ("A", "A type of recursion that uses an explicit stack."),
                        ("B", "A type of recursion where the recursive call is the last thing executed by the function."),
                        ("C", "A type of recursion that involves multiple recursive calls."),
                        ("D", "A type of recursion that is slower than iterative solutions.")
                    ],
                    "answer": ("B", "A type of recursion where the recursive call is the last thing executed by the function.")
                },
                {
                    "question": "Which of the following is NOT required for a recursive function?",
                    "options": [
                        ("A", "Base case"),
                        ("B", "Recursive case"),
                        ("C", "Initialization"),
                        ("D", "Iteration")
                    ],
                    "answer": ("D", "Iteration")
                },
                {
                    "question": "What is indirect recursion?",
                    "options": [
                        ("A", "A recursion that calls itself indirectly through another function."),
                        ("B", "A recursion that does not have a base case."),
                        ("C", "A recursion that is faster than direct recursion."),
                        ("D", "A recursion that has only one recursive call.")
                    ],
                    "answer": ("A", "A recursion that calls itself indirectly through another function.")
                },
                {
                    "question": "Which of the following algorithms uses recursion?",
                    "options": [
                        ("A", "Merge Sort"),
                        ("B", "Bubble Sort"),
                        ("C", "Insertion Sort"),
                        ("D", "Selection Sort")
                    ],
                    "answer": ("A", "Merge Sort")
                })


            
            return jsonify({'r': qna_pairs}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500



@app.route('/section_visibility', methods=['GET', 'POST'])
def visibility():
    if request.method == 'POST':
        data = request.get_json()
        if not data:
            app.logger.error('No data provided')
            return jsonify({'error': 'No data provided'}), 400

        collection_id = data.get('collection_id')
        section_id = data.get('section_id')
        visibility = data.get('visibility')

        if not collection_id or not section_id or not visibility:
            app.logger.error('Collection ID, Section ID or visibility not provided')
            return jsonify({'error': 'Collection ID, Section ID or visibility not provided'}), 400

        section_ref = db.collection('collections').document(collection_id).collection('sections').document(section_id)
        
        try:
            section_ref.update({'visibility': visibility})
            app.logger.info(f'Visibility updated to {visibility} for collection {collection_id}, section {section_id}')
        except Exception as e:
            app.logger.error(f'Error updating visibility: {e}')
            return jsonify({'error': 'Failed to update visibility'}), 500

        return jsonify({'message': 'Visibility updated successfully'})

    elif request.method == 'GET':
        collection_id = request.args.get('collection_id')
        section_id = request.args.get('section_id')

        if not collection_id or not section_id:
            app.logger.error('Collection ID or Section ID not provided')
            return jsonify({'error': 'Collection ID or Section ID not provided'}), 400

        section_ref = db.collection('collections').document(collection_id).collection('sections').document(section_id)
        section = section_ref.get()

        if not section.exists:
            app.logger.error('Section not found')
            return jsonify({'error': 'Section not found'}), 404

        return jsonify({'section': section.to_dict()})
@app.route('/clone_section', methods=['POST'])
def clone_section():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    section_id = data.get('sectionId')
    if not section_id:
        return jsonify({'error': 'Section ID not provided'}), 400

    try:
        # Query all collections to find the section
        collections = db.collection('collections').stream()
        for collection in collections:
            section_ref = collection.reference.collection('sections').document(section_id)
            section_doc = section_ref.get()
            if section_doc.exists:
                # Section found, retrieve associated notes
                notes_ref = section_ref.collection('notes_in_section')
                notes = [note.to_dict() for note in notes_ref.stream()]
                break
        else:
            # Section not found in any collection
            return jsonify({'error': 'Section not found'}), 404

        # Get the payload data
        add_to_existing = data.get('addToExisting')
        if add_to_existing is None:
            return jsonify({'error': 'Parameter "addToExisting" not provided'}), 400

        if add_to_existing:
            # Add to an existing collection
            collection_id = data.get('collectionId')
            if not collection_id:
                return jsonify({'error': 'Collection ID not provided'}), 400

            # Check if the specified collection exists
            collection_ref = db.collection('collections').document(collection_id)
            if not collection_ref.get().exists:
                return jsonify({'error': 'Collection not found'}), 404

            # Copy the section to the specified collection
            new_section_ref = collection_ref.collection('sections').document()
            new_section_ref.set(section_doc.to_dict())
            # Copy associated notes
            for note in notes:
                new_note_ref = new_section_ref.collection('notes_in_section').document()
                new_note_ref.set(note)

        else:
            # Create a new collection and add the section to it
            username = data.get('username')
            if not username:
                return jsonify({'error': 'Username not provided'}), 400

            collection_name = data.get('collectionName')
            if(check_profanity(collection_name)):
                return jsonify({'error':'Profanity detected'}), 400
    
            if not collection_name:
                return jsonify({'error': 'Collection name not provided'}), 400

            # Create a new collection with the specified username and title
            new_collection_ref = db.collection('collections').document()
            new_collection_ref.set({'collectionIdentification':new_collection_ref.id,'username': username, 'data': {'title': collection_name}})

            # Copy the section to the new collection
            new_section_ref = new_collection_ref.collection('sections').document()
            new_section_ref.set(section_doc.to_dict())
            # Copy associated notes
            for note in notes:
                new_note_ref = new_section_ref.collection('notes_in_section').document()
                new_note_ref.set(note)

        return jsonify({'message': 'Section cloned successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

async def practtest() -> str:  # Change the return type to str
    notes = request.args.get('notes')

    async with SydneyClient() as sydney:
        question = f'''
        Task: Generate 10 practice test questions based on the following notes. The question should cover the main topics. Ensure the question is of difficulty difficulty and is question_type.

        Notes:
        {notes}

        Question Format: Multiple Choice
        Question: [Your question here]
        Options: [Option 1, Option 2, Option 3, Option 4]
        Answer: [Correct answer]
        '''
        response = await sydney.ask(question, citations=True)
    return response

@app.route('/generatepracticetest', methods=['GET'])
async def generatepracticetest():
    response = await practtest()
    return jsonify({'questions': response})


# @app.route('/recommend_sections', methods=['GET'])
# def recommend_sections():
#     username = request.args.get('username')
#     if not username:
#         return jsonify({'error': 'Username not provided'}), 400

#     # Retrieve the user's recent sections
#     user_sections = db.collection('collections').where('username', '==', username).stream()

#     recent_sections = []
#     for doc in user_sections:
#         collection_id = doc.id
#         section_docs = db.collection('collections').document(collection_id).collection('sections').order_by('last_accessed', direction=firestore.Query.DESCENDING).limit(5).stream()

#         for section_doc in section_docs:
#             doc_data = section_doc.to_dict()
#             recent_sections.append(section_doc.id)

#     # Find users with similar recent sections
#     similar_users = db.collection('collections').where('sections', 'array_contains_any', recent_sections).stream()

#     recommended_sections = []
#     for doc in similar_users:
#         if doc.to_dict()['username'] != username:  # Exclude current user
#             collection_id = doc.id
#             section_docs = db.collection('collections').document(collection_id).collection('sections').where('visibility', '==', 'public').stream()

#             for section_doc in section_docs:
#                 section_data = section_doc.to_dict()
#                 title = section_data.get('section_name', '')
#                 if section_doc.id not in recent_sections:  # Exclude sections user already has
#                     recommended_sections.append({'id': section_doc.id, 'title': title})

#     return jsonify({'recommended_sections': recommended_sections})

# app.route('/clone_section', methods=['POST'])
# def clone_section():
#     data = request.get_json()
#     if not data:
#         return jsonify({'error': 'No data provided'}), 400

#     section_id = data.get('sectionId')
#     add_to_existing = data.get('addToExisting', False)
#     collection_id = data.get('collectionId')  # If add_to_existing is True
#     collection_name = data.get('collectionName')  # If add_to_existing is False

#     if not section_id:
#         return jsonify({'error': 'Section ID not provided'}), 400

#     if not add_to_existing and not collection_name:
#         return jsonify({'error': 'Collection name not provided for creating a new collection'}), 400

#     try:
#         # Get the section data from the database based on the provided section_id
#         section_ref = db.collection('sections').document(section_id)
#         section_data = section_ref.get().to_dict()

#         if not section_data:
#             return jsonify({'error': 'Section not found'}), 404

#         if add_to_existing:
#             if not collection_id:
#                 return jsonify({'error': 'Collection ID not provided for adding to an existing collection'}), 400

#             # Add the cloned section to the specified existing collection
#             collection_ref = db.collection('collections').document(collection_id)
#             new_section_ref = collection_ref.collection('sections').document()
#             new_section_ref.set(section_data)
#             new_section_ref.headers.add('Access-Control-Allow-Origin', '*')  # Adjust the origin as needed

#             return jsonify({'message': 'Section cloned and added to the existing collection successfully', 'newSectionId': new_section_ref.id}), 500
#         else:
#             # Create a new collection and add the cloned section to it
#             new_collection_ref = db.collection('collections').document()
#             new_collection_name = collection_name.strip()
#             new_section_ref = new_collection_ref.collection('sections').document()
#             new_section_ref.set(section_data)
#             new_collection_ref.set({
#                 'name': new_collection_name,
#                 'sections': [new_section_ref.id]
#             })
#             new_collection_ref.headers.add('Access-Control-Allow-Origin', '*')  # Adjust the origin as needed

#             return jsonify({'message': 'Section cloned and added to a new collection successfully', 'newCollectionId': new_collection_ref.id, 'newSectionId': new_section_ref.id}), 500
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
# # Load pre-trained word embeddings model (you need to train or download one)
# word_embeddings_model = Word2Vec.load("word2vec.model")

# @app.route('/recommend_public_sections', methods=['GET'])
# def recommend_public_sections():
#     username = request.args.get('username')

#     if not username:
#         return jsonify({'error': 'Username not provided'}), 400

#     user_sections = []
#     public_sections = []

#     # Get user's sections
#     user_section_docs = db.collection('collections').where('username', '==', username).stream()
#     for doc in user_section_docs:
#         collection_id = doc.id
#         section_docs = db.collection('collections').document(collection_id).collection('sections').stream()
        
#         for section_doc in section_docs:
#             doc_data = section_doc.to_dict()
#             title = doc_data.get('section_name', '')
#             user_sections.append(title)

#     # Get public sections
#     section_docs = db.collection('collections').stream()
#     for doc in section_docs:
#         section_ref = doc.reference.collection('sections').where('visibility', '==', 'public').stream()
#         for section_doc in section_ref:
#             section_data = section_doc.to_dict()
#             title = section_data.get('section_name', '')
#             # Calculate semantic similarity using word embeddings
#             similarity_scores = [word_embeddings_model.wv.similarity(title.lower(), user_section.lower()) for user_section in user_sections]
#             average_similarity = sum(similarity_scores) / len(similarity_scores)
#             if average_similarity > 0.5:  # You can adjust this threshold based on your requirements
#                 public_sections.append({'id': section_doc.id, 'title': title, 'similarity': average_similarity})

#     # Sort recommended sections by similarity score
#     public_sections.sort(key=lambda x: x['similarity'], reverse=True)

#     return jsonify({'recommended_sections': public_sections})


if __name__ == '__main__':
    # app.run(host='localhost', port=5000, debug=True)
    app.run(debug=True, port=5000)