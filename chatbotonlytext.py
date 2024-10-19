import gradio as gr
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceEndpoint

import chromadb
from unidecode import unidecode

from transformers import AutoTokenizer
import transformers
import torch
import tqdm 
import accelerate
import re

import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "key"

list_llm = ["mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mixtral-8x7B-Instruct-v0.1", "mistralai/Mistral-7B-Instruct-v0.1", \
    "google/gemma-7b-it","google/gemma-2b-it", \
    "HuggingFaceH4/zephyr-7b-beta", "HuggingFaceH4/zephyr-7b-gemma-v0.1", \
    "meta-llama/Llama-2-7b-chat-hf", "microsoft/phi-2", \
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "mosaicml/mpt-7b-instruct", "tiiuae/falcon-7b-instruct", \
    "google/flan-t5-xxl"
]
list_llm_simple = [os.path.basename(llm) for llm in list_llm]

# Load raw text and create doc splits
def load_doc(text, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size, 
        chunk_overlap = chunk_overlap)
    doc_splits = text_splitter.create_documents([text])
    return doc_splits

# Create vector database
def create_db(splits, collection_name):
    embedding = HuggingFaceEmbeddings()
    new_client = chromadb.EphemeralClient()
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        client=new_client,
        collection_name=collection_name,
    )
    return vectordb

# Load vector database
def load_db():
    embedding = HuggingFaceEmbeddings()
    vectordb = Chroma(
        embedding_function=embedding)
    return vectordb

# Initialize langchain LLM chain
def initialize_llmchain(llm_model, temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
    progress(0.1, desc="Initializing HF tokenizer...")
    progress(0.5, desc="Initializing HF Hub...")
    if llm_model == "mistralai/Mixtral-8x7B-Instruct-v0.1":
        llm = HuggingFaceEndpoint(
            repo_id=llm_model, 
            temperature = temperature,
            max_new_tokens = max_tokens,
            top_k = top_k,
            load_in_8bit = True,
        )
    elif llm_model in ["HuggingFaceH4/zephyr-7b-gemma-v0.1","mosaicml/mpt-7b-instruct"]:
        raise gr.Error("LLM model is too large to be loaded automatically on free inference endpoint")
        llm = HuggingFaceEndpoint(
            repo_id=llm_model, 
            temperature = temperature,
            max_new_tokens = max_tokens,
            top_k = top_k,
        )
    elif llm_model == "microsoft/phi-2":
        llm = HuggingFaceEndpoint(
            repo_id=llm_model, 
            temperature = temperature,
            max_new_tokens = max_tokens,
            top_k = top_k,
            trust_remote_code = True,
            torch_dtype = "auto",
        )
    elif llm_model == "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
        llm = HuggingFaceEndpoint(
            repo_id=llm_model, 
            temperature = temperature,
            max_new_tokens = 250,
            top_k = top_k,
        )
    elif llm_model == "meta-llama/Llama-2-7b-chat-hf":
        raise gr.Error("Llama-2-7b-chat-hf model requires a Pro subscription...")
        llm = HuggingFaceEndpoint(
            repo_id=llm_model, 
            temperature = temperature,
            max_new_tokens = max_tokens,
            top_k = top_k,
        )
    else:
        llm = HuggingFaceEndpoint(
            repo_id=llm_model, 
            temperature = temperature,
            max_new_tokens = max_tokens,
            top_k = top_k,
        )
    
    progress(0.75, desc="Defining buffer memory...")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='answer',
        return_messages=True
    )
    retriever = vector_db.as_retriever()
    progress(0.8, desc="Defining retrieval chain...")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        chain_type="stuff", 
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    progress(0.9, desc="Done!")
    return qa_chain

# Generate collection name for vector database
def create_collection_name(text):
    collection_name = unidecode(text[:50]).replace(" ","-")
    collection_name = re.sub('[^A-Za-z0-9]+', '-', collection_name)
    if len(collection_name) < 3:
        collection_name = collection_name + 'xyz'
    if not collection_name[0].isalnum():
        collection_name = 'A' + collection_name[1:]
    if not collection_name[-1].isalnum():
        collection_name = collection_name[:-1] + 'Z'
    print('Collection name: ', collection_name)
    return collection_name

# Initialize database
def initialize_database(text, chunk_size, chunk_overlap, progress=gr.Progress()):
    progress(0.1, desc="Creating collection name...")
    collection_name = create_collection_name(text)
    progress(0.25, desc="Loading document...")
    doc_splits = load_doc(text, chunk_size, chunk_overlap)
    progress(0.5, desc="Generating vector database...")
    vector_db = create_db(doc_splits, collection_name)
    progress(0.9, desc="Done!")
    return vector_db, collection_name, "Complete!"

def initialize_LLM(llm_option, llm_temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
    llm_name = llm_option
    print("llm_name: ",llm_name)
    qa_chain = initialize_llmchain(llm_name, llm_temperature, max_tokens, top_k, vector_db, progress)
    return qa_chain, "Complete!"

def format_chat_history(message, chat_history):
    formatted_chat_history = []
    for user_message, bot_message in chat_history:
        formatted_chat_history.append(f"User: {user_message}")
        formatted_chat_history.append(f"Assistant: {bot_message}")
    return formatted_chat_history

def conversation(qa_chain, message, history):
    formatted_chat_history = format_chat_history(message, history)
    response = qa_chain({"question": message, "chat_history": formatted_chat_history})
    response_answer = response["answer"]
    if response_answer.find("Helpful Answer:") != -1:
        response_answer = response_answer.split("Helpful Answer:")[-1]
    response_sources = response["source_documents"]
    
    # Initialize variables to handle missing keys
    response_source1 = response_source1_page = response_source2 = response_source2_page = response_source3 = response_source3_page = "Not available"

    if len(response_sources) > 0:
        response_source1 = response_sources[0].page_content.strip()
        response_source1_page = response_sources[0].metadata.get("page", "Not available") + "1"
    if len(response_sources) > 1:
        response_source2 = response_sources[1].page_content.strip()
        response_source2_page = response_sources[1].metadata.get("page", "Not available") + "1"
    if len(response_sources) > 2:
        response_source3 = response_sources[2].page_content.strip()
        response_source3_page = response_sources[2].metadata.get("page", "Not available") +"1"
    
    new_history = history + [(message, response_answer)]
    return qa_chain, gr.update(value=""), new_history, response_source1, response_source1_page, response_source2, response_source2_page, response_source3, response_source3_page

def demo():
    with gr.Blocks(theme="base") as demo:
        vector_db = gr.State()
        qa_chain = gr.State()
        collection_name = gr.State()
        
        gr.Markdown(
        """<center><h2>Text-based chatbot</center></h2>
        <h3>Ask any questions about your text documents</h3>""")
        gr.Markdown(
        """<b>Note:</b> This AI assistant, using Langchain and open-source LLMs, performs retrieval-augmented generation (RAG) from your text documents. \
        The user interface explicitely shows multiple steps to help understand the RAG workflow. 
        This chatbot takes past questions into account when generating answers (via conversational memory), and includes document references for clarity purposes.<br>
        <br><b>Warning:</b> This space uses the free CPU Basic hardware from Hugging Face. Some steps and LLM models used below (free inference endpoints) can be very slow. \
        For faster processing and to access more powerful models, you can duplicate this space and upgrade the hardware in your version.
        """)
        with gr.Tab("1. Upload document and prepare vector database"):
            with gr.Row():
                with gr.Column(scale=4):
                    text_input = gr.Textbox(
                        label="Enter text",
                        placeholder="Enter your text here...",
                        lines=10,
                    )
                with gr.Column(scale=1):
                    chunk_size = gr.Slider(
                        label="Chunk size", value=500, minimum=200, maximum=1500, step=100)
                    chunk_overlap = gr.Slider(
                        label="Chunk overlap", value=30, minimum=0, maximum=200, step=10)
                    start_button = gr.Button(value="Start", variant="primary")
                with gr.Column(scale=1):
                    collection_name_output = gr.Textbox(
                        label="Collection name", placeholder="collection_name", interactive=False, lines=1, max_lines=1)
                    db_output = gr.Textbox(label="Status", placeholder="empty", interactive=False, lines=1, max_lines=1)

        with gr.Tab("2. Select LLM and create chain"):
            with gr.Row():
                with gr.Column(scale=4):
                    llm_model = gr.Dropdown(
                        choices=list_llm, label="LLM model", value="mistralai/Mistral-7B-Instruct-v0.2")
                with gr.Column(scale=1):
                    temperature = gr.Slider(
                        label="Temperature", value=0.3, minimum=0.0, maximum=1.2, step=0.1)
                    max_tokens = gr.Slider(
                        label="Max new tokens", value=512, minimum=128, maximum=2048, step=16)
                    top_k = gr.Slider(
                        label="Top_k", value=20, minimum=1, maximum=50, step=1)
                    create_chain = gr.Button(value="Create chain", variant="primary")
                with gr.Column(scale=1):
                    chain_output = gr.Textbox(
                        label="Status", placeholder="empty", interactive=False, lines=1, max_lines=1)

        with gr.Tab("3. Conversational retrieval QA"):
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(label="Conversational chat with LLM")
                    msg = gr.Textbox(label="Question", placeholder="Ask a question and hit ENTER")
                with gr.Column(scale=2):
                    response_source1 = gr.Textbox(
                        label="Reference 1", placeholder="empty", interactive=False, lines=10, max_lines=10)
                    response_source1_page = gr.Number(
                        label="Page", value=0, interactive=False)
                    response_source2 = gr.Textbox(
                        label="Reference 2", placeholder="empty", interactive=False, lines=10, max_lines=10)
                    response_source2_page = gr.Number(
                        label="Page", value=0, interactive=False)
                    response_source3 = gr.Textbox(
                        label="Reference 3", placeholder="empty", interactive=False, lines=10, max_lines=10)
                    response_source3_page = gr.Number(
                        label="Page", value=0, interactive=False)
            with gr.Row():
                clear = gr.Button(value="Clear conversation")

        start_button.click(initialize_database, inputs=[
            text_input, chunk_size, chunk_overlap], outputs=[vector_db, collection_name, db_output])
        start_button.click(lambda collection_name: collection_name, inputs=collection_name, outputs=collection_name_output)
        create_chain.click(initialize_LLM, inputs=[
            llm_model, temperature, max_tokens, top_k, vector_db], outputs=[qa_chain, chain_output])
        msg.submit(conversation, [qa_chain, msg, chatbot], [qa_chain, msg, chatbot, response_source1, response_source1_page, response_source2, response_source2_page, response_source3, response_source3_page])
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue()
    demo.launch()

demo()
