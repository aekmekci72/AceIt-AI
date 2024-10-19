import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub
import streamlit as st

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize HuggingFace Hub API token
huggingfacehub_api_token = os.environ["HUGGINGFACEHUB_API_TOKEN"]

@st.cache_resource
def load_pdf(file):
    loader = PyPDFLoader(file)
    pages = loader.load()
    return pages

@st.cache_resource
def split_text(pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(pages)
    return splits

@st.cache_resource
def create_embeddings():
    embeddings = HuggingFaceEmbeddings()
    return embeddings

@st.cache_resource
def create_vector_store(splits, embeddings):
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

@st.cache_resource
def create_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="cvachet/pdf-chatbot", model_kwargs={"temperature": 0.5, "max_length": 512})
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    
    return conversation_chain

def main():
    st.set_page_config(page_title="PDF Chatbot")
    st.header("Chat with your PDF")

    # File uploader
    pdf_file = st.file_uploader("Upload your PDF", type="pdf")

    if pdf_file is not None:
        # Process the PDF
        with st.spinner("Processing PDF..."):
            pages = load_pdf(pdf_file)
            splits = split_text(pages)
            embeddings = create_embeddings()
            vectorstore = create_vector_store(splits, embeddings)
            conversation_chain = create_conversation_chain(vectorstore)

        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question about your PDF"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = conversation_chain({"question": prompt})
                    st.markdown(response['answer'])
                    st.session_state.messages.append({"role": "assistant", "content": response['answer']})

        # Display RAG workflow steps
        with st.expander("RAG Workflow Steps"):
            st.write("1. PDF Loading")
            st.write("2. Text Splitting")
            st.write("3. Embedding Creation")
            st.write("4. Vector Store Creation")
            st.write("5. Question Answering")

if __name__ == "__main__":
    main()