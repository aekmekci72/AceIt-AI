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

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "key"

# List of LLM models
list_llm = ["mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mixtral-8x7B-Instruct-v0.1", "mistralai/Mistral-7B-Instruct-v0.1", 
    "google/gemma-7b-it", "google/gemma-2b-it", 
    "HuggingFaceH4/zephyr-7b-beta", "HuggingFaceH4/zephyr-7b-gemma-v0.1", 
    "meta-llama/Llama-2-7b-chat-hf", "microsoft/phi-2", 
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "mosaicml/mpt-7b-instruct", "tiiuae/falcon-7b-instruct", 
    "google/flan-t5-xxl"]

# Default index for the LLM model
default_llm_index = 5

# Load raw text and create document splits
def load_doc(text, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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

# Initialize LLM chain
def initialize_llmchain(llm_model, temperature, max_tokens, top_k, vector_db):
    if llm_model == "mistralai/Mixtral-8x7B-Instruct-v0.1":
        llm = HuggingFaceEndpoint(repo_id=llm_model, temperature=temperature, max_new_tokens=max_tokens, top_k=top_k, load_in_8bit=True)
    elif llm_model in ["HuggingFaceH4/zephyr-7b-gemma-v0.1", "mosaicml/mpt-7b-instruct"]:
        raise Exception("LLM model is too large to be loaded automatically on free inference endpoint")
    elif llm_model == "microsoft/phi-2":
        llm = HuggingFaceEndpoint(repo_id=llm_model, temperature=temperature, max_new_tokens=max_tokens, top_k=top_k, trust_remote_code=True, torch_dtype="auto")
    elif llm_model == "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
        llm = HuggingFaceEndpoint(repo_id=llm_model, temperature=temperature, max_new_tokens=250, top_k=top_k)
    elif llm_model == "meta-llama/Llama-2-7b-chat-hf":
        raise Exception("Llama-2-7b-chat-hf model requires a Pro subscription")
    else:
        llm = HuggingFaceEndpoint(repo_id=llm_model, temperature=temperature, max_new_tokens=max_tokens, top_k=top_k)
    
    memory = ConversationBufferMemory(memory_key="chat_history", output_key='answer', return_messages=True)
    retriever = vector_db.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, chain_type="stuff", memory=memory, return_source_documents=True, verbose=False)
    
    return qa_chain

# Initialize database
def initialize_database(text, chunk_size=500, chunk_overlap=30):
    collection_name = create_collection_name(text)
    doc_splits = load_doc(text, chunk_size, chunk_overlap)
    vector_db = create_db(doc_splits, collection_name)
    return vector_db, collection_name

# Format chat history for display
def format_chat_history(chat_history):
    formatted_chat_history = []
    for user_message, bot_message in chat_history:
        formatted_chat_history.append(f"User: {user_message}")
        formatted_chat_history.append(f"Assistant: {bot_message}")
    return formatted_chat_history

# Chat function
def chat(qa_chain, message, chat_history):
    formatted_chat_history = format_chat_history(chat_history)
    response = qa_chain({"question": message, "chat_history": formatted_chat_history})
    response_answer = response["answer"]
    if "Helpful Answer:" in response_answer:
        response_answer = response_answer.split("Helpful Answer:")[-1]
    return response_answer

# Main method to initialize and chat
def main(text):
    vector_db, collection_name = initialize_database(text)
    llm_model = list_llm[default_llm_index]
    temperature = 0.3
    max_tokens = 512
    top_k = 20
    qa_chain = initialize_llmchain(llm_model, temperature, max_tokens, top_k, vector_db)

    chat_history = []
    print("Chatbot initialized. Start chatting (type 'exit' to quit):")
    while True:
        message = input("User: ")
        if message.lower() == "exit":
            break
        response = chat(qa_chain, message, chat_history)
        print(f"Assistant: {response}")
        chat_history.append((message, response))

if __name__ == "__main__":
    # Replace with your text input
    text_input = """
Anna Ekmekci
Mechatronics Research Lab at Bergen County Academies
The Use of Data Analysis Predicting Disease Development (Health Buddy)

I am incredibly grateful for any amount of data able to be provided in aiding the development of Health Buddy and furthering this research progress. Below are descriptions of what data would be useful, what type of model it would be used for, the general research plan and rationale for developing this software, and what has already been done to develop it. 

Data Necessary	1
Machine Learning Model	1
Research Plan	2
Current Project Status	6
Data Necessary
The most useful pieces of data that I need to improve training for my machine learning model for Health Buddy include:
Orphan disease
Age
Sex
Height
Weight
Symptoms
Risk factors
Family history
Other medical history/diagnoses

Any format to receive this data would be appreciated, including but not limited to CSV, JSON, XML, and Excel. 

I would like to express my immense gratitude once again for any data able to be provided to me and for helping my research project move forward. 
Machine Learning Model
The data required will be utilized to help train my artificial neural network (ANN), to explain specifically what it is required for. This ANN will be utilized to predict the likelihood of one developing certain orphan diseases and involves several key design decisions and considerations. The input features required include age, sex (encoded as categorical variables), height (normalized or standardized), weight (normalized or standardized), symptoms (encoded as binary features indicating presence/absence), risk factors (encoded as binary features), family history (encoded as binary features), and other medical history and diagnoses (encoded as binary features). 

The architecture for this model will go roughly as follows:
An input layer with its number of neurons corresponding to input features
Hidden layers to capture complex interactions between input features, allowing the network to learn hierarchical representations and nonlinear relationships in the data
An output layer for multi-class classification, providing the predicted probability of developing the diseases

Activation functions will include linear activation, Rectified Linear Unit, hyperbolic tangent, and sigmoid activation functions. The loss function will likely be binary cross-entropy, and the optimizer will be an Adam optimizer. 

The ANN will be trained based on a percentage of the patient data (randomly selected for each orphan disease) and fine-tuned with back-propagation. 
Research Plan
Rationale

Orphan diseases pose significant diagnostic challenges due to their rarity and the limited familiarity among healthcare professionals. Patients often endure years of uncertainty, visiting multiple doctors and thus undergoing ineffective treatments. This delay in diagnosis not only exacerbates suffering but also increases the risk of misdiagnosis, further complicating proper care ("Rare Diseases Difficult to Diagnose, Cures Hard to Come By").

With nearly 7,000 rare diseases affecting 30 million Americans, the breadth of required knowledge is extensive, yet many conditions lack established diagnostic criteria and efficient testing protocols ("Rare and Orphan Diseases"). Specialists and extensive testing are often necessary, prolonging the diagnostic process and delaying treatment initiation.

Diagnosing orphan diseases is resource-intensive, involving high costs for research, testing, and treatment. Limited financial incentives for pharmaceutical companies also hinder research and development, leaving many conditions underfunded and under-researched ("Orphan Disease Center"). These challenges highlight the urgent need for improved medical education, streamlined diagnostics, and increased research funding to enhance patient outcomes and quality of life.

Addressing these issues requires a concerted effort to refine diagnostic protocols for rare disease research. By tackling these challenges, the medical community can significantly improve the diagnosis and treatment of orphan diseases, ultimately enhancing patient outcomes and quality of life.

Goal and Expected Outcomes

Goal: 

In this project, I am to develop Health Buddy, a software application designed to expedite the diagnosis and prediction of orphan diseases. Health Buddy will utilize advanced data analysis and machine learning techniques including artificial neural networks to predict an individual’s susceptibility to various rare diseases based on their current medical data. The software will provide users with a personalized risk assessment, indicating their match to risk factors for these diseases. Additionally, Health Buddy will offer tailored recommendations and strategies to mitigate these risks, empowering individuals to adopt proactive measures toward healthier lifestyles. By focusing on orphan diseases, which are often under-researched and less understood by the general public, this project aims to increase awareness and early detection of these conditions, potentially leading to improved health outcomes and quality of life for affected individuals. 

Expected Outcomes: 

From this software, I can expect a platform for comprehensive health analysis that will be enhanced through AI and machine learning. This will enable more precise predictions of health risks associated with orphan diseases, offering personalized medical advice to mitigate these risks effectively. By leveraging advanced technologies, the software aims to reduce the likelihood of health complications related to rare diseases. 

Procedure

Outline the wireframes of the UI of the software, along with the database schematic. List the different APIs and technologies that can be utilized to enhance the web app’s functionality.
These sketches clarify what kinds of endpoints may be necessary to include in the server along with algorithms used for each step of the prediction process.
Write the code for the input of one’s medical profile on the user side, along with starting the development of the server to dynamically update the software and database.
There are several pieces of data that the program will take in, with necessary pages that the user will need to access. These include, based on wireframes and important features of Health Buddy: 
General information (age, sex, height, weight)
Family History (disease, family member)
Medical History (disease, year diagnosed)
Symptoms (symptom, date/time period)
Lifestyle Risk Factors (risk factor)
Implement the database connection to the server-side ensuring the storage of user information
Use MySQL, a relational database suitable for tasks necessary for this research project
Implement all of the tables needed with some sample testing data scraped currently from NORD and Orphanet pages
In the future, find data from medical centers with specialties and more patient specifics for better prediction of potential diseases
Create sample users to test all future algorithms to be implemented
Develop a sophisticated algorithm with more data collected with patient-specific information to train a machine-learning model
Use Keras and other Python libraries to train an artificial neural network (categorization model) to provide insights into diseases a user may be likely to develop in the future
Collect larger amounts of relevant data to be put into the database to make the algorithm even more effective 
This will give the software a larger scope of diseases it can predict the development of.
This data can be collected with web scraping methods, along with specific information from medical centers, complying to HIPAA regulations
Test various scenarios to see if the algorithm analyzes data in the expected and desired way with patient data collected
Make any changes if necessary
Continue to modify and improve different parts of Health Buddy to maximize effectiveness and efficiency

Data Analysis

To analyze Health Buddy’s effectiveness, frequent testing of the software will be done in order to ensure that it is working properly. Listed below are the tests that I will be running to ensure that it is working in the desired manner:
To test the algorithm of Health Buddy, different pieces of data will regularly be given to it to see how predicted outcomes change. This will include experimenting with the consistency of some data and modifying other data put in by the user. A few variables that will be viewed: 
The same basic information (age, sex, height, weight, medical and family history) but different symptoms experienced at different points in time
Different user information but the same symptoms
Different family histories
Different medical histories
Time series analysis (to view how trends and patterns in symptom development and progression can affect the predictions)
Consistent unit testing, ensuring different components of the software work, including testing data input processing, database interactions, and more, which should also produce the expected output
Error handling and recovery testing, making sure there are lots of catches to improve how the software can handle errors and exceptions, improving the user experience
Load testing, to see if the system can maintain functionality and performance to assess whether Health Buddy can handle more users and data without degrading performance

Bibliography

Ameisen, Emmanuel. Building Machine Learning Powered Applications: Going from
Idea to Product. O'Reilly Media, 2020.

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and
TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems.
O'Reilly Media, 2019.

Hernberg-Ståhl, Elizabeth, and Michelle L. Rose. Orphan Drugs: A Global Development
and Policy Perspective. CRC Press, 2013.

Holzinger, Andreas. Machine Learning for Health Informatics: State-of-the-Art and
Future Challenges. Springer, 2016.

Jiang, Fei, et al. "Artificial Intelligence in Healthcare: Past, Present and Future." Stroke
and Vascular Neurology, vol. 2, no. 4, 2017, pp. 230-243.

National Academies of Sciences, Engineering, and Medicine. Improving Diagnosis in
Health Care. The National Academies Press, 2015.

Nguengang Wakap, Stéphanie, et al. "Estimating Cumulative Point Prevalence of Rare
Diseases: Analysis of the Orphanet Database." European Journal of Human
Genetics, vol. 28, no. 2, 2020, pp. 165-173.

“Orphan Disease Center.” Orphan Disease Center,
www.orphandiseasecenter.med.upenn.edu/. Accessed 23 June 2024.

“Rare and Orphan Diseases.” Www.ncsl.org,
www.ncsl.org/health/rare-and-orphan-diseases.

“Rare Diseases Difficult to Diagnose, Cures Hard to Come By.” AAMC,
www.aamc.org/news/rare-diseases-difficult-diagnose-cures-hard-come.

Schieppati, Arrigo, et al. "Why Rare Diseases Are an Important Medical and Social
Issue." The Lancet, vol. 371, no. 9629, 2008, pp. 2039-2041.

Current Project Status
Health Buddy is currently in the development process and as of now, has been worked on for the past 16 months as a research project. It is a functional full-stack web application and currently has data scraped from NORD and Orphanet collections. Although it has a prediction algorithm based on a weighting metric, the progress has plateaued without a way to test the accuracy of Health Buddy. With data focused on specific individuals, not only would there be a metric to test whatever model or algorithm is used, but a more sophisticated artificial neural network may be developed for a higher accuracy of prediction personalized to users. Further steps that are being taken to improve the usefulness of Health Buddy include integration with wearables, and both an iOS and web app version will be available upon deployment. 

    """
    main(text_input)
