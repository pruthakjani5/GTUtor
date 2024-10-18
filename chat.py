import streamlit as st
import requests
from pypdf import PdfReader
import os
import google.generativeai as genai
import chromadb
from typing import List, Dict
from dotenv import load_dotenv
import tempfile
import json
import uuid
import clipboard
import markdown

# Load environment variables
load_dotenv()

# Set up Gemini API
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("Gemini API Key not provided or incorrect. Please provide a valid GEMINI_API_KEY in .env file.")
genai.configure(api_key=gemini_api_key)

# Create a permanent directory for the databases and chat histories
#data_folder = os.path.join(os.path.expanduser("~"), "gtutor_data")
data_folder = os.path.join(os.getcwd(), "gtutor_data")
db_folder = os.path.join(data_folder, "dbs")
history_folder = os.path.join(data_folder, "chat_histories")
os.makedirs(db_folder, exist_ok=True)
os.makedirs(history_folder, exist_ok=True)

# File to store subject names
subjects_file = os.path.join(data_folder, "subjects.json")

# Load existing subjects
def load_subjects():
    if os.path.exists(subjects_file):
        with open(subjects_file, 'r') as f:
            return json.load(f)
    return []

# Save subjects
def save_subjects(subjects):
    with open(subjects_file, 'w') as f:
        json.dump(subjects, f)

# Load existing subjects
subjects = load_subjects()

# Initialize databases for existing subjects
dbs: Dict[str, chromadb.Collection] = {}

# Function to create or get a database for a subject
def get_or_create_db(subject):
    if subject not in dbs:
        subject_db_path = os.path.join(db_folder, subject.lower().replace(" ", "_"))
        os.makedirs(subject_db_path, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=subject_db_path)
        try:
            dbs[subject] = chroma_client.get_collection(name=subject)
        except ValueError:
            dbs[subject] = chroma_client.create_collection(name=subject)
    return dbs[subject]

# Function to load chat history for a subject
def load_chat_history(subject):
    history_file = os.path.join(history_folder, f"{subject.lower().replace(' ', '_')}_history.json")
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            return json.load(f)
    return []

# Function to save chat history for a subject
def save_chat_history(subject, history):
    history_file = os.path.join(history_folder, f"{subject.lower().replace(' ', '_')}_history.json")
    with open(history_file, 'w') as f:
        json.dump(history, f)

# Function to download PDF from URL
def download_pdf(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        st.error(f"Failed to download PDF from {url}. Error: {str(e)}")
        return None

# Function to extract text from PDF in chunks
def extract_text_from_pdf(pdf_content, chunk_size=1000):
    pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf_file.write(pdf_content)
    pdf_file.close()

    reader = PdfReader(pdf_file.name)
    total_pages = len(reader.pages)
    
    for page_num in range(total_pages):
        page = reader.pages[page_num]
        text = page.extract_text()
        
        # Split text into chunks
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        for chunk in chunks:
            yield chunk, page_num

    os.unlink(pdf_file.name)

# Function to add document to the database
def add_document_to_db(pdf_content, source, subject):
    db = get_or_create_db(subject)
    chunk_generator = extract_text_from_pdf(pdf_content)
    
    for i, (chunk, page_num) in enumerate(chunk_generator):
        unique_id = f"{source}_page{page_num}_chunk{i}"
        db.add(
            documents=[chunk],
            metadatas=[{"source": source, "page": page_num}],
            ids=[unique_id]
        )
    st.success(f"Successfully added {source} to the {subject} database.")

# Function to get relevant passages
def get_relevant_passage(query: str, subject: str, n_results: int = 5):
    # db = get_or_create_db(subject)
    # results = db.query(query_texts=[query], n_results=n_results)
    # return results['documents'][0]
    db = get_or_create_db(subject)
    results = db.query(query_texts=[query], n_results=n_results)
    return results['documents'][0] if results['documents'][0] else []

# Function to construct the RAG prompt
def make_rag_prompt(query: str, relevant_passages: List[str], subject: str, chat_history: List[Dict]):
    escaped_passages = [p.replace("'", "").replace('"', "").replace("\n", " ") for p in relevant_passages]
    passages_text = "\n".join(f"PASSAGE {i+1}: {p}" for i, p in enumerate(escaped_passages))
    
    history_text = "\n".join([f"Human: {turn['human']}\nAssistant: {turn['ai']}" for turn in chat_history[-5:]])
    
    prompt = f"""You are GTUtor, a helpful and informative AI assistant specializing in {subject} for GTU (Gujarat Technological University) students.
Answer the question using the reference passages below, your knowledge of {subject}, and the chat history provided and in a detailed and well-structured manner. Include all relevant information and specify the page numbers, line numbers, and PDF names where the information is found. If the answer requires additional knowledge beyond the provided context, clearly state this limitation and provide relevant information or insights using your knowledge. Do not provide incorrect information.
* Maintain a formal and academic tone throughout your response which is also simple to understand and informative. Answer as per required depth and weightage to the topic in subject.
If the information is not in the passages, state that and then use your own knowledge to answer.

Chat History:
{history_text}

Reference Passages:
{passages_text}

QUESTION: '{query}'

ANSWER:
"""
    return prompt

    # Initialize session state for the query input
    if 'query' not in st.session_state:
        st.session_state.query = ""

    # Subject selection or creation
    previous_subject = st.session_state.get('previous_subject', '')
    subject_option = st.selectbox("Select a subject or create a new one", [""] + subjects + ["Create New Subject"])

    if subject_option != previous_subject:
        st.session_state.query = ""  # Clear the query when subject changes
        st.session_state.previous_subject = subject_option

    if subject_option == "Create New Subject":
        new_subject = st.text_input("Enter the name of the new subject")
        if new_subject and new_subject not in subjects:
            subjects.append(new_subject)
            save_subjects(subjects)
            st.success(f"New subject '{new_subject}' created successfully!")
            subject_option = new_subject

    selected_subject = subject_option if subject_option != "Create New Subject" else new_subject
# Function to generate an answer using Gemini Pro API
@st.cache_data
def generate_answer(prompt: str):
    try:
        model = genai.GenerativeModel('gemini-pro')
        result = model.generate_content(prompt)
        return result.text
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return None

# Function to generate an answer using LLM's knowledge
def generate_llm_answer(query: str, subject: str = None, chat_history: List[Dict] = None):
    history_text = "\n".join([f"Human: {turn['human']}\nAssistant: {turn['ai']}" for turn in (chat_history or [])[-5:]])
    
    if subject:
        prompt = f"""You are GTUtor, a helpful and informative AI assistant specializing in {subject} for GTU (Gujarat Technological University) students. 
You have in-depth knowledge about GTU's curriculum and courses related to {subject}.
Please provide a comprehensive and informative answer to the following question, using your specialized knowledge and considering the chat history:

Chat History:
{history_text}

QUESTION: {query}

ANSWER:
"""
    else:
        prompt = f"""You are GTUtor, a helpful and informative AI assistant for GTU (Gujarat Technological University) students. 
You have general knowledge about GTU's curriculum and various courses.
Please provide a comprehensive and informative answer to the following question, using your knowledge and considering the chat history:

Chat History:
{history_text}

QUESTION: {query}

ANSWER:
"""
    return generate_answer(prompt)

# Streamlit UI

# # Initialize session state for chat history
if 'chat_histories' not in st.session_state:
    st.session_state.chat_histories = {}

st.markdown("""
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
.bot .message p {
    margin-bottom: 0.5rem;
}
.bot .message ul, .bot .message ol {
    margin-left: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

st.title("GTUtor: Dynamic Multi-Subject Chat System")

# Subject selection or creation
subject_option = st.selectbox("Select a subject or create a new one", [""] + subjects + ["Create New Subject"])

if subject_option == "Create New Subject":
    new_subject = st.text_input("Enter the name of the new subject")
    if new_subject and new_subject not in subjects:
        subjects.append(new_subject)
        save_subjects(subjects)
        st.success(f"New subject '{new_subject}' created successfully!")
        subject_option = new_subject

selected_subject = subject_option if subject_option != "Create New Subject" else new_subject

# Load chat history for the selected subject
if selected_subject and selected_subject not in st.session_state.chat_histories:
    st.session_state.chat_histories[selected_subject] = load_chat_history(selected_subject)

# File upload and URL input for the selected subject
if selected_subject:
    st.subheader(f"Add Documents to {selected_subject}")
    uploaded_file = st.file_uploader(f"Choose a PDF file for {selected_subject} (max 10MB)", type="pdf")
    pdf_url = st.text_input(f"Or enter a PDF URL for {selected_subject}")

    if uploaded_file is not None:
        # if uploaded_file.size > 10 * 1024 * 1024:  # 10MB limit
        #     st.error("File size exceeds 10MB limit. Please upload a smaller file.")
        # else:
        pdf_content = uploaded_file.read()
        with st.spinner("Processing PDF..."):
            add_document_to_db(pdf_content, uploaded_file.name, selected_subject)

    elif pdf_url:
        with st.spinner("Downloading PDF..."):
            pdf_content = download_pdf(pdf_url)
        if pdf_content:
            with st.spinner("Processing PDF..."):
                add_document_to_db(pdf_content, pdf_url, selected_subject)

# Display chat history with enhanced UI
if selected_subject:
    st.subheader(f"Chat History - {selected_subject}")
    for i, turn in enumerate(st.session_state.chat_histories.get(selected_subject, [])):
        # User message
        st.markdown(f'<div class="chat-message user"><div class="avatar"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRhCtDRFGo8W5fLw1wg12N0zHKONLsTXgY3Ko1MDaYBc2INdt3-EU1MGJR9Thaq9lzC730&usqp=CAU"/></div><div class="message">{turn["human"]}</div></div>', unsafe_allow_html=True)
        cols = st.columns([0.85, 0.15])
        cols[1].button("🗑️", key=f"delete_msg_{i}", on_click=lambda idx=i: delete_message(selected_subject, idx))
        
        # Bot message (render as markdown)
        bot_message_html = markdown.markdown(turn["ai"])
        st.markdown(f'<div class="chat-message bot"><div class="avatar"><img src="https://img.freepik.com/premium-vector/ai-logo-template-vector-with-white-background_1023984-15069.jpg"/></div><div class="message">{bot_message_html}</div></div>', unsafe_allow_html=True)
        
        # Copy buttons
        cols = st.columns(2)
        cols[0].button("Copy Question", key=f"copy_q_{i}", on_click=lambda q=turn["human"]: clipboard.copy(q))
        cols[1].button("Copy Answer", key=f"copy_a_{i}", on_click=lambda a=turn["ai"]: clipboard.copy(a))

# Query input
query = st.text_input("Enter your question")

if query:
    with st.spinner("Generating answer..."):
        if selected_subject:
            relevant_texts = get_relevant_passage(query, selected_subject)
            chat_history = st.session_state.chat_histories.get(selected_subject, [])
            # if relevant_texts:
            #     final_prompt = make_rag_prompt(query, relevant_texts, selected_subject, chat_history)
            #     answer = generate_answer(final_prompt)
            # else:
            #     st.info(f"No relevant information found in the {selected_subject} database. Using LLM's knowledge to answer.")
            #     answer = generate_llm_answer(query, selected_subject, chat_history)
            final_prompt = make_rag_prompt(query, relevant_texts, selected_subject, chat_history)
            answer = generate_answer(final_prompt)
        else:
            st.info("No subject selected. Using general knowledge to answer.")
            answer = generate_llm_answer(query)
        
        if answer:
        # Display the new conversation
            st.markdown(f'<div class="chat-message user"><div class="avatar"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRhCtDRFGo8W5fLw1wg12N0zHKONLsTXgY3Ko1MDaYBc2INdt3-EU1MGJR9Thaq9lzC730&usqp=CAU"/></div><div class="message">{query}</div></div>', unsafe_allow_html=True)
            answer_html = markdown.markdown(answer)
            st.markdown(f'<div class="chat-message bot"><div class="avatar"><img src="https://img.freepik.com/premium-vector/ai-logo-template-vector-with-white-background_1023984-15069.jpg"/></div><div class="message">{answer_html}</div></div>', unsafe_allow_html=True)
            
            # Copy buttons for the new conversation
            cols = st.columns(2)
            cols[0].button("Copy Question", key="copy_current_q", on_click=lambda: clipboard.copy(query))
            cols[1].button("Copy Answer", key="copy_current_a", on_click=lambda: clipboard.copy(answer))
            
            # Update chat history
            if selected_subject:
                st.session_state.chat_histories.setdefault(selected_subject, []).append({
                    'human': query,
                    'ai': answer
                })
                save_chat_history(selected_subject, st.session_state.chat_histories[selected_subject])


# Sidebar information and buttons
st.sidebar.title("GTUtor Controls")

if selected_subject:
    db = get_or_create_db(selected_subject)
    total_docs = db.count()
    st.sidebar.write(f"Total documents in {selected_subject} database: {total_docs}")

    # Clear database button
    if st.sidebar.button(f"Clear {selected_subject} Database"):
        db.delete(delete_all=True)
        st.session_state.chat_histories[selected_subject] = []
        save_chat_history(selected_subject, [])
        st.sidebar.success(f"{selected_subject} database and chat history cleared successfully.")
        st.rerun()

    # Delete subject button
    if st.sidebar.button(f"Delete {selected_subject} Subject"):
        # Remove from subjects list
        subjects.remove(selected_subject)
        save_subjects(subjects)
        
        # Delete database
        db_path = os.path.join(db_folder, selected_subject.lower().replace(" ", "_"))
        if os.path.exists(db_path):
            import shutil
            shutil.rmtree(db_path)
        
        # Delete chat history
        if selected_subject in st.session_state.chat_histories:
            del st.session_state.chat_histories[selected_subject]
        history_file = os.path.join(history_folder, f"{selected_subject.lower().replace(' ', '_')}_history.json")
        if os.path.exists(history_file):
            os.remove(history_file)
        
        st.sidebar.success(f"{selected_subject} subject deleted successfully.")
        st.rerun()

# Option to start a new conversation
if st.sidebar.button("Start New Conversation"):
    if selected_subject:
        st.session_state.chat_histories[selected_subject] = []
        save_chat_history(selected_subject, [])
        st.success("New conversation started.")
        st.rerun()
    else:
        st.warning("Please select a subject before starting a new conversation.")

# Function to delete a specific message
def delete_message(subject, index):
    if subject in st.session_state.chat_histories:
        del st.session_state.chat_histories[subject][index]
        save_chat_history(subject, st.session_state.chat_histories[subject])
        st.rerun()

# Add custom CSS to improve readability
st.markdown("""
<style>
.stTextArea textarea {
    font-size: 16px !important;
}
</style>
""", unsafe_allow_html=True)
