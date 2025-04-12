import streamlit as st
from llama_index.core.agent import ReActAgent
import tempfile
import os
import shutil
from llama_index.core import SimpleDirectoryReader, Document
from typing import List
import uuid

def start_chat(agent: ReActAgent):
    # Initialize session state for history if not already done
    if 'history' not in st.session_state:
        st.session_state.history = []  # list to store Q&A pairs

    # Display the Q&A history in chronological order (oldest at the top)
    for qa in st.session_state.history:
        msg1 = st.chat_message("user")
        msg1.write(f"**Q:** {qa['Query']}")
        msg2 = st.chat_message("assistant")
        msg2.write(f"**A:** {qa['Answer']}")
        #st.write("---")  # Divider between Q&A pairs

    # Input field at the bottom of the chat history
    query = st.chat_input("Say something")
    # Append the Q&A to history if a question is entered
    if query:
        # chat_response = query_engine.query(query)
        chat_response = agent.chat(query)
        st.session_state.history.append({"Query": query, "Answer": chat_response})
        msg1 = st.chat_message("user")
        msg1.write(f"**Q:** {query}")
        msg2 = st.chat_message("assistant")
        msg2.write(f"**A:** {chat_response}")

def init_chat_ui(chat_title: str, version: str):
    # Set up the app title
    st.title(chat_title)
    st.subheader(version)
    documents = []
    uploaded_files = []
    uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True)

    if uploaded_files:
        st.write(f"Processing {len(uploaded_files)} files...")
        # Process the uploaded files
        documents = process_uploaded_files(uploaded_files)
    
        # Display information about the processed documents
        st.write(f"Successfully processed {len(documents)} documents")
    
        # Show document info
        for i, doc in enumerate(documents):
            st.write(f"Document {i+1}:")
            st.write(f"- ID: {doc.doc_id}")
            st.write(f"- Metadata: {doc.metadata}")
            st.text(f"- Preview: {doc.text[:100]}...")
    
    return documents

def process_uploaded_files(uploaded_files):
    """
    Process uploaded files from Streamlit using LlamaIndex's SimpleDirectoryReader
    to ensure proper metadata inclusion.
    
    Args:
        uploaded_files: List of files from st.file_uploader
        
    Returns:
        List of LlamaIndex Document objects with proper metadata
    """
    if not uploaded_files:
        return []
    
    # Create a temporary directory to store the uploaded files
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploaded files to the temporary directory
        file_paths = []
        for uploaded_file in uploaded_files:
            # Generate a unique filename to avoid collisions
            file_name = f"{uuid.uuid4()}_{uploaded_file.name}"
            file_path = os.path.join(temp_dir, file_name)
            
            # Save the file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)
        
        # Use SimpleDirectoryReader to properly load the documents with metadata
        reader = SimpleDirectoryReader(input_dir=temp_dir)
        documents = reader.load_data()
        
        # # You can add additional metadata if needed
        # for i, doc in enumerate(documents):
        #     # Add original filename to metadata
        #     original_filename = uploaded_files[i].name
        #     if not hasattr(doc, 'metadata') or doc.metadata is None:
        #         doc.metadata = {}
        #     doc.metadata["original_filename"] = original_filename
        
        return documents
        
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)
  
