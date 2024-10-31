import os 
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document 
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st 
from dotenv import load_dotenv

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


pdf_loader = PyPDFLoader(file_path='../../../data/pilot-manual-787.pdf')

openai_api_key = os.environ.get('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings(api_key=openai_api_key, model='text-embedding-3-small')
semantic_splitter = SemanticChunker(
    embeddings=embeddings, 
    breakpoint_threshold_amount=0.8)
documents = pdf_loader.load()
chunks = semantic_splitter.split_documents(documents)

bm25_retriever = BM25Retriever.from_documents(
    documents=chunks, k=5)

# Designing a prompt template. 
prompt = ChatPromptTemplate.from_template("""
Use the following pieces of context to answer the question at the end.
If you don't know the answer, say that you don't know.
Context:{context}
Question:{question}
""")

llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)


chain = ({"context":bm25_retriever, "question":RunnablePassthrough()}
         | prompt
         | llm 
         | StrOutputParser())


##--------------------------------------- Streamlit app

# Set up the app title
st.title("e-x-p-l-a-i-n")

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
    chat_response = chain.invoke(query)
    st.session_state.history.append({"Query": query, "Answer": chat_response})
    msg1 = st.chat_message("user")
    msg1.write(f"**Q:** {query}")
    msg2 = st.chat_message("assistant")
    msg2.write(f"**A:** {chat_response}")


