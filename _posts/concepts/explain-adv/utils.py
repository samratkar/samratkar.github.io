from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
import streamlit as st

def get_router_query_engine(file_path: str, llm = None, embed_model = None):
    """Get router query engine."""
    llm = llm or OpenAI(model="gpt-3.5-turbo")
    embed_model = embed_model or OpenAIEmbedding(model="text-embedding-ada-002")
    
    # load documents
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    
    summary_index = SummaryIndex(nodes)
    vector_index = VectorStoreIndex(nodes, embed_model=embed_model)
    
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
        llm=llm
    )
    vector_query_engine = vector_index.as_query_engine(llm=llm)
    
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description=(
            "Useful for summarization questions related to the document"
        ),
    )
    
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=(
            "Useful for retrieving specific context from the document."
        ),
    )
    
    query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[
            summary_tool,
            vector_tool,
        ],
        verbose=True
    )
    return query_engine

def render_chat_ui(query_engine):
    # Set up the app title
    st.title("e-x-p-l-a-i-n agentic")

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
        chat_response = query_engine.query(query)
        st.session_state.history.append({"Query": query, "Answer": chat_response})
        msg1 = st.chat_message("user")
        msg1.write(f"**Q:** {query}")
        msg2 = st.chat_message("assistant")
        msg2.write(f"**A:** {chat_response}")

