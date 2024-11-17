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
import os 
from pathlib import Path 
import geopandas as gpd
import folium
from llama_index.core.tools import FunctionTool

graph_tool = None 
summary_tool = None
vector_tool = None
file_path_global = None 

llm = OpenAI(model="gpt-3.5-turbo")

def render_fpln():
    """Renders and visualizes the flight plan."""
    # Read the KML file
    st.write (file_path_global)
    gdf = gpd.read_file(file_path_global)

    # Calculate the centroid of the geometries
    centroid = gdf.geometry.centroid

    # Create a Folium map
    # m = folium.Map(location=[gdf.centroid.y, gdf.centroid.x], zoom_start=10)
    m = folium.Map(location=[centroid.y.mean(), centroid.x.mean()], zoom_start=10)


    # Add the flight path to the map
    folium.GeoJson(gdf, style_function=lambda x: {'color': 'blue', 'weight': 3}).add_to(m)

    # Display the map
    return st.components.v1.html(m._repr_html_(), height=600)

def get_router_query_engine(file_path: str, llm = None, embed_model = None):
    file_path_global = file_path
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

    graph_tool = FunctionTool.from_defaults(
    name="graph_tool",
    fn=render_fpln)
    
    query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[
            summary_tool,
            vector_tool,
        ],
        verbose=True
    )
    return query_engine, summary_tool, vector_tool, graph_tool

def render_chat_ui(query_engine, summary_tool, vector_tool, graph_tool, file_path):
    file_path_global = file_path 
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
        chat_response = llm.predict_and_call(
                    [summary_tool, vector_tool, graph_tool], 
                    query, 
                    verbose=True
                )
        st.session_state.history.append({"Query": query, "Answer": chat_response})
        msg1 = st.chat_message("user")
        msg1.write(f"**Q:** {query}")
        msg2 = st.chat_message("assistant")
        msg2.write(f"**A:** {chat_response}")

def init_chat_ui(chat_title: str, version: str):
    # Set up the app title
    st.title(chat_title)
    st.subheader(version)
    
    # Loop until a file is uploaded
    file_path = None 
    uploaded_file = None
    uploaded_file = st.file_uploader("Upload a file", key="file_uploader")
    cwd = Path.cwd()
    if uploaded_file is not None:
        file_path = cwd / uploaded_file.name 
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None
  
