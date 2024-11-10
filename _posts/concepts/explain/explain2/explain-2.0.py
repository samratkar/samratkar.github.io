import os
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


##--------------------------------------- Streamlit app
def render_chat_ui():
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


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")


# documents = SimpleDirectoryReader(input_files=["../../../../data/metagpt.pdf"]).load_data()
documents = SimpleDirectoryReader(input_files=["../../../../data/master_flight_data-9.csv"]).load_data()
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)

summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)

summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
vector_query_engine = vector_index.as_query_engine()

summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description=(
        "Useful for summarization questions."
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific context."
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

render_chat_ui()
# response = query_engine.query("Tell me about the ablation study results?")
# print(str(response))

