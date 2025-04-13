from llama_index.core.tools import QueryEngineTool
from llama_index.core import VectorStoreIndex, SummaryIndex
from llama_index.core.node_parser import SentenceSplitter
def get_doc_nodes(documents):
    """Get document nodes from file path."""
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes

### Define Query Engine Tools
def get_vector_search_tool(nodes) -> QueryEngineTool:
    """Get the vector QueryEngine tool."""
    # Create a vector index
    vector_index = VectorStoreIndex(nodes)
    vector_query_engine = vector_index.as_query_engine()

    # Create vector search tool
    vector_search_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        name="vector_search",
        description="Useful for searching information in a knowledge base of AI and ML concepts"
    )
    return vector_search_tool

def get_summary_tool(nodes) -> QueryEngineTool:
    """Get the summary QueryEngine tool."""
    # Create a summary index
    summary_index = SummaryIndex(nodes)
    summary_query_engine = summary_index.as_query_engine()

    # Create summary tool
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        name="summary",
        description="Useful for summarization questions related to the documents"
    )
    return summary_tool
