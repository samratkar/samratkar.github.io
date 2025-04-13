from llama_index.core.agent import ReActAgent

def create_react_agent(function_tools):
    agent = ReActAgent.from_tools(
    function_tools,
    verbose=True,
    system_prompt="""You are a helpful assistant with access to the following tools:
    - add: For adding two numbers
    - subtract: For subtracting one number from another
    - vector_search: For searching information in a knowledge base about AI and ML concepts
    - summarize: For summarizing text to extract key points
    
    Based on the user's query, determine which tool is most appropriate and use it to respond.
    For math operations, use the add or subtract tools.
    For information retrieval about AI/ML, use the vector_search tool.
    For condensing long text, use the summarize tool.
    """)
    return agent
