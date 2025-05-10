import global_settings
import function_tools
import chat_utils
import llm_utils
import agent_utils

# render the chat UI and import files
documents = chat_utils.init_chat_ui ("e-x-p-l-a-i-n", "ver 4.0 | agentic vector, summary, tool calling")
if documents is not None:
    # tokenize and transform the tokens into llama index nodes
    nodes = llm_utils.get_doc_nodes(documents)

    # instantiate the query engine tools and function tools
    vector_tool = llm_utils.get_vector_search_tool(nodes)
    summary_tool = llm_utils.get_summary_tool(nodes)
    render_flight_plan_tool = function_tools.get_render_flight_plan_tool()
    add_tool = function_tools.get_add_tool()   
    subtract_tool = function_tools.get_subtract_tool()
    mystery_tool = function_tools.get_mystery_tool()
    all_tools = [vector_tool, summary_tool, render_flight_plan_tool, add_tool, subtract_tool, mystery_tool]

    # create an agent to run the tools 
    agent = agent_utils.create_react_agent(all_tools)

    # start the chat using the UI layer
    chat_utils.start_chat(agent)
