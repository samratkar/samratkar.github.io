import helper
import utils
file_path = utils.init_chat_ui ("e-x-p-l-a-i-n", "ver 3.1 | agentic tool calling")
if file_path is not None:
    [query_engine,summary_tool, vector_tool, graph_tool ] = utils.get_router_query_engine(file_path)
    utils.render_chat_ui(query_engine, summary_tool, vector_tool, graph_tool, file_path)