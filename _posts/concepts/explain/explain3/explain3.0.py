import helper
import utils
file_path = utils.init_chat_ui ("e-x-p-l-a-i-n", "ver 2.1 | agentic | file upload")
if file_path is not None:
    query_engine = utils.get_router_query_engine(file_path)
    utils.render_chat_ui(query_engine)