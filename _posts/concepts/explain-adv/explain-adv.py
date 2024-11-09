import helper
import utils
query_engine = utils.get_router_query_engine("../../../data/pilot-manual-787.pdf")
utils.render_chat_ui(query_engine)