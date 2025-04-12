# Add your utilities or helper functions to this file.
import os
from dotenv import load_dotenv, find_dotenv
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.embeddings.openai import OpenAIEmbedding
from pathlib import Path


# these expect to find a .env file at the directory above the lesson.                                                                                                                     # the format for that file is (without the comment)                                                                                                                                       #API_KEYNAME=AStringThatIsTheLongAPIKeyFromSomeService                                                                                                                                     
def load_env():
    _ = load_dotenv(find_dotenv())

def get_openai_api_key():
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key

def initialize_settings():
    # Set up LlamaIndex global settings
    load_env()
    llama_debug = LlamaDebugHandler()
    Settings.callback_manager = CallbackManager([llama_debug])
    Settings.llm = OpenAI(model="gpt-4")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    Settings.verbose = True

