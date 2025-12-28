# Sleep Assistant Chatbot Configuration

# Server Configuration
HOST = '0.0.0.0'
PORT = 5000
DEBUG = True

# Vector Database
VECTORDB_PATH = 'vectordb.json'

# Search Configuration
TOP_K_RESULTS = 3
MIN_SIMILARITY_THRESHOLD = 0.1

# Embedding Configuration
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # For sentence-transformers
EMBEDDING_DIMENSION = 384

# OpenAI Configuration (Optional)
OPENAI_MODEL = 'gpt-3.5-turbo'
OPENAI_MAX_TOKENS = 300
OPENAI_TEMPERATURE = 0.7

# Response Configuration
MAX_EXCERPT_LENGTH = 500
CONTEXT_WINDOW = 2  # Number of documents to use for context

# CORS Configuration
CORS_ORIGINS = ['http://localhost:4000', 'https://*.github.io']

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
