"""
Sleep Assistant Chatbot - Python Backend (Optional)
This Flask-based backend provides advanced features like:
- Better semantic search using sentence-transformers
- OpenAI GPT integration for natural responses
- Chat history persistence
- Advanced RAG (Retrieval Augmented Generation)

To use this backend:
1. Install dependencies: pip install flask flask-cors numpy sentence-transformers openai
2. Set OPENAI_API_KEY environment variable (optional, for GPT responses)
3. Run: python app.py
4. The chatbot will run on http://localhost:5000
5. Update chatbot.js to use this API endpoint instead of local processing
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
from typing import List, Dict, Tuple
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for GitHub Pages

# Global variables
vector_db = None
embeddings = None

# Optional: Use sentence transformers for better embeddings
USE_TRANSFORMERS = False
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    USE_TRANSFORMERS = True
    print("✓ Using Sentence Transformers for embeddings")
except ImportError:
    print("⚠ Sentence Transformers not available, using simple embeddings")

# Optional: Use OpenAI for better responses
USE_OPENAI = False
try:
    import openai
    openai.api_key = os.getenv('OPENAI_API_KEY')
    if openai.api_key:
        USE_OPENAI = True
        print("✓ OpenAI API configured")
except ImportError:
    print("⚠ OpenAI library not available")


def load_vector_database(filepath='vectordb.json'):
    """Load the vector database from JSON file"""
    global vector_db, embeddings
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            vector_db = json.load(f)
        
        # Convert embeddings if they exist in the file
        if 'embeddings' in vector_db:
            embeddings = np.array(vector_db['embeddings'])
        else:
            # Generate embeddings for documents
            print("Generating embeddings for documents...")
            embeddings = generate_embeddings(vector_db['documents'])
        
        print(f"✓ Loaded {len(vector_db['documents'])} documents")
        return True
    except Exception as e:
        print(f"✗ Error loading vector database: {e}")
        return False


def generate_embeddings(texts: List[str]) -> np.ndarray:
    """Generate embeddings for a list of texts"""
    if USE_TRANSFORMERS:
        return model.encode(texts, show_progress_bar=True)
    else:
        # Simple TF-IDF style embeddings
        return np.array([simple_embedding(text) for text in texts])


def simple_embedding(text: str, dim: int = 384) -> np.ndarray:
    """Simple hash-based embedding (fallback)"""
    words = text.lower().split()
    embedding = np.zeros(dim)
    
    for idx, word in enumerate(words):
        for i, char in enumerate(word):
            index = (ord(char) * (i + 1) + idx) % dim
            embedding[index] += 1.0 / (len(word) + 1)
    
    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def search_documents(query: str, top_k: int = 3) -> List[Dict]:
    """Search for relevant documents using semantic similarity"""
    if vector_db is None or embeddings is None:
        return []
    
    # Generate query embedding
    if USE_TRANSFORMERS:
        query_embedding = model.encode([query])[0]
    else:
        query_embedding = simple_embedding(query)
    
    # Calculate similarities
    similarities = []
    for idx, doc_embedding in enumerate(embeddings):
        similarity = cosine_similarity(query_embedding, doc_embedding)
        similarities.append({
            'index': idx,
            'document': vector_db['documents'][idx],
            'similarity': float(similarity)
        })
    
    # Sort by similarity and return top K
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    return similarities[:top_k]


def generate_response(query: str, relevant_docs: List[Dict]) -> str:
    """Generate a response based on relevant documents"""
    if not relevant_docs:
        return ("I couldn't find specific information about that in my knowledge base. "
                "Could you rephrase your question or ask about sleep stages, disorders, "
                "or sleep recommendations?")
    
    if USE_OPENAI:
        # Use OpenAI GPT for natural response generation
        context = "\\n\\n".join([doc['document'][:500] for doc in relevant_docs[:2]])
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful sleep science assistant. "
                     "Answer questions based on the provided research context. Be accurate and concise."},
                    {"role": "user", "content": f"Context:\\n{context}\\n\\nQuestion: {query}"}
                ],
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            # Fall back to simple response
    
    # Simple response using top document
    top_doc = relevant_docs[0]['document']
    excerpt_length = 500
    
    # Find most relevant excerpt
    query_words = set(query.lower().split())
    best_excerpt = top_doc[:excerpt_length]
    
    # Try to find excerpt with most query words
    best_score = 0
    for i in range(0, len(top_doc) - excerpt_length, 100):
        excerpt = top_doc[i:i + excerpt_length].lower()
        score = sum(1 for word in query_words if word in excerpt)
        if score > best_score:
            best_score = score
            best_excerpt = top_doc[i:i + excerpt_length]
    
    return f"Based on the research literature:\\n\\n{best_excerpt.strip()}..."


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'vector_db_loaded': vector_db is not None,
        'documents_count': len(vector_db['documents']) if vector_db else 0,
        'use_transformers': USE_TRANSFORMERS,
        'use_openai': USE_OPENAI
    })


@app.route('/search', methods=['POST'])
def search():
    """Search endpoint for finding relevant documents"""
    data = request.json
    query = data.get('query', '')
    top_k = data.get('top_k', 3)
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    relevant_docs = search_documents(query, top_k)
    
    return jsonify({
        'query': query,
        'results': relevant_docs
    })


@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    # Search for relevant documents
    relevant_docs = search_documents(query, top_k=3)
    
    # Generate response
    response = generate_response(query, relevant_docs)
    
    return jsonify({
        'query': query,
        'response': response,
        'sources': [{'index': doc['index'], 'similarity': doc['similarity']} 
                   for doc in relevant_docs],
        'confidence': relevant_docs[0]['similarity'] if relevant_docs else 0.0
    })


@app.route('/')
def index():
    """Root endpoint with API information"""
    return jsonify({
        'message': 'Sleep Assistant Chatbot API',
        'version': '1.0.0',
        'endpoints': {
            '/health': 'GET - Health check',
            '/search': 'POST - Search documents',
            '/chat': 'POST - Chat with the assistant'
        }
    })


if __name__ == '__main__':
    print("="*50)
    print("Sleep Assistant Chatbot Backend")
    print("="*50)
    
    # Load vector database
    if load_vector_database():
        print("✓ Server ready to start")
        print("="*50)
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("✗ Failed to load vector database")
        print("Please ensure vectordb.json is in the same directory")
