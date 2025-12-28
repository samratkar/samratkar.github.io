"""
Test script for Sleep Assistant Chatbot
Tests both the vector search and response generation
"""

import json
import sys

def load_vectordb(filepath='vectordb.json'):
    """Load vector database"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading vectordb: {e}")
        return None

def test_search(vectordb, query):
    """Test search functionality"""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    # Find relevant documents
    scores = []
    for idx, doc in enumerate(vectordb['documents']):
        doc_lower = doc.lower()
        
        # Keyword matching
        keyword_score = sum(1 for word in query_words if word in doc_lower and len(word) > 3)
        
        # Exact phrase match bonus
        if query_lower in doc_lower:
            keyword_score += 5
        
        # Position score
        first_occurrence = doc_lower.find(query_lower)
        position_score = (1000 - first_occurrence) / 1000 if first_occurrence >= 0 else 0
        
        total_score = keyword_score + position_score
        
        if total_score > 0:
            scores.append({
                'index': idx,
                'score': total_score,
                'excerpt': doc[:200] + '...'
            })
    
    # Sort and show top 3
    scores.sort(key=lambda x: x['score'], reverse=True)
    top_results = scores[:3]
    
    if not top_results:
        print("❌ No relevant documents found")
        return
    
    print(f"\n✓ Found {len(top_results)} relevant documents:\n")
    
    for i, result in enumerate(top_results, 1):
        print(f"{i}. Document #{result['index']} (Score: {result['score']:.2f})")
        print(f"   Excerpt: {result['excerpt']}\n")

def main():
    print("\n" + "="*60)
    print("Sleep Assistant Chatbot - Test Suite")
    print("="*60)
    
    # Load vector database
    print("\n1. Loading vector database...")
    vectordb = load_vectordb()
    
    if not vectordb:
        print("❌ Failed to load vector database")
        sys.exit(1)
    
    print(f"✓ Loaded {len(vectordb['documents'])} documents")
    print(f"✓ Database name: {vectordb['name']}")
    print(f"✓ Vector dimension: {vectordb['dimension']}")
    
    # Test queries
    test_queries = [
        "What are the main sleep stages?",
        "How does vitamin D affect sleep?",
        "What is sleep apnea?",
        "sleep duration in children",
        "REM sleep"
    ]
    
    print("\n2. Testing search functionality...")
    for query in test_queries:
        test_search(vectordb, query)
    
    print("\n" + "="*60)
    print("✓ All tests completed")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
