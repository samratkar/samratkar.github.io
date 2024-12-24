---
title : "Retrieval Optimization"
date : 2024-10-05
---
# Retrieval Optimization

Retrieval Augmented Generation contains two steps - 

1. A retriever searches a large document corpus to find relevant information.
2. Then a generator uses this information to produce accurate and contextually relevant results for user's query.

This article is about optimizing the 1st step - retrieval. 

The first thing a retrieval system does is tokenization. Tokenisation is the process of representing creating a sequence of numbers representing words or parts of words. There are several ways to represent a string of words into tokens. The different tokenization techniques are -

1. wordpiece
2. byte pair encoding
3. unigram tokenization

# Quality metrics to measure the search efficiency

1. HNSW - Hierarchical navigable small words - Vector databases use specialized data structure to optimize and approximate the search for nearest neighbours. HNSW is one such data structure. It gives some parameters to measure how good the approximation is.
   HNSW is built on a multi layer graph.
2. The idea is to balance the parameters used for forming and searching the HNSW graph for higher speed and maximum relevance.

# Quantization techniques

If there are **millions** **of vectors** to search, then storing, indexing and searching can be resource intensive and slow. If a pilot assistant app is to be created that gathers all the dynamic data from the realtime streaming of the QAR, realtime weather information, realtime traffic information, and all the static corpus of data the comprises of the pilot operational manuals, standard operating procedures, etc, then we might have really huge set of vectors to search from. And adding up with the historical data, these vectors that needs to be searched for a given query would be in millions. information 
