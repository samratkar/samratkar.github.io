---
layout: post
title:  "The Journey"
date:   2024-10-05 15:49:55 +0530
categories: phd
---
# 1. Realtime data analytics

* [X] 1-Did the realtime data analytics of the ADSB data being streamed from [open-sky.](https://openskynetwork.github.io/opensky-api/)
* [X] 2-Integrated Mongo Db atlas database the analytics on the realtime data being streamed in the [airspace metrics chart](https://charts.mongodb.com/charts-project-0-ohwbybj/public/dashboards/c7ea23df-7b65-4361-a644-f9b1344504aa).
* [X] 3-Integrated Tableau with the data and built EDA dashboard on the realtime streaming data on [Tableau charts](https://public.tableau.com/views/airdata-viz/Dashboard1?:language=en-GB&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link) as well.
* [X] 4-Ported the code into databricks community edition as well.

Code base - [gitlab-repo](https://gitlab.com/samratk/mtech-cloudcomputing/-/tree/main/4sem/final-yr-project/code?ref_type=heads).

# 2. Ollama embedding

1. [X] 1-Installed Ollama in Linux.
2. [X] 2-Downladed the phi-3 SLM from Ollama to the machine.
3. [X] 3-Embedded some static data on the model file and queried the model and qualitatively evaluated the training with the emeddings.

![ollama](/assets/img/ollama-phi3.jpg)

# 3. RAG creation and  optimization

1. [X] Created a RAG on open AI API to load and embedd on pdf file.
2. [ ] Optimizing the RAG to tokenize and quantize `<in progress>`
3. [ ] RAG using graph db

## 3.1. Optimization of retrieval of the rag

Optimization of retrieval piece of a RAG is of prime importance while building a virtual co pilot or a flight bot in the flight deck. This is especially important when we need to search for a given query a huge corpus of data. If there are millions of vectors to search, then storing, indexing and searching can be resource intensive and slow.

If a pilot assistant app is to be created that gathers all the dynamic data from the realtime streaming of the QAR, realtime weather information, realtime traffic information, and all the static corpus of data the comprises of the pilot operational manuals, standard operating procedures, etc, then we might have really huge set of vectors to search from. And adding up with the historical data, these vectors that need to be searched for a given query would be in millions.

To reduce this and simplify the search, quantization is done, that greatly helps the need of huge memory to do the computation. This exploration is towards understanding RAG and getting deep into how to optimize from tokenization to quantization.

* [ ] 1-[Details notes of the optimization process. ](https://samratkar.github.io/2024/10/05/retrieval-optimization.html)`<in progress>`
* [ ] 2-[Code base for rag processes](https://github.com/samratkar/research-papers/tree/main/courses/rag-datacamp) `<in progress>`

# 4. Exploration on AI safety - In progress

1. This page has the details on this exploration - [https://samratkar.github.io/journals/2024/10/09/managing-ai-risk.html](https://samratkar.github.io/journals/2024/10/09/managing-ai-risk.html)
2. I have been doing a research survey of the work that is happening in the area of AI safety. My aim is to do my deep dive into AI alignment problem. For doing so, I am planning to do few concepts revisions on reinforcement learning.
3. I also completed one survey paper on the application of generative AI on flight deck.
4. Currently doing course on explainable AI.
