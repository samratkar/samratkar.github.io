---
layout: post
title:  "The Journey"
date:   2024-10-05 15:49:55 +0530
categories: phd updates
---
# 1. Realtime data analytics - Done

1. Did the realtime data analytics of the ADSB data being streamed from [open-sky.](https://openskynetwork.github.io/opensky-api/)
2. Integrated Mongo Db atlas database the analytics on the realtime data being streamed in the [airspace metrics chart](https://charts.mongodb.com/charts-project-0-ohwbybj/public/dashboards/c7ea23df-7b65-4361-a644-f9b1344504aa).
3. Integrated Tableau with the data and built EDA dashboard on the realtime streaming data on [Tableau charts](https://public.tableau.com/views/airdata-viz/Dashboard1?:language=en-GB&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link) as well.
4. Ported the code into databricks community edition as well.
5. Code base - [gitlab-repo](https://gitlab.com/samratk/mtech-cloudcomputing/-/tree/main/4sem/final-yr-project/code?ref_type=heads).
6. Architecture diagram -

![arch-dia](/assets/img/open-sky-rt-stream.jpg)

![data-st-viz](/assets/img/data-stream-viz.jpg)

# 2. Ollama embedding - Done

1. Installed Ollama in Linux.
2. Downladed the phi-3 SLM from Ollama to the machine.
3. Embedded some static data on the model file and queried the model and qualitatively evaluated the training with the emeddings.

![ollama](/assets/img/ollama-phi3.jpg)

# 3. Architecture diagram - building a rag flight bot - Done

![arch-flight-bot](/assets/img/rag.jpg)

# 4. Optimization of retrieval of the rag - In progress

Optimization of retrieval piece of a RAG is of prime importance while building a virtual co pilot or a flight bot in the flight deck. This is especially important when we need to search for a given query a huge corpus of data. If there are millions of vectors to search, then storing, indexing and searching can be resource intensive and slow.

If a pilot assistant app is to be created that gathers all the dynamic data from the realtime streaming of the QAR, realtime weather information, realtime traffic information, and all the static corpus of data the comprises of the pilot operational manuals, standard operating procedures, etc, then we might have really huge set of vectors to search from. And adding up with the historical data, these vectors that need to be searched for a given query would be in millions.

To reduce this and simplify the search, quantization is done, that greatly helps the need of huge memory to do the computation. This exploration is towards understanding RAG and getting deep into how to optimize from tokenization to quantization.

[Details notes of the optimization process. ](https://samratkar.github.io/2024/10/05/retrieval-optimization.html)

# 5. On hands usage of the off the shelf LLMSs - In progress

Focussing on doing some on-hands coding for using off the shelf LLMs to know how to use them. I am following these videos to do this -

1. https://www.youtube.com/watch?v=oNA6pTNfHCY&t=3180s - open source LLMs coding and tuning
2. https://www.youtube.com/watch?v=ou-BDb_3zLc&list=PL65wd6VFbIoC5Si5YO8VsdpmkzArUAlAz&index=6 - building RAG in azure
3. https://www.youtube.com/watch?v=sVcwVQRHIc8&list=PL65wd6VFbIoC5Si5YO8VsdpmkzArUAlAz&index=4 - RAG in python

# 6. Exploration on AI safety - In progress

This page has the details on this exploration - [Managing AI Risk] (/_posts/2024-10-09-managing-ai-risk)
