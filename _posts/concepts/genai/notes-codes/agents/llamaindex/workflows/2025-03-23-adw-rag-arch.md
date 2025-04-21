---
layout: mermaid
title: "Agentic Document Workflow (ADW)"
author: Samrat Kar
---

# RAG architecture
![](/images/genai/rag-arch.png)


![](/images/genai/rag-workflow.svg)

# Limitation of RAG
When the query is a complex one with multi-part questions, the RAG system may not be able to retrieve all the relevant information from the knowledge base. The context generated is less focussed for a complex query. And hence the search is less focussed. 
![rag limit](/images/genai/rag-limit.png)

# Solution
Break the question into smaller simpler questions. And then pass to the RAG system. And then combine the results. Do this splitting up and combining using LLM.
![](/images/genai/solution-rag-limit.png)

# Event based Agentic document workflow (ADW)
Workflows are building blocks of agentic systems in llamaindex. They are event based system that lets one to define a series of steps connected by events and pass information from step to step. Following complications can be introduced into the workflows. 
1. parallel execution
2. loops
3. branches
![](/images/genai/workflows.png)

# Mindmap of the entire concept
![](/assets/)

