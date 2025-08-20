---
layout : mermaid
title: "LLM Memory"
description: "Exploring the concept of memory in LLMs, including its importance and implementation."
tags: ["LLM", "Memory", "Concepts"]
date: 2025-08-20
author : "Samrat Kar"
---

#### Types of memory 
1. Semantic memory : 
   - General knowledge about the world, facts, concepts, and relationships.
   - Example: Knowing that Paris is the capital of France.
   - facts about users with whom the agent has interacted with.
2. Episodic memory :
   - Personal experiences and specific events in one's life.
   - Example: Remembering your last birthday party.
   - callback to exact past conversations or events. past agent actions.
3. Procedural memory :
   - Skills and how to perform tasks.
   - Example: Riding a bicycle or playing a musical instrument.
   - This type of memory is often implicit, meaning you may not consciously recall how to do it, but you can perform the task.
   - This is typically agent system prompts.
4. Working memory :
   - Short-term memory used for temporarily holding and manipulating information.
   - Example: Remembering a phone number long enough to dial it.
   - It has a limited capacity and duration, typically lasting only seconds to minutes.


#### Mechanisms of memory management 

![](/assets/genai/mem/mechanisms.png)

![](/assets/genai/mem/memtypes-integration.png)