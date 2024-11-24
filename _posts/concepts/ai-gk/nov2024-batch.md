---
title: AI-GK November 2024
type : concepts
---

1. function calling or tool calling is reliable from GPT4. Prior to GPT 4, it was not reliable.
2. these functions can be any API that can be called. 
3. Today, LLMs can decide to call functions to search for information for retrieval-augmented generation (RAG), execute code,  send emails, place orders online, and much more.
4. Recently, Anthropic released a version of its model that is capable of computer use, using mouse-clicks and keystrokes to operate a computer (usually a virtual machine).
5. In a much smaller number of cases, developers who are working on very valuable applications will fine-tune LLMs to carry out particular agentic functions more reliably. For example, even though many LLMs support function calling natively, they do so by taking as input a description of the functions available and then (hopefully) generating output tokens to request the right function call. For mission-critical applications where generating the right function call is important, fine-tuning a model for your application’s specific function calls significantly increases reliability. (But please avoid premature optimization! Today I still see too many teams fine-tuning when they should probably spend more time on prompting before they resort to this.)
6. 