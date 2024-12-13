---
title: AI-GK Batch Dec 2024
type : batch
tags : #amazon-models
---

## Amazon LLM models

1. 3 vision language models 
   1. Nova Premier (2025)
   2. Nova Pro - comparable to claude anthropic 3.5 sonnet, OpenAI GPT-4o, Google Gemini Pro. 
      1. Processing time : 95 tokens per second.(GPT-4o (115 tokens per second))
      2. Pricing : $0.80/$3.20 per million tokens input/output. (GPT-4o ($2.50/$10) and Claude 3.5 Sonnet ($3/$15) )
      3. Token input context window : 300K
   3. Nova Lite - Claude Haiku, Google Gemini 1.5 Flash, and OpenAI GPT-4o Mini.
      1. Processing time : 157 tokens per second. Gemini 1.5 Flash (189 tokens per second).
      2. Pricing : $0.06/$0.24 per million tokens of input/output (GPT-4o mini ($0.15/$0.60), Claude 3.5 Haiku ($0.80/$4), or Gemini 1.5 Flash ($0.075/$0.30))
      3. Token input context window : 300K
2. language model
   1. Nova Micro
      1. Processing Time : 210 tokens per second. Gemini Flash 8B (284.2 tokens per second).
      2. Pricing : $0.035/$0.14 per million input/output tokens. Gemini Flash 8B ($0.0375/$0.15) and Llama 3.1 8B ($0.10/$0.10)
      3. Token input context window : 128K
3. Image generator
   1. Nova Canvas : 
      1. input text context - 1024 characters
      2. output image - 4.2 megapixel. does inpainting, outpainting and background removal.
4. Video generator
   1. Nova Reel
      1. input text context - 512 characters
      2. output image - 720X280 pixels.

## Amazon bedrock platform

The company launched Bedrock in April 2023 with Stability AI’s Stable Diffusion for image generation, Anthropic’s Claude and AI21’s Jurassic-2 for text generation, and its own Titan models for text generation and embeddings. Not long afterward, it added language models from Cohere as well as services for agentic applications and medical applications. It plans to continue to provide models from other companies (including Anthropic), offering a range of choices.

## Pricing

Nova’s pricing continues the rapid drop in AI prices over the last year. Falling per-token prices help make AI agents or applications that process large inputs more practical. For example, Simon Willison, developer of the Django Python framework for web applications, found that Nova Lite generated descriptions for his photo library (tens of thousands of images) for less than $10.

