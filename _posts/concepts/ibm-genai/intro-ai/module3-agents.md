---
type : notes 
title: "Module 3: AI Agents"
date: 12.4.2024
---

## AI Agents

### Definition

AI agents are software programs that `interact` with their environment, `collect and process data`, and `perform tasks on their own` to achieve `goals set by humans`. They can make decisions, solve problems and adapt to a new information without needing constant human help. 

```mermaid
graph LR
    A[Goals set by humans] --> B[AI Agent]
    C[Input from Environment data] --> B
    B --> E[Process data on their own]
    B --> F[Perform Tasks on their own]
    B --> G[Make decisions on their own]
    B --> H[Adapt to new information on their own]
    I[Without needing constant human help] --> B
    E --> J[Output to the Environment]
    F --> J
    G --> J
    H --> J
```

### Working of an AI agent

```mermaid
graph LR
    A[Perception] --> B[Understanding]
    B --> C[Decision making]
    C --> D[Action]
    D --> E[Learning]
```