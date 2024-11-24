---
type : concepts
title: Agentic Systems
date: 2024-11-24
theme : notes 
source : https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/?utm_campaign=The%20Batch&utm_source=hs_email&utm_medium=email&_hsenc=p2ANqtz-_gIrmbxcITc28FhuvGDCyEatfevaCrKevCJqk0DMR46aWOdQblPdiiop0C21jprkMtzx6e 
---
## Agentic Systems

### zero shot prompting

1. zero shot prompting : Using an LLM in zero shot means prompting the model to perform a task without providing any specific examples or training data related to the task that needs to be performed. The model relies on its general knowledge and understanding of language to generate the response. 
zero shot prompting can be less accurate compared to fine tuned models. And LLMs may struggle with highly specialised or domain specific tasks. 

### agentic workflows

with agentic workflow, however we can ask the LLM to iterate over a document many times. For example it might take a sequence of steps such as :

1. plan an outline
2. decide whether, if any web searches are needed to gather more information.
3. write a first draft
4. read over the first draft to sport arguments or extraneous information.
5. revise draft taking into account any weaknesses spotted.
6. and so on.

Such interactive workflow yields much better results than writing in a single pass.

### agentic design patterns

1. reflection : the LLM examines its own work to come up with the ways to improve it.
2. tool use : the LLM is given tools such as web search, code execution, or any other function to help it gather information, take actions, or process data.
3. planning : the LLM comes with and executes, a multi-step plan to achieve a goal
4. multi agent collaboration : more than one AI agent work together, splitting up tasks and discussing and debating ideas, to come up with better solutions than a single agent world.

#### reflection

1. it is about letting the LLM review it's working by giving appropriate prompts to validate itself.
2. Take the task of asking an LLM to write code. We can prompt it to generate the desired code directly to carry out some task X. After that, we can prompt it to reflect on its own output, perhaps as follows:
Here’s code intended for task X: [previously generated code]
Check the code carefully for correctness, style, and efficiency, and give constructive criticism for how to improve it.
basically, send the previous response as a new prompt, adding prompts on evaluating the response.
3. *tools to do evaluation and correction* : And we can go beyond self-reflection by giving the LLM tools that help evaluate its output; for example, running its code through a few unit tests to check whether it generates correct results on test cases or searching the web to double-check text output. Then it can reflect on any errors it found and come up with ideas for improvement.
4. *multi-agents evaluation* - Further, we can implement Reflection using a multi-agent framework. I've found it convenient to create two different agents, one prompted to generate good outputs and the other prompted to give constructive criticism of the first agent's output. The resulting discussion between the two agents leads to improved responses.

### tool use

### planning

### multi agent collaboration
