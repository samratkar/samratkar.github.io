1. agents are software programs that are trigerred by LLM models, using reasoning. They take action on the behalf of the user. Following are the components of an agent -
   - Reasoning - powered by LLMs
   - Routing - interpreting request and determining the correct tool
   - Action - executing code / tools (utilizing APIs, making LLM calls)
3. Evaluating LLMS
   - LLM model evaluation - measure the general language understanding of the foundational model.
         - benchmark data sets >> LLMs. The datasets are off the shelf. 
         - eg - MMLU - massive multi task language understanding. tests the capabilit of language models to understand and respond to complex questions across diverse domains.
         - eg - HumanEval - tests the code generation skills of LLM.
         - eg - Stanford Question Answering Dataset - SQuAD
   - LLM system evaluation - measure how the entire system including the LLM performs meeting the business requirements.
         - testing datasets >> LLM based applications. Testing datasets can be manually created, synthesized or curated from the application itself.
         - the testing datasets are typically custom created for that particular system.
         - the evaluation need to happen for the following for a given user query & output - prompt, tools, data sources, memory, routing components
4. The paradigm shift of testing LLMs based systems over traditional software systems.
   - Non deterministic in nature. for same inputs, the outputs can vary.
   - focus on application's ability to respond to user specific tasks.
   - the testing metrics are qualitative. relevance, coherence, etc.
5. Evaluation types -
   - Hallucinations - is the LLM system accurately using the provided context or is it making things up.
   - Retrieval relevance - The query and documents retrieved by the system is relevant to the query.
   - Q&A accuracy on the retrieved data - Does the response match the ground truth or the user needs.
   - Toxicity - whether the system outputing harmful or undesirable language
   - Summarization performance 
   - Code writing correctness and readability
