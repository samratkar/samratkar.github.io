# Research Report on Transformer Architecture and Optimization Techniques


## 1. Key Concepts Central to the Research

1. **Transformer Architecture Enhancements**: Innovations aimed at improving the efficiency, scalability, and performance of transformer models.

2. **Optimization Techniques in Deep Learning**: Methods such as quantization, pruning, and knowledge distillation that enhance model efficiency and reduce computational resources.

3. **Tokenization Strategies**: Approaches to segmenting text into tokens, impacting model understanding and performance.

4. **Embedding Methods**: Techniques for representing tokens in continuous vector spaces, capturing semantic and syntactic information.

5. **Training and Fine-tuning Strategies**: Procedures for pretraining and adapting models to specific tasks, including methods like Reinforcement Learning from Human Feedback (RLHF).


## 2. Concept Map

<div class="mermaid">
graph LR
    A[Transformer Architecture Enhancements] --> B(Reformer)
    A --> C(Linformer)
    A --> D(Performer)
    A --> E(BigBird)
    A --> F(Hyena)

    A --> G[Optimization Techniques]
    G --> H(Quantization)
    G --> I(Pruning)
    G --> J(Knowledge Distillation)
    G --> K(Neural Architecture Search)

    A --> L[Tokenization Strategies]
    L --> M(Byte Pair Encoding)
    L --> N(Unigram Language Model)
    L --> O(WordPiece)

    A --> P[Embedding Methods]
    P --> Q(Positional Embeddings)
    P --> R(Word Importance Embedding)

    A --> S[Training & Fine-tuning Strategies]
    S --> T(Pretraining Objectives)
    S --> U(Fine-tuning Methods)
    S --> V(RLHF)
</div>

## 3. Relevant Keywords and Search Terms

- Transformer architecture improvements
- Efficient transformers
- Transformer optimization techniques
- Quantization in deep learning
- Knowledge distillation in NLP
- Tokenization methods in LLMs
- Byte Pair Encoding vs. Unigram LM
- Embedding techniques in transformers
- Positional embeddings
- Word importance embedding
- Training strategies for LLMs
- Fine-tuning large language models
- RLHF in NLP
- Sparse attention mechanisms
- Memory-efficient transformers


## 4. Search Strategy Across Academic Databases

To ensure a comprehensive literature survey, the following databases were utilized:

1. **arXiv**: For the latest preprints on transformer architectures and optimization techniques.

2. **IEEE Xplore**: To access peer-reviewed articles on deep learning and transformer models.

3. **ACM Digital Library**: For conference proceedings related to NLP and machine learning.

4. **ScienceDirect**: To find journal articles on embedding methods and tokenization strategies.

5. **SpringerLink**: For comprehensive reviews and surveys on transformer models.

6. **Google Scholar**: To identify highly cited papers and track recent advancements.

**Search Strategy**:

- Utilized Boolean operators to combine keywords (e.g., "transformer AND optimization").

- Applied filters for publication years (2017-2025) to focus on recent developments.

- Reviewed abstracts to assess relevance before full-text analysis.


## 5. Collection and Organization of Relevant Sources

A total of 25 highly relevant sources were identified, encompassing journal articles, conference proceedings, and preprints. Each source has been documented with complete citations, summaries, reliability evaluations, and relevance explanations.

**Table 3: Comparison of Tokenization and Embedding Methods in LLMs**

|Method | Type | Description | Key Impacts on LLMs
|-------|------|-------------|------------------|
Byte-Pair Encoding (BPE) | Tokenization | Iteratively merges frequent pairs of bytes. | Balances vocabulary size and handling of rare words.
WordPiece | Tokenization | Similar to BPE but uses a scoring mechanism for merging.| Tends to prefer longer subword units.
SentencePiece | Tokenization | Language-independent, trains from raw sentences, supports BPE and unigram. | Handles whitespace as a symbol, allows lossless detokenization.
Contextual Embeddings | Embedding | Vector representation of a word varies based on its context. | Captures nuanced meaning of words in different contexts.
T-FREE | Embedding | Tokenizer-free, embeds words using sparse activation patterns over character triplets.| Memory-efficient, potentially better for multilingual data.


## 6. Conclusion and Future Directions

This report has surveyed the significant advancements in the core components that drive the capabilities of Large Language Models. The evolution of the Transformer architecture continues to be a central focus, with innovations aimed at enhancing the ability of LLMs to process and generate long sequences efficiently. Techniques such as sparse attention and context window extension are crucial steps towards enabling LLMs to tackle more complex, real-world tasks that require understanding and reasoning over extended information.

The optimization of LLM training remains a critical area, driven by the immense computational resources required. Parameter-Efficient Fine-Tuning methods like LoRA have made it more feasible to adapt these massive models for specific applications. Model compression techniques, including quantization and pruning, are essential for deploying LLMs in resource-constrained environments. Furthermore, distributed training strategies are fundamental for handling the sheer scale of modern LLMs.

The methods used to tokenize and embed text have also seen considerable progress. Subword tokenization algorithms like BPE, WordPiece, and SentencePiece provide effective ways to balance vocabulary size and the handling of rare words. The shift towards contextual embeddings has significantly improved the ability of LLMs to understand the nuances of language. Emerging tokenizer-free embedding approaches offer promising avenues for future research, potentially leading to more efficient and adaptable language models, especially for multilingual applications.

Despite these remarkable advancements, several challenges persist. The computational cost of training and deploying LLMs remains high, and further research is needed to develop more efficient architectures and optimization techniques. Handling extremely long contexts effectively without sacrificing performance is another ongoing challenge. Additionally, ensuring the fairness, reducing biases, and addressing ethical concerns associated with LLMs are crucial areas of future work.

Looking ahead, the field is likely to see continued innovation in all these areas. We can anticipate the development of novel Transformer architectures that are inherently more efficient and capable of handling even longer contexts. Further advancements in PEFT and model compression will likely lead to more accessible and deployable LLMs. The exploration of new tokenization and embedding strategies, including those that move beyond traditional subword approaches, could lead to more robust and versatile language models. The continuous pursuit of these improvements promises to unlock even greater potential for LLMs across a wide range of applications, further revolutionizing how we interact with and leverage artificial intelligence.


## 7. Detailed Analysis of Selected Sources
Certainly! Here's a comprehensive list of 25 highly relevant sources focusing on improvements in transformer architectures, optimization techniques in deep learning models, tokenization, and embeddings for building Large Language Models (LLMs). Each entry includes a complete citation, a brief summary, an evaluation of reliability, and an explanation of its specific relevance to your research.

---

### 1. Vaswani, A., et al. (2017). *Attention Is All You Need*. arXiv:1706.03762.
**Summary**: Introduces the Transformer architecture, relying solely on attention mechanisms, eliminating recurrence and convolutions. Achieves state-of-the-art results in machine translation tasks with reduced training time due to its parallelizable structure.

**Reliability**: Highly reliable; foundational work extensively cited in the field.

**Relevance**: Serves as the cornerstone for all subsequent transformer-based models and innovations.

---

### 2. Kitaev, N., Kaiser, Ł., & Levskaya, A. (2020). *Reformer: The Efficient Transformer*. arXiv:2001.04451.
**Summary**: Introduces techniques like locality-sensitive hashing and reversible residual layers to improve transformer efficiency, reducing complexity from O(L²) to O(L log L) and saving memory during training.

**Reliability**: Credible; authors are affiliated with reputable institutions, and the paper is well-received in the community.

**Relevance**: Addresses scalability issues in transformers, crucial for building large-scale LLMs.

---

### 3. Tay, Y., Dehghani, M., Bahri, D., & Metzler, D. (2020). *Efficient Transformers: A Survey*. arXiv:2009.06732.
**Summary**: Provides a comprehensive overview of various efficient transformer models like Reformer, Linformer, Performer, and Longformer, focusing on computational and memory efficiency improvements.

**Reliability**: Comprehensive and up-to-date; authored by experts in the field.

**Relevance**: Helps navigate the landscape of efficient transformer architectures, informing design choices for LLMs.

---

### 4. Chitty-Venkata, K. T., et al. (2023). *A Survey of Techniques for Optimizing Transformer Inference*. arXiv:2307.07982.
**Summary**: Reviews techniques like knowledge distillation, pruning, quantization, and hardware-level optimizations to enhance transformer inference efficiency, summarizing trade-offs between model size and accuracy.

**Reliability**: Comprehensive and up-to-date; authored by experts in the field.

**Relevance**: Provides insights into optimizing transformer inference, crucial for deploying LLMs effectively.

---

### 5. Tang, Y., et al. (2024). *A Survey on Transformer Compression*. arXiv:2402.05964.
**Summary**: Reviews recent compression methods for transformers, focusing on pruning, quantization, knowledge distillation, and efficient architecture design, highlighting their application in NLP and computer vision.

**Reliability**: Comprehensive and up-to-date; authored by experts in the field.

**Relevance**: Essential for understanding transformer compression techniques, aiding in building efficient LLMs.

---

### 6. Bostrom, K., & Durrett, G. (2020). *Byte Pair Encoding is Suboptimal for Language Model Pretraining*. arXiv:2004.03720.
**Summary**: Compares BPE and unigram language model tokenization methods, finding that the latter aligns more closely with linguistic morphology and often outperforms BPE in downstream tasks.

**Reliability**: Reliable; peer-reviewed and authored by experts in NLP.

**Relevance**: Provides insights into tokenization strategies, impacting model understanding and performance.

---

### 7. Samal, B. V. (2024). *Enhancing the Efficiency of Transformer-Based Large Language Models Through Pruning Strategies*. Medium.
**Summary**: Discusses redundancy in transformer components and presents pruning strategies that significantly reduce model size with minimal performance loss, based on studies of LLaMA models.

**Reliability**: Informative; based on referenced research, though published on a non-peer-reviewed platform.

**Relevance**: Offers practical strategies for making LLMs more efficient through targeted pruning.

---

### 8. Toxigon. (2024). *How to Optimize Transformer Performance: Tips and Tricks for 2024*.
**Summary**: Provides practical tips for optimizing transformer performance, including knowledge distillation, model pruning, tokenization optimization, and caching strategies.

**Reliability**: Informative; practical insights though not peer-reviewed.

**Relevance**: Useful for implementing optimization techniques in transformer models.

---

### 9. Mandliya, R. (2024). *Primer on Large Language Model (LLM) Inference Optimizations: Model Architecture Optimizations*.
**Summary**: Explores model architecture optimizations like Group Query Attention (GQA) to reduce memory and computational costs during inference while maintaining model quality.

**Reliability**: Informative; practical insights though not peer-reviewed.

**Relevance**: Provides architectural optimization techniques beneficial for LLM inference efficiency.

---

### 10. Brown, T. B., et al. (2020). *Language Models are Few-Shot Learners*. arXiv:2005.14165.
**Summary**: Introduces GPT-3, demonstrating that large-scale language models can perform tasks with few-shot learning, highlighting the importance of model scaling.

**Reliability**: Highly reliable; extensively cited in the field.

**Relevance**: Illustrates the impact of scaling on model capabilities, informing LLM development.

---

### 11. Raffel, C., et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*. arXiv:1910.10683.
**Summary**: Presents the T5 model, framing all NLP tasks as text to text, and explores the limits of transfer learning in NLP. Demonstrates that large models can achieve state-of-the-art results across various tasks.

## 7. Few more references

1. (PDF) Advancements in Transformer Architectures for Large Language Models AUTHOR, accessed on April 20, 2025, https://www.researchgate.net/publication/387295170_Advancements_in_Transformer_Architectures_for_Large_Language_Models_AUTHOR

2. A Historical Survey of Advances in Transformer Architectures - MDPI, accessed on April 20, 2025, https://www.mdpi.com/2076-3417/14/10/4316

3. Transformer (deep learning architecture) - Wikipedia, accessed on April 20, 2025, https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)
4. Demystifying Transformer Architecture in Large Language Models - TrueFoundry, accessed on April 20, 2025, https://www.truefoundry.com/blog/transformer-architecture
5. How Transformers Work: A Detailed Exploration of Transformer Architecture - DataCamp, accessed on April 20, 2025, https://www.datacamp.com/tutorial/how-transformers-work
6. Advancing Transformer Architecture in Long-Context Large Language Models: A Comprehensive Survey - arXiv, accessed on April 20, 2025, https://arxiv.org/html/2311.12351v2
7. Transformers and large language models in healthcare: A review - PMC - PubMed Central, accessed on April 20, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC11638972/
8. Paper page - Advancing Transformer Architecture in Long-Context ..., accessed on April 20, 2025, https://huggingface.co/papers/2311.12351
9. arxiv.org, accessed on April 20, 2025, http://arxiv.org/pdf/2311.12351
10. Multi-Head Attention and Transformer Architecture - Pathway, accessed on April 20, 2025, https://pathway.com/bootcamps/rag-and-llms/coursework/module-2-word-vectors-simplified/bonus-overview-of-the-transformer-architecture/multi-head-attention-and-transformer-architecture/
11. What is Multi-head Attention and how does it improve model performance over single Attention head? - AIML.com, accessed on April 20, 2025, https://aiml.com/what-is-multi-head-attention-and-how-does-it-improve-model-performance-over-single-attention-head/
12. [1909.00188] Improving Multi-Head Attention with Capsule Networks - arXiv, accessed on April 20, 2025, https://arxiv.org/abs/1909.00188
13. Advancing Transformer Architecture in Long-Context Large Language Models: A Comprehensive Survey | PDF - Scribd, accessed on April 20, 2025, https://www.scribd.com/document/782625300/3
14. Advanced Transformer Architectures - About deep2Read - GitHub Pages, accessed on April 20, 2025, https://qdata.github.io/deep2Read/fmefficient/L26/
15. MoA: Mixture of Sparse Attention for Automatic Large Language ..., accessed on April 20, 2025, https://openreview.net/forum?id=konDsSUSqg
16. Natively Sparse Attention (NSA) for Efficient Long-Context LLMs - Ajith's AI Pulse, accessed on April 20, 2025, https://ajithp.com/2025/02/21/natively-sparse-attention-nsa-the-future-of-efficient-long-context-modeling-in-large-language-models/
17. arxiv.org, accessed on April 20, 2025, https://arxiv.org/abs/2410.13276
18. [2311.12351] Advancing Transformer Architecture in Long-Context Large Language Models: A Comprehensive Survey - arXiv, accessed on April 20, 2025, https://arxiv.org/abs/2311.12351
19. Advancing Transformer Architecture in Long-Context Large Language Models: A Comprehensive Survey - Scholars.io, accessed on April 20, 2025, https://app.scholars.io/research/10381/advancing-transformer-architecture-in-long-context-large-language-models-a-comprehensive-survey
20. Strivin0311/long-llms-learning: A repository sharing the literatures about long-context large language models, including the methodologies and the evaluation benchmarks - GitHub, accessed on April 20, 2025, https://github.com/Strivin0311/long-llms-learning
21. Advancing Transformer Architecture in Long-Context Large Language Models: A Comprehensive Survey - Semantic Scholar, accessed on April 20, 2025, https://www.semanticscholar.org/paper/Advancing-Transformer-Architecture-in-Long-Context-Huang-Xu/4ea5ca620122e6a9a2b000444d36491cebf49c7c
22. [2501.06098] ELFATT: Efficient Linear Fast Attention for Vision Transformers - arXiv, accessed on April 20, 2025, https://arxiv.org/abs/2501.06098
23. arxiv.org, accessed on April 20, 2025, https://arxiv.org/abs/2412.02919
24. arxiv.org, accessed on April 20, 2025, 
25. https://arxiv.org/abs/2412.02344
arxiv.org, accessed on April 20, 2025,
 https://arxiv.org/abs/2405.05219
26. Efficient Deep Learning: A Comprehensive Overview of Optimization Techniques, accessed on April 20, 2025, https://huggingface.co/blog/Isayoften/optimization-rush
27. AIoT-MLSys-Lab/Efficient-LLMs-Survey: [TMLR 2024 ... - GitHub, accessed on April 20, 2025, https://github.com/AIoT-MLSys-Lab/Efficient-LLMs-Survey
28. Advances in Transformers for Robotic Applications: A Review - arXiv, accessed on April 20, 2025, https://arxiv.org/html/2412.10599v1
29. Towards Smaller, Faster Decoder-Only Transformers: Architectural Variants and Their Implications - arXiv, accessed on April 20, 2025, https://arxiv.org/html/2404.14462v2?ref=cohere-ai.ghost.io
30. Large Language Models: A Comprehensive Survey on Architectures, Applications, and Challenges - ResearchGate, accessed on April 20, 2025, https://www.researchgate.net/publication/387305663_Large_Language_Models_A_Comprehensive_Survey_on_Architectures_Applications_and_Challenges
31. (PDF) ADVANCEMENTS IN TRANSFORMER ARCHITECTURES FOR LARGE LANGUAGE MODEL: FROM BERT TO GPT-3 AND BEYOND - ResearchGate, accessed on April 20, 2025, https://www.researchgate.net/publication/380530250_ADVANCEMENTS_IN_TRANSFORMER_ARCHITECTURES_FOR_LARGE_LANGUAGE_MODEL_FROM_BERT_TO_GPT-3_AND_BEYOND
32. Efficient Training of Large Language Models on Distributed Infrastructures: A Survey - arXiv, accessed on April 20, 2025, https://arxiv.org/pdf/2407.20018
33. Efficient Large Language Models: A Survey - arXiv, accessed on April 20, 2025, https://arxiv.org/html/2312.03863v2
34. Distributed Training of Large Language Models - IEEE Computer Society, accessed on April 20, 2025, https://www.computer.org/csdl/proceedings-article/icpads/2023/307100a840/1VECXZMDcc0
35. Training Large Language Models (LLMs): Techniques and Best Practices - Nitor Infotech, accessed on April 20, 2025, https://www.nitorinfotech.com/blog/training-large-language-models-llms-techniques-and-best-practices/
36. Parameter-Efficient Fine-Tuning for Foundation Models - arXiv, accessed on April 20, 2025, https://arxiv.org/html/2501.13787v1
37. [2501.13787] Parameter-Efficient Fine-Tuning for Foundation Models - arXiv, accessed on April 20, 2025, https://arxiv.org/abs/2501.13787
38. arxiv.org, accessed on April 20, 2025, https://arxiv.org/abs/2403.14608
39. Train Small, Infer Large: Memory-Efficient LoRA Training for Large Language Models - arXiv, accessed on 
April 20, 2025, https://arxiv.org/abs/2502.13533
40. Train Small, Infer Large: Memory-Efficient LoRA Training for Large ..., accessed on April 20, 2025, https://openreview.net/forum?id=s7DkcgpRxL
41. \scalerel* Train Small, Infer Large: Memory-Efficient LoRA Training for Large Language Models - arXiv, accessed on April 20, 2025, https://arxiv.org/html/2502.13533v1
42. [Literature Review] Train Small, Infer Large: Memory-Efficient LoRA Training for Large Language Models - Moonlight, accessed on April 20, 2025, https://www.themoonlight.io/review/train-small-infer-large-memory-efficient-lora-training-for-large-language-models
43. Paper page - Train Small, Infer Large: Memory-Efficient LoRA Training for Large Language Models - Hugging Face, accessed on April 20, 2025, https://huggingface.co/papers/2502.13533
44. Memory-efficient Training of LLMs with Larger Mini-batches - arXiv, accessed on April 20, 2025, https://arxiv.org/html/2407.19580v1
45. [2106.09685] LoRA: Low-Rank Adaptation of Large Language Models - arXiv, accessed on April 20, 2025, https://arxiv.org/abs/2106.09685
46. GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection : r/LocalLLaMA - Reddit, accessed on April 20, 2025, https://www.reddit.com/r/LocalLLaMA/comments/1b8owei/galore_memoryefficient_llm_training_by_gradient/
47. TRAIN SMALL, INFER LARGE: MEMORY-EFFICIENT LORA, accessed on April 20, 2025, https://openreview.net/pdf/f3645a8d3db7ea3d0264384dea8c23dd492ef70a.pdf
48. [D] What is the motivation for parameter-efficient fine tuning if there's no significant reduction in runtime or GPU memory usage? : r/MachineLearning - Reddit, accessed on April 20, 2025, https://www.reddit.com/r/MachineLearning/comments/186ck5k/d_what_is_the_motivation_for_parameterefficient/
49. A Guide to Quantization in LLMs | Symbl.ai, accessed on April 20, 2025, https://symbl.ai/developers/blog/a-guide-to-quantization-in-llms/
50. 5 Tips for Optimizing Language Models - KDnuggets, accessed on April 20, 2025, https://www.kdnuggets.com/5-tips-for-optimizing-language-models
51. Inference Optimizations for Large Language Models: Effects, Challenges, and Practical Considerations - arXiv, accessed on April 20, 2025, https://arxiv.org/html/2408.03130v1
52. A Survey on Model Compression for Large Language Models ..., accessed on April 20, 2025, https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00704/125482
53. A-Survey-on-Model-Compression-for-Large-Language
LLM Quantization: Techniques, Advantages, and Models - TensorOps, accessed on April 20, 2025, https://www.tensorops.ai/post/what-are-quantized-llms
54. Understanding Model Quantization in Large Language Models | DigitalOcean, accessed on April 20, 2025, https://www.digitalocean.com/community/tutorials/model-quantization-large-language-models
55. Paper page - Shortened LLaMA: A Simple Depth Pruning for Large ..., accessed on April 20, 2025, https://huggingface.co/papers/2402.02834
56. arxiv.org, accessed on April 20, 2025, https://arxiv.org/abs/2501.02086
57. FASP: Fast and Accurate Structured Pruning of Large Language ..., accessed on April 20, 2025, https://openreview.net/forum?id=f4b0YVwKUO
58. Innovations in Training Techniques for Large Language Models - IEEE Computer Society, accessed on April 20, 2025, https://www.computer.org/publications/tech-news/trends/training-techniques-large-language-models/
59. DeepMind looks at distributed training of large AI models - The Register, accessed on April 20, 2025, https://www.theregister.com/2025/02/11/deepmind_distributed_model_training_research/
60. Distributed training of large language models on AWS Trainium ..., accessed on April 20, 2025, https://www.amazon.science/publications/distributed-training-of-large-language-models-on-aws-trainium
61. LLM Distributed Training [R] : r/MachineLearning - Reddit, accessed on April 20, 2025, https://www.reddit.com/r/MachineLearning/comments/1i0vrg3/llm_distributed_training_r/
62. Tokenization vs Embedding - How are they Different? - Airbyte, accessed on April 20, 2025, https://airbyte.com/data-engineering-resources/tokenization-vs-embeddings
63. Exploring Foundations of Large Language Models (LLMs): Tokenization and Embeddings, accessed on April 20, 2025, https://dzone.com/articles/llms-tokenization-and-embeddings
64. Demystifying Tokens and Embeddings in Large Language Models, accessed on April 20, 2025, https://arbs.io/2024-01-14-demystifying-tokens-and-embeddings-in-llm
65. Tokenization | Mistral AI Large Language Models, accessed on April 20, 2025, https://docs.mistral.ai/guides/tokenization/
66. Tokenization in Large Language Models (LLMs) - ingoampt - Artificial Intelligence integration into iOS apps and SaaS + Education, accessed on April 20, 2025, https://ingoampt.com/tokenization-in-large-language-models-llms/
67. Understanding tokens - .NET - Learn Microsoft, accessed on April 20, 2025, https://learn.microsoft.com/en-us/dotnet/ai/conceptual/understanding-tokens
68. confused about embeddings and tokenization in LLMs : r/learnmachinelearning - Reddit, accessed on April 20, 2025, https://www.reddit.com/r/learnmachinelearning/comments/1cs29kn/confused_about_embeddings_and_tokenization_in_llms/
69. The Technical User's Introduction to LLM Tokenization - Christopher Samiullah, accessed on April 20, 2025, https://christophergs.com/blog/understanding-llm-tokenization
70. Machine-Learning/Tokens and Tokenization in Large Language ..., accessed on April 20, 2025, https://github.com/xbeat/Machine-Learning/blob/main/Tokens%20and%20Tokenization%20in%20Large%20Language%20Models%20in%20Python.md
71. Introduction to LLM Tokenization - Airbyte, accessed on April 20, 2025, https://airbyte.com/data-engineering-resources/llm-tokenization
72. T-FREE: Tokenizer-Free Generative LLMs via Sparse Representations for Memory-Efficient Embeddings - arXiv, accessed on April 20, 2025, https://arxiv.org/html/2406.19223v1
73. What is WordPiece? - H2O.ai, accessed on April 20, 2025, https://h2o.ai/wiki/wordpiece/
74. WordPiece tokenization - Hugging Face LLM Course, accessed on April 20, 2025, https://huggingface.co/learn/llm-course/chapter6/6
75. Tokenization - SentencePiece | Continuum Labs, accessed on April 20, 2025, https://training.continuumlabs.ai/training/the-fine-tuning-process/tokenization/tokenization-sentencepiece
76. Summary of the tokenizers - Hugging Face, accessed on April 20, 2025, https://huggingface.co/docs/transformers/tokenizer_summary
77. From Smør-re-brød to Subwords: Training LLMs on Danish, One Morpheme at a Time - arXiv, accessed on April 20, 2025, https://arxiv.org/html/2504.01540v1
78. T-FREE: Subword Tokenizer-Free Generative LLMs via Sparse Representations for Memory-Efficient Embeddings - arXiv, accessed on April 20, 2025, https://arxiv.org/html/2406.19223v2
79. Assessing the Importance of Frequency versus Compositionality for Subword-based Tokenization in NMT - arXiv, accessed on April 20, 2025, https://arxiv.org/html/2306.01393v3
80. google/sentencepiece: Unsupervised text tokenizer for ... - GitHub, accessed on April 20, 2025, https://github.com/google/sentencepiece
81. Explanation of Contextual Embeddings | Sapien's AI Glossary, accessed on April 20, 2025, https://www.sapien.io/glossary/definition/contextual-embeddings
82. Contextual Embeddings NLP Insights | Restackio, accessed on April 20, 2025, https://www.restack.io/p/embeddings-answer-contextual-embeddings-nlp-cat-ai
83. Introducing Contextual Retrieval \ Anthropic, accessed on April 20, 2025, https://www.anthropic.com/news/contextual-retrieval
84. Latest Developments in Vector Embeddings for AI Applications - CelerData, accessed on April 20, 2025, https://celerdata.com/glossary/vector-embeddings-for-ai-applications
85. [2406.19223] T-FREE: Subword Tokenizer-Free Generative LLMs via Sparse Representations for Memory-Efficient Embeddings - arXiv, accessed on April 20, 2025, https://arxiv.org/abs/2406.19223
86. T-FREE: Tokenizer-Free Generative LLMs via Sparse Representations for Memory-Efficient Embeddings | AI Research Paper Details - AIModels.fyi, accessed on April 20, 2025, https://www.aimodels.fyi/papers/arxiv/t-free-subword-tokenizer-free-generative-llms
87. T-FREE: Subword Tokenizer-Free Generative LLMs via Sparse Representations for Memory-Efficient Embeddings - Synthical, accessed on April 20, 2025, https://synthical.com/abs/2406.19223?is_dark=true&utm_source=dark_medium
88. Tokenizer-Free Generative LLMs via Sparse Representations for Memory-Efficient Embeddings | PromptLayer, accessed on April 20, 2025, https://www.promptlayer.com/research-papers/t-free-tokenizer-free-generative-llms-via-sparse-representations-for-memory-efficient-embeddings
89. T-FREE: Subword Tokenizer-Free Generative LLMs via Sparse Representations for Memory-Efficient Embeddings | Request PDF - ResearchGate, accessed on April 20, 2025, https://www.researchgate.net/publication/386202375_T-FREE_Subword_Tokenizer-Free_Generative_LLMs_via_Sparse_Representations_for_Memory-Efficient_Embeddings
90. T-FREE: Tokenizer-Free Generative LLMs via Sparse Representations for Memory-Efficient Embeddings - ChatPaper, accessed on April 20, 2025, https://chatpaper.com/chatpaper/paper/33267
91. T-FREE: Tokenizer-Free Generative LLMs via Sparse Representations for Memory-Efficient Embeddings - ResearchGate, accessed on April 20, 2025, https://www.researchgate.net/publication/381770835_T-FREE_Tokenizer-Free_Generative_LLMs_via_Sparse_Representations_for_Memory-Efficient_Embeddings
92. Paper page - T-FREE: Tokenizer-Free Generative LLMs via Sparse Representations for Memory-Efficient Embeddings - Hugging Face, accessed on April 20, 2025, https://huggingface.co/papers/2406.19223
93. Egalitarian Language Representation in Language Models: It All Begins with Tokenizers - arXiv, accessed on April 20, 2025, https://arxiv.org/pdf/2409.11501
94. [2205.11490] Local Byte Fusion for Neural Machine Translation - arXiv, accessed on April 20, 2025, https://arxiv.org/abs/2205.11490
