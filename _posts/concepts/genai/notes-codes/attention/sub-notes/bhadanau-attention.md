---
layout: mermaid
title: Bahdanau Attention
---
[Neural machine translation by jointly learning to align and translate](https://arxiv.org/abs/1409.0473) (Bahdanau et al., 2014) is a seminal paper that introduced the concept of attention in neural networks. The paper proposed a method for aligning and translating sequences of variable lengths, which was particularly useful for machine translation tasks.
The key idea behind Bahdanau attention is to compute a context vector that summarizes the relevant information from the input sequence for each output time step. This is done by using a neural network to learn an alignment score for each input time step, which indicates how much attention should be paid to that time step when generating the output.

**Attention**: The attention mechanism was introduced to allow models to focus on **specific parts of the input sequence** when generating each output token. This was particularly useful for tasks like machine translation, where the alignment between input and output sequences is crucial.

**Attention** is about selectively accessing parts of input sequence with different weights, rather than using the entire input sequence equally. This allows the model to focus on relevant parts of the input when generating each output token.

For example to translate the sentence "I will eat" to French "Je vais manger", the model can focus on the word "I" when generating the word "Je", rather than considering all the words in the input sequence equally. That is the challenge, how to select which past tokens are how much important quantitatively for predicting the next token. This is done by what is known as **attention scores** for every hidden state. 

1. The relevant past tokens might or might not be 1:1 corresponding with the next token. For example, in the figure below the correspondence is as mentioned below. In the sentence formation the French word for zone is written before that of the economic. 
   - French - zone economique europeenne
   - English - economic zone area
![](/images/genai/french.png)
Note that in the figure above, if there was a 1:1 correspondence only the diagonal elements would be highlighted. But there are many cells which are off diagonal that are highlighted.
Attention mechanism does two things as we see here - 
   - translates the actual words.
   - identifies the right order of the words. And this order can change from that of the input sequence, based on the construct of the language. 

2. The **Bhadanav Attention** mechanism coupled RNN + Attention mechanism. It was not earlier imagined to be used for text summarization as well. It was primarily for text translation. 
In the figure below, the BLEU score is used to measure the quality of the translation. 
**RNNsearch-50** and **RNNsearch-30** are models that include attention mechanism. Others do not have the attention mechanism. 
![](/images/genai/bleuscore.png) 
The LSTMs without Attention mechanism would start failing as the length of the input sentence keeps growing, as shown in the figure above. 

3. Bhadanav attention mechanism was typically used for text translation. So, the attention was from the source language to the target language. The attention scores were calculated between the source and target language.

![](/images/genai/bhadanav-attention.svg)
