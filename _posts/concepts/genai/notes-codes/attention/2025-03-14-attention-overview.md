---
layout : mermaid
type : concept
title : "Attention - An overview"
---
## The main idea of attention
The evolution of NLP has been as follows - 

|**Date** | **Model** | **Description** |
|---------|-----------|------------------|
1966 | Eliza | First chatbot |
1980s | RNN | last hidden state would have the compressed past context |
1997 | LSTM | The long term cell state will give equal importance to all the past hidden states|
2014 | Bahdanau Attention | **Attention** + RNN (RNN was used for both encoder and decoder) |
2017 | **Attention** + **Transformer** | **RNN was removed** and Transformer was introduced. But it had again both **encoder and decoder.** **BERT** is an example for encoder + decoder architecture. [BERT paper](https://arxiv.org/abs/1810.04805) |
2018 | **Attention** + **General Purpose Transformer** | **No Encoder. Only decoder**. Encoder as a separate block was removed from original transformer|

### RNN - Recurrent Neural Networks
**RNNs**: [Recurrent Neural Networks (RNNs)](https://samratkar.github.io/2025/02/01/RNN-theo.html) were the first to be used for sequence-to-sequence tasks, but they struggled with long-range dependencies.
![ ](/images/genai/rnn-unrolled.svg)
<details>
  <summary>Click to expand</summary>
  <img src="/images/genai/rnn-block.svg" alt="Description" width="300">
</details>

### LSTM - Long Short-Term Memory

**LSTM**: [Long Short-Term Memory (LSTM)](https://samratkar.github.io/2025/02/15/LSTM-theory.html) networks improved upon RNNs by introducing memory cells and gates, allowing them to capture long-range dependencies more effectively. But still the problem was that all the previous hidden states were used to generate the next hidden state, which made it difficult to focus on specific parts of the input sequence.
<details>
   <summary>Click to expand</summary>
      <video width="640" height="360" controls>
         <source src="/images/genai/lstm-visualization.mp4" type="video/mp4">
         Your browser does not support the video tag.
      </video>
</details>

### Attention Mechanism
#### Bhadanau Attention 
[Neural machine translation by jointly learning to align and translate](https://arxiv.org/abs/1409.0473) (Bahdanau et al., 2014) is a seminal paper that introduced the concept of attention in neural networks. The paper proposed a method for aligning and translating sequences of variable lengths, which was particularly useful for machine translation tasks.
The key idea behind Bahdanau attention is to compute a context vector that summarizes the relevant information from the input sequence for each output time step. This is done by using a neural network to learn an alignment score for each input time step, which indicates how much attention should be paid to that time step when generating the output.

**Attention**: The attention mechanism was introduced to allow models to focus on **specific parts of the input sequence** when generating each output token. This was particularly useful for tasks like machine translation, where the alignment between input and output sequences is crucial.

1. **Attention** is about selectively accessing parts of input sequence with different weights, rather than using the entire input sequence equally. This allows the model to focus on relevant parts of the input when generating each output token.

For example to translate the sentence "I will eat" to French "Je vais manger", the model can focus on the word "I" when generating the word "Je", rather than considering all the words in the input sequence equally. That is the challenge, how to select which past tokens are how much important quantitatively for predicting the next token. This is done by what is known as **attention scores** for every hidden state. 

2. The relevant past tokens might or might not be 1:1 corresponding with the next token. For example, in the figure below the correspondence is as mentioned below. In the sentence formation the French word for zone is written before that of the economic. 
   - French - zone economique europeenne
   - English - economic zone area
![](/images/genai/french.png)
Note that in the figure above, if there was a 1:1 correspondence only the diagonal elements would be highlighted. But there are many cells which are off diagonal that are highlighted.
Attention mechanism does two things as we see here - 
   - translates the actual words.
   - identifies the right order of the words. And this order can change from that of the input sequence, based on the construct of the language. 

3. The **Bhadanav Attention** mechanism coupled RNN + Attention mechanism. It was not earlier imagined to be used for text summarization as well. It was primarily for text translation. 
In the figure below, the BLEU score is used to measure the quality of the translation. 
**RNNsearch-50** and **RNNsearch-30** are models that include attention mechanism. Others do not have the attention mechanism. 
![](/images/genai/bleuscore.png) 
4. 


### Self attention with trainable weights. 

1. In the above case where the Bhadanav attention was used, the attention scores were created to be able to translate the words of one language to the other. 
Self attention is the case where same concept is being used for text generation. 

2. In self attention we look into only one sequence of words. This is the main mechanism that is used to predict the next token. We take one token, and check how much attention score will that token have for all the other tokens in the sequence.
For example, take the token - "next". We need to find the attention scores of "next" with all the other tokens in the sequence, both front and back.
![](/images/genai/query-key-atten-score.svg)

The token that we are focusing currently is known as **query**. The other tokens are known as **keys**. The attention scores are calculated between the query and the keys.

#### How to compute the attention scores?
1. **Dot product** - 
By taking the dot product we can get an intuitive attention score which will demonstrate the closeness of one query with the other keys. But the problem is that if their are two keys, whose **magnitude is same but their directions are different**, the value of dot products will be same. Like for example if we have the keys as follows, the magnitude of the key "dog" is the same as that of the key "ball". In that case if we have a value say, "it", the dot products will be same for both the cases.  
![](/images/genai/query-key-dot-prod.svg)

2. **Train a neural network**
Start with a random initialized matrix for query and keys. And then give the right attention scores as inputs and train a neural network to learn the weights of the matrix. 
Assume the embedding size is 3. And assume two weight matrices named ***Query*** and ***Key***. The weight matrices are initialized randomly.
$$
Query \ weight = W_q = 
\begin{bmatrix} 
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1 \\
\end{bmatrix}
$$ 

$$
Key \ weight = W_k =
\begin{bmatrix}
1 & 0 & 0 \\
0 & 2 & 0 \\
0 & 0 & 1 \\
\end{bmatrix}
$$

$$
x = Input \ matrix = 
\begin{bmatrix}
0.9 & 0.1 & 0.1 \\
0.1 & 0.9 & 0.1 \\
0.5 & 0.5 & 0.1 \\
\end{bmatrix} 
$$

$$
Query = W_q \times Input \ matrix = 
\begin{bmatrix}
0.9 & 0.1 & 0.1 \\
0.1 & 0.9 & 0.1 \\ 
0.5 & 0.5 & 0.1 \\
\end{bmatrix} 
$$

$$
Key = W_k \times Input \ matrix =
\begin{bmatrix}
0.9 & 0.1 & 0.1 \\
0.2 & 1.8 & 0.2 \\
0.5 & 0.5 & 0.1 \\
\end{bmatrix}
$$

$$
Attention \ score = softmax(Query \cdot Key^T) =
\begin{bmatrix}
0.4231 & 0.2697 & 0.3072 \\ 
0.1487 & 0.6466 & 0.2047 \\
0.2728 & 0.4543 & 0.2728 \\
\end{bmatrix}
$$

##### Input matrices 
|Word|x = Input matrix|Query = $W_q \cdot x$ |Key = $W_k \cdot x$|
|---|---|---|---|
|the|$\begin{bmatrix} 0.9 & 0.1 & 0.1 \end{bmatrix}$ |$\begin{bmatrix} 0.9 & 0.1 & 0.1 \end{bmatrix}$|$\begin{bmatrix} 0.9 & 0.1 & 0.1 \end{bmatrix}$|
|next|$\begin{bmatrix} 0.1 & 0.9 & 0.1 \end{bmatrix}$|$\begin{bmatrix} 0.1 & 0.9 & 0.1 \end{bmatrix}$|$\begin{bmatrix} 0.2 & 1.8 & 0.2 \end{bmatrix}$|
|is|$\begin{bmatrix} 0.5 & 0.5 & 0.1 \end{bmatrix}$|$\begin{bmatrix} 0.5 & 0.5 & 0.1 \end{bmatrix}$|$\begin{bmatrix} 0.5 & 0.5 & 0.1 \end{bmatrix}$|

##### Attention scores
|Word|word with Max Attention score| score for the | score for next | sore for is |
|---|---|---|---|---|
|the|the|**0.4231**|0.2697|0.3072|
|next|next|0.1487|**0.6466**|0.2047|
|is|next|0.2728|**0.4543**|0.2728| 


## References

1. [What is the big deal about attention - vizuara substack](https://substack.com/inbox/post/158574020)
2. [Attention is all you need - arxiv](https://arxiv.org/abs/1706.03762)
3. Attention mechanism 1 hour video 
<iframe width="560" height="315" src="https://www.youtube.com/embed/K45ze9Yd5UE?si=FAJ3YPArq9Wu-uQ3" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>