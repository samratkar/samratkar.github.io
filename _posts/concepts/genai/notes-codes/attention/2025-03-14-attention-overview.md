---
layout : mermaid
type : concept
title : "Attention - An overview"
---
## The main idea of attention
The evolution of NLP has been as follows - 
1. **RNNs**: Recurrent Neural Networks (RNNs) were the first to be used for sequence-to-sequence tasks, but they struggled with long-range dependencies.
![ ](/images/genai/rnn-unrolled.svg)
<details>
  <summary> Click to expand for details </summary>
![ ](/images/genai/rnn-block.svg)
![ ](/images/genai/rnn-details.svg)
</details>

2. **LSTM**: Long Short-Term Memory (LSTM) networks improved upon RNNs by introducing memory cells and gates, allowing them to capture long-range dependencies more effectively. But still the problem was that all the previous hidden states were used to generate the next hidden state, which made it difficult to focus on specific parts of the input sequence.
3. **Attention**: The attention mechanism was introduced to allow models to focus on specific parts of the input sequence when generating each output token. This was particularly useful for tasks like machine translation, where the alignment between input and output sequences is crucial.
## Bahdanau Attention

[Neural machine translation by jointly learning to align and translate](https://arxiv.org/abs/1409.0473) (Bahdanau et al., 2014) is a seminal paper that introduced the concept of attention in neural networks. The paper proposed a method for aligning and translating sequences of variable lengths, which was particularly useful for machine translation tasks.
The key idea behind Bahdanau attention is to compute a context vector that summarizes the relevant information from the input sequence for each output time step. This is done by using a neural network to learn an alignment score for each input time step, which indicates how much attention should be paid to that time step when generating the output.
The attention mechanism is implemented as follows:
1. **Alignment Scores**: For each output time step, compute alignment scores for all input time steps using a feedforward neural network. The alignment scores are computed as:
   $$ e_{ij} = v_a^T \tanh(W_a h_i + U_a s_j) $$
   where \( h_i \) is the hidden state of the encoder at time step \( i \), \( s_j \) is the hidden state of the decoder at time step \( j \), and \( v_a, W_a, U_a \) are learnable parameters.
2. **Attention Weights**: Normalize the alignment scores using a softmax function to obtain attention weights:
3. $$ \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})} $$
   where \( T_x \) is the length of the input sequence.
4. **Context Vector**: Compute the context vector as a weighted sum of the encoder hidden states:
5. $$ c_j = \sum_{i=1}^{T_x} \alpha_{ij} h_i $$
6. **Output**: The context vector \( c_j \) is then concatenated with the decoder hidden state \( s_j \) and passed through a feedforward neural network to produce the final output:
7. $$ y_j = \text{softmax}(W_y [s_j; c_j]) $$
   where \( W_y \) is a learnable parameter and \( [s_j; c_j] \) denotes concatenation.

## References

1. [What is the big deal about attention - vizuara substack](https://substack.com/inbox/post/158574020)
2. [Attention is all you need - arxiv](https://arxiv.org/abs/1706.03762)
3. Attention mechanism 1 hour video 
<iframe width="560" height="315" src="https://www.youtube.com/embed/K45ze9Yd5UE?si=FAJ3YPArq9Wu-uQ3" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>