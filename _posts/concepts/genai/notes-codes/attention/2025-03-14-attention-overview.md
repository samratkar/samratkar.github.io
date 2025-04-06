---
layout : mermaid
type : concept
title : "Attention - An overview"
---

## The main idea of attention
The evolution of NLP has been as follows -
![](/images/genai/chronology.svg) 

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
![](/_includes/genai/attention/sub-notes/bhadanav-attention.md)

{% capture my_markdown %}
{% include genai/attention/sub-notes/bhadanav-attention.md %}
{% endcapture %}
{{ my_markdown | markdownify }}


### Self attention with trainable weights

![](/images/genai/self-attention-overview.svg)

1. In the above case where the Bhadanav attention was used, the attention scores were created to be able to translate the words of one language to the other. 
Self attention is the case where same concept is being used for text generation. So, source and targets are the same input text stream. This is known as **Auto-regression**. The model is trained to predict the next token in the sequence based on the previous tokens. The attention scores are calculated between the tokens in the same sequence.
When the attention mechanism directed such that the source and target is the same sequence of words, it is known as **Self Attention**.

2. In self attention we look into only one sequence of words. This is the main mechanism that is used to predict the next token. We take one token, and check how much attention score will that token have for all the other tokens in the sequence.
For example, take the token - "next". We need to find the attention scores of "next" with all the other tokens in the sequence, both front and back.
![](/images/genai/query-key-atten-score.svg)

The token that we are focusing currently is known as **query**. The other tokens are known as **keys**. The attention scores are calculated between the query and the keys.



### How to compute the attention scores?
#### Dot product
By taking the dot product we can get an intuitive attention score which will demonstrate the closeness of one query with the other keys. But the problem is that if their are two keys, whose **magnitude is same but their directions are different**, the value of dot products will be same. Like for example if we have the keys as follows, the magnitude of the key "dog" is the same as that of the key "ball". In that case if we have a value say, "it", the dot products will be same for both the cases.  
![](/images/genai/query-key-dot-prod.svg)

#### Train a neural network
Start with a random initialized matrix for query and keys. And then give the right attention scores as inputs and train a neural network to learn the weights of the matrix. 
Assume the embedding size is 3. And assume two weight matrices named ***Query*** and ***Key***. The weight matrices are initialized randomly.

##### Query 
The current word or token has a "question" about its context. **"What should I pay attention to?"**

##### Key 
Every token provides a "key", acting as a label or signal to answer queries from other tokens. It indicates : **"Here is what information I can provide"**

##### Value
Once a match (query - key) is found, this is the actual information provided. It says, **"Here is the meaning or content you will get if you attend to me"**

### Self Attention : Workflow for creation of context vector
#### The big picture : End to end Workflow of Self Attention 
![](/images/genai/self-attention-big-pic.svg)

#### Details of single head self attention
![](/images/genai/self-attention.svg)

### Details of the workflow of self attention
![](/images/genai/self-attention-det.svg)

## Implementation of Self attention
### Step 1 : Start with the input embedding matrix 
Embedding dimension or input dimension : 8
Context length : 5
```python
import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89, 0.17, 0.23, 0.19, 0.38, 0.44], # The     (x^1)
   [0.55, 0.87, 0.66, 0.51, 0.49, 0.3, 0.2, 0.1], # next  (x^2)
   [0.57, 0.85, 0.64, 0.8, 0.1, 0.4, 0.21, 0.39], # day  (x^3)
   [0.22, 0.58, 0.33, 0.4, 0.4, 0.4, 0.1, 0.3], # is     (x^4)
   [0.77, 0.25, 0.10, 0.1, 0.9, 0.3, 0.3, 0.2]] # bright     (x^5)
)
```
#### Illustration of the code 
![](/images/genai/iembed.svg)
### Step 2 : Set the input and output dimensions
```python
d_in = inputs.shape[1]
d_out = 4
print(x_2)
print(d_in)
print(d_out)
Output >>
tensor([0.5500, 0.8700, 0.6600, 0.5100, 0.4900, 0.3000, 0.2000, 0.1000])
8
4
```
#### Illustration of the code
![](/images/genai/iodim.svg)
### Step 3 : Initialize the weight matrices for query, key and value. 
###### Wq, Wk, Wv are just trainable weight matrices. They have no relationships with the input embeddings of the text sequence.

```python
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

print(W_query)
Output >> 
Parameter containing:
### Wq is as follows. 
### It is just a trainable weight matrix. It has no relationship with the input embeddings of the text sequence.
tensor([[0.2961, 0.5166, 0.2517, 0.6886],
        [0.0740, 0.8665, 0.1366, 0.1025],
        [0.1841, 0.7264, 0.3153, 0.6871],
        [0.0756, 0.1966, 0.3164, 0.4017],
        [0.1186, 0.8274, 0.3821, 0.6605],
        [0.8536, 0.5932, 0.6367, 0.9826],
        [0.2745, 0.6584, 0.2775, 0.8573],
        [0.8993, 0.0390, 0.9268, 0.7388]])

print(W_key)
Output >>
Parameter containing:
tensor([[0.7179, 0.7058, 0.9156, 0.4340],
        [0.0772, 0.3565, 0.1479, 0.5331],
        [0.4066, 0.2318, 0.4545, 0.9737],
        [0.4606, 0.5159, 0.4220, 0.5786],
        [0.9455, 0.8057, 0.6775, 0.6087],
        [0.6179, 0.6932, 0.4354, 0.0353],
        [0.1908, 0.9268, 0.5299, 0.0950],
        [0.5789, 0.9131, 0.0275, 0.1634]])

print(W_value)
Output >>
Parameter containing:
tensor([[0.3009, 0.5201, 0.3834, 0.4451],
        [0.0126, 0.7341, 0.9389, 0.8056],
        [0.1459, 0.0969, 0.7076, 0.5112],
        [0.7050, 0.0114, 0.4702, 0.8526],
        [0.7320, 0.5183, 0.5983, 0.4527],
        [0.2251, 0.3111, 0.1955, 0.9153],
        [0.7751, 0.6749, 0.1166, 0.8858],
        [0.6568, 0.8459, 0.3033, 0.6060]])
```
#### Illustration of the code
![](/images/genai/wqwkwv.svg)
### Step 4: Computation of Query, Key and Value matrices
```python
keys = inputs @ W_key
values = inputs @ W_value
queries = inputs @ W_query
print("keys.shape:", keys.shape)
Output >> 
keys.shape: torch.Size([5, 4])

print("values.shape:", values.shape)
Output >>
values.shape: torch.Size([5, 4])

print("queries.shape:", queries.shape)
Output >>
queries.shape: torch.Size([5, 4])
``` 
**Note the dimensions of the keys, values and queries.** We have moved from din to dout.
#### Illustration of the code
![](/images/genai/qkv.svg)

#### finding the Q, K, V matrices for "next" token
**Note :** - The entire Wq is multiplied by the input vector to determine the Q for the input vector for "next".
```python
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)
```
###### Each element of Q, K, V is a linear combination of the input embedding, done by scaling up using the weight matrices 
![](/images/genai/matmul.svg)


### Step 5: Compute the attention scores
```python
attn_scores = queries @ keys.T 
print(attn_scores)
Output >>
tensor([[ 8.7252, 10.8803, 11.0007,  7.7678,  9.7598],
        [ 9.7351, 12.0370, 12.2923,  8.7149, 10.9628],
        [10.4691, 12.9987, 13.1878,  9.3438, 11.8256],
        [ 7.7531,  9.6199,  9.7608,  6.9217,  8.7864],
        [ 8.8185, 10.9612, 11.1314,  7.8699,  9.8633]])
```
#### computing the attention scores for next token with next token w22
```python
keys_2 = keys[1] 
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)
Output >>
tensor(12.0370)
```
#### Generalizing this to get the attention scores for all keys for token "next"
```python
attn_scores_2 = query_2 @ keys.T # All attention scores for given query
print(attn_scores_2)
Output >>
tensor([ 9.7351, 12.0370, 12.2923,  8.7149, 10.9628])
```

### Step 6: Compute the attention weights
#### Why divide by sqrt(d_k)?
##### Reason 1: For stability in learning

The softmax function is sensitive to the magnitudes of its inputs. When the inputs are large, the differences between the exponential values of each input become much more pronounced. This causes the softmax output to become "peaky," where the highest value receives almost all the probability mass, and the rest receive very little.

In attention mechanisms, particularly in transformers, if the dot products between query and key vectors become too large (like multiplying by 8 in this example), the attention scores can become very large. This results in a very sharp softmax distribution, making the model overly confident in one particular "key." Such sharp distributions can make learning unstable

###### Illustration on how softmax() increases the magnitude of the result for higher values
```python
import torch

# Define the tensor
tensor = torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])

# Apply softmax without scaling
softmax_result = torch.softmax(tensor, dim=-1)
print("Softmax without scaling:", softmax_result)

# Multiply the tensor by 8 and then apply softmax
scaled_tensor = tensor * 8
softmax_scaled_result = torch.softmax(scaled_tensor, dim=-1)
print("Softmax after scaling (tensor * 8):", softmax_scaled_result)

Output >>
Softmax without scaling: tensor([0.1925, 0.1426, 0.2351, 0.1426, 0.2872])
Softmax after scaling (tensor * 8): tensor([0.0326, 0.0030, 0.1615, 0.0030, 0.8000])
```
**0.8000 is higher than other values in the order of 100x.**

##### Reason 2 : To make the variance of the dot product stable

The dot product of  Q and K increases the variance because multiplying two random numbers increases the variance.
The increase in variance grows with the dimension.
Dividing by sqrt (dimension) keeps the variance close to 1

###### Illustration on how the variance increases with the dimension of the vector
Imagine you’re rolling dice. Consider two cases:
###### Case 1: Rolling one standard die (1–6):

1. **The average (mean)** : 3.5
2. **The variance** : 2.9 (calculated as the average of the squared differences from the mean). The variance is relatively small.

###### Case 2: Rolling and summing 100 dice:

1. **The average (mean)** : $100 \times 3.5 = 350$ 
2. **The variance** : $100 \times 2.9 = 290$. The variance significantly grows.

Now, outcomes fluctuate widely (e.g., you might get sums like 320, 350, or 380). The distribution spreads out drastically. Outcomes become unpredictable.

###### Dot Product without normalization:

Think of dimensions as "dice." Increasing the number of dimensions is like rolling more dice and summing results.
Each dimension (dice) contributes some variance. As dimensions grow, variance accumulates.

**Result: Dot products (before softmax) become either extremely large or small, making attention weights unstable and erratic.**

###### Dot Product with normalization (dividing by $\sqrt{d}$):

This effectively scales down the variance, ensuring the summed results remain stable.
It’s like taking the average roll per dice rather than summing them up, stabilizing your expected outcomes.
**Result: Attention weights become more stable, predictable, and informative, enabling the model to learn effectively.**


```python
# dim = -1 implies the operation is done column-wise.
attn_weights_final = torch.softmax(attn_scores / d_k**0.5, dim=-1)
print(attn_weights_final)
row_sums = attn_weights_final.sum(dim=1)
print("\nSum of Each Row:")
print(row_sums)
Output >> 
tensor([[0.1069, 0.3140, 0.3335, 0.0662, 0.1793],
        [0.0980, 0.3099, 0.3521, 0.0589, 0.1811],
        [0.0911, 0.3227, 0.3547, 0.0519, 0.1795],
        [0.1162, 0.2954, 0.3170, 0.0767, 0.1947],
        [0.1063, 0.3103, 0.3379, 0.0662, 0.1793]])

Sum of Each Row:
tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000])
```
### Step 7: Compute the context vector
```python
context_vec = attn_weights_final @ values
print(context_vec)
Output >>
tensor([[1.3246, 1.5236, 1.8652, 2.3285],
        [1.3301, 1.5304, 1.8753, 2.3433],
        [1.3325, 1.5353, 1.8866, 2.3537],
        [1.3211, 1.5153, 1.8390, 2.3002],
        [1.3253, 1.5242, 1.8657, 2.3304]])
```
## Modular implementation of self attention 
### Version 1
```python
import torch.nn as nn

class SelfAttention_v1(nn.Module):

#### Step 1 : Initialize the weight matrices for query, key and value.
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

#### Step 2 : Compute the Query, Key and Value matrices
        self.d_in = d_in
        self.d_out = d_out
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

#### Step 3 : Compute the attention scores
        attn_scores = queries @ keys.T # omega

#### Step 4 : Compute the attention weights (normalization and then softmax)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec

```

#### Example usage of computing the context matrix for any input embedding matrix
```python
# Initializing the self attention with the input dimension (embedding) and the output dimension (context)
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
# Directly computing the Context matrix in one line, from any input embedding matrix
print(sa_v1(inputs))
Output >>
tensor([[1.3246, 1.5236, 1.8652, 2.3285],
        [1.3301, 1.5304, 1.8753, 2.3433],
        [1.3325, 1.5353, 1.8866, 2.3537],
        [1.3211, 1.5153, 1.8390, 2.3002],
        [1.3253, 1.5242, 1.8657, 2.3304]], grad_fn=<MmBackward0>)
```
### Version 2 : Using nn.Linear to initialize the weight matrices for query, key and value. This makes the training stabler.

1. The random seeds are chosen by the linear neural network
2. Also Q, K, V are just nn.Linear(din, dout, bias=false). This computes the output layer of the nn in a way it does the sum of products anyways. So, matrix multiplication is not done explicitly but **$nn.Linear()$** is used to determine Q, K, V matrices.
3. Here ***linear neural network*** means that output is a linear combination of the input. The output is a weighted sum of the inputs. The weights are the parameters of the linear layer, and they are learned during training. The bias term is also a parameter that is learned during training. The output is computed as follows:
    **$output = (W \cdot input + b)$** where W is the weight matrix and b is the bias vector.
4. Here the **non-linear** layer of the neural network is not used. That is implemente by passing the output of the linear layer into the **activation function**.

```python
class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(789)

inputs = torch.tensor(
  [[0.43, 0.15, 0.89, 0.17, 0.23, 0.19, 0.38, 0.44], # The     (x^1)
   [0.55, 0.87, 0.66, 0.51, 0.49, 0.3, 0.2, 0.1], # next  (x^2)
   [0.57, 0.85, 0.64, 0.8, 0.1, 0.4, 0.21, 0.39], # day  (x^3)
   [0.22, 0.58, 0.33, 0.4, 0.4, 0.4, 0.1, 0.3], # is     (x^4)
   [0.77, 0.25, 0.10, 0.1, 0.9, 0.3, 0.3, 0.2]] # bright     (x^5)
)

d_in = 8
d_out = 4

sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))
Output >>
tensor([[ 0.0174,  0.0553, -0.1093,  0.1026],
        [ 0.0175,  0.0556, -0.1089,  0.1024],
        [ 0.0175,  0.0559, -0.1087,  0.1022],
        [ 0.0179,  0.0544, -0.1091,  0.1028],
        [ 0.0172,  0.0543, -0.1105,  0.1032]], grad_fn=<MmBackward0>)
```
### Step 8 : Causal Attention
#### The attention weights computed above in the self attention method are for all tokens.
But the attentions weights for future queries are not required, when computing the attention scores for the current query as per the auto-regressive self attention. 
![](/images/genai/causal-attention.svg)
![](/images/genai/infity-mask.png)

#### Minus Infinity Mask 
<div class="mermaid">
graph TD;
    A(Attention scores) --> B(Negative Infinity Masking)
    B --> C(Causal Attention score)
    C --multiply--> D(Value matrix)
    D --> E(Context Vector)
</div>

## Uni-direction attention with Dropout
There are some neurons which do no to any work in a neural network. **Dropout** is added to avoid this lazy neuron problem. This basically randomly offs some neurons in the network. This is done to avoid overfitting. The dropout rate is typically set between 0.1 and 0.5, meaning that 10% to 50% of the neurons are randomly turned off during training.
This forces the network to learn more robust features and prevents it from relying too heavily on any single neuron.
In a particular iteration the grayed cells are the neurons that are turned off. In some other iteration other neurons will be randomly turned off.
![](/images/genai/dropout.png)


## Multi-head attention
In the single head causal self attention explained above we have only one attention head. This means that the model is only able to focus on one perspective of the input sequence at a time. This can be limiting, especially for complex tasks where multiple parts of the input sequence may be relevant at the same time.

For example, consider this sentence : **The artist painted the portrait of a woman with a brush**


## Layer normalization
Layer normalization is done on input embedding. Information is not lost because it is relative scaling. 
Layer normalization is needed to handle the following - 
### Vanishing or exploding gradient problem
Gradients need to be constrained. If the gradients are too large, the model will not be able to learn. If the gradients are too small, the model will not be able to learn. This is known as the vanishing gradient problem. The gradients need to be constrained to a certain range. This is done by using layer normalization. 

The **output layer** plays an important role in determining the gradient of a layer.

## Feed-forward Neural Network

### Disadvantages of Relu
1. not differentiable between negative and zero inputs. 
2. when input is negative, relu is zero. So the gradient is zero. This is known as the **dying relu** problem. Dead Neuron. Neuron is dead. It is not able to learn anything.

### Leaky relu 

### GELU activation function 

### Covariate shift

### Value clipping

## All the illustrations and mindmaps referenced in this article : 
![](/assets/genai/attention/excalidraws/attention/attention.excalidraw.png)

## References

1. [What is the big deal about attention - vizuara substack](https://substack.com/inbox/post/158574020)
2. [Attention is all you need - arxiv](https://arxiv.org/abs/1706.03762)
3. Attention mechanism 1 hour video 
<iframe width="560" height="315" src="https://www.youtube.com/embed/K45ze9Yd5UE?si=FAJ3YPArq9Wu-uQ3" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


