---
layout : mermaid
title : "Position Embeddings"
author : Samrat Kar
---
>**The dog chased the ball, but "it" could not catch "it" even after running the whole field round and round.**


In the above sentence,  first **"it"** refers to the dog but the second **"it"** refers to the ball. The token id for **"it"** would be same, and so would be the embedding. But based on their positions, they need to be differentiated. This differentiation based on the position of the same token in a given input text is created by adding the position embedding to the token embeddings.

The position embedding is a matrix of size (context_size, embedding_size) where context_size number of token processed by LLM at a time. And embedding size is the number of dimensions to which the tokens are represented in vector form. The token embedding is added to the position embedding to create the final input embedding that is provided as input to the LLM. 

$$\text{Final Input Embedding} = \text{Token Embedding} + \text{Position Embedding}$$

![](/images/genai/position-embeddings.svg)

### Key Points

1. This input embedding is random to begin with. It is passed to the transformer. It undergoes training along with the positional embedding matrix.
2. The loss function of the token embedding matrix is calculated based on the output of the transformer that gives the actual output. The expected output is already part of the output matrix. This gives the actual and expected values whose difference becomes an input to the cost function. That is minimized using back propagation.
3. The cost function of the position embedding matrix is calculated based on the token embedding matrix. 
4. Since the positional embeddings are part of the input representation, their values are adjusted to minimize the main training objective of the model. The loss function depends on the task:
 - For language modeling (e.g., GPT, causal transformers): The loss is usually cross-entropy loss between predicted and actual next-token probabilities.
 - For masked language modeling (e.g., BERT): The loss is cross-entropy over masked token predictions.
 - For translation tasks (e.g., Transformer): The loss could be cross-entropy over predicted translated words.
5. The positional embeddings help the model differentiate between word order in a way that improves performance on the overall task. If a particular position is important for predicting the next word or reconstructing a masked word, the model adjusts the positional embeddings accordingly.
Hence, Positional embeddings are learned implicitly as part of the model's backpropagation process. There’s no separate cost function for them; they are updated to minimize the primary loss of the model.
5. The same positional embedding and token embedding is used for all the tokens in the input text. They act as lookup.
6. Position embeddings have nothing to do with the meaning. It is there only to quantify the position. The size of the positional embedding is same as the token embedding just because they are added together. Otherwise we don't need such large vector for positional embedding.


### Code snippets

1. Initializing a data loader of context size = max_length = 4, batch size = 8, stride = same as context size. Creating the token ids to start with - 

#### Getting token ids
```python
max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter) # 1st iteration. getting the input target pair.
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

Output >>
Token IDs:
 tensor([[   40,   367,  2885,  1464],
        [ 1807,  3619,   402,   271],
        [10899,  2138,   257,  7026],
        [15632,   438,  2016,   257],
        [  922,  5891,  1576,   438],
        [  568,   340,   373,   645],
        [ 1049,  5975,   284,   502],
        [  284,  3285,   326,    11]])

Inputs shape:
 torch.Size([8, 4])

```

#### Getting token embedding matrix. The step where the 2D token id matrix is converted into a 3D token embedding matrix!
```python
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

Output >>
torch.Size([8, 4, 256])
```

#### Getting position embedding matrix. 
```python
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print(pos_embeddings.shape)

Output >>
torch.Size([4, 256])
```

#### Adding the token embedding and position embedding matrices to get input matrix
```python
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)

Output >>
torch.Size([8, 4, 256])
```