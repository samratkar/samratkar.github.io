---
layout: mermaid
type: concept 
title: "Generative AI Fundamentals-session 3,4 - RNN"
date: 2024-02-1
tags: introduction to LLM
book: build llm from scratch
course: vizuara
author: Samrat Kar
course: gen ai fundamentals - vizuara
class : 3,4
---

**Courtesy and Thanks** to sessions from [Vizuara.ai](https://vizuara.ai) that has helped me to understand the concepts discussed here below. 

## RNN

1. The memory is retained from the hidden layer. Previous state is passed to the current state.
2. Intestellar RNN 
3. RNN can be used as a language translator.
4. All burden of compression is in the last hidden state.
5. Video of Jay Allamar - what is passed to the decoder is the last hidden state.
6. This is a disadvantage of RNN. They have context limitation.

## Build an RNN from scratch

English to Hindi translation using RNN

### Block diagram for RNN


![alt text](../../../../../images/vizuara/rnn-data-flow.png)

#### Mathematical relationships for RNN

1. Equation for the hidden state at time t -
$h_t = \tanh(Wh_{t-1} + Ux_{t} + b)$  
2. Equation for the output at time t -
$y_t = softmax(Vh_t + c)$

#### Few observations 

1. The hidden state at time t is a function of the hidden state at time t-1.=, the input at time t and the bias term.
2. W and U are the weight matrices. That are trained using the training set. In the beginning, they are randomly initialized. And then they are trained, using the corpus so that the loss is minimized.
3. b is the bias term. It is also randomly initialized and then trained.
4. $x_t$ is the input at time t. It is the word embedding of the word at time t.
5. $h_t$ is the hidden state at time t. It is the memory of the network at time t.

#### Steps to calculate the hidden state [Encoding]

<div class=mermaid>
    graph TD;
    A(Step 1. Convert input words into embeddings) --> B(Step 2. Decide hidden state size)
    B --> C(Step 3. Initialize the first hidden state of encoder h0 based on hidden size)
    C --> D(Step 4. Initialize the weight matrices and bias term randomly for encoder)
    D --> E[Step 5. Calculate the hidden states h1, h2 using RNN matrix equation]
    E --> F[Step 6. Output the last hidden state to decoder]
</div>

#### Steps to translate the hidden state to the output [Decoding]

<div class=mermaid>
    graph TD;
    A(Step 1. Convert Hindi words into embeddings) --> B(Step 2. Initialize the hidden size of decoder same as that of the encoder)
    B --> C(Step 3. Initialize the 1st hidden state of decoder to teh final state of the encoder s0 = h0)
    C --> D(Step 4. Initialize the weight matrices and bias term randomly for decoder)
    D --> E[Step 5. Calculate the 1st hidden states s1 using RNN matrix equation using weight and biases]
    E --> F[Step 6. Compute the logits matrix]
</div>


1. The matrix W, U and b are initialized randomly.

$$
    W = \begin{bmatrix}
    0.3 & -0.1 \\
    0 & 0.2
    \end{bmatrix}_{2*2}
$$

$$
    U = \begin{bmatrix}
    0.5 \\
    0.7
    \end{bmatrix}_{2*1}
$$

$$
    b = \begin{bmatrix}
    0 \\
    0
    \end{bmatrix}_{2*1}
$$

Hence the 1st and 2nd hidden states are calculated as follows - 

$h_1 = \tanh(Wh_{0} + Ux_{1} + b)$  
$h_2 = \tanh(Wh_{1} + Ux_{2} + b)$

1. Step 1 - Convert input words into embeddings. $x_1 = 1$ and $x_2 = 2$  
2. Step 2 - Initialize the hidden state. $h_0 = 0$
3. Step 3 - Calculate the hidden state. $h_1 = \tanh(Wh_{0} + Ux_{1}x + b)$
4. Parameter matrices seeding randomly - 

$$
A = \begin{bmatrix} 
1 & 2 & 3 \\ 
4 & 5 & 6 \\ 
7 & 8 & 9 
\end{bmatrix}
$$

