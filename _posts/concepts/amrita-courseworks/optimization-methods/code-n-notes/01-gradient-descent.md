---
layout: mermaid
title: "Gradient Descent"
---

# Understanding cost or loss function in ML
## Liner regression
Consider a dataset $(x_1,y_1),(x_2,y_2), (x_3,y_3)$
x is input and y is output. 
If we have a linear model $y_i = w_ix + b$ where $w$ is the weight and $b$ is the bias, that learns from the above data.
After learning the model predicts $\hat y_i$ for the input $x_i$, as $\hat y_i = w_ix + b$.
### Loss function 
Loss function is a measure of how well the model predicts the output. 
It is a function of the model parameters $w$ and $b$.
Loss $L(w,b)$ is Mean squared error (MSE) defined as follows : 
$$
L(w,b)=-\frac{1}{n} \sum_1^n(y_i-\hat y)^2 \\
\Rightarrow L(w,b)=\frac{1}{n} \sum_1^n(y_i-(w_ix+b))^2
$$

### The objective
Find the optimal values of $w$ and $b$ that minimize the loss function.

# Gradient descent
Gradient descent 

Parameters = $\theta = (w_1, w_2, \dots , w_n)$ for a neural network
$\theta ← \theta - \eta ∇_\theta L(\theta)$
η = learning rate
$∇_\theta L(\theta)= $



# How to use gradient descent to minimize lost


# Understanding back propagation in ML


# Practical considerations and examples