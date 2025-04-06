---
layout: mermaid
title: "Multi-head attention"
type: sub-note
---
In the single head causal self attention explained above we have only one attention head. This means that the model is only able to focus on one perspective of the input sequence at a time. This can be limiting, especially for complex tasks where multiple parts of the input sequence may be relevant at the same time.

For example, consider this sentence : **The artist painted the portrait of a woman with a brush**
