---
layout : mermaid
title : "L2 - Coding the first neuron"
type : concept
author : Samrat Kar
---

## Coding the 1st neuron 

<iframe width="560" height="315" src="https://www.youtube.com/embed/vbeanwfm0Q4?si=Wv5wb0WS7z4Jbs9K" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


```python
# One neuron
input = [1,2,3]
weights = [0.2, 0.8, -0.5]
bias = 2.0
output = input[0]*weights[0] + input[1]*weights[1] + input[2]*weights[2] + bias
print(output)

Output >>
2.3
```


## Coding 3 neurons

<iframe width="560" height="315" src="https://www.youtube.com/embed/Uvngs6sWyBg?si=AjKHKXID42rcCIxi" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

```python
# Layer : 1
# Inputs : 4 
# Number of neurons in each layer : 3
input = [1,2,3,2.5]
# number of rows = number of neurons. number of columns = number of inputs
# focus on the neuron, and consider it as a puller of strings of all inputs. So, all inputs need to be a list themselves.
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
# biases = number of neurons in the layer
biases = [2.0, 3.0, 0.5]
output_l1 = []
neuron_index = 0
for neuron_index_w_list in weights:
    # Calculate the dot product of inputs and weights for each neuron
    s = sum(x * w for x, w in zip(input, neuron_index_w_list)) + biases[neuron_index]
    output_l1.append(s)
    neuron_index += 1
print(output_l1)
# Number of outputs = number of neurons in the layer.
```

## Representing 3 Dimensional matrix

<iframe width="560" height="315" src="https://www.youtube.com/embed/z_fcBg6_bKU?si=C8b1hwy6x3RFArI6" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

```python
# Dimensions : 1
nodes = [3, 2, 5, 6] # 1 x 4 or just 4

# Dimensions : 2
nodes = [[3, 2, 5, 6],
         [5,2,3, 8]] # 1 x 2 x 4 or just 2 x 4

# Dimensions : 3
nodes = [[[3, 2, 5, 6],
          [5,2,3, 8]],
         [[1,2,3,4],
          [5,6,7,8]],
         [[3,2,5,6],
          [5,2,3,8]]] # 3 x 2 x 4
```