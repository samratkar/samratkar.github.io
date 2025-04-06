Layer normalization is done on input embedding. Information is not lost because it is relative scaling. 
Layer normalization is needed to handle the following - 
### Vanishing or exploding gradient problem
Gradients need to be constrained. If the gradients are too large, the model will not be able to learn. If the gradients are too small, the model will not be able to learn. This is known as the vanishing gradient problem. The gradients need to be constrained to a certain range. This is done by using layer normalization. 

The **output layer** plays an important role in determining the gradient of a layer.