---
layout : mermaid
title : Building LLM from Scratch
author : Samrat Kar 
---
## Dataset 
Dataset chosen - Julius Caesar text. [Link](/assets/genai/attention/data/julius-caesar.txt)

## Train the LLM with the dataset
### Setting up the training 
```python
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            tokens_seen += input_batch.numel() # Returns the total number of elements (or tokens) in the input_batch.
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen
```
### Training the model
```python
import time
start_time = time.time()

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1) #A

num_epochs =50
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=1,
    start_context="He said we came here", tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")
>>> Training completed in 13.78 minutes.
```
## Model configuration
1. learning rate = 1e-4
2. max iters = 10000
3. warmup steps = 2000 
4. min lr = 5e-4 
5. eval iters = 500 
6. batch size = 8 
7. block size = 128 
8. vocab size=50257 
9. block size=128 
10. number of layers=4 
11. number of heads=4 
12. embedding dimension=768 
13. dropout=0.01 
14. bias=True

## Training & validation loss
### Epoch : 500

![](/assets/genai/lfs-assignment3/training-validation-loss.png)

### Epoch : 4000 
![](/assets/genai/attention/jc-slm/4kep.png)

### Epoch : 10,000 
![](/assets/genai/attention/jc-slm/10kep.png)

### Epoch : 20,000
![](/assets/genai/attention/jc-slm/20kep.png)
