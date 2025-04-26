---
layout : mermaid
title : Diffusion model for Devanagari script generation
author : Samrat Kar
---

# Devanagari Script Generation Diffusion Model Analysis

## Executive Summary

This report presents a comprehensive analysis of a diffusion model designed for generating Devanagari script characters. The model successfully generates high-quality Devanagari characters after just 2 epochs of training on a dataset of 92,000 images. This document outlines the model's architecture, examines the current configuration, and proposes optimization strategies to enhance performance while maintaining generation quality.

## 1. Introduction to Diffusion Models

Diffusion models represent a powerful class of generative models that learn to reverse a gradual noising process. The process begins with adding noise to images until they become pure noise, then training the model to reverse this process step by step. For Devanagari script generation, this approach is particularly suitable due to the complex structure and intricate details of the characters.

### 1.1 Diffusion Model Theory

Diffusion models are based on two key processes:
1. **Forward Process (Diffusion)**: A Markov chain that gradually adds Gaussian noise to the data over T timesteps
2. **Reverse Process (Denoising)**: A learned process that gradually removes noise to recover the data distribution

### 1.2 Diffusion Pipeline Stages

The complete diffusion model pipeline consists of:

1. **Data Preparation**: Preprocessing the Devanagari script images
2. **Forward Diffusion**: Adding predetermined noise to training images
3. **Model Training**: Learning to predict and remove noise
4. **Sampling**: Generating new Devanagari characters through iterative denoising
5. **Post-processing**: Enhancing the generated images if needed

## 2. Detailed Model Architecture Analysis

### 2.1 Core Architecture

The implemented diffusion model uses a U-Net architecture with the following key components:

#### 2.1.1 U-Net Architecture Breakdown

The U-Net architecture consists of:

1. **Input Layer**: 
   - Accepts 32×32 pixel images
   - Processes the noise level embedding (timestep)
   - Initial convolution to map to the feature space

2. **Encoder Path**:
   - Sequential downsampling blocks that progressively reduce spatial dimensions
   - Feature maps become deeper as they become spatially smaller
   - Downsampling operations: convolutions with stride 2
   - 6 downsampling stages with channel dimensions [128, 128, 256, 256, 512, 512]

3. **Bottleneck**:
   - Middle blocks with highest feature depth (512 channels)
   - Contains self-attention layers for global information processing
   - Captures highest-level abstract features of the Devanagari script

4. **Decoder Path**:
   - Sequential upsampling blocks that restore spatial dimensions
   - Feature maps become shallower as they expand spatially
   - Upsampling operations: transposed convolutions or upsample + convolution
   - 6 upsampling stages mirroring the encoder path
   - Skip connections from encoder to decoder for preserving detailed information

5. **Skip Connections**:
   - Connect corresponding layers in encoder and decoder
   - Allow detailed spatial information to bypass the bottleneck
   - Essential for preserving fine details in the generated Devanagari characters

6. **Time Embedding**:
   - Processes the diffusion timestep using sinusoidal positional embeddings
   - Transformed through MLPs to condition each block's operation
   - Allows the model to adapt its behavior based on the denoising stage

7. **Output Layer**:
   - Final convolution to map features to the output space
   - Produces noise prediction to guide the denoising process

### 2.2 Block Configuration

The model implements a sophisticated block structure:

#### 2.2.1 Downsampling Blocks (Encoder)

The implementation uses six down blocks with increasing feature dimensions:
```python
block_out_channels=(128, 128, 256, 256, 512, 512)
down_block_types=(
    "DownBlock2D",
    "DownBlock2D",
    "DownBlock2D",
    "DownBlock2D",
    "AttnDownBlock2D",
    "DownBlock2D",
)
```

Each **DownBlock2D** typically contains:
- ResNet-style blocks with convolutions, normalization, and activations
- Spatial downsampling via strided convolutions
- Timestep embedding injection through adaptive normalization or addition

The **AttnDownBlock2D** additionally contains:
- Self-attention mechanisms operating on feature maps
- Enables the model to capture relationships between different parts of the characters
- Particularly important for maintaining structural coherence in complex scripts

#### 2.2.2 Upsampling Blocks (Decoder)

The upsampling path contains six blocks:
```python
up_block_types=(
    "UpBlock2D",
    "AttnUpBlock2D",
    "UpBlock2D",
    "UpBlock2D",
    "UpBlock2D",
    "UpBlock2D",
)
```

Each **UpBlock2D** typically contains:
- ResNet-style blocks with convolutions, normalization, and activations
- Spatial upsampling via transposed convolutions or upsampling + convolution
- Skip connection integration from corresponding encoder blocks
- Timestep embedding injection

The **AttnUpBlock2D** additionally contains:
- Self-attention mechanisms similar to the encoder path
- Helps maintain global coherence during the upsampling process

### 2.3 Attention Mechanism In-Depth

The attention blocks implement a mechanism similar to the transformer's self-attention:

1. **Query, Key, Value Projections**:
   - Features are projected into query (Q), key (K), and value (V) spaces
   - Projections are performed using learned convolutional layers

2. **Attention Computation**:
   - Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√d)V
   - Allows the model to focus on relevant features across the spatial domain

3. **Multi-Head Implementation**:
   - Multiple attention heads operating in parallel
   - Each head captures different relationship patterns
   - Outputs are concatenated and projected back to the original feature space

This attention mechanism is crucial for capturing the intricate relationships between different strokes and components of Devanagari characters.

### 2.4 Time Conditioning

Time conditioning is fundamental to diffusion models:

1. **Timestep Embedding**:
   - Converts scalar timestep t to a high-dimensional embedding using sinusoidal functions
   - Embedding dimension is typically 4 times the base channel dimension
   
2. **MLP Projection**:
   - Processes the embedding through a small MLP network
   - Transforms the embedding into format suitable for modulating the U-Net blocks

3. **Feature Modulation**:
   - Injected into each block through adaptive normalization or additive bias
   - Controls how each layer processes features based on the denoising stage

This time conditioning allows the model to adapt its behavior based on the noise level, which is critical for the progressive denoising process.

## 3. Training Configuration Analysis

### 3.1 Current Configuration

```python
class TrainingConfig:
    image_size = 32  # assumes images are square
    train_batch_size = 32
    eval_batch_size = 32
    num_epochs = 2  # Reduced from 20 to 2
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 30
    mixed_precision = "fp16"
    output_dir = output_folder
    overwrite_output_dir = True
    seed = 0
    dataset_name = "data128"
```

### 3.2 Optimizer and Scheduler

- **Optimizer**: Adam - efficient stochastic optimization with adaptive learning rates
- **Scheduler**: Cosine schedule with warmup - gradually increases learning rate during initial training, then decreases it following a cosine curve

### 3.3 Performance Insights

The model achieving good results after just 2 epochs (reduced from 20) suggests:
- The dataset of 92,000 images provides sufficient examples for learning
- The model architecture is well-suited for the task
- The complexity of Devanagari script generation might be lower than initially anticipated
- The chosen hyperparameters create an efficient learning environment

## 4. Industry Best Practices for Diffusion Model Optimization

### 4.1 Efficient Training with Limited Data

#### 4.1.1 Data Augmentation Strategies

Effective data augmentation is crucial when working with limited training data:

1. **Geometric Transformations**:
   - **Rotation**: Small rotations (±5-10°) preserve character integrity while adding variety
   - **Slight Scaling**: Scale variations (0.9-1.1x) increase robustness
   - **Translation**: Small shifts preserve local patterns
   - **Flipping**: Should be used carefully for script data as it may affect readability

2. **Appearance Transformations**:
   - **Brightness/Contrast**: Subtle adjustments simulate different writing conditions
   - **Elastic Deformations**: Gentle warping imitates different writing styles
   - **Noise Injection**: Adding small amounts of structured noise before the diffusion process

3. **Script-Specific Augmentations**:
   - **Stroke Width Variation**: Simulate different pen widths
   - **Component Recombination**: For compound characters, mix components from different samples
   - **Style Transfer**: Apply minor style changes across characters

**Implementation Example**:
```python
def augment_devanagari(image):
    # Geometric transformations
    angle = random.uniform(-5, 5)
    scale = random.uniform(0.95, 1.05)
    image = tf.image.rot90(image, k=angle/90)
    
    # Appearance transformations
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    
    # Script-specific transformations
    # Implement custom transforms based on Devanagari structure
    
    return image
```

#### 4.1.2 Latent Diffusion Models

Industry has moved toward latent diffusion models to improve efficiency:

1. **Perceptual Compression**:
   - Use a pre-trained VAE/autoencoder to compress images to a latent space
   - Perform diffusion in the compressed latent space rather than pixel space
   - Significantly reduces computational requirements while preserving quality

2. **Implementation Process**:
   - Train or use a pre-trained encoder-decoder model
   - Encode training images to latent representations
   - Train diffusion model on these latent codes
   - During generation, decode the latent samples to produce images

3. **Benefits for Limited Data**:
   - Reduced dimensionality means fewer parameters to learn
   - More efficient learning from limited examples
   - Better generalization properties

### 4.2 Noise Schedule Optimization

The noise schedule critically impacts model performance:

1. **Linear vs. Cosine Schedules**:
   - Linear schedules: Simple but often sub-optimal
   - Cosine schedules: Better preservation of low-frequency information
   - Sigmoid schedules: Can help balance early and late stage denoising

2. **Learned Variance Schedules**:
   - Allow the model to learn optimal noise schedules during training
   - Particularly effective for specialized domains like script generation

3. **Noise Schedule for Script Generation**:
   - Early denoising steps: Focus on overall character structure
   - Middle steps: Develop major strokes and components
   - Late steps: Refine details and connections between strokes

**Industry Best Practice**: Implement a "warm-start" noise schedule that preserves more structure in early diffusion steps:
```python
def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in Improved DDPM paper
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)
```

### 4.3 Advanced Conditioning Techniques

Conditioning the diffusion model improves performance with limited data:

1. **Classifier Guidance**:
   - Use a pre-trained classifier to guide the generation process
   - Steer the diffusion sampling toward regions with high classifier confidence
   - Implementation: Modify the score function with classifier gradients

2. **Classifier-Free Guidance**:
   - Train the model to work both with and without conditioning
   - During inference, interpolate between conditional and unconditional predictions
   - More efficient than classifier guidance and doesn't require a separate classifier

3. **Conditioning Types for Script Generation**:
   - **Character Identity**: Condition on the character class/ID
   - **Style Parameters**: Condition on style indicators (thickness, slant)
   - **Context Characters**: Condition on neighboring characters for sequence generation

**Code Example for Classifier-Free Guidance**:
```python
def sample_with_cfg(model, x, t, class_labels, cfg_scale=3.0):
    # Predict noise with conditioning
    noise_cond = model(x, t, class_labels)
    
    # Predict noise without conditioning (null conditioning)
    noise_uncond = model(x, t, None)
    
    # Combine predictions using CFG
    noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
    
    return noise_pred
```

### 4.4 Training Efficiency Techniques

Industry has developed several techniques to improve training efficiency:

1. **Progressive Distillation**:
   - Train a teacher model with many diffusion steps
   - Distill knowledge into a student model with fewer steps
   - Allows for high-quality generation with fewer inference steps

2. **Offset Noise Optimization**:
   - Introduce small offsets to the noise prediction target
   - Helps the model focus on the most informative aspects of the signal
   - Particularly useful for detailed features in scripts

3. **Prediction Targets**:
   - Predict the noise directly (ε-prediction)
   - Predict the denoised image (x₀-prediction)
   - Predict velocity (v-prediction)
   - Industry finding: v-prediction often works best for detailed structures

4. **Efficient Attention Mechanisms**:
   - Linear attention mechanisms reduce computational complexity
   - Sparse attention focuses on the most relevant portions of the feature maps
   - Memory-efficient attention implementations for larger batch sizes

**Example Implementation of v-prediction**:
```python
def v_prediction_loss(model, x_0, t, noise):
    # Add noise to image according to timestep
    x_t = q_sample(x_0, t, noise)
    
    # Calculate velocity target
    alpha_t = extract(alphas_cumprod, t, x_0.shape)
    sigma_t = extract(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    v_target = alpha_t * noise - sigma_t * x_0
    
    # Predict v
    v_pred = model(x_t, t)
    
    # Calculate loss
    return F.mse_loss(v_pred, v_target)
```

### 4.5 Loss Function Specialization

Specialized loss functions improve training with limited data:

1. **Perceptual Losses**:
   - Incorporate losses based on pre-trained feature extractors
   - Emphasize perceptually important features of the script
   - Example: Use features from a CNN trained on script recognition

2. **Structural Losses**:
   - Incorporate losses that emphasize structural integrity
   - Example: Use SSIM (Structural Similarity Index) as a component of the loss

3. **Adversarial Losses**:
   - Incorporate a discriminator network for adversarial training
   - Can improve visual quality and adherence to the script style

4. **Focal Loss Variants**:
   - Adapt the standard MSE loss to focus more on difficult-to-predict regions
   - Particularly useful for intricate details in Devanagari characters

**Industry Best Practice**: Combine multiple loss components with appropriate weighting:
```python
def combined_loss(model_output, target, perceptual_extractor, alpha=1.0, beta=0.1):
    # Standard diffusion loss
    mse_loss = F.mse_loss(model_output, target)
    
    # Perceptual loss
    perceptual_features_pred = perceptual_extractor(model_output)
    perceptual_features_target = perceptual_extractor(target)
    perceptual_loss = F.mse_loss(perceptual_features_pred, perceptual_features_target)
    
    # Combine losses
    total_loss = mse_loss + alpha * perceptual_loss
    
    return total_loss
```

## 5. Optimization Strategies for Your Model

### 5.1 Architecture Optimizations

#### 5.1.1 Block Configuration Refinement

**Current Configuration**:
```python
block_out_channels=(128, 128, 256, 256, 512, 512)
```

**Potential Optimizations**:
- **Graduated Channel Scaling**: Implement a more gradual increase in channel dimensions
  ```python
  block_out_channels=(64, 128, 192, 256, 384, 512)
  ```
- **Balanced Attention Blocks**: Distribute attention blocks more evenly
  ```python
  down_block_types=(
      "DownBlock2D",
      "DownBlock2D",
      "AttnDownBlock2D",
      "DownBlock2D",
      "AttnDownBlock2D",
      "DownBlock2D",
  )
  up_block_types=(
      "UpBlock2D",
      "AttnUpBlock2D",
      "UpBlock2D",
      "AttnUpBlock2D",
      "UpBlock2D",
      "UpBlock2D",
  )
  ```

#### 5.1.2 Resolution Considerations

The current 32×32 resolution may be limiting detail capture. Consider:
- Increasing to 64×64 for better detail preservation
- Implementing a multi-scale approach with additional blocks:
  ```python
  block_out_channels=(64, 128, 192, 256, 384, 512, 512)
  ```

### 5.2 Training Optimizations

#### 5.2.1 Learning Rate Tuning

- **Learning Rate Range Test**: Implement a test to find optimal learning rate range
- **Cyclic Learning Rates**: Consider implementing instead of cosine annealing
  ```python
  learning_rate = 5e-5  # Lower base learning rate
  # With cyclic scheduler implementation
  ```

#### 5.2.2 Batch Size Adjustments

- **Batch Size Increase**: If hardware allows, increase batch size for more stable gradients
  ```python
  train_batch_size = 64  # Up from 32
  eval_batch_size = 64
  ```
- **Gradient Accumulation**: Increase steps for virtual batch size increase
  ```python
  gradient_accumulation_steps = 4  # Up from 1
  ```

#### 5.2.3 Advanced Regularization

- **Dropout**: Add dropout layers to prevent overfitting
- **Data Augmentation**: Implement rotation, slight scaling, and minor distortions

### 5.3 Noise Schedule Optimization

- **Custom Noise Schedule**: Implement a noise schedule tailored to Devanagari features
- **Progressive Distillation**: Train subsequent models with fewer diffusion steps

## 6. Evaluation Metrics and Monitoring

### 6.1 Quality Assessment

- **FID (Fréchet Inception Distance)**: Implement to measure generated image quality
- **Character Recognition Accuracy**: Test with OCR systems to ensure legibility
- **Human Evaluation**: Conduct studies with Devanagari readers

### 6.2 Performance Monitoring

- **Training Dynamics**: Track loss curves during training
- **Inference Speed**: Measure generation time per character
- **Model Size vs. Quality**: Evaluate quality against model size tradeoffs

## 7. Conclusion and Recommendations

The current diffusion model for Devanagari script generation demonstrates remarkable efficiency by producing high-quality results after just 2 epochs. This suggests that:

1. The architecture is well-aligned with the complexity of the task
2. The dataset provides comprehensive coverage of Devanagari characters
3. The optimization strategy effectively navigates the solution space

**Key Recommendations**:

1. **Short-term Improvements**:
   - Experiment with balanced attention blocks
   - Implement targeted data augmentation
   - Test lower learning rates (3e-5 to 8e-5)

2. **Medium-term Enhancements**:
   - Increase image resolution to 64×64
   - Implement custom noise scheduling
   - Add conditional generation capabilities

3. **Long-term Research Directions**:
   - Explore knowledge distillation to create smaller models
   - Implement classifier-free guidance to improve generation quality
   - Develop specialized loss functions for Devanagari script characteristics

By implementing these recommendations systematically, the model can be further optimized for efficiency, quality, and versatility in Devanagari script generation.

## Output:
![](/assets/genai/diffusion/diffusion_process3_devnagri.gif)
![](/assets/genai/diffusion/diffusion_process_grid3.png)
![](/assets/genai/diffusion/diffusion_steps_visualization3.png)

## Architecture of diffusion model 
![](/assets/genai/diffusion/diffusion-model-architecture.mermaid)

## Code base: 
[Code base for the implementation of the diffusion model to generate Devangari Scripts](https://colab.research.google.com/github/samratkar/samratkar.github.io/blob/main/_posts/concepts/genai/notes-codes/diffusion/code/Diffusion_Sanskrit_Gen_img.ipynb)

## References

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems.
2. Nichol, A., & Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. International Conference on Machine Learning.
3. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-resolution image synthesis with latent diffusion models. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
4. Saharia, C., Chan, W., Saxena, S., Li, L., et al. (2022). Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems.
5. Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-based generative modeling through stochastic differential equations. International Conference on Learning Representations.
6. Dhariwal, P., & Nichol, A. (2021). Diffusion models beat GANs on image synthesis. Advances in Neural Information Processing Systems.
7. Karras, T., Aittala, M., Laine, S., Härkönen, E., Hellsten, J., Lehtinen, J., & Aila, T. (2022). Elucidating the design space of diffusion-based generative models. Advances in Neural Information Processing Systems.