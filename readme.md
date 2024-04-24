## Introduction
This guide is a barebones implimentation of a Vision Transformer. This model is built off of model from Tin Nguyen [https://github.com/tintn/vision-transformer-from-scratch][1].
The purpose of this repository is to provide additional pieces of information on the working components of the transformer as well as details on how to alter to user's own need.


### Overview
---

Here we go through each individual part of the code and relate it to the overall model, as shown in *figure 1*

![alt text](/images/vit.png)  
*Figure 1: Vision Transformer overview*

Note: When looking through this code, you'll often see references to **config**. This is not in the base code and is instead used in the Jupyter notebook from the original author. It looks like this:

```
config = {
    "patch_size": 4,  # Input image size: 32x32 -> 8x8 patches
    "hidden_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 48, # 4 * hidden_size
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 10, # num_classes of CIFAR10
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": True,
}
```
Obviously, these are parameters that can be changed, although it shouldn't be essential. Additional parameters that are actually useful for testing different training parameters is also from the notebook:
```
exp_name = 'vit-with-100-epochs' #@param {type:"string"}
batch_size = 32 #@param {type: "integer"}
epochs = 100 #@param {type: "integer"}
lr = 1e-4  #@param {type: "number"}
save_model_every = 0 #@param {type: "integer"} 
```
To start off, let's break down how images are transformed to be used in the transformer. 

### Patch Embedding
---
![alt text](/images/patchembed.png)  
*Figure 2: Patch embedding process*

As alluded to before, transformers weren't originally meant to handle data like images. It was a sort of replacement for Recurrent Neural Networks (RNNs). While the transformer improved The paper __AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE__ gives the following method for making image data readable by transformers:

Given an image of size $x \in \mathbb{R}^{H \times W \times C}$, split the images into 2D patches and flatten into size $x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$
* $C$ is the channel number, $(H,W)$ is the resolution of the original image
* $(P,P)$ is the resolution of each patch
* $N = \frac{HW}{P^2}$ is the resulting number of patches

Once we have this flattened patch, the resulting sequence is the input for the transformer. Because the transformer uses $D$ dimensions as a constant latent vector size through all its layers, the input will be mapped to $D$ dimensions through a trainable linear projection, which is defined as:$$z_0 = [x_{class}; x_p^1E; x_p^2E;...;x_p^NE] + E_{pos}$$
$$E \in \mathbb{R}^{(P^2 \cdot C) \times D}, E_{pos} \in \mathbb{R}^{(N+1) \times D}$$
The shapes here appear confusing. Simply put, they are training to understand how each patch should be linearily mapped to a dimension $D$


That's pretty much it for image information. Now al we have to do is go through how the transformer behaves from here. 

### Self-Attention 
---

![alt text](/images/100epochs.png)  
*Figure ?: Visualized attention*

### Multi-Head Self-Attention
---
Multihead Self-Attention
$$[Q,K,V] = zU_{QKV}$$
$$U \in \mathbb{R}^{D\times 3D_h}$$ 
$$A = softmax(\frac{QK^\top}{\sqrt{D_h}}) \; A \in \mathbb{R}^{N \times N} $$

### Feed-Forward Network
---
Feed-forward networks are simply common neural networks, the most basic form. Normal transformer architectures that are used for *generation* purposes will utilize an encoder and decoder, due to their self-supervised nature. The decoder isn't necessary in classification processes because it's not generating anything. Encoding gives you the most basic information needed to describe data and the relationships between all bits of information. Decoders use the minimal information to restore the data, while all the multi-layer perceptron (MLP) needs to do is learn to classify off of this information.

For activation functions, the Gaussian Error Linear Unit (GELU) is used. From the original paper, **Gaussian Error Linear Units (GELUs)** (https://arxiv.org/abs/1606.08415), the GELU is:
$$GELU(x) = x \cdot \frac{1}{2} [1 + erf(\frac{x}{\sqrt 2})]$$ 
Which can be approximated as:
$$0.5 \cdot x (1+tanh[\sqrt{\frac{2}{\pi}(x + 0.044715x^3)])}$$

$$z_{\ell} = MLP(LN(z_{\ell}^{'})) + z_{\ell}^{'}$$
$$\ell = 1...L$$

### Bringing It All Together
---





[1]: https://github.com/tintn/vision-transformer-from-scratch