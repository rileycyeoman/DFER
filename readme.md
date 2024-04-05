## Introduction
This guide is a barebones implimentation of a Vision Transformer. This model is built off of model from Tin Nguyen https://github.com/tintn/vision-transformer-from-scratch.
The purpose of this repository is to provide additional pieces of information on the working components of the transformer as well as details on how to alter to user's own need.


### Overview
---

Here we go through each individual part of the code and relate it to the overall model, as shown in *figure 1*

![alt text](/images/vit.png)  
*Figure 1: Vision Transformer overview*

To start off, let's break down how images are transformed to be used in the transformer. 

### Patch Embedding
---
![alt text](/images/patchembed.png)  
*Figure 2: Patch embedding process*



Multihead Self-Attention
$$[q,k,v] = zU_{qkv}, \; U \in \mathbb{R}^{D\times 3D_h}$$ 
$$A = softmax(\frac{qk^\top}{\sqrt{D_h}}) \; A \in \mathbb{R}^{N \times N} $$

That's pretty much it for image information. Now al we have to do is go through how the transformer behaves from here. 

### Self-Attention 
---

![alt text](/images/100epochs.png)  
*Figure ?: Visualized attention*

### Multi-Head Self-Attention
---


### Feed-Forward Network
---
Feed-forward networks are simply common neural networks, the most basic form. Normal transformer architectures that are used for *generation* purposes will utilize an encoder and decoder, due to their self-supervised nature. The decoder isn't necessary in classification processes because it's not generating anything. Encoding gives you the most basic information needed to describe data and the relationships between all bits of information. Decoders use the minimal information to restore the data, while all the multi-layer perceptron (MLP) needs to do is learn to classify off of this information.

For activation functions, the Gaussian Error Linear Unit (GELU) is used. From the original paper (https://arxiv.org/abs/1606.08415), the GELU is:
$$GELU(x) = x \cdot \frac{1}{2} [1 + erf(\frac{x}{\sqrt 2})]$$ 
Which can be approximated as:
$$0.5 \cdot x (1+tanh[\sqrt{\frac{2}{\pi}(x + 0.044715x^3)])}$$



### Bringing It All Together
---