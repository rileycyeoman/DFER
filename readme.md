This guide is a barebones implimentation of a Vision Transformer. This model is built off of model from Tin Nguyen https://github.com/tintn/vision-transformer-from-scratch.
The purpose of this repository is to provide additional pieces of information on the working components of the transformer as well as details on how to alter to user's own need.


### Patch Embedding

![alt text](/images/patchembed.png)  
*Figure 1: Patch embedding process*



For activation functions, the Gaussian Error Linear Unit (GELU) is used. From the original paper (https://arxiv.org/abs/1606.08415), the GELU is:
$$GELU(x) = x \cdot \frac{1}{2} [1 + erf(\frac{x}{\sqrt 2})]$$ 
Which can be approximated as:
$$0.5 \cdot x (1+tanh[\sqrt{\frac{2}{\pi}(x + 0.044715x^3)])}$$

Multihead Self-Attention
$$[q,k,v] = zU_{qkv}, \; U \in \mathbb{R}^{D\times 3D_h}$$ 
$$A = softmax(\frac{qk^\top}{\sqrt{D_h}}) \; A \in \mathbb{R}^{N \times N} $$

