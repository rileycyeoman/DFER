
"""
#######
Imports
#######
"""

import math
import torch
from torch import nn

"""
########################
GELU Activation Function
########################
"""
class GELU(nn.Module):
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


"""
################
Patch Embeddings
################
"""

class PatchEmbeddings(nn.Module):
    def __init__(self,config):
        super().__init__()
        #Configuration parameters
        self.image_size = config["image_size"] #base image size
        self.patch_size = config["patch_size"] #patch size of image taken
        self.num_channels = config["num_channels"] #color channels
        self.hidden_size = config["hidden_size"]

        self.num_patches = (self.image_size // self.patch_size) ** 2
        #Image of size HxWxC is projected into a space of Nx(PxPxC)
        #PxP is the patch size, N is the number of patches (HW/P^2)
        #conv2d(in channels, out channels, kerel size, stride)
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size = self.patch_size, stride = self.patch_size)
    
    #forward pass
    def forward(self, x):
        x = self.projection(x)
        #flatten turns the tensor into 2 dimensions, tranpose restores original shape
        x = x.flatten(2).transpose(1,2)
        return x #(batch_size, num_patches, hidden_size)
    
class Embeddings(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)
        #Learnable class (CLS) token
        self.cls_token = nn.Parameter(torch.randn(1,1,config["hidden_size"]))
        #learnable position embeddings, 1 is added to account for class token
        #nn.Parameter allows this to be a trained parameter
        #Tensor of size (1, num_patches + 1, hidden_size)
        self.position_embeddings = \
            nn.Parameter(torch.randn(1,self.patch_embeddings.num_patches + 1, config["hidden_size"]))
        #regularization via random dropout of nodes
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    #forward pass
    def forward(self,x):
        x = self.patch_embeddings(x) #PxPxC
        batch_size,_,_ = x.size()
        #Expand class token to the batch size
        """
        EXPLAIN
        """
        cls_tokens = self.cls_token.expand(batch_size,-1,-1) #(1,1,hidden_size) -> (batch_size, 1, hidden_size)
        #Concatenate with class token at the front
        x = torch.cat((cls_tokens,x), dim = 1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x
    

class AttentionHead(nn.Module):
    
    def __init__(self, hidden_size, attention_head_size, dropout, bias = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        #Q,K,V, learnable value projection layers for the transformer
        self.Q = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.K = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.V = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)

    #Forward pass, learn Q, K, and V
    #Calculate 
    def forward(self, x):
        #query
        Q = self.Q(x)
        #key
        K = self.K(x)
        #value
        V = self.V(x)
        #A = softmax(Q*K)/sqrt(D_h), D_h is size of head
        attention_scores = torch.matmul(Q, K.transpose(-1,-2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim = -1)
        attention_probs = self.dropout(attention_probs)
        #SA(z) = A * V
        attention_output = torch.matmul(attention_probs, V)
        return (attention_output, attention_probs)
    

class MultiHeadAttention(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        #Boolean for use of bias in QKV projections
        self.qkv_bias = config["qkv_bias"]
        
        #list of attention heads
        self.heads = nn.ModuleList([])
        for _ in range (self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias
            )
            self.heads.append(head)
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])
    
    def forward(self, x, output_attentions = False):
        #Create head for each iter in the list, calculate output for each
        attention_outputs = [head(x) for head in self.heads]
        #Concatenate output from each heads
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim = -1)
        #project back to hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        #return the attention output and the attention probabilities
        if not output_attentions:
            return (attention_output, None)
        else:
            attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim = -1)
            return (attention_output, attention_probs)
        
class FasterMHA(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]

        #Create a lienar layer to project QKV
        self.qkv_projection = nn.Linear(self.hidden_size, self.all_head_size * 3, bias = self.qkv_bias)
        self.attn_dropoiut = nn.Dropout(config["attention_probs_dropout_prob"])
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions = False):
        #Project Q,K,V
        qkv = self.qkv_projection(x)
        # Split the projected query, key, and value into query, key, and value
        # (batch_size, sequence_length, all_head_size * 3) -> (batch_size, sequence_length, all_head_size)
        Q, K, V = torch.chunk(qkv, 3, dim = -1) #split the tensor into 3 chunks
        #resize QKV to (batch_size, num_attention_heads, sequence_length, attention_head_size)
        batch_size, sequence_length = Q.size()
        Q = Q.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1,2)
        K = K.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1,2)
        V = V.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1,2)

        #Calculate attention scores
        attention_scores = torch.matmul(Q, K.tranpose(-1,-2)) #negative indicates reverse ordering
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) #D_h
        attention_probs = nn.functional.softmax(attention_scores, dim = -1) #-1 ensures the last dimension is what is softmaxed
        attention_probs = self.attn_dropout(attention_probs)
        #Calculate attention output
        attention_output = torch.matmul(attention_probs, V)
        attention_output = self.output_dropout(attention_output)

        if not output_attentions:
            return (attention_output, None)
        else:
            return (attention_output, attention_probs)


class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = GELU()
        self.dense_2 = nn.Linear(config["intermediate_size"],config["hidden_size"])
        self.dropout = nn.Dropout(["hidden_dropout_prob"])

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x
    

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_faster_attention = config.get("use_faster_attention", False)
        if self.use_faster_attention:
            self.attention = FasterMHA(config)
        else: 
            self.attention = MultiHeadAttention(config)
        
        self.layernrom_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernrom_2 = nn.LayerNorm(config["hidden_size"])
    
    def forward(self, x, output_attentions = False):
        #self-attention
        attention_output, attention_probs = \
            self.attention(self.layernorm_1(x), output_attentions = output_attentions)
        # Skip connection
        x = x + attention_output
        #feed forward
        mlp_output = self.mlp(self.layernrom_2(x))
        x = x + mlp_output
        if not output_attentions:
            return (x, None)
        else: 
            return (x, attention_probs)
        


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            self.blocks.append(block)
        
    def forward(self, x, output_attentions = False):
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions = output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
        if not output_attentions:
            return(x, None)
        else: 
            return (x, all_attentions)
        
class ViTForClassfication(nn.Module):
    """
    The ViT model for classification.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        # Create the embedding module
        self.embedding = Embeddings(config)
        # Create the transformer encoder module
        self.encoder = Encoder(config)
        # Create a linear layer to project the encoder's output to the number of classes
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        # Initialize the weights
        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
        # Calculate the embedding output
        embedding_output = self.embedding(x)
        # Calculate the encoder's output
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
        # Calculate the logits, take the [CLS] token's output as features for classification
        logits = self.classifier(encoder_output[:, 0, :])
        # Return the logits and the attention probabilities (optional)
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)