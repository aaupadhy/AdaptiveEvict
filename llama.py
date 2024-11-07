import torch
import torch.nn as nn
import torch.nn.functional as F


# T  -->  Number of Tokens / Vocab Size
# B  -->  Batch Size
# E  -->  Embedding Dimension
# F  -->  Forward Multiplier
# S  -->  Sequence Length   
# Q  -->  Query Sequence length (equal to S for self-attention)
# K  -->  Key Sequence length   (equal to S for self-attention)
# V  -->  Value Sequence length (equal to S for self-attention)
# H  -->  Number of heads
# HE -->  Head Embedding Dimension = E/H
# G  -->  Number of Experts
# TG -->  Number of Top Experts


class TokenEmbedding(nn.Module):
    """
    Class for Embedding word indices.

    Parameters:
        vocab_size (int) : Number of words (T)
        embed_dim  (int) : Embedding dimension (E)

    Input:
        x (tensor): Long Tensor of shape B, S containing token indices 
    
    Returns:
        Tensor: Embedding of the tokens of shape B, S, E
    """   

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)                                  # Embedding for each token

    def forward(self, x):                                                                   
        x = self.token_embedding(x)                                                                 # B, S  -->   B, S, E
        return x


class RotatoryPositionEmbedding(nn.Module):
    """
    Class for creating rotational positional embeddings.

    Parameters:
        max_seq_len (int) : Maximum sequence length used for training. This is used to cache rotational positional embeddings.
        embed_dim   (int) : Embedding dimension

    Input:
        x (tensor): Tensor of shape B, S, H, HE containing query/keys projections.
    
    Returns:
        Tensor: Embedding of the tokens of shape B, S, H, HE after rotating using rotational positional embeddings
    """   

    def __init__(self, seq_len, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        x_sin, x_cos = self.build_rope(seq_len)                                   # 1, S, 1, E    ,    1, S, 1, E
        self.register_buffer("x_cos", x_cos)                                      # Register_buffer for easy switching of device
        self.register_buffer("x_sin", x_sin)                                      # Register_buffer for easy switching of device

    def build_rope(self, seq_len):
        '''
        Create theta as per the equation in the RoPe paper: theta = 10000 ^ -2(i-1)/d for i belongs to [1, 2, ... d/2].  
        '''
        sequence   = torch.arange(seq_len).float().unsqueeze(-1)
        thetas     = - torch.arange(start=0, end=self.embed_dim, step=2).float() / self.embed_dim       # E//2
        thetas     = torch.repeat_interleave(thetas, 2, 0)                                              # E
        thetas     = torch.pow(10000, thetas)                                                           # E
        angles     = sequence * thetas                                                                  # S, 1 * E --> S, E
        cos_values = torch.cos(angles).unsqueeze(1).unsqueeze(0)                                        # S, E     --> 1, S, 1, E      Precompute and store cos values
        sin_values = torch.sin(angles).unsqueeze(1).unsqueeze(0)                                        # S, E     --> 1, S, 1, E      Precompute and store sin values
        return sin_values, cos_values       

    def forward(self, x, token_loc=None):
        '''
        function to apply rotation on queries and keys

        Input:
            x         (tensor) : torch tensors of shape B, S, H, E
            token_loc (int)    : To be used when a single element in a sequence is passed and its location index is specified using "token_loc". (To be used with KV Cache while inference)
        
        Returns:
            Tensors: x 

        '''
        if token_loc is None:
            x1 = x * self.x_cos[:, :x.shape[1], :, :]                              # B, S, H, E  *  1, S, 1, E          -->  B, S, H, E            Multiply with its cos factor
            x_shifted = torch.stack((-x[:, :, :, 1::2], x[:, :, :, ::2]), -1)      # B, S, H, E//2 stack B, S, H, E//2  -->  B, S, H, E//2, 2      Shift values for sin multiplications
            x_shifted = x_shifted.reshape(x.shape)                                 # B, S, H, E//2, 2                   -->  B, S, H, E            Reshape to the original size
            x2 = x_shifted * self.x_sin[:, :x.shape[1], :, :]                      # B, S, H, E  *  1, S, 1, E          -->  B, S, H, E            Multiply x with its sin factor
            x = x1 + x2                                                            # B, S, H, E  +  B, S, H, E          -->  B, S, H, E            Add sin and cosine value
        else:
            x1 = x * self.x_cos[:, token_loc, :, :].unsqueeze(1)                   # B, S, H, E  *  1, S, 1, E          -->  B, S, H, E            Multiply with its cos factor
            x_shifted = torch.stack((-x[:, :, :, 1::2], x[:, :, :, ::2]), -1)      # B, S, H, E//2 stack B, S, H, E//2  -->  B, S, H, E//2, 2      Shift values for sin multiplications
            x_shifted = x_shifted.reshape(x.shape)                                 # B, S, H, E//2, 2                   -->  B, S, H, E            Reshape to the original size
            x2 = x_shifted * self.x_sin[:, token_loc, :, :].unsqueeze(1)           # B, S, H, E  *  1, S, 1, E          -->  B, S, H, E            Multiply x with its sin factor
            x = x1 + x2                                                            # B, S, H, E  +  B, S, H, E          -->  B, S, H, E            Add sin and cosine value            
        return x


class MultiHeadSelfAttentionWithRope(nn.Module):
    """
    Class for computing Multi head Self-Attention with Rope Positional embedding. Also, applies causal attention mask so tokens can attend only to current or earlier tokens.

    Parameters:
        max_seq_len       (int) : Maximum sequence length used for training. This creates causal attention mask (att_mask)
        n_attention_heads (int) : Number of attention heads to use for performing MultiHeadAttention
        embed_dim         (int) : Embedding dimension
    
    Input:
        x (tensor): Tensor of shape B, S, E

    Returns:
        Tensor: Output after Self-Attention Module of shape B, S, E
    """  

    def __init__(self, embed_dim, n_attention_heads, max_seq_len):
        super().__init__()
        self.embed_dim         = embed_dim
        self.n_attention_heads = n_attention_heads
        self.max_seq_len       = max_seq_len
        self.head_embed_dim    = embed_dim // n_attention_heads

        self.queries           = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)        # Queries projection
        self.keys              = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)        # Keys projection
        self.values            = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)        # Values projection
        self.out_projection    = nn.Linear(self.head_embed_dim * self.n_attention_heads, self.embed_dim)        # Out projection

        # Mask to hide future words from current word / Causal Attention
        att_mask = torch.ones(1, 1, max_seq_len, max_seq_len)                                                   # 1, 1, S, S
        att_mask = torch.triu(att_mask, diagonal=1).bool()                                                      # 1, 1, S, S
        self.register_buffer("att_mask", att_mask)                                                              # Register_buffer for easy switching of device

        # Rotational Positional Embedding
        self.rotary_embedding  = RotatoryPositionEmbedding(seq_len=max_seq_len, embed_dim=self.head_embed_dim)  # Rotation for Queries and keys (Used for both because it applies the same funtions).

        # KV Cache for storing keys and values for the previous tokens. 
        self.reset_cache()

    # KV Cache for faster inference by storing KV cache of previous tokens. Also, useful to store precomputed instruction prompt.
    def reset_cache(self):
        self.cache = {'k': None, 'v': None}

    def forward(self, x, token_loc=None):
        if self.training:                                                                       # Training mode is normal. Inference mode uses KV cache
            b, s, e = x.shape                                                                   # Note: In case of self-attention Q, K and V are all equal to S

            xq = self.queries(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)     # B, Q, E      -->  B, Q, (H*HE)  -->  B, Q, H, HE
            xk = self.keys(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)        # B, K, E      -->  B, K, (H*HE)  -->  B, K, H, HE
            xv = self.values(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)      # B, V, E      -->  B, V, (H*HE)  -->  B, V, H, HE

            # Rotate Queries and Keys only
            xq = self.rotary_embedding(xq)                                                      # B, Q, H, HE  -->  B, Q, H, HE
            xk = self.rotary_embedding(xk)                                                      # B, K, H, HE  -->  B, K, H, HE  

            # Transpose head and sequence dims
            xq = xq.permute(0, 2, 1, 3)                                                         # B, Q, H, HE  -->  B, H, Q, HE
            xk = xk.permute(0, 2, 1, 3)                                                         # B, K, H, HE  -->  B, H, K, HE
            xv = xv.permute(0, 2, 1, 3)                                                         # B, V, H, HE  -->  B, H, V, HE

            # Compute Attention presoftmax values
            xk = xk.permute(0, 1, 3, 2)                                                         # B, H, K, HE  -->  B, H, HE, K
            x_attention = torch.matmul(xq, xk)                                                  # B, H, Q, HE   *   B, H, HE, K   -->  B, H, Q, K    (Matmul tutorial eg: A, B, C, D  *  A, B, E, F  -->  A, B, C, F   if D==E)

            x_attention /= float(self.head_embed_dim) ** 0.5                                    # Scale presoftmax values for stability

            # Apply Mask on future words / Causal attention
            x_attention = x_attention.masked_fill(self.att_mask[:, :, :s, :s], -torch.inf)      # B, H, Q, K    *   1, 1, S, S    -->  B, H, Q, K      Fill future values with -inf and Trim causal mask for smaller sequences

            x_attention = torch.softmax(x_attention, dim=-1)                                    # Compute Attention Matrix

            x = torch.matmul(x_attention, xv)                                                   # B, H, Q, K  *  B, H, V, HE  -->  B, H, Q, HE     Compute Attention product with Values

            # Format the output
            x = x.permute(0, 2, 1, 3)                                                           # B, H, Q, HE  --> B, Q, H, HE
            x = x.reshape(b, s, e)                                                              # B, Q, H, HE  --> B, Q, (H*HE)

            x = self.out_projection(x)                                                          # B, Q, (H*HE) --> B, Q, E
        else:
            # While inference KV Cache is used and only last token is used in the query to generate next token.

            b, s, e = x.shape                                                                   # B, 1, E      In case of inference input sequence has just 1 length (current token). Earlier tokens keys and values are stored in cache.

            xq = self.queries(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)     # B, 1, E      -->  B, 1, (H*HE)  -->  B, 1, H, HE
            xk = self.keys(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)        # B, 1, E      -->  B, 1, (H*HE)  -->  B, 1, H, HE
            xv = self.values(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)      # B, 1, E      -->  B, 1, (H*HE)  -->  B, 1, H, HE

            if self.cache['k'] is not None and self.cache['v'] is not None :
                xk = torch.cat((self.cache['k'], xk), 1)                                        # B, K-1, H, E  cat  B, 1, H, E   --> B, K, H, E
                xv = torch.cat((self.cache['v'], xv), 1)                                        # B, V-1, H, E  cat  B, 1, H, E   --> B, V, H, E

            self.cache['k'] = xk[:, -(self.max_seq_len-1):, :, :].detach()                      # B, K-1, H, E
            self.cache['v'] = xv[:, -(self.max_seq_len-1):, :, :].detach()                      # B, V-1, H, E

            # Rotate Queries and Keys only
            xq_rotated = self.rotary_embedding(xq, token_loc)                                   # B, 1, H, HE  -->  B, 1, H, HE
            xk_rotated = self.rotary_embedding(xk)                                              # B, K, H, HE  -->  B, K, H, HE  

            # Transpose head and sequence dims
            xq_rotated = xq_rotated.permute(0, 2, 1, 3)                                         # B, 1, H, HE  -->  B, H, 1, HE
            xk_rotated = xk_rotated.permute(0, 2, 1, 3)                                         # B, K, H, HE  -->  B, H, K, HE
            xv = xv.permute(0, 2, 1, 3)                                                         # B, V, H, HE  -->  B, H, V, HE

            # Compute Attention presoftmax values
            xk_rotated_ = xk_rotated.permute(0, 1, 3, 2)                                        # B, H, K, HE  -->  B, H, HE, K
            x_attention = torch.matmul(xq_rotated, xk_rotated_)                                 # B, H, 1, HE   *   B, H, HE, K   -->  B, H, 1, K    (Matmul tutorial eg: A, B, C, D  *  A, B, E, F  -->  A, B, C, F   if D==E)

            x_attention /= float(self.head_embed_dim) ** 0.5                                    # Scale presoftmax values for stability

            # Note: No Attention Mask / Causal attention required while inference as the future tokens are unavailable.

            x_attention = torch.softmax(x_attention, dim=-1)                                    # B, H, 1, K        Compute Attention Matrix

            x = torch.matmul(x_attention, xv)                                                   # B, H, 1, K  *  B, H, V, HE  -->  B, H, 1, HE     Compute Attention product with Values

            # Format the output
            x = x.permute(0, 2, 1, 3)                                                           # B, H, 1, HE  --> B, 1, H, HE
            # x = x[:, -1, :, :].unsqueeze(1)
            x = x.reshape(b, s, e)                                                              # B, 1, H, HE  --> B, 1, (H*HE)

            x = self.out_projection(x)                                                          # B, 1, (H*HE) --> B, 1, E

        return x


class RMSNorm(nn.Module):
    """
    Class for RMSNorm for normalizing data by scaling the data.

    Parameters:
        embed_dim (int) : Embedding dimension
    
    Input:
        x (tensor): Tensor of shape B, S, E

    Returns:
        Tensor: Rescaled tensor of shape B, S, E
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(1, 1, embed_dim))       

    def forward(self, x):                                                   
        x_deno = torch.sqrt(x.pow(2).mean(-1, keepdims=True))               # B, S, E  -->  B, S, 1
        x      = x / (x_deno + 1e-6)                                        # B, E, E   /   B, S, 1  -->  B, S, E
        x      = x * self.weights                                           # B, S, E   *   1, 1, E  -->  B, S, E
        return x                                                            # B, S, E


class SwiGLU(nn.Module):
    """
    Class for Swish Gated Linear Unit (SwiGLU)

    Parameters:
        in_dim   (int) : Input Embedding dimension
        out_dim  (int) : Output Embedding dimension
    
    Input:
        x (tensor): Tensor of shape B, S, E

    Returns:
        Tensor: Output tensor shape B, S, E
    """ 

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.fc(x)                                                  # B, S, E  -->  B, S, E
        x = x * torch.sigmoid(x)                                        # B, S, E  -->  B, S, E
        return x


class FF(nn.Module):
    """
    Class for creating optimized feed-forward layers

    Parameters:
        embed_dim    (int)  : Embedding dimension
        forward_mul (float) : Used to calculate dimension of the hidden fc layer = embed_dim * forward_mul
        dropout     (float) : Dropout parameter
    
    Input:
        x (tensor): Tensor of shape B, S, E

    Returns:
        Tensor: Output tensor shape B, S, E
    """ 

    def __init__(self, embed_dim=64, forward_mul=2, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * forward_mul)
        self.act = SwiGLU(embed_dim, embed_dim * forward_mul)
        self.fc2 = nn.Linear(embed_dim * forward_mul, embed_dim)

    def forward(self, x):                                                        
        x = self.act(x) * self.fc1(x)                                           # B, S, (E*F)  *  B, S, (E*F)  -->  B, S, (E*F)     Note: SwiGLU is multiplied.
        x = self.fc2(x)                                                         # B, S, (E*F) --> B, S, E
        return x


class MoeFF(nn.Module):
    """
    Class for creating optimized feed-forward layers with Mixture of Experts

    Parameters:
        n_experts     (int)  : Number of experts to create
        n_top_experts (int)  : Number of experts to select
        embed_dim     (int)  : Embedding dimension
        forward_mul (float) : Used to calculate dimension of the hidden fc layer = embed_dim * forward_mul
        dropout     (float) : Dropout parameter
    
    Input:
        x (tensor): Tensor of shape B, S, E

    Returns:
        Tensor: Output tensor shape B, S, E
    """ 

    def __init__(self, n_experts, n_top_experts, embed_dim, forward_mul, dropout):
        super().__init__()
        self.n_experts      = n_experts
        self.n_top_experts  = n_top_experts
        self.embed_dim      = embed_dim
        self.mid_dim        = embed_dim * forward_mul

        self.gate      = nn.Linear(embed_dim, n_experts)
        self.experts   = nn.ModuleList([FF(embed_dim, forward_mul) for _ in range(n_experts)])

    def forward(self, x):
        x_gate = F.softmax(self.gate(x), -1)                                                                # B, S, E     -->  B, S, G
        experts_weight, selected_experts = torch.topk(x_gate, self.n_top_experts, dim=-1)                   # B, S, G     -->  B, S, TG  ,   B, S, TG
        experts_weight = experts_weight/experts_weight.sum(-1, keepdims=True)                               # B, S, TG    -->  B, S, TG
        experts_weight = experts_weight.unsqueeze(-1)                                                       # B, S, TG    -->  B, S, TG, 1
         
        if self.training:    
            x = torch.stack([self.experts[i](x) for i in range(len(self.experts))])                         # G, B, S, E
            x = x.permute(1, 2, 0, 3)                                                                       # G, B, S, E  --> B, S, G, E

            selected_experts = selected_experts.unsqueeze(-1).expand(-1, -1, -1, self.embed_dim)            # B, S, TG    -->  B, S, TG, E
            x = torch.gather(x, 2, selected_experts)                                                        # B, S, G, E  -->  B, S, TG, E  Select selected experts
        else:                                                                                               
            # Note: Requires KV Cache as values for tokens before the current token will be incorrect as we are chosing the only experts selected for current sequence step. 
            x = torch.stack([self.experts[i](x) for i in selected_experts[0, -1, :].tolist()])              # TG, B, S, E
            x = x.permute(1, 2, 0, 3)                                                                       # TG, B, S, E -->  B, S, TG, E

        x = x * experts_weight                                                                              # B, S, TG, E  *   B, S, TG, E  -->  B, S, TG, E
        x = x.sum(-2)                                                                                       # B, S, TG, E -->  B, S, E
        return x
        

class Encoder(nn.Module):
    """
    Class for creating an encoder layer

    Parameters:
        embed_dim (int)      : Embedding dimension
        n_heads (int)        : Number of attention heads to use for performing MultiHeadAttention
        forward_mul (float)  : Used to calculate dimension of the hidden fc layer = embed_dim * forward_mul
        max_seq_len (int)    : Maximum sequence length used for training. This creates causal attention mask in self-attention layer (att_mask)
        dropout (float)      : Dropout parameter
    
    Input:
        x (tensor): Tensor of shape B, S, E

    Returns:
        Tensor: Output of the encoder block of shape B, S, E
    """ 

    def __init__(self, embed_dim, n_heads, forward_mul, max_seq_len, n_expert, n_top_experts, dropout):
        super().__init__()
        self.norm1      = RMSNorm(embed_dim)
        self.attention  = MultiHeadSelfAttentionWithRope(embed_dim, n_heads, max_seq_len)
        self.dropout1   = nn.Dropout(dropout)
        
        self.norm2      = RMSNorm(embed_dim)
        self.moe_ff     = MoeFF(n_expert, n_top_experts, embed_dim, forward_mul, dropout)
        self.dropout2   = nn.Dropout(dropout)

    def forward(self, x, token_loc=None):
        x = x + self.dropout1(self.attention(self.norm1(x), token_loc))                     # Skip connections
        x = x + self.dropout2(self.moe_ff(self.norm2(x)))                                   # Skip connections
        return x


class Classifier(nn.Module):
    """
    Classification module of the Transformer. Uses the embedding of the tokens to generate logits for each token.

    Parameters:
        embed_dim (int) : Embedding dimension
        n_classes (int) : Number of classes (Vocal size)
    
    Input:
        x (tensor): Tensor of shape B, S, E

    Returns:
        Tensor: Logits of shape B, S, T
    """    

    def __init__(self, embed_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = self.fc(x)                                  # B, S, E  -->  B, S, T
        return x


class LLAMA(nn.Module):
    """
    LLAMA Class.

    Parameters:
        vocab size  (int)      : Number of unique tokens
        embed_dim   (int)      : Embedding dimension
        max_seq_len (int)      : Maximum sequence length used for training.        
        n_layers    (int)      : Number of encoder blocks to use
        n_heads     (int)      : Number of attention heads to use for performing MultiHeadAttention
        forward_mul (float)    : Used to calculate dimension of the hidden fc layer = embed_dim * forward_mul
        dropout     (float)    : dropout value
    
    Input:
        x (tensor): Long Tensor of shape B, S containing token indices 

    Returns:
        Tensor: Logits of shape B, S, T
    """    

    def __init__(self, vocab_size, embed_dim, max_seq_len, n_layers, n_heads, forward_mul, n_experts=1, n_top_experts=1, dropout=0.1):
        super().__init__()
        self.max_seq_len   = max_seq_len

        self.embedding     = TokenEmbedding(vocab_size, embed_dim)
        self.encoder       = nn.ModuleList([Encoder(embed_dim, n_heads, forward_mul, max_seq_len, n_experts, n_top_experts, dropout=dropout) for _ in range(n_layers)])
        self.norm          = RMSNorm(embed_dim)                                    # Final normalization layer after the last block
        self.classifier    = Classifier(embed_dim, vocab_size)

    def reset_cache(self):
        for block in self.encoder:
            block.attention.reset_cache()

    def forward(self, x):
        if self.training:
            x = self.embedding(x)                           # B, S, E  Get word embeddings
            for block in self.encoder:                      # B, S, E  Loop through the encoders
                x = block(x)           
            x = self.norm(x)                                # B, S, E  Output normalization
            x = self.classifier(x)                          # B, S, T  Classification
        else:
            # During inference only use the last generated token. Keys and Values for previous tokens is stored in the KV Cache.
            token_loc = min(self.max_seq_len, x.shape[-1]) - 1

            x = x[:, -1].unsqueeze(-1)
            x = self.embedding(x)                           # B, S, E  Get word embeddings
            for block in self.encoder:                      # B, S, E  Loop through the encoders
                x = block(x, token_loc)           
            x = self.norm(x)                                # B, S, E  Output normalization
            x = self.classifier(x)                          # B, S, T  Classification
        return x
