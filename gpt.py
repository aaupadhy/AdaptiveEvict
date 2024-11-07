import torch
import torch.nn as nn

# T  -->  Number of Tokens / Vocab Size
# B  -->  Batch Size
# E  -->  Embedding Dimension
# S  -->  Sequence Length   
# Q  -->  Query Sequence length (equal to S for self-attention)
# K  -->  Key Sequence length   (equal to S for self-attention)
# V  -->  Value Sequence length (equal to S for self-attention)
# H  -->  Number of heads
# HE -->  Head Embedding Dimension = E/H


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


class SinusoidalPositionalEncoding(nn.Module):
    """
    Class for creating sinusoidal positional encoding.

    Parameters:
        max_seq_len (int) : Maximum sequence length used for training. This is used to cache sinusoidal positional encoding.
        embed_dim   (int) : Embedding dimension

    Input:
        x (tensor): Tensor of shape B, S, E containing token embeddings 
    
    Returns:
        Tensor: Embedding of the tokens of shape B, S, E after adding positional encoding
    """   

    def __init__(self, max_seq_len, embed_dim):
        super().__init__()

        pos_embedding = self.generate_sin_encoding(max_seq_len, embed_dim)                          # 1, S, E
        self.register_buffer("pos_embedding", pos_embedding)                                        # 1, S, E

    def generate_sin_encoding(self, seq_len, embed_dim):
        # Sequence
        sequence = torch.arange(seq_len).reshape(-1, 1)                                             # S, 1

        # Denominator
        denominator = torch.pow(10000, torch.arange(0, embed_dim, 2) / embed_dim)                   # 1, E//2
        denominator = sequence / denominator                                                        # S, E//2

        # Create an empty tensor and fill with sin and cos values as per sinusoidal encoding
        pos_embedding = torch.zeros(1, sequence.shape[0], embed_dim)                                # 1, S, E
        pos_embedding[:, :, ::2]  = torch.sin(denominator)                                          # 1, S, E//2
        pos_embedding[:, :, 1::2] = torch.cos(denominator)                                          # 1, S, E//2                              
        return pos_embedding                                                                        # 1, S, E

    def forward(self, x):
        x = x + self.pos_embedding[:, :x.shape[1], :]                                               # B, S, E  +  1, S, E  -->  B, S, E
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Class for computing Multi head Self-Attention. Applies causal attention mask so tokens can attend only to current or earlier tokens.

    Parameters:
        max_seq_len       (int) : Maximum sequence length used for training. This creates causal attention mask (att_mask)
        n_attention_heads (int) : Number of attention heads to use for performing MultiHeadAttention
        embed_dim         (int) : Embedding dimension
    
    Input:
        x (tensor): Tensor of shape B, S, E

    Returns:
        Tensor: Output after Self-Attention Module of shape B, S, E
    """  

    def __init__(self, max_seq_len, n_attention_heads, embed_dim):
        super().__init__()
        self.embed_dim         = embed_dim
        self.n_attention_heads = n_attention_heads
        self.head_embed_dim    = embed_dim // n_attention_heads

        self.queries           = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)   # Queries projection
        self.keys              = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)   # Keys projection
        self.values            = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)   # Values projection
        self.out_projection    = nn.Linear(self.head_embed_dim * self.n_attention_heads, self.embed_dim)   # Out projection

        # Mask to hide future words from current word / Causal Attention
        att_mask = torch.ones(1, 1, max_seq_len, max_seq_len)                               # 1, 1, S, S
        att_mask = torch.triu(att_mask, diagonal=1).bool()                                  # 1, 1, S, S
        self.register_buffer("att_mask", att_mask)                                          # Register_buffer for easy switching of device

    def forward(self, x):
        b, s, e = x.shape                                                                   # Note: Here Q, K and V are all equal to S (Not in Llama due to Grouped Query Attention.)

        xq = self.queries(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)     # B, Q, E      -->  B, Q, (H*HE)  -->  B, Q, H, HE
        xq = xq.permute(0, 2, 1, 3)                                                         # B, Q, H, HE  -->  B, H, Q, HE
        xk = self.keys(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)        # B, K, E      -->  B, K, (H*HE)  -->  B, K, H, HE
        xk = xk.permute(0, 2, 1, 3)                                                         # B, K, H, HE  -->  B, H, K, HE
        xv = self.values(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)      # B, V, E      -->  B, V, (H*HE)  -->  B, V, H, HE
        xv = xv.permute(0, 2, 1, 3)                                                         # B, V, H, HE  -->  B, H, V, HE

        # Compute Attention presoftmax values
        xk = xk.permute(0, 1, 3, 2)                                                         # B, H, K, HE  -->  B, H, HE, K
        x_attention = torch.matmul(xq, xk)                                                  # B, H, Q, HE   @   B, H, HE, K   -->  B, H, Q, K    (Matmul tutorial eg: A, B, C, D    A, B, E, F  -->  A, B, C, F   if D==E)

        x_attention /= float(self.head_embed_dim) ** 0.5                                    # Scale presoftmax values for stability

        # Apply Mask on future words / Causal attention
        x_attention = x_attention.masked_fill(self.att_mask[:, :, :s, :s], -torch.inf)      # B, H, Q, K    *   1, 1, S, S    -->  B, H, Q, K      Fill future values with -inf and Trim causal mask for smaller sequences

        x_attention = torch.softmax(x_attention, dim=-1)                                    # Compute Attention Matrix

        x = torch.matmul(x_attention, xv)                                                   # B, H, Q, K    @   B, H, V, HE   -->  B, H, Q, HE     Compute Attention product with Values

        # Format the output
        x = x.permute(0, 2, 1, 3)                                                           # B, H, Q, HE  -->  B, Q, H, HE
        x = x.reshape(b, s, e)                                                              # B, Q, H, HE  -->  B, Q, (H*HE)

        x = self.out_projection(x)                                                          # B, Q, (H*HE) -->  B, Q, E
        return x


class Encoder(nn.Module):
    """
    Class for creating an encoder layer

    Parameters:
        embed_dim   (int)   : Embedding dimension
        n_heads     (int)   : Number of attention heads to use for performing MultiHeadAttention
        forward_mul (float) : Used to calculate dimension of the hidden fc layer = embed_dim * forward_mul
        max_seq_len (int)   : Maximum sequence length used for training. This creates causal attention mask in self-attention layer (att_mask)
        dropout     (float) : Dropout parameter
    
    Input:
        x (tensor): Tensor of shape B, S, E

    Returns:
        Tensor: Output of the encoder block of shape B, S, E
    """ 

    def __init__(self, embed_dim, n_heads, forward_mul, max_seq_len, dropout=0.0):
        super().__init__()

        self.attention  = MultiHeadSelfAttention(max_seq_len, n_heads, embed_dim)
        self.norm1      = nn.LayerNorm(embed_dim)
        self.dropout1   = nn.Dropout(dropout)
        
        self.ffn        = nn.Sequential(*[nn.Linear(embed_dim, embed_dim * forward_mul), 
                                          nn.GELU(), 
                                          nn.Linear(embed_dim * forward_mul, embed_dim)])
        self.norm2      = nn.LayerNorm(embed_dim)
        self.dropout2   = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout1(self.norm1(self.attention(x)))                                # Skip connections
        x = x + self.dropout2(self.norm2(self.ffn(x)))                                      # Skip connections
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
        x = self.fc(x)                                                                  # B, S, E  -->  B, S, T
        return x


class GPT(nn.Module):
    """
    GPT Class.

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

    def __init__(self, vocab_size, embed_dim, max_seq_len, n_layers, n_heads, forward_mul, dropout=0.1):
        super().__init__()
        self.embedding     = TokenEmbedding(vocab_size, embed_dim)
        self.pos_embedding = SinusoidalPositionalEncoding(max_seq_len, embed_dim)
        self.encoder       = nn.ModuleList([Encoder(embed_dim, n_heads, forward_mul, max_seq_len, dropout=dropout) for _ in range(n_layers)])
        self.classifier    = Classifier(embed_dim, vocab_size)

    def forward(self, x, kv_cache=None):                # B, S
        x = self.embedding(x)                           # B, S, E  Get word embeddings
        for block in self.encoder:                      # B, S, E  Loop through the encoders
            x = block(x)           
        x = self.classifier(x)                          # B, S, T  Classification
        return x
