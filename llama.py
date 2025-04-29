import torch
import torch.nn as nn
import torch.nn.functional as F

# (No change to helper classes: TokenEmbedding, RotatoryPositionEmbedding, etc.)

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.token_embedding(x).to(dtype=torch.float16)


class RotatoryPositionEmbedding(nn.Module):
    def __init__(self, seq_len, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len

    def build_rope(self, seq_len):
        sequence = torch.arange(seq_len).float().unsqueeze(-1)
        thetas = -torch.arange(start=0, end=self.embed_dim, step=2).float() / self.embed_dim
        thetas = torch.repeat_interleave(thetas, 2, 0)
        thetas = torch.pow(10000, thetas)
        angles = sequence * thetas
        cos_values = torch.cos(angles).unsqueeze(1).unsqueeze(0)
        sin_values = torch.sin(angles).unsqueeze(1).unsqueeze(0)
        return sin_values, cos_values

    def forward(self, x, token_loc=None):
        seq_len = x.shape[1]
        x_sin, x_cos = self.build_rope(seq_len)
        x_sin = x_sin.to(x.device).to(dtype=torch.float16)
        x_cos = x_cos.to(x.device).to(dtype=torch.float16)
        
        if token_loc is None:
            x1 = x * x_cos[:, :seq_len, :, :]
            x_shifted = torch.stack((-x[:, :, :, 1::2], x[:, :, :, ::2]), -1).reshape(x.shape)
            x2 = x_shifted * x_sin[:, :seq_len, :, :]
            x = x1 + x2
        else:
            x1 = x * x_cos[:, token_loc, :, :].unsqueeze(1)
            x_shifted = torch.stack((-x[:, :, :, 1::2], x[:, :, :, ::2]), -1).reshape(x.shape)
            x2 = x_shifted * x_sin[:, token_loc, :, :].unsqueeze(1)
            x = x1 + x2
        return x


class MultiHeadSelfAttentionWithRope(nn.Module):
    def __init__(self, embed_dim, n_attention_heads, max_seq_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_attention_heads = n_attention_heads
        self.max_seq_len = max_seq_len
        self.head_embed_dim = embed_dim // n_attention_heads

        self.queries = nn.Linear(embed_dim, self.head_embed_dim * n_attention_heads)
        self.keys = nn.Linear(embed_dim, self.head_embed_dim * n_attention_heads)
        self.values = nn.Linear(embed_dim, self.head_embed_dim * n_attention_heads)
        self.out_projection = nn.Linear(self.head_embed_dim * n_attention_heads, embed_dim)

        att_mask = torch.ones(1, 1, max_seq_len, max_seq_len)
        att_mask = torch.triu(att_mask, diagonal=1).bool()
        self.register_buffer("att_mask", att_mask)

        self.rotary_embedding = RotatoryPositionEmbedding(seq_len=max_seq_len, embed_dim=self.head_embed_dim)
        self.reset_cache()

    def reset_cache(self):
        self.cache = {'k': None, 'v': None}

    def forward(self, x, token_loc=None, kv_cache=False):
        self.last_attention_scores = None

        if self.training or not kv_cache:
            b, s, e = x.shape

            xq = self.queries(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim).to(dtype=torch.float16)
            xk = self.keys(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim).to(dtype=torch.float16)
            xv = self.values(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim).to(dtype=torch.float16)

            xq = self.rotary_embedding(xq)
            xk = self.rotary_embedding(xk)

            xq = xq.permute(0, 2, 1, 3)
            xk = xk.permute(0, 2, 1, 3)
            xv = xv.permute(0, 2, 1, 3)

            xk = xk.permute(0, 1, 3, 2)
            x_attention = torch.matmul(xq, xk) / (self.head_embed_dim ** 0.5)
            if s <= self.max_seq_len:
                mask = self.att_mask[:, :, :s, :s]
            else:
                mask = torch.triu(torch.ones(1, 1, s, s, dtype=torch.bool, device=x_attention.device), diagonal=1)
            x_attention = x_attention.masked_fill(mask, -torch.inf)
            x_attention = torch.softmax(x_attention, dim=-1)
            
            self.last_attention_scores = x_attention.detach()

            x = torch.matmul(x_attention, xv)
            x = x.permute(0, 2, 1, 3).reshape(b, s, e)
            x = self.out_projection(x).to(dtype=torch.float16)
        else:
            b, s, e = x.shape

            xq = self.queries(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim).to(dtype=torch.float16)
            xk = self.keys(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim).to(dtype=torch.float16)
            xv = self.values(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim).to(dtype=torch.float16)

            if self.cache['k'] is not None and self.cache['v'] is not None:
                xk = torch.cat((self.cache['k'], xk), 1)
                xv = torch.cat((self.cache['v'], xv), 1)

            self.cache['k'] = xk[:, -(self.max_seq_len-1):, :, :].detach()
            self.cache['v'] = xv[:, -(self.max_seq_len-1):, :, :].detach()

            xq_rotated = self.rotary_embedding(xq, token_loc)
            xk_rotated = self.rotary_embedding(xk)

            xq_rotated = xq_rotated.permute(0, 2, 1, 3)
            xk_rotated = xk_rotated.permute(0, 2, 1, 3)
            xv = xv.permute(0, 2, 1, 3)

            xk_rotated_ = xk_rotated.permute(0, 1, 3, 2)
            x_attention = torch.matmul(xq_rotated, xk_rotated_) / (self.head_embed_dim ** 0.5)
            if s <= self.max_seq_len:
                mask = self.att_mask[:, :, :s, :s]
            else:
                mask = torch.triu(torch.ones(1, 1, s, s, dtype=torch.bool, device=x_attention.device), diagonal=1)
            x_attention = x_attention.masked_fill(mask, -torch.inf)
            x_attention = torch.softmax(x_attention, dim=-1)
            
            self.last_attention_scores = x_attention.detach()

            x = torch.matmul(x_attention, xv)
            x = x.permute(0, 2, 1, 3).reshape(b, s, e)
            x = self.out_projection(x).to(dtype=torch.float16)

        return x


class RMSNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(1, 1, embed_dim))

    def forward(self, x):
        x_deno = torch.sqrt(x.pow(2).mean(-1, keepdims=True))
        x = x / (x_deno + 1e-6)
        x = x * self.weights
        return x.to(dtype=torch.float16)


class SwiGLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.fc(x)
        x = x * torch.sigmoid(x)
        return x.to(dtype=torch.float16)


class FF(nn.Module):
    def __init__(self, embed_dim=64, forward_mul=2, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * forward_mul)
        self.act = SwiGLU(embed_dim, embed_dim * forward_mul)
        self.fc2 = nn.Linear(embed_dim * forward_mul, embed_dim)

    def forward(self, x):
        x = self.act(x) * self.fc1(x)
        x = self.fc2(x)
        return x.to(dtype=torch.float16)


class MoeFF(nn.Module):
    def __init__(self, n_experts, n_top_experts, embed_dim, forward_mul, dropout):
        super().__init__()
        self.n_experts = n_experts
        self.n_top_experts = n_top_experts
        self.embed_dim = embed_dim

        self.gate = nn.Linear(embed_dim, n_experts)
        self.experts = nn.ModuleList([FF(embed_dim, forward_mul) for _ in range(n_experts)])

    def forward(self, x, kv_cache):
        x_gate = F.softmax(self.gate(x), -1)
        experts_weight, selected_experts = torch.topk(x_gate, self.n_top_experts, dim=-1)
        experts_weight = experts_weight / experts_weight.sum(-1, keepdims=True)
        experts_weight = experts_weight.unsqueeze(-1)

        if self.training or (not kv_cache):
            x = torch.stack([self.experts[i](x) for i in range(len(self.experts))])
            x = x.permute(1, 2, 0, 3)
            selected_experts = selected_experts.unsqueeze(-1).expand(-1, -1, -1, self.embed_dim)
            x = torch.gather(x, 2, selected_experts)
        else:
            x = torch.stack([self.experts[i](x) for i in selected_experts[0, -1, :].tolist()])
            x = x.permute(1, 2, 0, 3)

        x = x * experts_weight
        x = x.sum(-2)
        return x.to(dtype=torch.float16)


class Encoder(nn.Module):
    def __init__(self, embed_dim, n_heads, forward_mul, max_seq_len, n_expert, n_top_experts, kv_cache, dropout):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attention = MultiHeadSelfAttentionWithRope(embed_dim, n_heads, max_seq_len)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(embed_dim)
        self.moe_ff = MoeFF(n_expert, n_top_experts, embed_dim, forward_mul, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, token_loc=None, kv_cache=False):
        x = x + self.dropout1(self.attention(self.norm1(x), token_loc, kv_cache))
        x = x + self.dropout2(self.moe_ff(self.norm2(x), kv_cache))
        return x.to(dtype=torch.float16)


class Classifier(nn.Module):
    def __init__(self, embed_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        return self.fc(x).to(dtype=torch.float16)


class LLAMA(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len, n_layers, n_heads, forward_mul, n_experts=4, n_top_experts=1, kv_cache=True, dropout=0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedding = TokenEmbedding(vocab_size, embed_dim)
        self.encoder = nn.ModuleList([Encoder(embed_dim, n_heads, forward_mul, max_seq_len, n_experts, n_top_experts, kv_cache, dropout=dropout) for _ in range(n_layers)])
        self.norm = RMSNorm(embed_dim)
        self.classifier = Classifier(embed_dim, vocab_size)

    def reset_cache(self):
        for block in self.encoder:
            block.attention.reset_cache()

    def forward(self, x, kv_cache=True):
        if self.training or not kv_cache:
            x = self.embedding(x)
            for block in self.encoder:
                x = block(x)
            x = self.norm(x)
            x = self.classifier(x)
        else:
            token_loc = min(self.max_seq_len, x.shape[-1]) - 1
            x = x[:, -1].unsqueeze(-1)
            x = self.embedding(x)
            for block in self.encoder:
                x = block(x, token_loc, kv_cache=True)
            x = self.norm(x)
            x = self.classifier(x)
        return x.to(dtype=torch.float16)

    def get_last_layer_attention(self):
        if self.encoder[-1].attention.last_attention_scores is None:
            return torch.zeros(1)
        return self.encoder[-1].attention.last_attention_scores

    def get_token_gradients(self):
        if hasattr(self.embedding.token_embedding.weight, "grad") and self.embedding.token_embedding.weight.grad is not None:
            return self.embedding.token_embedding.weight.grad
        return None

    def get_token_embedding(self, token_id):
        """
        Returns the embedding vector for a given token id.
        """
        return self.embedding.token_embedding.weight[token_id]

    def get_logits(self, input_ids):
        self.eval()
        with torch.no_grad():
            input_ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
            logits = self.forward(input_ids, kv_cache=False)
        return logits.squeeze(0)

    def calculate_perplexity(self, input_ids):
        """
        Calculates the perplexity for a given sequence of input IDs.
        """
        self.eval()
        with torch.no_grad():
            input_tensor = input_ids[:, :-1]
            target_tensor = input_ids[:, 1:]

            logits = self.forward(input_tensor, kv_cache=False)
            log_probs = F.log_softmax(logits, dim=-1)

            # Ensure target_tensor matches the dimensions of log_probs
            target_tensor = target_tensor[:, :log_probs.size(1)]

            # Perform gather operation
            target_log_probs = log_probs.gather(2, target_tensor.unsqueeze(-1)).squeeze(-1)

            # Compute perplexity
            avg_log_prob = target_log_probs.mean()
            perplexity = torch.exp(-avg_log_prob).item()

        return perplexity

