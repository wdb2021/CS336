import torch
import torch.nn as nn
from a_myLM import Dropout, Function, LinearModel, RMSNorm, SwiGLU, CausalMultiheadAttention

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, batch_first: bool = False,
                 device: torch.device = None, d_ff=None, method='round'):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        # 输入归一化层
        self.norm1 = RMSNorm(embed_dim, device=device)
        self.attn = CausalMultiheadAttention(embed_dim, num_heads, dropout, batch_first, device=self.device)
        # FFN前归一化层
        self.norm2 = RMSNorm(embed_dim, device=device)
        if d_ff is not None:
            self.ffn = SwiGLU(embed_dim, d_ff, method, device)
        else:
            self.ffn = SwiGLU(embed_dim, device=device)

        self.dropout = Dropout(dropout, device=device)

    def forward(self, hidden_states: torch.Tensor, kv_cache: tuple = None,
                key_padding_mask: torch.Tensor = None):
        # 1. MHA
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)

        attn_output, new_kv_cache = self.attn(
        # attn_output = self.attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            kv_cache=kv_cache,
            key_padding_mask=key_padding_mask)
        hidden_states = attn_output + residual

        # 2. FFN
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)

        ffn_output = self.ffn(hidden_states)
        hidden_states = ffn_output + residual

        return hidden_states, new_kv_cache
        # return hidden_states

class TransformerStack(nn.Module):
    def __init__(self, num_layers: int, embed_dim: int, num_heads: int, dropout=0.1,
                 batch_first=False,device=None, d_ff=None, method='round'):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                num_heads,
                dropout,
                batch_first,
                device=self.device,
                d_ff=d_ff,
                method=method)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.batch_first = batch_first

    def forward(self, hidden_states, key_padding_mask=None, kv_caches=None):
        """
        参数:
            hidden_states: 输入张量 [batch, seq_len, embed_dim]
            key_padding_mask: 键填充掩码
            kv_caches: 每层的KV缓存列表，每个元素为 (key_cache, value_cache)

        返回:
            hidden_states: 输出张量
            new_kv_caches: 更新后的KV缓存列表
        """
        new_kv_caches = []

        if kv_caches is None:
            kv_caches = [None] * self.num_layers

        for i, layer in enumerate(self.layers):
            hidden_states, new_kv_cache = layer(
                hidden_states,
                kv_cache=kv_caches[i],
                key_padding_mask=key_padding_mask)
            new_kv_caches.append(new_kv_cache)

        return hidden_states, new_kv_caches
