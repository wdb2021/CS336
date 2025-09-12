import torch
import torch.nn as nn
from a_myLM import Dropout, Function, LinearModel, RMSNorm, SwiGLU, CausalMultiheadAttention

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, batch_first: bool = False,device: torch.device = None, d_ff=None, method='round'):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    def forward(self, hidden_states: torch.Tensor, key_padding_mask: torch.Tensor = None):
        # 1. MHA
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)

        attn_output = self.attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask)
        hidden_states = attn_output + residual

        # 2. FFN
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)

        ffn_output = self.ffn(hidden_states)
        hidden_states = ffn_output + residual

        return hidden_states
