import torch
import torch.nn as nn

import sampling
from a_myLM import Function, LinearModel, LayerNorm, RMSNorm, SwiGLU, CausalMultiheadAttention
from a_tokenizer import BPETokenizer
from a_transformer import TransformerBlock

tokenizer = BPETokenizer(
    vocab_id2token_file='vocab_5M_id2token.json',
    merges_file='merges_5M.txt',
    special_tokens=["<|endoftext|>"],
    space_replacement="Ġ"
)

texts = [
    "hello, world",
    "tell me a story.",
    "the quick brown fox",
    "jumps over the lazy dog"
]

device = torch.device('cuda')
batch_tensor, padding_mask = tokenizer.batch_encode(texts, padding=True, device=device)

vocab_size = tokenizer.vocab_size
d_model = 768
weights = nn.Parameter(torch.randn(vocab_size, d_model, device=device))
nn.init.normal_(weights, mean=0, std=0.02)

dense_vectors = Function.embedding(vocab_size, d_model, weights, batch_tensor)
rope_vectors = Function.apply_rope(dense_vectors)
norm_vectors = RMSNorm(d_model, device=device)(rope_vectors)

num_layers = 32
num_heads = 8
blocks = nn.Sequential(*[
    TransformerBlock(d_model, num_heads, dropout=0.1, batch_first=True, device=device)
    for _ in range(num_layers)
])

for i, block in enumerate(blocks):
    print(f"通过块 {i+1} 前形状: {norm_vectors.shape}")
    norm_vectors = block(norm_vectors, padding_mask)
    print(f"通过块 {i+1} 后形状: {norm_vectors.shape}")

final_norm = RMSNorm(d_model)(norm_vectors)
logits = LinearModel(d_model, vocab_size)(final_norm)
probs = Function.softmax(logits, dim=-1)

predicted_ids = sampling.top_p_sampling(probs, p=0.9)
if predicted_ids.dim() == 1:
    predicted_ids = predicted_ids.unsqueeze(0) # 单样本转为多样本
id_lists = predicted_ids.tolist()

output_text = [tokenizer.decode(ids) for ids in id_lists]
for i, text in enumerate(output_text):
    print(f"样本 {i+1} 预测文本: {text}")


