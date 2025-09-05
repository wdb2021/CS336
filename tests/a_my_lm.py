import json
import tokenize
import torch.nn as nn
import torch.nn.functional as F
import adapters

import torch
from a_get_tokenizer import BPETokenizer

input_path = "D:\Model\cs336\\assignment1-basics-main\\tests\\fixtures\\tinystories_sample_5M.txt"
vocab_size = 1500
special_tokens = ["<|endoftext|>"] #"<unk>", "<bos>", "<eos>",
merges_file = 'merges_5M.txt'
vocab_file = 'vocab_5M.json'
vocab_file_id2token = 'vocab_5M_id2token.json'

# 读取merges
merges_loaded = []
with open(merges_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            a = parts[0].replace('Ġ', ' ').encode('latin-1')
            b = parts[1].replace('Ġ', ' ').encode('latin-1')
            merges_loaded.append((a, b))

# 读取vocab
with open(vocab_file_id2token, 'r', encoding='utf-8') as f:
    vocab_json = json.load(f)
vocab_loaded = {}
for id_str, token_str in vocab_json.items():
    # 将 id 字符串转换为整数
    token_id = int(id_str)
    token_bytes = token_str.encode('latin1')
    vocab_loaded[token_id] = token_bytes

tokenizer = BPETokenizer.get_tokenizer(vocab_loaded, merges_loaded, special_tokens)

# 单样本预测
text = "hello, world"  #[285, 288, 111, 44, 196, 160, 573, 705]
token_ids_tensor = torch.tensor(tokenizer.encode(text), dtype=torch.long).unsqueeze(0)
print("token_ids_tensor 张量形状:", token_ids_tensor.shape)
print(token_ids_tensor.dim())

# 多样本预测
texts = [
    "hello, world",    #[285, 288, 111, 44, 196, 160, 573, 705]
    "tell me a story.",     # [116, 635, 196, 160, 601, 196, 160, 97, 196, 160, 546, 805, 46]
    "the quick brown fox",
    "jumps over the lazy dog"
]
token_ids_list = [tokenizer.encode(text) for text in texts]
token_ids_tensors = [torch.tensor(ids, dtype=torch.long) for ids in token_ids_list]

# 批次处理(单批次或多批次)
batch_tensor = nn.utils.rnn.pad_sequence(token_ids_tensors, batch_first=True, padding_value=0)
print("批次张量形状:", batch_tensor.shape)  # torch.Size([2, 13])
print("批次张量内容:\n", batch_tensor)

# 嵌入维度
d_model = 128

# embedding_layer = nn.Embedding(vocab_size, d_model)
# dense_vectors = embedding_layer(token_ids_tensor)
# weights = torch.randn(vocab_size, d_model)
# 权重初始化
weights = nn.Parameter(torch.randn(vocab_size, d_model))
nn.init.normal_(weights, mean=0, std=0.02)  # xavier初始化：nn.init.xavier_normal_(weights)

dense_vectors = adapters.run_embedding(vocab_size, d_model, weights, batch_tensor)
print("嵌入张量形状:", dense_vectors.shape)
rope_vectors = tokenizer.apply_rope(dense_vectors)
# print(rope_vectors)
print("RoPE 处理后形状:", rope_vectors.shape)  #RoPE 处理后形状: torch.Size([1, 8, 128])

# norm_vectors = F.normalize(rope_vectors, p=2, dim=-1)  #（L2归一化）余弦相似度计算，k-means聚类
layer_norm = nn.LayerNorm(d_model)   #（层归一化）
norm_vectors = layer_norm(rope_vectors) # 等价于：norm_vectors = nn.LayerNorm(d_model)(rope_vectors)

## todo: 实现层归一化
# print(norm_vectors)
print("归一化处理后形状:", norm_vectors.shape)  # 归一化处理后形状: torch.Size([1, 8, 128])

# 自注意力子层（Pre-LN）
multihead_attn = nn.MultiheadAttention(
    embed_dim=d_model,
    num_heads=4,
    dropout=0.1,      # 防止过拟合
    batch_first=True  # 输入输出为[batch, seq, features]
)
print(multihead_attn)
## todo: ablation study -- 尝试不同的attention heads数量

attn_output, attn_weights = multihead_attn(
    query=norm_vectors,  # 输入
    key=norm_vectors,    # 与query相同
    value=norm_vectors,  # 与query相同
    need_weights=True,         # 返回注意力权重
    # avarage_attn_weights=True,
)

print("注意力输出形状:", attn_output.shape)  # [batch_size, seq_len, d_model]  torch.Size([1, 8, 128])
print("attn_weights shape:", attn_weights.shape)  # [batch_size, query_len, key_len]  torch.Size([1, 8, 8])

residual_attn = attn_output + rope_vectors  # 残差连接使用原始输入
print("残差输出形状:", residual_attn.shape)

layer_norm = nn.LayerNorm(d_model)  # 两个实例的开销？
norm_ffn = layer_norm(residual_attn) # norm_output = nn.LayerNorm(d_model)(residual_output)
print("第二次归一化处理后形状:", norm_ffn.shape)

# 前馈网络子层（Pre-LN）
d_ff = d_model * 4
ffn = nn.Sequential(
    nn.Linear(d_model, d_ff),
    nn.ReLU(),
    nn.Linear(d_ff, d_model),
)
ffn_output = ffn(norm_ffn)
print("全连接输出形状:", ffn_output.shape)
residual_final = residual_attn + ffn_output
print("残差输出形状:", residual_final.shape)

# 对最后一层输出归一化
final_norm = nn.LayerNorm(d_model)(residual_final)
print("第三次归一化处理后形状:", final_norm.shape)

output_layer = nn.Linear(d_model, vocab_size)
logits = output_layer(final_norm)
print("输出形状:", logits.shape)

probs = F.softmax(logits, dim=-1)
print("概率形状:", probs.shape)

predicted_ids = torch.argmax(probs, dim=-1)
print("预测ID:", predicted_ids)

if predicted_ids.dim() == 1:
    predicted_ids = predicted_ids.unsqueeze(0) # 单样本转为多样本
id_lists = predicted_ids.tolist()

# for i, ids in enumerate(id_lists):
#     print(f"样本 {i+1} 预测ID: {ids}")

# 输出文本
output_text = [tokenizer.decode(ids) for ids in id_lists]
for i, text in enumerate(output_text):
    print(f"样本 {i+1} 预测文本: {text}")