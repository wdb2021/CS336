import json
import tokenize
import torch.nn as nn
import torch.nn.functional as F

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

text = "hello, world"
print(tokenizer.encode(text))
print(tokenizer.decode([285, 288, 111, 44, 196, 160, 573, 705]))

token_ids_list = tokenizer.encode(text)
token_ids_tensor = torch.tensor(token_ids_list, dtype=torch.long).unsqueeze(0)
print("Token IDs 张量形状:", token_ids_tensor.shape)
print(token_ids_tensor.dim())

# 嵌入维度
d_model = 128
embedding_layer = nn.Embedding(vocab_size, d_model)
dense_vectors = embedding_layer(token_ids_tensor)

print("embedding dense vectors形状:", dense_vectors.shape)
rope_vectors = tokenizer.apply_rope(dense_vectors)
print("RoPE 处理后形状:", rope_vectors.shape)
print(rope_vectors)