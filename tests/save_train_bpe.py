from datetime import datetime

import torch
from torch import nn

from adapters import run_train_bpe
import json
import os
import time
from a_get_tokenizer import BPETokenizer

input_path = "D:\Model\cs336\\assignment1-basics-main\\tests\\fixtures\\tinystories_sample_5M.txt"
vocab_size = 1500
special_tokens = ["<|endoftext|>"] #"<unk>", "<bos>", "<eos>",
merges_file = 'merges_5M.txt'
vocab_file = 'vocab_5M.json'
vocab_file_id2token = 'vocab_5M_id2token.json'

print("run_train_bpe")
start_time = datetime.now()
# vocab, merges = run_train_bpe(input_path, vocab_size, special_tokens, num_merges=100)
# print(f"vocab({len(vocab)}): {vocab}")
# print(f"merges({len(merges)}): {merges}")
end_time = datetime.now()

# 将vocab保存为JSON文件
# vocab_dict = {}
# for token, id in vocab.items():
#     token_str = token.decode('latin-1')
#     token_str = token_str.replace(' ', 'Ġ') #替换空格
#     vocab_dict[token_str] = id
#
# with open(vocab_file, 'w', encoding='utf-8') as f:
#     json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
#

# 将merges保存为文本文件
# """OpenAI 风格的 merges，将空格替换为'Ġ' """
# with open(merges_file, 'w', encoding='utf-8') as f:
#     for a, b in merges:
#         # 同样转换为字符串
#         a_str = a.decode('latin-1').replace(' ', 'Ġ')
#         b_str = b.decode('latin-1').replace(' ', 'Ġ')
#         f.write(f"{a_str} {b_str}\n")

# ssh -T git@github.com
#start time: 2025-08-27 10:31:55.450740, end time: 2025-08-27 10:35:03.237085
# 300: 300
# 压缩比: 1.55
# 压缩比: 1.55 字节/Token
# 吞吐率: 117526.89 字节/秒

# # 将merges以16进制直接保存为文本文件
# with open(merges_file, 'w', encoding='utf-8') as f:
#     for a, b in merges:
#         # 直接写入字节的十六进制表示
#         a_hex = a.hex()
#         b_hex = b.hex()
#         f.write(f"{a_hex} {b_hex}\n")
# def load_merges(filepath):
#     merges = []
#     with open(filepath, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) == 2:
#                 # 将十六进制字符串转换回字节
#                 a = bytes.fromhex(parts[0])
#                 b = bytes.fromhex(parts[1])
#                 merges.append((a, b))
#     return merges
#
# merges_loaded = load_merges(merges_file)
# print(f"merges loaded: {merges_loaded}")

# special_tokens = ["<|endoftext|>"]
# print([s.encode('utf-8') for s in special_tokens])
# print("<|endoftext|>".encode('utf-8'))

# with open(input_path, "rb") as f:
#     text = f.read()  # text 是 bytes 对象
#     int_text = list(text)
#     print(text)
#     print(int_text)
#     print(len(text))
#     end_index = text.decode('utf-8').find("<|endoftext|>")
#     print(end_index)
#     print(text[:end_index])
#
# print(hex(ord('中')))
# print(list('中'.encode('utf-8')))
# print(chr(97))

# 读取merges
merges_loaded = []
with open(merges_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            a = parts[0].replace('Ġ', ' ').encode('latin-1')
            b = parts[1].replace('Ġ', ' ').encode('latin-1')
            merges_loaded.append((a, b))
print(merges_loaded)
print(f"start time: {start_time}, end time: {end_time}")
print(f"vocab: {vocab_size}, merges: {len(merges_loaded)}")

# 读取vocab
with open(vocab_file_id2token, 'r', encoding='utf-8') as f:
    vocab_json = json.load(f)

# for token, id in vocab_json.items():
#     vocab_loaded[id] = token.encode('latin-1')
vocab_loaded = {}
for id_str, token_str in vocab_json.items():
    # 将 id 字符串转换为整数
    token_id = int(id_str)
    token_bytes = token_str.encode('latin1')
    vocab_loaded[token_id] = token_bytes
print(vocab_loaded)

tokenizer = BPETokenizer.get_tokenizer(vocab_loaded, merges_loaded, special_tokens)

text = "hello, world"
print(tokenizer.encode(text))
print(tokenizer.decode([285, 288, 111, 44, 196, 160, 573, 705]))

token_ids_list = tokenizer.encode(text)
token_ids_tensor = torch.tensor(token_ids_list, dtype=torch.long).unsqueeze(0)

print("Token IDs 张量形状:", token_ids_tensor.shape)  # torch.Size([1, 8])
# todo: Ablation Study, 尝试找到最好的d_model值
# 嵌入维度
d_model = 128

embedding_layer = nn.Embedding(vocab_size, d_model)
dense_vectors = embedding_layer(token_ids_tensor)
