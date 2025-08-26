from adapters import run_train_bpe
import json

input_path = "D:\Model\cs336\\assignment1-basics-main\\tests\\fixtures\\tinystories_sample.txt"
vocab_size = 280
special_tokens = ["<|endoftext|>"] #"<unk>", "<bos>", "<eos>",
print("run_train_bpe")
vocab, merges = run_train_bpe(input_path, vocab_size, special_tokens, num_merges=100)
print(f"vocab({len(vocab)}): {vocab}")
print(f"merges({len(merges)}): {merges}")

# 将vocab保存为JSON文件
# vocab_dict = {id: token.decode('latin-1') for id, token in vocab.items()}
# with open('vocab.json', 'w', encoding='utf-8') as f:
#     json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

# 将merges保存为文本文件
with open('merges.txt', 'w', encoding='utf-8') as f:
    for a, b in merges:
        # 直接写入字节的十六进制表示
        a_hex = a.hex()
        b_hex = b.hex()
        f.write(f"{a_hex} {b_hex}\n")

def load_merges(filepath):
    merges = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                # 将十六进制字符串转换回字节
                a = bytes.fromhex(parts[0])
                b = bytes.fromhex(parts[1])
                merges.append((a, b))
    return merges

merges_loaded = load_merges("merges.txt")
print(f"merges loaded: {merges_loaded}")

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