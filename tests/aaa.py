from collections import defaultdict
text = """So Ben took the vase home and he was so proud of it! He called his friends over and showed them the amazing vase. All his friends thought the vase was beautiful and couldn't believe how lucky Ben was.
And that's how Ben found an amazing vase in the store!<|endoftext|>"""
special_token = b"|endoftext|"

data_bytes = text.encode("utf-8")
print(f"原始字节: {list(data_bytes)}")
print(f"原始文本: {text}")

vocab = {i: bytes([i]) for i in range(256)}
# print(f"{vocab[97]}, {bytes([97])}, {'a'.encode('utf-8')}")
next_id = 256

tokens = [bytes([b]) for b in data_bytes]
print(f"初始tokens: {[t for t in tokens]}")

merges = []

def get_pair_frequency(tokens):
    freq = defaultdict(int)
    for i in range(len(tokens)-1):
        pair = tokens[i]+tokens[i+1]
        freq[pair] += 1
    return freq

while len(vocab) < 400:
    freq = get_pair_frequency(tokens)
    if not freq:
        break

    most_frequent_pair = max(freq, key=freq.get)
    A, B = most_frequent_pair
    frequency = freq[most_frequent_pair]
    print(f"合并: {A} + {B} (频率: {frequency})")

    new_token = A+B
    vocab[next_id] = new_token
    next_id += 1

    merges.append((A, B))

    new_tokens = []
    i = 0
    while i < len(tokens):
        if tokens[i]==A and tokens[i+1]==B and i < len(tokens)-1:
            new_tokens.append(new_token)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    tokens = new_tokens

    # 打印当前状态（仅用于演示）
    print(f"当前词汇表大小: {len(vocab)}")
    print(f"当前tokens: {[t for t in tokens[:20]]}...")
    print("---")

print("\n最终词汇表大小:", len(vocab))
print("合并操作数量:", len(merges))
print("前10个合并操作:", merges[:10])
print("后10个合并操作:", merges[-10])
