import time
import os
import multiprocessing as mp
from collections import defaultdict
from functools import partial
import numpy as np


def get_pair_frequency(tokens_chunk):
    """并行优化的频率统计函数"""
    freq = defaultdict(int)
    for i in range(len(tokens_chunk) - 1):
        pair = (tokens_chunk[i], tokens_chunk[i + 1])
        freq[pair] += 1
    return freq


def apply_merge(tokens_chunk, merge_rules):
    """并行优化的合并应用函数"""
    # 应用所有合并规则到当前分块
    for A, B in merge_rules:
        new_token = A + B
        new_tokens = []
        i = 0
        while i < len(tokens_chunk):
            if i < len(tokens_chunk) - 1 and tokens_chunk[i] == A and tokens_chunk[i + 1] == B:
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens_chunk[i])
                i += 1
        tokens_chunk = new_tokens
    return tokens_chunk


def run_train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        num_processes: int = None,
        batch_size: int = 10
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """并行优化的 BPE 训练函数"""
    start_time = time.time()

    # 初始化词汇表
    vocab = {bytes([i]): i for i in range(256)}
    next_id = 256

    # 读取文件
    with open(input_path, "rb") as f:
        text = f.read()

    original_byte_count = len(text)

    # 分割特殊标记
    text_segs = text.split(special_tokens[0].encode('utf-8'))

    # 初始令牌化
    tokens = [bytes([b]) for text_seg in text_segs for b in text_seg]
    print(f"初始tokens({len(tokens)}): {[t for t in tokens[:10]]}...")

    merges = []

    # 设置进程数
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 8)  # 最多使用8个进程

    # 分块函数
    def chunk_data(data, chunk_size):
        """将数据分割为多个块"""
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]

    while len(vocab) < vocab_size:
        # 将令牌分成多个块进行并行处理
        chunk_size = max(1000, len(tokens) // num_processes)
        chunks = list(chunk_data(tokens, chunk_size))

        # 并行计算频率
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(get_pair_frequency, chunks)

        # 合并频率结果
        freq = defaultdict(int)
        for res in results:
            for pair, count in res.items():
                freq[pair] += count

        if not freq:
            break

        # 找出多个最高频的 token 对
        batch_merges = []
        batch_indices = []
        for _ in range(min(batch_size, len(freq))):
            if not freq:
                break

            most_frequent_pair = max(freq, key=freq.get)
            batch_merges.append(most_frequent_pair)
            batch_indices.append(freq[most_frequent_pair])
            del freq[most_frequent_pair]

        # 应用批量合并
        for i, (A, B) in enumerate(batch_merges):
            frequency = batch_indices[i]
            print(f"合并: {A} + {B} (频率: {frequency})")

            new_token = A + B
            vocab[new_token] = next_id
            next_id += 1
            merges.append((A, B))

        # 应用所有新增的合并规则
        with mp.Pool(processes=num_processes) as pool:
            apply_merge_func = partial(apply_merge, merge_rules=batch_merges)
            tokens_chunks = pool.map(apply_merge_func, chunks)

        # 合并处理后的块
        tokens = []
        for chunk in tokens_chunks:
            tokens.extend(chunk)

        # 打印当前状态
        print(f"当前词汇表大小: {len(vocab)}")
        print(f"当前tokens: {[t for t in tokens[:10]]}...")
        print("---")

    # 添加特殊 token
    for special_token in special_tokens:
        special_bytes = special_token.encode('utf-8')
        if special_bytes not in vocab.values():
            vocab[special_bytes] = next_id
            next_id += 1

    # 计算压缩比
    final_token_count = len(tokens)
    compression_ratio = original_byte_count / final_token_count if final_token_count > 0 else 0

    # 计算吞吐率
    end_time = time.time()
    processing_time = end_time - start_time
    throughput = original_byte_count / processing_time if processing_time > 0 else 0

    print(f"压缩比: {compression_ratio:.2f} 字节/Token")
    print(f"吞吐率: {throughput:.2f} 字节/秒")

    return vocab, merges