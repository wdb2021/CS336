import re
from typing import Any, Dict, List, Optional, Tuple

import torch


class BPETokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: Optional[List[str]] = None):
        """
        初始化 BPE 分词器（使用 Ġ 表示空格）
        Args:
            vocab: 词汇表字典，格式为 {token_id: token_bytes}
            merges: 合并规则列表，格式为 [(token1, token2), ...]
            special_tokens: 特殊 token 列表（如有）
        """
        self.id2token = vocab
        self.token2id = {token: id for id, token in vocab.items()}

        if special_tokens:
            next_id = max(vocab.keys()) + 1
            for token_str in special_tokens:
                token_bytes = token_str.encode('utf-8')
                if token_bytes not in self.token2id:
                    self.id2token[next_id] = token_bytes
                    self.token2id[token_bytes] = next_id
                    next_id += 1

        self.merges = merges
    def encode(self, text: str) -> List[int]:
        """
        将文本编码为 token ID 列表
        Args:
            text: 输入文本
        Returns:
            编码后的 token ID 列表
        """

        # 将空格替换为 Ġ
        text = text.replace(' ', 'Ġ')

        # 将文本转换为字节序列
        byte_sequence = text.encode('utf-8')

        # 初始化为单个字节 tokens
        tokens = [bytes([b]) for b in byte_sequence]

        # 应用所有合并规则
        for a, b in self.merges:
            new_token = a + b
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i+1] == b:
                    new_tokens.append(new_token)
                    i += 2  # 跳过两个已处理的token
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # 将 tokens 转换为 token IDs
        token_ids = []
        for token in tokens:
            if token in self.token2id:
                token_ids.append(self.token2id[token])
            else:
                # 回退：拆分为单个字节并处理
                for b in token:
                    byte_token = bytes([b])
                    if byte_token in self.token2id:
                        token_ids.append(self.token2id[byte_token])
        return token_ids


    def decode(self, token_ids: List[int]) -> str:
        """
        将 token ID 列表解码为文本
        Args:
            indices: 输入 token ID 列表
        Returns:
            解码后的文本
        """
        # 获取所有 token 的字节表示并拼接
        byte_sequence = b''.join(self.id2token[id] for id in token_ids)

        # 解码为 UTF-8 字符串
        decoded_text = byte_sequence.decode('utf-8', errors='replace')

        # 将 Ġ 替换回空格
        return decoded_text.replace('Ġ', ' ')

    def get_tokenizer(
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None = None,
    ) -> Any:
        """
        创建并返回一个 BPE 分词器（使用 Ġ 表示空格）

        Args:
            vocab: 词汇表字典，格式为 {token_id: token_bytes}
            merges: 合并规则列表，格式为 [(token1, token2), ...]
            special_tokens: 特殊 token 列表（如有）

        Returns:
            BPETokenizer 实例
        """
        return BPETokenizer(vocab, merges, special_tokens)

    def apply_rope(self, x, positions=None, rope_freq=10000.0):
        """
        应用 RoPE 位置编码，简易实现（支持完整序列和自回归生成）

        参数:
            x: 输入张量 (batch_size, seq_len, d_model)
            rope_freq: RoPE 频率参数

        返回:
            添加了 RoPE 的嵌入向量
        """
        batch_size, seq_len, d_model = x.shape
        device = x.device

        # 处理位置索引
        if positions is None:
            # 默认连续位置索引[0, 1, 2, ..., seq_len-1]
            positions = torch.arange(seq_len, device=device).float().unsqueeze(0)
        else:
            # 确保位置索引形状为(batch_size, seq_len)
            positions = positions.to(device).float()
            if positions.dim() == 1:
                positions = positions.unsqueeze(0)

        # 当用户只提供了单个序列的位置索引，但需要处理多个序列时
        # 扩展位置索引以匹配批次大小
        if positions.size(0) == 1 and batch_size > 1:
            positions = positions.expand(batch_size, positions.size(1))

        # 创建频率项
        dim_indices = torch.arange(0, d_model, 2, device=device).float()
        inv_freq = 1.0 / (rope_freq ** (dim_indices / d_model))

        # 计算正余弦值
        freqs = torch.einsum('bi, j->bij', positions, inv_freq)
        sin = torch.sin(freqs)
        cos = torch.cos(freqs)

        # 将嵌入向量分为两部分
        x1 = x[:, :, 0::2]    #偶数索引
        x2 = x[..., :, 1::2]  #奇数索引

        # 应用 RoPE 旋转编码
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # 重新组合
        rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)
        rotated = rotated.reshape(batch_size, seq_len, d_model)
        # reshape可以处理非连续张量，而view只能处理连续张量
        # rotated = rotated.view(batch_size, seq_len, d_model)

        return rotated