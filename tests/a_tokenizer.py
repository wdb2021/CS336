import re
from typing import Any, Dict, List, Optional, Tuple
import json
import torch
import torch.nn as nn

class BPETokenizer:
    def __init__(
            self,
            # vocab: Dict[int, bytes],
            # merges: List[Tuple[bytes, bytes]],
            vocab_id2token_file: str,
            merges_file: str,
            special_tokens: Optional[List[str]] = None,
            space_replacement: str = "Ġ"):
        """
        初始化 BPE 分词器（使用 Ġ 表示空格）
        Args:
            vocab_file: 词汇表文件路径
            merges_file: 合并规则文件路径
            special_tokens: 特殊 token 列表
            space_replacement: 空格替换字符
            vocab: 词汇表字典，格式为 {token_id: token_bytes}
            merges: 合并规则列表，格式为 [(token1, token2), ...]
            special_tokens: 特殊 token 列表（如有）
        """
        self.vocab_id2token_file = vocab_id2token_file
        self.merges_file = merges_file
        self.space_replacement = space_replacement
        self.special_tokens = special_tokens or []

        self.vocab = self._load_vocab()
        self.merges = self._load_merges()
        self.vocab_size = len(self.vocab)

        self.id2token = self.vocab
        self.token2id = {token: id for id, token in self.vocab.items()}


        if special_tokens:
            next_id = max(self.vocab.keys()) + 1
            for token_str in special_tokens:
                token_bytes = token_str.encode('utf-8')
                if token_bytes not in self.token2id:
                    self.id2token[next_id] = token_bytes
                    self.token2id[token_bytes] = next_id
                    next_id += 1

    def _load_vocab(self) -> Dict[int, bytes]:
        # 读取vocab
        try:
            with open(self.vocab_id2token_file, 'r', encoding='utf-8') as f:
                vocab_json = json.load(f)
            vocab_loaded = {}
            for id_str, token_str in vocab_json.items():
                token_id = int(id_str)
                token_bytes = token_str.encode('latin1')
                vocab_loaded[token_id] = token_bytes

            return vocab_loaded

        except FileNotFoundError:
            raise FileNotFoundError(f"词汇表文件未找到: {self.vocab_id2token_file}")
        except json.JSONDecodeError:
            raise ValueError(f"词汇表文件格式错误: {self.vocab_id2token_file}")

    def _load_merges(self) -> List[Tuple[bytes, bytes]]:
        # 读取vocab
        try:
            merges_loaded = []
            with open(self.merges_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        a = parts[0].replace('Ġ', ' ').encode('latin-1')
                        b = parts[1].replace('Ġ', ' ').encode('latin-1')
                        merges_loaded.append((a, b))
            return merges_loaded

        except FileNotFoundError:
            raise FileNotFoundError(f"合并规则文件未找到: {self.merges_file}")

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
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

        # if add_special_tokens:
            # token_ids = [self.token2id[b'<s>']] + token_ids + [self.token2id[b'</s>']]
        return token_ids

    def batch_encode(
            self,
            texts: List[str],
            add_special_tokens: bool = True,
            padding: bool = True,
            max_length: Optional[int] = None,
            device: torch.device = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量编码文本

        参数:
            texts: 文本列表
            add_special_tokens: 是否添加特殊 token
            padding: 是否填充到相同长度
            max_length: 最大长度限制

        返回:
            包含 input_ids 和 attention_mask 的字典
        """
        # 编码文本
        token_ids_list = [self.encode(text, add_special_tokens) for text in texts]

        # 长度限制
        if max_length is not None:
            token_ids_list = [token_ids[:max_length] for token_ids in token_ids_list]

        # 转换为张量
        token_ids_tensors = [
            torch.tensor(token_ids, dtype=torch.long)
            for token_ids in token_ids_list
        ]

        if padding:
            batch_tensor = nn.utils.rnn.pad_sequence(
                token_ids_tensors,
                batch_first=True,
                padding_value=0  # 默认填充值为0
            )
            padding_mask = (batch_tensor != 0)
        else:
            # 不填充时，堆叠不同长度的张量
            batch_tensor = torch.nested.nested_tensor(token_ids_tensors)

            # 创建注意力掩码（全1）
            padding_mask = torch.nested.nested_tensor([
                torch.ones(len(ids), dtype=torch.long) for ids in token_ids_list
            ])

        if device:
            batch_tensor = batch_tensor.to(device)
            padding_mask = padding_mask.to(device)

        return batch_tensor, padding_mask


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
