import torch
import torch.nn.functional as F

def top_k_sampling(probs, k=50):
    """
    Top-K采样

    参数:
        probs: 概率分布 [batch_size, seq_len, vocab_size]
        k: 保留的最高概率词汇数量

    返回:
        sampled_ids: 采样结果 [batch_size, seq_len]
    """
    # 获取top-k概率和索引
    topk_probs, topk_indices = torch.topk(probs, k, dim=-1)

    # 归一化top-k概率
    norm_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

    # 从top-k中采样
    sampled_indices = torch.multinomial(norm_probs, num_samples=1)

    # 获取实际token ID
    sampled_ids = topk_indices.gather(-1, sampled_indices).squeeze(-1)

    return sampled_ids


def top_p_sampling(probs, p=0.9):
    """
    Top-P (Nucleus) 采样 - 支持批次和序列

    参数:
        probs: 概率分布 [batch_size, seq_len, vocab_size]
        p: 累积概率阈值 (0.0-1.0)

    返回:
        sampled_ids: 采样结果 [batch_size, seq_len]
    """
    # 保存原始形状
    original_shape = probs.shape
    batch_size, seq_len, vocab_size = original_shape

    # 重塑为 2D: [batch_size * seq_len, vocab_size]
    flat_probs = probs.view(-1, vocab_size)

    # 对概率排序
    sorted_probs, sorted_indices = torch.sort(flat_probs, dim=-1, descending=True)

    # 计算累积概率
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    # 创建掩码：移除累积概率超过p的词汇
    mask = cum_probs <= p

    # 确保至少有一个词汇
    mask[:, 0] = True

    # 获取有效词汇
    valid_probs = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))

    # 归一化有效概率
    norm_probs = valid_probs / valid_probs.sum(dim=-1, keepdim=True)

    # 从有效词汇中采样
    sampled_indices = torch.multinomial(norm_probs, num_samples=1)

    # 获取实际token ID
    sampled_ids = sorted_indices.gather(-1, sampled_indices)

    # 恢复原始形状 [batch_size, seq_len]
    sampled_ids = sampled_ids.view(batch_size, seq_len)

    return sampled_ids


def temperature_sampling(logits, temperature=0.7):
    """
    温度控制采样

    参数:
        logits: 原始logits [batch_size, seq_len, vocab_size]
        temperature: 温度参数 (0.0-1.0)

    返回:
        sampled_ids: 采样结果 [batch_size, seq_len]
    """
    # 应用温度
    scaled_logits = logits / temperature

    # 转换为概率
    probs = F.softmax(scaled_logits, dim=-1)

    # 采样
    sampled_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return sampled_ids