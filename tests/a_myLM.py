import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class LinearModel(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                bias: bool = False):
        """
        手动实现的线性层，默认使用 Kaiming 初始化
        参数:
            in_features: 输入特征数
            out_features: 输出特征数
            bias: 是否使用偏置
            init_method: 初始化方法 ('kaiming', 'xavier', 'normal')
            a: Kaiming 初始化的负斜率参数
            mode: 'fan_in' 或 'fan_out'
            nonlinearity: 非线性函数 ('relu', 'leaky_relu', 'linear')
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.empty(out_features, in_features,
                                               device=device, dtype=dtype))
        torch.nn.init.kaiming_uniform_(
            self.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='linear'
        )
        ## todo: 添加偏置，了解初始化的各个函数细节，如：激活函数的负斜率（负半轴斜率）gain bound

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features,
                                                device=device, dtype=dtype))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias',  None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：y = xA^T + b"""
        return torch.nn.functional.linear(x, self.weight, self.bias)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        """
        自定义层归一化实现

        参数:
            normalized_shape: 输入形状中需要归一化的维度
            eps: 数值稳定性常数，防止除以零
            elementwise_affine: 是否添加可学习的缩放和偏移参数
        """
        super().__init__()
        self.normalized_shape = (normalized_shape, ) \
            if isinstance(normalized_shape, int) else normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            # 可学习的缩放参数 (gamma)
            self.weight = nn.Parameter(torch.ones(*self.normalized_shape))
            # 可学习的偏移参数 (beta)
            self.bias = nn.Parameter(torch.zeros(*self.normalized_shape))
        else:
            # 没有可学习的参数，注册为None
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        前向传播：应用层归一化

        公式:
            y = (x - mean) / sqrt(var + eps) * weight + bias
        """
        # 计算均值和方差（在指定的维度上）
        # 注意：PyTorch的var默认使用无偏估计（除以n-1），但我们使用有偏估计（除以n）
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # 归一化
        x_normed = (x - mean) / torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            return self.weight * x_normed + self.bias
        return x_normed

    def extra_repr(self) -> str:
        """返回层的额外信息"""
        return f'normalized_shape={self.normalized_shape}, eps={self.eps}, ' \
               f'elementwise_affine={self.elementwise_affine}'

class RMSNorm(nn.Module):
    """
    RMS Normalization (RMSNorm)
    RMSNorm(x) = x / sqrt(Var(x) + eps)
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = (normalized_shape, ) \
            if isinstance(normalized_shape, int) else normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            # 可学习的缩放参数 (gamma)
            self.weight = nn.Parameter(torch.ones(*self.normalized_shape))
        else:
            self.register_parameter('weight', None)
        self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：RMS Normalization (RMSNorm)
            y = x / RMS(x) * weight
            RMS(x) = sqrt(mean(x²) + eps)
        """
        # 均方根, L2范数计算
        # rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        ## TODO：手动实现该CUDA和Triton写法
        rms = x.norm(2, dim=-1, keepdim=True) / (x.size(-1) ** 0.5)
        rms = rms.clamp(min=self.eps)  # 确保不小于 eps

        # 归一化
        x_normed = x / rms

        if self.elementwise_affine:
            return self.weight * x_normed
        return x_normed

    def extra_repr(self) -> str:
        """返回层的额外信息"""
        return f'normalized_shape={self.normalized_shape}, eps={self.eps}, ' \
               f'elementwise_affine={self.elementwise_affine}'


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (SwishGLU)
    SwiGLU(x) = Swish(xW + b) ⊗ (xV + c)
    Swish(x) = x * sigmoid(βx), β=1
    """
    def __init__(self, d_model: int, method='round'):
        super().__init__()
        # 计算d_ff
        self.d_ff = self.calculate_d_ff(d_model, method)

        # 创建两个线性层,一次性创建更简洁高效
        ## todo: 两种方式性能测试benchmark
        # 第一个线性变换 (门控路径)
        # self.gate_proj = LinearModel(d_model, self.d_ff)
        # 第二个线性变换 (值路径)
        # self.value_proj = LinearModel(d_model, self.d_ff)
        self.input_proj = LinearModel(d_model, 2 * self.d_ff)
        self.output_proj = LinearModel(self.d_ff, d_model)
        self.swish = lambda x: x * torch.sigmoid(x)

    @staticmethod
    def calculate_d_ff(d_model, method='round'):
        base = 8/3 * d_model
        if method == 'round':
            return round(base)
        elif method == 'ceil':
            return math.ceil(base)
        elif method == 'floor':
            return math.floor(base)
        else:
            raise ValueError(f'Invalid method: {method}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：SwiGLU(x) = Swish(xW + b) ⊗ (xV + c)
        """
        # 输入投影
        x_proj = self.input_proj(x)  # (batch, seq, 2*d_ff)
        # print("x_proj的形状：", x_proj.shape)
        gate, value = x_proj.chunk(2, dim=-1)

        activated = self.swish(gate) * value
        # print("activated的形状：", activated.shape)  # (batch, seq, d_ff)

        return self.output_proj(activated) # (batch, seq, d_model)
    # 基准测试


class Dropout(nn.Module):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        """
        自定义 Dropout 层

        参数:
            p (float): 元素被置零的概率 (默认: 0.5)
            inplace (bool): 是否在原张量上进行操作 (默认: False)
        """
        super().__init__()
        if p<=0 or p>1:
            raise ValueError(f'Invalid dropout probability: {p}')

        self.p = p
        self.inplace = inplace

        #计算保留概率
        self.keep_prob = 1 - p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：Dropout
        训练模式下：以概率 p 将元素置零，并缩放剩余元素
        评估模式下：直接返回输入
        """
        if not self.training:
            return x

        # 1. 创建与输入相同形状的随机张量，值在 [0, 1) 之间
        random_tensor = torch.rand_like(x)

        # 2. 创建掩码：小于保留概率的位置为1，否则为0
        mask = (random_tensor < self.keep_prob).float()

        # 3. 应用掩码并缩放
        if self.inplace:
            x.mul(mask)
            x.div(self.keep_prob)
            return x
        else:
            return x * mask / self.keep_prob

class Function(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        手动实现 Softmax 函数
        softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))

        参数:
            logits: 输入张量，包含未归一化的分数
            dim: 应用 Softmax 的维度（默认：最后一个维度）

        返回:
            概率分布张量，形状与输入相同
        """
        max_vals = torch.max(logits, dim=dim, keepdim=True).values
        shifted = logits - max_vals

        # 计算指数
        exp_vals = torch.exp(shifted)

        # 计算分母(指数和)
        sum_exp = torch.sum(exp_vals, dim=dim, keepdim=True)

        # 计算概率
        probs = exp_vals / sum_exp
        return probs

class CausalMultiheadAttention(nn.Module):
    """
     causal multi-head attention
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, batch_first: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = Dropout(dropout)
        self.batch_first = batch_first

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Linear projections
        self.q_proj = LinearModel(embed_dim, embed_dim)
        self.k_proj = LinearModel(embed_dim, embed_dim)
        self.v_proj = LinearModel(embed_dim, embed_dim)
        self.out_proj = LinearModel(embed_dim, embed_dim)

        # Scaling factor
        # self.scaling = self.head_dim ** -0.5
        # Dropout
        self.dropout = Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: torch.Tensor = None, key_padding_mask: torch.Tensor = None):
        """
        前向传播

        参数:
            query, key, value: 输入张量
            attn_mask: 自定义注意力掩码
            key_padding_mask: 键填充掩码，处理变长序列，忽略填充部分（padding tokens）
        返回:
            attn_output: 注意力输出 [batch, seq, embed_dim]
            attn_weights: 注意力权重 [batch, num_heads, seq, seq]
        """
        if self.batch_first:
            # 转换为 [seq_len, batch, embed_dim]
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        tgt_len, batch_size, embed_dim = query.size()
        src_len = key.size(0)
        assert embed_dim == self.embed_dim, "embed_dim must be equal to self.embed_dim"

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 分割多头
        q = q.view(tgt_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(src_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(src_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)

        # 计算注意力分数
        # attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        # attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / (self.head_dim ** 0.5)

        # 应用因果编码
        casual_mask = self.generate_casual_mask(tgt_len, src_len, device=attn_scores.device)
        attn_scores += casual_mask

        if attn_mask is not None:
            attn_scores += attn_mask

        # 应用键填充
        if key_padding_mask is not None:
            # 确保掩码形状正确 [batch_size, src_len]
            attn_scores = attn_scores.view(batch_size, self.num_heads, tgt_len, src_len)
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'),
            )
            attn_scores = attn_scores.view(batch_size * self.num_heads, tgt_len, src_len)

        # 计算注意力权重
        attn_weights = Function.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权值
        # attn_output = torch.matmul(attn_weights, v)
        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).reshape(tgt_len, batch_size, embed_dim)

        # 合并多头
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, batch_size, embed_dim)

        # 输出投影
        attn_output = self.out_proj(attn_output)

        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, attn_weights

    def generate_casual_mask(self, tgt_len: int, src_len: int, device: torch.device) -> torch.Tensor:
        """
        生成一个因果编码的注意力掩码
        参数:
            tgt_len: 目标序列长度
            src_len: 源序列长度
            device: 设备
        返回:
            causal_mask: causal mask [tgt_len, src_len]
        """
        # causal_mask = torch.triu(torch.ones(tgt_len, src_len, device=device), diagonal=1).bool()
        causal_mask = torch.triu(torch.full((tgt_len, src_len), float('-inf'), device=device), diagonal=1)
        return causal_mask