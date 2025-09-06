import torch
from torch import nn
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

