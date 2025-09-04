import torch
import math
#
# a = torch.tensor([1,2,3])
# b = torch.tensor([4,5,6])
#
# add = a + b
# sub = a - b
# mul = a * b
# div = a / b
# dot = torch.dot(a,b)
#
# matrix = torch.tensor([[1,2,3],[4,5,6]])
# scalar = torch.tensor([1,2,3])
#
# a_col = a.view(-1, 1)
# b_row = b.view(1, -1)
# print(a_col.shape, b_row.shape)
# matmul = a_col @ b_row
# reverse_matmul = torch.matmul(b_row, a_col)
# print(matmul)
# print(f"dot product: {torch.dot(b_row.view(-1), a_col.view(-1))}")
# print(f"sum_all: {matmul.sum()}")
# print(f"sum_dim: {matmul.sum(dim=0)}")
# print(f"mean_dim: {matmul.float().mean(dim=1)}")
# max_val, max_idx = matmul.max(dim=0)
# max_val_cow, max_idx_cow = matmul.max(dim=1)
# print(f"max_val: {max_val}, max_idx: {max_idx}")
# print(f"max_val_cow: {max_val_cow}, max_idx_cow: {max_idx_cow}")
#
# tensor = torch.arange(0, 9)
# reshaped = tensor.view(3, 3)
# print(reshaped)
# row = reshaped[0]
# col = reshaped[:, 0]
# submatrix = reshaped[0:2, 0:2]
# mask = reshaped > 5
# filter = reshaped[mask]
# print(row, col)
# print(submatrix)
# print(mask)
# print(f"filter: {filter}")
# print(f"~filter: {~filter+1}")
# print(f"filter reverse: {reshaped[~mask].view(-1, 3)}")

# x = torch.tensor(2.0, requires_grad=True)
# y = torch.tensor(3.0, requires_grad=True)
#
# z = x**2 + y**3 + x*y
# z.backward()
# print(x.grad)
# print(y.grad)
#
# model_params = torch.randn(3, requires_grad=True)
# print(model_params)
# input_data = torch.tensor([1.0, 2.0, 3.0])
#
# predictions = torch.dot(model_params, input_data)
# target = torch.tensor(10.0)
#
# loss = (predictions - target) ** 2
# loss.backward()
# print(model_params.grad)


# def compute_gradient():
#     """
#     计算函数 f(x,y) = sin(x² + y²) 在(1,2)处的梯度
#     """
#
#     x = torch.tensor(1.0, requires_grad=True)
#     y = torch.tensor(2.0, requires_grad=True)
#
#     z = torch.sin(x**2 + y**2)
#     z.backward()
#     print(f"grad_x: {x.grad}, {x.grad.item()}")
#     print(f"grad_y: {y.grad}, {y.grad.item()}")
#
#     return x.grad, y.grad
#
# grad_x, grad_y = compute_gradient()
# print(f"函数在 (1,2) 处的梯度:")
# print(f"∂f/∂x = {grad_x:.6f}")
# print(f"∂f/∂y = {grad_y:.6f}")
#
# # 数学验证
# # f = sin(x² + y²)
# # ∂f/∂x = cos(x² + y²) * 2x
# # ∂f/∂y = cos(x² + y²) * 2y
#
# x_val, y_val = 1.0, 2.0
# cos_val = math.cos(x_val**2 + y_val**2)
# expected_grad_x = cos_val * 2 * x_val
# expected_grad_y = cos_val * 2 * y_val
#
# print("\n数学验证:")
# print(f"预期 ∂f/∂x = {expected_grad_x:.6f}")
# print(f"预期 ∂f/∂y = {expected_grad_y:.6f}")

# class SimpleReLULayer:
#     def __init__(self, input_size, output_size):
#         self.W = torch.randn(output_size, input_size, requires_grad=True)
#         self.b = torch.randn(output_size, requires_grad=True)
#
#     def forward(self, x):
#         """
#         前向传播
#         :param x: 输入张量
#         :return: 输出张量
#         """
#         # 线性变换
#         linear = torch.matmul(self.W, x) + self.b
#
#         # ReLU激活
#         self.output = torch.relu(linear)
#         return self.output
#
#     def backward(self, upstream_grad):
#         """
#         反向传播
#         :param upstream_grad: 上游梯度
#         :return: 下游梯度
#         """
#         # 计算ReLU的梯度
#         relu_grad = (self.output > 0).float()
#
#         # 计算线性部分的梯度
#         linear_grad = upstream_grad * relu_grad
#
#         # 计算W和b的梯度
#         self.W.grad = torch.outer(linear_grad, self.x)  # ∂L/∂W = ∂L/∂z * ∂z/∂W
#         self.b.grad = linear_grad  # ∂L/∂b = ∂L/∂z * ∂z/∂b
#
#         # 计算输入的梯度
#         input_grad = torch.matmul(self.W.T, linear_grad)  # ∂L/∂x = ∂L/∂z * ∂z/∂x
#         return input_grad
#
#     def manual_backward(self, x, upstream_grad):
#         """
#         手动计算梯度
#         """
#         # 保存输入用于反向传播
#         self.x =x.clone().detach()
#
#         self.forward(x)
#
#         return self.backward(upstream_grad   )

# def test_relu_layer():
#     layer = SimpleReLULayer(input_size=2, output_size=3)
#
#     x = torch.tensor([1.0, 2.0])
#
#     output = layer.forward(x)
#     print("\n前向传播结果:")
#     print(f"输入: {x}")
#     print(f"权重 W:\n{layer.W}")
#     print(f"偏置 b: {layer.b}")
#     print(f"输出: {output}")
#
#     upstream_grad = torch.tensor([0.5, -1.0, 0.3])
#
#     input_grad = layer.manual_backward(x, upstream_grad)
#
#     print("\n反向传播结果:")
#     print(f"权重梯度 ∂L/∂W:\n{layer.W.grad}")
#     print(f"偏置梯度 ∂L/∂b: {layer.b.grad}")
#     print(f"输入梯度 ∂L/∂x: {input_grad}")
#     # 使用PyTorch自动求导验证
#     # 重新创建层和输入
#     layer_auto = SimpleReLULayer(input_size=2, output_size=3)
#     x_auto =x.clone().requires_grad_(True)
#
#     output_auto = layer_auto.forward(x_auto)
#
#     output_auto.backward(upstream_grad)
#
#     print("\nPyTorch自动求导验证:")
#     print(f"权重梯度 ∂L/∂W:\n{layer_auto.W.grad}")
#     print(f"偏置梯度 ∂L/∂b: {layer_auto.b.grad}")
#     print(f"输入梯度 ∂L/∂x: {x_auto.grad}")
#
# # 运行测试
# test_relu_layer()

# def gradient_check(layer, x, eps=1e-4):
#     """
#     梯度检查
#     :param layer: 层对象
#     :param x: 输入张量
#     :param eps: 误差
#     :return:
#     """
#
#     W_orig = layer.W.clone()
#     b_orig = layer.b.clone()
#
#     numerical_grad_W = torch.zeros_like(layer.W)
#     for i in range(layer.W.shape[0]):
#         for j in range(layer.W.shape[1]):
#             layer.W[i, j] = W_orig[i, j] + eps
#             output_plus = layer.forward(x)
#
#             layer.W[i, j] = W_orig[i, j] - eps
#             output_minus = layer.forward(x)
#
#             numerical_grad_W[i, j] = (output_plus - output_minus).sum() / (2 * eps)
#
#             layer.W[i, j] = W_orig[i, j]
#
#     analytic_grad = layer.backward(torch.ones_like(layer.output))
#
#     diff_W = torch.norm(numerical_grad_W - layer.W.grad) / torch.norm(numerical_grad_W + layer.W.grad)
#     print(f"权重梯度差异: {diff_W.item():.6f}")

import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = SimpleNN(input_size=784, hidden_size=256, output_size=10)
print(model)

input_data = torch.randn(32,784)
print(f"input_data: {input_data}")
output = model(input_data)
print(f"output: {output}")