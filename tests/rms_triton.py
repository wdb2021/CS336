import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
        x_ptr,  # 输入指针
        rms_ptr,  # 输出RMS指针
        n_elements,  # 元素总数
        dim_size,  # 维度大小
        eps,  # 数值稳定常数
        BLOCK_SIZE: tl.constexpr,
):
    # 程序ID
    pid = tl.program_id(axis=0)

    # 计算偏移
    block_start = pid * dim_size
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 边界检查 - 确保不超过当前行的范围
    mask = (offsets < (pid + 1) * dim_size) & (offsets < n_elements)

    # 加载数据
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # 计算平方
    x_sq = x * x

    # 归约计算平方和
    sum_sq = tl.sum(x_sq, axis=0)

    # 计算均方根
    mean_sq = sum_sq / dim_size
    rms = tl.sqrt(mean_sq + eps)

    # 存储结果
    tl.store(rms_ptr + pid, rms)


def rms_norm_triton(x, dim=-1, eps=1e-6):
    # 获取形状信息
    dim_size = x.size(dim)
    n_elements = x.numel()
    n_batches = n_elements // dim_size

    # 准备输出
    rms = torch.empty(n_batches, device=x.device)

    # 计算合适的 BLOCK_SIZE（2 的幂）
    # 找到大于等于 dim_size 的最小 2 的幂
    block_size = 1
    while block_size < dim_size:
        block_size *= 2

    # 启动内核
    grid = lambda meta: (n_batches,)
    rms_norm_kernel[grid](
        x, rms, n_elements, dim_size, eps,
        BLOCK_SIZE=block_size
    )

    return rms