#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int num_elements,
    const int dim_size,
    const float eps) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    const int batch_idx = idx / dim_size;
    const int dim_idx = idx % dim_size;

    // 共享内存存储平方和
    extern __shared__ float shared_sum[];

    // 计算局部平方
    float local_val = static_cast<float>(input[idx]);
    float local_sq = local_val * local_val;

    // 线程内归约
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + offset];
    }
        __syncthreads();
    }

    // 第一个线程计算均方根
    if (threadIdx.x == 0) {
        float mean_sq = shared_sum[0] / dim_size;
        float rms = sqrtf(mean_sq + eps);

        // 广播到所有线程
        for (int i = 0; i < blockDim.x; i++) {
            output[batch_idx * dim_size + i] = rms;
        }
    }
}

torch::Tensor rms_norm_cuda(torch::Tensor input, int dim, float eps) {
    // 输入检查
    TORCH_CHECK(input.is_cuda(), "输入必须在CUDA设备上");

    // 获取维度信息
    auto sizes = input.sizes();
    int dim_size = sizes[dim];
    int num_elements = input.numel();
    int num_batches = num_elements / dim_size;

    // 准备输出
    auto output = torch::empty_like(input);

    // 启动内核
    dim3 blocks((num_elements + 255) / 256);
    dim3 threads(256);
    size_t shared_mem_size = threads.x * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_cuda", ([&] {
        rms_norm_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            num_elements,
            dim_size,
            eps
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm_cuda", &rms_norm_cuda, "RMS归一化(CUDA)");
}