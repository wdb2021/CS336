import time
import torch
# from matplotlib import pyplot as plt
from a_tokenizer import BPETokenizer as plt
from torch.profiler import profile, ProfilerActivity, record_function
from typing import Callable
from a_myLM import Function, LinearModel, LayerNorm, RMSNorm, SwiGLU, CausalMultiheadAttention
import rms_triton
from torch.utils.cpp_extension import load



def simple_profiler(description: str, func: Callable, num_warmups: int = 1, with_stack: bool = False):
    """
    简易性能分析工具

    参数:
        description: 操作描述
        func: 要分析的函数
        num_warmups: 预热次数
        with_stack: 是否收集调用栈信息
    """
    # 预热阶段 - 避免冷启动影响
    for _ in range(num_warmups):
        func()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 等待CUDA操作完成

    # 配置并运行性能分析
    with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA if torch.cuda.is_available() else None
            ],
            with_stack=with_stack,
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
    ) as prof:
        func()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # 生成并返回分析结果表格
    return prof.key_averages().table(
        sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
        max_name_column_width=80,
        row_limit=10
    )


def run_operation(dim: int, operation: Callable):
    """
    运行指定操作的辅助函数

    参数:
        dim: 张量维度
        operation: 要执行的操作函数
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建随机张量
    a = torch.randn(dim, dim, device=device)
    b = torch.randn(dim, dim, device=device)

    # 定义实际运行的函数
    def run():
        operation(a, b)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    return run


def run_mlp(dim: int, num_layers: int, batch_size: int, num_steps: int):
    """
    运行多层感知机(MLP)的辅助函数
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建MLP模型
    layers = []
    for _ in range(num_layers):
        layers.append(torch.nn.Linear(dim, dim))
        layers.append(torch.nn.ReLU())
    model = torch.nn.Sequential(*layers).to(device)

    # 创建输入数据
    inputs = torch.randn(batch_size, dim, device=device)

    # 定义实际运行的函数
    def run():
        for _ in range(num_steps):
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    return run


def demo_profiling():
    """演示各种操作的性能分析"""
    print("PyTorch Profiler 演示")
    print("=" * 50)

    # 1. 分析sleep操作
    sleep_func = lambda: time.sleep(0.05)  # 50ms延迟
    sleep_profile = simple_profiler("sleep", sleep_func)
    print("\n## sleep操作分析")
    print(sleep_profile)

    # 2. 分析加法操作
    add_func = lambda a, b: a + b
    add_profile = simple_profiler("add", run_operation(2048, add_func))
    print("\n## 加法操作分析 (2048x2048)")
    print(add_profile)

    # 3. 分析矩阵乘法
    matmul_func = lambda a, b: a @ b
    matmul_profile = simple_profiler("matmul", run_operation(2048, matmul_func))
    print("\n## 矩阵乘法分析 (2048x2048)")
    print(matmul_profile)

    # 4. 分析小规模矩阵乘法
    matmul_small_profile = simple_profiler("matmul_small", run_operation(128, matmul_func))
    print("\n## 矩阵乘法分析 (128x128)")
    print(matmul_small_profile)

    # 5. 分析softmax操作
    softmax_func = lambda a, b: torch.nn.functional.softmax(a + b, dim=-1)
    softmax_profile = simple_profiler("softmax", run_operation(2048, softmax_func))
    print("\n## Softmax操作分析")
    print(softmax_profile)

    # 6. 分析GELU操作
    gelu_func = lambda a, b: torch.nn.functional.gelu(a + b)
    gelu_profile = simple_profiler("gelu", run_operation(2048, gelu_func))
    print("\n## GELU操作分析")
    print(gelu_profile)

    # 7. 分析MLP模型
    if torch.cuda.is_available():
        mlp_func = run_mlp(2048, 64, 1024, 2)
    else:
        mlp_func = run_mlp(128, 16, 128, 2)

    mlp_profile = simple_profiler("mlp", mlp_func, with_stack=True)
    print("\n## MLP模型分析")
    print(mlp_profile)


class CustomProfiler:
    def __init__(self):
        self.results = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"分析将在 {'GPU' if self.device == 'cuda' else 'CPU'} 上执行")

    def profile_function(self, name: str, func: Callable, *args, **kwargs):
        """
        分析单个函数的性能

        参数:
            name: 函数名称
            func: 要分析的函数
            *args, **kwargs: 函数参数
        """
        # 预热
        for _ in range(2):
            func(*args, **kwargs)
            if self.device == "cuda":
                torch.cuda.synchronize()

        # 运行分析
        with profile(
                activities=[
                    ProfilerActivity.CPU,
                    ProfilerActivity.CUDA if self.device == "cuda" else None
                ],
                record_shapes=True,
                with_stack=True,
                experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
        ) as prof:
            with record_function(name):
                func(*args, **kwargs)
                if self.device == "cuda":
                    torch.cuda.synchronize()

        # 保存结果
        table = prof.key_averages().table(
            sort_by="cuda_time_total" if self.device == "cuda" else "cpu_time_total",
            max_name_column_width=120,
            row_limit=15
        )

        self.results[name] = {
            "table": table,
            "profiler": prof
        }

        return table

    def profile_custom_operations(self, dim=512, seq_len=128, batch_size=32):
        """分析自定义函数库中的关键操作"""
        # 创建测试数据
        x = torch.randn(batch_size, seq_len, dim, device=self.device)
        y = torch.randn(batch_size, seq_len, dim, device=self.device)

        print("\n" + "=" * 60)
        print(f"开始分析自定义函数 (dim={dim}, seq_len={seq_len}, batch_size={batch_size})")
        print("=" * 60)

        # 1. 分析RMSNorm
        rms_norm = RMSNorm(dim).to(self.device)
        rms_table = self.profile_function("RMSNorm", rms_norm, x)
        print("\n## RMSNorm 分析结果")
        print(rms_table)

        # 2. 分析SwiGLU
        swiglu = SwiGLU(dim).to(self.device)
        swiglu_table = self.profile_function("SwiGLU", swiglu, x)
        print("\n## SwiGLU 分析结果")
        print(swiglu_table)

        # 3. 分析Softmax
        logits = torch.randn(batch_size, seq_len, dim, device=self.device)
        softmax_table = self.profile_function("CustomSoftmax", Function.softmax, logits, dim=-1)
        print("\n## CustomSoftmax 分析结果")
        print(softmax_table)

        # 4. 分析多头注意力
        attn = CausalMultiheadAttention(dim, num_heads=8).to(self.device)
        attn_table = self.profile_function("CausalMultiheadAttention", attn, x,x,x)
        print("\n## CausalMultiheadAttention 分析结果")
        print(attn_table)

        # 5. 分析组合操作
        def combined_operation(x):
            norm_x = rms_norm(x)
            swiglu_out = swiglu(norm_x)
            attn_out = attn(swiglu_out, swiglu_out, swiglu_out)
            return Function.softmax(attn_out, dim=-1)

        combined_table = self.profile_function("CombinedOperation", combined_operation, x)
        print("\n## 组合操作分析结果")
        print(combined_table)

    def visualize_results(self):
        """可视化分析结果"""
        if not self.results:
            print("没有可用的分析结果")
            return

        print("\n" + "=" * 60)
        print("性能分析可视化")
        print("=" * 60)

        # 1. 比较各操作的总时间
        times = {}
        for name, data in self.results.items():
            # 提取总时间 - 更可靠的方法
            table_lines = data["table"].split("\n")
            cpu_time = None
            cuda_time = None

            for line in table_lines:
                if "Self CPU time total" in line:
                    # 提取时间值（最后一个单词）
                    time_str = line.split()[-1]
                    # 移除单位并转换为浮点数
                    if time_str.endswith("ms"):
                        cpu_time = float(time_str[:-2])
                    elif time_str.endswith("us"):
                        cpu_time = float(time_str[:-2]) / 1000  # 微秒转毫秒
                    elif time_str.endswith("s"):
                        cpu_time = float(time_str[:-1]) * 1000  # 秒转毫秒

                elif "Self CUDA time total" in line:
                    time_str = line.split()[-1]
                    if time_str.endswith("ms"):
                        cuda_time = float(time_str[:-2])
                    elif time_str.endswith("us"):
                        cuda_time = float(time_str[:-2]) / 1000
                    elif time_str.endswith("s"):
                        cuda_time = float(time_str[:-1]) * 1000

            # 优先使用CUDA时间（如果可用）
            if self.device == "cuda" and cuda_time is not None:
                times[name] = cuda_time
            elif cpu_time is not None:
                times[name] = cpu_time
            else:
                print(f"警告: 无法提取 {name} 的时间信息")
                times[name] = 0  # 默认值

        # 创建柱状图
        plt.figure(figsize=(12, 6))
        names = list(times.keys())
        values = list(times.values())

        plt.bar(names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        plt.title("自定义函数性能比较")
        plt.ylabel("执行时间 (ms)")
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("custom_functions_performance.png")
        print("已保存性能比较图: custom_functions_performance.png")

        # 2. 火焰图（需要安装flameprof）
        try:
            from flameprof import FlameGraph
            for name, data in self.results.items():
                # 生成火焰图
                stacks = data["profiler"].export_stacks(f"stacks_{name}.txt", "self_cuda_time_total")
                flame_graph = FlameGraph.from_file(f"stacks_{name}.txt")
                flame_graph.to_svg(f"flamegraph_{name}.svg")
                print(f"已保存火焰图: flamegraph_{name}.svg")
        except ImportError:
            print("未安装flameprof，跳过火焰图生成")

        # 3. 内存使用分析
        if self.device == "cuda":
            print("\nGPU内存使用情况:")
            for name, data in self.results.items():
                # 提取内存信息
                table_lines = data["table"].split("\n")
                mem_usage = 0
                for line in table_lines:
                    if "Self CUDA Mem" in line:
                        parts = line.split()
                        if len(parts) > 4:
                            mem_str = parts[4]
                            if mem_str.endswith("MB"):
                                mem_usage = max(mem_usage, float(mem_str[:-2]))
                            elif mem_str.endswith("KB"):
                                mem_usage = max(mem_usage, float(mem_str[:-2]) / 1024)
                            elif mem_str.endswith("GB"):
                                mem_usage = max(mem_usage, float(mem_str[:-2]) * 1024)
                print(f"{name}: {mem_usage:.2f} MB")

        # 4. 操作耗时分布
        plt.figure(figsize=(10, 8))
        for i, (name, data) in enumerate(self.results.items()):
            # 提取各操作的耗时
            table_lines = data["table"].split("\n")
            op_times = []
            op_names = []

            # 跳过表头和汇总行
            for line in table_lines[3:-3]:
                parts = line.split()
                if len(parts) > 4:
                    # 提取时间值
                    time_str = parts[3] if self.device == "cuda" else parts[2]
                    if time_str.endswith("ms"):
                        time_val = float(time_str[:-2])
                    elif time_str.endswith("us"):
                        time_val = float(time_str[:-2]) / 1000
                    elif time_str.endswith("s"):
                        time_val = float(time_str[:-1]) * 1000
                    else:
                        continue

                    if time_val > 0.01:  # 只显示耗时超过0.01ms的操作
                        op_times.append(time_val)
                        op_names.append(" ".join(parts[0:-4]))

            if not op_times:
                continue

            # 创建饼图
            plt.subplot(2, 3, i + 1)
            plt.pie(op_times, labels=op_names, autopct='%1.1f%%', startangle=90)
            plt.title(f"{name} 操作耗时分布")

        plt.tight_layout()
        plt.savefig("operation_time_distribution.png")
        print("已保存操作耗时分布图: operation_time_distribution.png")


def efficient_rms_norm(x, dim=-1, eps=1e-6):
    dim_size = x.size(dim)
    # 一步计算均方根
    return torch.sqrt(torch.mean(x**2, dim=dim, keepdim=True) + eps)

def rms_norm(x, dim=-1, eps=1e-6):
    if x.is_cuda:
        return rms_extension.rms_norm_cuda(x, dim, eps)
    else:
        return efficient_rms_norm(x, dim, eps)

def atest_performance(func, name):
    start = time.time()
    for _ in range(1000):
        rms = func(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"{name}: {elapsed * 1000:.3f} ms")

if __name__ == "__main__":
    # 检查PyTorch版本和CUDA可用性
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {'是' if torch.cuda.is_available() else '否'}")

    # 运行性能分析演示
    # demo_profiling()
    ##############
    # profiler = CustomProfiler()
    #
    # # 分析不同规模的操作
    # for dim in [128, 256, 512]:
    #     profiler.profile_custom_operations(dim=dim, seq_len=128, batch_size=32)

    # 可视化结果
    # profiler.visualize_results()
    if torch.cuda.is_available():
        rms_extension = load(
            name='rms_norm',
            sources=['rms_kernel.cu'],
            extra_cuda_cflags=['-O3', '--use_fast_math']
        )

    x = torch.randn(1024, 768, device='cuda')


    # 运行测试
    atest_performance(lambda x: efficient_rms_norm(x, dim=-1), "PyTorch高效版")
    atest_performance(lambda x: rms_norm(x, dim=-1), "CUDA扩展版")
    atest_performance(lambda x: rms_triton.rms_norm_triton(x, dim=-1), "Triton版")

    # PyTorch高效版: 271.022ms
    # CUDA扩展版: 48.593ms
    # Triton版: 3617.369ms