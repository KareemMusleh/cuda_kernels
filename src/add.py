import config
import torch
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel and C++ wrapper
cuda_source = '''
__global__ void add_kernel(const float* a, const float* b, float* c, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < len) {
        c[i] = a[i] + b[i];
    }
}

torch::Tensor add(torch::Tensor a, torch::Tensor b) {
    if (!a.is_cuda() || !b.is_cuda()) {
        throw std::runtime_error("Tensors must be CUDA tensors");
    }

    const auto len = a.size(0);

    auto c = torch::empty_like(a);

    dim3 threads_per_block(256);
    dim3 number_of_blocks((len + threads_per_block.x - 1) / threads_per_block.x);

    add_kernel<<<number_of_blocks, threads_per_block>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), len);

    return c;
    }
'''

cpp_source = "torch::Tensor add(torch::Tensor a, torch::Tensor b);"

# Load the CUDA kernel as a PyTorch extension
ext = load_inline(
    name='ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['add'],
    with_cuda=True,
    extra_cuda_cflags=["-O2"],
    build_directory=config.build_directory,
    # extra_cuda_cflags=['--expt-relaxed-constexpr']
)

a = torch.arange(6, device='cuda').to(torch.float32)
b = torch.zeros(6, device='cuda').to(torch.float32)

print(ext.add(a, b))
