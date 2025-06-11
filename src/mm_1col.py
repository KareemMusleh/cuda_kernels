import os
# Set to my Compute Capability by default
os.environ['TORCH_CUDA_ARCH_LIST'] = os.environ.get("TORCH_CUDA_ARCH_LIST", "6.1")

import torch
from torch.utils.cpp_extension import load_inline

# From PMPP chapter 3. This is exercise 1.b
cuda_source = '''
__global__ void matmul_kernel(const float* A, const float* B, float* C, int N, int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N) {
        float acc;
        for (int m = 0; m < M; m++) {
            acc = 0.0;
            for (int k = 0; k < K; k++) {
                acc += A[row * K + k] * B[M * k + m];
            }
            C[row * M + m] = acc;
        }
    }
}

torch::Tensor matmul(torch::Tensor A, torch::Tensor B) {
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::runtime_error("Tensors must be CUDA tensors");
    }
    if (A.size(-1) != B.size(0)) {
        throw std::runtime_error("shapes are incorrect for matmul");
    }

    const auto N = A.size(0);
    const auto K = A.size(1);
    const auto M = B.size(1);

    auto options = torch::TensorOptions()
        .device(A.device())
        .dtype(A.dtype());
    auto matmul_t = torch::empty({N, M}, options);

    dim3 ts(32);
    dim3 bs((N + ts.x - 1) / ts.x);

    matmul_kernel<<<bs, ts>>>(A.data_ptr<float>(), B.data_ptr<float>(), matmul_t.data_ptr<float>(), N, M, K);
    return matmul_t;
    }
'''

cpp_source = "torch::Tensor matmul(torch::Tensor A, torch::Tensor B);"

# Load the CUDA kernel as a PyTorch extension
ext = load_inline(
    name='ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['matmul'],
    with_cuda=True,
    extra_cuda_cflags=["-O2"],
    build_directory='./build',
    # extra_cuda_cflags=['--expt-relaxed-constexpr']
)
size = 128
A = torch.randn((size, size), device='cuda')
B = torch.randn((size, size), device='cuda')
cuda_matmul = ext.matmul(A, B)
pytorch_matmul = torch.matmul(A, B)
tols = {'rtol' : 1e-4, 'atol' : 1e-4}
# print(cuda_matmul[0][:16], pytorch_matmul[0][:16], sep='\n')
if torch.allclose(cuda_matmul, pytorch_matmul, **tols):
    print("WOOOHOO IT'S WORKING")
else:
    print("IT ISN'T WORKING :(")