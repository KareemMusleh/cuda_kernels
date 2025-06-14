import config
import torch
from torch.utils.cpp_extension import load_inline

# From PMPP chapter 5. Fig 5.9
cuda_source = '''
#define TILE_WIDTH 32
__global__ void matmul_kernel(const float* A, const float* B, float* C, int width) {
    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    
    float Pvalue = 0;
    for (int ph = 0; ph < width/TILE_WIDTH; ph++) {
        Ads[ty][tx] = A[row * width + ph * TILE_WIDTH + tx];
        Bds[ty][tx] = B[(ph * TILE_WIDTH + ty) * width + col];
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++) {
            Pvalue += Ads[ty][k] * Bds[k][tx];
        }
        __syncthreads();
    }
    C[row * width + col] = Pvalue;
}

torch::Tensor matmul(torch::Tensor A, torch::Tensor B) {
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::runtime_error("Tensors must be CUDA tensors");
    }
    const auto N = A.size(0);
    const auto K = A.size(1);
    const auto M = B.size(1);
    if (!(N == K && K == M)) {
        throw std::runtime_error("This function only supports squared matrix multiplication");
    }

    auto options = torch::TensorOptions()
        .device(A.device())
        .dtype(A.dtype());
    auto matmul_t = torch::empty({N, M}, options);

    dim3 ts(32, 32);
    dim3 bs((M + ts.x - 1) / ts.x, (N + ts.y - 1) / ts.y);

    matmul_kernel<<<bs, ts>>>(A.data_ptr<float>(), B.data_ptr<float>(), matmul_t.data_ptr<float>(), N);
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
    build_directory=config.build_directory,
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