import config
import torch
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel and C++ wrapper
cuda_source = '''
__global__ void grayscale_kernel(const float* image, float* grayscale_t, int size) {
    int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = g_idx * 3;
    grayscale_t[g_idx] = image[i] * 0.2989 + image[i + 1] * 0.5870 + image[i + 2] * 0.1140;
}

torch::Tensor grayscale(torch::Tensor image) {
    if (!image.is_cuda()) {
        throw std::runtime_error("Tensors must be CUDA tensors");
    }

    const auto size = image.size(0);

    auto options = torch::TensorOptions()
        .device(image.device())
        .dtype(image.dtype());
    auto grayscale_t = torch::empty({size, size}, options);

    dim3 threads_per_block(size);
    dim3 number_of_blocks(size);

    grayscale_kernel<<<number_of_blocks, threads_per_block>>>(image.data_ptr<float>(), grayscale_t.data_ptr<float>(), size);
    return grayscale_t;
    }
'''

cpp_source = "torch::Tensor grayscale(torch::Tensor image);"

# Load the CUDA kernel as a PyTorch extension
ext = load_inline(
    name='ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['grayscale'],
    with_cuda=True,
    extra_cuda_cflags=["-O2"],
    build_directory=config.build_directory,
    # extra_cuda_cflags=['--expt-relaxed-constexpr']
)
size = 128
image = torch.randn((size, size, 3), device='cuda')
def grayscale(image):
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=image.device)
    return (image * weights).sum(2)
cuda_grayscale = ext.grayscale(image)
torch_grayscale = grayscale(image)
tols = {'rtol' : 1e-4, 'atol' : 1e-4}
if torch.allclose(cuda_grayscale, torch_grayscale, **tols):
    print("WOOOHOO IT'S WORKING")
else:
    print("IT ISN'T WORKING :(")