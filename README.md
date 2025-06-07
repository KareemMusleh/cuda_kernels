# cuda_kernels

Run from the root of the repo:
```bash
mkdir build
TORCH_CUDA_ARCH_LIST=<YOUR_CC> python src/add.py
```

You can also profile the kernels using:
```bash
sudo TORCH_CUDA_ARCH_LIST=<YOUR_CC> nsys profile python src/add.py
```
you can also use `ncu` but it doesn't work on my arch