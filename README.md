# cuda_kernels

You can check your Compute Capability by running:
```bash
git clone github.com/KareemMusleh/cuda_kernels/
cd csrc/device_query
make
./device_query
```

To run the python programs that are located in `src`. Execute the following from the root of the repo:
```bash
mkdir build
TORCH_CUDA_ARCH_LIST=<YOUR_CC> python src/add.py
```

You can also profile the kernels using:
```bash
sudo TORCH_CUDA_ARCH_LIST=<YOUR_CC> nsys profile python src/add.py
```
you can also use `ncu` but my CC is too old