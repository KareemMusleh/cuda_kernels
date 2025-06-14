# cuda_kernels

You can check your Compute Capability by running:
```bash
git clone github.com/KareemMusleh/cuda_kernels/
cd csrc/device_query
make
./device_query
```
Then change it in `config.yaml`

To run the python programs that are located in `src`:
```bash
mkdir build
python src/add.py
```

You can also profile the kernels using:
```bash
sudo nsys profile python src/add.py
```
you can also use `ncu` but my CC is too old