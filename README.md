##  CUDA Programming

The GPU architecture has many blocks which in turn contains multiple threads which are capable of executing operations in parallel.\
GPU is optimized for throughput, but not necessarily for latency.\
Each GPU core is slow but there are thousands of it.\
GPU works well for massively parallel tasks such as matrix multiplication, but it can be quite inefficient for tasks where massive parallelization is impossible or difficult.

These are the main steps to run you programme on parallel threads of GPU
- Initate The Input Data in HOST(CPU)
- Allocate Memory on Device(GPU) for input and output variables
- Copy the Input data from HOST to DEVICE
- Launch a kernel (call the GPU code) 
- Copy the output from DEVICE to HOST 
- FREE the Allocated memory on GPU

|Programme | Links |
| -------- | ------|
| Square of numbers | [Click Here](https://github.com/RheagalFire/CUDA_Programming/blob/main/square.cu)|
| Adding Vectors    | [Click Here](https://github.com/RheagalFire/CUDA_Programming/blob/main/Vector_Add.cu)|
|Barrier Synchronisation | [Click Here](https://github.com/RheagalFire/CUDA_Programming/blob/main/Barrier_sync.cu) |


