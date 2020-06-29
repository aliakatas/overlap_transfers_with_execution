# Overlap transfers with execution on GPU
Test different approaches of offloading work to GPU and compare their performance.

Additionally, just-in-time compilation is used for the kernels. The code contains a mix of CUDA Runtime and Driver APIs.

There is also provision to test operations on very large matrices that do not necessarily fit in device memory.

### Build
The project uses CUDA Toolkit 10.2, MSVC and VS 2019. There are two configurations, one for single and another for double precision. 
