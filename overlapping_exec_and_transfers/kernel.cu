/*
Define the kernels to be used for comparison. 

There are two types of kernels:
        1) Addition of two float arrays and store into a third one.
        2) Calculating the tanh() of each element of two float arrays and storing into a third one. 

Some of the kernels work with linear indices only. However, the arrays tested are 2D.
*/
#include "definitions.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/******************************************************/
extern "C" __global__ void addKernel(real *c, const real *a, const real *b, const size_t rows, const size_t cols, const real dt)
{
    size_t i = threadIdx.x + blockDim.x * blockIdx.x;
    size_t j = threadIdx.y + blockDim.y * blockIdx.y;

    if (i < cols && j < rows)
    {
        size_t idx = j * cols + i;
        c[idx] = c[idx] + dt * (a[idx] + b[idx]);
    }
}

/******************************************************/
extern "C" __global__ void modifyKernel(real* d, const real* a, const real* b, const real* c, const real dt, const size_t rows, const size_t cols)
{
    size_t i = threadIdx.x + blockDim.x * blockIdx.x;
    size_t j = threadIdx.y + blockDim.y * blockIdx.y;

    if (i < cols && j < rows)
    {
        size_t idx = j * cols + i;
        d[idx] = c[idx] * c[idx] + dt * (tanh(a[idx]) + tanh(b[idx])) / real(2.0);
    }
}

/******************************************************/
extern "C" __global__ void addKernel_part(real* c, const real* a, const real* b, const size_t rows, const size_t cols, const size_t stride, const real dt)
{
    size_t i = threadIdx.x + blockDim.x * blockIdx.x;
    size_t j = threadIdx.y + blockDim.y * blockIdx.y;

    if (i < cols && j < rows)
    {
        size_t idx = j * stride + i;
        c[idx] = c[idx] + dt * (a[idx] + b[idx]);
    }
}

/******************************************************/
extern "C" __global__ void modifyKernel_part(real* d, const real* a, const real* b, const real* c, const real dt, const size_t rows, const size_t cols, const size_t stride)
{
    size_t i = threadIdx.x + blockDim.x * blockIdx.x;
    size_t j = threadIdx.y + blockDim.y * blockIdx.y;

    if (i < cols && j < rows)
    {
        size_t idx = j * stride + i;
        d[idx] = c[idx] * c[idx] + dt * (tanh(a[idx]) + tanh(b[idx])) / real(2.0);
    }
}


