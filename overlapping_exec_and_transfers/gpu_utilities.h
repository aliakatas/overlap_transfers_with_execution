#pragma once
#include "definitions.h"

#include "cuda_runtime.h"

// Load PTX file(s) containing the kernels (JIT)
bool load_all_ptx();

// Run on GPU with all the workload in one take
cudaError_t execute_GPU_in_one_take(real* c_from_d, const real* a, const real* b, const size_t nrows, const size_t ncols, const size_t bytes, 
	const size_t reps, const real dt);

// Run on GPU with all the workload using streams
cudaError_t execute_GPU_with_streams(real* c_from_d, const real* a, const real* b, const size_t bytes, const size_t nrows, const size_t ncols, 
	const size_t nstreams, const size_t reps, const real dt, const size_t hostPitch, const bool useGPUPitch = false);

// Run on GPU with the workload partitioned and utilising streams
cudaError_t execute_GPU_chunk_by_chunk(real* c_from_d, const real* a, const real* b, const size_t bytes, const size_t nrows, const size_t ncols, 
	const size_t nstreams, const size_t nparts, const size_t reps, const real dt, const bool useGPUPitch = false);

// Decide how to partition the matrix
void domain_partitioning(const size_t nrows, const size_t ncols, const int narraysReal, const int narraysChar,
	const size_t gpuMemBytes, size_t& rowPartitions, size_t& colPartitions, size_t& propRowPartitions, size_t& propColPartitions);

// Run a preliminary query to find CUDA-supported devices
bool deviceQuery();


