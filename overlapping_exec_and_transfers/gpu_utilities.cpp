#include "gpu_utilities.h"
#include "definitions.h"
#include "jit4ptx.h"

#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <cuda.h>

// Define the pointers to the kernels
CUfunction haddKernel = nullptr;
CUfunction hmodifyKernel = nullptr;
CUfunction haddKernel_part = nullptr;
CUfunction hmodifyKernel_part = nullptr;
CUfunction haddKernel_pitch = nullptr;
CUfunction hmodifyKernel_pitch = nullptr;

// Load PTX file(s) containing the kernels (JIT)
bool load_all_ptx()
{
	// Define the name of the ptx
	std::string kernelPTX = "kernel.ptx";
	
	CUmodule hmodule = nullptr;
	CUlinkState lState = nullptr;
	
	// Load and JIT the ptx - if ok, load the kernels
	if (ptxJIT(kernelPTX, &hmodule, &lState, 0))
	{
		cuModuleGetFunction(&haddKernel, hmodule, "addKernel");
		cuModuleGetFunction(&hmodifyKernel, hmodule, "modifyKernel");
		cuModuleGetFunction(&haddKernel_part, hmodule, "addKernel_part");
		cuModuleGetFunction(&hmodifyKernel_part, hmodule, "modifyKernel_part");
		cuModuleGetFunction(&haddKernel_pitch, hmodule, "addKernel_pitch");
		cuModuleGetFunction(&hmodifyKernel_pitch, hmodule, "modifyKernel_pitch");
	}
	else
	{
		printf("\nFailed to JIT  kernels.cu ! \n");
		return false;
	}
	return true;
}

// Run on GPU with all the workload in one take
cudaError_t execute_GPU_in_one_take(real* c_from_d, const real* a, const real* b, const size_t nrows, const size_t ncols, const size_t bytes, const size_t reps, const real dt)
{
	// Define necessary variables
	cudaError_t exitStat = cudaSuccess;
	real* a_d = nullptr;
	real* b_d = nullptr;
	real* c_d = nullptr;
	real* w_d = nullptr;
	
	// Allocate memory on GPU
	cudaError_t allocStat = cudaMalloc((void**)&a_d, bytes);
	if (allocStat != cudaSuccess)
	{
	    printf("Error allocating a_d... \n");
	    printf("%s \n", cudaGetErrorString(allocStat));
		return allocStat;
	}
	else
		printf("Allocated GPU memory for a_d... \n");
	
	allocStat = cudaMalloc((void**)&b_d, bytes);
	if (allocStat != cudaSuccess)
	{
	    printf("Error allocating b_d... \n");
	    printf("%s \n", cudaGetErrorString(allocStat));
		return allocStat;
	}
	else
		printf("Allocated GPU memory for b_d... \n");

	allocStat = cudaMalloc((void**)&c_d, bytes);
	if (allocStat != cudaSuccess)
	{
		printf("Error allocating c_d... \n");
		printf("%s \n", cudaGetErrorString(allocStat));
		return allocStat;
	}
	else
		printf("Allocated GPU memory for c_d... \n");

	allocStat = cudaMalloc((void**)&w_d, bytes);
	if (allocStat != cudaSuccess)
	{
		printf("Error allocating w_d... \n");
		printf("%s \n", cudaGetErrorString(allocStat));
		return allocStat;
	}
	else
		printf("Allocated GPU memory for w_d... \n");

	// Init to zero for intermediate array only
	cudaMemset(w_d, 0, bytes);
	
	// Everything in one go
	dim3 blockSize(BLOCKX, BLOCKY, 1);
	dim3 gridSize((ncols + blockSize.x - 1) / blockSize.x, (nrows + blockSize.y - 1) / blockSize.y, 1);
	
	// Transfer data in: 1/2
	cudaError_t transferStat = cudaMemcpy(a_d, a, bytes, cudaMemcpyHostToDevice);
	if (transferStat != cudaSuccess)
	{
	    printf("Error transferring a -> a_d... \n");
	    printf("%s \n", cudaGetErrorString(transferStat));
		return transferStat;
	}
	
	// Transfer data in: 2/2
	transferStat = cudaMemcpy(b_d, b, bytes, cudaMemcpyHostToDevice);
	if (transferStat != cudaSuccess)
	{
	    printf("Error transferring b -> b_d... \n");
	    printf("%s \n", cudaGetErrorString(transferStat));
		return transferStat;
	}

	// Workload
	for (auto ij = 0; ij < reps; ++ij)
	{
		// Launch first kernel
		//addKernel(real *c, const real *a, const real *b, const size_t rows, const size_t cols, const real dt)
		void* args[6] = { &w_d, &a_d, &b_d, (void*)&nrows, (void*)&ncols, (void*)&dt };
		cuLaunchKernel(haddKernel, gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, NULL, args, NULL);

		// Launch second kernel
		//modifyKernel(real* d, const real* a, const real* b, const real* c, const real dt, const size_t rows, const size_t cols)
		void* argsM[7] = { &c_d, &a_d, &b_d, &w_d, (void*)&dt, (void*)&nrows, (void*)&ncols };
		cuLaunchKernel(hmodifyKernel, gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, NULL, argsM, NULL);

		cudaMemcpy(w_d, c_d, bytes, cudaMemcpyDeviceToDevice);
		cudaError_t asyncError = cudaGetLastError();
		if (asyncError != cudaSuccess)
		{
			printf("Async Error... \n");
			printf("%s \n", cudaGetErrorString(asyncError));
			return asyncError;
		}
	}
	
	// Transfer data out: 1/1
	transferStat = cudaMemcpy(c_from_d, c_d, bytes, cudaMemcpyDeviceToHost);
	if (transferStat != cudaSuccess)
	{
	    printf("Error transferring c_d -> c_from_d... \n");
	    printf("%s \n", cudaGetErrorString(transferStat));
		return transferStat;
	}

	// Release resources
	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);
	cudaFree(w_d);

	return exitStat;
}

// Run on GPU with all the workload using streams
cudaError_t execute_GPU_with_streams(real* c_from_d, const real* a, const real* b, const size_t bytes, const size_t nrows, const size_t ncols, const size_t nstreams, const size_t reps, const real dt)
{
	// Define necessary variables
	cudaError_t exitStat = cudaSuccess;
	real* a_d = nullptr;
	real* b_d = nullptr;
	real* c_d = nullptr;
	real* w_d = nullptr;
	
	// Allocate memory on GPU
	cudaError_t allocStat = cudaMalloc((void**)&a_d, bytes);
	if (allocStat != cudaSuccess)
	{
	    printf("Error allocating a_d... \n");
	    printf("%s \n", cudaGetErrorString(allocStat));
		return allocStat;
	}
	else
		printf("Allocated GPU memory for a_d... \n");
	
	allocStat = cudaMalloc((void**)&b_d, bytes);
	if (allocStat != cudaSuccess)
	{
	    printf("Error allocating b_d... \n");
	    printf("%s \n", cudaGetErrorString(allocStat));
		return allocStat;
	}
	else
		printf("Allocated GPU memory for b_d... \n");
	
	allocStat = cudaMalloc((void**)&c_d, bytes);
	if (allocStat != cudaSuccess)
	{
	    printf("Error allocating c_d... \n");
	    printf("%s \n", cudaGetErrorString(allocStat));
		return allocStat;
	}
	else
		printf("Allocated GPU memory for c_d... \n");

	allocStat = cudaMalloc((void**)&w_d, bytes);
	if (allocStat != cudaSuccess)
	{
		printf("Error allocating w_d... \n");
		printf("%s \n", cudaGetErrorString(allocStat));
		return allocStat;
	}
	else
		printf("Allocated GPU memory for w_d... \n");
	
	// Init to zero for intermediate array only
	cudaMemset(w_d, 0, bytes);

	// Create the streams as requested...
	cudaStream_t* streams = (cudaStream_t*)malloc(nstreams * sizeof(cudaStream_t));
	for (int i = 0; i < nstreams; i++) {
	    cudaStreamCreate(&streams[i]);
	}
	
	// Sort out logistics
	size_t rowsPerStream = (nrows + nstreams - 1) / nstreams;
	size_t colsPerStream = (ncols + nstreams - 1) / nstreams;
	unsigned int i_stream = 0;
	
	// Workload
	for (auto irep = 0; irep < reps; ++irep)
	{
		// Go over the entire array
		for (auto ir = 0; ir < nstreams; ++ir)
		{
			for (auto ic = 0; ic < nstreams; ++ic)
			{
				// Calculate offsets and size of chunks
				size_t nrows_local = rowsPerStream;
				size_t ncols_local = colsPerStream;

				size_t rowOffset = ir * nrows_local;
				size_t colOffset = ic * ncols_local;

				// Do not exceed real size of the array
				if (rowOffset + nrows_local > nrows)
					nrows_local = nrows - rowOffset;
				if (colOffset + ncols_local > ncols)
					ncols_local = ncols - colOffset;

				dim3 block(BLOCKX, BLOCKY, 1);
				dim3 grid((ncols_local + block.x - 1) / block.x, (nrows_local + block.y - 1) / block.y, 1);

				size_t ioffset = rowOffset * ncols + colOffset;

				cudaMemcpy2DAsync(&a_d[ioffset], ncols * sizeof(real), &a[ioffset], ncols * sizeof(real), ncols_local * sizeof(real), nrows_local, cudaMemcpyHostToDevice, streams[i_stream]);
				cudaMemcpy2DAsync(&b_d[ioffset], ncols * sizeof(real), &b[ioffset], ncols * sizeof(real), ncols_local * sizeof(real), nrows_local, cudaMemcpyHostToDevice, streams[i_stream]);

				// Unfortunately, one cannot simply pass the addresses...
				real* w_dd = &w_d[ioffset];
				real* c_dd = &c_d[ioffset];
				real* a_dd = &a_d[ioffset];
				real* b_dd = &b_d[ioffset];

				//addKernel_part(real* c, const real* a, const real* b, const size_t rows, const size_t cols, const size_t stride, const real dt)
				void* args[7] = { (void*)&w_dd, (void*)&a_dd, (void*)&b_dd, (void*)&nrows_local, (void*)&ncols_local, (void*)&ncols, (void*)&dt };
				CUresult launchErr = cuLaunchKernel(haddKernel_part, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, streams[i_stream], args, NULL);
				if (launchErr != CUDA_SUCCESS)
				{
					printf("CUresult :: %u \n", launchErr);
					return cudaErrorLaunchFailure;
				}

				//modifyKernel_part(real* d, const real* a, const real* b, const real* c, const real dt, const size_t rows, const size_t cols, const size_t stride)
				void* argsM[8] = { (void*)&c_dd, (void*)&a_dd, (void*)&b_dd, (void*)&w_dd, (void*)&dt, (void*)&nrows_local, (void*)&ncols_local, (void*)&ncols };
				launchErr = cuLaunchKernel(hmodifyKernel_part, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, streams[i_stream], argsM, NULL);
				if (launchErr != CUDA_SUCCESS)
				{
					printf("CUresult :: %u \n", launchErr);
					return cudaErrorLaunchFailure;
				}

				cudaMemcpy2DAsync(&w_d[ioffset], ncols * sizeof(real), &c_d[ioffset], ncols * sizeof(real), ncols_local * sizeof(real), nrows_local, cudaMemcpyDeviceToDevice, streams[i_stream]);

				i_stream = (i_stream + 1) % nstreams;
			}
		}
	}
	
	cudaError_t syncError = cudaMemcpy2D(c_from_d, ncols * sizeof(real), c_d, ncols * sizeof(real), ncols * sizeof(real), nrows, cudaMemcpyDeviceToHost);
	if (syncError != cudaSuccess)
	{
		printf("Error synchronising device... \n");
		printf("%s \n", cudaGetErrorString(syncError));
	}

	// Release resources
	for (int i = 0; i < nstreams; i++) {
		cudaStreamDestroy(streams[i]);
	}

	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);
	cudaFree(w_d);

	return syncError;
}

// Run on GPU with the workload partitioned and utilising streams
cudaError_t execute_GPU_chunk_by_chunk(float* c_from_d, const float* a, const float* b, const size_t bytes, const size_t arraySize, const int nstreams, const int nparts, const int reps)
{
	cudaError_t exitStat = cudaSuccess;

	size_t iArray = arraySize / nparts;
	size_t rem = arraySize - iArray * nparts;

	for (int i = 0; i < nparts; ++i)
	{
		int ioffset = i * iArray;

		if (i == nparts - 1)
			iArray += rem;

		//execute_GPU_with_streams(&c_from_d[ioffset], &a[ioffset], &b[ioffset], iArray * sizeof(float), iArray, nstreams, reps);
	}

	return exitStat;
}

// Run on GPU with all the workload using streams (maintaining 2D format)
cudaError_t execute_GPU_with_streams_pitch(float* c_from_d, const float* a, const float* b, const size_t nrows, const size_t ncols, const int reps)
{
	cudaError_t exitStat = cudaSuccess;
	float* a_d = nullptr;
	float* b_d = nullptr;
	float* c_d = nullptr;

	size_t gpuPitchA;
	size_t gpuPitchB;
	size_t gpuPitchC;

	cudaError_t allocStat = cudaMallocPitch((void**)&a_d, &gpuPitchA, ncols * sizeof(float), nrows);
	if (allocStat != cudaSuccess)
	{
		printf("Error allocating a_d... \n");
		printf("%s \n", cudaGetErrorString(allocStat));
		return allocStat;
	}
	else
		printf("Allocated GPU memory for a_d... \n");

	allocStat = cudaMallocPitch((void**)&b_d, &gpuPitchB, ncols * sizeof(float), nrows);
	if (allocStat != cudaSuccess)
	{
		printf("Error allocating b_d... \n");
		printf("%s \n", cudaGetErrorString(allocStat));
		return allocStat;
	}
	else
		printf("Allocated GPU memory for b_d... \n");

	allocStat = cudaMallocPitch((void**)&c_d, &gpuPitchC, ncols * sizeof(float), nrows);
	if (allocStat != cudaSuccess)
	{
		printf("Error allocating c_d... \n");
		printf("%s \n", cudaGetErrorString(allocStat));
		return allocStat;
	}
	else
		printf("Allocated GPU memory for c_d... \n");

	size_t rowPartitions = 0;
	size_t colPartitions = 0;
	size_t rowChunkPartitions = 0;
	size_t colChunkPartitions = 0;
	domain_partitioning(nrows, ncols, 12, 4, 1024 * 1024 * 1024, rowPartitions, colPartitions, rowChunkPartitions, colChunkPartitions);

	size_t stream_x = (colPartitions + colChunkPartitions - 1) / colChunkPartitions;
	size_t stream_y = (rowPartitions + rowChunkPartitions - 1) / rowChunkPartitions;
	size_t nstreams = stream_x * stream_y;
#ifdef _DEBUG
	nstreams = 2;
#else
	nstreams = 5;
#endif 
	cudaStream_t * streams = (cudaStream_t*)malloc(nstreams * sizeof(cudaStream_t));
	for (int i = 0; i < nstreams; i++) {
		cudaStreamCreate(&streams[i]);
	}

	size_t rowPerStream = (nrows + nstreams - 1) / nstreams;
	size_t colPerStream = (ncols + nstreams - 1) / nstreams;

	size_t iterOverRows = (nrows + rowPerStream - 1) / rowPerStream;
	size_t iterOverCols = (ncols + colPerStream - 1) / colPerStream;

	unsigned int istream = 0;
	for (auto ir = 0; ir < iterOverRows; ++ir)
	{
		for (auto ic = 0; ic < iterOverCols; ++ic)
		{
			size_t rowOffset = ir * rowPerStream;
			size_t colOffset = ic * colPerStream;
				
			size_t rowNum = rowPerStream;
			size_t colNum = colPerStream;

			// Get the number of rows and columns on which this stream operates
			if (rowOffset + rowNum > nrows)
				rowNum = nrows - rowOffset;
			if (colOffset + colNum > ncols)
				colNum = ncols - colOffset;

			size_t h_idx = rowOffset * ncols + colOffset;
			size_t colPitch = (gpuPitchA / sizeof(float));
			size_t d_idxA = rowOffset * (gpuPitchA / sizeof(float)) + colOffset;
			size_t d_idxB = rowOffset * (gpuPitchB / sizeof(float)) + colOffset;
			size_t d_idxC = rowOffset * (gpuPitchC / sizeof(float)) + colOffset;

			//T * pElement = (T*)((char*)BaseAddress + Row * pitch) + Column;

			cudaMemcpy2DAsync(&a_d[d_idxA], gpuPitchA, &a[h_idx], ncols * sizeof(float), colNum * sizeof(float), rowNum, cudaMemcpyHostToDevice, streams[istream]);
			cudaMemcpy2DAsync(&b_d[d_idxB], gpuPitchB, &b[h_idx], ncols * sizeof(float), colNum * sizeof(float), rowNum, cudaMemcpyHostToDevice, streams[istream]);

			float* c_dd = &c_d[d_idxC];
			float* a_dd = &a_d[d_idxA];
			float* b_dd = &b_d[d_idxB];
			void* args[7] = { (void*)&c_dd, (void*)&a_dd, (void*)&b_dd, (void*)&reps, (void*)&rowNum, (void*)&colNum, (void*)&gpuPitchC };

			dim3 block(16, 16, 1);
			dim3 grid((colNum + block.x - 1) / block.x, (rowNum + block.y - 1) / block.y, 1);

			cuLaunchKernel(haddKernel_pitch, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, streams[istream], args, NULL);
			cuLaunchKernel(hmodifyKernel_pitch, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, streams[istream], args, NULL);

			cudaMemcpy2DAsync(&c_from_d[h_idx], ncols * sizeof(float), &c_d[d_idxC], gpuPitchC, colNum * sizeof(float), rowNum, cudaMemcpyDeviceToHost, streams[istream]);
			istream = ++istream % nstreams;
		}
	}
	
	cudaError_t syncError = cudaDeviceSynchronize();
	if (syncError != cudaSuccess)
	{
		printf("Error synchronising device... \n");
		printf("%s \n", cudaGetErrorString(syncError));
	}

	for (int i = 0; i < nstreams; i++) {
		cudaStreamDestroy(streams[i]);
	}

	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);

	return syncError;
}

// Decide how to partition the matrix
void domain_partitioning(const size_t nrows, const size_t ncols, const int narraysReal, const int narraysChar, 
	const size_t gpuMemBytes, size_t& rowPartitions, size_t& colPartitions, size_t& propRowPartitions, size_t& propColPartitions)
{
	printf("\n");

	// This is the smallest amount of information to be processed by a block of threads
	const size_t minNelements = BLOCKX * BLOCKY;

	// This is the memory needed in bytes for all the matrices required in a block of threads
	const size_t minMemNeed = (narraysChar * sizeof(char) + narraysReal * sizeof(float)) * minNelements;

	// The quantum of a domain is: BLOCKX x BLOCKY
	rowPartitions = (nrows + BLOCKY - 1) / BLOCKY;
	colPartitions = (ncols + BLOCKX - 1) / BLOCKX;
	size_t subDomainsNeeded = rowPartitions * colPartitions;

	// This is the overall memory need given the partitioning (overestimating, but not that bad...)
	const size_t overMemNeed = rowPartitions * colPartitions * minMemNeed;

	// This is how many rounds it takes to process all blocks given the limited amount of memory
	const size_t nRounds = (overMemNeed + gpuMemBytes - 1) / gpuMemBytes;

	// This is how many min blocks fit in memory at any given time
	const size_t numOfBlockFitting = gpuMemBytes / minMemNeed;

	printf("                   Processing rows x cols :: %u x %u \n", nrows, ncols);
	printf("                        Min block of work :: %u x %u \n", BLOCKY, BLOCKX);
	printf("                       Partitions of work :: %u x %u \n", rowPartitions, colPartitions);
	printf("               Memory needs for min block :: %u Bytes \n", minMemNeed);
	printf("              Memory needs for all blocks :: %u Bytes \n", overMemNeed);
	printf("                         Available memory :: %u Bytes \n", gpuMemBytes);
	printf("       Number of blocks fitting in memory :: %u \n", numOfBlockFitting);
	printf(" Number of rounds to process all elements :: %u \n", nRounds);
	
	propRowPartitions = 1;
	propColPartitions = colPartitions;
	while (true)
	{
		propRowPartitions = numOfBlockFitting / propColPartitions;
		if (propRowPartitions > 0)
			break;
		
		propColPartitions /= 2;
	}

	printf("                       Proposed partition :: % u x %u  (%u) \n", propRowPartitions, propColPartitions, propColPartitions * propRowPartitions);







	

	//size_t remainingRows = nrows % BLOCKY;
	//size_t remainingCols = ncols % BLOCKX;
	//if (remainingRows > 0) ++rowParts;
	//if (remainingCols > 0) ++colParts;

	//printf("      Processing rows x cols :: %d x %d \n", nrows, ncols);
	//printf("    Number of row partitions :: %d \n", rowParts);
	//printf(" Number of column partitions :: %d \n", colParts);
	//printf("   Remaining rows to process :: %d \n", remainingRows);
	//printf("Remaining columns to process :: %d \n", remainingCols);



}

// Run a preliminary query to find CUDA-supported devices
bool deviceQuery()
{
	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess) {
		printf("cudaGetDeviceCount returned %d\n-> %s\n",
			static_cast<int>(error_id), cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
		return false;
	}

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0) 
		printf("There are no available device(s) that support CUDA\n");
	else
		printf("Detected %d CUDA Capable device(s)\n", deviceCount);
	
	int driverVersion = 0, runtimeVersion = 0;

	for (auto dev = 0; dev < deviceCount; ++dev) 
	{
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

		// Console log
		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
			driverVersion / 1000, (driverVersion % 100) / 10,
			runtimeVersion / 1000, (runtimeVersion % 100) / 10);
		printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
			deviceProp.major, deviceProp.minor);

		char msg[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		sprintf_s(msg, sizeof(msg),
			"  Total amount of global memory:                 %.0f MBytes "
			"(%llu bytes)\n",
			static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
			(unsigned long long)deviceProp.totalGlobalMem);
#else
		snprintf(msg, sizeof(msg),
			"  Total amount of global memory:                 %.0f MBytes "
			"(%llu bytes)\n",
			static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
			(unsigned long long)deviceProp.totalGlobalMem);
#endif
		printf("%s", msg);

		printf(
			"  GPU Max Clock rate:                            %.0f MHz (%0.2f "
			"GHz)\n",
			deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

#if CUDART_VERSION >= 5000
		// This is supported in CUDA 5.0 (runtime API device properties)
		printf("  Memory Clock rate:                             %.0f Mhz\n",
			deviceProp.memoryClockRate * 1e-3f);
		printf("  Memory Bus Width:                              %d-bit\n",
			deviceProp.memoryBusWidth);

		if (deviceProp.l2CacheSize) {
			printf("  L2 Cache Size:                                 %d bytes\n",
				deviceProp.l2CacheSize);
		}

#else
		// This only available in CUDA 4.0-4.2 (but these were only exposed in the
		// CUDA Driver API)
		int memoryClock;
		getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
			dev);
		printf("  Memory Clock rate:                             %.0f Mhz\n",
			memoryClock * 1e-3f);
		int memBusWidth;
		getCudaAttribute<int>(&memBusWidth,
			CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
		printf("  Memory Bus Width:                              %d-bit\n",
			memBusWidth);
		int L2CacheSize;
		getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

		if (L2CacheSize) {
			printf("  L2 Cache Size:                                 %d bytes\n",
				L2CacheSize);
		}

#endif

		printf(
			"  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, "
			"%d), 3D=(%d, %d, %d)\n",
			deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
			deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
			deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
		printf(
			"  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
			deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
		printf(
			"  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
			"layers\n",
			deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
			deviceProp.maxTexture2DLayered[2]);

		printf("  Total amount of constant memory:               %lu bytes\n",
			deviceProp.totalConstMem);
		printf("  Total amount of shared memory per block:       %lu bytes\n",
			deviceProp.sharedMemPerBlock);
		printf("  Total number of registers available per block: %d\n",
			deviceProp.regsPerBlock);
		printf("  Warp size:                                     %d\n",
			deviceProp.warpSize);
		printf("  Maximum number of threads per multiprocessor:  %d\n",
			deviceProp.maxThreadsPerMultiProcessor);
		printf("  Maximum number of threads per block:           %d\n",
			deviceProp.maxThreadsPerBlock);
		printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
			deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);
		printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
			deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
			deviceProp.maxGridSize[2]);
		printf("  Maximum memory pitch:                          %lu bytes\n",
			deviceProp.memPitch);
		printf("  Texture alignment:                             %lu bytes\n",
			deviceProp.textureAlignment);
		printf(
			"  Concurrent copy and kernel execution:          %s with %d copy "
			"engine(s)\n",
			(deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
		printf("  Run time limit on kernels:                     %s\n",
			deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
		printf("  Integrated GPU sharing Host Memory:            %s\n",
			deviceProp.integrated ? "Yes" : "No");
		printf("  Support host page-locked memory mapping:       %s\n",
			deviceProp.canMapHostMemory ? "Yes" : "No");
		printf("  Alignment requirement for Surfaces:            %s\n",
			deviceProp.surfaceAlignment ? "Yes" : "No");
		printf("  Device has ECC support:                        %s\n",
			deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
			deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)"
			: "WDDM (Windows Display Driver Model)");
#endif
		printf("  Device supports Unified Addressing (UVA):      %s\n",
			deviceProp.unifiedAddressing ? "Yes" : "No");
		printf("  Device supports Compute Preemption:            %s\n",
			deviceProp.computePreemptionSupported ? "Yes" : "No");
		printf("  Supports Cooperative Kernel Launch:            %s\n",
			deviceProp.cooperativeLaunch ? "Yes" : "No");
		printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
			deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
		printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
			deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

		const char* sComputeMode[] = {
			"Default (multiple host threads can use ::cudaSetDevice() with device "
			"simultaneously)",
			"Exclusive (only one host thread in one process is able to use "
			"::cudaSetDevice() with this device)",
			"Prohibited (no host thread can use ::cudaSetDevice() with this "
			"device)",
			"Exclusive Process (many threads in one process is able to use "
			"::cudaSetDevice() with this device)",
			"Unknown",
			NULL };
		printf("  Compute Mode:\n");
		printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
	}
	return true;
}