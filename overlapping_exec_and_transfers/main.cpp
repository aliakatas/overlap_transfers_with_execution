#include "CmdParser.h"
#include "cpu_utilities.h"
#include "definitions.h"
#include "gpu_utilities.h"
#include "Timer.h"

#include <stdio.h>
#include "cuda_runtime.h"
#include "cuda.h"

int main(int argc, char** argv)
{
    // Create a timer...
    Timer timer;
    // Create the configuration
    RunConfiguration myRC;
    // Modify configuration as per user input
    parseArguments(argc, argv, myRC);

    // Create some general info...
    const size_t arraySize = myRC.ncols * myRC.nrows;
    const size_t bytesPerMatrix = arraySize * sizeof(real);
    const size_t numOfHostMatrices = 8;
    const size_t numOfDeviceMatrices = 4;

    // Show some info on the work to be done...
    printf("\n");
#ifdef DP
    printf("    Running with Precision :: Double -> %u bytes / real number.\n", sizeof(real));
#else
    printf("    Running with Precision :: Single -> %u bytes / real number.\n", sizeof(real));
#endif
    printf("            Rows x Columns :: %llu x %llu \n", myRC.nrows, myRC.ncols);
    printf(" Total Elements per matrix :: %llu \n", arraySize);
    printf("          Bytes per matrix :: %llu Bytes | %f KB | %f MB \n", bytesPerMatrix, real(bytesPerMatrix) / 1024.0, real(bytesPerMatrix) / 1024.0 / 1024.0);
    printf("        Max host RAM usage :: %llu Bytes | %f KB | %f MB \n", numOfHostMatrices * bytesPerMatrix, numOfHostMatrices * real(bytesPerMatrix) / 1024.0, numOfHostMatrices * real(bytesPerMatrix) / 1024.0 / 1024.0);
    printf("      Max device RAM usage :: %llu Bytes | %f KB | %f MB \n", numOfDeviceMatrices * bytesPerMatrix, numOfDeviceMatrices * real(bytesPerMatrix) / 1024.0, numOfDeviceMatrices * real(bytesPerMatrix) / 1024.0 / 1024.0);
    printf("               Repetitions :: %u \n", myRC.reps);
    printf("         Number of streams :: %u \n", myRC.n_streams);
    printf("      Number of partitions :: %u \n", myRC.parts);
    printf("                 Tolerance :: %f \n", myRC.tolerance);
    printf("\n");

    // Declare the arrays
    real* a_h = nullptr;
    real* b_h = nullptr;
    real* c_h = nullptr;
    real* c_from_GPU_one = nullptr;
    real* c_from_GPU_stream = nullptr;
    real* c_from_GPU_chunk = nullptr;
    real *c_from_GPU_2Dstreams = nullptr;

    // Allocate host arrays
    cudaError_t pin_alloc = cudaMallocHost((void**)&a_h, bytesPerMatrix);
    if (pin_alloc != cudaSuccess)
    {
        printf("Error allocating a_h... \n");
        printf("%s \n", cudaGetErrorString(pin_alloc));
        return -999;
    }
    else
        printf("Allocated pinned memory for a_h. \n");

    pin_alloc = cudaMallocHost((void**)&b_h, bytesPerMatrix);
    if (pin_alloc != cudaSuccess)
    {
        printf("Error allocating b_h... \n");
        printf("%s \n", cudaGetErrorString(pin_alloc));
        return -999;
    }
    else
        printf("Allocated pinned memory for b_h. \n");

    c_h = (real*)malloc(bytesPerMatrix);
    if (!c_h)
    {
        printf("Error allocating c_h... \n");
        return -999;
    }
    else
        printf("Allocated pageable memory for c_h. \n");
    
    pin_alloc = cudaMallocHost((void**)&c_from_GPU_one, bytesPerMatrix);
    if (pin_alloc != cudaSuccess)
    {
        printf("Error allocating c_from_GPU_one... \n");
        printf("%s \n", cudaGetErrorString(pin_alloc));
        return -999;
    }
    else
        printf("Allocated pinned memory for c_from_GPU_one. \n");

    pin_alloc = cudaMallocHost((void**)&c_from_GPU_stream, bytesPerMatrix);
    if (pin_alloc != cudaSuccess)
    {
        printf("Error allocating c_from_GPU_stream... \n");
        printf("%s \n", cudaGetErrorString(pin_alloc));
        return -999;
    }
    else
        printf("Allocated pinned memory for c_from_GPU_stream. \n");

    pin_alloc = cudaMallocHost((void**)&c_from_GPU_chunk, bytesPerMatrix);
    if (pin_alloc != cudaSuccess)
    {
        printf("Error allocating c_from_GPU_chunk... \n");
        printf("%s \n", cudaGetErrorString(pin_alloc));
        return -999;
    }
    else
        printf("Allocated pinned memory for c_from_GPU_chunk. \n");

    pin_alloc = cudaMallocHost((void**)&c_from_GPU_2Dstreams, bytesPerMatrix);
    if (pin_alloc != cudaSuccess)
    {
        printf("Error allocating c_from_GPU_2Dstreams... \n");
        printf("%s \n", cudaGetErrorString(pin_alloc));
        return -999;
    }
    else
        printf("Allocated pinned memory for c_from_GPU_2Dstreams. \n");

    // Initialise the input matrices
    initialise(a_h, myRC.nrows, myRC.ncols);
    initialise(b_h, myRC.nrows, myRC.ncols);

    // Run work on CPU
    timer.start("CPU execution ");
    printf("Performing calculations on CPU... \n");
    calculateOnCPU(c_h, a_h, b_h, myRC.nrows, myRC.ncols, myRC.dt, myRC.reps);
    timer.stop();

#ifdef _DEBUG
    dump_to_csv("CPU_out.csv", c_h, myRC.nrows, myRC.ncols);
#endif // _DEBUG

    // Check for CUDA-supported devices...
    if (!deviceQuery())
        return -99;

    // Use the first available device (naive, but ok)
    cudaError_t devSet = cudaSetDevice(0);
    if (devSet != cudaSuccess) {
        printf("Error setting the device... \n");
        printf("%s \n", cudaGetErrorString(devSet));
        return -9;
    }
    else
        printf("Device set. \n");

    // Load the PTX
    printf("Loading PTX... \n");
    if (!load_all_ptx())
    {
        printf("Error loading the PTX content... \n");
        return -10;
    }

    // Run on GPU - all data across, simple approach
    printf("GPU execution (one take)... \n");
    timer.start("GPU execution (one take) ");
    cudaError_t devExec = execute_GPU_in_one_take(c_from_GPU_one, a_h, b_h, myRC.nrows, myRC.ncols, bytesPerMatrix, myRC.reps, myRC.dt);
    if (devExec != cudaSuccess)
    {
        printf("Error executing in one take on GPU... \n");
        return -1;
    }
    timer.stop();

#ifdef _DEBUG
    dump_to_csv("GPU_out.csv", c_from_GPU_one, myRC.nrows, myRC.ncols);
#endif // _DEBUG

    // Run on GPU - all data across, use streams
    printf("GPU execution (streams)... \n");
    timer.start("GPU execution (streams) ");
    devExec = execute_GPU_with_streams(c_from_GPU_stream, a_h, b_h, bytes, arraySize, n_streams, reps);
    if (devExec != cudaSuccess)
    {
        printf("Error executing with streams on GPU... \n");
        return -2;
    }
    timer.stop();

    /*
    printf("GPU execution (chunks)... \n");
    timer.start("GPU execution (chunks) ");
    devExec = execute_GPU_chunk_by_chunk(c_from_GPU_chunk, a_h, b_h, bytes, arraySize, n_streams, n_parts, reps);
    if (devExec != cudaSuccess)
    {
        printf("Error executing chunk by chunk on GPU... \n");
        return -3;
    }
    timer.stop();

    printf("GPU execution (streams with 2D)... \n");
    timer.start("GPU execution (streams with 2D) ");
    devExec = execute_GPU_with_streams_pitch(c_from_GPU_2Dstreams, a_h, b_h, nrows, ncols, reps);
    if (devExec != cudaSuccess)
    {
        printf("Error executing chunk by chunk on GPU... \n");
        return -4;
    }
    timer.stop();*/

    // Check results!
    printf("\n ************************************************** \n");
    size_t diff = check_answer(c_h, c_from_GPU_one, myRC.nrows, myRC.ncols, myRC.tolerance);
    if (diff)
        printf("Discrepancy detected in results from kernel in one take! -> %llu errors \n", diff);
    else
        printf("Results (one take) OK! \n");

    diff = check_answer(c_h, c_from_GPU_stream, myRC.nrows, myRC.ncols, myRC.tolerance);
    if (diff)
        printf("Discrepancy detected in results from kernel with streams! -> %llu errors \n", diff);
    else
        printf("Results (streams) OK! \n");

    /*
    diff = check_answer(c_h, c_from_GPU_chunk, nrows, ncols, epsilon);
    if (diff)
    {
        printf("Discrepancy detected in results from chunks! # %llu \n", diff);
    }
    else
    {
        printf("Results (chunks) OK! \n");
    }

    diff = check_answer(c_h, c_from_GPU_2Dstreams, nrows, ncols, epsilon);
    if (diff)
    {
        printf("Discrepancy detected in results from 2D streams! # %llu \n", diff);
    }
    else
    {
        printf("Results (2D streams) OK! \n");
    }
*/

    // Timings
    timer.print();

    // Reset device
    devSet = cudaDeviceReset();
    if (devSet != cudaSuccess)
    {
        printf("Error resetting the device... \n");
        printf("%s \n", cudaGetErrorString(devSet));
        return -1;
    }
    else
        printf("Device reset: OK! \n");

	return 0;
}
