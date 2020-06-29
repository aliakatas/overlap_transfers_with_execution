#include "jit4ptx.h"
#include "string_operations.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <Windows.h>
#include <string>

// Check if a file exists
inline bool file_exists(const std::string& name) {
    if (FILE* file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    }
    else {
        return false;
    }
}

// Determine the path to the file and extract source code (PTX)
bool inline findModulePath(const std::string module_file, std::string& module_path, std::string& ptx_source)
{
    const int nSize = 512;
    char* currentPath = new char[nSize];
    
    std::string actualPath;
    std::vector<std::string> wholePath;
    const int actualSize = GetModuleFileNameA(NULL, currentPath, nSize);
    if (actualSize > 0)
    {
        for (auto i = 0; i < actualSize; ++i)
            actualPath.push_back(currentPath[i]);

        wholePath = splitpath(actualPath);
    }

    if (module_file.empty())
        return false;

    module_path = "";
    for (auto i = 0; i < wholePath.size() - 1; ++i)
    {
        std::string partPath = wholePath[i];
        for (auto c = 0; c < partPath.length(); ++c)
            module_path.push_back(partPath[c]);
        module_path.push_back('\\');
    }
    module_path += module_file;
    
    if (module_path.empty())
        return false;
    else if (!file_exists(module_path.c_str()))
        return false;
    else 
    {
        if (module_path.rfind(".ptx") != std::string::npos) {
            FILE* fp = fopen(module_path.c_str(), "rb");
            fseek(fp, 0, SEEK_END);
            int file_size = ftell(fp);
            char* buf = new char[file_size + 1];
            fseek(fp, 0, SEEK_SET);
            fread(buf, sizeof(char), file_size, fp);
            fclose(fp);
            buf[file_size] = '\0';
            ptx_source = buf;
            delete[] buf;
        }
        else
            return false;
        return true;
    }
}

// Perform Just-In-Time compilation for the PTX source
bool ptxJIT(std::string ptx_file, CUmodule* phModule, CUlinkState* lState, int device)
{
    CUjit_option options[6];
    void* optionVals[6];
    float walltime;

    char* error_log = new char[16384];
    char* info_log = new char[16384];
    unsigned int logSize = 16384;
    void* cuOut;
    size_t outSize;
    int myErr = 0;
    std::string module_path, ptx_source;

    // Setup linker options
    // Return walltime from JIT compilation
    options[0] = CU_JIT_WALL_TIME;
    optionVals[0] = (void*)&walltime;
    // Pass a buffer for info messages
    options[1] = CU_JIT_INFO_LOG_BUFFER;
    optionVals[1] = (void*)info_log;
    // Pass the size of the info buffer
    options[2] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    optionVals[2] = (void*)(long)logSize;
    // Pass a buffer for error message
    options[3] = CU_JIT_ERROR_LOG_BUFFER;
    optionVals[3] = (void*)error_log;
    // Pass the size of the error buffer
    options[4] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
    optionVals[4] = (void*)(long)logSize;
    // Make the linker verbose
    options[5] = CU_JIT_LOG_VERBOSE;
    optionVals[5] = (void*)1;
    
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error with cuDevicePrimaryCtxRetain ::  %d\n", err);
    }
    cudaFree(0);

    // Create a pending linker invocation
    myErr = cuLinkCreate(6, options, optionVals, lState);
    if (myErr != CUDA_SUCCESS) {
        fprintf(stderr, "Error with cuLinkCreate ::  %d\n", myErr);
    }

    // first search for the module path before we load the results
    if (!findModulePath(ptx_file, module_path, ptx_source))
        return false;

    // Load the PTX from the ptx file
    myErr = cuLinkAddData(*lState, CU_JIT_INPUT_PTX, (void*)ptx_source.c_str(), strlen(ptx_source.c_str()) + 1, 0, 0, 0, 0);

    if (myErr != CUDA_SUCCESS) {
        // Errors will be put in error_log, per CU_JIT_ERROR_LOG_BUFFER option
        // above.
        fprintf(stderr, "PTX Linker Error:\n%s\n", error_log);
    }

    // Complete the linker step
    cuLinkComplete(*lState, &cuOut, &outSize);

    // Linker walltime and info_log were requested in options above.
    printf("CUDA Link Completed in %fms. Linker Output:\n%s\n", walltime, info_log);

    // Load resulting cuBin into module
    cuModuleLoadData(phModule, cuOut);

    // Destroy the linker invocation
    cuLinkDestroy(*lState);
    return true;
}