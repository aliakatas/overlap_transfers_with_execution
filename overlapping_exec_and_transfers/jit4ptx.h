#pragma once

#define CUDA_DRIVER_API

#include <cuda.h>
#include <string>

// Check if a file exists
inline bool file_exists(const std::string& name);

// Determine the path to the file and extract source code (PTX)
bool inline findModulePath(const std::string module_file, std::string& module_path, std::string& ptx_source);

// Perform Just-In-Time compilation for the PTX source
bool ptxJIT(std::string ptx_file, CUmodule* phModule, CUlinkState* lState, int device);
