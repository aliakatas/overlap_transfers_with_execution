#pragma once
#include "definitions.h"
#include <string>

// Initialise an array with random numbers.
void initialise(real* x, const size_t nrows, const size_t ncols);

// Perform the addition on the CPU to get baseline performance.
void addWithCPU(real* xpy, const real* x, const real* y, const size_t rows, const size_t cols, const real dt);

// Perform a modification on the CPU to get baseline performance.
void modifyWithCPU(real* xpy, const real* x, const real* y, const real* w, const size_t rows, const size_t cols, const real dt);

// Perform a full cycle of computations
void calculateOnCPU(real* xpy, const real* x, const real* y, const size_t rows, const size_t cols, const real dt, const size_t rep,
	const size_t* idxRow, const size_t* idxCol, const size_t nBC);

// Simulate the application of boundary conditions
void applyBC(real* xpy, const size_t rows, const size_t cols, const real val, const size_t* idxRow, const size_t* idxCol, const size_t num);

// Compare results.
size_t check_answer(const real *ref, const real* test, const size_t nrows, const size_t ncols, const real tol);

// Save an array in csv format
void dump_to_csv(const std::string fname, const real* myArray, const size_t nrows, const size_t ncols);
