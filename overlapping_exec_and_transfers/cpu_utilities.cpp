#include "cpu_utilities.h"
#include "definitions.h"

#include <math.h>
#include <stdlib.h>
#include <string>
#include <fstream>

// Initialise an array with random numbers.
void initialise(real* x, const size_t nrows, const size_t ncols)
{
    for (auto i = 0; i < nrows; ++i)
        for (auto j = 0; j < ncols; ++j)
            x[i * ncols + j] = static_cast <real>(static_cast <real> (rand()) / static_cast <real> (RAND_MAX));
}

// Perform the addition on the CPU to get baseline performance.
void addWithCPU(real* xpy, const real* x, const real* y, const size_t rows, const size_t cols, const real dt)
{
    size_t idx = 0;
    for (auto i = 0; i < rows; ++i)
    {
        for (auto j = 0; j < cols; ++j)
        {
            idx = i * cols + j;
            xpy[idx] = xpy[idx] + dt * (x[idx] + y[idx]);
        }
    }
}

// Perform a modification on the CPU to get baseline performance.
void modifyWithCPU(real* xpy, const real* x, const real* y, const real* w, const size_t rows, const size_t cols, const real dt)
{
    size_t idx = 0;
    for (auto i = 0; i < rows; ++i)
    {
        for (auto j = 0; j < cols; ++j)
        {
            idx = i * cols + j;
            xpy[idx] = w[idx] * w[idx] + dt * (tanh(x[idx]) + tanh(y[idx])) / real(2.0);
        }
    }
}

// Perform a full cycle of computations
void calculateOnCPU(real* xpy, const real* x, const real* y, const size_t rows, const size_t cols, const real dt, const size_t rep)
{
    real* temp = (real*)malloc(rows * cols * sizeof(real));
    memset(temp, 0, rows * cols * sizeof(real));
    for (auto i = 0; i < rep; ++i)
    {
        addWithCPU(temp, x, y, rows, cols, dt);
        modifyWithCPU(xpy, x, y, temp, rows, cols, dt);
        memcpy(temp, xpy, rows * cols * sizeof(real));
    }
    // This is not doing anything fancy. Just keeps the CPU busy for some time.
}

// Compare results.
size_t check_answer(const real* ref, const real* test, const size_t nrows, const size_t ncols, const real tol)
{
    size_t idx = 0;
    size_t count = 0;
    for (auto irow = 0; irow < nrows; irow++) {
        for (auto icol = 0; icol < ncols; icol++)
        {
            idx = icol + ncols * irow;
            if (fabs(ref[idx] - test[idx]) > tol)
                ++count;
        }
    }
    return count;
}

// Save an array in csv format
void dump_to_csv(const std::string fname, const real* myArray, const size_t nrows, const size_t ncols)
{
    std::ofstream csvstream;
    csvstream.open(fname);
    size_t idx = 0;
    for (auto irow = 0; irow < nrows; ++irow)
    {
        for (auto icol = 0; icol < ncols; ++icol)
        {
            idx = irow * ncols + icol;
            csvstream << std::to_string(myArray[idx]) + ", ";
        }
        csvstream << "\n";
    }
    csvstream.close();
}