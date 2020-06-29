#pragma once
#include "definitions.h"

#include <vector>
#include <string>

// Enumeration for the acceptable inputs above
enum class InputArgs {
	SIZE,
	ROWS,
	COLUMNS,
	REPETITIONS,
	TSTEP,
	STREAMS,
	PARTITIONS,
	TOLERANCE,
	UNKNOWN
};

// Keeping the configuration tidy - default values
struct RunConfiguration
{
#ifdef _DEBUG
	size_t nrows = 801;                 // number of rows
	size_t ncols = 601;                 // number of columns
#else
	size_t nrows = 8001;                // number of rows
	size_t ncols = 6001;                // number of columns
#endif 
	size_t reps = 250;
	real dt = real(0.01);
	size_t n_streams = 5;
	size_t parts = 2;
	real tolerance = real(0.0001);
};

// Parse user inputs from the command-line and modify the configuration of the program.
void parseArguments(int argc, char** argv, RunConfiguration& rc);

// Modify a specific configuration parameter according to user input.
void setRunConfigurationParameter(const std::string istr, const std::string val, RunConfiguration& rc);


