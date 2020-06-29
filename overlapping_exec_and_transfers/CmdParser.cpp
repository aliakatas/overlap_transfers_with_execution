#include "CmdParser.h"
// Define the command line options here: { "number of row and columns", "number of rows", "number of columns", "number of repetitions",
//                                         "timestep", "number of streams", "number of partitions", "tolerance" }
std::string options[] = { "-n", "-r", "-c", "-rep", "-dt", "-s", "-p", "-tol" };

/**/
void parseArguments(int argc, char** argv, RunConfiguration &rc)
{
	int iarg = 0;
	while (iarg < argc)
	{
		for (auto i = 0; i < size(options); ++i)
		{
			if (options[i].compare(argv[iarg]) == 0)
				setRunConfigurationParameter(options[i], argv[++iarg], rc);
		}
		++iarg;
	}
}

/**/
void setRunConfigurationParameter(const std::string istr, const std::string val, RunConfiguration& rc)
{
	if (istr.compare("-n") == 0)
	{
		rc.ncols = std::stoi(val);
		rc.nrows = std::stoi(val);
	}
	else if (istr.compare("-r") == 0)
		rc.nrows = std::stoi(val);
	else if (istr.compare("-c") == 0)
		rc.ncols = std::stoi(val);
	else if (istr.compare("-rep") == 0)
		rc.reps = std::stoi(val);
	else if (istr.compare("-dt") == 0)
	{
		if (sizeof(real) == sizeof(float))
			rc.dt = std::stof(val);
		else
			rc.dt = std::stod(val);
	}
	else if (istr.compare("-s") == 0)
		rc.n_streams = std::stoi(val);
	else if (istr.compare("-p") == 0)
		rc.parts = std::stoi(val);
	else if (istr.compare("-tol") == 0)
	{
		if (sizeof(real) == sizeof(float))
			rc.tolerance = std::stof(val);
		else
			rc.tolerance = std::stod(val);
	}
}