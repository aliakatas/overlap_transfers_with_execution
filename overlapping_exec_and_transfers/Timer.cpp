#include <stdlib.h>
#include <chrono>
#include <iostream>

#include "Timer.h"

Timer::Timer()
{
	n = 0;
}

Timer::~Timer()
{
}

void Timer::print()
{
	std::cout << "\n";
	std::cout << "Action :: time/s Time resolution = " << 1.f / (float)CLOCKS_PER_SEC << "\n";
	std::cout << "------" << "\n";

	for (auto i = 0; i < n; ++i)
		std::cout << labels[i] << " :: " << (times[2 * i + 1] - times[2 * i + 0]) / (float)CLOCKS_PER_SEC << "\n";
}

void Timer::start(std::string label)
{
	if (n < 20)
	{
		labels[n] = label;
		times[2 * n] = clock();
	}
	else {
		std::cerr << "No more timers, " << label
			<< " will not be timed." << std::endl;
	}
}

void Timer::stop()
{
	times[2 * n + 1] = clock();
	n++;
}

void Timer::reset()
{
	n = 0;
}