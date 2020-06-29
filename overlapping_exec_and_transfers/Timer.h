#pragma once

#include <string>

class Timer
{
public:
	Timer();
	~Timer();

	void start(std::string label);
	void stop();
	void reset();
	void print();

private:
	std::string labels[20];
	float times[40];
	int n;
};

