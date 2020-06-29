#include "string_operations.hpp"

#include <algorithm> 
#include <set>
#include <string>
#include <vector>

// trim from start (in place)
void ltrim(std::string& s) {
	s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
		return !std::isspace(ch);
		}));
}

// trim from end (in place)
void rtrim(std::string& s) {
	s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
		return !std::isspace(ch);
		}).base(), s.end());
}

// trim from both ends (in place)
void trim(std::string& s) {
	ltrim(s);
	rtrim(s);
}

// trim from start (copying)
std::string ltrim_copy(std::string s) {
	ltrim(s);
	return s;
}

// trim from end (copying)
std::string rtrim_copy(std::string s) {
	rtrim(s);
	return s;
}

// trim from both ends (copying)
std::string trim_copy(std::string s) {
	trim(s);
	return s;
}

// Convert the contents of a string to lowercase.
void toLowerCase(std::string s)
{
	std::for_each(s.begin(), s.end(), [](char& c) {
		c = ::tolower(c);
		});
}

// Convert the contents of a string to uppercase.
void toUpperCase(std::string s)
{
	std::for_each(s.begin(), s.end(), [](char& c) {
		c = ::toupper(c);
		});
}

// Break down a full path to its components
std::vector<std::string> splitpath(const std::string& str, const std::set<char> delimiters)
{
	std::vector<std::string> result;

	char const* pch = str.c_str();
	char const* start = pch;
	for (; *pch; ++pch)
	{
		if (delimiters.find(*pch) != delimiters.end())
		{
			if (start != pch)
			{
				std::string str(start, pch);
				result.push_back(str);
			}
			else
				result.push_back("");

			start = pch + 1;
		}
	}
	result.push_back(start);

	return result;
}

// Replace the extension of a file to use elsewhere.
void replaceExtension(std::string& str, const std::string ext)
{
	size_t pos = str.find_last_of('.');
	if (pos > str.length())
		str.append("." + ext);
	else if (pos == str.length() - 1)
		str.append(ext);
	else
		str = str.substr(0, pos + 1) + ext;
}