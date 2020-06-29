#pragma once

#include <set>
#include <string>
#include <vector>

/** Trim a string from left side.
Operates directly on the string passed as argument.
@param [in][out] s The string on which to operate.
*/
void ltrim(std::string& s);

/** Trim a string from right side.
Operates directly on the string passed as argument.
@param [in][out] s The string on which to operate.
*/
void rtrim(std::string& s);

/** Trim a string from left AND right side.
Operates directly on the string passed as argument.
@param [in][out] s The string on which to operate.
*/
void trim(std::string& s);

/** Trim a string from left side.
Creates a copy of the trimmed string on return.
@param [in] s The string on which to operate.
@return String trimmed from left hand-side.
*/
std::string ltrim_copy(std::string s);

/** Trim a string from right side.
Creates a copy of the trimmed string on return.
@param [in] s The string on which to operate.
@return String trimmed from right hand-side.
*/
std::string rtrim_copy(std::string s);

/** Trim a string from left AND right side.
Creates a copy of the trimmed string on return.
@param [in] s The string on which to operate.
@return String trimmed from left AND right hand-side.
*/
std::string trim_copy(std::string s);

/** Convert the contents of a string to lowercase.
This acts directly on the string.
@param [in][out] s The string on which to operate.
*/
void toLowerCase(std::string s);

/** Convert the contents of a string to uppercase.
This acts directly on the string.
@param [in][out] s The string on which to operate.
*/
void toUpperCase(std::string s);

/** Retrieve the components of a full or relative path string.
@param [in] str The string to operate on.
@param [in] delimiters The acceptable delimiter for the path. Defaults to Windows-specific "\".
@return A vector of strings containing the components of the path.
*/
std::vector<std::string> splitpath(const std::string& str, const std::set<char> delimiters = { '\\' });

/** Replace the extension of a file. If no extension is present, then
the given one is appended.
@param [in][out] str The string on which to operate.
@param [in] ext The extension to use for replacement of the current one.
*/
void replaceExtension(std::string& str, const std::string ext);