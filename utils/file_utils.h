#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <vector>
#include <string>
#include <fstream>
#include <random>

#include <glob.h>
#include "json.hpp"

#define ALPHABET "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

// Read filenames into string vector
inline std::vector<std::string> getFilenamesFromFolder(const std::string& pattern){
    glob_t glob_result;
    std::vector<std::string> files;

    glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);

    for(unsigned int i = 0; i < glob_result.gl_pathc; ++i){
        files.push_back(std::string(glob_result.gl_pathv[i]));
    }

    globfree(&glob_result);
    return files;
}

// Read JSON File into nlohmann::json object
inline nlohmann::json read_json(const std::string& json_name){
    nlohmann::json json_parser;

    // Read json as file
    std::ifstream json_filename(json_name);
    
    // pass input stream in to json
    json_filename >> json_parser;

    return json_parser;
}

//Generate Random Filename
inline std::string generate_random_filename(int string_len, std::string ext){
    std::string str(ALPHABET);

    std::random_device rd;
    std::mt19937 generator(rd());

    std::shuffle(str.begin(), str.end(), generator);

    return str.substr(0, string_len) + ext;
}

#endif

