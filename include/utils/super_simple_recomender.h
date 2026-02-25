#ifndef SUPER_SIMPLE_STRING_H
#define SUPER_SIMPLE_STRING_H

#include <cstdio>
#include <iostream>
#include <vector>
#include <sstream>
#include <string>

namespace {

std::vector<std::string> parseSpaceSeparated(const std::string& input) {
    std::vector<std::string> result;
    std::stringstream ss(input);
    std::string token;
    
    while (ss >> token) {  // >> автоматически разделяет по пробелам
        result.push_back(token);
    }
    
    return result;
}

} // namespace

class SimpleRecommender {
private:
    std::string model_path;
    
    std::string exec(const std::string& cmd) {
        char buffer[128];
        std::string result;
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) return "ERROR";
        
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            result += buffer;
        }
        pclose(pipe);
        return result;
    }
    
public:
    SimpleRecommender(const std::string& path) : model_path(path) {}
    
    std::vector<std::string> getTopAlgoritmth(const std::string& problem, int count = 3) {
        std::string cmd = std::string("python3 ../model/heuristic_recommender.py --mode names_only ") +
                          "--model \"" + model_path + "\" " +
                          "--problem \"" + problem + "\" --top_n " + std::to_string(count);
        
        std::string output = exec(cmd);

        if (!output.empty() && output.back() == '\n') {
            output.pop_back();
        }
        
        return parseSpaceSeparated(output);
    }
};

#endif