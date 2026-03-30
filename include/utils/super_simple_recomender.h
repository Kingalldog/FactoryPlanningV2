#ifndef SUPER_SIMPLE_STRING_H
#define SUPER_SIMPLE_STRING_H

#include <cstdio>
#include <iostream>
#include <vector>
#include <sstream>
#include <string>

namespace {

const std::string PYTHON_SCRIPT = "../../model/heuristic_recommender.py";
const std::string DYNAMIC_ML_NAME = "../../model/dynamic_recommender.py";

std::vector<std::string> parseSpaceSeparated(const std::string& input) {
    std::vector<std::string> result;
    
    std::string name = "";
    for(auto e : input) {
        if (e == ' ') {
            result.push_back(name);
            name.clear();
        } else {
            name.push_back(e);
        }
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
    
    std::vector<std::string> getTopAlgoritmth(const std::string& problem, bool isDynamicMl = false, int count = 3) {
        std::string cmd = std::string("python3 " + PYTHON_SCRIPT + " --mode names_only ") +
                          "--model \"" + model_path + "\" " +
                          "--problem \"" + problem + "\" --top_n " + std::to_string(count);
        if (!isDynamicMl) {
            cmd = std::string("python3 " + PYTHON_SCRIPT + " --mode names_only ") +
                          "--model \"" + model_path + "\" " +
                          "--problem \"" + problem + "\" --top_n " + std::to_string(count);
        } else {
            cmd = std::string("python3 " + DYNAMIC_ML_NAME + " --mode recommend ") +
                          "--model \"" + model_path + "\" " +
                          "--problem \"" + problem + "\" --top_n " + std::to_string(count);
        }
        
        std::string output = exec(cmd);

        if (!output.empty() && output.back() == '\n') {
            output.pop_back();
        }

        return parseSpaceSeparated(output);
    }
};

#endif