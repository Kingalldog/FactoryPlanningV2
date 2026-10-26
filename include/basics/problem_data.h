#pragma once
#include <fstream>
#include <string>
#include <unordered_map>

#include "basics/operation.h"
#include "basics/tool.h"
#include "basics/work.h"

TimePoint ParseTimePoint(const std::string& str);

Duration ParseDuration(const std::string& str);

class ProblemData {
public:
    ProblemData(){};
    ProblemData(std::ifstream& input);

    std::vector<Tool> tools;
    std::vector<WorkPtr> works;
    std::vector<Operation> operations;
    std::vector<std::vector<Duration>> times_matrix;
int count_operations = 0;
private:
    void FillTools(std::ifstream& input);
    void FillOperations(std::ifstream& input);
    void FillTimes(std::ifstream& input);
    void AddWork(std::ifstream& input);

    Tool ParseTool(std::ifstream& input, size_t id);
};


ProblemData DeepCopy(const ProblemData& src) ;