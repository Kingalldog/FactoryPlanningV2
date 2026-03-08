#pragma once

#include "basics/problem_data.h"
#include "basics/operation.h"
#include "defines.h"

#include <algorithm>

// this class need's only for ml nodel
// then we appoint operation we changed task
// so logic that after appointment
// we can solve another task
class ProblemHandler {
public:
    ProblemHandler() = default;
    ProblemHandler(ProblemData data) {
        data_.tools = data.tools;

        data_.works.clear();
        for (const auto& work : data.works) {
            data_.works.push_back(std::make_shared<Work>(*work));
        }
        data_.times_matrix = data.times_matrix;
        data_.operations = data.operations;
    }
    ProblemData data();
    void setData(ProblemData data);
    void AppointmentProcess(
        size_t chooseToolId,
        const Operation& operation);
private:
    ProblemData data_;
};
