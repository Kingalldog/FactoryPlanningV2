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
        std::unordered_map<const Work*, WorkPtr> old_to_new;
        for (const auto& work : data.works) {
            auto copy = std::make_shared<Work>(*work);
            old_to_new[work.get()] = copy;
            data_.works.push_back(copy);
        }
        data_.times_matrix = data.times_matrix;
        data_.operations = data.operations;

        // но у каждой операции надо перепривязать Work:
        for (auto& op : data_.operations) {
            if (auto p = op.PtrToWork()) {
                auto it = old_to_new.find(p.get());
                // если не нашли — значит ptr_to_work_ указывал мимо src.works,
                // это отдельный баг, стоит как минимум assert'нуть
                op.SetWorkPtr(it != old_to_new.end() ? it->second : nullptr);
            }
    }

        operations_count_ = data.operations.size();
    }
    ProblemHandler(const ProblemHandler& other) {
        data_.tools = other.data_.tools;
        
        data_.works.clear();
        for (const auto& work : other.data_.works) {
            data_.works.push_back(std::make_shared<Work>(*work));
        }
        
        data_.times_matrix = other.data_.times_matrix;
        data_.operations = other.data_.operations;

        operations_count_ = other.operations_count_;
    }
    ProblemHandler& operator=(const ProblemHandler& other) {
        if (this == &other) return *this;
        ProblemHandler tmp(other);
        std::swap(data_, tmp.data_);
        return *this;
    }
    ProblemData data();
    void setData(ProblemData data);
    void AppointmentProcess(
        size_t chooseToolId,
        const Operation& operation);

int operations_count_ = 0;
private:
    ProblemData data_;
    
};
