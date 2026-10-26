#pragma once
#include <algorithm>
#include <iostream>

#include "basics/operation.h"
#include "defines.h"
#include "utils/count_set.h"
#include "utils/printer.h"
#include "problemHandler.h"


enum class MlDatasetCreationType {
    Finished,
    Working
};

using TimePointsQueue = std::set<TimePoint>;

class ProblemDataAndFront {
public:
    ProblemDataAndFront(const ProblemData& data)
        : data_(DeepCopy(data)), problemHandler_(data) {
            preFront_ = CalcPreFront(data);
            timestamps_ = GetStartTimes(data);
        }
    ProblemDataAndFront(const ProblemDataAndFront& other)
        : data_(DeepCopy(other.data_)),
        preFront_(other.preFront_),
        currentTime_(other.currentTime_),
        timestamps_(other.timestamps_),
        problemHandler_(other.problemHandler_)   // его copy ctor делает deep copy
    {}
    ProblemData& data() {
        return data_;
    }
    CountSet<size_t>& preFront() {
        return preFront_;
    }
    TimePoint& currentTime() {
        return currentTime_;
    }
    TimePointsQueue& timestamps() {
        return timestamps_;
    }
    ProblemHandler& problemHandler() {
        return problemHandler_;
    }

    void fixTimes() {
        if (timestamps_.size() == 0) {
            return ;
        }
        currentTime_ = *timestamps_.begin();
        for (auto& work : problemHandler_.data().works) {
            work->start_time_ = std::max(work->start_time_, currentTime_);
        }
    }

private:
    static CountSet<size_t> CalcPreFront(const ProblemData& data) {
        CountSet<size_t> pre_front;
        std::for_each(data.operations.begin(), data.operations.end(),
                      [&](const Operation& op) {
                          if (op.Dependencies().empty()) {
                              pre_front.Insert(op.ID());
                          }
                      });
        return pre_front;
    }

    TimePointsQueue GetStartTimes(const ProblemData& data) {
        TimePointsQueue times_set;
        for (const auto& work : data.works) {
            times_set.insert(work->StartTime());
        }

        for (const auto& tool : data.tools) {
            for (const auto& interval : tool.GetShedule()) {
                times_set.insert(interval.Start());
            }
        }

        return times_set;
    }

    ProblemData data_;
    CountSet<size_t> preFront_;
    TimePoint currentTime_;
    TimePointsQueue timestamps_;
    ProblemHandler problemHandler_;
};

inline std::ostream& operator<<(std::ostream& out, ProblemDataAndFront& problemDataAndFront)
{
    std::set<size_t> operations;
    const auto preFront = problemDataAndFront.preFront();
    for (auto it = preFront.begin(); it != preFront.end(); it++) {
        operations.insert(*it);
    }
    out << "PREFRONT\n" << operations.size() << std::endl;
    for (auto e : operations) {
        out << e << " ";
    }

    out << "\nOPERATIONS_LEFT\n";
    out << problemDataAndFront.problemHandler().operations_count_;
    out << std::endl;
    Printer::PrintProblemData(problemDataAndFront.problemHandler().data(), out);

    return out;
}