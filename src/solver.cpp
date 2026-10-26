#include "solvers/solver.h"

#include "defines.h"
#include "utils/count_set.h"

#include <iostream>

TimePointsQueue Solver::GetStartTimes(const ProblemData& data) {
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

IdsVec Solver::FindFront(const CountSet<size_t>& pre_front,
                         const ProblemData& data, TimePoint current_time) {
    IdsVec front;
    for (auto it = pre_front.UniqueBegin(); it != pre_front.UniqueEnd(); ++it) {
        size_t oper = *it;
        const Operation& operation = data.operations[oper];
        if (operation.CanBeAppointed(current_time)) {
            front.push_back(operation.ID());
        }
    }
    return front;
}

IdsSet Solver::FindPossibleTools(const IdsVec& front, const ProblemData& data,
                                 TimePoint current_time) {
    IdsSet possible_tools;
    for (size_t oper : front) {
        for (size_t i : data.operations[oper].PossibleTools()) {
            if (data.tools[i]
                    .CanStartWork(data.operations[oper], current_time,
                                  data.times_matrix[oper][i])
                    .first) {
                possible_tools.insert(i);
            }
        }
    }
    return possible_tools;
}

void Solver::AppointmentProcess(const IdsVec& front,
                                CountSet<size_t>& pre_front,
                                IdsSet& possible_tools, TimePoint current_time,
                                ProblemData& data,
                                TimePointsQueue& timestamps,
                                ProblemHandler& problemHandler) {
    std::set<size_t> removedPreforntIndex;
    for (size_t oper : front) {
        auto& operation = data.operations[oper];
        if (!operation.CanBeAppointed(current_time)) {
            continue;
        }

        size_t best_tool_id = INT64_MAX;
        Duration best_duration = Duration(INT64_MAX);
        for (size_t r : possible_tools) {
            auto result = data.tools[r].CanStartWork(
                operation, current_time, data.times_matrix[oper][r]);
            if (operation.PossibleTools().contains(r) && result.first) {
                if (result.second < best_duration) {
                    best_duration = result.second;
                    best_tool_id = r;
                }
            }
        }
        if (best_tool_id != INT64_MAX) {
            data.tools[best_tool_id].Appoint(
                operation, current_time, data.times_matrix[oper][best_tool_id],
                data.operations);
            timestamps.insert(operation.GetStEndTimes().second);
            possible_tools.erase(best_tool_id);
            // if (oper == 0) {
            //     std::cout << "DELETENG 0" << std::endl << std::endl;
            // }
            // if (operation.ID() == 0) {
            //     std::cout << "DELETENG 0 vvvv" << std::endl << std::endl;
            // }
            pre_front.EraseOne(operation.ID());
            // if (operation.ID() == 0) {
            //     std::cout << pre_front.Count(0) << std::endl << std::endl;
            // }
            for (size_t id : operation.Depended()) {
                pre_front.Insert(id);
            }

            removedPreforntIndex.insert(oper);
            problemHandler.AppointmentProcess(best_tool_id, operation);
            data.count_operations--;
        }
    }
    
}