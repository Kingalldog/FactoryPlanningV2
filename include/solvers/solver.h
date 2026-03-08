#pragma once
#include <algorithm>

#include "basics/operation.h"
#include "defines.h"
#include "solvers/sorters.h"
#include "utils/count_set.h"
#include "problemHandler.h"

using TimePointsQueue = std::set<TimePoint>;

class Solver {
public:
    template <CanSort Sorter>
    static void Solve(ProblemData& data, int seed = 185643241) {
        ProblemHandler problemHandler(data);
        RandomGenerator gena(seed);
        Sorter sorter;
        TimePoint current_time;
        TimePointsQueue timestamps(GetStartTimes(data));

        CountSet<size_t> pre_front(CalcPreFront(data));

        while (!timestamps.empty()) {
            current_time = *timestamps.begin();
            timestamps.erase(current_time);

            IdsVec front = FindFront(pre_front, data, current_time);
            IdsSet possible_tools =
                FindPossibleTools(front, data, current_time);
            std::shuffle(front.begin(), front.end(), gena.GetGen());
            gena.GetGen().seed(gena.GetGen()());
            sorter.SortFront(problemHandler.data(), front, possible_tools);

            // назначение
            // вот здесь надо не первый попавшийся инструмент использовать
            // а для каждой операции в текущем фронте искать самый оптимальный
            AppointmentProcess(front, pre_front, possible_tools, current_time,
                               data, timestamps, problemHandler);
        }
    }

protected:
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
    static TimePointsQueue GetStartTimes(const ProblemData& data);

    static IdsVec FindFront(const CountSet<size_t>& pre_front,
                            const ProblemData& data, TimePoint current_time);
    static IdsSet FindPossibleTools(const IdsVec& front,
                                    const ProblemData& data,
                                    TimePoint current_time);
    static void AppointmentProcess(const IdsVec& front,
                                   CountSet<size_t>& pre_front,
                                   IdsSet& possible_tools,
                                   TimePoint current_time, ProblemData& data,
                                   TimePointsQueue& timestamps,
                                   ProblemHandler& problemHandler);
    RandomGenerator gena_;
};