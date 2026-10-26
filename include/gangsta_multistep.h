#pragma once
#include <cfloat>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <iomanip>   // for std::setw
#include <memory>
#include <stack>

#include "basics/problem_data.h"
#include "defines.h"
#include "multistep_solver.h"
#include "solvers/solver.h"
#include "utils/count_set.h"

/*
Score DFS(ProblemData& data, TimerQueue times, size_t depth) {
    if (depth == 0) {
        return Scorer::CalcScore(data);
    }

    Score min_time = MAX_TIME;
    for (size_t i; i < bfs_size; ++i) {
        IdsVec front = GetFront(data, timer_queue.top());
        sorter.SortFront(front);
        auto for_dp, new_times, top_time = Appointment(data, front);
        Score temp = DFS(data, [times, new_times], depth - 1);
        if (min_time > temp) {
            min_time = temp;
            dp[depth] = for_dp;
        }
        // отмена (восстановление задачи, временной очереди)
        Undo(data, for_dp, times, top_time);
    }
}

*/

// нода дерева в которой будем хранить все
// назначения данного такта и состояние очереди тактов планирования
// на момент последнего назначения.

struct Appointment {
    size_t operation_idx;
    size_t tool_idx;
    TimePoint start;
    TimePoint end;
};

struct Node;
using NodePtr = std::shared_ptr<Node>;

struct Node {
    std::stack<Appointment> appointments;  // назначения в 1м такте
    std::vector<NodePtr> childs;           // потомки
};

using namespace std::chrono_literals;

void PrintVector(IdsVec vec) {
    std::cout << '|';
    for (size_t id : vec) {
        std::cout << std::setw(4) << id << " ";
    }
    std::cout << '|';
}

class GangstaMultistep : Solver {
public:
    template <CanSort OneStepSorter, CanScoreInter Scorer>
    static ProblemData Solve(const ProblemData& data, size_t max_depth,
                             size_t beam_size, int seed = 123432524) {
        std::mt19937 gen(seed);
        OneStepSorter sorter;
        ProblemData answer = data;
        TimePointsQueue timestamps(GetStartTimes(data));
        CountSet<size_t> pre_front(CalcPreFront(data));
        NodePtr start = std::make_shared<Node>(Node{{}, {}});

        while (!AllOpperationsAppointed(answer) && !timestamps.empty()) {
            NodePtr best_leaf;
            Score best_score{DBL_MAX, DBL_MAX};

            std::stack<std::pair<NodePtr, size_t>> traversal_stack;
            traversal_stack.push({start, 0});
            while (!traversal_stack.empty()) {
                auto& [current_node, next_child_id] = traversal_stack.top();
                /*if (timestamps.empty()) {
                    traversal_stack.pop();
                    continue;
                }*/
                if (next_child_id == 0) {
                    TimePoint current_time{*timestamps.begin()};
                    timestamps.erase(current_time);

                    IdsVec front = FindFront(pre_front, answer, current_time);
                    IdsSet possible_tools =
                        FindPossibleTools(front, answer, current_time);
                    std::shuffle(front.begin(), front.end(), gen);
                    gen.seed(gen());
                    sorter.SortFront(data, front, possible_tools);
                    current_node->appointments =
                        AppointmentProcess(front, pre_front, possible_tools,
                                           current_time, answer, timestamps);
                }

                size_t current_depth = traversal_stack.size();

                if (current_depth >= max_depth || next_child_id >= beam_size) {
                    if (current_depth >= max_depth) {
                        Score score{Scorer::CalcScore(answer)};
                        if (score < best_score) {
                            best_score = score;
                            best_leaf = current_node;
                        }
                    }
                    UndoChanges(answer, timestamps, pre_front, current_node);
                    traversal_stack.pop();
                    continue;
                }

                NodePtr child = std::make_shared<Node>(Node{{}, {}});
                current_node->childs.push_back(child);
                ++next_child_id;
                traversal_stack.push({child, 0});
            }

            // вот тут найдем путь до лучшего "листа" (т. е. все назначения)
            // поставим start на лучшую ноду и от неё построим следующее дерево
            std::stack<NodePtr> way_to_best_reverse;
            FindPath(start, best_leaf, way_to_best_reverse);
            std::vector<NodePtr> way_to_best(ReverseStack(way_to_best_reverse));

            for (NodePtr node : way_to_best) {
                auto appointments = ReverseStack(node->appointments);
                for (Appointment app : appointments) {
                    answer.tools[app.tool_idx].Appoint(
                        answer.operations[app.operation_idx], app.start,
                        answer.times_matrix[app.operation_idx][app.tool_idx],
                        answer.operations);
                    timestamps.insert(app.end);
                    timestamps.erase(app.start);
                    for (size_t id :
                         data.operations[app.operation_idx].Depended()) {
                        pre_front.Insert(id, 1);
                    }
                }
            }

            start = best_leaf;
            start->appointments = {};
        }
        return answer;
    }

private:
    template <typename T>
    static std::vector<T> ReverseStack(std::stack<T> stack) {
        std::vector<T> answer(stack.size());
        size_t right = stack.size() - 1;
        while (!stack.empty()) {
            answer[right] = stack.top();
            --right;
            stack.pop();
        }

        return answer;
    }
    static bool FindPath(NodePtr current, NodePtr target,
                         std::stack<NodePtr>& path) {
        if (!current) {
            return false;
        }

        path.push(current);

        if (current == target) {
            return true;
        }

        for (NodePtr child : current->childs) {
            if (FindPath(child, target, path)) {
                return true;
            }
        }

        path.pop();
        return false;
    }

    static bool AllOpperationsAppointed(const ProblemData& data) {
        return std::all_of(data.operations.begin(), data.operations.end(),
                           [&](const Operation& op) { return op.Appointed(); });
    }

    static void UndoChanges(ProblemData& data, TimePointsQueue& timestamps,
                            CountSet<size_t>& pre_front, const NodePtr node) {
        std::stack<Appointment> buffer = node->appointments;
        while (!buffer.empty()) {
            auto [op_idx, tool_idx, start, end] = buffer.top();
            buffer.pop();
            data.operations[op_idx].UnAppoint(data.operations);
            data.tools[tool_idx].RemOpFromWorkProceess(op_idx);
            timestamps.erase(end);
            timestamps.insert(start);
            pre_front.Insert(op_idx);
            for (size_t op_dep_idx : data.operations[op_idx].Depended()) {
                pre_front.EraseOne(op_dep_idx);
            }
        }
    }

    static std::stack<Appointment> AppointmentProcess(
        const IdsVec& front, CountSet<size_t>& pre_front,
        IdsSet& possible_tools, TimePoint current_time, ProblemData& data,
        TimePointsQueue& timestamps) {
        std::stack<Appointment> answer;
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
                    operation, current_time,
                    data.times_matrix[oper][best_tool_id], data.operations);
                timestamps.insert(operation.GetStEndTimes().second);
                possible_tools.erase(best_tool_id);
                pre_front.EraseOne(oper);
                for (size_t id : operation.Depended()) {
                    pre_front.Insert(id, 1);
                }
                answer.push(Appointment{oper, best_tool_id, current_time,
                                        operation.GetStEndTimes().second});
            }
        }
        return answer;
    }
};