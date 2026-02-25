#pragma once
#include <concepts>
#include <deque>
#include <functional>
#include <queue>
#include <unordered_map>

#include "basics/problem_data.h"
#include "defines.h"

template <typename T>
concept CanSort = requires(T obj, const ProblemData& data, IdsVec& front,
                           const IdsSet& tools) {
    { obj.SortFront(data, front, tools) } -> std::same_as<void>;
};

struct DummySorter {
    static void SortFront(const ProblemData& data, IdsVec& front,
                          const IdsSet& tools) {}
};

struct DirectiveTimeSorter {
    static void SortFront(const ProblemData& data, IdsVec& front,
                          const IdsSet& tools) {
        std::sort(front.begin(), front.end(), [&](size_t a, size_t b) {
            return data.operations[a].PtrToWork()->Directive() <
                   data.operations[b].PtrToWork()->Directive();
        });
    }
};

struct StoppableSorter {
    static void SortFront(const ProblemData& data, IdsVec& front,
                          const IdsSet& tools) {
        std::sort(front.begin(), front.end(), [&](size_t a, size_t b) {
            return data.operations[a].Stoppable() >
                   data.operations[b].Stoppable();
        });
    }
};

struct FineSorter {
    static void SortFront(const ProblemData& data, IdsVec& front,
                          const IdsSet& tools) {
        std::sort(front.begin(), front.end(), [&](size_t a, size_t b) {
            return data.operations[a].PtrToWork()->FineCoef() >
                   data.operations[b].PtrToWork()->FineCoef();
        });
    }
};

struct DependedSorter {
    static void SortFront(const ProblemData& data, IdsVec& front,
                          const IdsSet& tools) {
        std::sort(front.begin(), front.end(), [&](size_t a, size_t b) {
            return data.operations[a].Depended().size() >
                   data.operations[b].Depended().size();
        });
    }
};

struct RoundRobinSorter {
    void SortFront(const ProblemData& data, IdsVec& front,
                   const IdsSet& tools) {
        if (works_q.empty()) {
            for (auto work : data.works) {
                works_q.push_back(work->ID());
            }
        }
        std::unordered_map<size_t, size_t> position_map;

        for (size_t i = 0; i < works_q.size(); ++i) {
            position_map[works_q[i]] = i;
        }

        std::sort(front.begin(), front.end(), [&](size_t a, size_t b) {
            return position_map[data.operations[a].PtrToWork()->ID()] <
                   position_map[data.operations[b].PtrToWork()->ID()];
        });

        works_q.push_back(works_q.front());
        works_q.pop_front();
    }

    std::deque<size_t> works_q;
};

template <CanSort... Sorter>
class SorterAggregator {
private:
    using SorterFunc =
        std::function<void(const ProblemData&, IdsVec&, const IdsSet&)>;
    std::queue<SorterFunc> queue_;

public:
    template <CanSort SorterOne>
    void AddSorter() {
        SorterOne sorter;
        queue_.emplace(
            [&](const ProblemData& data, IdsVec& front, const IdsSet& tools) {
                sorter.SortFront(data, front, tools);
            });
    }

    SorterAggregator() { (AddSorter<Sorter>(), ...); }

    void SortFront(const ProblemData& data, IdsVec& front,
                   const IdsSet& tools) {
        auto curr_sorter = queue_.front();
        queue_.pop();
        curr_sorter(data, front, tools);
        queue_.push(curr_sorter);
    }
};
