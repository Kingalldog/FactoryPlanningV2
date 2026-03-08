#pragma once

#include "sorters.h"
#include "../utils/super_simple_recomender.h"
#include "../utils/printer.h"

#include <variant>
#include <mutex>
#include <thread>

namespace {

const std::string DUMMMY_SORTER = "Dummy";
const std::string DIRECTIVE_SORTER = "Directive";
const std::string STOPABLE_SORTER = "Stoppable";
const std::string ROBIN_SORTER = "Robin";
const std::string FINE_SORTER = "FineSorter";
const std::string DEPEND_SORTER = "Dependent";

const std::string MODEL_PATH = "../../model/heuristic_model.pkl";
const std::string DATA_PATH = "tmp.txt";

using SorterVariant = std::variant<
    DummySorter,
    DirectiveTimeSorter,
    StoppableSorter,
    FineSorter,
    DependedSorter,
    RoundRobinSorter
>;

// Функция для применения сортировки
void applySort(SorterVariant& sorter, const ProblemData& data, 
               IdsVec& front, const IdsSet& tools) {
    std::visit([&](auto& s) {
        s.SortFront(data, front, tools);
    }, sorter);
}

SorterVariant toSorter(const std::string& sorter)
{
    if (sorter == DUMMMY_SORTER) {
        return DummySorter{};
    } else if (sorter == DIRECTIVE_SORTER) {
        return DirectiveTimeSorter{};
    } else if (sorter == STOPABLE_SORTER) {
        return StoppableSorter{};
    } else if (sorter == FINE_SORTER) {
        return FineSorter{};
    } else if (sorter == DEPEND_SORTER) {
        return DependedSorter{};
    } else if (sorter == ROBIN_SORTER) {
        return RoundRobinSorter{};
    }
    throw std::runtime_error("Unknown sorter type");
}

SorterVariant getBestSorter(const ProblemData& data, int cnt = 0)
{
    static std::mutex fileMutex;
    
    try {
        // Записываем в файл под защитой мьютекса
        std::string path = DATA_PATH;
        {
            std::lock_guard<std::mutex> lock(fileMutex);
            
            std::ofstream out(path);
            if (!out.is_open()) {
                throw std::runtime_error("Cannot open file for writing: " + std::string(DATA_PATH));
            }
            
            Printer::PrintProblemData(data, out);
            // out закроется автоматически здесь
        } // Мьютекс освобождается после записи
        
        // Читаем из файла без блокировки (если другие потоки не пишут одновременно)
        SimpleRecommender recomender(MODEL_PATH);

        auto recommendedAlgoritmhs = recomender.getTopAlgoritmth(path);
        
        std::cout << recommendedAlgoritmhs[0] << " " << recommendedAlgoritmhs[1] << std::endl;
        return toSorter(recommendedAlgoritmhs[0]);
        
    } catch (std::exception& e) {
        std::cout << "Error in getBestSorter: " << e.what() << std::endl;
        return DummySorter{};
    }
}

} // namespace

struct MlSorter {
    void SortFront(const ProblemData& data, IdsVec& front,
                          const IdsSet& tools) {

        if (countSortings == 0) {
            sorter = getBestSorter(data);
            std::cout << "Ml choose sorter" << std::endl;
        }
        applySort(sorter, data, front, tools);
        countSortings++;
    }

    SorterVariant sorter;
    int countSortings = 0;
};


struct MlHardSorter {
    void SortFront(const ProblemData& data, IdsVec& front,
                          const IdsSet& tools) {
        if (countSortings % freqSorting == 0) {
            sorter = getBestSorter(data, countSortings);
            std::cout << "Changed Hard solver on sorting=" << countSortings << std::endl;
        }
        
        applySort(sorter, data, front, tools);
        countSortings++;
    }

    std::atomic<int> countSortings = 0;
    const int freqSorting = 250;
    SorterVariant sorter = DummySorter{};
};

// using TimePointsQueue = std::set<TimePoint>;
// class MlSolver {
// public:
//     static void Solve(ProblemData& data, int seed = 185643241) {
//         RandomGenerator gena(seed);

//         auto sorter = getBestSorter(data);

//         TimePoint current_time;
//         TimePointsQueue timestamps(GetStartTimes(data));

//         CountSet<size_t> pre_front(CalcPreFront(data));

//         while (!timestamps.empty()) {
//             current_time = *timestamps.begin();
//             timestamps.erase(current_time);

//             IdsVec front = FindFront(pre_front, data, current_time);
//             IdsSet possible_tools =
//                 FindPossibleTools(front, data, current_time);
//             std::shuffle(front.begin(), front.end(), gena.GetGen());
//             gena.GetGen().seed(gena.GetGen()());
//             applySort(sorter, data, front, possible_tools);

//             // назначение
//             // вот здесь надо не первый попавшийся инструмент использовать
//             // а для каждой операции в текущем фронте искать самый оптимальный
//             AppointmentProcess(front, pre_front, possible_tools, current_time,
//                                data, timestamps);
//         }
//     }

// protected:
//     static CountSet<size_t> CalcPreFront(const ProblemData& data) {
//         CountSet<size_t> pre_front;
//         std::for_each(data.operations.begin(), data.operations.end(),
//                       [&](const Operation& op) {
//                           if (op.Dependencies().empty()) {
//                               pre_front.Insert(op.ID());
//                           }
//                       });
//         return pre_front;
//     }
//     static TimePointsQueue GetStartTimes(const ProblemData& data);

//     static IdsVec FindFront(const CountSet<size_t>& pre_front,
//                             const ProblemData& data, TimePoint current_time);
//     static IdsSet FindPossibleTools(const IdsVec& front,
//                                     const ProblemData& data,
//                                     TimePoint current_time);
//     static void AppointmentProcess(const IdsVec& front,
//                                    CountSet<size_t>& pre_front,
//                                    IdsSet& possible_tools,
//                                    TimePoint current_time, ProblemData& data,
//                                    TimePointsQueue& timestamps);
//     RandomGenerator gena_;
// };