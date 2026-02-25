#pragma once

#include "sorters.h"
#include "super_simple_recomender.h"
#include "../utils/printer.h"

#include <variant>

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
    }
    throw std::runtime_error("Unknown sorter type");
}

} // namespace

struct MlSorter {
    static void SortFront(const ProblemData& data, IdsVec& front,
                          const IdsSet& tools) {
        std::ofstream out(DATA_PATH);
        Printer::PrintProblemData(data, out);

        SimpleRecommender recomender(MODEL_PATH);
        auto recommendedAlgoritmhs = recomender.getTopAlgoritmth(DATA_PATH);

        auto sorter = toSorter(recommendedAlgoritmhs[0]);
        applySort(sorter, data, front, tools);
    }
};