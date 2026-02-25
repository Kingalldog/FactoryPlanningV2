#include <atomic>
#include <chrono>
#include <fstream>
#include <optional>
#include <ostream>
#include <thread>

#include "basics/problem_data.h"
#include "gangsta_multistep.h"
#include "multistep_solver.h"
#include "scorers/scorer.h"
#include "solvers/solution_checker.h"
#include "solvers/ml_solver.h"
#include "solvers/solver.h"
#include "solvers/sorters.h"
#include "utils/concurent_grid.h"
#include "utils/dataset.h"
#include "utils/printer.h"

constexpr int kSolverSeedFirst = 123432524;
constexpr int kSolverSeedSecond = 546392715;
constexpr int kSolverSeedThird = 235392578;
constexpr double kScoreMult = 100;

const std::string folderName = "tmp";

template <CanSort Sorter>
double Pipeline(const ProblemData& data) {
    ProblemData data_copy = data;
    Solver::Solve<Sorter>(data_copy, kSolverSeedFirst);
    SolutionChecker::Check(data_copy);
    Score score_1 = BasicScorer::CalcScore(data_copy, kScoreMult);

    data_copy = data;
    Solver::Solve<Sorter>(data_copy, kSolverSeedSecond);
    SolutionChecker::Check(data_copy);
    Score score_2 = BasicScorer::CalcScore(data_copy, kScoreMult);

    data_copy = data;
    Solver::Solve<Sorter>(data_copy, kSolverSeedThird);
    SolutionChecker::Check(data_copy);
    Score score_3 = BasicScorer::CalcScore(data_copy, kScoreMult);
    return (score_1.appointed_fine + score_2.appointed_fine +
            score_3.appointed_fine) /
           3;
}

template <CanSort Sorter, size_t depth, size_t beam_size>
double PipelineForMultistep(const ProblemData& data) {
    ProblemData data_copy = data;
    data_copy = GangstaMultistep::Solve<Sorter, SimpleScorer>(
        data_copy, depth, beam_size, kSolverSeedFirst);
    SolutionChecker::Check(data_copy);
    Score score = BasicScorer::CalcScore(data_copy, kScoreMult);
    return score.appointed_fine;
}

int main() {
    std::vector<std::string> row_names;
    std::vector<std::string> col_names{
        "Dummy",          "Directive",     "Stoppable",      "Robin",
        "FineSorter",     "Dependent",     /*"MDummy6_2",      "MDirective6_2",
        "MStoppable6_2",  "MRobin6_2",     "MFineSorter6_2", "MDependent6_2",
        "MDummy3_4",      "MDirective3_4", "MStoppable3_4",  "MRobin3_4",
        "MFineSorter3_4", "MDependent3_4", "MDummy2_8",      "MDirective2_8",
        "MStoppable2_8",  "MRobin2_8",     "MFineSorter2_8", "MDependent2_8",
        "Aggregator"*/};
    Dataset dataset("../../dataset_directory_big/" + folderName, row_names);
    ConcurentGrid<double> table(row_names.size(), col_names.size(), row_names,
                                col_names, 999999);
    std::vector<std::thread> threads;
    std::atomic<size_t> counter = 0;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&]() {
            ProblemData data;
            std::optional<std::string> name{"dummy"};
            while (name.has_value()) {
                name = dataset.GetNext(&data);
                if (name.has_value()) {
                    auto start = std::chrono::system_clock::now();
                    table.Set(name.value(), "Dummy",
                              Pipeline<DummySorter>(data));
                    table.Set(name.value(), "Directive",
                              Pipeline<DirectiveTimeSorter>(data));
                    table.Set(name.value(), "Stoppable",
                              Pipeline<StoppableSorter>(data));
                    table.Set(name.value(), "Robin",
                              Pipeline<RoundRobinSorter>(data));
                    table.Set(name.value(), "FineSorter",
                              Pipeline<FineSorter>(data));
                    table.Set(name.value(), "Dependent",
                              Pipeline<DependedSorter>(data));
                    table.Set(name.value(), "MlSorter",
                              Pipeline<MlSorter>(data));
                    // table.Set(name.value(), "MDummy6_2",
                    //           PipelineForMultistep<DummySorter, 6, 2>(data));
                    // table.Set(
                    //     name.value(), "MDirective6_2",
                    //     PipelineForMultistep<DirectiveTimeSorter, 6, 2>(data));
                    // table.Set(
                    //     name.value(), "MStoppable6_2",
                    //     PipelineForMultistep<StoppableSorter, 6, 2>(data));
                    // table.Set(
                    //     name.value(), "MRobin6_2",
                    //     PipelineForMultistep<RoundRobinSorter, 6, 2>(data));
                    // table.Set(name.value(), "MFineSorter6_2",
                    //           PipelineForMultistep<FineSorter, 6, 2>(data));
                    // table.Set(name.value(), "MDependent6_2",
                    //           PipelineForMultistep<DependedSorter, 6, 2>(data));

                    // table.Set(name.value(), "MDummy3_4",
                    //           PipelineForMultistep<DummySorter, 3, 4>(data));
                    // table.Set(
                    //     name.value(), "MDirective3_4",
                    //     PipelineForMultistep<DirectiveTimeSorter, 3, 4>(data));
                    // table.Set(
                    //     name.value(), "MStoppable3_4",
                    //     PipelineForMultistep<StoppableSorter, 3, 4>(data));
                    // table.Set(
                    //     name.value(), "MRobin3_4",
                    //     PipelineForMultistep<RoundRobinSorter, 3, 4>(data));
                    // table.Set(name.value(), "MFineSorter3_4",
                    //           PipelineForMultistep<FineSorter, 3, 4>(data));
                    // table.Set(name.value(), "MDependent3_4",
                    //           PipelineForMultistep<DependedSorter, 3, 4>(data));

                    // table.Set(name.value(), "MDummy2_8",
                    //           PipelineForMultistep<DummySorter, 2, 8>(data));
                    // table.Set(
                    //     name.value(), "MDirective2_8",
                    //     PipelineForMultistep<DirectiveTimeSorter, 2, 8>(data));
                    // table.Set(
                    //     name.value(), "MStoppable2_8",
                    //     PipelineForMultistep<StoppableSorter, 2, 8>(data));
                    // table.Set(
                    //     name.value(), "MRobin2_8",
                    //     PipelineForMultistep<RoundRobinSorter, 2, 8>(data));
                    // table.Set(name.value(), "MFineSorter2_8",
                    //           PipelineForMultistep<FineSorter, 2, 8>(data));
                    // table.Set(name.value(), "MDependent2_8",
                    //           PipelineForMultistep<DependedSorter, 2, 8>(data));

                    // table.Set(
                    //     name.value(), "Aggregator",
                    //     PipelineForMultistep<
                    //         SorterAggregator<DummySorter, DirectiveTimeSorter,
                    //                          StoppableSorter, FineSorter,
                    //                          DependedSorter>,
                    //         4, 8>(data));
                    std::cout << "\rProblem #" << std::setw(5)
                              << counter.fetch_add(1)
                              << " solution taked: " << std::setw(8)
                              << std::chrono::duration_cast<
                                     std::chrono::milliseconds>(
                                     std::chrono::system_clock::now() - start)
                              << std::flush;
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    std::ofstream out_file("../model/task_results/" + folderName + ".csv");
    Printer::PrintGridCSV(table, out_file);
    return 0;
}