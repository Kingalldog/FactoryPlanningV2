#include <atomic>
#include <chrono>
#include <fstream>
#include <optional>
#include <ostream>
#include <thread>
#include <unistd.h>

#include "basics/problem_data.h"
#include "gangsta_multistep.h"
#include "multistep_solver.h"
#include "scorers/scorer.h"
#include "solvers/solution_checker.h"
#include "solvers/ml_sorter.h"
#include "solvers/solver.h"
#include "solvers/problem_data_and_front.h"
#include "solvers/sorters.h"
#include "utils/concurent_grid.h"
#include "utils/dataset.h"
#include "utils/printer.h"

constexpr int kSolverSeedFirst = 123432524;
constexpr int kSolverSeedSecond = 546392715;
constexpr int kSolverSeedThird = 235392578;
constexpr double kScoreMult = 100;
const int countThreds = 10;

namespace {

const std::string folderPath = "../../../factory/dataset_directory_big/";
const std::string mlRemainsPath = "../../../factory/dataset_directory_big/ml_remains/";
const std::string folderName = "for_ml_traning";

void savedData(const std::string& name, ProblemDataAndFront& dataAndFront) {
    std::ofstream out(mlRemainsPath + name, std::ios::app);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + std::string(mlRemainsPath + name));
    }
    
    out << dataAndFront;
    // dataAndFront.Print(out);
    // Printer::PrintProblemData(dataAndFront.data(), out);
    out.close();
}

struct TaskData {
    std::string name;
    std::string algoname;
    double fine;
};
} // namespace

template <CanSort Sorter>
double Pipeline(const ProblemData& data) {
    ProblemData data_copy = data;
    ProblemDataAndFront problemDataAndFront(data);
    Solver::Solve<Sorter>(problemDataAndFront, kSolverSeedFirst);
    SolutionChecker::Check(data_copy);
    Score score_1 = BasicScorer::CalcScore(problemDataAndFront.data(), kScoreMult);

    // data_copy = data;
    // Solver::Solve<Sorter>(data_copy, kSolverSeedSecond);
    // SolutionChecker::Check(data_copy);
    // Score score_2 = BasicScorer::CalcScore(data_copy, kScoreMult);

    // data_copy = data;
    // Solver::Solve<Sorter>(data_copy, kSolverSeedThird);
    // SolutionChecker::Check(data_copy);
    // Score score_3 = BasicScorer::CalcScore(data_copy, kScoreMult);
    // return (score_1.appointed_fine + score_2.appointed_fine +
    //         score_3.appointed_fine) /
    //        3;
    return score_1.appointed_fine;
}

template <CanSort Sorter>
TaskData oneRunning(
    ProblemDataAndFront dataProblemAndFontCopy,
    const std::string& name,
    const std::string algoName)
{
    Solver::Solve<Sorter>(dataProblemAndFontCopy, kSolverSeedFirst, true);
    // solve all with dummy sorting
    Solver::Solve<DummySorter>(dataProblemAndFontCopy, kSolverSeedFirst);
    SolutionChecker::Check(dataProblemAndFontCopy.data());
    Score score_1 = BasicScorer::CalcScore(dataProblemAndFontCopy.data(), kScoreMult);

    return {
        .name = name,
        .algoname = algoName,
        .fine = score_1.appointed_fine
    };
}


void PipelineForMl(
    const ProblemData& data,
    ConcurentGrid<double>& table,
    const std::string& name)
{
    ProblemDataAndFront problemDataAndFront(data);
    auto operationType = MlDatasetCreationType::Working;
    int iteration = 0;
    while (operationType == MlDatasetCreationType::Working) {
        if (iteration > 10) {
            return;
        }
        auto savedName = name.substr(folderPath.length() + folderName.length(), name.length()) +"_itaration_" + std::to_string(iteration);
        savedData(savedName, problemDataAndFront);

        std::vector<TaskData> taskDatas;
        
        // taskDatas.push_back(
        //     oneRunning<DummySorter>(problemDataAndFront, savedName, "Dummy")
        // );
        taskDatas.push_back(
            oneRunning<DummySorter>(problemDataAndFront, savedName, "Dummy")
        );
        taskDatas.push_back(
            oneRunning<DirectiveTimeSorter>(problemDataAndFront, savedName, "Directive")
        );
        taskDatas.push_back(
            oneRunning<StoppableSorter>(problemDataAndFront, savedName, "Stoppable")
        );
        taskDatas.push_back(
            oneRunning<RoundRobinSorter>(problemDataAndFront, savedName, "Robin")
        );
        taskDatas.push_back(
            oneRunning<FineSorter>(problemDataAndFront, savedName, "FineSorter")
        );
        taskDatas.push_back(
            oneRunning<DependedSorter>(problemDataAndFront, savedName, "Dependent")
        );
        
        // solve all with dummy sorting
        std::set<double> fines;
        for(const auto& taskData : taskDatas) {
            fines.insert(taskData.fine);
        }
        // if (fines.size() < 4 || iteration > 10) {
        //     return;
        // }
        for(const auto& taskData : taskDatas) {
            table.Set(taskData.name, taskData.algoname, taskData.fine);
        }

        operationType = Solver::Solve<DummySorter>(problemDataAndFront, kSolverSeedFirst, true);
        ++iteration;
    }
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
        "Dummy",         "Directive",     "Stoppable",      "Robin",
        "FineSorter",     "Dependent"/*,"MlSorter", "MlHardSorter","MDummy6_2",      "MDirective6_2",
        "MStoppable6_2",  "MRobin6_2",     "MFineSorter6_2", "MDependent6_2",
        "MDummy3_4",      "MDirective3_4", "MStoppable3_4",  "MRobin3_4",
        "MFineSorter3_4", "MDependent3_4", "MDummy2_8",      "MDirective2_8",
        "MStoppable2_8",  "MRobin2_8",     "MFineSorter2_8", "MDependent2_8",
        "Aggregator"*/};
    Dataset dataset("../../../factory/dataset_directory_big/" + folderName, row_names);
    ConcurentGrid<double> table(row_names.size(), col_names.size(), row_names,
                                col_names, 999999);
    std::vector<std::thread> threads;
    std::atomic<size_t> counter = 0;
    for (int i = 0; i < countThreds; ++i) {
        threads.emplace_back([&]() {
            ProblemData data;
            std::optional<std::string> name{"dummy"};
            while (name.has_value()) {
                try {
                    name = dataset.GetNext(&data);
                    std::cout << "solving " << name.value_or("") << std::endl;
                    if (name.has_value()) {
                        auto start = std::chrono::system_clock::now();
                        // table.Set(name.value(), "Dummy",
                        //         Pipeline<DummySorter>(data));
                        PipelineForMl(data, table, name.value());
                        // PipelineForMl<DummySorter>(data, table, name.value(), "Dummy");
                        // PipelineForMl<DirectiveTimeSorter>(data, table, name.value(), "Directive");
                        // PipelineForMl<StoppableSorter>(data, table, name.value(), "Stoppable");
                        // PipelineForMl<RoundRobinSorter>(data, table, name.value(), "Robin");
                        // PipelineForMl<FineSorter>(data, table, name.value(), "FineSorter");
                        // PipelineForMl<DependedSorter>(data, table, name.value(), "Dependent");
                        // table.Set(name.value(), "Dummy",
                        //         Pipeline<DummySorter>(data));
                        // table.Set(name.value(), "Directive",
                        //         Pipeline<DirectiveTimeSorter>(data));
                        // table.Set(name.value(), "Stoppable",
                        //         Pipeline<StoppableSorter>(data));
                        // table.Set(name.value(), "Robin",
                        //         Pipeline<RoundRobinSorter>(data));
                        // table.Set(name.value(), "FineSorter",
                        //         Pipeline<FineSorter>(data));
                        // table.Set(name.value(), "Dependent",
                        //         Pipeline<DependedSorter>(data));
                        // table.Set(name.value(), "MlSorter",
                        //         Pipeline<MlSorter>(data));
                        // table.Set(name.value(), "MlHardSorter",
                        //         Pipeline<MlHardSorter>(data));
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
                        usleep(500000);
                    }
                } catch (std::exception& e) {
                    std::cout << e.what() <<
                        " in file " << name.value_or("");
                }
            }
            });
    }

    for (auto& t : threads) {
        t.join();
    }

    std::ofstream out_file(folderName + ".csv");
    Printer::PrintGridCSV(table, out_file);
    return 0;
}