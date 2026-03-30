#include <time.h>

#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>

#include "basics/problem_data.h"
#include "gangsta_multistep.h"
#include "scorers/scorer.h"
#include "solvers/solution_checker.h"
#include "solvers/solver.h"
#include "solvers/sorters.h"
#include "utils/generator.h"
#include "utils/printer.h"


using namespace std::chrono_literals;

constexpr int kSolverSeed = 1232412;
constexpr double kScoreMult = 100;

template <CanSort Sorter>
void Pipeline(const ProblemData& generated_data, std::string out_filename) {
    std::ofstream out_file(out_filename);
    ProblemData data_copy = generated_data;
    ProblemDataAndFront problemDataAndFront(data_copy);
    Solver::Solve<Sorter>(problemDataAndFront, kSolverSeed);
    SolutionChecker::Check(data_copy);

    Score score = BasicScorer::CalcScore(data_copy, kScoreMult);

    Printer::PrintAnswerJSON(data_copy, score, out_file);
    std::cout << out_filename << ": " << score.appointed_fine << " || "
              << score.not_appointed_fine << std::endl;
}

constexpr size_t kCntTools = 20;
constexpr size_t kCntIntervals = 20;
constexpr std::pair<int, int> kSpacerDur = {100, 200};
constexpr std::pair<int, int> kIntervalDur = {1000, 2000};
constexpr double kOpGenProb = 0.5;
constexpr double kAddEdgeProb = 0.5;
constexpr double kAddToolProb = 0.5;
constexpr int kSeedGenerator = 65783928;

int main() {
    Generator gen;
    GenerationParams params{
        kCntTools,    kCntIntervals, ch::system_clock::now(),
        kSpacerDur,   kIntervalDur,  kOpGenProb,
        kAddEdgeProb, kAddToolProb,  kSeedGenerator};
    ProblemData data(gen.Generate(params));
    std::cout << "data created!\n";

    Pipeline<DummySorter>(data, "solution_dummy.json");
    Pipeline<DirectiveTimeSorter>(data, "solution_directive.json");
    Pipeline<StoppableSorter>(data, "solution_stoppable.json");
    Pipeline<RoundRobinSorter>(data, "solution_robin.json");
    Pipeline<FineSorter>(data, "solution_finework.json");
    Pipeline<DependedSorter>(data, "solution_depends.json");
    return 0;
}