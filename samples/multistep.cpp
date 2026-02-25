#include <iostream>

#include "basics/problem_data.h"
#include "gangsta_multistep.h"
#include "scorers/scorer.h"
#include "solvers/solution_checker.h"
#include "solvers/sorters.h"
#include "utils/generator.h"

constexpr size_t kCntTools = 10;
constexpr size_t kCntIntervals = 20;
constexpr std::pair<int, int> kSpacerDur = {100, 150};
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
    ProblemData data3 = GangstaMultistep::Solve<StoppableSorter, SimpleScorer>(
        data, 4, 5, 123432524);
    SolutionChecker::Check(data3);
    Score score = BasicScorer::CalcScore(data3, 100);
    std::cout << "Basic Multistep: " << score.appointed_fine << " || "
              << score.not_appointed_fine << '\n';

    ProblemData data2 = GangstaMultistep::Solve<
        SorterAggregator<DummySorter, DirectiveTimeSorter, StoppableSorter,
                         FineSorter, DependedSorter>,
        SimpleScorer>(data, 4, 8, 123432524);
    SolutionChecker::Check(data2);
    score = BasicScorer::CalcScore(data2, 100);
    std::cout << "Cool Multistep: " << score.appointed_fine << " || "
              << score.not_appointed_fine << '\n';

    ProblemData data_copy = data;
    Solver::Solve<StoppableSorter>(data_copy, 123432524);
    SolutionChecker::Check(data_copy);
    Score score_2 = BasicScorer::CalcScore(data_copy, 100);
    std::cout << "One step: " << score_2.appointed_fine << " || "
              << score_2.not_appointed_fine << '\n';

    return 0;
}