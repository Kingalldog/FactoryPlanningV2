#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <chrono>

#include "basics/problem_data.h"
#include "basics/time_interval.h"
#include "basics/tool.h"
#include "defines.h"
#include "solvers/solver.h"
#include "solvers/sorters.h"
#include "utils/generator.h"

bool CheckToolDoThisOperation(const Tool& tool, size_t id) {
    if (tool.GetWorkProcess().empty()) {
        return false;
    }
    return std::any_of(tool.GetWorkProcess().begin(),
                       tool.GetWorkProcess().end(),
                       [&](const NamedTimeInterval& interval) {
                           return interval.OperationID() == id;
                       });
}

IdsSet GetWhatToolDo(const Tool& tool) {
    IdsSet answer;
    for (const auto& inter : tool.GetWorkProcess()) {
        answer.insert(inter.OperationID());
    }
    return answer;
}

TEST_CASE("Tool deleting operation") {
    Generator genadiy;
    GenerationParams params{10,         10,          ch::system_clock::now(),
                            {100, 200}, {500, 1000}, 0.5,
                            0.5,        0.5,         132143};

    ProblemData data(genadiy.Generate(params));
    CHECK(!CheckToolDoThisOperation(data.tools[0], 213));
    ProblemDataAndFront problemDataAndFront(data);
    Solver::Solve<DummySorter>(problemDataAndFront, 13212423);

    for (auto& tool : data.tools) {
        IdsSet ids = GetWhatToolDo(tool);
        for (size_t id : ids) {
            CHECK(CheckToolDoThisOperation(tool, id));
            tool.RemOpFromWorkProceess(id);
            CHECK(!CheckToolDoThisOperation(tool, id));
        }
    }
}