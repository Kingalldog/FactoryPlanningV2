#include <catch2/catch_test_macros.hpp>

#include "basics/operation.h"
#include "basics/work.h"
#include "defines.h"
#include "solvers/solver.h"
#include "solvers/sorters.h"
#include "utils/generator.h"

TEST_CASE("Operation creation") {
    Operation op{777, false};
    CHECK(op.ID() == 777);
    CHECK(op.Stoppable() == false);
    CHECK(op.HasDependencies() == false);
    CHECK(op.PtrToWork() == nullptr);
    CHECK(op.CntDeps() == 0);
    CHECK(op.Appointed() == false);

    auto st_end = op.GetStEndTimes();
    CHECK(st_end.first == kStartTimePoint);
    CHECK(st_end.second == kStartTimePoint);
}

TEST_CASE("Operation dependencies") {
    // master -> slave
    Operation master(666, true);
    Operation slave(777, true);

    CHECK(master.AddDepended(slave.ID()) == true);
    CHECK(master.AddDepended(slave.ID()) == false);
    CHECK(slave.AddDependency(master.ID()) == true);
    CHECK(master.Depended().size() == 1);
    CHECK(*master.Depended().begin() == 777);
    CHECK(slave.Dependencies().size() == 1);
    CHECK(*slave.Dependencies().begin() == 666);
}

TEST_CASE("Possible tools") {
    Operation op(888, false);
    op.AddPossibleTool(12);
    op.AddPossibleTool(12);
    CHECK(op.PossibleTools().size() == 1);
}

TEST_CASE("Work and operation") {
    WorkPtr work(
        new Work(ch::system_clock::now(), ch::system_clock::now(), 0.5, 888));
    std::vector<Operation> operations;
    operations.reserve(100);
    Operation master_op{0, true};
    operations.push_back(master_op);
    work->AddOperation(master_op.ID());
    for (size_t i = 1; i < operations.capacity(); ++i) {
        Operation operation{i, true};
        operation.SetWorkPtr(work);
        master_op.AddDepended(operation.ID());
        operation.AddDependency(master_op.ID());
        operations.push_back(std::move(operation));
    }

    CHECK(work->OperationIDs().size() == 100);
    CHECK(operations.back().PtrToWork() == work);

    CHECK(master_op.Depended().size() == 99);
    CHECK(master_op.Dependencies().size() == 0);
    CHECK(operations.back().CntDeps() == 1);
    CHECK(operations.back().Depended().size() == 0);

    auto now = ch::system_clock::now();
    master_op.SetTimes(now, now, now - now, operations);
    CHECK(operations.back().CntDeps() == 0);
}

TEST_CASE("Operation UnAppointment") {
    WorkPtr work(
        new Work(ch::system_clock::now(), ch::system_clock::now(), 100, 0));
    std::vector<Operation> ops{Operation(0, true), Operation(1, true)};
    ops[0].AddDepended(1);
    ops[0].SetWorkPtr(work);
    ops[1].AddDependency(0);
    ops[1].SetWorkPtr(work);
    CHECK(ops[1].Dependencies().size() == 1);
    CHECK(ops[0].Depended().size() == 1);

    auto now = ch::system_clock::now();
    CHECK(ops[1].PossibleStart() == kStartTimePoint);
    ops[0].SetTimes(now, now, ch::system_clock::now() - kStartTimePoint, ops);
    CHECK(ops[0].Appointed());
    CHECK(ops[1].PossibleStart() == now);
    CHECK(ops[1].CanBeAppointed(ch::system_clock::now()));
    ops[0].UnAppoint(ops);
    CHECK(!ops[0].Appointed());
    CHECK(!ops[1].CanBeAppointed(now));
    CHECK(ops[1].PossibleStart() == kStartTimePoint);
}

TEST_CASE("Advanced UnAppointment") {
    Generator genadiy;
    GenerationParams params{10,         10,          ch::system_clock::now(),
                            {100, 200}, {500, 1000}, 0.5,
                            0.5,        0.5,         132143};

    ProblemData data(genadiy.Generate(params));
    ProblemDataAndFront problemDataAndFront(data);
    Solver::Solve<DummySorter>(problemDataAndFront, 13212423);
    for (auto& oper : data.operations) {
        CHECK(oper.Appointed());
        std::vector<TimePoint> possible_starts;
        for (size_t idx : oper.Depended()) {
            possible_starts.push_back(data.operations[idx].PossibleStart());
        }
        oper.UnAppoint(data.operations);
        CHECK(!oper.Appointed());
        size_t counter = 0;
        for (size_t idx : oper.Depended()) {
            CHECK(data.operations[idx].PossibleStart() <=
                  possible_starts[counter]);
            ++counter;
        }
    }
}