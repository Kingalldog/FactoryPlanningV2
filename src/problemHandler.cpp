#include "solvers/problemHandler.h"

namespace {

// void changeToolSchedule(
//     Tool& tool,
//     const Operation& operation,
//     bool choosenTool)
// {
//     const auto& [startTime, endTime] = operation.GetStEndTimes();
//     auto schedule = tool.shedule();
//     for (auto& interval : schedule) {
//         tool.removeTimeInterval(interval);

//         if (interval.Start() <= endTime) {
//             if (!choosenTool) {
//                 tool.addTimeInterval(TimeInterval(
//                     std::max(startTime, interval.Start()),
//                     std::min(endTime, interval.End())));
//             } else {
//                 // значит мы на этот станок как раз и назначили операцию
//                 tool.addTimeInterval(TimeInterval(
//                     std::max(endTime, interval.Start()),
//                     std::min(endTime, interval.End())));
//             }
//         }
//     }
// }

} // namespace

ProblemData ProblemHandler::data()
{
    return data_;
}

void ProblemHandler::AppointmentProcess(
    size_t chooseToolId,
    const Operation& operation)
{
    // // clearing tool.schedule_
    const auto& [startTime, endTime] = operation.GetStEndTimes();
    // for (size_t toolId = 0; toolId < data_.tools.size(); ++toolId) {
    //     changeToolSchedule(data_.tools[toolId], operation, toolId == chooseToolId);
    // }

    // remove depencies
    for (auto& dataOperation : data_.operations) {
        dataOperation.DelDependency(operation.ID(), endTime);
    }

    operations_count_--;

    // remove operations from work
    std::vector<size_t> removeWorkIndex;
    for (size_t i = 0; i < data_.works.size(); ++i) {
        auto& work = data_.works[i];
        work->removeOperations(operation.ID());
        if (work->OperationIDs().size() == 0) {
            removeWorkIndex.push_back(i);
        }
    }

    for(auto i = removeWorkIndex.rbegin(); i != removeWorkIndex.rend(); ++i) {
        data_.works.erase(data_.works.begin() + *i);
    }
}

void ProblemHandler::setData(ProblemData data)
{
    data_ = data;
    operations_count_ = data.operations.size();
}