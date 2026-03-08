#pragma once

#include "basics/operation.h"
#include "basics/time_interval.h"

class Tool {
public:
    Tool(const std::set<TimeInterval> shedule);
    Tool(std::initializer_list<TimeInterval>&& shedule);
    std::pair<bool, Duration> CanStartWork(const Operation& operation,
                                           TimePoint stamp,
                                           Duration span) const;
    void Appoint(Operation& operation, TimePoint stamp, Duration span,
                 std::vector<Operation>& all_operations);
    // метод отката изменений в расписании (удаление операции)
    void RemOpFromWorkProceess(size_t op_idx);
    const std::set<TimeInterval>& GetShedule() const;
    const std::set<NamedTimeInterval>& GetWorkProcess() const;
    const std::set<TimeInterval> shedule();
    void removeTimeInterval(TimeInterval timeInterval);
    void addTimeInterval(TimeInterval timeInterval);
private:
    std::set<TimeInterval> shedule_;
    std::set<NamedTimeInterval> work_process_;
    std::set<TimeInterval>::const_iterator GetStartIterator(
        TimePoint stamp) const;
    bool IntersectsWithWorkProc(TimePoint timestamp) const;
};
