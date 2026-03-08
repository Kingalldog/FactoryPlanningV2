#include "basics/tool.h"

Tool::Tool(const std::set<TimeInterval> shedule) : shedule_(shedule) {}
Tool::Tool(std::initializer_list<TimeInterval>&& shedule) : shedule_(shedule) {}

std::pair<bool, Duration> Tool::CanStartWork(const Operation& operation,
                                             TimePoint stamp,
                                             Duration span) const {
    auto it = GetStartIterator(stamp);
    if (it == shedule_.end()) {
        return {false, Duration(0)};
    }

    if (IntersectsWithWorkProc(stamp)) {
        return {false, Duration(0)};
    }

    Duration time(0);
    if (operation.Stoppable()) {
        Duration total_time(0);
        while (it != shedule_.end() && time < span) {
            time += it->GetTimeSpan(stamp);
            total_time += it->GetTimeSpan(stamp);
            if (time < span) {
                total_time += std::next(it)->Start() - it->End();
            }
            ++it;
            stamp = it->Start();
        }
        return {time >= span, total_time};
    }
    time = it->GetTimeSpan(stamp);
    return {time >= span, time};
}

void Tool::Appoint(Operation& operation, TimePoint stamp, Duration span,
                   std::vector<Operation>& all_operations) {
    Duration time(0);
    auto it = GetStartIterator(stamp);
    TimePoint start_time = stamp;
    while (time < span) {
        NamedTimeInterval interval{operation.ID(), std::max(stamp, it->Start()),
                                   std::min(it->End(), stamp + span - time)};
        work_process_.insert(interval);
        time += interval.GetTimeSpan();
        ++it;
        stamp = it->Start();
    }
    TimePoint end_time = std::prev(work_process_.end())->End();
    operation.SetTimes(start_time, end_time, time, all_operations);
}

const std::set<TimeInterval>& Tool::GetShedule() const { return shedule_; }

const std::set<NamedTimeInterval>& Tool::GetWorkProcess() const {
    return work_process_;
}

bool Tool::IntersectsWithWorkProc(TimePoint stamp) const {
    NamedTimeInterval interval(666, stamp, stamp);
    if (!work_process_.empty()) {
        const auto& last = *work_process_.rbegin();
        if (interval.Start() < last.End()) {
            return true;
        }

        if (last.End() <= interval.Start()) {
            return false;
        }
    }

    return false;
}
std::set<TimeInterval>::const_iterator Tool::GetStartIterator(
    TimePoint stamp) const {
    if (shedule_.empty()) {
        return shedule_.end();
    }

    TimeInterval interval(stamp, stamp);
    auto it = shedule_.lower_bound(interval);

    if (it == shedule_.begin() && it->Start() > stamp) {
        return shedule_.end();
    }

    if (it == shedule_.end() || it->Start() > stamp) {
        it = std::prev(it);
    }

    if (stamp >= it->Start() && stamp < it->End()) {
        return it;
    }

    return shedule_.end();
}

void Tool::RemOpFromWorkProceess(size_t op_idx) {
    auto it = work_process_.begin();
    while (it != work_process_.end()) {
        if (it->OperationID() == op_idx) {
            it = work_process_.erase(it);
        } else {
            ++it;
        }
    }
}

const std::set<TimeInterval> Tool::shedule()
{
    return shedule_;
}

void Tool::removeTimeInterval(TimeInterval timeInterval)
{
    if (shedule_.find(timeInterval) != shedule_.end()) {
        shedule_.erase(timeInterval);
    }
}

void Tool::addTimeInterval(TimeInterval timeInterval)
{
    shedule_.insert(timeInterval);
}