#pragma once
#include <stack>

#include "basics/work.h"
#include "defines.h"

class Operation {
public:
    Operation(size_t id, bool stoppable);
    // добавляет id операции в список зависимостей (other -> this)
    bool AddDependency(size_t id);
    // добавляет id операции в список зависимых (this -> other)
    bool AddDepended(size_t id);
    // добавляет номер инструмента на котором можно будет выполнить операцию
    bool AddPossibleTool(size_t id);

    bool HasDependencies();

    bool Appointed() const;

    bool SetTimes(TimePoint start, TimePoint end, Duration span,
                  std::vector<Operation>& all_operations);

    // cделает операцию снова не назначенной и добавит её к потомкам
    void UnAppoint(std::vector<Operation>& all_operations) {
        start_time_ = kStartTimePoint;
        end_time_ = kStartTimePoint;

        // мы назначили время выполнения операции, значит надо уменьшить число
        // зависимостей для всех зависимых
        for (size_t dep_id : depended_) {
            all_operations[dep_id].AddDependency(id_);
            all_operations[dep_id].possible_starts_.pop();
        }
    }

    void SetWorkPtr(WorkPtr work);

    StEndTimes GetStEndTimes() const;

    bool CanBeAppointed(TimePoint stamp) const;

    bool DelDependency(size_t dep_id, TimePoint dep_end);

    bool Stoppable() const;
    const IdsSet& PossibleTools()
        const;  // вернет список инструментов, которые могут выполнить операцию
    const IdsSet& Dependencies() const;  // вернет список зависимостей
    const IdsSet& Depended() const;  // вернёт список зависимых
    size_t ID() const;
    size_t CntDeps() const;
    WorkPtr PtrToWork() const;

    TimePoint PossibleStart() const { return possible_starts_.top(); }

private:
    WorkPtr ptr_to_work_ = nullptr;
    size_t id_;       // номер операции
    bool stoppable_;  // прерываемость
    size_t cnt_deps_ = 0;  // число неназначенных родителей
    IdsSet depended_;      // множество зависимых от неё
    IdsSet dependencies_;  // множество зависимостей
    IdsSet possible_tools_;  // множество возможных исполнителей
    TimePoint start_time_ = kStartTimePoint;
    TimePoint end_time_ = kStartTimePoint;
    std::stack<TimePoint> possible_starts_;
    // TimePoint possible_start_ =
    //     START_TIME_POINT;  // момент возможного старта операции

    // friend class Operation;
};
