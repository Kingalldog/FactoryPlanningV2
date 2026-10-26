#pragma once
#include <memory>

#include "defines.h"

class Work {
public:
    Work(TimePoint start, TimePoint directive, double fine_coef, size_t id);

    TimePoint StartTime() const;

    TimePoint Directive() const;

    double FineCoef() const;

    size_t ID() const;

    const IdsSet& OperationIDs() const;

    bool AddOperation(size_t id);

    void removeOperations(size_t id);
TimePoint start_time_;
private:
    bool DelOperation(size_t id);
    TimePoint directive_;
    double fine_coef_;
    size_t id_;
    IdsSet operation_ids_;

    friend class Generator;
};

using WorkPtr = std::shared_ptr<Work>;