#include "basics/work.h"

Work::Work(TimePoint start, TimePoint directive, double fine_coef, size_t id)
    : start_time_(start),
      directive_(directive),
      fine_coef_(fine_coef),
      id_(id) {}

TimePoint Work::StartTime() const { return start_time_; }

TimePoint Work::Directive() const { return directive_; }

double Work::FineCoef() const { return fine_coef_; }

size_t Work::ID() const { return id_; }

bool Work::AddOperation(size_t id) { return operation_ids_.insert(id).second; }
bool Work::DelOperation(size_t id) { return operation_ids_.erase(id) != 0U; }

const IdsSet& Work::OperationIDs() const { return operation_ids_; }

void Work::removeOperations(size_t id)
{
    if (operation_ids_.find(id) != operation_ids_.end()) {
      operation_ids_.erase(id);
    }
}