#include <scorers/scorer.h>

#include "basics/operation.h"
#include "basics/problem_data.h"
#include "basics/work.h"
#include "defines.h"

#include <iostream>

namespace {

// double calculatePenalty(double seconds, double fineCoef) {
//     const double SECONDS_IN_DAY = 86400.0;
//     const double DAYS_SCALE = 30.0; // масштаб для логарифма
//     const double BASE_PENALTY = 100.0; // базовый штраф
    
//     double days = seconds / SECONDS_IN_DAY;
//     if (days <= 0) return 0;
    
//     // Логарифмический рост: ln(1 + days/DAYS_SCALE)
//     // С каждым днём прирост замедляется
//     double penalty = BASE_PENALTY * log(1 + days / DAYS_SCALE);
    
//     return penalty * fineCoef;
// }

} // namespace

std::pair<std::vector<WorkPtr>, std::vector<WorkPtr>> SeparateWorks(
    const ProblemData& data) {
    std::vector<WorkPtr> fully_appointed;
    std::vector<WorkPtr> not_fully_appointed;
    for (WorkPtr work : data.works) {
        bool flag = true;
        for (size_t id : work->OperationIDs()) {
            if (!data.operations[id].Appointed()) {
                flag = false;
                break;
            }
        }
        if (flag) {
            fully_appointed.push_back(work);
        } else {
            not_fully_appointed.push_back(work);
        }
    }

    return {fully_appointed, not_fully_appointed};
}

double CalcNotAppointedFine(const ProblemData& data,
                            const std::vector<WorkPtr> not_fully_appointed,
                            double not_appointed_coef) {
    double answer = 0;
    for (WorkPtr work : not_fully_appointed) {
        for (size_t id : work->OperationIDs()) {
            if (!data.operations[id].Appointed()) {
                answer += work->FineCoef() * not_appointed_coef;
            }
        }
    }

    return answer;
}

Score BasicScorer::CalcScore(const ProblemData& data,
                             double not_appointed_coef) {
    Score score;
    auto [fully_appointed, not_fully_appointed] = SeparateWorks(data);
    
    // auto minTime = kStartTimePoint;
    // for (WorkPtr work : fully_appointed) {
    //     // TimePoint end_time = kStartTimePoint;
    //     for (size_t id : work->OperationIDs()) {
    //         if (minTime == kStartTimePoint) { 
    //             minTime = data.operations[id].GetStEndTimes().first;
    //         } else if (minTime > data.operations[id].GetStEndTimes().first) {
    //             minTime = data.operations[id].GetStEndTimes().first;
    //         }
    //     }

    // }

    for (WorkPtr work : fully_appointed) {
        TimePoint end_time = kStartTimePoint;
        for (size_t id : work->OperationIDs()) {
            end_time =
                std::max(data.operations[id].GetStEndTimes().second, end_time);
        }
        // std::cout << std::chrono::duration<double>(
        //         std::max(end_time - work->Directive(), Duration(0)))
        //         .count() << std::endl;
        score.appointed_fine +=
            work->FineCoef() *
            std::chrono::duration<double>(
                std::max(end_time - work->Directive(), Duration(0)))
                .count();
        // score.appointed_fine +=
        //     work->FineCoef() *
        //     std::chrono::duration<double>(
        //         std::max(end_time - minTime, Duration(0)))
        //         .count();
    }

    score.not_appointed_fine =
        CalcNotAppointedFine(data, not_fully_appointed, not_appointed_coef);
    return score;
}

Score VolumeScorer::CalcScore(const ProblemData& data,
                              double not_appointed_coef) {
    Score score;
    auto [fully_appointed, not_fully_appointed] = SeparateWorks(data);

    for (WorkPtr work : fully_appointed) {
        for (size_t id : work->OperationIDs()) {
            const Operation& operation(data.operations[id]);
            score.appointed_fine +=
                work->FineCoef() *
                std::chrono::duration<double>(
                    std::max(
                        operation.GetStEndTimes().second - work->Directive(),
                        Duration(0)))
                    .count();
        }
    }

    score.not_appointed_fine =
        CalcNotAppointedFine(data, not_fully_appointed, not_appointed_coef);

    return score;
};