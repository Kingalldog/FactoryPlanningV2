#include "utils/printer.h"

#include <nlohmann/json.hpp>

using Json = nlohmann::json;

std::string TimePointToStr(TimePoint tp) {
    std::time_t time = ch::system_clock::to_time_t(tp);
    std::tm tm = *std::localtime(&time);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%d.%m.%y_%H:%M:%S");
    return oss.str();
}

std::string DurationToStr(Duration dur) {
    auto days = ch::duration_cast<ch::days>(dur);
    auto hours = ch::duration_cast<ch::hours>(dur - days);
    auto minutes = ch::duration_cast<ch::minutes>(dur - days - hours);
    auto seconds = ch::duration_cast<ch::seconds>(dur - days - hours - minutes);

    std::ostringstream oss;
    oss << days.count() << '_' << std::setfill('0') << std::setw(2)
        << hours.count() << ':' << std::setw(2) << minutes.count() << ':'
        << std::setw(2) << seconds.count();

    return oss.str();
}

void Printer::PrintProblemData(const ProblemData& data, std::ostream& out) {
    out << "TOOLS\n" << data.tools.size() << std::endl;

    for (const auto& tool : data.tools) {
        out << "i";
        for (const auto& interval : tool.GetShedule()) {
            out << '(' << TimePointToStr(interval.Start()) << ' '
                << TimePointToStr(interval.End()) << ") ";
        }
        out << '\n';
    }

    out << "OPERATIONS\n" << data.operations.size() << std::endl;
    for (const auto& op : data.operations) {
        out << op.Stoppable() << ' ';
    }

    out << "\nTIMES\n";
    for (size_t i = 0; i < data.times_matrix.size(); ++i) {
        for (size_t j = 0; j < data.times_matrix[i].size(); ++j) {
            out << DurationToStr(data.times_matrix[i][j]) << ' ';
        }
        out << std::endl;
    }

    for (const auto& work : data.works) {
        size_t cnt_edges = 0;
        for (size_t op_id : work->OperationIDs()) {
            cnt_edges += data.operations[op_id].CntDeps();
        }

        out << "\nwork\n"
            << TimePointToStr(work->StartTime()) << ' '
            << TimePointToStr(work->Directive()) << ' ' << work->FineCoef()
            << ' ' << work->ID() << ' ' << cnt_edges << ' '
            << work->OperationIDs().size() << std::endl;
        for (size_t op_id : work->OperationIDs()) {
            out << op_id << ' ';
        }
        out << std::endl;
        for (size_t op_id : work->OperationIDs()) {
            for (size_t dep_id : data.operations[op_id].Depended()) {
                out << op_id << ' ' << dep_id << std::endl;
            }
        }
    }
}

void Printer::PrintGants(const ProblemData& data, std::ostream& out) {
    for (size_t idx = 0; idx < data.tools.size(); ++idx) {
        out << idx << ") \n";
        for (const auto& interval : data.tools[idx].GetWorkProcess()) {
            out << "(" << interval.OperationID() << ", "
                << TimePointToStr(interval.Start()) << ", "
                << TimePointToStr(interval.End()) << ")\n";
        }
        out << '\n';
    }
}

void Printer::PrintShedules(const ProblemData& data, std::ostream& out) {
    for (size_t idx = 0; idx < data.tools.size(); ++idx) {
        out << idx << ")\n";
        for (const auto& interval : data.tools[idx].GetShedule()) {
            out << "(" << TimePointToStr(interval.Start()) << ", "
                << TimePointToStr(interval.End()) << ")\n";
        }
        out << '\n';
    }
}

void Printer::PrintOperations(const ProblemData& data, std::ostream& out) {
    for (const auto& operation : data.operations) {
        out << operation.ID() << ") " << operation.Stoppable()
            << " start: " << TimePointToStr(operation.GetStEndTimes().first)
            << " end: " << (TimePointToStr(operation.GetStEndTimes().second))
            << " Appointed: " << operation.Appointed() << "\n";
    }
}

void Printer::PrintWorks(const ProblemData& data, std::ostream& out) {
    for (const auto& work : data.works) {
        out << "id: " << work->ID() << " start: " << work->StartTime()
            << " directive: " << work->Directive()
            << " fine: " << work->FineCoef() << " operations: ";
        for (size_t op_id : work->OperationIDs()) {
            out << op_id << " ";
        }
        out << std::endl;
    }
}

Json ToolToJson(const Tool& tool, const ProblemData& data, size_t id) {
    Json works_json = Json::array();
    Json shedule_json = Json::array();
    for (const auto& work_interval : tool.GetWorkProcess()) {
        auto op_id = work_interval.OperationID();
        auto wo_id = data.operations[op_id].PtrToWork()->ID();
        works_json.push_back({{"work", wo_id},
                              {"operation", op_id},
                              {"start", TimePointToStr(work_interval.Start())},
                              {"end", TimePointToStr(work_interval.End())}});
    }

    for (const auto& interval : tool.GetShedule()) {
        shedule_json.push_back({{"start", TimePointToStr(interval.Start())},
                                {"end", TimePointToStr(interval.End())}});
    }

    return {
        {"id", id}, {"work_process", works_json}, {"shedule", shedule_json}};
}

Json IdsSetToJson(const IdsSet& ids) {
    Json answer = Json::array();
    for (size_t id : ids) {
        answer.push_back(id);
    }
    return answer;
}

void Printer::PrintAnswerJSON(const ProblemData& data, Score score,
                              std::ostream& out) {
    Json answer;
    answer["score"].push_back(
        {{"appointed_fine", score.appointed_fine},
         {"not_appointed_fine", score.not_appointed_fine}});

    for (size_t i = 0; i < data.tools.size(); ++i) {
        answer["tools"].push_back(ToolToJson(data.tools[i], data, i));
    }

    for (const auto& oper : data.operations) {
        answer["operations"].push_back(
            {{"id", oper.ID()},
             {"stoppable", oper.Stoppable()},
             {"appointed", oper.Appointed()},
             {"depended", IdsSetToJson(oper.Depended())},
             {"start", TimePointToStr(oper.GetStEndTimes().first)},
             {"end", TimePointToStr(oper.GetStEndTimes().second)}});
    }

    for (const auto& work : data.works) {
        answer["works"].push_back(
            {{"id", work->ID()},
             {"fine_coef", work->FineCoef()},
             {"start_time", TimePointToStr(work->StartTime())},
             {"directive", TimePointToStr(work->Directive())},
             {"operations", IdsSetToJson(work->OperationIDs())}});
    }
    out << answer.dump(4);
}
