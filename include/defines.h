#pragma once
#include <chrono>
#include <random>
#include <set>
#include <utility>
#include <ostream>      // or #include <iostream>

namespace ch = std::chrono;

using Duration = ch::system_clock::duration;
using TimePoint = ch::system_clock::time_point;
using IdsSet = std::set<size_t>;
using IdsVec = std::vector<std::size_t>;
using StEndTimes = std::pair<TimePoint, TimePoint>;

constexpr TimePoint kStartTimePoint{ch::time_point<ch::system_clock>::min()};

inline std::ostream& operator<<(std::ostream& os, const TimePoint& tp)
{
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(
        tp.time_since_epoch()
    ).count();
    os << seconds;
    return os;
}

class RandomGenerator {
public:
    explicit RandomGenerator(int seed) : gen_(std::mt19937(seed)) {}

    int GetInt(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(gen_);
    }

    double GetDouble(double min, double max) {
        std::uniform_real_distribution<double> dist(min, max);
        return dist(gen_);
    }

    bool GetBool(double prob = 0.5) {
        std::bernoulli_distribution dist(prob);
        return dist(gen_);
    }

    std::mt19937 GetGen() { return gen_; }

private:
    std::mt19937 gen_;
};

struct GenerationParams {
    // число инструментов
    size_t cnt_tools;
    // число интервалов в расписании каждого инструмента
    size_t interval_per_tool;
    // время начала задачи планирования
    TimePoint start;
    // границы мин макс для пробелов
    std::pair<int, int> spacer_dur;
    // границы мин макс для интервалов расписания
    std::pair<int, int> interval_dur;
    // вероятность создания операции
    double operation_create_prob;
    // вероятность создания связи
    double edge_create_prob;
    // вероятность добавить исполнителя в список возможных
    double add_tool_prob;
    // сид для генератора
    int seed;
};

struct Score {
    double appointed_fine = 0;
    double not_appointed_fine = 0;

    bool operator<(Score other) {
        return std::abs(this->not_appointed_fine - other.not_appointed_fine) <
                       1e-10
                   ? this->appointed_fine < other.appointed_fine
                   : this->not_appointed_fine < other.not_appointed_fine;
    }
};
