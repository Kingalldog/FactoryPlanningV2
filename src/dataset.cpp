#include "utils/dataset.h"

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>

#include "basics/problem_data.h"
#include "defines.h"
#include "utils/generator.h"
#include "utils/printer.h"
constexpr std::array<int, 6> kSeeds{5465634, 465677, 465677,
                                    5613244, 68123,  2349008};
constexpr std::array<size_t, 3> kToolsCnts{70};
constexpr std::array<size_t, 3> kIntervalsCnts{20, 50, 70};
constexpr std::array<double, 3> kOpCreateProbs{0.45, 0.69};
constexpr std::array<double, 3> kEdCreateProbs{0.45, 0.69};
constexpr std::array<double, 3> kToolAddiProbs{0.45, 0.69};

constexpr std::array<std::pair<int, int>, 2> kIntervalLens{
    std::pair<int, int>{1000, 2000}, std::pair<int, int>{200, 600}};
constexpr std::array<std::pair<int, int>, 2> kSpacerLens{
    std::pair<int, int>{100, 200}, std::pair<int, int>{500, 600}};

int Dataset::CreateDataset(const std::string& path_name, size_t size) {
    try {
        std::filesystem::create_directories(path_name);
    } catch (const std::filesystem::filesystem_error& ex) {
        return -1;
    }

    Generator gena;
    RandomGenerator rand_gena(kSeeds[0]);

    for (size_t i = 0; i < size; ++i) {
        GenerationParams params{
            kToolsCnts[rand_gena.GetInt(0, kToolsCnts.size() - 1)],
            kIntervalsCnts[rand_gena.GetInt(0, kIntervalsCnts.size() - 1)],
            kStartTimePoint,
            kSpacerLens[rand_gena.GetInt(0, kSpacerLens.size() - 1)],
            kIntervalLens[rand_gena.GetInt(0, kIntervalLens.size() - 1)],
            kOpCreateProbs[rand_gena.GetInt(0, kOpCreateProbs.size() - 1)],
            kEdCreateProbs[rand_gena.GetInt(0, kEdCreateProbs.size() - 1)],
            kToolAddiProbs[rand_gena.GetInt(0, kToolAddiProbs.size() - 1)],
            kSeeds[rand_gena.GetInt(0, kSeeds.size() - 1)]};
        ProblemData data(gena.Generate(params));
        std::string filename = std::to_string(params.seed) + "_" +
                               std::to_string(params.cnt_tools) + "_" +
                               std::to_string(params.interval_per_tool) + "_" +
                               std::to_string(params.operation_create_prob) +
                               "_" + std::to_string(params.edge_create_prob) +
                               "_" + std::to_string(params.add_tool_prob) +
                               "_" + std::to_string(params.interval_dur.first) +
                               "_" + std::to_string(params.spacer_dur.first) +
                               "_" + std::to_string(i) + ".txt";
        std::filesystem::path file_path = path_name + "/" + filename;
        std::ofstream out_file(file_path);
        Printer::PrintProblemData(data, out_file);
    }
    return 0;
}

int Dataset::CreateBigDataset(const std::string& path_name) {
    try {
        std::filesystem::create_directories(path_name);
    } catch (const std::filesystem::filesystem_error& ex) {
        return -1;
    }
    Generator gena;

    for (size_t tool_cnt : kToolsCnts) {
        for (size_t int_cnt : kIntervalsCnts) {
            std::string local_path = path_name + "/" +
                                     std::to_string(tool_cnt) + "_" +
                                     std::to_string(int_cnt);
            try {
                std::filesystem::create_directories(local_path);
            } catch (const std::filesystem::filesystem_error& ex) {
                return -1;
            }

            std::cout << "New path filling: " << "tools - " << tool_cnt
                      << " intervals - " << int_cnt << std::endl;

            for (auto op_prob : kOpCreateProbs) {
                for (auto tool_prob : kToolAddiProbs) {
                    for (auto edge_prob : kEdCreateProbs) {
                        for (auto int_len : kIntervalLens) {
                            for (auto space_len : kSpacerLens) {
                                for (auto seed : kSeeds) {
                                    GenerationParams params{
                                        tool_cnt,
                                        int_cnt,
                                        std::chrono::system_clock::now(),
                                        space_len,
                                        int_len,
                                        op_prob,
                                        edge_prob,
                                        tool_prob,
                                        seed};

                                    ProblemData data(gena.Generate(params));
                                    std::string filename =
                                        std::to_string(params.seed) + "_" +
                                        std::to_string(params.cnt_tools) + "_" +
                                        std::to_string(
                                            params.interval_per_tool) +
                                        "_" +
                                        std::to_string(
                                            params.operation_create_prob) +
                                        "_" +
                                        std::to_string(
                                            params.edge_create_prob) +
                                        "_" +
                                        std::to_string(params.add_tool_prob) +
                                        "_" +
                                        std::to_string(
                                            params.interval_dur.first) +
                                        "_" +
                                        std::to_string(
                                            params.spacer_dur.first) +
                                        "_" + ".txt";
                                    std::string file_path =
                                        local_path + "/" + filename;
                                    std::ofstream out_file(file_path);
                                    Printer::PrintProblemData(data, out_file);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}

Dataset::Dataset(const std::string& folder_path,
                 std::vector<std::string>& file_names) {
    namespace fs = std::filesystem;

    try {
        for (const auto& entry : fs::directory_iterator(folder_path)) {
            if (entry.is_regular_file()) {
                file_paths_.push_back(entry.path().string());
            }
        }
        std::sort(file_paths_.begin(), file_paths_.end());
        std::cout << "Прочитано " << file_paths_.size() << " задач"
                  << std::endl;
        file_names = file_paths_;
    } catch (const fs::filesystem_error& ex) {
        std::cerr << "Ошибка чтения папки: " << ex.what() << std::endl;
    }
}

std::optional<std::string> Dataset::GetNext(ProblemData* data) {
    size_t index = current_index_.fetch_add(1, std::memory_order_relaxed);

    if (index >= file_paths_.size()) {
        return std::nullopt;
    }

    std::ifstream file(file_paths_[index]);
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << file_paths_[index] << std::endl;
        return std::nullopt;
    }
    // std::cout << std::this_thread::get_id() << " " << index << std::endl;
    *data = ProblemData(file);
    return file_paths_[index];
}