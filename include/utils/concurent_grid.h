#pragma once

#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

using LabelsMap = std::unordered_map<std::string, size_t>;

template <typename T>
class ConcurentGrid {
private:
    std::vector<std::vector<T>> data_;
    LabelsMap row_labels_;
    LabelsMap col_labels_;
    mutable std::shared_mutex mutex_;

public:
    using value_type = T;
    ConcurentGrid(size_t rows, size_t cols, const T& default_value = T{})
        : data_(rows, std::vector<T>(cols, default_value)) {}

    ConcurentGrid(size_t rows, size_t cols,
                  const std::vector<std::string>& row_names,
                  const std::vector<std::string>& col_names,
                  const T& default_value = T{})
        : data_(rows, std::vector<T>(cols, default_value)) {
        SetRowLabels(row_names);
        SetColumnLabels(col_names);
    }

    void SetRowLabels(const std::vector<std::string>& labels) {
        std::unique_lock lock(mutex_);
        row_labels_.clear();
        for (size_t i = 0; i < labels.size() && i < data_.size(); ++i) {
            row_labels_[labels[i]] = i;
        }
    }

    void SetColumnLabels(const std::vector<std::string>& labels) {
        std::unique_lock lock(mutex_);
        col_labels_.clear();
        for (size_t i = 0;
             i < labels.size() && i < (data_.empty() ? 0 : data_[0].size());
             ++i) {
            col_labels_[labels[i]] = i;
        }
    }

    void Set(size_t row, size_t col, const T& value) {
        std::unique_lock lock(mutex_);
        data_[row][col] = value;
    }

    bool Set(const std::string& row_label, const std::string& col_label,
             const T& value) {
        std::unique_lock lock(mutex_);
        auto row_it = row_labels_.find(row_label);
        auto col_it = col_labels_.find(col_label);

        if (row_it == row_labels_.end()) {
            row_labels_[row_label] = data_.size();
            data_.push_back(std::vector<T>(col_labels_.size()));
        }

        row_it = row_labels_.find(row_label);
        if (row_it != row_labels_.end() && col_it != col_labels_.end()) {
            size_t row = row_it->second;
            size_t col = col_it->second;
            data_[row][col] = value;
            return true;
        }
        
        return false;
    }

    std::optional<T> Get(size_t row, size_t col) const {
        std::shared_lock lock(mutex_);
        if (row < data_.size() && col < data_[row].size()) {
            return data_[row][col];
        }
        return std::nullopt;
    }

    std::optional<T> Get(const std::string& row_label,
                         const std::string& col_label) const {
        std::shared_lock lock(mutex_);
        auto row_it = row_labels_.find(row_label);
        auto col_it = col_labels_.find(col_label);

        if (row_it != row_labels_.end() && col_it != col_labels_.end()) {
            size_t row = row_it->second;
            size_t col = col_it->second;
            if (row < data_.size() && col < data_[row].size()) {
                return data_[row][col];
            }
        }
        return std::nullopt;
    }

    std::pair<size_t, size_t> Size() const {
        std::shared_lock lock(mutex_);
        if (data_.empty()) {
            return {0, 0};
        }
        return {data_.size(), data_[0].size()};
    }

    std::vector<std::string> GetRowLabels() const {
        std::shared_lock lock(mutex_);
        std::vector<std::string> labels(row_labels_.size());
        for (const auto& [label, index] : row_labels_) {
            if (index < labels.size()) {
                labels[index] = label;
            }
        }
        return labels;
    }

    std::vector<std::string> GetColumnLabels() const {
        std::shared_lock lock(mutex_);
        std::vector<std::string> labels(col_labels_.size());
        for (const auto& [label, index] : col_labels_) {
            if (index < labels.size()) {
                labels[index] = label;
            }
        }
        return labels;
    }
};