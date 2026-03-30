#pragma once

#include <cstddef>
#include <iterator>
#include <map>

template <typename T>
class CountSet {
private:
    std::map<T, size_t> data_;

public:
    class Iterator {
    private:
        using MapIterator = typename std::map<T, size_t>::const_iterator;
        MapIterator it_;
        size_t current_pos_;

    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = const T*;
        using reference = const T&;

        Iterator(MapIterator map_it, size_t pos = 0)
            : it_(map_it), current_pos_(pos) {}

        reference operator*() const { return it_->first; }
        pointer operator->() const { return &it_->first; }

        Iterator& operator++() {
            if (it_ != MapIterator()) {
                ++current_pos_;
                if (current_pos_ >= it_->second) {
                    ++it_;
                    current_pos_ = 0;
                }
            }
            return *this;
        }

        Iterator operator++(int) {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const Iterator& other) const {
            return it_ == other.it_ && current_pos_ == other.current_pos_;
        }

        bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }
    };

    class UniqueIterator {
    private:
        using MapIterator = typename std::map<T, size_t>::const_iterator;
        MapIterator it_;

    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = const T*;
        using reference = const T&;

        UniqueIterator(MapIterator map_it) : it_(map_it) {}

        reference operator*() const { return it_->first; }
        pointer operator->() const { return &it_->first; }

        UniqueIterator& operator++() {
            ++it_;
            return *this;
        }

        UniqueIterator operator++(int) {
            UniqueIterator tmp = *this;
            ++it_;
            return tmp;
        }

        bool operator==(const UniqueIterator& other) const {
            return it_ == other.it_;
        }

        bool operator!=(const UniqueIterator& other) const {
            return it_ != other.it_;
        }
    };

    void Insert(const T& value) {
        if (!data_.contains(value)) {
            data_[value] = 1;
        } else {
            ++data_[value];
        }
    }
    void Insert(const T& value, size_t count) { data_[value] += count; }

    bool EraseOne(const T& value) {
        auto it = data_.find(value);
        if (it != data_.end() && it->second > 0) {
            // it->second--;
            // if (it->second == 0) {
            //     data_.erase(it);
            // }
            data_.erase(it);
            return true;
        }
        return false;
    }

    size_t Count(const T& value) const {
        auto it = data_.find(value);
        return it != data_.end() ? it->second : 0;
    }

    size_t size() {
        return data_.size();
    }

    Iterator begin() const { return Iterator(data_.begin()); }
    Iterator end() const { return Iterator(data_.end()); }

    UniqueIterator UniqueBegin() const { return UniqueIterator(data_.begin()); }
    UniqueIterator UniqueEnd() const { return UniqueIterator(data_.end()); }

    void Clear() { data_.clear(); }
};