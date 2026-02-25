#include <iostream>
#include "super_simple_recomender.h"

int main() {
    // Создаем объект
    SimpleRecommender recommender("../model/heuristic_model.pkl");
    
    // Получаем топ-3 алгоритма для задачи
    auto algorithms = recommender.getTopAlgoritmth("../../dataset_directory_big/tmp/465677_20_20_0.300000_0.450000_0.300000_1000_500_.txt");
    
    // Выводим результаты
    std::cout << "Топ-3 алгоритма:" << std::endl;
    for(auto& algorithm : algorithms) {
        std::cout << algorithm << " ";
    }
    
    return 0;
}