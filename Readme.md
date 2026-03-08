# Скачать архив с GitHub
mkdir third_party
cd third_party
wget https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz

# Распаковать архив
tar -xzf v3.11.3.tar.gz

# Проверить, что распаковалось правильно
ls -la
# Должны увидеть: json-3.11.3/  и  v3.11.3.tar.gz

# Проверить содержимое библиотеки
ls -la json-3.11.3/include/nlohmann/
# Должны увидеть: json.hpp  и папку detail/


# Sample 1: statistic_calculation
add_executable(statistic_calculation statistic_calculation.cpp)
target_link_libraries(statistic_calculation 
    PRIVATE 
        ${PROJECT_NAME}LIB
        Threads::Threads
)
# add_executable(statistic_calculation ${CMAKE_SOURCE_DIR}/samples/statistic_calculation.cpp)
# target_link_libraries(statistic_calculation ${PROJECT_NAME}LIB)
# target_include_directories(statistic_calculation PRIVATE ${CMAKE_SOURCE_DIR}/include)