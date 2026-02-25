add_executable(${PROJECT_NAME}.sample ${CMAKE_SOURCE_DIR}/samples/sample.cpp)
target_link_libraries(${PROJECT_NAME}.sample ${PROJECT_NAME}LIB)
target_include_directories(${PROJECT_NAME}.sample PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(dataset_creation ${CMAKE_SOURCE_DIR}/samples/dataset_creation.cpp)
target_link_libraries(dataset_creation ${PROJECT_NAME}LIB)
target_include_directories(dataset_creation PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(concurent_grid ${CMAKE_SOURCE_DIR}/samples/concurent_grid.cpp)
target_link_libraries(concurent_grid ${PROJECT_NAME}LIB)
target_include_directories(concurent_grid PRIVATE ${CMAKE_SOURCE_DIR}/include)

find_package(Threads REQUIRED)

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

add_executable(multistep ${CMAKE_SOURCE_DIR}/samples/multistep.cpp)
target_link_libraries(multistep ${PROJECT_NAME}LIB)
target_include_directories(multistep PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(tree_of_changing ${CMAKE_SOURCE_DIR}/samples/change_tree.cpp)

