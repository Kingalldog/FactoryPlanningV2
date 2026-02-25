cmake_minimum_required(VERSION 3.9.0)
project(FPToolsV2 VERSION 0.1.0)

# Найти библиотеку Threads (pthread)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wsign-compare")

set(CMAKE_CXX_FLAGS_ASAN "-g -fsanitize=address,undefined -fno-sanitize-recover=all -O2"
    CACHE STRING "Compiler flags in asan build"
    FORCE)

set(CMAKE_CXX_FLAGS_TSAN "-g -fsanitize=thread -fno-sanitize-recover=all"
    CACHE STRING "Compiler flags in tsan build"
    FORCE)

set(CMAKE_CXX_FLAGS_REL "-O2 -DNDEBUG" 
    CACHE STRING "Compiler flags for the release version" 
    FORCE)

message("General compiler flags: " ${CMAKE_CXX_FLAGS})
message("Debug build compiler flags: " ${CMAKE_CXX_FLAGS_DEBUG})
message("ASAN build compiler flags: " ${CMAKE_CXX_FLAGS_ASAN})
message("TSAN build compiler flags: " ${CMAKE_CXX_FLAGS_TSAN})
message("Release build compiler flags: " ${CMAKE_CXX_FLAGS_REL})

# ------------------------------------------------------------------
# Ручное подключение nlohmann/json
# ------------------------------------------------------------------

set(NLOHMANN_JSON_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/json-3.11.3")

if(NOT EXISTS ${NLOHMANN_JSON_PATH}/include/nlohmann/json.hpp)
    message(FATAL_ERROR 
        "\n❌ nlohmann/json not found at ${NLOHMANN_JSON_PATH}/include/nlohmann/json.hpp!\n"
        "Please download it manually:\n"
        "  cd third_party\n"
        "  wget https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz\n"
        "  tar -xzf v3.11.3.tar.gz\n"
    )
endif()

message("✅ Found nlohmann_json at: ${NLOHMANN_JSON_PATH}")

add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE
    ${NLOHMANN_JSON_PATH}/include
)

add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)

# ------------------------------------------------------------------
# Сборка основной библиотеки
# ------------------------------------------------------------------

file(GLOB_RECURSE LIBRARY_SRC "src/*.cpp")

add_library(${PROJECT_NAME}LIB ${LIBRARY_SRC})

target_include_directories(${PROJECT_NAME}LIB 
    PUBLIC include
)

target_link_libraries(${PROJECT_NAME}LIB 
    PRIVATE 
        nlohmann_json::nlohmann_json
        Threads::Threads  # Добавлено для поддержки потоков
)

add_subdirectory(tests)
add_subdirectory(samples)
