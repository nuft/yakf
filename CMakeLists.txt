
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(kalman-filter)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

include_directories(
    ./
    tests/
    )

add_executable(
    tests
    test/main.cpp
    test/NumericalIntegrationTest.cpp
    )

target_link_libraries(
    tests
    gtest
    )

# Run unit tests
add_custom_target(check ./tests -c DEPENDS tests)
