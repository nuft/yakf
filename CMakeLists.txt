cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(kalmanfilter)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")

# Eigen library
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Google test library
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
    test/ExtendedKalmanFilterTest.cpp
    )

target_link_libraries(
    tests
    Eigen3::Eigen
    gtest
    )

# Run unit tests
add_custom_target(check ./tests -c DEPENDS tests)
