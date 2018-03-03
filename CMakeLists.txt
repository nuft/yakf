
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(kalman-filter)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")

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
