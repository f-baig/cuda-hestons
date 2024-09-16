# CMake
cmake_minimum_required(VERSION 3.10)

project(MonteCarloOptionPricing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

include_directories(include)

find_package(CUDA REQUIRED)

add_executable(MonteCarloOptionPricing
    main.cpp
    src/MonteCarloSimulator.cpp
    src/MonteCarloSimulatorCUDA.cu
)
