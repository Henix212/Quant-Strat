cmake_minimum_required(VERSION 3.10)
project(OptionPricingDL)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories("C:/Users/hugog/Documents/cpp_tools/eigen-5.0.1/eigen-5.0.1")

include_directories(modules/include)

set(SOURCES
    main.cpp
    modules/activationFunction.cpp
    modules/perceptron.cpp
    modules/mlp.cpp
)

add_executable(optionPricingDl ${SOURCES})