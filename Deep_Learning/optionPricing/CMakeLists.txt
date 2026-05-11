cmake_minimum_required(VERSION 3.10)
project(OptionPricingDL)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 QUIET)

if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 non trouvé via find_package, tentative via chemin manuel...")
    
    if(APPLE)
        include_directories("/opt/homebrew/include/eigen3")
    elseif(WIN32)
        include_directories("C:/Users/hugog/Documents/cpp_tools/eigen-5.0.1/eigen-5.0.1")
    endif()
else()
    message(STATUS "Eigen3 trouvé !")
endif()

include_directories(modules/include)

set(SOURCES
    main.cpp
    modules/activationFunction.cpp
    modules/perceptron.cpp
    modules/mlp.cpp
)

add_executable(optionPricingDl ${SOURCES})

if(TARGET Eigen3::Eigen)
    target_link_libraries(optionPricingDl PRIVATE Eigen3::Eigen)
endif()