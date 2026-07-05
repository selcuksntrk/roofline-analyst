cmake_minimum_required(VERSION 4.2)
project(roofline_analyst_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


add_executable(roofline_hw src/cpp/main.cpp)
