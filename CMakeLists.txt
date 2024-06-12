cmake_minimum_required(VERSION 3.15)

set(PROJECT_NAME TinyOCL)

project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(${PROJECT_SOURCE_DIR}/src)

option(ENABLE_EXAMPLE "Enable examples of TinyOCL." OFF)

# if (ENABLE_EXAMPLE)
add_subdirectory(${PROJECT_SOURCE_DIR}/examples)
# endif()