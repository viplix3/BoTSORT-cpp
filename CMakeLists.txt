CMAKE_MINIMUM_REQUIRED(VERSION 3.20)

# Set project name
project(botsort_tracker_example VERSION 0.1.0 LANGUAGES CXX)

# Build options
option(BUILD_EXAMPLES "Build examples" ON)

# Set Build Type if not set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
        "MinSizeRel" "RelWithDebInfo")
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Set install directories
set(CMAKE_INSTALL_BINDIR bin)
set(CMAKE_INSTALL_LIBDIR lib)
set(CMAKE_INSTALL_INCLUDEDIR include)

# Check OpenCV and Eigen3
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# add botsort
add_subdirectory(botsort)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()