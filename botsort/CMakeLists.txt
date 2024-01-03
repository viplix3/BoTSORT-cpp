CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

PROJECT(botsort VERSION 1.0 LANGUAGES CXX)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set Build Type if not set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
        "MinSizeRel" "RelWithDebInfo")
endif()

# Collect all source files
file(GLOB_RECURSE SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")

# Create library
add_library(${PROJECT_NAME} SHARED ${SOURCES})

# Set include directories
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)

# Find and link OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# Find and link Eigen3
find_package(Eigen3 REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${EIGEN3_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

# Find and link Boost
find_package(Boost REQUIRED COMPONENTS filesystem)
target_include_directories(${PROJECT_NAME} PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})

# Find and link CUDA
find_package(CUDA QUIET)

if(NOT ${CUDA_FOUND})
    message(WARNING "CUDA not found, ReID won't be built")
else()
    message(STATUS "CUDA version ${CUDA_VERSION_STRING} found")
    target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})

    target_sources(${PROJECT_NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/src/TRT_InferenceEngine/TensorRT_InferenceEngine.cpp
        ${PROJECT_SOURCE_DIR}/src/ReID.cpp)
    target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} nvonnxparser nvinfer)
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
else()
    add_definitions(-DPROFILE=1)
    add_compile_options(-O3)
endif()