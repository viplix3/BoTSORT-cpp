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
if(MSVC)
    add_library(${PROJECT_NAME} STATIC ${SOURCES})
else()
    add_library(${PROJECT_NAME} SHARED ${SOURCES})
endif()

if(MSVC)
    target_link_options(${PROJECT_NAME} PRIVATE "/NODEFAULTLIB:LIBCMT")
    #target_link_options(${PROJECT_NAME} PRIVATE "/NODEFAULTLIB:LIBCMTD")
    #target_link_options(${PROJECT_NAME} PRIVATE "/NODEFAULTLIB:MSVCRTD")
    target_link_libraries(${PROJECT_NAME} ws2_32)
endif()


# Set include directories
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Find and link OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# Find and link Eigen3
find_package(Eigen3 REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${EIGEN3_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

# Find and link CUDA
find_package(CUDA QUIET)

if(NOT ${CUDA_FOUND})
    message(WARNING "CUDA not found, ReID will be built with opencv_dnn")
    target_sources(${PROJECT_NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/src/ReID.cpp)

    remove_definitions(-DUSE_TRT_REID)
else()
    message(STATUS "CUDA version ${CUDA_VERSION_STRING} found")
    target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})

    find_library(TRT_NVINFER NAMES nvinfer PATHS ${TENSORRT_DIR} PATH_SUFFIXES lib lib64)
    find_library(TRT_NVONNXPARSER NAMES nvonnxparser PATHS ${TENSORRT_DIR} PATH_SUFFIXES lib lib64)
    find_path(TRT_INCLUDE_DIR NAMES NvInfer.h PATHS ${TENSORRT_DIR} PATH_SUFFIXES include)

    if (NOT TRT_NVINFER OR NOT TRT_NVONNXPARSER OR NOT TRT_INCLUDE_DIR)
        message(FATAL_ERROR "Not all TensorRT was found. Check TENSORRT_DIR.")
    else()
        message("TRT_INCLUDE_DIR: ${TRT_INCLUDE_DIR}, TRT_NVINFER: ${TRT_NVINFER}, TRT_NVONNXPARSER: ${TRT_NVONNXPARSER}")
    endif()

    target_include_directories(${PROJECT_NAME} PRIVATE ${TRT_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} ${TRT_NVINFER})
    target_link_libraries(${PROJECT_NAME} ${TRT_NVONNXPARSER})

    target_sources(${PROJECT_NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/src/TRT_InferenceEngine/TensorRT_InferenceEngine.cpp
        ${PROJECT_SOURCE_DIR}/src/ReID.cpp)
    target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})

    add_definitions(-DUSE_TRT_REID)
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
