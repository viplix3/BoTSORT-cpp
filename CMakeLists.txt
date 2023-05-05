CMAKE_MINIMUM_REQUIRED(VERSION 3.20)

# CMP0076: The target_sources() command converts relative paths to absolute.
cmake_policy(SET CMP0076 NEW)

# CMP0079: target_link_libraries() allows use with targets in other directories.
cmake_policy(SET CMP0079 NEW)

# CMP0116: Ninja generators transform DEPFILE s from add_custom_command().
cmake_policy(SET CMP0116 NEW)

# Double colon in target name means ALIAS or IMPORTED target.
cmake_policy(SET CMP0028 NEW)

# Support new IN_LIST if() operator
cmake_policy(SET CMP0057 NEW)

PROJECT(multi_object_tracking VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB_RECURSE SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")

add_executable(${PROJECT_NAME} ${SOURCES})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
