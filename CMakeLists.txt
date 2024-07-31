cmake_minimum_required(VERSION 3.10)

#project(yolov8 LANGUAGES CXX CUDA)
project(auto_zoom)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb3")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)


# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV Libs: \n${OpenCV_LIBS}\n")
message(STATUS "OpenCV Libraries: \n${OpenCV_LIBRARIES}\n")
message(STATUS "OpenCV Headers: \n${OpenCV_INCLUDE_DIRS}\n")

list(APPEND INCLUDE_DIRS
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/src/AutoZoom/*.h
    )

list(APPEND ALL_LIBS
        ${OpenCV_LIBRARIES}
    )

include_directories(${INCLUDE_DIRS})
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/AutoZoom/*.cc)
message(STATUS "Source files: \n${SRCS}\n")
add_executable(${PROJECT_NAME} src/main.cc ${SRCS})
target_link_directories(${PROJECT_NAME} PUBLIC ${ALL_LIBS})
target_link_libraries(${PROJECT_NAME} ${ALL_LIBS} ${OpenCV_LIBS})