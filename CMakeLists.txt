cmake_minimum_required(VERSION 2.6)

project(chelp)

include_directories(src/)

aux_source_directory(src/ SRC_LIST)

add_library(${PROJECT_NAME} SHARED ${SRC_LIST})

