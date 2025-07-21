cmake_minimum_required(VERSION 3.12)
project(AvxToNeon C)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/../)
include_directories(${CMAKE_SOURCE_DIR}/../data)

set(SOURCE_FILES
	a2ntest.c
	main.c
)

message(STATUS "CMAKE_C_COMPILER_ID: ${CMAKE_C_COMPILER_ID}")
if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
	message("[MY MSG] Using C Compiler MSVC")
else()
	message("[MY MSG] Using C Compiler ${CMAKE_C_COMPILER_ID}")
	set(ARCH_CFLAGS "-march=armv8-a+fp+simd+crc+fp16")
endif()

add_executable(test ${SOURCE_FILES})
target_compile_options(test PRIVATE ${ARCH_CFLAGS})
