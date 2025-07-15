cmake_minimum_required(VERSION 3.12)
project(AvxToNeon C)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/../)
include_directories(${CMAKE_SOURCE_DIR}/../data)

set(SOURCE_FILES
	a2ntest.c
	main.c
)

set(ARCH_CFLAGS "-march=armv8-a+fp+simd+crc+fp16")
add_executable(test ${SOURCE_FILES})
target_compile_options(test PRIVATE ${ARCH_CFLAGS})
