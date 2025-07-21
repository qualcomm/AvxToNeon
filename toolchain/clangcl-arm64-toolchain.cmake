set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR ARM64)

# clang-cl compiler
set(CMAKE_C_COMPILER "C:/Users/haozen/software/clang+llvm-20.1.7-x86_64-pc-windows-msvc/bin/clang-cl.exe")
set(CMAKE_CXX_COMPILER "C:/Users/haozen/software/clang+llvm-20.1.7-x86_64-pc-windows-msvc/bin/clang-cl.exe")
set(CMAKE_LINKER "C:/Users/haozen/software/clang+llvm-20.1.7-x86_64-pc-windows-msvc/bin/lld-link.exe")

# 指定 Windows SDK 路径（根据你安装的版本调整）
set(CMAKE_INCLUDE_PATH "C:/Program Files (x86)/Windows Kits/10/Lib/10.0.26100.0/um")
set(CMAKE_LIBRARY_PATH "C:/Program Files (x86)/Windows Kits/10/Lib/10.0.26100.0/um/arm64")

# 指定目标架构为 ARM64 Windows
set(CMAKE_C_FLAGS "--target=arm64-windows-msvc")
set(CMAKE_CXX_FLAGS "--target=arm64-windows-msvc")