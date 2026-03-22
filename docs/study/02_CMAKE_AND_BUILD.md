# 02 - CMake Va Build Trong Project Nay

## CMake la gi?
CMake khong compile truc tiep. CMake generate build system (Ninja/Make), sau do build system moi goi compiler.

Flow:
- CMakeLists.txt -> generate
- Ninja/Make -> compile + link

## Nhung khoi lenh quan trong
1. Khai bao project
- cmake_minimum_required(...)
- project(... LANGUAGES C CXX)

2. Chuan ngon ngu
- set(CMAKE_CXX_STANDARD 23)
- set(CMAKE_CXX_STANDARD_REQUIRED ON)

3. Khai bao libs/executables
- add_library(...)
- add_executable(...)

4. Dependency
- target_include_directories(...)
- target_link_libraries(...)
- find_package(OpenCV REQUIRED)
- find_library(ONNXRUNTIME_LIB ...)

## CMake cua repo nay (tom tat)
- Libs:
  - ocrplate_utils
  - ocrplate_tracking
  - ocrplate_services
  - ocrplate_pipeline
- Executables:
  - main
  - benchmark

## Dependency target-level
- ocrplate_pipeline link:
  - ocrplate_services
  - ocrplate_utils
  - ocrplate_tracking
- main/benchmark link:
  - ocrplate_pipeline
  - onnxruntime
  - OpenCV
  - pthread, dl

## CMakePresets
CMakePresets.json giup build dong nhat theo profile (debug/release), giam sai khac moi truong.

## Script build/run
- build.sh: parse option, configure, build
- run.sh: chay main/benchmark va pass args
