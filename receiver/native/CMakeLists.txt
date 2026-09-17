cmake_minimum_required(VERSION 3.18)

project(bt_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BT_NATIVE_VERSION "0.1.0" CACHE STRING "bt_native extension version")
set(BT_NATIVE_PYBIND11_ROOT "" CACHE PATH "Optional path to a pybind11 source checkout")

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE BT_NATIVE_PYBIND11_CMAKE_DIR
    RESULT_VARIABLE BT_NATIVE_PYBIND11_CMAKE_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(BT_NATIVE_PYBIND11_CMAKE_RESULT EQUAL 0 AND EXISTS "${BT_NATIVE_PYBIND11_CMAKE_DIR}/pybind11Config.cmake")
    list(PREPEND CMAKE_PREFIX_PATH "${BT_NATIVE_PYBIND11_CMAKE_DIR}")
endif()

find_package(pybind11 CONFIG QUIET)

set(BT_NATIVE_PYBIND11_INCLUDE_DIR "")
if(NOT TARGET pybind11::headers AND BT_NATIVE_PYBIND11_ROOT)
    set(BT_NATIVE_PYBIND11_INCLUDE_DIR "${BT_NATIVE_PYBIND11_ROOT}/include")
endif()

if(NOT TARGET pybind11::headers AND NOT EXISTS "${BT_NATIVE_PYBIND11_INCLUDE_DIR}/pybind11/pybind11.h")
    message(FATAL_ERROR "pybind11 not found. Install pybind11 or configure with -DBT_NATIVE_PYBIND11_ROOT=/path/to/pybind11.")
endif()

Python3_add_library(bt_native MODULE WITH_SOABI
    python_bindings/bt_native_module.cpp
    src/signal.cpp
    src/version.cpp
)

target_include_directories(bt_native PRIVATE include)
if(TARGET pybind11::headers)
    target_link_libraries(bt_native PRIVATE pybind11::headers)
else()
    target_include_directories(bt_native PRIVATE "${BT_NATIVE_PYBIND11_INCLUDE_DIR}")
endif()
target_compile_definitions(bt_native PRIVATE BT_NATIVE_VERSION="${BT_NATIVE_VERSION}")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(bt_native PRIVATE -O3)
endif()
