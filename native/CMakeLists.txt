cmake_minimum_required(VERSION 3.18)

project(phantom_bt_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PHANTOM_BLE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../receiver" CACHE PATH
    "PhantomChannel receiver source tree used by the native facade")
set(PHANTOM_BT_NATIVE_VERSION "phantomchannel-local-0.2.0" CACHE STRING
    "Version reported by the PhantomChannel-local native facade")

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PHANTOM_PYBIND11_CMAKE_DIR
    RESULT_VARIABLE PHANTOM_PYBIND11_CMAKE_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(PHANTOM_PYBIND11_CMAKE_RESULT EQUAL 0 AND
   EXISTS "${PHANTOM_PYBIND11_CMAKE_DIR}/pybind11Config.cmake")
    list(PREPEND CMAKE_PREFIX_PATH "${PHANTOM_PYBIND11_CMAKE_DIR}")
endif()

find_package(pybind11 CONFIG QUIET)
if(NOT TARGET pybind11::headers)
    message(FATAL_ERROR "pybind11 was not found for the selected Python interpreter")
endif()

set(PHANTOM_BLE_NATIVE_INCLUDE "${PHANTOM_BLE_ROOT}/native/include")
set(PHANTOM_BLE_SIGNAL_SOURCE "${PHANTOM_BLE_ROOT}/native/src/signal.cpp")
set(PHANTOM_BLE_VERSION_SOURCE "${PHANTOM_BLE_ROOT}/native/src/version.cpp")

foreach(required_file IN ITEMS
        "${PHANTOM_BLE_NATIVE_INCLUDE}/bt_native/signal.hpp"
        "${PHANTOM_BLE_SIGNAL_SOURCE}"
        "${PHANTOM_BLE_VERSION_SOURCE}")
    if(NOT EXISTS "${required_file}")
        message(FATAL_ERROR "Required read-only BLE source is missing: ${required_file}")
    endif()
endforeach()

Python3_add_library(bt_native MODULE WITH_SOABI
    phantom_bt_native_module.cpp
    "${PHANTOM_BLE_SIGNAL_SOURCE}"
    "${PHANTOM_BLE_VERSION_SOURCE}"
)

target_include_directories(bt_native PRIVATE "${PHANTOM_BLE_NATIVE_INCLUDE}")
target_link_libraries(bt_native PRIVATE pybind11::headers)
target_compile_definitions(
    bt_native PRIVATE
    BT_NATIVE_VERSION="${PHANTOM_BT_NATIVE_VERSION}"
)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(bt_native PRIVATE -O3)
endif()
