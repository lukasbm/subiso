cmake_minimum_required(VERSION 3.22)
project(subiso_glasgow LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

set(GLASGOW_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/glasgow" CACHE PATH "Glasgow subgraph solver checkout")
if (NOT EXISTS "${GLASGOW_DIR}/gss/homomorphism.hh")
    message(FATAL_ERROR
        "Glasgow sources not found at ${GLASGOW_DIR}. "
        "Initialise the vendor/glasgow submodule and build from the repository checkout "
        "(uv pip install ./packages/subiso-glasgow), or point GLASGOW_DIR at a checkout.")
endif()

find_path(GMP_INCLUDE_DIR gmp.h PATHS "$ENV{HOME}/.local/include" /usr/local/include /usr/include)
find_library(GMP_LIBRARY NAMES gmp PATHS "$ENV{HOME}/.local/libs" "$ENV{HOME}/.local/lib" /usr/local/lib /usr/lib)
if (NOT GMP_INCLUDE_DIR OR NOT GMP_LIBRARY)
    message(FATAL_ERROR "GMP not found (looked in ~/.local, /usr/local, /usr).")
endif()

file(GLOB_RECURSE GLASGOW_SOURCES "${GLASGOW_DIR}/gss/*.cc")
# Catch2-based tests are not part of the shim.
list(FILTER GLASGOW_SOURCES EXCLUDE REGEX "_test\\.cc$")

add_library(glasgow_subgraphs STATIC ${GLASGOW_SOURCES})
target_include_directories(glasgow_subgraphs PUBLIC "${GLASGOW_DIR}" "${GMP_INCLUDE_DIR}")
target_link_libraries(glasgow_subgraphs PUBLIC "${GMP_LIBRARY}" Threads::Threads)
target_compile_options(glasgow_subgraphs PRIVATE -O3)

pybind11_add_module(_glasgow src/shim.cc)
target_link_libraries(_glasgow PRIVATE glasgow_subgraphs)

install(TARGETS _glasgow LIBRARY DESTINATION subiso_glasgow)
