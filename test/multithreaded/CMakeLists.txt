cmake_minimum_required(VERSION 3.14...3.22)

project(mallocMCMultithreadedTests LANGUAGES CXX)

# ---- Options ----

option(ENABLE_TEST_COVERAGE "Enable test coverage" OFF)
option(TEST_INSTALLED_VERSION "Test the version found by find_package" OFF)

# --- Import tools ----

include(../../cmake/tools.cmake)

# ---- Dependencies ----

include(../../cmake/CPM.cmake)

CPMAddPackage("gh:catchorg/Catch2@3.7.0")
CPMAddPackage("gh:TheLartians/Format.cmake@1.7.3")

CPMAddPackage(
  NAME alpaka
  GIT_TAG 1.2.0
  GITHUB_REPOSITORY alpaka-group/alpaka
  OPTIONS
  "alpaka_CXX_STANDARD 20"
)

if(TEST_INSTALLED_VERSION)
  find_package(mallocMC REQUIRED)
else()
  CPMAddPackage(NAME mallocMC SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../..)
endif()

# ---- Create binary ----

file(GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)
alpaka_add_executable(${PROJECT_NAME} ${sources})
target_link_libraries(${PROJECT_NAME} mallocMC::mallocMC alpaka::alpaka Catch2::Catch2WithMain)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20)

# enable compiler warnings
if(NOT TEST_INSTALLED_VERSION)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(mallocMC INTERFACE -Wall -Wpedantic -Wextra -Werror)
  elseif(MSVC)
    target_compile_options(mallocMC INTERFACE /W4 /WX)
  endif()
endif()

# ---- Add mallocMCTests ----

enable_testing()
add_test(${PROJECT_NAME} ${PROJECT_NAME})

# ---- code coverage ----

if(ENABLE_TEST_COVERAGE)
  target_compile_options(mallocMC INTERFACE -O0 -g -fprofile-arcs -ftest-coverage)
  target_link_options(mallocMC INTERFACE -fprofile-arcs -ftest-coverage)
endif()
