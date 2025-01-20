cmake_minimum_required(VERSION 3.14...3.22)

project(mallocMCUnitTests LANGUAGES CXX)

# ---- Options ----

option(ENABLE_TEST_COVERAGE "Enable test coverage" OFF)
option(TEST_INSTALLED_VERSION "Test the version found by find_package" OFF)

# --- Import tools ----

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/tools.cmake)

# ---- Dependencies ----

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/CPM_0.40.2.cmake)
CPMUsePackageLock(${CMAKE_CURRENT_LIST_DIR}/../../cmake/package-lock.cmake)

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/add_controlled.cmake)

add_controlled("alpaka" REQUIRED PREFIX mallocMC)
add_controlled("Catch2" REQUIRED PREFIX mallocMC)

if(NOT TARGET mallocMC)
  if(TEST_INSTALLED_VERSION)
    find_package(mallocMC REQUIRED)
  else()
    CPMAddPackage(NAME mallocMC SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../..)
  endif()
endif()

# ---- Create binary ----

file(GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)
alpaka_add_executable(${PROJECT_NAME} ${sources})
target_link_libraries(${PROJECT_NAME} mallocMC::mallocMC alpaka::alpaka Catch2::Catch2WithMain)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20)

# enable compiler warnings
if(NOT TEST_INSTALLED_VERSION)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(mallocMC INTERFACE -Wall -Wpedantic -Wextra -Werror "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:default-stream-launch>")
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
