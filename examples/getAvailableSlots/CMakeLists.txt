cmake_minimum_required(VERSION 3.14...3.22)

project(mallocMCExampleGetAvailableSlots LANGUAGES CXX)

# --- Import tools ----

include(../../cmake/tools.cmake)

# ---- Dependencies ----

include(../../cmake/CPM.cmake)

CPMAddPackage(
  NAME alpaka
  GIT_TAG 1.2.0
  GITHUB_REPOSITORY alpaka-group/alpaka
  OPTIONS
  "alpaka_CXX_STANDARD 20"
)

CPMAddPackage(NAME mallocMC SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../..)

# ---- Create standalone executable ----

alpaka_add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/source/main.cpp)

set_target_properties(${PROJECT_NAME}
    PROPERTIES
        CXX_STANDARD 20
        OUTPUT_NAME ${PROJECT_NAME}
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)

target_link_libraries(${PROJECT_NAME} mallocMC::mallocMC alpaka::alpaka)
