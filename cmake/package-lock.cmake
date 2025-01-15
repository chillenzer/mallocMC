# CPM Package Lock
# This file should be committed to version control

# PackageProject.cmake
CPMDeclarePackage(PackageProject.cmake
  VERSION 1.8.0
  GITHUB_REPOSITORY TheLartians/PackageProject.cmake
  SYSTEM YES
  EXCLUDE_FROM_ALL YES
  REQUIRED TRUE
)
# alpaka
CPMDeclarePackage(alpaka
  NAME alpaka
  GIT_TAG 1.2.0
  GITHUB_REPOSITORY alpaka-group/alpaka
  OPTIONS
    "alpaka_CXX_STANDARD 20"
    "REQUIRED"
    "TRUE"
)
# cmake-scripts
CPMDeclarePackage(cmake-scripts
  GIT_TAG 24.04
  GITHUB_REPOSITORY StableCoder/cmake-scripts
  SYSTEM YES
  EXCLUDE_FROM_ALL YES
)
# Catch2
CPMDeclarePackage(Catch2
  VERSION 3.7.0
  GITHUB_REPOSITORY catchorg/Catch2
  SYSTEM YES
  EXCLUDE_FROM_ALL YES
  REQUIRED TRUE
)
