#----------------------------------------------------------------------------------#
#    Copyright (c) 2010-2018 Pauli Parkkinen, Eelis Solala, Wen-Hua Xu,            #
#                            Sergio Losilla, Elias Toivanen, Jonas Juselius        #
#                                                                                  #
#    Permission is hereby granted, free of charge, to any person obtaining a copy  #
#    of this software and associated documentation files (the "Software"), to deal #
#    in the Software without restriction, including without limitation the rights  #
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell     #
#    copies of the Software, and to permit persons to whom the Software is         #
#    furnished to do so, subject to the following conditions:                      #
#                                                                                  #
#    The above copyright notice and this permission notice shall be included in all#
#    copies or substantial portions of the Software.                               #
#                                                                                  #
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    #
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      #
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE   #
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        #
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, #
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE #
#    SOFTWARE.                                                                     #
#----------------------------------------------------------------------------------# 
# Master CMakeLists.txt file for libbubbles

set(F2PY_LIBS   "")
set(F2PY_INCLUDE_DIRS   "")

cmake_minimum_required(VERSION 2.6)
project(BUBBLES) # Enables C/C++ support
enable_language(Fortran)
find_package(PythonInterp 2.7 REQUIRED)

# Define bubbles specific CMake modules in the folder `cmake'
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/cmake
)


# Build/install in Debug configuration if not specified otherwise
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE
        Debug
        CACHE STRING
        "Choose the type of build. Options are: Debug, Release."
        FORCE
        )
endif()

# Check that the build type is sane
string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_tolower)
if(NOT cmake_build_type_tolower STREQUAL "debug"
        AND NOT cmake_build_type_tolower STREQUAL "release")
    #        AND NOT cmake_build_type_tolower STREQUAL "relwithdebinfo")
    message(FATAL_ERROR "Unknown build type \"${CMAKE_BUILD_TYPE}\".
    Allowed values are Debug, Release (case-insensitive).")
endif()

# Build options (May or may not take effect depending on
# compiler/machine you are running)
option(ENABLE_DFT            "Enable DFT (LibXC Required)"      OFF)
option(ENABLE_PROFILING      "Profile execution"                OFF)
option(ENABLE_BOUNDS_CHECK   "Runtime bounds checking"          OFF)
option(ENABLE_CODE_COVERAGE  "Profile code coverage"            OFF)
option(ENABLE_MPI            "MPI parallelization"              OFF)
option(ENABLE_OMP            "OpenMP parallelization"           OFF)
option(ENABLE_CUDA           "CUDA GPU acceleration"            OFF)
option(ENABLE_CUDA_PROFILING "Enable CUDA profiling"            OFF)
option(ENABLE_NVTX           "Enable NVTX"                      OFF)
option(DEBUG_CUDA            "Debug CUDA"                       OFF)
option(DEBUG_MEMORY_LEAKS    "Debug Memory Leaks"               OFF)
option(ENABLE_64BIT_INTEGERS "Control the default integer size" OFF)

set(LIBS)
include(Revision) # The version number of current build (Relies on git)
include(Archs) # Control of architecture dependent settings (Currently unused)
include(Compilers) # Default compiler flags
include(Linalg) # Finds BLAS and LAPACK

# Uncomment this to not rebuild if rules (compiler flags) change
# this has advantages and disadvantages
# set(CMAKE_SKIP_RULE_DEPENDENCY TRUE)

set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS}")


if (ENABLE_DFT)
    find_library(LIBXC xc)
    find_library(LIBXCF90 xcf90)
    if(LIBXC AND LIBXCF90)
        set(LIBS "${LIBS} -lxc -lxcf90")   
        # message("libs ${LIBS}")
        set(F2PY_LIBS "${F2PY_LIBS} -lxc -lxcf90")
        set(F2PY_INCLUDE_DIRS ${F2PY_INCLUDE_DIRS} ${LIBXC} ${LIBXCF90})
        # message("f2py inc dir ${F2PY_INCLUDE_DIRS}")
        add_definitions(-DHAVE_DFT)
    else()
        message("LibXC was not found")
    endif()
endif()

# Configure compiler/linker according to enabled features
if(ENABLE_MPI)
    find_package(MPI)
    if(MPI_FOUND)
        add_definitions(-DHAVE_MPI)
        set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${MPI_COMPILE_FLAGS}")
        set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
        set(CMAKE_Fortran_COMPILER ${MPI_Fortran_COMPILER})
    
        message("-- CMake fortran flags \"${MPI_COMPILE_FLAGS}\"")
        message("-- MPI CXX compiler \"${MPI_CXX_COMPILER}\"")
        message("-- MPI Fortran compiler \"${MPI_Fortran_COMPILER}\"")
        message("-- MPI Include path \"${MPI_INCLUDE_PATH}\"")
        message("\"${CMAKE_INSTALL_PREFIX}\"/include")
    endif()
endif()

if(ENABLE_OMP)
    message("!!! Open MP enabled")
    add_definitions(-DHAVE_OMP)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fopenmp")
    set(F2PY_LIBS "${F2PY_LIBS} -lgomp")
    #set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${OpenMP_COMPILE_FLAGS}")
    #include_directories(${OpenMP_INCLUDE_PATH})
    
endif()

if (ENABLE_CUDA)
    find_package(CUDA REQUIRED)
    if(CUDA_FOUND)
        add_definitions(-DHAVE_CUDA)
        include_directories(${CUDA_INCLUDE_DIRS})
        message("Cuda include directories ${CUDA_INCLUDE_DIRS}")
        message("Cuda libraries ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES}")
        set(LIBS ${LIBS} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES})
        set(F2PY_INCLUDE_DIRS ${F2PY_INCLUDE_DIRS} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES})
        set(F2PY_LIBS "${F2PY_LIBS} -lcublas -lcudart -lcuda")
        if (ENABLE_CUDA_PROFILING)
            set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -lineinfo --ptxas-options;-v)
            add_definitions(-DHAVE_CUDA_PROFILING)
        endif()
        if (ENABLE_NVTX)
            set(F2PY_LIBS "${F2PY_LIBS} -lnvToolsExt")
            add_definitions(-DHAVE_NVTX)
        endif()
        set(CUDA_NVCC_ARCH "60;37"
            CACHE STRING
            "List of CUDA Compute capability (35, 37, 60 for 3.5, 3.7, 6.0, respectively)"
            )
        foreach (ARCH ${CUDA_NVCC_ARCH})
            set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
                        "-gencode arch=compute_${ARCH},code=sm_${ARCH}"
                        )
        endforeach()
        
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
                        "-I${CMAKE_INSTALL_PREFIX}/include/"
                        )
        
        message("cuda nvcc flags ${CUDA_NVCC_FLAGS}")
        message(WARNING "
        >> Remember to set CUDA_HOST_COMPILER to the correct version
        (e.g. gcc <= 4.6 for CUDA 5.0)")
        if (DEBUG_CUDA)
            add_definitions(-DDEBUG_CUDA)
        endif()
    endif()
endif ()

add_definitions(-DBUBBLES_REVISION_NUMBER=\"${BUBBLES_REVISION_NUMBER}\")

if (DEBUG_MEMORY_LEAKS)
    add_definitions(-DDEBUG_MEMORY_LEAKS) 
endif ()



# Where to look for Fortran module files
include_directories(
    ${PROJECT_BINARY_DIR}/modules
    ${CMAKE_INSTALL_PREFIX}/include
    /usr/include
)

# Where to dump libraries...
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/lib
)

# Where to dump libraries...
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/lib
)

# ... and where to dump modules
set(CMAKE_Fortran_MODULE_DIRECTORY
    ${PROJECT_BINARY_DIR}/modules
    )

include_directories(
    ${CMAKE_INSTALL_PREFIX}/lib
)



add_subdirectory(src)
add_subdirectory(utils)
add_subdirectory(tests)
add_subdirectory(sandbox)
add_subdirectory(docs)
add_subdirectory(python)


