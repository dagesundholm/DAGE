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
set(SOURCES
        #boysf.F90
        action.F90
        bubbles.F90
        cartiter.F90
        generator.F90
        gauss_generator.F90
        slater_generator.F90
        cubeplot.F90
        multipole.F90
        multipole_tools.F90
        helmholtz_multipole_tools.F90
        #datadump.F90
        analytical.F90
        exponentials.F90
        file_format.F90
        function3d_types.F90
        function3d.F90
        gauss.F90
        #getkw.F90
        globals.F90
        gopenmol.F90
        grid.F90
        harmonic.F90
        harmonics.F90
        evaluators.F90
        io.F90
        lip.F90
        mpi.F90
        #operator.F90
        pbc.F90
        potential.F90
        pprinter.F90
        realarrays.F90
        timer.F90
        xmatrix.F90
        lcao.F90
        scf.F90
        core_evaluator.F90
        helmholtz3d.F90
        scf_cycle.F90
        chrono.F90
        coulomb.F90
        laplacian.F90
        #projector.F90
        parallel_info.F90
        diis.F90
        settings.F90
        core.F90
        cuda_object.F90
        memory_leak_checker.F90
        scf_energetics.F90
        #schrodinger.F90
        ../gbfmm/gbfmm_function3d.F90
        ../gbfmm/gbfmm_coulomb3d.F90
        ../gbfmm/gbfmm_laplacian3d.F90
        ../gbfmm/gbfmm_parallel_info.F90
        ../gbfmm/gbfmm_helmholtz3d.F90
        #xc.F90
        #xc_lebedev.F90
        #schrodinger.F90
        #modified_bessel.F90
        initial_guess.F90
        tensor_decomposition.F90
        ci.F90
        points.F90
        memory_follower.cpp
        memory_leak_operators.cpp
)

if (ENABLE_DFT AND LIBXC)
    set(SOURCES ${SOURCES}
        xc.F90
        xc_lebedev.F90)
endif()


if(CUDA_FOUND)
    set(SOURCES ${SOURCES}
        cuda.F90
        streamcontainer.cpp
        cube.cu
        cublas_iface.cpp
        bubbles_cuda.cu
        injector_cuda.F90
        spherical_harmonics_cuda.cu
        bubbles_multiplier.cu
        function3d_multiplier.cu
        cuda_blas.cu
        grid.cu
        integrator.cu
        bessels.cu
        cuda_profiling.cu
        evaluators.cu
        points.cu
        gpu_info.cu
        gpu_info.F90
        ../gbfmm/gbfmm_helmholtz3d.cu
        ../gbfmm/gbfmm_coulomb3d.cu
        ../gbfmm/gbfmm_potential_operator.cu)
    cuda_add_library(bubbles SHARED
        ${SOURCES}
        )
    #Require compute capability 1.3 to handle doubles. Compute cabability 35 needed for optimal performance
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
else()
    add_library(bubbles SHARED ${SOURCES})
endif()

set_target_properties(bubbles PROPERTIES LINKER_LANGUAGE CXX)
install (TARGETS bubbles LIBRARY DESTINATION lib COMPONENT bubbles)
install (DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}/ DESTINATION include/bubbles 
         COMPONENT bubbles FILES_MATCHING PATTERN "*.mod")


