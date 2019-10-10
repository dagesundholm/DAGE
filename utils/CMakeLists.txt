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
# add_executable(interpolator.x interpolator.F90)
# target_link_libraries(interpolator.x bubbles ${LIBS})
# install (TARGETS interpolator.x COMPONENT utils DESTINATION bin)
# 
# add_executable(combine.x combine.F90)
# target_link_libraries(combine.x bubbles getkw ${LIBS})
# install (TARGETS combine.x COMPONENT utils DESTINATION bin)
# 
# add_executable(trim_bubbles.x trim_bubbles.F90)
# target_link_libraries(trim_bubbles.x bubbles getkw ${LIBS})
# install (TARGETS trim_bubbles.x COMPONENT utils DESTINATION bin)
# 
# add_executable(funinfo.x funinfo.F90)
# target_link_libraries(funinfo.x bubbles getkw ${LIBS})
# install (TARGETS funinfo.x COMPONENT utils DESTINATION bin)
# 
# add_executable(fun2xyz.x fun2xyz.F90)
# target_link_libraries(fun2xyz.x bubbles getkw ${LIBS})
# install (TARGETS fun2xyz.x COMPONENT utils DESTINATION bin)
# 
# add_executable(lcao_rhf_energy.x lcao_rhf_energy.F90)
# target_link_libraries(lcao_rhf_energy.x bubbles getkw ${LIBS})
# install (TARGETS lcao_rhf_energy.x COMPONENT utils DESTINATION bin)

#add_executable(lcao_rdft_energy.x lcao_rdft_energy.F90)
#target_link_libraries(lcao_rdft_energy.x bubbles getkw ${LIBS})
#install (TARGETS lcao_rdft_energy.x COMPONENT utils DESTINATION bin)

# add_executable(print_harmonic.x print_harmonic.F90)
# target_link_libraries(print_harmonic.x bubbles getkw ${LIBS})
# install (TARGETS print_harmonic.x COMPONENT utils DESTINATION bin)
# 
# 
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
message("-- include_directories: ${F2PY_INCLUDE_DIRS}")
message("-- compiler: ${CMAKE_Fortran_COMPILER}")
message("-- f90 flags: ${CMAKE_Fortran_FLAGS}")
message("-- binary dir: ${PROJECT_BINARY_DIR}")
# get_filename_component(PYTHON_NAME ${PYTHON_EXECUTABLE} NAME)
# add_custom_target(dage_py ALL DEPENDS dage_py)
get_filename_component(PYTHON_NAME ${Python3_EXECUTABLE} NAME)
add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/dage_fortran.so
    COMMAND f2py 
        -I${CMAKE_Fortran_MODULE_DIRECTORY}
        -L${PROJECT_BINARY_DIR}/lib
        -L${CMAKE_INSTALL_PREFIX}/lib
        -L/appl/opt/cuda/8.0/lib64
        --f90exec=${CMAKE_Fortran_COMPILER}
        --f90flags=${CMAKE_Fortran_FLAGS}
        -c ${CMAKE_CURRENT_SOURCE_DIR}/python_interface.F90
        -m dage_fortran
        --no-lower
        -lbubbles
        ${F2PY_LIBS}
    )

add_custom_target(dage ALL DEPENDS ${PROJECT_BINARY_DIR}/dage_fortran.so)
add_dependencies(dage bubbles)

set(VERSION_STRING ".cpython-37m-x86_64-linux-gnu") # yes, this is not in any way portable

# target_link_libraries(dage bubbles getkw ${LIBS})
install(FILES ${PROJECT_BINARY_DIR}/utils/dage_fortran${VERSION_STRING}.so DESTINATION
    lib/${PYTHON_NAME}/site-packages/dage COMPONENT dage)
#add_dependencies(bubbles python_interface)


