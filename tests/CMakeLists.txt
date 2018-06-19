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
file(GLOB test_files 
gauss_test.F90
io_test.F90
iterator_test.F90
mpi_test.F90
multipoles_test.F90
multipole_test.F90
slater_test.F90
helmholtz_test.F90
multipole_translation_test.F90
gbfmm_potential_test.F90
gbfmm_update_test.F90
)

#foreach (test ${test_files})
#    get_filename_component(exe ${test} NAME_WE)
#    add_executable(${exe}.x ${test})
#    target_link_libraries(${exe}.x bubbles ${LIBS})
#    install (TARGETS ${exe}.x COMPONENT tests DESTINATION bin)
#endforeach(test)

