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
set(MATH_FOUND  FALSE)
set(MATH_LANG   "Fortran")

# user sets

# unset(USERDEFINED_LIBS CACHE)
# unset(MATH_LIBS CACHE)

set(USERDEFINED_LIBS
# comment out. otherwise the variable gets longer and longer 
# after each run of 'cmake ..'
         "${USERDEFINED_LIBS}"
#       /usr/lib/libblas.so
#       /usr/lib/liblapack.so
#       ${PROJECT_SOURCE_DIR}/lib/libxcf90.a
#       ${PROJECT_SOURCE_DIR}/lib/libxc.a;
	CACHE STRING
	"User set math libraries"
	FORCE
	)
if(NOT "${USERDEFINED_LIBS}" STREQUAL "")
	set(MATH_LIBS
		"${USERDEFINED_LIBS}"
		CACHE STRING
		"User set math libraries"
		FORCE
		)
	message("-- User set math libraries: ${MATH_LIBS}")
        
        set(F2PY_LIBS ${F2PY_LIBS} ${MATH_LIBS})
	set(MATH_FOUND TRUE)
endif()

# try to find the best library using environment variables
if(NOT MATH_FOUND)
	find_package(BLAS)
	find_package(LAPACK)
	if(BLAS_FOUND AND LAPACK_FOUND)
		set(MATH_LIBS
			${BLAS_LIBRARIES}
			${LAPACK_LIBRARIES}
			)
		set(MATH_FOUND TRUE)
		set(F2PY_LIBS ${F2PY_LIBS} "-llapack" "-lblas")
	endif()
endif()

if(MATH_FOUND)
	set(LIBS
		${LIBS}
		${MATH_LIBS}
		)
	add_definitions(-DHAVE_BLAS)
	add_definitions(-DHAVE_LAPACK)
else()
	message("-- No external math libraries found")
endif()

