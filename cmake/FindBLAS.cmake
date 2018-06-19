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
# - Find a BLAS library
#
# This module will first look in BLAS_ROOT before considering the default
# system pahts.
# The linker language can be defined by setting the varable BLAS_LANG
#
# This module defines:
#
#  BLAS_INCLUDE_DIRS Where to find blas.h (or equivalent)
#  BLAS_LIBRARIES Libraries to link against to use BLAS
#  BLAS_FOUND Defined if BLAS is available
#  HAVE_BLAS To be used in #ifdefs
#  BLAS_H Name of BLAS header file
#
# None of the above will be defined unless BLAS can be found.
#
#=============================================================================
# Copyright 2011 Jonas Juselius <jonas.juselius@uit.no>
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distributed this file outside of CMake, substitute the full
#  License text for the above reference.)

include(MathLibFunctions)

if (EXISTS $ENV{MATH_ROOT})
	if (NOT DEFINED BLAS_ROOT})
		set(BLAS_ROOT $ENV{MATH_ROOT})
	endif()
endif()

if (EXISTS $ENV{BLAS_ROOT})
	if (NOT DEFINED BLAS_ROOT})
		set(BLAS_ROOT $ENV{BLAS_ROOT})
	endif()
endif()

if (MATH_LANG STREQUAL "C")
	set(blas_h cblas.h)
	set(blas_libs cblas)
else()
	set(blas_libs blas)
endif()

if (BLAS_INCLUDE_DIRS AND BLAS_LIBRARIES)
  set(BLAS_FIND_QUIETLY TRUE)
endif ()

if (NOT BLAS_FIND_COMPONENTS)
	set(BLAS_FIND_COMPONENTS MKL Atlas default)
endif()

function(find_blas)
	foreach (blas ${BLAS_FIND_COMPONENTS})
		if (${blas} MATCHES "MKL")
			find_mkl()
		elseif (${blas} MATCHES "Atlas")
			find_atlas()
		else()
			find_default()
		endif()
		if (BLAS_FOUND)
			break()
		endif()
	endforeach()
endfunction()

function(find_default)
	set(path_suffixes lib)
	find_math_header(blas)
	find_math_libs(blas)
	cache_math_result(default blas)
endfunction()

function(find_atlas)
	set(path_suffixes lib lib/atlas)
	if (MATH_LANG STREQUAL "C")
		set(blas_libs cblas atlas f77blas)
	else()
		set(blas_libs atlas f77blas)
	endif()

	find_math_header(blas)
	find_math_libs(blas)
	cache_math_result(Atlas blas)
endfunction()

function(find_mkl)
	if (MATH_LANG STREQUAL "C")
		set(blas_h mkl_cblas.h)
	endif()

	if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
		set(path_suffixes lib/intel64 lib/em64t)
		set(blas_libs mkl_core mkl_intel_lp64 mkl_sequential guide pthread m)
	else()
		set(path_suffixes lib/ia32 lib/32)
		set(blas_libs mkl_core mkl_intel mkl_sequential guide pthread m)
	endif()

	find_math_header(blas)
	find_math_libs(blas)
	if(blas_libraries)
		set(blas_libraries -Wl,--start-group ${blas_libraries} -Wl,--end-group )
	endif()
	cache_math_result(MKL blas)
endfunction()

find_blas()

if(BLAS_LIBRARIES)
   set(BLAS_FOUND TRUE)
endif()

unset(blas_h)
unset(blas_libs)
