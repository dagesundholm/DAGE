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
# - Find a LAPACK library
#
# This module will first look in LAPACK_ROOT before considering the default
# system pahts.
# The linker language can be defined by setting the varable LAPACK_LANG
#
# This module defines:
#
#  LAPACK_INCLUDE_DIRS Where to find lapack.h (or equivalent)
#  LAPACK_LIBRARIES Libraries to link against to use LAPACK
#  LAPACK_FOUND Defined if LAPACK is available
#  HAVE_LAPACK To be used in #ifdefs
#  LAPACK_H Name of LAPACK header file
#
# None of the above will be defined unless LAPACK can be found.
#
#=============================================================================
# Copyright 2011 Jonas Juselius <jonas.juselius@uit.no>
#                Radovan Bast   <radovan.bast@uit.no>
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
	if (NOT DEFINED LAPACK_ROOT})
		set(LAPACK_ROOT $ENV{MATH_ROOT})
	endif()
endif()

if (EXISTS $ENV{LAPACK_ROOT})
	if (NOT DEFINED LAPACK_ROOT})
		set(LAPACK_ROOT $ENV{LAPACK_ROOT})
	endif()
endif()

# Default names for the headers
if (MATH_LANG STREQUAL "C")
	set(lapack_h clapack.h)
endif()
set(lapack_libs lapack)

if (LAPACK_INCLUDE_DIRS AND LAPACK_LIBRARIES)
  set(LAPACK_FIND_QUIETLY TRUE)
endif ()

if (NOT LAPACK_FIND_COMPONENTS)
	set(LAPACK_FIND_COMPONENTS MKL Atlas default)
endif()

function(find_lapack)
	foreach (lapack ${LAPACK_FIND_COMPONENTS})
		if (${lapack} MATCHES "MKL")
			find_mkl()
		elseif (${lapack} MATCHES "Atlas")
			find_atlas()
		else()
			find_default()
		endif()
		if (LAPACK_FOUND)
			break()
		endif()
	endforeach()
endfunction()

function(find_default)
	set(path_suffixes lib)
	find_math_header(lapack)
	find_math_libs(lapack)
	cache_math_result(default lapack)
endfunction()

function(find_atlas)
	set(lapack_libs lapack_atlas)

	set(path_suffixes include/atlas include)
	find_math_header(lapack)
	set(path_suffixes lib lib/atlas)
	find_math_libs(lapack)
	cache_math_result(Atlas lapack)
endfunction()

function(find_mkl)
	if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
		set(path_suffixes lib/intel64 lib/em64t)
		set(lapack_libs mkl_lapack)
	else()
		set(path_suffixes lib/ia32 lib/32)
		set(lapack_libs mkl_lapack)
	endif()

	find_math_header(lapack)
	find_math_libs(lapack)

	if(lapack_libraries)
		set(lapack_libraries
			-Wl,--start-group ${lapack_libraries} -Wl,--end-group )
	endif()
	cache_math_result(MKL lapack)
endfunction()

find_lapack()

if(LAPACK_LIBRARIES)
   set(LAPACK_FOUND TRUE)
endif()

unset(lapack_h)
unset(lapack_libs)
