#----------------------------------------------------------------------------------#
#    Copyright (c) 2010-2018 Pauli Parkkinen, Eelis Solala, Wen-Hua Xu,            #
#                            Sergio Losilla, Elias Toivanen, Jonas Juselius,       #
#                            Radovan Bast                                          #
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

include(FindPackageHandleStandardArgs)

if (NOT MATH_LANG)
	set(MATH_LANG C)
elseif(MATH_LANG STREQUAL "C" OR MATH_LANG STREQUAL "CXX")
	set(MATH_LANG C)
elseif(NOT MATH_LANG STREQUAL "Fortran")
	message(FATAL_ERROR "Invalid math library linker language: ${MATH_LANG}")
endif()

macro(find_math_header _service)
	string(TOUPPER ${_service} _SERVICE)
	if (${_service}_h)
		find_path(${_service}_include_dirs
			NAMES ${${_service}_h}
			PATHS ${${_SERVICE}_ROOT}
			PATH_SUFFIXES include ${path_suffixes}
			NO_DEFAULT_PATH
			)
		find_path(${_service}_include_dirs
			NAMES ${${_service}_h}
			PATH_SUFFIXES include
			)
	endif()
endmacro()

macro(find_math_libs _service)
	string(TOUPPER ${_service} _SERVICE)
	foreach(${_service}lib ${${_service}_libs})
		find_library(_lib ${${_service}lib}
			PATHS ${${_SERVICE}_ROOT}
			PATH_SUFFIXES ${path_suffixes}
			NO_DEFAULT_PATH
			)
		find_library(_lib ${${_service}lib}
			PATH_SUFFIXES ${path_suffixes}
			)
		if(_lib)
			set(${_service}_libraries ${${_service}_libraries} ${_lib})
			unset(_lib CACHE)
		else()
			break()
		endif()
	endforeach()
	unset(${_service}lib)
	unset(_lib CACHE)
endmacro()

macro(cache_math_result math_type _service)
	string(TOUPPER ${_service} _SERVICE)
	if (${_service}_h)
		set(${_SERVICE}_H ${${_service}_h})
	endif()
	if (${_service}_include_dirs)
		set(${_SERVICE}_INCLUDE_DIRS ${${_service}_include_dirs})
	endif()
	if (${_service}_libraries)
		set(${_SERVICE}_LIBRARIES ${${_service}_libraries})
	endif()
	unset(${_service}_h)
	unset(${_service}_include_dirs)
	unset(${_service}_libraries)

	if (${_SERVICE}_H)
		find_package_handle_standard_args(${_SERVICE}
			"Could NOT find ${math_type} ${_SERVICE}"
			${_SERVICE}_INCLUDE_DIRS ${_SERVICE}_LIBRARIES ${_SERVICE}_H)
	else()
		find_package_handle_standard_args(${_SERVICE}
			"Could NOT find ${math_type} ${_SERVICE}" ${_SERVICE}_LIBRARIES)
	endif()

	if (${_SERVICE}_FOUND)
		set(HAVE_${_SERVICE} ON CACHE INTERNAL "Defined if ${_SERVICE} is available")
		set(${_SERVICE}_LIBRARIES ${${_SERVICE}_LIBRARIES} CACHE STRING "${_SERVICE} libraries")
		mark_as_advanced(${_SERVICE}_LIBRARIES)
		if (${_SERVICE}_H)
			set(${_SERVICE}_H ${${_SERVICE}_H} CACHE STRING "Name of ${_SERVICE} header")
			set(${_SERVICE}_INCLUDE_DIRS ${${_SERVICE}_INCLUDE_DIRS}
				CACHE STRING "${_SERVICE} include directory")
			mark_as_advanced(${_SERVICE}_INCLUDE_DIRS ${_SERVICE}_H)
		endif()
	endif()
endmacro()
