/*----------------------------------------------------------------------------------*
 *    Copyright (c) 2010-2018 Pauli Parkkinen, Eelis Solala, Wen-Hua Xu,            *
 *                            Sergio Losilla, Elias Toivanen, Jonas Juselius        *
 *                                                                                  *
 *    Permission is hereby granted, free of charge, to any person obtaining a copy  *
 *    of this software and associated documentation files (the "Software"), to deal *
 *    in the Software without restriction, including without limitation the rights  *
 *    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell     *
 *    copies of the Software, and to permit persons to whom the Software is         *
 *    furnished to do so, subject to the following conditions:                      *
 *                                                                                  *
 *    The above copyright notice and this permission notice shall be included in all*
 *    copies or substantial portions of the Software.                               *
 *                                                                                  *
 *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    *
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      *
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE   *
 *    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        *
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, *
 *    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE *
 *    SOFTWARE.                                                                     *
 *----------------------------------------------------------------------------------*/
#ifdef DEBUG_MEMORY_LEAKS
#ifndef MEMCHECK_H
#define MEMCHECK_H
#include <iostream>
#include <stdio.h>
#include <malloc.h>

#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

void *operator new (size_t, const char *, int);
void operator delete (void *, const char *, int);
void *operator new[] (size_t, const char *, int);      // for arays
void operator delete[] (void *, const char *, int);
#ifdef HAVE_CUDA
cudaError_t myCudaFreeHost(void *ptr, const char *file, int lineno);
cudaError_t myCudaHostAlloc(void **ptr, size_t size, unsigned int flags, const char *file, int lineno);
cudaError_t myCudaHostRegister(void *ptr, size_t size, unsigned int flags, const char *file, int lineno);  
cudaError_t myCudaHostUnregister(void *ptr, const char *file, int lineno);



//#define cudaHostRegister(x, y, z) myCudaHostRegister(x, y, z, __FILE__, __LINE__)
//#define cudaHostUnregister(p)     myCudaHostUnregister(p, __FILE__, __LINE__)
#define cudaHostAlloc(x, y, z)    myCudaHostAlloc(x, y, z, __FILE__, __LINE__)
#define cudaFreeHost(p)           myCudaFreeHost(p, __FILE__, __LINE__)
#endif

#define new new(__FILE__, __LINE__)
#endif 
#endif