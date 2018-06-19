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
#ifndef INCLUDE_MEMORY_FOLLOWER
#define INCLUDE_MEMORY_FOLLOWER
#include <iostream>
#include <stdio.h>

#include <vector>
using namespace std;

class MemoryAllocation {
    public:
        MemoryAllocation(void *pointer, size_t size, int line_number, const char *file);
        void * pointer;
        size_t size;
        int line_number;
        const char *file;
};

class MemoryFollower {
    public:
        vector<MemoryAllocation *> memoryAllocations;
        /** Memory allocated */
        size_t allocated;
        /** Memory deallocated */
        size_t deallocated;
        /** parent follower */
        MemoryFollower *parentMemoryFollower;
        /* init */
        MemoryFollower();
        void allocate(void *pointer, size_t size, int line_number, const char *file);
        void deallocate(void *pointer);
        /* Destroy and report */
        void destroy();
        
};

void allocate(void *pointer, size_t size, int line_number, const char *file);
void deallocate(void *pointer);
extern MemoryFollower *currentMemoryFollower;
#endif
#endif