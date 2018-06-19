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
#include "memory_follower.h"
#include <vector>
#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#include <iostream>
#include <stdio.h>

#include <vector>
using namespace std;
#ifdef DEBUG_MEMORY_LEAKS

MemoryFollower *currentMemoryFollower = NULL;

MemoryAllocation::MemoryAllocation(void *pointer, size_t size, int line_number, const char *file) {
    this->pointer = pointer;
    this->size = size;
    this->line_number = line_number;
    this->file = file;
}

MemoryFollower::MemoryFollower() {
    this->allocated = 0;
    this->deallocated = 0;
    // if the pointer is not null, store the pointer
    if (currentMemoryFollower) {
        this->parentMemoryFollower = currentMemoryFollower;
    }
    else {
        this->parentMemoryFollower = NULL;
    }
}

void MemoryFollower::destroy() {
    
    if (this->parentMemoryFollower) {
        this->parentMemoryFollower->allocated += currentMemoryFollower->allocated;
        this->parentMemoryFollower->deallocated += currentMemoryFollower->deallocated;
        currentMemoryFollower = this->parentMemoryFollower;
    }
    else {
        currentMemoryFollower = NULL;
    }
    
    
    for(int i = 0; i < this->memoryAllocations.size(); i++ ) {
        MemoryAllocation *memoryAllocation = this->memoryAllocations.at(i);
        if (currentMemoryFollower) {
            printf("Memory not deallocated, pushing to parent (%ld): file: %s, line %d.\n", memoryAllocation->size, memoryAllocation->file, memoryAllocation->line_number);
            currentMemoryFollower->memoryAllocations.push_back(memoryAllocation);
        }
        else {
            printf("Memory not deallocated (%ld): file: %s, line %d.\n", memoryAllocation->size, memoryAllocation->file, memoryAllocation->line_number);
            delete memoryAllocation;
        }
    }
    
    printf("Finalizing memory follower, allocated: %ld, deallocated: %ld, still available: %ld\n", 
           this->allocated, this->deallocated,
           this->allocated - this->deallocated 
          );
    if (currentMemoryFollower) {
        printf("Parent status: allocated: %ld, deallocated: %ld, still available: %ld\n", 
            currentMemoryFollower->allocated, currentMemoryFollower->deallocated,
            currentMemoryFollower->allocated - currentMemoryFollower->deallocated
        );
    }
    
    
}

void MemoryFollower::allocate(void *ptr, size_t size, int line_number, const char *file) {
    MemoryAllocation *memoryAllocation = new MemoryAllocation(ptr, size, line_number, file);
    this->memoryAllocations.push_back(memoryAllocation);
    this->allocated += size;
}

void MemoryFollower::deallocate(void *ptr) {
    bool deallocated = false;
    for(int i = 0; i < this->memoryAllocations.size(); i++ ) {
        MemoryAllocation *memoryAllocation = this->memoryAllocations.at(i);
        if (memoryAllocation->pointer == ptr) {
            this->deallocated += memoryAllocation->size;
            memoryAllocations.erase(memoryAllocations.begin() +i);
            delete memoryAllocation;
            deallocated = true;
            break;
        }
    }
    if (!deallocated && this->parentMemoryFollower) {
        this->parentMemoryFollower->deallocate(ptr);
    }
        
}




extern "C" void memoryfollower_init_() {
    currentMemoryFollower = new MemoryFollower();
}

extern "C" void memoryfollower_destroy_() {
    if (currentMemoryFollower) {
        currentMemoryFollower->destroy();
    }
}


#endif

extern "C" void memoryfollower_print_status_() {
#ifdef DEBUG_MEMORY_LEAKS
    if (currentMemoryFollower) {
        printf("--- Memory Follower status: allocated: %ld, deallocated: %ld, still available: %ld\n", 
            currentMemoryFollower->allocated, currentMemoryFollower->deallocated,
            currentMemoryFollower->allocated - currentMemoryFollower->deallocated);
    }
#endif
}

void allocate(void *ptr, size_t size, int line_number, const char *file) {
#ifdef DEBUG_MEMORY_LEAKS
    if (currentMemoryFollower) {
        currentMemoryFollower->allocate(ptr, size, line_number, file);
    }
#endif
}

void deallocate(void *ptr) {
#ifdef DEBUG_MEMORY_LEAKS
    if (currentMemoryFollower) {
        currentMemoryFollower->deallocate(ptr);
    }
#endif
}

extern "C" void check_memory_cuda_() {
#ifdef HAVE_CUDA
    int device_count;
    cudaGetDeviceCount(&device_count); 
    for (int i = 0; i < device_count; i++) {
        cudaSetDevice(i);
        size_t mem_tot_0 = 0;
        size_t mem_free_0 = 0;
        cudaMemGetInfo  (&mem_free_0, &mem_tot_0);
        printf("Free memory on device: %d: %ld / %ld\n ", i, mem_free_0, mem_tot_0);
    }
#endif
}

