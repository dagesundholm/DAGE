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
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <stdio.h>

#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <malloc.h>
#include "memory_follower.h"
#undef new
#undef delete


/* Prototypes for our hooks.  */
static void my_init_hook (void);
static void *my_malloc_hook (size_t, const void *);
static void my_free_hook (void*, const void *);
static void *(* old_malloc_hook)(size_t, const void *);
static void (* old_free_hook)(void *, const void *);


 /* Override initializing hook from the C library. */
 void (*__malloc_initialize_hook) (void) = my_init_hook;


static void my_init_hook (void) {
  old_malloc_hook = __malloc_hook;
  old_free_hook = __free_hook;
  __malloc_hook = my_malloc_hook;
  __free_hook = my_free_hook;
}


static void * my_malloc_hook (size_t size, const void *caller) {
  void *result;
  
  /* Restore all old hooks */
  __malloc_hook = old_malloc_hook;
  __free_hook = old_free_hook;
  
  /* Call recursively */
  result = malloc (size);
  
  /* Save underlying hooks */
  old_malloc_hook = __malloc_hook;
  old_free_hook = __free_hook;
  
  /* printf might call malloc, so protect it too. */
  printf ("malloc (%u) returns %p\n", (unsigned int) size, result);
  
  /* Restore our own hooks */
  __malloc_hook = my_malloc_hook;
  __free_hook = my_free_hook;
  return result;
}

static void my_free_hook (void *ptr, const void *caller) {
  /* Restore all old hooks */
  __malloc_hook = old_malloc_hook;
  __free_hook = old_free_hook;
  
  /* Call recursively */
  free (ptr);
  
  /* Save underlying hooks */
  old_malloc_hook = __malloc_hook;
  old_free_hook = __free_hook;
  
  /* printf might call free, so protect it too. */
  printf ("freed pointer %p\n", ptr);
  
  /* Restore our own hooks */
  __malloc_hook = my_malloc_hook;
  __free_hook = my_free_hook;
}

int main () {
    my_init_hook ();
    return 0;
}

#ifdef HAVE_CUDA
cudaError_t myCudaFreeHost(void *ptr, const char *file, int lineno) {    
    if (currentMemoryFollower) {
        currentMemoryFollower->deallocate(ptr);
    }
    return cudaFreeHost(ptr);
}

     
cudaError_t myCudaHostAlloc(void **ptr, size_t size, unsigned int flags, const char *file, int lineno) {    
    cudaError_t error = cudaHostAlloc(ptr, size, flags);
    if (currentMemoryFollower) {
        currentMemoryFollower->allocate(*ptr, size, lineno, file);
    }
    return error;
}

cudaError_t myCudaHostRegister(void *ptr, size_t size, unsigned int flags, const char *file, int lineno) {     
    cudaError_t error = cudaHostRegister(ptr, size, flags);  
    if (currentMemoryFollower) {
        currentMemoryFollower->allocate(ptr, size, lineno, file);
    }
    return error;
}     
     
cudaError_t myCudaHostUnregister(void *ptr, const char *file, int lineno) {    
    if (currentMemoryFollower) {
        currentMemoryFollower->deallocate(ptr);
    }
    return cudaHostUnregister(ptr);
}
#endif


void *operator new (size_t size, const char *file, int lineno) {
    void *pointer =  new char[size];
    if (currentMemoryFollower) {
        currentMemoryFollower->allocate(pointer, size, lineno, file);
    }
    return pointer;
}

// Overload array new
void* operator new[](size_t size, const char* file, int line) {
    return operator new(size, file, line);
}

void operator delete (void *ptr) {
    if (currentMemoryFollower) {
        currentMemoryFollower->deallocate(ptr);
    }
    free(ptr);   
}

void operator delete (void *ptr, const char* file, int line) {
    if (currentMemoryFollower) {
        currentMemoryFollower->deallocate(ptr);
    }
    free(ptr);   
}

// Override array delete
void operator delete[](void* p, const char* file, int line) {
    operator delete(p, file, line);
} 


#endif
