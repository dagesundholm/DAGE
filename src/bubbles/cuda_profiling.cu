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
#ifdef HAVE_CUDA_PROFILING

#include <stdio.h>
#include "nvToolsExt.h"
#include "cuda_profiler_api.h"

const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);
#ifdef HAVE_NVTX
#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif
/*extern "C" void initialize_cuda_profiler_() {
    cudaProfilerInitialize();       
}*/


int color_number = 0;
bool cuda_profiler_started = false;


extern "C" void start_cuda_profiling_() {
    printf("------------------Starting cuda profiling----------------\n");
    int number_of_devices;
    cudaGetDeviceCount(&number_of_devices);
    for (int device = 0; device < number_of_devices; device ++) {
        cudaSetDevice(device);
        cudaProfilerStart();
    }
    cuda_profiler_started = true;
}

extern "C" void stop_cuda_profiling_() {
    printf("------------------Stopping cuda profiling----------------\n");
    int number_of_devices;
    cudaGetDeviceCount(&number_of_devices);
    for (int device = 0; device < number_of_devices; device ++) {
        cudaSetDevice(device);
        cudaProfilerStop();
    }
    cuda_profiler_started = false;
}

extern "C" void start_nvtx_timing_(char * title) {
    if (cuda_profiler_started) {
        PUSH_RANGE(title, color_number);
        color_number ++;
    }
}

extern "C" void stop_nvtx_timing_() {
    if (cuda_profiler_started) {
        POP_RANGE;
    }
}
#endif

