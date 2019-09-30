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
#ifndef INCLUDE_STREAMCONTAINER
#define INCLUDE_STREAMCONTAINER
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
using namespace std;

class StreamContainer {
    private:
        int streams_per_device;
        int number_of_devices;
        int *device_event_counter;
        int **stream_event_counter;
        int *device_numbers;
        bool subcontainer;
        cudaEvent_t **deviceEvents;
        cudaEvent_t **streamEvents;
        cudaStream_t **streams;
    public:
        StreamContainer(int streams_per_device = 4);
        StreamContainer(StreamContainer *parentContainer, int subcontainer_order_number, int total_subcontainers);
        StreamContainer(StreamContainer *parentContainer, int *device_order_numbers, int number_of_devices);
        cudaStream_t * getStream(int device_order_number, int stream);
        //cudaStream_t * StreamContainer::selectSequentialStream(int order_number);
        //cudaStream_t * StreamContainer::selectSplitStream(int order_number);
        void recordAndWaitEvent(int device, int stream);
        void recordAndWaitEvent(int device);
        void recordAndWaitEvent();
        cudaEvent_t *recordStreamEvent(int device, int stream);
        cudaEvent_t *recordDeviceEvent(int device);
        cudaEvent_t **recordDeviceEvents();
        
        int getDeviceOrderNumber(int device_number);
        int getDeviceNumber(int device_order_number);
        StreamContainer *getSingleDeviceContainer(int device_order_number);
        void synchronizeAllDevices();
        void synchronizeDevice(int device_order_number);
        int getNumberOfDevices();
        int getStreamsPerDevice();
        void setDevice(int device_order_number);
        void destroy();
};


__host__ inline void handleLastError(cudaError_t error, const char *filename, const int line_number) {
    if(error != cudaSuccess)
    {
        printf("CUDA error at %s:%i: %s\n", filename, line_number, cudaGetErrorString(error));
        exit(-1);
    }
}


__host__ inline void check_errors_and_lock(const char *filename, const int line_number) {
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    handleLastError(error, filename, line_number);
    //else {
    //    printf("No errors!\n");
    //}
}

__host__ inline void check_errors(const char *filename, const int line_number) {
#ifdef DEBUG_CUDA
    cudaDeviceSynchronize();
#endif

    cudaError_t error = cudaGetLastError();
    handleLastError(error, filename, line_number);
    //else {
    //    printf("No errors!\n");
    //}
}

extern "C" void streamcontainer_init(StreamContainer **streamContainer, int streams_per_device);
extern "C" void streamcontainer_destroy(StreamContainer *streamContainer);

#endif
