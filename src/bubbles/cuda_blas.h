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
#ifndef INCLUDE_CUDABLAS
#define INCLUDE_CUDABLAS
#include "streamcontainer.h"
#include "cube.h"
using namespace std;

class CudaBlas {
    private:
        // pointer to streamcontainer used in the matrix
        StreamContainer *streamContainer;
        // 
        cublasHandle_t ** device_handles;
        int stream_order_number;
        int pointers_per_stream;
        int **stream_pointers_offset;
        double **cube1_slices;
        double **cube2_slices;
        double **result_slices;
        double **cube1_device_slices;
        double **cube2_device_slices;
        double **result_cube_device_slices;
    public:
        CudaBlas(StreamContainer *streamContainer, int pointers_per_stream);
        void matrixMatrixMultiplication(int device_order_number, int stream_order_number, CudaCube *slice1, CudaCube *slice2, CudaCube *result_slice, double alpha, double beta);
        void matrixMatrixMultiplicationBatched(int device_order_number, int stream_order_number,
                                               CudaCube *cube1, CudaCube *cube2, CudaCube *result_cube,
                                               int slice_dimension, double alpha, double beta, cudaEvent_t *waited_event);
        int getNumberOfDevices();
        int getStreamsPerDevice();
        // destroy all cuda related objects
        void destroy();
};

#endif