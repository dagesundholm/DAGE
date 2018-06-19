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
#include <cuda.h>
#include <cuda_runtime.h>
//#include <cublas.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdint.h>

#include "cube.h"
#include "streamcontainer.h"
#include "cuda_blas.h"
#include "memory_leak_operators.h"


__host__ inline void check_cuda_blas_errors(const char *filename, const int line_number) {
#ifdef DEBUG_CUDA
    cudaThreadSynchronize();
#endif
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      printf("CUDA error at %s:%i: %s\n", filename, line_number, cudaGetErrorString(error));
      exit(-1);
    }
    
}

/*****************************************************
 * Constructors & Destructors                        *
 *****************************************************/

CudaBlas::CudaBlas(StreamContainer *streamContainer, int pointers_per_stream) {
    this->streamContainer = streamContainer;
    this->pointers_per_stream = pointers_per_stream;
    this->stream_order_number = 0;
    this->stream_pointers_offset = new int*[streamContainer->getNumberOfDevices()];
    
    check_cuda_blas_errors(__FILE__, __LINE__);
    
    // allocate space for device handles
    cudaHostAlloc((void **)&this->device_handles, 
                  sizeof(cublasHandle_t *) * streamContainer->getNumberOfDevices(),
                  cudaHostAllocPortable);

    // allocate space for pitched multiplication pointers
    cudaHostAlloc((void **)&this->cube1_device_slices, 
                  sizeof(double *) * streamContainer->getNumberOfDevices() * streamContainer->getStreamsPerDevice(),
                  cudaHostAllocPortable);
    // allocate space for pitched multiplication pointers
    cudaHostAlloc((void **)&this->cube2_device_slices, 
                  sizeof(double *) * streamContainer->getNumberOfDevices() * streamContainer->getStreamsPerDevice(),
                  cudaHostAllocPortable);
    // allocate space for pitched multiplication pointers
    cudaHostAlloc((void **)&this->result_cube_device_slices, 
                  sizeof(double *) * streamContainer->getNumberOfDevices() * streamContainer->getStreamsPerDevice(),
                  cudaHostAllocPortable);
    // allocate space for batched gemm-pointers
    cudaHostAlloc((void **)&this->cube1_slices, 
                  sizeof(double *) * streamContainer->getNumberOfDevices() * streamContainer->getStreamsPerDevice() * this->pointers_per_stream,
                  cudaHostAllocPortable);
    // allocate space for batched gemm-pointers
    cudaHostAlloc((void **)&this->cube2_slices, 
                  sizeof(double *) * streamContainer->getNumberOfDevices() * streamContainer->getStreamsPerDevice() * this->pointers_per_stream,
                  cudaHostAllocPortable);
    // allocate space for batched gemm-pointers
    cudaHostAlloc((void **)&this->result_slices, 
                  sizeof(double *) * streamContainer->getNumberOfDevices() * streamContainer->getStreamsPerDevice() * this->pointers_per_stream,
                  cudaHostAllocPortable);
    check_cuda_blas_errors(__FILE__, __LINE__);
    // initialize handles
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device++) {
        this->streamContainer->setDevice(device);
        this->device_handles[device] = new cublasHandle_t;
        cublasCreate(this->device_handles[device]);
        this->stream_pointers_offset[device] = new int[this->streamContainer->getStreamsPerDevice()];
        check_cuda_blas_errors(__FILE__, __LINE__);
        for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream++) {
            this->stream_pointers_offset[device][stream] = 0;
            cudaMalloc(&this->cube1_device_slices[device * this->streamContainer->getStreamsPerDevice() + stream], this->pointers_per_stream*sizeof(double*));
            cudaMalloc(&this->cube2_device_slices[device * this->streamContainer->getStreamsPerDevice() + stream], this->pointers_per_stream*sizeof(double*));
            cudaMalloc(&this->result_cube_device_slices[device * this->streamContainer->getStreamsPerDevice() + stream], this->pointers_per_stream*sizeof(double*));
        }
    }
}

void CudaBlas::destroy() {
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device++) {
        this->streamContainer->setDevice(device);
        cublasDestroy(*this->device_handles[device]);
        delete this->device_handles[device];
        delete[] this->stream_pointers_offset[device];
        for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream++) {
            cudaFree(this->cube1_device_slices[device * this->streamContainer->getStreamsPerDevice() + stream]);
            cudaFree(this->cube2_device_slices[device * this->streamContainer->getStreamsPerDevice() + stream]);
            cudaFree(this->result_cube_device_slices[device * this->streamContainer->getStreamsPerDevice() + stream]);
            
            
        }
        check_cuda_blas_errors(__FILE__, __LINE__);
    }
    cudaFreeHost(this->device_handles);
    cudaFreeHost(this->cube1_slices);
    cudaFreeHost(this->cube2_slices);
    cudaFreeHost(this->result_slices);
    cudaFreeHost(this->cube1_device_slices);
    cudaFreeHost(this->cube2_device_slices);
    cudaFreeHost(this->result_cube_device_slices);
    delete[] this->stream_pointers_offset;
    check_cuda_blas_errors(__FILE__, __LINE__);
} 

/*****************************************************
 * Operations                                        *
 *****************************************************/

/*
 *  Performs Matrix-Matrix multiplication for two Matrices and stores the result to result_slice.
 *  This method is a wrapper for cublasDgemm-method.
 */
void CudaBlas::matrixMatrixMultiplication(int device_order_number, int stream_order_number, CudaCube *slice1,
                                          CudaCube *slice2, CudaCube *result_slice, double alpha, double beta) {
    int device = device_order_number % this->streamContainer->getNumberOfDevices();
    int stream = stream_order_number % this->streamContainer->getStreamsPerDevice();
    this->streamContainer->setDevice(device);
    
    // assert that the input cubes are indeed slices
    if (slice1->isSlice() && slice2->isSlice()) {
        // get the number of rows in the cube1 and number of columns and rows in cube2
        int cube1_row_count = slice1->getShape(0);
        int cube2_row_count = slice2->getShape(0);
        int cube2_column_count = slice2->getShape(1);
        // get the leading dimensions
        int lda = slice1->getDeviceSliceLeadingDimension(device);
        int ldb = slice2->getDeviceSliceLeadingDimension(device);
        int ldc = result_slice->getDeviceSliceLeadingDimension(device);
        // get the slice pointers
        double *slice1_pointer = slice1->getDevicePointer(device);
        double *slice2_pointer = slice2->getDevicePointer(device);
        double *result_slice_pointer = result_slice->getDevicePointer(device); 
        
        
        // set the correct stream
        cublasSetStream(*this->device_handles[device], *this->streamContainer->getStream(device, stream));
        
             
    
        // do the cublas call
        cublasDgemm(*this->device_handles[device], CUBLAS_OP_N, CUBLAS_OP_N,
            cube1_row_count, cube2_column_count, cube2_row_count, &alpha,
            slice1_pointer, lda,
            slice2_pointer, ldb,
            &beta,
            result_slice_pointer, ldc
        );
    }
    else {
        printf("Attempting to perform matrix-multiplication for a non-slice CudaCube objects. This is not allowed.");
        exit(-1);
    }
    
    
}

/*
 *  Performs Batch of Matrix-Matrix multiplications by slicing one of input cubes to 'slice_dimension',
 *  multiplying with the other cube (which is a slice) and stores the result to result_cube.
 * 
 *  This method uses cublasDgemmBatched-method to perform the operations.
 */
void CudaBlas::matrixMatrixMultiplicationBatched(int device_order_number, int stream_order_number,
                                                 CudaCube *cube1, CudaCube *cube2, CudaCube *result_cube,
                                                 int slice_dimension, double alpha, double beta, cudaEvent_t *waited_event) {
    int device = device_order_number % this->streamContainer->getNumberOfDevices();
    int stream = stream_order_number % this->streamContainer->getStreamsPerDevice();
    
    this->streamContainer->setDevice(device);
    
    int batch_count = result_cube->getShape(slice_dimension);
    int lda, ldb, ldc;
    int cube2_row_count; 
    int cube2_column_count;
    int result_cube_row_count;
    int result_cube_column_count;
    int cube1_row_count;
    
    
    // result  counts
    if (slice_dimension == 2) {
        result_cube_row_count = result_cube->getShape(0);
        result_cube_column_count = result_cube->getShape(1);
    }
    else if (slice_dimension == 1) {
        result_cube_row_count = result_cube->getShape(0);
        result_cube_column_count = result_cube->getShape(2);
    }
    else {
        result_cube_row_count = result_cube->getShape(1);
        result_cube_column_count = result_cube->getShape(2);
    }
    
    
    if (this->stream_pointers_offset[device][stream] + batch_count >= this->pointers_per_stream) {
        this->stream_pointers_offset[device][stream] = 0;
    }
    int pointers_offset = this->stream_pointers_offset[device][stream];
    
    double **result_slices =
        &this->result_slices[(device*this->streamContainer->getStreamsPerDevice() + stream) * this->pointers_per_stream + pointers_offset];
    double **cube1_slices  =
        &this->cube1_slices [(device*this->streamContainer->getStreamsPerDevice() + stream) * this->pointers_per_stream + pointers_offset];
    double **cube2_slices  = &this->cube2_slices [(device*this->streamContainer->getStreamsPerDevice() + stream) * this->pointers_per_stream + pointers_offset];
    
    
    
    // check which of the input cubes is a slice
    if (cube1->isSlice()) {
        // get the number of rows in the cube1
        cube1_row_count = cube1->getShape(0);
        lda = cube1->getDeviceSliceLeadingDimension(device);
        
        double *cube1_pointer = cube1->getDevicePointer(device);
        for (int i = 0; i < batch_count; i++) {
            CudaCube *slice2 = cube2->getSlice(i, slice_dimension);
            CudaCube *result_slice = result_cube->getSlice(i, slice_dimension);
            
            // get the leading dimensions for cube1 and result cube slices
            // and the row & column counts for the cube2 slice
            if (i == 0) {
                // leading dimensions
                ldb = slice2->getDeviceSliceLeadingDimension(device);
                ldc = result_slice->getDeviceSliceLeadingDimension(device);
                
                // counts
                cube2_row_count = cube2->getShape(0);
                cube2_column_count = cube2->getShape(1);
                
            }
            
            // get the pointers to the slices and store them to arrays
            cube1_slices[i] = cube1_pointer;
            cube2_slices[i] = slice2->getDevicePointer(device);
            result_slices[i] = result_slice->getDevicePointer(device);
            
            
            // cleanup
            slice2->destroy();
            result_slice->destroy();
            delete slice2;
            delete result_slice;
        }
    }
    else if (cube2->isSlice()) {
        // get the number of rows & columns in the cube2
        cube2_row_count = cube2->getShape(0);
        cube2_column_count = cube2->getShape(1);
        
        // get the leading 
        ldb = cube2->getDeviceSliceLeadingDimension(device);
        double *cube2_pointer = cube2->getDevicePointer(device);
        
       
        for (int i = 0; i < batch_count; i++) {
            CudaCube *slice1 = cube1->getSlice(i, slice_dimension);
            CudaCube *result_slice = result_cube->getSlice(i, slice_dimension);
            
            // get the leading dimensions for cube1 and result cube slices
            // and the cube1 row count
            if (i == 0) {
                // leading dimensions
                lda = slice1->getDeviceSliceLeadingDimension(device);
                ldc = result_slice->getDeviceSliceLeadingDimension(device);
                
                // get the number of rows in the cube1
                cube1_row_count = slice1->getShape(0);
            }
            
            // get the pointers to the slices and store them to arrays
            cube1_slices[i] = slice1->getDevicePointer(device);
            cube2_slices[i] = cube2_pointer;
            result_slices[i] = result_slice->getDevicePointer(device);
            
            
            // cleanup
            slice1->destroy();
            result_slice->destroy();
            delete slice1;
            delete result_slice;
        }
    }
    
    
    // Copy the host array of device pointers to the device
    cudaMemcpyAsync(&this->cube1_device_slices[device*this->streamContainer->getStreamsPerDevice() + stream][pointers_offset],
                    cube1_slices, batch_count*sizeof(double*), cudaMemcpyHostToDevice, *this->streamContainer->getStream(device, stream)); 
    cudaMemcpyAsync(&this->cube2_device_slices[device*this->streamContainer->getStreamsPerDevice() + stream][pointers_offset],
                    cube2_slices, batch_count * sizeof(double*), cudaMemcpyHostToDevice, *this->streamContainer->getStream(device, stream));
    cudaMemcpyAsync(&this->result_cube_device_slices[device*this->streamContainer->getStreamsPerDevice() + stream][pointers_offset],
                    result_slices, batch_count*sizeof(double*), cudaMemcpyHostToDevice, *this->streamContainer->getStream(device, stream));
    // set the correct stream
    cublasSetStream(*this->device_handles[device], *this->streamContainer->getStream(device, stream));
    
    // if waited event is not NULL wait for the event to be called
    if (waited_event) {
        cudaStreamWaitEvent(*this->streamContainer->getStream(device, stream), *waited_event, 0);
    }
    // do the cublas call
    cublasDgemmBatched(*this->device_handles[device], CUBLAS_OP_N, CUBLAS_OP_N,
        result_cube_row_count, result_cube_column_count, cube2_row_count, &alpha,
        (const double **)&this->cube1_device_slices[device*this->streamContainer->getStreamsPerDevice() + stream][pointers_offset], lda,
        (const double **)&this->cube2_device_slices[device*this->streamContainer->getStreamsPerDevice() + stream][pointers_offset], ldb,
        &beta,
        (double **)&this->result_cube_device_slices[device*this->streamContainer->getStreamsPerDevice() + stream][pointers_offset], ldc, 
        batch_count
    );
    
    this->stream_pointers_offset[device][stream] += batch_count; 
    
}

/*****************************************************
 * MISC stuff                                        *
 *****************************************************/

int CudaBlas::getNumberOfDevices() {
    return this->streamContainer->getNumberOfDevices();
}

int CudaBlas::getStreamsPerDevice() {
    return this->streamContainer->getStreamsPerDevice();
}

/*****************************************************
 * Fortran interfaces                                *
 *****************************************************/

extern "C" CudaBlas *cudablas_init_cuda(StreamContainer *streamContainer, int pointers_per_stream) {
    CudaBlas *cudaBlas = new CudaBlas(streamContainer, pointers_per_stream); 
    return cudaBlas;
}

extern "C" void cudablas_destroy_cuda(CudaBlas *cudaBlas) {
    cudaBlas->destroy();
    delete cudaBlas;
}

extern "C" int cudablas_get_number_of_devices_cuda(CudaBlas *cudaBlas) {
    return cudaBlas->getNumberOfDevices();
}

extern "C" int cudablas_get_streams_per_device_cuda(CudaBlas *cudaBlas) {
    return cudaBlas->getStreamsPerDevice();
}

extern "C" void cudablas_mm_multiplication_batched_cuda(CudaBlas *cudaBlas, int device, int stream,
                                                        CudaCube *cube1, CudaCube *cube2, CudaCube *result_cube,
                                                        int slice_dimension, double alpha, double beta, 
                                                        cudaEvent_t *waited_event) {
    // go from fortran indexing to C indexing and deduct ones
    cudaBlas->matrixMatrixMultiplicationBatched(device-1, stream, cube1, cube2, result_cube, slice_dimension-1, alpha, beta, waited_event);
}

extern "C" void cudablas_mm_multiplication_cuda(CudaBlas *cudaBlas, int device, int stream,
                                                CudaCube *slice1, CudaCube *slice2, CudaCube *result_slice,
                                                double alpha, double beta) {
    // go from fortran indexing to C indexing and deduct ones
    cudaBlas->matrixMatrixMultiplication(device-1, stream-1, slice1, slice2, result_slice, alpha, beta);
}