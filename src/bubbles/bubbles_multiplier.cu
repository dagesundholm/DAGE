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
#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
//#include <algorithm> *std::max_element(result_cube, result_cube + totalPointCount)
#include "bubbles_cuda.h"
#include "streamcontainer.h"
#include "grid.h"
#include "bubbles_multiplier.h"
#include "function3d_multiplier.h"
#include "memory_leak_operators.h"

__host__ inline void check_multiplier_errors(const char *filename, const int line_number) {
#ifdef CUDA_DEBUG
    cudaThreadSynchronize();
#endif
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      printf("CUDA error at %s:%i: %s\n", filename, line_number, cudaGetErrorString(error));
      exit(-1);
    }
}

template<typename T>
__device__ __forceinline__ T ldg(const T* ptr) {
#if __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

// the multiplying kernel functions
__device__ inline int product_identifier(const int idx1, const int idx2, const int lmax)  {
    int is=min(idx1,idx2)+1;
    int il=max(idx1,idx2)+1;
    int nmax = (lmax+1)*(lmax+1);
    int result = (is-1)*(2*nmax-is)/2+il -1;
    return result;
}


__global__ void Bubble_product_kernel(Bubble *bubble, Bubble *bubble1, Bubble *result_bubble, 
                                      const double* __restrict__ coefficients, const int* __restrict__ number_of_terms,
                                      const int* __restrict__ result_lm, const int* __restrict__ positions,
                                      const int offset, const int max_id, const size_t device_f_pitch, double factor) {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    const int index= id + offset;
    int result_index, prod_idx, i, lm_counter = 0, lm_counter1 = 0, term_id;
    const int lmax0 = bubble->lmax, lmax1 = bubble1->lmax, lmax2 = bubble->lmax + bubble1->lmax;
    const int nmax0 = (lmax0 +1) * (lmax0+1), nmax1 = (lmax1 +1) * (lmax1+1);
    double value, value1, value12, value2;
    if (id < max_id) {
        // go through all l, m values of input bubble 'bubble'
        for (lm_counter = 0; lm_counter < nmax0; lm_counter++) {
            // get the value for the point 'index' for 'bubble' with current l, m values
            value   =  bubble->f[lm_counter * device_f_pitch / sizeof(double) + index];
            value12 = (lm_counter < nmax1) ? bubble1->f[lm_counter * device_f_pitch / sizeof(double) + index] : 0.0;
            for (lm_counter1 = lm_counter; lm_counter1 < nmax1 ; lm_counter1++) {
                prod_idx = product_identifier(lm_counter, lm_counter1, lmax2);
                term_id = ldg<int>(&positions[prod_idx])-1;
                // get the value for the point 'index' for 'bubble' with current l1, m1 values
                value1  = value * bubble1->f[lm_counter1 * device_f_pitch / sizeof(double)+index];
                value2  = (lm_counter == lm_counter1 || lm_counter1 >= nmax0)
                          ? 0.0 : value12 * bubble->f[lm_counter1 * device_f_pitch / sizeof(double)+index];
                value1 += value2;
                for (i = 0; i < ldg<int>(&number_of_terms[prod_idx]); i++) {
                    result_index = (ldg<int>(&result_lm[term_id]) - 1) * device_f_pitch / sizeof(double) + index;
                    result_bubble->f[result_index] += factor * ldg<double>(&coefficients[term_id]) * value1;
                    term_id ++;
                }
                         
            }
        }
    }
}

// BubblesMultiplier-class functions
BubblesMultiplier::BubblesMultiplier(Bubbles *bubbles1, Bubbles *bubbles2, Bubbles *result_bubbles, 
                                     Bubbles *taylor_series_bubbles1, Bubbles *taylor_series_bubbles2,
                                     int lmax,
                                     double *coefficients, int *number_of_terms, int *result_lm, int *positions,
                                     int result_lm_size, int processor_order_number,
                                     int number_of_processors, StreamContainer *streamContainer) {

    this->bubbles1 = bubbles1;
    this->bubbles2 = bubbles2;
    this->result_bubbles = result_bubbles;
    this->taylor_series_bubbles1 = taylor_series_bubbles1;
    this->taylor_series_bubbles2 = taylor_series_bubbles2;
    
    this->processor_order_number = processor_order_number;
    this->number_of_processors = number_of_processors;
    this->streamContainer = streamContainer;

    
    
    this->result_bubbles->setProcessorConfiguration(this->processor_order_number, this->number_of_processors);
    this->bubbles1->setProcessorConfiguration(this->processor_order_number, this->number_of_processors);
    this->bubbles2->setProcessorConfiguration(this->processor_order_number, this->number_of_processors);
    if (this->taylor_series_bubbles1) {
        this->taylor_series_bubbles1->setProcessorConfiguration(this->processor_order_number, this->number_of_processors);
    }
    if (this->taylor_series_bubbles2) {
        this->taylor_series_bubbles2->setProcessorConfiguration(this->processor_order_number, this->number_of_processors);
    }


    // allocate the arrays to contain device-wise pointers
    cudaHostAlloc((void **)&this->device_number_of_terms, sizeof(int * ) * this->streamContainer->getNumberOfDevices(), cudaHostAllocPortable);
    check_multiplier_errors(__FILE__, __LINE__);
    cudaHostAlloc((void **)&this->device_positions, sizeof(int * ) * this->streamContainer->getNumberOfDevices(), cudaHostAllocPortable);
    check_multiplier_errors(__FILE__, __LINE__);
    cudaHostAlloc((void **)&this->device_result_lm, sizeof(int * ) * this->streamContainer->getNumberOfDevices(), cudaHostAllocPortable);
    check_multiplier_errors(__FILE__, __LINE__);
    cudaHostAlloc((void **)&this->device_coefficients, sizeof(double * ) * this->streamContainer->getNumberOfDevices(), cudaHostAllocPortable);
    check_multiplier_errors(__FILE__, __LINE__);

    
    
    // allocate & copy the array containing the number of result terms per l,m -pair: 'number_of_terms'
    int nmax = (lmax + 1) * (lmax + 1);
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        cudaSetDevice(device);
        
        size_t size = nmax*(nmax+1)/2 * sizeof(int);

        cudaMalloc(&this->device_number_of_terms[device], size);
        check_multiplier_errors(__FILE__, __LINE__);
        cudaMemcpy(this->device_number_of_terms[device], number_of_terms, size, cudaMemcpyHostToDevice);
        check_multiplier_errors(__FILE__, __LINE__);

        cudaMalloc(&this->device_positions[device], size);
        check_multiplier_errors(__FILE__, __LINE__);
        cudaMemcpy(this->device_positions[device], positions, size, cudaMemcpyHostToDevice);
        check_multiplier_errors(__FILE__, __LINE__);

        
        // allocate & copy the array containing the result 'ilm'-addresses (l, m pair) of the result bubbles 
        size = result_lm_size * sizeof(int);
        cudaMalloc(&this->device_result_lm[device], size);
        check_multiplier_errors(__FILE__, __LINE__);
        cudaMemcpy(this->device_result_lm[device], result_lm, size, cudaMemcpyHostToDevice);
        check_multiplier_errors(__FILE__, __LINE__);

        
        // allocate & copy the array containing the result 'coefficients' (l, m pair) of the result bubbles 
        size = result_lm_size * sizeof(double);
        cudaMalloc(&this->device_coefficients[device], size);
        check_multiplier_errors(__FILE__, __LINE__);
        cudaMemcpy(this->device_coefficients[device], coefficients, size, cudaMemcpyHostToDevice);
        check_multiplier_errors(__FILE__, __LINE__);

    }
}



void BubblesMultiplier::multiplyBubble(int ibub, Bubbles* bubbles1, Bubbles* bubbles2, Bubbles* result_bubbles, double factor, int first_cell, int last_cell) {
    // calculate the total number of points in the bubbles each l,m -pair, 
    int total_point_count;
    if (first_cell >= 0 && last_cell >= 0 ) {
        int ncell = last_cell - first_cell;
        total_point_count = ncell * (bubbles1->getBubble(ibub)->grid->nlip - 1) +1;
    }
    else {
        total_point_count = bubbles1->getBubble(ibub)->grid->ncell * (bubbles1->getBubble(ibub)->grid->nlip - 1) +1;
    }
    check_multiplier_errors(__FILE__, __LINE__);

    // determine how many of the points belong to the current mpi-node
    int remainder = total_point_count % this->number_of_processors;
    int processor_point_count = total_point_count / this->number_of_processors 
                                + ( remainder > this->processor_order_number);
    // get the offset to the f-array caused by other processors
    int offset = processor_order_number * total_point_count / this->number_of_processors +
                 ((remainder < processor_order_number) ? remainder : processor_order_number);
    
    int block_size = 256;
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        cudaSetDevice(device);
        // detemine how many of the mpi-node's points belong to this device (gpu)
        int device_point_count = processor_point_count / this->streamContainer->getNumberOfDevices() +
                                 ((processor_point_count % this->streamContainer->getNumberOfDevices()) > device);
        for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream ++) {
            // detemine the number of the points handled by this stream
            int stream_point_count = device_point_count / this->streamContainer->getStreamsPerDevice() +
                                    ((device_point_count % this->streamContainer->getStreamsPerDevice()) > stream);
            
            int grid_size = (stream_point_count + block_size - 1) / block_size;
            // call the kernel
            Bubble_product_kernel <<<grid_size, block_size, 0, *this->streamContainer->getStream(device, stream)>>> 
                (bubbles1->getBubble(ibub)->device_copies[device], bubbles2->getBubble(ibub)->device_copies[device], 
                 result_bubbles->getBubble(ibub)->device_copies[device], this->device_coefficients[device],
                 this->device_number_of_terms[device], this->device_result_lm[device], this->device_positions[device], offset, stream_point_count, 
                 bubbles1->getBubble(ibub)->device_f_pitch[device], factor);
            check_multiplier_errors(__FILE__, __LINE__);

            // increase the offset for the next calls.
            offset += stream_point_count;
        }
    }
}


void BubblesMultiplier::multiplyBubble(int ibub, double *bubble1_bf, double *bubble2_bf, double *result_bubble_bf,
                                       double *taylor_series_bubble1_bf, double *taylor_series_bubble2_bf,
                                       int lmax1, int lmax2, int tlmax1, int tlmax2
                                      ) {
    this->complex_multiplication = true;
    Bubble *result_bubble = this->result_bubbles->getBubble(ibub);
    Bubble *bubble1       = this->bubbles1->getBubble(ibub);
    Bubble *bubble2       = this->bubbles2->getBubble(ibub);
    
    // register the host target of result array and set the values of the device array to zero
    if (result_bubble && result_bubble_bf) {
        result_bubble->registerHost(result_bubble_bf);
        result_bubble->setToZero();
        bubble1->setToZero();
        bubble2->setToZero();
    }
    else {
        printf("ERROR: the result_bubble or result_bubble_bf should not be NULL");
    }
    
    if (bubble1_bf) {
        // upload the diagonal and off-diagonal parts of bubble 1 and add them together
        bubble1->upload(bubble1_bf, lmax1);
    }
    else {
        bubble1->setToZero();
    }
    check_multiplier_errors(__FILE__, __LINE__);
    
    if (taylor_series_bubble1_bf) {  
        Bubble *taylor_bubble1 = this->taylor_series_bubbles1->getBubble(ibub);
        taylor_bubble1->setToZero();
        taylor_bubble1->upload(taylor_series_bubble1_bf, tlmax1);
        bubble1->add(taylor_bubble1);
        check_multiplier_errors(__FILE__, __LINE__);
    }
    
    if (bubble2_bf) {
        // upload the diagonal and off-diagonal parts of bubble 2 and add them together
        bubble2->upload(bubble2_bf, lmax2, bubble1_bf != bubble2_bf);
    }
    else {
        bubble2->setToZero();
    }
    check_multiplier_errors(__FILE__, __LINE__);
        
    if (taylor_series_bubble2_bf) {
        Bubble *taylor_bubble2= this->taylor_series_bubbles2->getBubble(ibub);
        taylor_bubble2->setToZero();
        taylor_bubble2->upload(taylor_series_bubble2_bf, tlmax2, taylor_series_bubble1_bf != taylor_series_bubble2_bf);
        bubble2->add(taylor_bubble2);
        check_multiplier_errors(__FILE__, __LINE__);
    }
    
    
    // multiply the bubble1 with bubble2
    this->multiplyBubble(ibub, this->bubbles1, this->bubbles2, this->result_bubbles, 1.0);
    
    // deduct the taylor bubble1 * taylor_bubble2 from the result, if both are present
    if (taylor_series_bubble1_bf && taylor_series_bubble2_bf) {
        this->multiplyBubble(ibub, this->taylor_series_bubbles1, this->taylor_series_bubbles2, this->result_bubbles, -1.0);
    }
    check_multiplier_errors(__FILE__, __LINE__);
}

void BubblesMultiplier::downloadResult(int lmax, int *ibubs, int nbub) {
    if (nbub > 0) {
        for (int i = 0; i < nbub; i++) {
            this->result_bubbles->getBubble(ibubs[i])->download(lmax);
        }
        check_multiplier_errors(__FILE__, __LINE__);
        // as we are done with all uploading and downloading, 
        // unregister the host arrays of bubbles
        this->result_bubbles->unregister();
        check_multiplier_errors(__FILE__, __LINE__);
        this->bubbles1->unregister();
        check_multiplier_errors(__FILE__, __LINE__);
        // if the multiplied bubbles are different, then unregister the second
        // bubbles also
        if (this->bubbles2->getBubbleWithLocalOrderNumber(0)->f != this->bubbles1->getBubbleWithLocalOrderNumber(0)->f) {
            this->bubbles2->unregister();
            check_multiplier_errors(__FILE__, __LINE__);
        }
        if (this->complex_multiplication && this->taylor_series_bubbles1 && this->taylor_series_bubbles2) {
            this->taylor_series_bubbles1->unregister();
            check_multiplier_errors(__FILE__, __LINE__);
            
            // if the multiplied taylor series bubbles are different, then unregister the second
            // taylor series bubbles also
            if (this->taylor_series_bubbles1->getBubbleWithLocalOrderNumber(0)->f != this->taylor_series_bubbles2->getBubbleWithLocalOrderNumber(0)->f) {
                this->taylor_series_bubbles2->unregister();
                check_multiplier_errors(__FILE__, __LINE__);
            }
        }
    }
}

void BubblesMultiplier::setK(int bubble1_k, int bubble2_k, int result_bubble_k, int taylor_series_bubble1_k, int taylor_series_bubble2_k) {
    
    for (int i = 0; i < this->bubbles1->getBubbleCount(); i ++) {
        this->bubbles1->getBubbleWithLocalOrderNumber(i)->k = bubble1_k;
        this->bubbles2->getBubbleWithLocalOrderNumber(i)->k = bubble2_k;
        this->result_bubbles->getBubbleWithLocalOrderNumber(i)->k = result_bubble_k;
        if (this->taylor_series_bubbles1) this->taylor_series_bubbles1->getBubbleWithLocalOrderNumber(i)->k = taylor_series_bubble1_k;
        if (this->taylor_series_bubbles2) this->taylor_series_bubbles2->getBubbleWithLocalOrderNumber(i)->k = taylor_series_bubble2_k;
    }
    check_multiplier_errors(__FILE__, __LINE__);
}


Bubbles *BubblesMultiplier::getBubbles1() {
    return this->bubbles1;
}

Bubbles *BubblesMultiplier::getBubbles2() {
    return this->bubbles2;
}

Bubbles *BubblesMultiplier::getResultBubbles() {
    return this->result_bubbles;
}

/*
 * Destroy all cuda related objects owned by this, i.e.,
 * only the arrays
 */ 
void BubblesMultiplier::destroy() {
    
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        cudaSetDevice(device);
        cudaFree(this->device_number_of_terms[device]);
        cudaFree(this->device_positions[device]);
        cudaFree(this->device_result_lm[device]);
        cudaFree(this->device_coefficients[device]);
    }
    
    
    cudaFreeHost(this->device_number_of_terms);
    cudaFreeHost(this->device_result_lm);
    cudaFreeHost(this->device_positions);
    cudaFreeHost(this->device_coefficients);
}


/***********************************************************
 *             The Fortran Interfaces                      *
 ***********************************************************/

extern "C" void bubblesmultiplier_destroy_cuda(BubblesMultiplier *multiplier) {
    multiplier->destroy();
    check_multiplier_errors(__FILE__, __LINE__);
}

extern "C" BubblesMultiplier *bubblesmultiplier_init_cuda(Bubbles *bubbles1, Bubbles *bubbles2, Bubbles *result_bubbles, 
                                                          Bubbles *taylor_series_bubbles1, Bubbles *taylor_series_bubbles2, int lmax,
                                                          double *coefficients, int *number_of_terms, int *result_lm, int *positions,
                                                          int result_lm_size, int processor_order_number,
                                                          int number_of_processors, StreamContainer *streamContainer) {
    BubblesMultiplier *new_multiplier = new BubblesMultiplier(bubbles1, bubbles2, result_bubbles,
                                                              taylor_series_bubbles1, taylor_series_bubbles2, lmax, coefficients, 
                                                              number_of_terms, result_lm, positions, result_lm_size,
                                                              processor_order_number, number_of_processors, streamContainer);
    check_multiplier_errors(__FILE__, __LINE__);
    return new_multiplier;
}

extern "C" void bubblesmultiplier_download_result_cuda(BubblesMultiplier *multiplier, int lmax, int *ibubs, int nbub) {
    multiplier->downloadResult(lmax, ibubs, nbub);
    check_multiplier_errors(__FILE__, __LINE__);
}


extern "C" void bubblesmultiplier_multiply_bubble_cuda(BubblesMultiplier *multiplier, int ibub, double *bubble1_bf,
                                                       double *bubble2_bf, double *result_bubble_bf,
                                                       double *taylor_series_bubble1_bf, double *taylor_series_bubble2_bf, int lmax1, int lmax2, int tlmax1, int tlmax2) {
    
    check_multiplier_errors(__FILE__, __LINE__);
    multiplier->multiplyBubble(ibub, bubble1_bf, bubble2_bf, result_bubble_bf, taylor_series_bubble1_bf, taylor_series_bubble2_bf, lmax1, lmax2, tlmax1, tlmax2);
    check_multiplier_errors(__FILE__, __LINE__);
}

extern "C" void bubblesmultiplier_set_ks(BubblesMultiplier *multiplier, int bubble1_k,
                                        int bubble2_k, int result_bubble_k, int taylor_series_bubble1_k, int taylor_series_bubble2_k) {
    
    multiplier->setK(bubble1_k, bubble2_k, result_bubble_k, taylor_series_bubble1_k, taylor_series_bubble2_k);
    check_multiplier_errors(__FILE__, __LINE__);
}



