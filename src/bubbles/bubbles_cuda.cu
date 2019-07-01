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
/*! @file  bubbles_cuda.cu
 *! @brief CUDA implementation of the Bubbles.
 */
#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
//#include <algorithm> *std::max_element(result_cube, result_cube + totalPointCount)
#include "bubbles_cuda.h"
#include "streamcontainer.h"
#include "grid.h"
#include "spherical_harmonics_cuda.h"
#include "cube.h"
#include "function3d_multiplier.h"
#include "memory_leak_operators.h"
#include "evaluators.h"

#define X_ 0
#define Y_ 1
#define Z_ 2
#define R_ 3
#if (__CUDA_ARCH__ > 350)
#define INJECT_BLOCK_SIZE 256
#else
#define INJECT_BLOCK_SIZE 128
#endif
#define NLIP 7


/** \brief Size of the CUDA blocks in the X dimension */
#define BLOCKDIMX 8
/** \brief Size of the CUDA blocks in the Y dimension */
#define BLOCKDIMY 4
/** \brief Size of the CUDA blocks in the Z dimension */
#define BLOCKDIMZ 4

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

cudaError_t cudastat;
__constant__ int shape_x_, shape_y_, shape_z_, ncell_, nlip_, lmax_, ilmmin_, lmin_, ilmmax_, first_term_, normalization_, ijk_max_;
__constant__ double charge_, r_max_;
cudaStream_t **streams;
int streams_inited = 0;
int allocated = 0;
extern __shared__ double shared_memory[];

__host__ inline void check_memory(const char *filename, const int line_number) {
    
    size_t mem_tot_0 = 0;
    size_t mem_free_0 = 0;
    cudaMemGetInfo  (&mem_free_0, &mem_tot_0);
    printf("Free memory after: %ld, total: %ld\n ", mem_free_0, mem_tot_0);
}





template<typename T>
__device__ __forceinline__ T ldg(const T* ptr) {
#if __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

void cube_download(double *hstPtr, int width, int height ,int depth,
               void *devPtr, size_t pitch) {
//    Define copy "from device to host" parameters
    cudaMemcpy3DParms d2h={0};
    d2h.srcPtr =  make_cudaPitchedPtr(devPtr,
        pitch,width,height);
    d2h.dstPtr = make_cudaPitchedPtr((void *)hstPtr,
        width*sizeof(double),width,height);
    d2h.extent = make_cudaExtent(width * sizeof(double), height,
            depth);
//    cudaMemset3D( d2h.srcPtr, 999, d2h.extent);
    d2h.kind   = cudaMemcpyDeviceToHost;

//    cudastat=cudaMemset3D( d2h.srcPtr, 0, d2h.extent);

//    Copy to host 
    cudastat = cudaMemcpy3D( &d2h );
    check_errors(__FILE__, __LINE__);

    return;
}
void cube_upload(double *hstPtr, int *width ,int *height ,int *depth,
               void *devPtr, size_t pitch) {
//    Define copy "from host to device" parameters
    cudaMemcpy3DParms h2d={0};
    h2d.srcPtr = make_cudaPitchedPtr((void *)hstPtr,
        *width*sizeof(double),*width,*height);
    h2d.dstPtr =  make_cudaPitchedPtr(devPtr,
        pitch,*width,*height);
    h2d.extent = make_cudaExtent(*width * sizeof(double), *height,
            *depth);
    h2d.kind   = cudaMemcpyHostToDevice;

//    Copy to device 
    cudaMemcpy3D( &h2d );
    return;
}

__device__ int icell(double x, double *d, int n){
    if ( ( x > d[n] ) || ( x < d[0] ) ) {
        return -1;
    }

    int i[2];
    i[0]=0;
    i[1]=n;
    int im=(i[0]+i[1])/2;
    int j;
    int max=log((float)n)/log(2.)+1;

    for(j=0;j<max;j++){
        i[ x<d[im] ] = im;
        im=(i[0]+i[1])/2;
    }
    return im;
} 



__device__ void calc_rc(double dist_vec[3], double *dist, double ref[3],double x,
        double y, double z){
    dist_vec[X_]=x-ref[X_];
    dist_vec[Y_]=y-ref[Y_];
    dist_vec[Z_]=z-ref[Z_];
    *dist=sqrt(dist_vec[X_]*dist_vec[X_]+
               dist_vec[Y_]*dist_vec[Y_]+
               dist_vec[Z_]*dist_vec[Z_]);
    dist_vec[X_]/=*dist;
    dist_vec[Y_]/=*dist;
    dist_vec[Z_]/=*dist;
    return;
}

__device__ double eval_lip(int n, double *lip, double *f, double x){
    short i,j;
    double out=0.0;
    for (j=0;j<n;j++){
        double tmp=0.0;
        for (i=0;i<n;i++){
            tmp*= x;
            tmp+= *(lip++);
        }
        out+=tmp*f[j];
    }
    return out;
} 

__device__ double eval_poly(int n, double *c, double x){
    double r=0.0;
    while (n-- > 0) {
        r *= x;
        r += *(c++);
    }
    return r;
}

/*
 * the following function precalculates some common values for the injection.
 * 
 * NOTE: We are setting the cf-array to have 8 * (lmax+1) * (lmax+1) size
 * This has several advantages (even if we are using more space and have 
 * blank spots in the array). 1) Every cell read is coalesced and we don't
 * have overlapping requests! Additionally, we avoid divergence of the threads
 * of one warp in the injection.
 */
__global__ void calc_cf(Bubble *bub, int offset, int number_of_points, size_t device_f_pitch) {
    // get the index within this kernel call
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    // get the global index
    const int id= index + offset;

    const int icell=id%bub->grid->ncell;
    const int ilm=id/bub->grid->ncell;
    const int nlip = bub->grid->nlip;
    __shared__ double shared_lip[49];
    __shared__ double derivative_lip[42];
    __shared__ double lower_derivative_lip[30];
    __shared__ double cf_results[8*64];
    __shared__ double df_results[8*64];
    double f_i;

    // load the Lagrange interpolation polynomials coefficients to
    // the shared memory
    if (threadIdx.x < (nlip) * (nlip)) {
        shared_lip[threadIdx.x] = bub->grid->lip[threadIdx.x];
    }
    if (threadIdx.x < (nlip) * (nlip-1)) {
        derivative_lip[threadIdx.x] = bub->grid->derivative_lip[threadIdx.x];
    }
    if (threadIdx.x < (nlip-2) * (nlip-1)) {
        lower_derivative_lip[threadIdx.x] = bub->grid->lower_derivative_lip[threadIdx.x];
    }
    __syncthreads();


    if ( index < number_of_points && ilm < ((bub->lmax+1)*(bub->lmax+1)) ) {
        double *f  = bub->f + ilm * device_f_pitch / sizeof(double) + (icell * (bub->grid->nlip-1));
        double *cf = bub->cf + ( ilm * bub->grid->ncell + icell ) * 8;
        double *df = bub->df + ( ilm * bub->grid->ncell + icell ) * 8;
        short i,j;
        double one_per_cell_step = 1.0 / bub->grid->h[icell];
        double *lip=&shared_lip[0];
        double *dlip=&derivative_lip[0];
        double *ldlip=&lower_derivative_lip[0];
        

        // set the shared memory result array to zero
        for (i=0; i < 8; i++) {
            cf_results[threadIdx.x * 8 + i]=0.0;
            df_results[threadIdx.x * 8 + i]=0.0;
        }

        // evaluate the cf to shared memory
        for (i=0; i < nlip; i++) {
            f_i = f[i];
            for (j=0; j < nlip ;j++){
                cf_results[threadIdx.x * 8 + j] += f_i* (*(lip++));
            }

            // I (lnw) cannot see any good reason for this special case that is, the
            // derivative at the centre of each bubble should be zero, but why does it have
            // to be enforced?
            const bool ignore_first = true;
            if(ignore_first){
                // handle the special case of the first cell, where the first
                // data item most likely is not valid
                if (icell == 0) {
                    if (i != 0) {
                        for (j = 1 ; j <= nlip-2; j++) {
                            df_results[threadIdx.x * 8 + j] += f_i* (*(ldlip++));
                        }
                    }
                    else {
                        df_results[threadIdx.x * 8] = 0.0;
                    }
                }
                else {
                    for (j=0; j < nlip-1 ;j++) {
                        df_results[threadIdx.x * 8 + j] += f_i* (*(dlip++));
                    }
                }
            }
            else { // no special treatment
                for (j=0; j < nlip-1 ;j++) {
                    df_results[threadIdx.x * 8 + j] += f_i* (*(dlip++));
                }
            }

        }
        // copy the result to device memory
        for (i=0; i < 8; i++) {
            cf[i] = cf_results[threadIdx.x * 8 + i];
            df[i] = one_per_cell_step * df_results[threadIdx.x * 8 + i];
        }

    }
    return;
}




__device__ inline double evaluate_polynomials(int n, const double* __restrict__ c, const double x){
    double result=0.0;
    while (n-- > 0) {
        result *= x;
        result += *(c++);
    }
    return result;
}

//#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 350
/*
 *  Evaluates one granular polynomial for coefficients, and x
 * NOTE: each thread is different value for coefficient, when entering the function
 * NOTE: each x value must be the same for 8 consecutive threads
 * NOTE: upon return each thread has the same value.
 */
__inline__ __device__
double evaluate_polynomials_unit_shuffle(double coefficient, const double x) {
    double result = coefficient;
    for (int i = 1; i < 7; i++) {
        result *= x;
        result += __shfl_down(coefficient, i, 8);
    }
    return result;
}

__inline__ __device__
double evaluate_polynomials_unit_register(const double * __restrict__ coefficients, const double x, int nlip) {
    double result = 0.0;
    while (nlip-- > 0) {
        result *= x;
        result += *(coefficients++);
    }

    return result;
}

__device__ inline void horizontal_rotate_8f(double coefficients[8], unsigned int order_number) {
    coefficients[1] = __shfl(coefficients[1], (order_number+1)%8, 8);
    coefficients[2] = __shfl(coefficients[2], (order_number+2)%8, 8);
    coefficients[3] = __shfl(coefficients[3], (order_number+3)%8, 8);
    coefficients[4] = __shfl(coefficients[4], (order_number+4)%8, 8);
    coefficients[5] = __shfl(coefficients[5], (order_number+5)%8, 8);
    coefficients[6] = __shfl(coefficients[6], (order_number+6)%8, 8);
    coefficients[7] = __shfl(coefficients[7], (order_number+7)%8, 8);
}

__device__ inline void horizontal_rotate_8b(double coefficients[8], unsigned int order_number) {
    coefficients[1] = __shfl(coefficients[1], (order_number+7)%8, 8);
    coefficients[2] = __shfl(coefficients[2], (order_number+6)%8, 8);
    coefficients[3] = __shfl(coefficients[3], (order_number+5)%8, 8);
    coefficients[4] = __shfl(coefficients[4], (order_number+4)%8, 8);
    coefficients[5] = __shfl(coefficients[5], (order_number+3)%8, 8);
    coefficients[6] = __shfl(coefficients[6], (order_number+2)%8, 8);
    coefficients[7] = __shfl(coefficients[7], (order_number+1)%8, 8);
}

__device__ inline void vertical_rotate_8(double src[8], unsigned int order_number) {
    double tmp = src[0];
    src[0] = (order_number == 1) ? src[7] : src[0];
    src[7] = (order_number == 1) ? src[6] : src[7];
    src[6] = (order_number == 1) ? src[5] : src[6];
    src[5] = (order_number == 1) ? src[4] : src[5];
    src[4] = (order_number == 1) ? src[3] : src[4];
    src[3] = (order_number == 1) ? src[2] : src[3];
    src[2] = (order_number == 1) ? src[1] : src[2];
    src[1] = (order_number == 1) ? tmp    : src[1];
   
    src[1] = (order_number == 2) ? src[7] : src[1];
    src[0] = (order_number == 2) ? src[6] : src[0];
    src[7] = (order_number == 2) ? src[5] : src[7];
    src[6] = (order_number == 2) ? src[4] : src[6];
    src[5] = (order_number == 2) ? src[3] : src[5];
    src[4] = (order_number == 2) ? src[2] : src[4];
    src[3] = (order_number == 2) ? src[1] : src[3];
    src[2] = (order_number == 2) ? tmp    : src[2];
   
    src[2] = (order_number == 3) ? src[7] : src[2];
    src[1] = (order_number == 3) ? src[6] : src[1];
    src[0] = (order_number == 3) ? src[5] : src[0];
    src[7] = (order_number == 3) ? src[4] : src[7];
    src[6] = (order_number == 3) ? src[3] : src[6];
    src[5] = (order_number == 3) ? src[2] : src[5];
    src[4] = (order_number == 3) ? src[1] : src[4];
    src[3] = (order_number == 2) ? tmp    : src[3];
   
    src[3] = (order_number == 4) ? src[7] : src[3];
    src[2] = (order_number == 4) ? src[6] : src[2];
    src[1] = (order_number == 4) ? src[5] : src[1];
    src[0] = (order_number == 4) ? src[4] : src[0];
    src[7] = (order_number == 4) ? src[3] : src[7];
    src[6] = (order_number == 4) ? src[2] : src[6];
    src[5] = (order_number == 4) ? src[1] : src[5];
    src[4] = (order_number == 4) ? tmp    : src[4];
   
    src[4] = (order_number == 5) ? src[7] : src[4];
    src[3] = (order_number == 5) ? src[6] : src[3];
    src[2] = (order_number == 5) ? src[5] : src[2];
    src[1] = (order_number == 5) ? src[4] : src[1];
    src[0] = (order_number == 5) ? src[3] : src[0];
    src[7] = (order_number == 5) ? src[2] : src[7];
    src[6] = (order_number == 5) ? src[1] : src[6];
    src[5] = (order_number == 5) ? tmp    : src[5];
   
    src[5] = (order_number == 6) ? src[7] : src[5];
    src[4] = (order_number == 6) ? src[6] : src[4];
    src[3] = (order_number == 6) ? src[5] : src[3];
    src[2] = (order_number == 6) ? src[4] : src[2];
    src[1] = (order_number == 6) ? src[3] : src[1];
    src[0] = (order_number == 6) ? src[2] : src[0];
    src[7] = (order_number == 6) ? src[1] : src[7];
    src[6] = (order_number == 6) ? tmp    : src[6];
   
    src[6] = (order_number == 7) ? src[7] : src[6];
    src[5] = (order_number == 7) ? src[6] : src[5];
    src[4] = (order_number == 7) ? src[5] : src[4];
    src[3] = (order_number == 7) ? src[4] : src[3];
    src[2] = (order_number == 7) ? src[3] : src[2];
    src[1] = (order_number == 7) ? src[2] : src[1];
    src[0] = (order_number == 7) ? src[1] : src[0];
    src[7] = (order_number == 7) ? tmp    : src[7];
}


__device__ inline void transpose8(double coefficients[8], int order_number) {

    //printf("Original coefficients %d: %f, %f, %f, %f, %f, %f, %f, %f\n", order_number, coefficients[0], coefficients[1], coefficients[2], coefficients[3], coefficients[4], coefficients[5], coefficients[6], coefficients[7]);
    horizontal_rotate_8f(coefficients, order_number);
    vertical_rotate_8(coefficients, order_number);
    horizontal_rotate_8b(coefficients, order_number);
    //printf("Transposed coefficients coefficients %d: %f, %f, %f, %f, %f, %f, %f, %f\n", order_number, coefficients[0], coefficients[1], coefficients[2], coefficients[3], coefficients[4], coefficients[5], coefficients[6], coefficients[7]);
}


/*
 *  Evaluates the polynomials using shuffle actions. This saves the shared_memory significantly and allows
 *  the increase of the occupancy of the devices.
 *
 *  This function only needs blockDim.x * 8 bytes of shared memory. This allows the usage of any sized blocks
 *  that are practically useful.
 *
 *  The number of arithmetic operations is larger than for the version using shared memory only, and thus
 *  the effect to the execution speed remains to be seen.
 */
__device__ inline double evaluate_polynomials_shuffle(const int address,
                                                      const double * __restrict__ c,
                                                      const double x,
                                                      const int nlip) {
    double *result = &shared_memory[0];
    //double coefficients[8];
    //double res;
   
    int remainder =  threadIdx.x%8;
    int base_address = 8*(threadIdx.x/8);
    double res;
   
    for (int i = 0; i < 8; i ++) {
        // evaluate the polynomials
        // NOTE: __shfl(address, i, width=8)  gets the address needed by the thread i/8 in the thread group
        // NOTE: __shfl(x, i, width = 8) gets the coordinate x of the thread i/8 in the thread group
        // NOTE: the c access (global memory is coalesced),
        // NOTE: shared memorybank conflict should not occur, as every thread in the 8 thread group access
        //       the same address, thus resulting in broadcast.
        //coefficients[i] = c[__shfl(address, i, 8) + remainder];
         res  = evaluate_polynomials_unit_shuffle(
                                    c[__shfl(address, i, 8) + remainder],
                                    __shfl(x, i, 8));
        if (remainder == 0) result[base_address + i] = res;
    }
   
   
    // swap the coefficients to be with their rightful owners
    //transpose8(coefficients, remainder);
    return result[threadIdx.x];
    //return evaluate_polynomials_unit_register(coefficients, x, nlip);
}
#endif
//#endif

/*
 * Get the thread-id within block.
 */
__device__ inline int getThreadId() {
    return   threadIdx.x 
           + blockDim.x * threadIdx.y
           + blockDim.x * blockDim.y * threadIdx.z;
}

/*
 * @param c, bubbles coefficients in the global memory
 * @param x, the coordinate of the point in cell coordinates
 * 
 * NOTE: The parameter 'c' must be pitched for this function to be useful
 * NOTE: This function is made for NLIP:7, with other nlip values, the function must be remade
 */

template<int nlip>
__device__ inline
double evaluate_polynomials_shared(const int address, const double* __restrict__ c, const double x) {

    double *coefficients = &shared_memory[0];
    //const float *fc = (const float *)c;
    int threadId = getThreadId();
    
    int remainder =  threadId%8;
    int base_address = 8*(threadId/8);
    int id = base_address * 7 + remainder;
    /*int remainder =  threadId%16;
    base_address = 16*(threadId/16);
    id = base_address * 16 + remainder;
    int faddress = 2 * address;*/

#if (__CUDA_ARCH__ >= 350)
    // read the coefficients in the shared memory, 8 threads 
    // neighbouring each other are reading the global memory
    // coefficients for one thread at the time, starting from 0
    // and going to 7
    
    int address_7 = __shfl(address, 7, 8);
    
    if (remainder < 7) {
        coefficients[id]        = ldg<double>(&c[__shfl(address, 0, 8)  + remainder]);
        coefficients[id+7]      = ldg<double>(&c[__shfl(address, 1, 8)  + remainder]);
        coefficients[id+7*2]    = ldg<double>(&c[__shfl(address, 2, 8)  + remainder]);
        coefficients[id+7*3]    = ldg<double>(&c[__shfl(address, 3, 8)  + remainder]);
        coefficients[id+7*4]    = ldg<double>(&c[__shfl(address, 4, 8)  + remainder]);
        coefficients[id+7*5]    = ldg<double>(&c[__shfl(address, 5, 8)  + remainder]);
        coefficients[id+7*6]    = ldg<double>(&c[__shfl(address, 6, 8)  + remainder]);
        coefficients[id+7*7]    = ldg<double>(&c[address_7  + remainder]);
    }
    
    /*coefficients[id]    = c[__shfl(address, 0, 8)  + remainder];
    coefficients[id+8]  = c[__shfl(address, 1, 8)  + remainder];
    coefficients[id+16] = c[__shfl(address, 2, 8)  + remainder];
    coefficients[id+24] = c[__shfl(address, 3, 8)  + remainder];
    coefficients[id+32] = c[__shfl(address, 4, 8)  + remainder];
    coefficients[id+40] = c[__shfl(address, 5, 8)  + remainder];
    coefficients[id+48] = c[__shfl(address, 6, 8)  + remainder];
    coefficients[id+56] = c[__shfl(address, 7, 8)  + remainder];*/
    
    /*fcoefficients[id]     = fc[__shfl(faddress, 0, 16)  + remainder];
    fcoefficients[id+16]  = fc[__shfl(faddress, 1, 16)  + remainder];
    fcoefficients[id+32]  = fc[__shfl(faddress, 2, 16)  + remainder];
    fcoefficients[id+48]  = fc[__shfl(faddress, 3, 16)  + remainder];
    fcoefficients[id+64]  = fc[__shfl(faddress, 4, 16)  + remainder];
    fcoefficients[id+80]  = fc[__shfl(faddress, 5, 16)  + remainder];
    fcoefficients[id+96]  = fc[__shfl(faddress, 6, 16)  + remainder];
    fcoefficients[id+112] = fc[__shfl(faddress, 7, 16)  + remainder];
    fcoefficients[id+128] = fc[__shfl(faddress, 8, 16)  + remainder];
    fcoefficients[id+144] = fc[__shfl(faddress, 9, 16)  + remainder];
    fcoefficients[id+160] = fc[__shfl(faddress, 10, 16)  + remainder];
    fcoefficients[id+176] = fc[__shfl(faddress, 11, 16)  + remainder];
    fcoefficients[id+192] = fc[__shfl(faddress, 12, 16)  + remainder];
    fcoefficients[id+208] = fc[__shfl(faddress, 13, 16)  + remainder];
    fcoefficients[id+224] = fc[__shfl(faddress, 14, 16)  + remainder];
    fcoefficients[id+240] = fc[__shfl(faddress, 15, 16)  + remainder];*/
    
#else
    // store the addresses to the shared memory
    int *address_array = (int *) &shared_memory[8*blockDim.x * blockDim.y * blockDim.z];
    address_array[threadIdx.x] = address;
    
    coefficients[id]    = c[address_array[base_address]     + remainder];
    coefficients[id+8]  = c[address_array[base_address +1]  + remainder];
    coefficients[id+16] = c[address_array[base_address +2]  + remainder];
    coefficients[id+24] = c[address_array[base_address +3]  + remainder];
    coefficients[id+32] = c[address_array[base_address +4]  + remainder];
    coefficients[id+40] = c[address_array[base_address +5]  + remainder];
    coefficients[id+48] = c[address_array[base_address +6]  + remainder];
    coefficients[id+56] = c[address_array[base_address +7]  + remainder];
#endif
    
    double *coeff = &coefficients[threadId * 7];
    double result = coeff[0];
    
    if (nlip > 1) {
        result *= x;
        result += coeff[1];
    }
    if (nlip > 2) {
        result *= x;
        result += coeff[2];
    }
    if (nlip > 3) {
        result *= x;
        result += coeff[3];
    }
    if (nlip > 4) {
        result *= x;
        result += coeff[4];
    }
    
    if (nlip > 5) {
        result *= x;
        result += coeff[5];
    }
    
    if (nlip > 6) {
        result *= x;
        result += coeff[6];
    }

    return result;
}


__device__ inline int calculate_icell(double x, double *d, int n){
    if ( ( x > d[n] ) || ( x < d[0] ) ) {
        return -1;
    }

    int i[2];
    i[0]=0;
    i[1]=n;
    int im=(i[0]+i[1])/2;
    int j;
    int max=log((float)n)/log(2.)+1;

    for(j=0;j<max;j++){
        i[ x<d[im] ] = im;
        im=(i[0]+i[1])/2;
    }
    return im;
} 

__device__ inline void calculate_icell_radial(const double x, const double charge, const double r_max,
                                             const int ncell, const int nlip, 
                                             int *icell, double *in_cell_position) {
    const double dx = r_max/(double)ncell;
    const double c=8.0*rsqrt(charge)/charge;
    const double a = r_max + c;
    *icell = (int)(x * a / ((c + x)*dx));
    double x1 = c / (a/((*icell+1) * dx) - 1.0);
    double x0 = c / (a/(*icell * dx)     - 1.0);
    if (icell == 0) {
        x0 = 0.0;
    }
    double grid_step = (x1-x0) / (nlip-1);
    double center    = (x1+x0) / (2.0);
    *in_cell_position= (x - center)/grid_step;
   
}

inline __device__ void calculate_distance(double &dist_vec_x, double &dist_vec_y, double &dist_vec_z, double &dist,
                              const double reference_point_x, const double reference_point_y, const double reference_point_z, 
                              const double x, const double y, const double z){
    // calculate the vector relative to reference_point
    dist_vec_x=x-reference_point_x;
    dist_vec_y=y-reference_point_y;
    dist_vec_z=z-reference_point_z;
    
    // evaluate the length of the dist_vector, i.e., the distance between dist_vec and reference_point
    dist=sqrt(dist_vec_x * dist_vec_x +
              dist_vec_y * dist_vec_y +
              dist_vec_z * dist_vec_z);
    return;
}


/* 
 * Evaluates value of single bubble at a point. This is very similar with the
 * SolidHarmonics simple evaluation, but the results are multiplied with the
 * polynomial evaluations
 */

__device__ inline double Bubbles_evaluate_point_lmin(
                                         // x-coordinate relative to the center of the bubble
                                         const double &x,
                                         // y-coordinate relative to the center of the bubble
                                         const double &y,
                                         // z-coordinate relative to the center of the bubble
                                         const double &z,
                                         // relative distance from the center of the bubble
                                         const double &distance,
                                         // minimum quantum number 'l'
                                         const int &lmin,
                                         // maximum quantum number 'l'
                                         const int &lmax,
                                         // number of cells
                                         const int &ncell,
                                         // number of lagrange integration polyniomials per 
                                         // cell, i.e., the number of grid points per cell
                                         const int &nlip,
                                         // position inside the cell
                                         const double &r,
                                         // k value for the bubble
                                         const int &k,
                                         // the first address value in bubble for the selected cell
                                         const int &address,
                                         const double* __restrict__ cf
                                        ) {
    
    double result = 0.0;
    int lm_address = address, address2 = address;
    // NOTE: Here the nlip is replaced by 8 because this gives advantages in loading the stuff
    // also *cf this should be done
    const int ncell_nlip = ncell * 8;
    int l, m, l2; 
    double top = 0.0, bottom = 0.0, new_bottom = 0.0, prev1 = 0.0, prev2 = 0.0, current = 0.0;
    double multiplier = 0.0, multiplier2 = 0.0, one_per_r = 1.0 / distance;
    double r2 = x*x+y*y+z*z;
    l = 0;
    // set value for l=0, m=0
    if (lmin == 0) {
        //printf("x: %f, y: %f, z: %f, nlip: %d, ncell: %d, l: 0, address: %d, cf: %ld, r: %f\n", x, y, z, nlip, ncell, 0, lm_address, cf, r);
        //printf("shared_memory address: %ld\n");
        //printf("shared memory first value: %f", shared_memory[0]);
        result = evaluate_polynomials_shared<NLIP>(lm_address, cf, r);
    }
    
    
    if (lmax >= 1) { 
        l = 1;
        multiplier = one_per_r;
        // set value for l=1, m=-1
        lm_address += ncell_nlip;
        if (lmin <= 1) {
            result += y * evaluate_polynomials_shared<NLIP>(lm_address, cf, r) * multiplier;
        }
        
        // set all values where m=-1
        m = -1;
        prev1 = y;
        // the starting address has 1 item before from the l=0, 3 from l=1, and 1 from l=2
        address2 = address + ncell_nlip * 5;
        multiplier2 = multiplier * one_per_r;
        for (l = 2; l <= lmax; l++) {
            current =   ( 2.0*(double)l-1.0) * rsqrt( 1.0*(double)((l+m)*(l-m)) ) * z*prev1;
            if (l > 2) {
                current -=  sqrt( (double)((l+m-1)*(l-m-1)) /  (double)((l+m)*(l-m)) ) * r2 * prev2;
            }
            prev2 = prev1;
            prev1 = current;
            if (l >= lmin) {
                result += current * evaluate_polynomials_shared<NLIP>(address2, cf, r) * multiplier2;
            }
            
            // add the address2 to get to the next item with m=-1
            address2 += ncell_nlip * (2*l+2);
            multiplier2 *= one_per_r;
        }
        
        // set value for l=1, m=0
        lm_address += ncell_nlip;
        if (lmin <= 1) {
            result += z * evaluate_polynomials_shared<NLIP>(lm_address, cf, r) * multiplier;
        }
        
        // set all values where m=0
        prev1 = z;
        prev2 = 1.0;
        m = 0;
        // the starting address has 1 item before from the l=0, 3 from l=1, and 2 from l=2
        address2 = address + ncell_nlip * 6;
        multiplier2 = multiplier * one_per_r;
        for (l = 2; l <= lmax; l++) {
            current =   ( 2.0*(double)l-1.0) * rsqrt( 1.0*(double)((l+m)*(l-m)) ) * z * prev1;
            current -=  sqrt( (double)((l+m-1)*(l-m-1)) /  (double)((l+m)*(l-m)) ) * r2 * prev2;
            prev2 = prev1;
            prev1 = current;
            if (l >= lmin) {
                result += current * evaluate_polynomials_shared<NLIP>(address2, cf, r) * multiplier2;
            }
            
            // add the address2 to get to the next item with m=0 
            address2 += ncell_nlip * (2*l+2);
            multiplier2 *= one_per_r;
        }
        
        // set value for l=1, m=1
        lm_address += ncell_nlip;
        if (lmin <= 1) {
            result += x * evaluate_polynomials_shared<NLIP>(lm_address, cf, r) * multiplier;
        }
        // set all values where m=1
        prev1 = x;
        m = 1;
        // the starting address has 1 item before from the l=0, 3 from l=1, and 3 from l=2
        address2 = address + ncell_nlip * 7;
        multiplier2 = multiplier * one_per_r;
        for (l = 2; l <= lmax; l++) {
            current =   ( 2.0*(double)l-1.0) * rsqrt( 1.0*(double)((l+m)*(l-m)) ) * z*prev1;
            if (l > 2) {
                current -=  sqrt( (double)((l+m-1)*(l-m-1)) /  (double)((l+m)*(l-m)) ) * r2 * prev2;
            }
            prev2 = prev1;
            prev1 = current;
            
            if (l >= lmin) {
                result += current * evaluate_polynomials_shared<NLIP>(address2, cf, r) * multiplier2;
            }
            
            // add the address2 to get to the next item with m=1 
            address2 += ncell_nlip * (2*l+2);
            multiplier2 *= one_per_r;
        }
        
        // go through the rest of the stuff
        bottom = y; // bottom refers to solid harmonics value with l=l-1 and m=-(l-1)
        top = x;    // top    refers to solid harmonics value with l=l-1 and m=l-1
        lm_address += ncell_nlip;
        multiplier *= one_per_r;
        for (l=2; l <= lmax; l++) {
            
            new_bottom = sqrt((2.0*(double)l - 1.0) / (2.0*(double)l)) * 
                        ( y*top + x*bottom);
                        
            if (l >= lmin) {
                result += new_bottom * evaluate_polynomials_shared<NLIP>(lm_address, cf, r) * multiplier;
            }
            
            // set all values where m=-l
            m = -l;
            prev1 = new_bottom;
            address2 = lm_address + (2*l+2) * ncell_nlip;
            multiplier2 = multiplier * one_per_r;
            for (l2 = l+1; l2 <= lmax; l2++) {
                current =   ( 2.0*(double)l2-1.0) * rsqrt( 1.0*(double)((l2+m)*(l2-m)) ) * z*prev1;
                if (l2 > l+1) {
                    current -=  sqrt( (double)((l2+m-1)*(l2-m-1)) /  (double)((l2+m)*(l2-m)) ) * r2 * prev2;
                }
                prev2 = prev1;
                prev1 = current;
                
                if (l2 >= lmin) {
                    result += current * evaluate_polynomials_shared<NLIP>(address2, cf, r) * multiplier2;
                }
                // add the address2 to get to the next item with m=l
                address2 += ncell_nlip * (2*l2+2);
                multiplier2 *= one_per_r;
            }
            
            
            // get value for l=l, m=l. The address is 2*l items away from l=l, m=-l
            lm_address += 2*l * ncell_nlip;
            top = sqrt((2.0*(double)l - 1.0) / (2.0*(double)l)) * 
                        ( x*top-y*bottom );
            // set all values where m=l
            m = l;
            prev1 = top;
            address2 = lm_address + (2*l+2) * ncell_nlip;
            multiplier2 = multiplier * one_per_r;
            for (l2 = l+1; l2 <= lmax; l2++) {
                current =   ( 2.0*(double)l2-1.0) * rsqrt( 1.0*(double)((l2+m)*(l2-m)) ) * z*prev1;
                if (l2 > l+1) {
                    current -=  sqrt( (double)((l2+m-1)*(l2-m-1)) /  (double)((l2+m)*(l2-m)) ) * r2 * prev2;
                }
                prev2 = prev1;
                prev1 = current;
                
                if (l2 >= lmin) {
                    result += current * evaluate_polynomials_shared<NLIP>(address2, cf, r) * multiplier2;
                }
                // add the address2 to get to the next item with m=l
                address2 += ncell_nlip * (2*l2+2);
                multiplier2 *= one_per_r;
            }
            // store the new bottom: l=l, m=-l (we need the old bottom in calculation of top)
            bottom = new_bottom;
            if (l >= lmin) {
                result += top * evaluate_polynomials_shared<NLIP>(lm_address, cf, r) * multiplier;
            }
            // get next address
            lm_address += ncell_nlip;
            multiplier *= one_per_r;
        }
    }
    // multiply the result with r^k, if k is not 0
    // the distance is not too close to 0.0 as this is checked
    // earlier in this function
    if (k != 0 && distance > 1e-12) {
        result *= pow(distance, (double)k);
    }
    
    if (distance < 1e-8) {
        result = 1.0 * cf[0]; //evaluate_polynomials(nlip, &cf[address], r);
    }
    return result;
}



/*
 * (int nlip, int ncell, int l, int address, double *c, const double x)
 * Evaluates the value of gradient of a single bubble at a point. This is very similar with the
 * SolidHarmonics simple evaluation, but the results are multiplied with the
 * polynomial evaluations and summed together.
 */

template <bool evaluate_gradients_x, bool evaluate_gradients_y, bool evaluate_gradients_z >
__device__ inline void Bubbles_evaluate_gradient_point(
                                         // x-coordinate relative to the center of the bubble
                                         const double &x,
                                         // y-coordinate relative to the center of the bubble
                                         const double &y,
                                         // z-coordinate relative to the center of the bubble
                                         const double &z,
                                         // relative distance from the center of the bubble
                                         const double &distance,
                                         // maximum quantum number 'l'
                                         const int &lmax,
                                         // number of cells
                                         const int &ncell,
                                         // number of lagrange integration polyniomials per 
                                         // cell, i.e., the number of grid points per cell
                                         const int &nlip,
                                         // position inside the cell
                                         const double &r,
                                         // k value for the bubble
                                         const int &k,
                                         // the first address value in bubble for the selected cell
                                         const int &address,
                                         // constant pointer to a variable double array
                                         const double* __restrict__ cf,
                                         // constant pointer to a derivative variable double array
                                         const double* __restrict__ df,
                                         // if only the l = 0 is evaluated
                                         const bool only_spherical,
                                         // result
                                         double result[3]
                                        ) {
    
    int lm_address = address, address2;
    // NOTE: Here the nlip is replaced by 8 because this gives advantages in loading the stuff
    // also *cf this should be done
    const int ncell_nlip = ncell * 8;
    int l, l2; 
    double top, bottom, new_bottom, prev1, prev2, current, current_gradient[3], prev1_gradient[3], prev2_gradient[3], bottom_gradient[3], new_bottom_gradient, top_gradient[3];
    double one_per_r = 1.0 / distance;;
    double one_per_r_gradient[3] = {(-x) * one_per_r * one_per_r,
                                    (-y) * one_per_r * one_per_r,
                                    (-z) * one_per_r * one_per_r};
    l = 0;
    
    // set value for l=0, m=0
    
    double radial_value, radial_derivative;
    radial_derivative = evaluate_polynomials_shared<NLIP-1>(lm_address, df, r);
    if (evaluate_gradients_x) result[X_] =  radial_derivative * x;// * one_per_r;
    if (evaluate_gradients_y) result[Y_] =  radial_derivative * y;// * one_per_r;
    if (evaluate_gradients_z) result[Z_] =  radial_derivative * z;// * one_per_r;
    
    if (distance >= 0.0 && distance < 1e-12) {
        one_per_r = 0.0;
        if (evaluate_gradients_x) one_per_r_gradient[X_] = 0.0;
        if (evaluate_gradients_y) one_per_r_gradient[Y_] = 0.0;
        if (evaluate_gradients_z) one_per_r_gradient[Z_] = 0.0;
        if (evaluate_gradients_x) result[X_] = 0.0; //radial_derivative;
        if (evaluate_gradients_y) result[Y_] = 0.0; //radial_derivative;
        if (evaluate_gradients_z) result[Z_] = 0.0;//radial_derivative;
    }
    /*if (only_spherical) {
        one_per_r = 0.0;
        if (evaluate_gradients_x) one_per_r_gradient[X_] = 0.0;
        if (evaluate_gradients_y) one_per_r_gradient[Y_] = 0.0;
        if (evaluate_gradients_z) one_per_r_gradient[Z_] = 0.0;
    }*/
    if (lmax >= 1) {    
        // set all values where m=-1
        prev1 = y * one_per_r;
        if (evaluate_gradients_x) prev1_gradient[X_] = one_per_r_gradient[X_] * y;
        if (evaluate_gradients_y) prev1_gradient[Y_] = 1.0 + one_per_r_gradient[Y_] * y;
        if (evaluate_gradients_z) prev1_gradient[Z_] = one_per_r_gradient[Z_] * y;
        
        // set value for l=1, m=-1
        radial_value = evaluate_polynomials_shared<NLIP>(address+ncell_nlip, cf, r);
        radial_derivative = evaluate_polynomials_shared<NLIP-1>(address+ncell_nlip, df, r); 
        if (evaluate_gradients_x) result[X_] += radial_value * prev1_gradient[X_] + radial_derivative * prev1  * x;// * one_per_r;
        if (evaluate_gradients_y) result[Y_] += radial_value * prev1_gradient[Y_] + radial_derivative * prev1  * y;// * one_per_r;
        if (evaluate_gradients_z) result[Z_] += radial_value * prev1_gradient[Z_] + radial_derivative * prev1  * z;// * one_per_r;
        
        //if (only_spherical) printf("radial_value: %e, radial_derivative: %e, prev1, i.e., y/r: %e\n", radial_value, radial_derivative, prev1);
        //if (only_spherical && evaluate_gradients_x) printf("prev1-gradient-x: %e, x/r: %e\n", prev1_gradient[X_], x * one_per_r);
        //if (only_spherical && evaluate_gradients_y) printf("prev1-gradient-y: %e, y/r: %e\n", prev1_gradient[Y_], y * one_per_r);
        //if (only_spherical && evaluate_gradients_z) printf("prev1-gradient-z: %e, z/r: %e\n", prev1_gradient[Z_], z * one_per_r);
         
        // the starting address has 1 item before from the l=0, 3 from l=1, and 1 from l=2
        address2 = address + ncell_nlip * 5;
        for (l = 2; l <= lmax; l++) {
            double a = ( 2.0*(double)l-1.0) * rsqrt( 1.0*(double)((l-1)*(l+1)) );
            current =   a * z*prev1 * one_per_r;
            if (evaluate_gradients_x) current_gradient[X_] = a *(z * prev1 * one_per_r_gradient[X_] + z * one_per_r * prev1_gradient[X_]);
            if (evaluate_gradients_y) current_gradient[Y_] = a *(z * prev1 * one_per_r_gradient[Y_] + z * one_per_r * prev1_gradient[Y_]);
            if (evaluate_gradients_z) current_gradient[Z_] = a *(z * prev1 * one_per_r_gradient[Z_] + prev1 + z * one_per_r * prev1_gradient[Z_]);
            if (l > 2) {
                double b = sqrt( (double)((l-2)*(l)) /  (double)((l-1)*(l+1)) );
                current -=  b * prev2;
                if (evaluate_gradients_x) current_gradient[X_] -= b * prev2_gradient[X_];
                if (evaluate_gradients_y) current_gradient[Y_] -= b * prev2_gradient[Y_];
                if (evaluate_gradients_z) current_gradient[Z_] -= b * prev2_gradient[Z_];
            }
            radial_value = evaluate_polynomials_shared<NLIP>(address2, cf, r);
            radial_derivative = evaluate_polynomials_shared<NLIP-1>(address2, df, r);
            if (evaluate_gradients_x) result[X_] += radial_value * current_gradient[X_] + radial_derivative * current * x;// * one_per_r;
            if (evaluate_gradients_y) result[Y_] += radial_value * current_gradient[Y_] + radial_derivative * current * y;// * one_per_r;
            if (evaluate_gradients_z) result[Z_] += radial_value * current_gradient[Z_] + radial_derivative * current * z;// * one_per_r;
            prev2 = prev1;
            if (evaluate_gradients_x) prev2_gradient[X_] = prev1_gradient[X_];
            if (evaluate_gradients_y) prev2_gradient[Y_] = prev1_gradient[Y_];
            if (evaluate_gradients_z) prev2_gradient[Z_] = prev1_gradient[Z_];
            prev1 = current;
            if (evaluate_gradients_x) prev1_gradient[X_] = current_gradient[X_];
            if (evaluate_gradients_y) prev1_gradient[Y_] = current_gradient[Y_];
            if (evaluate_gradients_z) prev1_gradient[Z_] = current_gradient[Z_];
            
            // add the address2 to get to the next item with m=-1
            address2 += ncell_nlip * (2*l+2);
        }
        
        prev2 = 1.0;
        if (evaluate_gradients_x) prev2_gradient[X_] = 0.0;
        if (evaluate_gradients_y) prev2_gradient[Y_] = 0.0;
        if (evaluate_gradients_z) prev2_gradient[Z_] = 0.0;
        
        // set all values where m=0
        prev1 = z * one_per_r;
        if (evaluate_gradients_x) prev1_gradient[X_] = one_per_r_gradient[X_] * z;
        if (evaluate_gradients_y) prev1_gradient[Y_] = one_per_r_gradient[Y_] * z;
        if (evaluate_gradients_z) prev1_gradient[Z_] = 1.0  + one_per_r_gradient[Z_] * z;
        
        // set value for l=1, m=0
        radial_value = evaluate_polynomials_shared<NLIP>(address+2*ncell_nlip, cf, r);
        radial_derivative = evaluate_polynomials_shared<NLIP-1>(address+2*ncell_nlip, df, r);
        
        if (evaluate_gradients_x) result[X_] += radial_value * prev1_gradient[X_] + radial_derivative * prev1  * x;// * one_per_r;
        if (evaluate_gradients_y) result[Y_] += radial_value * prev1_gradient[Y_] + radial_derivative * prev1  * y;// * one_per_r;
        if (evaluate_gradients_z) result[Z_] += radial_value * prev1_gradient[Z_] + radial_derivative * prev1  * z;// * one_per_r;
        
        //if (only_spherical) printf("radial_value: %e, radial_derivative: %e, prev1, i.e., z/r: %e\n", radial_value, radial_derivative, prev1);
        //if (only_spherical && evaluate_gradients_x) printf("prev1-gradient-x: %e, x/r: %e\n", prev1_gradient[X_], x * one_per_r);
        //if (only_spherical && evaluate_gradients_y) printf("prev1-gradient-y: %e, y/r: %e\n", prev1_gradient[Y_], y * one_per_r);
        //if (only_spherical && evaluate_gradients_z) printf("prev1-gradient-z: %e, z/r: %e\n", prev1_gradient[Z_], z * one_per_r);
          
        
        
        // the starting address has 1 item before from the l=0, 3 from l=1, and 2 from l=2
        address2 = address + ncell_nlip * 6;
        for (l = 2; l <= lmax; l++) {
            double a = ( 2.0*(double)l-1.0) * rsqrt( 1.0*(double)((l)*(l)) );
            double b = sqrt( (double)((l-1)*(l-1)) /  (double)((l)*(l)) );
            current =   a * z * prev1 * one_per_r;
            if (evaluate_gradients_x) current_gradient[X_] = a *(z * prev1 * one_per_r_gradient[X_] + z * one_per_r * prev1_gradient[X_]);
            if (evaluate_gradients_y) current_gradient[Y_] = a *(z * prev1 * one_per_r_gradient[Y_] + z * one_per_r * prev1_gradient[Y_]);
            if (evaluate_gradients_z) current_gradient[Z_] = a *(z * prev1 * one_per_r_gradient[Z_] + prev1  + z * one_per_r * prev1_gradient[Z_]);
            current -=  b * prev2;
            if (evaluate_gradients_x) current_gradient[X_] -= b * prev2_gradient[X_];
            if (evaluate_gradients_y) current_gradient[Y_] -= b * prev2_gradient[Y_];
            if (evaluate_gradients_z) current_gradient[Z_] -= b * prev2_gradient[Z_];
                        
            radial_value = evaluate_polynomials_shared<NLIP>(address2, cf, r);
            radial_derivative = evaluate_polynomials_shared<NLIP-1>(address2, df, r);
            if (evaluate_gradients_x) result[X_] += radial_value * current_gradient[X_] + radial_derivative * current * x;// * one_per_r;
            if (evaluate_gradients_y) result[Y_] += radial_value * current_gradient[Y_] + radial_derivative * current * y;// * one_per_r;
            if (evaluate_gradients_z) result[Z_] += radial_value * current_gradient[Z_] + radial_derivative * current * z;// * one_per_r;
            prev2 = prev1;
            if (evaluate_gradients_x) prev2_gradient[X_] = prev1_gradient[X_];
            if (evaluate_gradients_y) prev2_gradient[Y_] = prev1_gradient[Y_];
            if (evaluate_gradients_z) prev2_gradient[Z_] = prev1_gradient[Z_];
            prev1 = current;
            if (evaluate_gradients_x) prev1_gradient[X_] = current_gradient[X_];
            if (evaluate_gradients_y) prev1_gradient[Y_] = current_gradient[Y_];
            if (evaluate_gradients_z) prev1_gradient[Z_] = current_gradient[Z_];
            
            // add the address2 to get to the next item with m=0
            address2 += ncell_nlip * (2*l+2);
            
        }
        
        
        // set all values where m=1
        prev1 = x  * one_per_r;
        if (evaluate_gradients_x) prev1_gradient[X_] = 1.0 + one_per_r_gradient[X_] * x;
        if (evaluate_gradients_y) prev1_gradient[Y_] = one_per_r_gradient[Y_] * x;
        if (evaluate_gradients_z) prev1_gradient[Z_] = one_per_r_gradient[Z_] * x;
        
        // set value for l=1, m=1
        radial_value = evaluate_polynomials_shared<NLIP>(address+3*ncell_nlip, cf, r);
        radial_derivative = evaluate_polynomials_shared<NLIP-1>(address+3*ncell_nlip, df, r);
        
        if (evaluate_gradients_x) result[X_] += radial_value * prev1_gradient[X_] + radial_derivative * prev1  * x;// * one_per_r;
        if (evaluate_gradients_y) result[Y_] += radial_value * prev1_gradient[Y_] + radial_derivative * prev1  * y;// * one_per_r;
        if (evaluate_gradients_z) result[Z_] += radial_value * prev1_gradient[Z_] + radial_derivative * prev1  * z;// * one_per_r;
        //if (only_spherical) printf("radial_value: %e, radial_derivative: %e, prev1, i.e., x/r: %e\n", radial_value, radial_derivative, prev1);
        //if (only_spherical && evaluate_gradients_x) printf("prev1-gradient-x: %e, x/r: %e\n", prev1_gradient[X_], x * one_per_r);
        //if (only_spherical && evaluate_gradients_y) printf("prev1-gradient-y: %e, y/r: %e\n", prev1_gradient[Y_], y * one_per_r);
        //if (only_spherical && evaluate_gradients_z) printf("prev1-gradient-z: %e, z/r: %e\n", prev1_gradient[Z_], z * one_per_r);
        
        // the starting address has 1 item before from the l=0, 3 from l=1, and 3 from l=2
        address2 = address + ncell_nlip * 7;
        for (l = 2; l <= lmax; l++) {   
            double a = ( 2.0*(double)l-1.0) * rsqrt( 1.0*(double)((l+1)*(l-1)) );
            current =   a * z*prev1 * one_per_r;
            if (evaluate_gradients_x) current_gradient[X_] = a *(z * prev1 * one_per_r_gradient[X_] + z * one_per_r * prev1_gradient[X_]);
            if (evaluate_gradients_y) current_gradient[Y_] = a *(z * prev1 * one_per_r_gradient[Y_] + z * one_per_r * prev1_gradient[Y_]);
            if (evaluate_gradients_z) current_gradient[Z_] = a *(z * prev1 * one_per_r_gradient[Z_] + prev1 + z * one_per_r * prev1_gradient[Z_]);
            if (l > 2) {
                double b = sqrt( (double)((l)*(l-2)) /  (double)((l+1)*(l-1)) );
                current -=  b * prev2;
                if (evaluate_gradients_x) current_gradient[X_] -= b * prev2_gradient[X_];
                if (evaluate_gradients_y) current_gradient[Y_] -= b * prev2_gradient[Y_];
                if (evaluate_gradients_z) current_gradient[Z_] -= b * prev2_gradient[Z_];
            }
            radial_value = evaluate_polynomials_shared<NLIP>(address2, cf, r);
            radial_derivative = evaluate_polynomials_shared<NLIP-1>(address2, df, r);
            if (evaluate_gradients_x) result[X_] += radial_value * current_gradient[X_] + radial_derivative * current * x;// * one_per_r;
            if (evaluate_gradients_y) result[Y_] += radial_value * current_gradient[Y_] + radial_derivative * current * y;// * one_per_r;
            if (evaluate_gradients_z) result[Z_] += radial_value * current_gradient[Z_] + radial_derivative * current * z;// * one_per_r;
            
            prev2 = prev1;
            if (evaluate_gradients_x) prev2_gradient[X_] = prev1_gradient[X_];
            if (evaluate_gradients_y) prev2_gradient[Y_] = prev1_gradient[Y_];
            if (evaluate_gradients_z) prev2_gradient[Z_] = prev1_gradient[Z_];
            prev1 = current;
            if (evaluate_gradients_x) prev1_gradient[X_] = current_gradient[X_];
            if (evaluate_gradients_y) prev1_gradient[Y_] = current_gradient[Y_];
            if (evaluate_gradients_z) prev1_gradient[Z_] = current_gradient[Z_];
            
            // add the address2 to get to the next item with m=-1
            address2 += ncell_nlip * (2*l+2);
        }
        
        // go through the rest of the stuff
        bottom = y * one_per_r; // bottom refers to solid harmonics value with l=l-1 and m=-(l-1)
        if (evaluate_gradients_x) bottom_gradient[X_] = one_per_r_gradient[X_] * y;
        if (evaluate_gradients_y) bottom_gradient[Y_] = 1.0 + one_per_r_gradient[Y_] * y;
        if (evaluate_gradients_z) bottom_gradient[Z_] = one_per_r_gradient[Z_] * y;
        top = x * one_per_r;    // top    refers to solid harmonics value with l=l-1 and m=l-1
        if (evaluate_gradients_x) top_gradient[X_] = 1.0 + one_per_r_gradient[X_] * x;
        if (evaluate_gradients_y) top_gradient[Y_] = one_per_r_gradient[Y_] * x;
        if (evaluate_gradients_z) top_gradient[Z_] = one_per_r_gradient[Z_] * x;
        lm_address += 4 * ncell_nlip;
        for (l=2; l <= lmax; l++) {
            double c = sqrt((2.0*(double)l - 1.0) / (2.0*(double)l));
            new_bottom = c * one_per_r * ( y*top + x*bottom);
            
            // get the gradients to x direction
            if (evaluate_gradients_x) new_bottom_gradient = c * (one_per_r_gradient[X_] * (y * top                    + x * bottom) +
                                       one_per_r              * (y * top_gradient[X_]       + x * bottom_gradient[X_] + bottom)) ;
            if (evaluate_gradients_x) top_gradient[X_]    = c * (one_per_r_gradient[X_] * (x * top                    - y * bottom) +
                                       one_per_r              * (x * top_gradient[X_] + top - y * bottom_gradient[X_]));
            if (evaluate_gradients_x) bottom_gradient[X_] = new_bottom_gradient;
            
            // get the gradients to y direction
            if (evaluate_gradients_y) new_bottom_gradient = c * (one_per_r_gradient[Y_] * (y * top                    + x * bottom) +
                                       one_per_r              * (y * top_gradient[Y_] + top + x * bottom_gradient[Y_]));
            if (evaluate_gradients_y) top_gradient[Y_] =    c * (one_per_r_gradient[Y_] * (x * top                    - y * bottom) +
                                       one_per_r              * (x * top_gradient[Y_]       - y * bottom_gradient[Y_] - bottom));
            if (evaluate_gradients_y) bottom_gradient[Y_] = new_bottom_gradient;
            
            // get the gradients to z direction
            if (evaluate_gradients_z) new_bottom_gradient = c * (one_per_r_gradient[Z_] * (y * top                    + x * bottom) +
                                       one_per_r              * (y * top_gradient[Z_]       + x * bottom_gradient[Z_]));
            if (evaluate_gradients_z) top_gradient[Z_] =    c * (one_per_r_gradient[Z_] * (x * top                    - y * bottom) +
                                       one_per_r              * (x * top_gradient[Z_]       - y * bottom_gradient[Z_]));
            if (evaluate_gradients_z) bottom_gradient[Z_] = new_bottom_gradient;
            
            
            top = c * one_per_r * ( x*top-y*bottom );
                        
            // store the new bottom: l=l, m=-l (we need the old bottom in calculation of top previously, so we 
            // have to sacrifice one register temporarily)
            bottom = new_bottom;
            
            radial_value = evaluate_polynomials_shared<NLIP>(lm_address, cf, r);
            radial_derivative = evaluate_polynomials_shared<NLIP-1>(lm_address, df, r);
            
            // get value for l=l, m=-l.
            if (evaluate_gradients_x) result[X_] += radial_value * bottom_gradient[X_] + radial_derivative * bottom  * x;// * one_per_r;
            if (evaluate_gradients_y) result[Y_] += radial_value * bottom_gradient[Y_] + radial_derivative * bottom  * y;// * one_per_r;
            if (evaluate_gradients_z) result[Z_] += radial_value * bottom_gradient[Z_] + radial_derivative * bottom  * z;// * one_per_r;
                      
            radial_value = evaluate_polynomials_shared<NLIP>(lm_address + 2*l * ncell_nlip, cf, r);
            radial_derivative = evaluate_polynomials_shared<NLIP-1>(lm_address + 2*l * ncell_nlip, df, r);
                      
            // get value for l=l, m=l. The address is 2*l items away from l=l, m=-l
            if (evaluate_gradients_x) result[X_] += radial_value * top_gradient[X_] + radial_derivative * top  * x;// * one_per_r;
            if (evaluate_gradients_y) result[Y_] += radial_value * top_gradient[Y_] + radial_derivative * top  * y;// * one_per_r;
            if (evaluate_gradients_z) result[Z_] += radial_value * top_gradient[Z_] + radial_derivative * top  * z;// * one_per_r;
            
            // set all values where m=-l
            prev1 = bottom;
            if (evaluate_gradients_x) prev1_gradient[X_] = bottom_gradient[X_];
            if (evaluate_gradients_y) prev1_gradient[Y_] = bottom_gradient[Y_];
            if (evaluate_gradients_z) prev1_gradient[Z_] = bottom_gradient[Z_];
            address2 = lm_address + (2*l+2) * ncell_nlip;
                        
            for (l2 = l+1; l2 <= lmax; l2++) {
                // evaluate spherical harmonics for l=l2, m=-l
                double a = ( 2.0*(double)l2-1.0) * rsqrt( 1.0*(double)((l2-l)*(l2+l)) );
                current =   a * z*prev1 * one_per_r;
                if (evaluate_gradients_x) current_gradient[X_] = a *(z * prev1 * one_per_r_gradient[X_] + z * one_per_r * prev1_gradient[X_]);
                if (evaluate_gradients_y) current_gradient[Y_] = a *(z * prev1 * one_per_r_gradient[Y_] + z * one_per_r * prev1_gradient[Y_]);
                if (evaluate_gradients_z) current_gradient[Z_] = a *(z * prev1 * one_per_r_gradient[Z_] + prev1 + z * one_per_r * prev1_gradient[Z_]);
                if (l2 > l+1) { 
                    double b = sqrt( (double)((l2-l-1)*(l2+l-1)) /  (double)((l2-l)*(l2+l)) );
                    current -=  b * prev2;
                    if (evaluate_gradients_x) current_gradient[X_] -= b * prev2_gradient[X_];
                    if (evaluate_gradients_y) current_gradient[Y_] -= b * prev2_gradient[Y_];
                    if (evaluate_gradients_z) current_gradient[Z_] -= b * prev2_gradient[Z_];
                }
                radial_value = evaluate_polynomials_shared<NLIP>(address2, cf, r);
                radial_derivative = evaluate_polynomials_shared<NLIP-1>(address2, df, r);
                
                if (evaluate_gradients_x) result[X_] += radial_value * current_gradient[X_] + radial_derivative * current * x;// * one_per_r;
                if (evaluate_gradients_y) result[Y_] += radial_value * current_gradient[Y_] + radial_derivative * current * y;// * one_per_r;
                if (evaluate_gradients_z) result[Z_] += radial_value * current_gradient[Z_] + radial_derivative * current * z;// * one_per_r;
                
                prev2 = prev1;
                if (evaluate_gradients_x) prev2_gradient[X_] = prev1_gradient[X_];
                if (evaluate_gradients_y) prev2_gradient[Y_] = prev1_gradient[Y_];
                if (evaluate_gradients_z) prev2_gradient[Z_] = prev1_gradient[Z_];
                prev1 = current;
                if (evaluate_gradients_x) prev1_gradient[X_] = current_gradient[X_];
                if (evaluate_gradients_y) prev1_gradient[Y_] = current_gradient[Y_];
                if (evaluate_gradients_z) prev1_gradient[Z_] = current_gradient[Z_];
                
                // add the address2 to get to the next item with m=-1
                address2 += ncell_nlip * (2*l2+2);
                
            }
            
            
            // set all values where m=l
            lm_address += 2*l * ncell_nlip;
            
            prev1 = top;
            if (evaluate_gradients_x) prev1_gradient[X_] = top_gradient[X_];
            if (evaluate_gradients_y) prev1_gradient[Y_] = top_gradient[Y_];
            if (evaluate_gradients_z) prev1_gradient[Z_] = top_gradient[Z_];
            address2 = lm_address + (2*l+2) * ncell_nlip;
            for (l2 = l+1; l2 <= lmax; l2++) {
                // evaluate spherical harmonics for l=l2, m=l
                double a = ( 2.0*(double)l2-1.0) * rsqrt( 1.0*(double)((l2+l)*(l2-l)) );
                current =   a * z*prev1 * one_per_r;
                if (evaluate_gradients_x) current_gradient[X_] = a *(z * prev1 * one_per_r_gradient[X_] + z * one_per_r * prev1_gradient[X_]);
                if (evaluate_gradients_y) current_gradient[Y_] = a *(z * prev1 * one_per_r_gradient[Y_] + z * one_per_r * prev1_gradient[Y_]);
                if (evaluate_gradients_z) current_gradient[Z_] = a *(z * prev1 * one_per_r_gradient[Z_] + prev1  + z * one_per_r * prev1_gradient[Z_]);
                if (l2 > l+1) {
                    double b = sqrt( (double)((l2+l-1)*(l2-l-1)) /  (double)((l2+l)*(l2-l)) );
                    current -=  b * prev2;
                    if (evaluate_gradients_x) current_gradient[X_] -= b * prev2_gradient[X_];
                    if (evaluate_gradients_y) current_gradient[Y_] -= b * prev2_gradient[Y_];
                    if (evaluate_gradients_z) current_gradient[Z_] -= b * prev2_gradient[Z_];
                }
                radial_value = evaluate_polynomials_shared<NLIP>(address2, cf, r);
                radial_derivative = evaluate_polynomials_shared<NLIP-1>(address2, df, r);
                
                if (evaluate_gradients_x) result[X_] += radial_value * current_gradient[X_] + radial_derivative * current * x;// * one_per_r;
                if (evaluate_gradients_y) result[Y_] += radial_value * current_gradient[Y_] + radial_derivative * current * y;// * one_per_r;
                if (evaluate_gradients_z) result[Z_] += radial_value * current_gradient[Z_] + radial_derivative * current * z;// * one_per_r;
                
                prev2 = prev1;
                if (evaluate_gradients_x) prev2_gradient[X_] = prev1_gradient[X_];
                if (evaluate_gradients_y) prev2_gradient[Y_] = prev1_gradient[Y_];
                if (evaluate_gradients_z) prev2_gradient[Z_] = prev1_gradient[Z_];
                prev1 = current;
                if (evaluate_gradients_x) prev1_gradient[X_] = current_gradient[X_];
                if (evaluate_gradients_y) prev1_gradient[Y_] = current_gradient[Y_];
                if (evaluate_gradients_z) prev1_gradient[Z_] = current_gradient[Z_];
                
                // add the address2 to get to the next item with m=-1
                address2 += ncell_nlip * (2*l2+2);
                
            }
            
            // get next address
            lm_address += ncell_nlip;
        }
    }
    result[X_] *= one_per_r;
    result[Y_] *= one_per_r;
    result[Z_] *= one_per_r;
    // multiply the result with r^k, if k is not 0
    // the distance is not too close to 0.0 as this is checked
    // earlier in this function, NOTE: should never happen, thus 
    // commented away
    //if (k != 0 && distance > 1e-12) {
    
    /*for (int i = 0; i < k; i ++) {
        result *= distance;
    }
    for (int i = 0; i < -k; i ++) {
        result *= one_per_r;
    }*/
    
    //}
    
    if (distance < 1e-12) {
        result[X_] = 0.0; // * evaluate_polynomials_shared<NLIP-1>(address, df, r);
        result[Y_] = 0.0;
        result[Z_] = 0.0;
    }
}


/* 
 * Evaluates value of single bubble at a point. This is very similar with the
 * SolidHarmonics simple evaluation, but the results are multiplied with the
 * polynomial evaluations
 */

__device__ inline double  Bubbles_evaluate_point(
                                         // x-coordinate relative to the center of the bubble
                                         const double &x,
                                         // y-coordinate relative to the center of the bubble
                                         const double &y,
                                         // z-coordinate relative to the center of the bubble
                                         const double &z,
                                         // relative distance from the center of the bubble
                                         const double &distance,
                                         // maximum quantum number 'l'
                                         const int &lmax,
                                         // number of cells
                                         const int &ncell,
                                         // number of lagrange integration polyniomials per 
                                         // cell, i.e., the number of grid points per cell
                                         const int &nlip,
                                         // position inside the cell
                                         const double &r,
                                         // k value for the bubble
                                         const int &k,
                                         // the first address value in bubble for the selected cell
                                         const int &address,
                                         // constant pointer to a variable double array
                                         const double* __restrict__ cf
                                        ) {
    
    double result = 0.0;
    int lm_address = address, address2;
    // NOTE: Here the nlip is replaced by 8 because this gives advantages in loading the stuff
    // also *cf this should be done
    const int ncell_nlip = ncell * 8;
    int l, l2; 
    double top, bottom, new_bottom, prev1, prev2, current, a, b, a2;
    const double one_per_r = 1.0 / distance;;
    l = 0;
    // set value for l=0, m=0
    //printf("x: %f, y: %f, z: %f, nlip: %d, ncell: %d, l: 0, address: %d, cf: %ld, r: %f\n", x, y, z, nlip, ncell, 0, lm_address, cf, r);
    //printf("shared_memory address: %ld\n");
    //printf("shared memory first value: %f", shared_memory[0]);
    result = evaluate_polynomials_shared<NLIP>(lm_address, cf, r);
    
    
    
    if (lmax >= 1) { 
        // set value for l=1, m=-1
        result += y * evaluate_polynomials_shared<NLIP>(address+ncell_nlip, cf, r) * one_per_r;
        // set value for l=1, m=0
        result += z * evaluate_polynomials_shared<NLIP>(address+2*ncell_nlip, cf, r) * one_per_r;
        // set value for l=1, m=1
        result += x * evaluate_polynomials_shared<NLIP>(address+3*ncell_nlip, cf, r) * one_per_r;
        
        // set all values where m=-1
        prev2 = 0.0;
        prev1 = y * one_per_r;
        
        // the starting address has 1 item before from the l=0, 3 from l=1, and 1 from l=2
        address2 = address + ncell_nlip * 5;
        l = threadIdx.x % 32;
        a =  ( 2.0*(double)l-1.0) * rsqrt( 1.0*(double)((l-1)*(l+1)) );
        b =  (l > 2) ? sqrt( (double)((l-2)*(l)) /  (double)((l-1)*(l+1)) ) : 0.0;
        for (l = 2; l <= lmax; l++) {
            current =  __shfl(a, l) * z*prev1 * one_per_r - __shfl(b, l) * prev2;
            result += current * evaluate_polynomials_shared<NLIP>(address2, cf, r) ;
            prev2 = prev1;
            prev1 = current;
            
            // add the address2 to get to the next item with m=-1
            address2 += ncell_nlip * (2*l+2);
        }
          
        
        // set all values where m=0
        prev1 = z * one_per_r;
        prev2 = 1.0;
        // the starting address has 1 item before from the l=0, 3 from l=1, and 2 from l=2
        address2 = address + ncell_nlip * 6;
        
        
        l = threadIdx.x % 32;
        a =  ( 2.0*(double)l-1.0) * rsqrt( 1.0*(double)((l)*(l)) );
        b =  sqrt( (double)((l-1)*(l-1)) /  (double)((l)*(l)) );
        for (l = 2; l <= lmax; l++) {
            current =   __shfl(a, l) * z * prev1 * one_per_r -  __shfl(b, l) * prev2;
            result += current * evaluate_polynomials_shared<NLIP>(address2, cf, r);
            prev2 = prev1;
            prev1 = current; 
            
            // add the address2 to get to the next item with m=0 
            address2 += ncell_nlip * (2*l+2);
            
        }
        
        // set all values where m=1
        prev1 = x * one_per_r;
        // the starting address has 1 item before from the l=0, 3 from l=1, and 3 from l=2
        address2 = address + ncell_nlip * 7;
        
        l = threadIdx.x % 32;
        a =  ( 2.0*(double)l-1.0) * rsqrt( 1.0*(double)((l+1)*(l-1)) );
        b =  (l > 2) ? sqrt( (double)((l)*(l-2)) /  (double)((l+1)*(l-1)) ) : 0.0;
        for (l = 2; l <= lmax; l++) {   
            current =  __shfl(a, l) * z*prev1 * one_per_r -  __shfl(b, l) * prev2;
            result += current * evaluate_polynomials_shared<NLIP>(address2, cf, r);
            prev2 = prev1;
            prev1 = current;
            
            // add the address2 to get to the next item with m=1 
            address2 += ncell_nlip * (2*l+2);
        }
        
        // go through the rest of the stuff
        bottom = y * one_per_r; // bottom refers to spherical harmonics value with l=l-1 and m=-(l-1)
        top = x * one_per_r;    // top    refers to spherical harmonics value with l=l-1 and m=l-1
        lm_address += 4 * ncell_nlip;
        l = threadIdx.x % 32;
        a = sqrt((2.0*(double)l - 1.0) / (2.0*(double)l));
        for (l=2; l <= lmax; l++) {
            
            new_bottom = __shfl(a, l) * one_per_r * ( y*top + x*bottom);
            top        = __shfl(a, l) * one_per_r * ( x*top - y*bottom );
                        
            // store the new bottom: l=l, m=-l (we need the old bottom in calculation of top previously, so we 
            // have to sacrifice one register temporarily)
            bottom = new_bottom;
            
            result += bottom * evaluate_polynomials_shared<NLIP>(lm_address, cf, r);
            // get value for l=l, m=l. The address is 2*l items away from l=l, m=-l
            result += top * evaluate_polynomials_shared<NLIP>(lm_address + 2*l * ncell_nlip, cf, r);
            
            // set all values where m=-l
            prev2 = 0.0;
            prev1 = bottom;
            address2 = lm_address + (2*l+2) * ncell_nlip;
                        
            // set all values where m=l
            lm_address += 2*l * ncell_nlip;
            l2 = threadIdx.x % 32;
            a2 =  ( 2.0*(double)l2-1.0) * rsqrt( 1.0*(double)((l2-l)*(l2+l)) );
            b =  (l2 > l+1) ? sqrt( (double)((l2-l-1)*(l2+l-1)) /  (double)((l2-l)*(l2+l)) ) : 0.0;
            for (l2 = l+1; l2 <= lmax; l2++) {
                // evaluate spherical harmonics for l=l2, m=-l
                current =  __shfl(a2, l2) * z*prev1 * one_per_r - __shfl(b, l2) *  prev2; 
                
                result += current * evaluate_polynomials_shared<NLIP>(address2, cf, r);
                prev2 = prev1;
                prev1 = current;
                
                // add the address2 to get to the next item with m=-l
                address2 += ncell_nlip * (2*l2+2);
            }
                
            
            prev2 = 0.0;
            prev1 = top;
            address2 = lm_address + (2*l+2) * ncell_nlip;
            
            l2 = threadIdx.x % 32;
            a2 =  ( 2.0*(double)l2-1.0) * rsqrt( 1.0*(double)((l2+l)*(l2-l)) ) ;
            b =  (l2 > l+1) ? sqrt( (double)((l2+l-1)*(l2-l-1)) /  (double)((l2+l)*(l2-l)) ) : 0.0;
            for (l2 = l+1; l2 <= lmax; l2++) {
                // evaluate spherical harmonics for l=l2, m=l
                current =  __shfl(a2, l2) * z*prev1 * one_per_r - __shfl(b, l2) * prev2;
                // the latter term will go to zero, if l2 <= l+1
                result += current * evaluate_polynomials_shared<NLIP>(address2, cf, r);
                
                prev2 = prev1;
                prev1 = current;
                
                // add the address3 to get to the next item with m=l
                address2 += ncell_nlip * (2*l2+2);
            }
            
            // get next address
            lm_address += ncell_nlip;
        }
    }
    // multiply the result with r^k, if k is not 0
    // the distance is not too close to 0.0 as this is checked
    // earlier in this function, NOTE: should never happen, thus 
    // commented away
    //if (k != 0 && distance > 1e-12) {
    
    
    
    if (distance < 1e-14) {
        result = 1.0 * evaluate_polynomials_shared<NLIP>(address, cf, r);
    }
    
    for (int i = 0; i < k; i ++) {
        result *= distance;
    }
    for (int i = 0; i < -k; i ++) {
        result *= one_per_r;
    }
    
    //}
    return result;
}



__device__ int getGlobalIdx_1D_1D() {
    int id=threadIdx.x + blockIdx.x * blockDim.x;
    return id;
}

__device__ int getGlobalIdx_3D_3D() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
        + (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x)
        + threadIdx.x;
    return threadId;
}

/*
 * Get the minimum/maximum and overwrite values with -1
 */
__device__ inline void minmax(int *first, int *second) {
    int temp;
    if (*first == -1) {
        *first = *second;
    }
    if (*second == -1) {
        *second = *first;
    }
    if (*second < *first) {
        temp = *second;
        *second = *first;
        *first = temp;
    }
   
}

/*
 * Find the minimum and maximum in array that is as large as a block, and store them as the first 
 * and last value of the input array. NOTE: The arrayLength must be a power of 2.
 */
__device__ void calculateMinimumMaximum(int *array, int blockThreadId, int arrayLength) {
    int division = arrayLength / 2;
    // order so that the larger values of pairs are at the second part of the array
    // and the smaller are at the end of the array
    if (blockThreadId < division) {
        // rearrange the values so that the larger is in the &array[blockThreadId + division]
        // and smaller is in &array[blockThreadId]
        minmax(&array[blockThreadId], &array[blockThreadId + division]);
    }
    __syncthreads();
    division = arrayLength / 4;
    // if the block
    while (division >= 1) {
        if (blockThreadId < division) {
            minmax(&array[blockThreadId], &array[blockThreadId + division]);
        }
        else if (blockThreadId > arrayLength - division) {
            minmax(&array[blockThreadId - division], &array[blockThreadId]);
        }
        division /= 2;
        __syncthreads();
    }
    
}



/* 
 * Evaluate Bubbles on a grid
 * 
 */
template <bool lmin_zero, bool evaluate_value, bool evaluate_gradients_x, bool evaluate_gradients_y, bool evaluate_gradients_z >
__device__ inline void 
Bubbles_evaluate_grid(const Bubble* __restrict__ bubble,
                      double* __restrict__ cube,
                      double* __restrict__ gradient_cube_x,
                      double* __restrict__ gradient_cube_y,
                      double* __restrict__ gradient_cube_z,
                      const double* __restrict__ grid_points_x,
                      const double* __restrict__ grid_points_y,
                      const double* __restrict__ grid_points_z,
                      const int shape_x,
                      const int shape_y,
                      const int shape_z,
                      const double zero_point_x,
                      const double zero_point_y,
                      const double zero_point_z, 
                      const int k,
                      const int slice_offset,
                      const size_t pitch,
                      const int memory_y_shape,
                      const int slice_count,
                      const int lmin,
                      const double multiplier) {    
    
    // The result array will be in fortran with indices l, x, y, z. 
    // This means that the x index will be the fastest to change.
    int x, y, z;
    getXYZ(&x, &y, &z);
    
    // get the offset from the input cube pointer
    const int id = getCubeOffset3D(x, y, z, pitch, memory_y_shape);
    
    double value, gradient[3];
    double in_cell_position = 0.0;
    const int ncell = bubble->grid->ncell, nlip = bubble->grid->nlip;
    int icell;
    double relative_position_x, relative_position_y, relative_position_z, distance;
                
    //printf("X: %f, cell_spacing: %f, ncell: %d", distance, bubble->cell_spacing, ncell);
    // Check that the point is within the block 
    if (x < shape_x && y < shape_y && z+slice_offset < shape_z && z < slice_count) {
        // calculate relative position to the zero-point and distance to it 
        calculate_distance(relative_position_x,
                           relative_position_y,
                           relative_position_z,
                           distance, 
                           zero_point_x, 
                           zero_point_y,
                           zero_point_z,
                           grid_points_x[x],
                           ldg<double>(&grid_points_y[y]),
                           ldg<double>(&grid_points_z[z+slice_offset]));
        
        
        
        // get the order number of cell the point resides in
        //icell = calculate_icell(distance, bubble->d, bubble->ncell);
        calculate_icell_radial(distance, bubble->charge, bubble->grid->r_max, ncell, nlip, &icell, &in_cell_position);
        //printf("x: %d, y: %d, z:%d, id:%d, vector_id: %d, vector_offset:%d, blockId: %d, blocks_per_vector: %d, %f, %f, %f, %d\n", x, y, z, id, vector_id, vector_offset, blockIdx.x, blocks_per_vector, grid_points_x[x], ldg(&grid_points_y[y]), ldg(&grid_points_z[z]), icell);
    }    
    else {
        icell = 1;
        distance = 0.1;
    }
    
    if (lmin_zero) {
        
        // calculate the bubble value for the point with lmin = 0
        if (evaluate_value) {
            value = Bubbles_evaluate_point( relative_position_x,
                                        relative_position_y,
                                        relative_position_z,
                                        distance,
                                        bubble->lmax,
                                        ncell,
                                        nlip,
                                        in_cell_position,
                                        k,
                                        icell * 8,
                                        bubble->cf);
        }
        // evaluate gradients if we are evaluating any
        if (evaluate_gradients_x || evaluate_gradients_y || evaluate_gradients_z) {
            Bubbles_evaluate_gradient_point
                <evaluate_gradients_x, evaluate_gradients_y, evaluate_gradients_z>
                                           (relative_position_x,
                                            relative_position_y,
                                            relative_position_z,
                                            distance,
                                            bubble->lmax,
                                            ncell,
                                            nlip,
                                            in_cell_position,
                                            k,
                                            icell * 8,
                                            bubble->cf,
                                            bubble->df,
                                            false,
                                            gradient
                                           );
        }
    }
    else {
        if (evaluate_value) {
            // calculate the bubble value for the point with lmin > 0
            value = Bubbles_evaluate_point_lmin( relative_position_x,
                                            relative_position_y,
                                            relative_position_z,
                                            distance,
                                            lmin,
                                            bubble->lmax,
                                            ncell,
                                            nlip,
                                            in_cell_position,
                                            k,
                                            icell * 8,
                                            bubble->cf
                                        );
        }
    }
    
    
    if (x < shape_x && y < shape_y && z+slice_offset < shape_z && z < slice_count && icell < ncell) {
        /*if (x == 0 && y == 0) {
            printf("%d: [x, y, z], id : [%d, %d, %d], %d, icell: %d, in_cell_position:%f, first_bubble-value:%e, distance:%f, coord: [%f, %f, %f] old-value: %e, value: %e, multiplier: %f\n", slice_offset, x, y, z+slice_offset, id, icell, in_cell_position, bubble->cf[icell*8], distance, relative_position_x, relative_position_y, relative_position_z, cube[id], value, multiplier);
        }*/
        if (evaluate_value) cube[id] += multiplier *  value;
        if (evaluate_gradients_x) gradient_cube_x[id] += multiplier * gradient[X_];
        if (evaluate_gradients_y) gradient_cube_y[id] += multiplier * gradient[Y_];
        if (evaluate_gradients_z) gradient_cube_z[id] += multiplier * gradient[Z_];
    }
    return;

}


/* 
 * Evaluate Bubbles on a grid
 */
__global__ void 
#if (__CUDA_ARCH__ <= 350)
__launch_bounds__(128, 6)
#else
__launch_bounds__(256)
#endif
Bubbles_evaluate_grid_lmin(const Bubble* __restrict__ bubble,
                           double* __restrict__ cube,
                           const double* __restrict__ grid_points_x,
                           const double* __restrict__ grid_points_y,
                           const double* __restrict__ grid_points_z,
                           const int shape_x,
                           const int shape_y,
                           const int shape_z,
                           const double zero_point_x,
                           const double zero_point_y,
                           const double zero_point_z, 
                           const int k,
                           const int slice_offset,
                           const size_t pitch,
                           const int memory_y_shape,
                           const int slice_count,
                           const int lmin,
                           const double multiplier) { 
    Bubbles_evaluate_grid <false, true, false, false, false> (
        bubble, cube, /*gradient_cube_x = */NULL,
        /*gradient_cube_y = */NULL, /*gradient_cube_z = */NULL,
        grid_points_x, grid_points_y, grid_points_z,
        shape_x, shape_y, shape_z,
        zero_point_x, zero_point_y, zero_point_z,
        k, slice_offset, pitch, memory_y_shape,
        slice_count, lmin, multiplier);
}


__global__ void 
#if (__CUDA_ARCH__ > 350)
__launch_bounds__(256)
#else
__launch_bounds__(128, 8)
#endif
Bubbles_evaluate_grid_pitched(const Bubble* __restrict__ bubble,
                              double* __restrict__ cube,
                              const double* __restrict__ grid_points_x,
                              const double* __restrict__ grid_points_y,
                              const double* __restrict__ grid_points_z,
                              const int shape_x,
                              const int shape_y,
                              const int shape_z,
                              const double zero_point_x,
                              const double zero_point_y,
                              const double zero_point_z, 
                              const int k,
                              const int slice_offset,
                              const size_t pitch,
                              const int memory_y_shape,
                              const int slice_count,
                              const double multiplier) {
    Bubbles_evaluate_grid <true, true, false, false, false> (
        bubble, cube, /*gradient_cube_x = */NULL,
        /*gradient_cube_y = */NULL, /*gradient_cube_z = */NULL,
        grid_points_x, grid_points_y, grid_points_z,
        shape_x, shape_y, shape_z,
        zero_point_x, zero_point_y, zero_point_z,
        k, slice_offset, pitch, memory_y_shape,
        slice_count, /*lmin = */0, multiplier);
}

template <bool lmin_zero, bool evaluate_value, bool evaluate_gradients_x, bool evaluate_gradients_y, bool evaluate_gradients_z >
__global__ void 
#if (__CUDA_ARCH__ > 350)
__launch_bounds__(256)
#else
__launch_bounds__(128, 5)
#endif
Bubbles_evaluate_grid_gradients(const Bubble* __restrict__ bubble,
                                double* __restrict__ cube,
                                double* __restrict__ gradient_cube_x,
                                double* __restrict__ gradient_cube_y,
                                double* __restrict__ gradient_cube_z,
                                const double* __restrict__ grid_points_x,
                                const double* __restrict__ grid_points_y,
                                const double* __restrict__ grid_points_z,
                                const int shape_x,
                                const int shape_y,
                                const int shape_z,
                                const double zero_point_x,
                                const double zero_point_y,
                                const double zero_point_z, 
                                const int k,
                                const int slice_offset,
                                const size_t pitch,
                                const int memory_y_shape,
                                const int slice_count,
                                const double multiplier) {
    Bubbles_evaluate_grid <lmin_zero, evaluate_value, evaluate_gradients_x, evaluate_gradients_y, evaluate_gradients_z> (
        bubble, cube, gradient_cube_x, 
        gradient_cube_y, gradient_cube_z,
        grid_points_x, grid_points_y, grid_points_z,
        shape_x, shape_y, shape_z,
        zero_point_x, zero_point_y, zero_point_z,
        k, slice_offset, pitch, memory_y_shape,
        slice_count, /*lmin = */0, multiplier);
}


/* 
 * Evaluate Bubbles at points
 */
template <bool lmin_zero, bool evaluate_value, bool evaluate_gradients_x, bool evaluate_gradients_y, bool evaluate_gradients_z>
__device__ inline void 
Bubbles_evaluate_points(const Bubble* __restrict__ bubble,
                        double* __restrict__ result_array,
                        double* __restrict__ device_gradients_x,
                        double* __restrict__ device_gradients_y,
                        double* __restrict__ device_gradients_z,
                        // a 3d array, where the x coordinates are first, 
                        // then y coordinates, and finally the z coordinates. This ordering
                        // is selected to get coalesced memory reads
                        const double* __restrict__ points,
                        // total number of points evaluated by this device
                        const int device_number_of_points,
                        // the zero point x-coordinate of bubbles
                        const double zero_point_x,
                        // the zero point y-coordinate of bubbles
                        const double zero_point_y,
                        // the zero point z-coordinate of bubbles
                        const double zero_point_z, 
                        // the k value of the bubbles
                        const int k,
                        // the lmin value evaluated
                        const int lmin,
                        // number of points in this kernel call
                        const int point_count,
                        // device_point_offset
                        const int device_point_offset,
                        const double multiplier
                       ) {    
    
    // Get the point order number within this kernel call
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    double value, gradient[3];
    double in_cell_position = 0.0;
    const int ncell = bubble->grid->ncell, nlip = bubble->grid->nlip;
    int icell = -1;
    double relative_position_x, relative_position_y, relative_position_z, distance, r_max = bubble->grid->r_max;
                
    //printf("X: %f, cell_spacing: %f, ncell: %d", distance, bubble->cell_spacing, ncell);
    // Check that the point is within the block 
    if (id + device_point_offset < device_number_of_points && id < point_count ) {
        // calculate relative position to the zero-point and distance to it 
        calculate_distance(relative_position_x,
                           relative_position_y,
                           relative_position_z,
                           distance, 
                           zero_point_x, 
                           zero_point_y,
                           zero_point_z,
                           points[id + device_point_offset],
                           points[id + device_point_offset + device_number_of_points],
                           points[id + device_point_offset + device_number_of_points*2]);
        
        
        
        // get the order number of cell the point resides in
        calculate_icell_radial(distance, bubble->charge, bubble->grid->r_max, ncell, nlip, &icell, &in_cell_position);
        
    }    
    else {
        icell = 1;
        distance = 0.1;
        
    }
    
    // calculate the bubble value for the point
    if (!lmin_zero) {
        if (evaluate_value) {
            value = Bubbles_evaluate_point_lmin( relative_position_x,
                                            relative_position_y,
                                            relative_position_z,
                                            distance,
                                            lmin,
                                            bubble->lmax,
                                            ncell,
                                            nlip,
                                            in_cell_position,
                                            k,
                                            icell * 8,
                                            bubble->cf
                                        );
           
        }
    }
    else {
        if (evaluate_gradients_x || evaluate_gradients_y || evaluate_gradients_z) {
            Bubbles_evaluate_gradient_point
                <evaluate_gradients_x, evaluate_gradients_y, evaluate_gradients_z>
                                           (relative_position_x,
                                            relative_position_y,
                                            relative_position_z,
                                            distance,
                                            bubble->lmax,
                                            ncell,
                                            nlip,
                                            in_cell_position,
                                            k,
                                            icell * 8,
                                            bubble->cf,
                                            bubble->df,
                                            false, //(evaluate_gradients_x != evaluate_gradients_y || evaluate_gradients_x != evaluate_gradients_z) && icell == 0,  //evaluate_gradients_x != evaluate_gradients_y || evaluate_gradients_x != evaluate_gradients_z,
                                            gradient
                                           );
        }
        if (evaluate_value) {
            value = Bubbles_evaluate_point( relative_position_x,
                                            relative_position_y,
                                            relative_position_z,
                                            distance,
                                            bubble->lmax,
                                            ncell,
                                            nlip,
                                            in_cell_position,
                                            k,
                                            icell * 8,
                                            bubble->cf
                                        );
        }
        
    }
    
    // store the result to the result array
    if (id + device_point_offset < device_number_of_points && id < point_count && distance < r_max && icell < ncell ) {
        
        if (evaluate_value) result_array[id+device_point_offset] += multiplier *  value;
        
        //if ((evaluate_gradients_x) && (id + device_point_offset <= 7)) printf("¤%¤%¤%%#¤%#¤ X: %d: pos: %f, %f, %f, val: %e, remainder: %e\n", id + device_point_offset, relative_position_x, relative_position_y, relative_position_z, gradient[X_], device_gradients_x[id+device_point_offset]);
        //if ((evaluate_gradients_y) && (id + device_point_offset <= 7)) printf("¤%¤%¤%%#¤%#¤ Y: %d: pos: %f, %f, %f, val: %e, remainder: %e\n", id + device_point_offset, relative_position_x, relative_position_y, relative_position_z, gradient[Y_], device_gradients_y[id+device_point_offset]);
        //if ((evaluate_gradients_z) && (id + device_point_offset <= 7)) printf("¤%¤%¤%%#¤%#¤ Z: %d: pos: %f, %f, %f, val: %e, remainder: %e\n", id + device_point_offset, relative_position_x, relative_position_y, relative_position_z, gradient[Z_], device_gradients_z[id+device_point_offset]);
        // add also the gradient value, if we are evaluating them
        if (evaluate_gradients_x) device_gradients_x[id+device_point_offset] += multiplier * gradient[X_];
        if (evaluate_gradients_y) device_gradients_y[id+device_point_offset] += multiplier * gradient[Y_];
        if (evaluate_gradients_z) device_gradients_z[id+device_point_offset] += multiplier * gradient[Z_];
    }
    return;

}


__device__ inline double get_damping_factor(double r) {
    double result;
    // erfc: error function
    if (r > 1e-12) {
        result = 0.5*erfc(r-2.0/r);
    }
    else {
        result = 1.0;
    }
    return result;
}


template <bool lmin_zero, bool evaluate_value, bool evaluate_gradients_x, bool evaluate_gradients_y, bool evaluate_gradients_z >
#if (__CUDA_ARCH__ <= 350)
__launch_bounds__(128, 4)
#else
__launch_bounds__(256)
#endif
__global__ void Bubbles_evaluate_gradient_points(
                        const Bubble* __restrict__ bubble,
                        double* __restrict__ result_array,
                        double* __restrict__ device_gradients_x,
                        double* __restrict__ device_gradients_y,
                        double* __restrict__ device_gradients_z,
                        // a 3d array, where the x coordinates are first, 
                        // then y coordinates, and finally the z coordinates. This ordering
                        // is selected to get coalesced memory reads
                        const double* __restrict__ points,
                        // total number of points evaluated by this device
                        const int device_number_of_points,
                        // the zero point x-coordinate of bubbles
                        const double zero_point_x,
                        // the zero point y-coordinate of bubbles
                        const double zero_point_y,
                        // the zero point z-coordinate of bubbles
                        const double zero_point_z, 
                        // the k value of the bubbles
                        const int k,
                        // number of points in this kernel call
                        const int point_count,
                        // device_point_offset
                        const int device_point_offset,
                        const double multiplier
                       ) { 
    Bubbles_evaluate_points<lmin_zero, evaluate_value, evaluate_gradients_x, evaluate_gradients_y, evaluate_gradients_z>(
        bubble,
        result_array,
        device_gradients_x,
        device_gradients_y,
        device_gradients_z,
        points,
        device_number_of_points,
        zero_point_x,
        zero_point_y,
        zero_point_z,
        k,
        0,
        point_count,
        device_point_offset,
        multiplier
    );
}

#if (__CUDA_ARCH__ <= 350)
__launch_bounds__(128, 7)
#else
__launch_bounds__(256)
#endif
__global__ void Bubbles_evaluate_points_simple(
                        const Bubble* __restrict__ bubble,
                        double* __restrict__ result_array,
                        // a 3d array, where the x coordinates are first, 
                        // then y coordinates, and finally the z coordinates. This ordering
                        // is selected to get coalesced memory reads
                        const double* __restrict__ points,
                        // total number of points evaluated by this device
                        const int device_number_of_points,
                        // the zero point x-coordinate of bubbles
                        const double zero_point_x,
                        // the zero point y-coordinate of bubbles
                        const double zero_point_y,
                        // the zero point z-coordinate of bubbles
                        const double zero_point_z, 
                        // the k value of the bubbles
                        const int k,
                        // number of points in this kernel call
                        const int point_count,
                        // device_point_offset
                        const int device_point_offset,
                        const double multiplier
                       ) { 
    Bubbles_evaluate_points<true, true, false, false, false>(
        bubble,
        result_array,
        /*device_gradients_x*/NULL,
        /*device_gradients_y*/NULL,
        /*device_gradients_z*/NULL,
        points,
        device_number_of_points,
        zero_point_x,
        zero_point_y,
        zero_point_z,
        k,
        0,
        point_count,
        device_point_offset,
        multiplier
    );
    
}




/*__global__ void Bubble_make_taylor_kernel(Bubble_t *result_bubble, int maximum_taylor_order, double *contaminants, 
                                          double *c2s_coefficients, int *c2s_lm_ids, int *c2s_term_starts, int offset) {
    const int index=threadIdx.x + blockIdx.x * blockDim.x + offset;
    extern __shared__ double shared_memory[]; 
    double *one_per_kappa_factorial =  &shared_memory[0];
    double *shared_contaminants = &shared_memory[maximum_taylor_order];
    int contaminants_size = (maximum_taylor_order+1)*(maximum_taylor_order+2)*(maximum_taylor_order+3)/6;
    
    // calculate the 1/kappa! terms to the shared memory
    if (threadIdx.x < maximum_taylor_order) {
        int kappa = 1;
        for (int i = 1; i <= threadIdx.x; i++) {
            kappa *= i+1;
        }
        one_per_kappa_factorial[threadIdx.x] = 1.0 / ((double) kappa);
    }
    
    // load the contaminats to the shared memory
    if (threadIdx.x < contaminants_size) {
        int id = threadIdx.x;
        while (id < contaminants_size) {
            shared_contaminants[id] = contaminants[id];
            id += blockDim.x;
        }
    }
    __syncthreads();
    
    // do the actual calculation
    double r = result_bubble->gridpoints[index];
    double prefactor;
    double damping_factor = get_damping_factor(r);
    int k = result_bubble->k, ncell= result_bubble->ncell, nlip = result_bubble->nlip;
    int result_index = 0, counter = 0, term_counter = 0;
    for (int x = 0; x <= maximum_taylor_order; x++) {
        for (int y = 0; y <= maximum_taylor_order - x; y++) {
            for (int z = 0; z <= maximum_taylor_order - x - y; z++) {
                prefactor = one_per_kappa_factorial[x+y+z]// 1/[x+y+z] 
                            * pow(r, (double)(x+y+z - k)) // r^x+y+z-k
                            * shared_contaminants[counter]  // c
                            * damping_factor;     
                // go through all l,m terms which get contribution from x,y,z -term 
                while (term_counter < c2s_term_starts[counter+1]) {
                    // get the index in the result array, note: the -1 is because the indices are in
                    // fortran format, starting from 1
                    result_index = (c2s_lm_ids[term_counter]-1) * (ncell * (nlip-1) +1) + index;
                    // add the prefactor times the coefficient from cartesion to spherical conversion
                    result_bubble->f[result_index] += c2s_coefficients[term_counter] * prefactor;
                    // add the counter value used to follow the c2s conversion
                    term_counter++;
                }
                // add the conter value used to follow cartesian terms
                counter ++;
            }
        }
    }
    
} */

/*
 *  Kernel that sums the f-values of two bubble objects together. The summation happens
 *  pointwise so that each thread calculates all l,m values for each point. The result
 *  is stored to the bubble_f.
 */
__global__ void Bubble_sum_kernel(double* __restrict__ bubble_f, const double* __restrict__ bubble1_f, const int lmax, const int max_id, const size_t device_f_pitch) {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (id < max_id) {
        // go through all l, m values of input bubble 'bubble'
        for (int ilm = 0; ilm < (lmax+1)*(lmax+1); ilm++) {
            bubble_f[ilm * device_f_pitch / sizeof(double) + id] += bubble1_f[ilm * device_f_pitch / sizeof(double) + id];
        }
    }
}

/*
 *  Decreases the k-value of a bubble by k_decrese. The operation happens
 *  pointwise so that each thread calculates all l,m values for each point. The result
 *  is stored to the bubble_f.
 *  
 *  k_decrease is how many k values is decreased
 */
__global__ void Bubble_decrease_k_kernel(double* __restrict__ bubble_f, const double* __restrict__ r, const int k_decrease, const int lmax, const int max_id, const size_t device_f_pitch) {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (id < max_id) {
        const double rpow = pow(r[id], (double) k_decrease);
        // go through all l, m values of input bubble 'bubble'
        for (int ilm = 0; ilm < (lmax+1)*(lmax+1); ilm++) {
            bubble_f[ilm * device_f_pitch / sizeof(double) + id] *= rpow;
        }
    }
}

/*
 * Multiply cubes 1 and 2 and store it to cube1
 */
__global__ void multiply_cubes(double *cube1, double *cube2, const int cube_size, const int offset) {
    // get the id of the point (We are using only the first )
    const int index=threadIdx.x + blockIdx.x * blockDim.x + offset;
    
    if (index < cube_size) {
        cube1[index] *= cube2[index];
    }
}

/**************************************************************
 *  Bubble-implementation                                    *
 **************************************************************/

/*
 * Evaluate the cf at ALL devices. This is a crucial preparation function for injection.
 * For correct results, on call the Bubble must have all f-values present.
 * 
 * NOTE: the function streaming is structured using number of l,m-pairs, like the uploadAll.
 */
void Bubble::calculateCf() {
    
    // calculate the cf
    int ilmmax = (this->lmax+1)*(this->lmax+1);
    int block_size = 64;
    int grid_size;
    int offset;
    check_errors(__FILE__, __LINE__);
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        this->streamContainer->setDevice(device);
        offset = 0;
        for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream ++) {
            
            int ilm_per_stream = ilmmax /  this->streamContainer->getStreamsPerDevice() + 
                               (( ilmmax %  this->streamContainer->getStreamsPerDevice()) > stream);
            int number_of_points = ilm_per_stream * this->grid->ncell;
            // verify that there is something to calculate the cf for (for instance if ilmmax is 1, some streams
            // can be left without any points, resulting to a cuda error)
            if (number_of_points > 0) {
                grid_size = (number_of_points + block_size - 1) / block_size;
                calc_cf <<< grid_size, block_size, 0, *this->streamContainer->getStream(device, stream) >>>
                        (this->device_copies[device], offset, number_of_points, this->device_f_pitch[device]);
                offset += number_of_points;
            }
            check_errors(__FILE__, __LINE__);
        }
    }
}

void Bubble::initDeviceMemory(int ibub, Grid1D *grid, double center[3], int lmax,
               int k, double charge, StreamContainer *streamContainer) {
    //cudaHostRegister(this, sizeof(Bubble), cudaHostRegisterPortable);
    //check_errors(__FILE__, __LINE__);
    
    this->ibub = ibub;
    this->lmax = lmax;
    this->device_memory_lmax = lmax;
    this->k = k;
    this->charge = charge;
    this->streamContainer = streamContainer;
    this->crd[X_] = center[X_];
    this->crd[Y_] = center[Y_];
    this->crd[Z_] = center[Z_];
    this->integrator = NULL;


    this->uploaded_events = new cudaEvent_t*[this->streamContainer->getNumberOfDevices()];
    this->device_copies = new Bubble * [this->streamContainer->getNumberOfDevices()];
    this->device_f = new double *[this->streamContainer->getNumberOfDevices()];
    this->device_f_pitch = new size_t [this->streamContainer->getNumberOfDevices()];
    this->device_cf = new double * [this->streamContainer->getNumberOfDevices()];
    this->device_df = new double * [this->streamContainer->getNumberOfDevices()];
    
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        this->streamContainer->setDevice(device);
        
        size_t sz=sizeof(double)*(grid->ncell*(grid->nlip-1)+1);
        cudaMallocPitch((void**)&device_f[device], &device_f_pitch[device], 
                    sz, (lmax+1)*(lmax+1)); 
        check_errors(__FILE__, __LINE__);
        
        cudaMemset(device_f[device], 0, device_f_pitch[device]*(lmax+1)*(lmax+1));
        check_errors(__FILE__, __LINE__);
        
        sz=sizeof(double)*grid->ncell*8*(lmax+1)*(lmax+1);
        cudaMalloc(&this->device_cf[device], sz);
        cudaMalloc(&this->device_df[device], sz);
        check_errors(__FILE__, __LINE__);
        
        // copy the bubble to the device, for which set the device pointers
        // to be the main-pointers
        this->f = this->device_f[device];
        this->cf = this->device_cf[device];
        this->df = this->device_df[device];
        this->grid = grid->device_copies[device];
        
        // allocate & copy the bubble to device
        cudaMalloc(&this->device_copies[device], sizeof(Bubble));
        cudaMemcpy(this->device_copies[device], this, sizeof(Bubble), cudaMemcpyHostToDevice);
        check_errors(__FILE__, __LINE__);
        
        
    }
    this->grid = grid;
    
}

Bubble::Bubble(int ibub, Grid1D *grid, double center[3], int lmax, int k, double *bf,
               double charge, StreamContainer *streamContainer) {

    this->initDeviceMemory(ibub, grid, center, lmax, k, charge, streamContainer);
    
    // set the host variables and register them for faster data transfer
    this->f = bf;
    /*cudaHostRegister(this->f, sizeof(double)*(grid->ncell*(grid->nlip-1)+1)*(lmax+1)*(lmax+1), cudaHostRegisterPortable);
    check_errors(__FILE__, __LINE__);*/
    
}

Bubble::Bubble(int ibub, Grid1D *grid, double center[3], int lmax, int k, double charge, StreamContainer *streamContainer) {
    this->initDeviceMemory(ibub, grid, center, lmax, k, charge, streamContainer);
}

Bubble::Bubble(Bubble *old_bubble, int lmax, int k) {
    this->initDeviceMemory(old_bubble->ibub, old_bubble->grid, old_bubble->crd, lmax, old_bubble->k, old_bubble->charge, old_bubble->streamContainer);
}

/*
 * Uploads all bubble data to all devices (gpus) on all nodes. This kind of approach 
 * is needed when injecting bubbles to cuda. With bubble-multiplication - the upload
 * -method is preferred.
 */
void Bubble::uploadAll(double *f, int lmax) {
    // set the host variables and register them for faster data transfer
    this->f = f;
    this->lmax = lmax;
    size_t host_pitch = (this->grid->ncell * (this->grid->nlip - 1) + 1) * sizeof(double);
    int ilmmax = (lmax+1)*(lmax+1);
    check_errors(__FILE__, __LINE__);
    Grid1D* host_grid = this->grid;
    // register the host array array
    //cudaHostRegister(this->f, host_pitch * ilmmax, cudaHostRegisterPortable);
    check_errors(__FILE__, __LINE__);
    
    double *device_f, *host_f;
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        this->streamContainer->setDevice(device);
        
        // get the preallocated device pointers
        device_f = this->device_f[device];
        
        // NOTE: for all devices the first pointer points to  the first value of each array
        host_f = this->f;
        
        for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream++) {
            int ilm_per_stream = ilmmax /  this->streamContainer->getStreamsPerDevice() + 
                               (( ilmmax %  this->streamContainer->getStreamsPerDevice()) > stream);
            // upload the stream data to device
            cudaMemcpy2DAsync((void *) device_f, this->device_f_pitch[device],
                              (void *) host_f, host_pitch,
                              host_pitch,
                              ilm_per_stream, 
                              cudaMemcpyHostToDevice,
                              *this->streamContainer->getStream(device, stream)
                             );
            check_errors(__FILE__, __LINE__);
                    
            // add to the pointers
            device_f += ilm_per_stream * this->device_f_pitch[device] / sizeof(double);
            host_f += ilm_per_stream * host_pitch / sizeof(double);
            
        }
        
        // copy the bubble to the device, for which set the device pointers
        // to be the main-pointers
        this->f = this->device_f[device];
        this->cf = this->device_cf[device];
        this->df = this->device_df[device];
        this->grid = host_grid->device_copies[device];
        this->lmax = lmax;
        
        // copy the bubble to device
        cudaMemcpyAsync(this->device_copies[device], this, sizeof(Bubble), cudaMemcpyHostToDevice, 
                              *this->streamContainer->getStream(device, 0));
        check_errors(__FILE__, __LINE__);
        
        this->f = f;
        this->grid = host_grid;
        
    }
    check_errors(__FILE__, __LINE__);
    this->streamContainer->synchronizeAllDevices();
    // calculate the cf
    this->calculateCf();
    // and synchronize the host with the device
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        this->uploaded_events[device] = this->streamContainer->recordDeviceEvent(device);
    }
    // we are not in any case downloading the data back, so we can unregister the array 
    //cudaHostUnregister(this->f);
    check_errors(__FILE__, __LINE__);
}


/*
 * Uploads part of a bubble to the device
 * 
 * NOTE: in order to use this, the bubble uploaded (i.e., the f-array given as input)
 * must have the same lmax value as the Bubble-object we are uploading to.
 * NOTE: registers the input array but does not unregister it, thus after calling this
 * the user must unregister the f elsewhere, for instance by calling the unregister function.
 * NOTE: this function is designed to function together with the bubble multiplication
 */
void Bubble::upload(double *f, int lmax, bool register_host) {
    // set the host variables and register them for faster data transfer
    this->f = f;
    check_errors(__FILE__, __LINE__);
    this->lmax = lmax;
    int ilmmax = (lmax + 1) * (lmax + 1);
    // calculate the total number of points in the bubbles each l,m -pair, 
    int total_point_count = this->grid->ncell * (this->grid->nlip - 1) +1;
    Grid1D* host_grid = this->grid;
    
    // register the host array, if not explicitly telling not to
    /*if (register_host) {
        cudaHostRegister(this->f, sizeof(double)*ilmmax*total_point_count, cudaHostRegisterPortable);
        check_errors(__FILE__, __LINE__);
    }*/
    
    // store the processor variables to be used at downloading time
    this->processor_order_number = processor_order_number;
    this->number_of_processors = number_of_processors;
    
    size_t host_pitch = total_point_count * sizeof(double);
    
    // determine how many of the points belong to the current mpi-node
    int processor_point_count = total_point_count / this->number_of_processors 
                                + ((total_point_count % number_of_processors) > processor_order_number);
    // get the offset to the f-array caused by other processors
    int remainder = total_point_count % this->number_of_processors;
    int offset = processor_order_number * total_point_count / number_of_processors +
                 ((remainder < processor_order_number) ? remainder : processor_order_number); 
    double *device_f;
    double *host_f = &this->f[offset];
    
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        this->streamContainer->setDevice(device);
        // get the preallocated device pointers, 
        // NOTE: The memory of bubble is allocated for its entire
        // length, thus we have to go to the part we want to upload
        device_f = this->device_f[device];
        device_f = &device_f[offset];
        
        // detemine how many of the mpi-nodes points belong to this device (gpu)
        int device_point_count = processor_point_count / this->streamContainer->getNumberOfDevices() +
                                 ((processor_point_count % this->streamContainer->getNumberOfDevices()) > device);
        for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream ++) {
            // detemine the number of the points handled by this stream
            int stream_point_count = device_point_count / this->streamContainer->getStreamsPerDevice() +
                                    ((device_point_count % this->streamContainer->getStreamsPerDevice()) > stream);  
            // upload the data to device, copy all ilmmax-rows for stream_point_count columns
            cudaMemcpy2DAsync((void *) device_f, this->device_f_pitch[device],
                              (void *) host_f, host_pitch,
                              stream_point_count * sizeof(double),
                              ilmmax, 
                              cudaMemcpyHostToDevice,
                              *this->streamContainer->getStream(device, stream)
                             );
            check_errors(__FILE__, __LINE__);
                
            offset += stream_point_count;
            device_f += stream_point_count;
            host_f += stream_point_count;
        }
        
        // copy the bubble to the device, for which set the device pointers
        // to be the main-pointers
        this->f = this->device_f[device];
        this->cf = this->device_cf[device];
        this->df = this->device_df[device];
        this->grid = host_grid->device_copies[device];
        this->lmax = lmax;
        
        // copy the bubble to device
        cudaMemcpyAsync(this->device_copies[device], this, sizeof(Bubble), cudaMemcpyHostToDevice, 
                              *this->streamContainer->getStream(device, 0));
        check_errors(__FILE__, __LINE__);
        
        this->f = f;
        this->grid = host_grid;
    }
    
    
    // and synchronize the host with the device
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        this->uploaded_events[device] = this->streamContainer->recordDeviceEvent(device);
    }
}

void Bubble::waitBubbleUploaded(int device, cudaStream_t *stream) {
    cudaStreamWaitEvent(*stream, *this->uploaded_events[device], 0);
}

void Bubble::waitBubbleUploaded(int device) {
    cudaStreamWaitEvent(0, *this->uploaded_events[device], 0);
}

/*
 * Sets bubble values to zero
 * 
 * NOTE: in order to use this, the bubble uploaded (i.e., the f-array given as input)
 * must have the same lmax value as the Bubble-object we are uploading to.
 * NOTE: registers the input array but does not unregister it, thus after calling this
 * the user must unregister the f elsewhere, for instance by calling the unregister function.
 * NOTE: this function is designed to function together with the bubble multiplication
 */
void Bubble::setToZero() {
    // set the host variables and register them for faster data transfer
    this->f = f;
    check_errors(__FILE__, __LINE__);
    
    int ilmmax = (this->device_memory_lmax + 1) * (this->device_memory_lmax + 1);
    // calculate the total number of points in the bubbles each l,m -pair, 
    int total_point_count = this->grid->ncell * (this->grid->nlip - 1) +1;
    
    
    // determine how many of the points belong to the current mpi-node
    int processor_point_count = total_point_count / this->number_of_processors 
                                + ((total_point_count % this->number_of_processors) > this->processor_order_number);
    // get the offset to the f-array caused by other processors
    int remainder = total_point_count % this->number_of_processors;
    int offset = this->processor_order_number * total_point_count / this->number_of_processors +
                 ((remainder < this->processor_order_number) ? remainder : this->processor_order_number); 
    double *device_f;
    
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        this->streamContainer->setDevice(device);
        
        // get the preallocated device pointers, 
        // NOTE: The memory of bubble is allocated for its entire
        // length, thus we have to go to the part we want to upload
        device_f = this->device_f[device];
        device_f = &device_f[offset];
        
        // detemine how many of the mpi-nodes points belong to this device (gpu)
        int device_point_count = processor_point_count / this->streamContainer->getNumberOfDevices() +
                                 ((processor_point_count % this->streamContainer->getNumberOfDevices()) > device);
        for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream ++) {
            // detemine the number of the points handled by this stream
            int stream_point_count = device_point_count / this->streamContainer->getStreamsPerDevice() +
                                    ((device_point_count % this->streamContainer->getStreamsPerDevice()) > stream);                      
            // upload the data to device, copy all ilmmax-rows for stream_point_count columns
            cudaMemset2DAsync((void *) device_f, this->device_f_pitch[device],
                              0,
                              stream_point_count * sizeof(double),
                              ilmmax, 
                              *this->streamContainer->getStream(device, stream)
                             );
            check_errors(__FILE__, __LINE__);
                
            offset += stream_point_count;
            device_f += stream_point_count;
        }
    }
}

/*
 * Downloads part of a bubble from the device. Downloads to host exactly the same
 * part as the upload function above uploads to device.
 * 
 * NOTE: this function is designed to function together with the bubble multiplication &
 *       summation
 */
void Bubble::download(int lmax) {
    // calculate the total number of points in the bubbles each l,m -pair, 
    int total_point_count = this->grid->ncell * (this->grid->nlip - 1) +1;
    size_t host_pitch = total_point_count * sizeof(double);
    int ilmmax = (lmax + 1) * (lmax + 1);
    
    // determine how many of the points belong to the current mpi-node
    int processor_point_count = total_point_count / this->number_of_processors 
                                + ((total_point_count % this->number_of_processors) > this->processor_order_number);
                             
    // get the offset to the f-array caused by other processors
    int remainder = total_point_count % this->number_of_processors;
    int offset = this->processor_order_number * total_point_count / this->number_of_processors 
                  + ((remainder < this->processor_order_number) ? remainder : this->processor_order_number); 
    double *device_f;
    double *host_f = &this->f[offset];
    check_errors(__FILE__, __LINE__);
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        this->streamContainer->setDevice(device);
        
        // get the preallocated device pointers, 
        // NOTE: The memory of bubble is allocated for its entire
        // length, thus we have to go to the part we want to upload
        device_f = this->device_f[device];
        device_f = &device_f[offset];
        
        // detemine how many of the mpi-nodes points belong to this device (gpu)
        int device_point_count = processor_point_count / this->streamContainer->getNumberOfDevices() +
                                 ((processor_point_count % this->streamContainer->getNumberOfDevices()) > device);
        for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream ++) {
            // detemine the number of the points handled by this stream
            int stream_point_count = device_point_count / this->streamContainer->getStreamsPerDevice() +
                                    ((device_point_count % this->streamContainer->getStreamsPerDevice()) > stream);
                                   
            // upload the data to device, copy all ilmmax-rows for stream_point_count columns
            cudaMemcpy2DAsync((void *) host_f, host_pitch,
                              (void *) device_f, this->device_f_pitch[device],
                              stream_point_count * sizeof(double),
                              ilmmax, 
                              cudaMemcpyDeviceToHost,
                              *this->streamContainer->getStream(device, stream)
                             );
            check_errors(__FILE__, __LINE__);
                
            offset += stream_point_count;
            device_f += stream_point_count;
            host_f += stream_point_count;
            check_errors(__FILE__, __LINE__);
        }
    }
}

/*
 * Adds together the f-values of 'this' and input bubble 'bubble'
 * 
 * NOTE: this function is designed to function together with the bubble multiplication
 * NOTE: this function assumes that the bubbles have identical grids and with that, 
 *       identical f_pitches
 */
void Bubble::add(Bubble *bubble) {
    
    // make sure that the k-values of the input functions are the same
    // this is done by decreasing the larger k-value to be equal
    // with the smaller
    check_errors(__FILE__, __LINE__);
    if (this->k > bubble->k) {
        this->decreaseK(this->k - bubble->k);   
    }
    else if (this->k < bubble->k) {
        bubble->decreaseK(bubble->k - this->k); 
    }
    check_errors(__FILE__, __LINE__);
    
    // calculate the total number of points in the bubbles each l,m -pair, 
    int total_point_count = this->grid->ncell * (this->grid->nlip - 1) +1;
    int smaller_lmax = min(this->lmax, bubble->lmax);

    
    // determine how many of the points belong to the current mpi-node
    int processor_point_count = total_point_count / this->number_of_processors 
                                + ((total_point_count % this->number_of_processors) > this->processor_order_number);
                                
    // get the offset to the f-array caused by other processors
    int remainder = total_point_count % this->number_of_processors;
    int offset = this->processor_order_number * total_point_count / this->number_of_processors 
                  + ((remainder < this->processor_order_number) ? remainder : this->processor_order_number); 
    double *device_f;
    double *device_f1;
    
    int block_size = 256;
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        this->streamContainer->setDevice(device);
        this->waitBubbleUploaded(device);
        bubble->waitBubbleUploaded(device);
        
        // get the preallocated device pointers, 
        // NOTE: The memory of bubble is allocated for its entire
        // length, thus we have to go to the part we want to upload
        device_f = this->device_f[device];
        device_f = &device_f[offset];
        device_f1 = bubble->device_f[device];
        device_f1 = &device_f1[offset];
        
        // detemine how many of the mpi-nodes points belong to this device (gpu)
        int device_point_count = processor_point_count / this->streamContainer->getNumberOfDevices() +
                                 ((processor_point_count % this->streamContainer->getNumberOfDevices()) > device);
        for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream ++) {
            // detemine the number of the points handled by this stream
            int stream_point_count = device_point_count / this->streamContainer->getStreamsPerDevice() +
                                    ((device_point_count % this->streamContainer->getStreamsPerDevice()) > stream);
                                    
            int grid_size = (stream_point_count + block_size - 1) / block_size;
            
            // call the kernel
            Bubble_sum_kernel <<<grid_size, block_size, 0, *this->streamContainer->getStream(device, stream)>>> 
                (device_f, device_f1, smaller_lmax, stream_point_count, this->device_f_pitch[device]);
                                   
           
            check_errors(__FILE__, __LINE__);
                
            // add the device pointers and the offset
            offset += stream_point_count;
            device_f += stream_point_count;
            device_f1 += stream_point_count;
        }
    }
}

/*
 * Decreases the k-value of a bubble by k_decrease
 * 
 * NOTE: this function is designed to function together with the bubble multiplication
 * NOTE: this function assumes that the bubbles have identical grids and with that, 
 *       identical f_pitches
 */
void Bubble::decreaseK(int k_decrease) {
    
    
    // calculate the total number of points in the bubbles each l,m -pair, 
    int total_point_count = this->grid->ncell * (this->grid->nlip - 1) +1;
    
    // determine how many of the points belong to the current mpi-node
    int processor_point_count = total_point_count / this->number_of_processors 
                                + ((total_point_count % this->number_of_processors) > this->processor_order_number);
                                
    // get the offset to the f-array caused by other processors
    int remainder = total_point_count % this->number_of_processors;
    int offset = this->processor_order_number * total_point_count / this->number_of_processors 
                  + ((remainder < this->processor_order_number) ? remainder : this->processor_order_number); 
    double *device_f;
    double *device_r;
    
    int block_size = 256;
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        this->streamContainer->setDevice(device);
        
        this->waitBubbleUploaded(device);
        
        // get the preallocated device pointers, 
        // NOTE: The memory of bubble is allocated for its entire
        // length, thus we have to go to the part we want to upload
        device_f = this->device_f[device];
        device_f = &device_f[offset];
        device_r = this->grid->device_gridpoints[device];
        device_r = &device_r[offset];
        
        // detemine how many of the mpi-nodes points belong to this device (gpu)
        int device_point_count = processor_point_count / this->streamContainer->getNumberOfDevices() +
                                 ((processor_point_count % this->streamContainer->getNumberOfDevices()) > device);
        for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream ++) {
            // detemine the number of the points handled by this stream
            int stream_point_count = device_point_count / this->streamContainer->getStreamsPerDevice() +
                                    ((device_point_count % this->streamContainer->getStreamsPerDevice()) > stream);
                                    
            int grid_size = (stream_point_count + block_size - 1) / block_size;
            
            // call the kernel
            Bubble_decrease_k_kernel <<<grid_size, block_size, 0, *this->streamContainer->getStream(device, stream)>>> 
               (device_f, device_r, k_decrease, this->lmax, stream_point_count, this->device_f_pitch[device]);
                                   
           
            check_errors(__FILE__, __LINE__);
                
            // add the device pointers and the offset
            offset += stream_point_count;
            device_f += stream_point_count;
            device_r += stream_point_count;
        }
    }
}

/*
 * Integrates over the bubble. We only need to integrate over the s-bubble.
 */ 
double Bubble::integrate() {
    
    // calculate the total number of points in the bubbles each l,m -pair, 
    int total_point_count = this->grid->getShape();
    
    // check if the integrator has been inited, if not, init it
    if (!this->integrator) {
        this->integrator = new Integrator1D(this->streamContainer, this->grid, this->processor_order_number, this->number_of_processors);
    }
    
    // upload the l,m=0 radial function f to the integrator
    this->integrator->upload(this->f);
    check_errors(__FILE__, __LINE__);
    
    // determine how many of the points belong to the current mpi-node
    int processor_point_count = total_point_count / this->number_of_processors 
                                + ((total_point_count % this->number_of_processors) > this->processor_order_number);
                                
    // get the offset to the f-array caused by other processors
    int remainder = total_point_count % this->number_of_processors;
    int offset = this->processor_order_number * total_point_count / this->number_of_processors 
                  + ((remainder < this->processor_order_number) ? remainder : this->processor_order_number); 
                  
    // get the partial s-bubble device vectors residing now in the integrators device memory
    double **device_vectors = this->integrator->getDeviceVectors();
    double *device_vector;
    double *device_r;
    
    // multiply the integration vector with r^(2+this->k)
    // get the times we have to multiply the vector with r, i.e., 2+this->k
    // NOTE: this must be larger or equal to zero
    int k_change = 2 + this->k;
    if (k_change > 0) {
        int block_size = 256;
        for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
            this->streamContainer->setDevice(device);
            
            // get the preallocated device pointers, 
            // NOTE: The memory of gridpoints is allocated for its entire
            // length, thus we have to go to the part we want to upload
            // however, the integrator only has the memory it needs, the we don't need to
            // offset the device_vector
            device_vector = device_vectors[device];
            device_r = this->grid->device_gridpoints[device];
            device_r = &device_r[offset];
            
            // detemine how many of the mpi-nodes points belong to this device (gpu)
            int device_point_count = processor_point_count / this->streamContainer->getNumberOfDevices() +
                                    ((processor_point_count % this->streamContainer->getNumberOfDevices()) > device);
            for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream ++) {
                // detemine the number of the points handled by this stream
                int stream_point_count = device_point_count / this->streamContainer->getStreamsPerDevice() +
                                        ((device_point_count % this->streamContainer->getStreamsPerDevice()) > stream);
                                        
                int grid_size = (stream_point_count + block_size - 1) / block_size;
                
                // call the decrease_k- kernel by using lmax = 0 
                Bubble_decrease_k_kernel <<<grid_size, block_size, 0, *this->streamContainer->getStream(device, stream)>>> 
                (device_vector, device_r, k_change, 0, stream_point_count, 0);
                check_errors(__FILE__, __LINE__);
                                    
            
                    
                // add the device pointers and the offset
                offset += stream_point_count;
                device_vector += stream_point_count;
                device_r += stream_point_count;
                check_errors(__FILE__, __LINE__);
            }
        }
    }
    else if (k_change < 0) {
        printf("Invalid k-value (%d) at bubble-integrate, must be larger or equal with -2. At file '%s', line number %d", this->k, __FILE__, __LINE__);
        exit(-1);
    }
    
    return  4.0 * M_PI * this->integrator->integrate(); // 
}

void Bubble::registerHost(double *f) {
    check_errors(__FILE__, __LINE__);
    this->f = f;
    /*int ilmmax = (this->lmax + 1) * (this->lmax + 1);
    // calculate the total number of points in the bubbles each l,m -pair, 
    int total_point_count = this->grid->ncell * (this->grid->nlip - 1) +1;
    cudaHostRegister(this->f, sizeof(double)*ilmmax*total_point_count, cudaHostRegisterPortable);*/
    check_errors(__FILE__, __LINE__);
}



void Bubble::destroy() {
    //this->grid->destroy();
    //check_errors(__FILE__, __LINE__);
    //delete this->grid;
    this->grid = NULL;
    
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device++) {
        this->streamContainer->setDevice(device);
        
        cudaFree(this->device_f[device]);
        check_errors(__FILE__, __LINE__);
        cudaFree(this->device_cf[device]);
        check_errors(__FILE__, __LINE__);
        cudaFree(this->device_df[device]);
        check_errors(__FILE__, __LINE__);
        cudaFree(this->device_copies[device]);
        check_errors(__FILE__, __LINE__);
    }
    delete[] this->device_copies;
    delete[] this->device_f;
    delete[] this->device_df;
    delete[] this->device_f_pitch;
    delete[] this->device_cf;
    delete[] this->uploaded_events;
   
 
    // check if integrator is null pointer, if not
    // delete the integrator
    if (this->integrator) {
        this->integrator->destroy();
        delete this->integrator;
        this->integrator = NULL;
    }
    check_errors(__FILE__, __LINE__);
    //cudaHostUnregister(this);
}

/*
 * Set MPI-configuration used by the bubble object.
 */ 
void Bubble::setProcessorConfiguration( int processor_order_number, int number_of_processors) {
    this->number_of_processors = number_of_processors;
    this->processor_order_number = processor_order_number;
}

/**************************************************************
 *  Bubbles-implementation                                    *
 **************************************************************/


int Bubbles::getBubbleCount() {
    return this->nbub;
}

Bubbles::Bubbles(int nbub) {
    this->nbub = nbub;
    this->bubbles = new Bubble*[nbub];
    this->is_sub_bubbles = false;
}

/*
 * Init new Bubbles by making a copy of the old. 
 * 
 * NOTE: This makes a deep copy of the old bubbles, meaning that 
 *       new memory places are allocated for the underlying Bubble objects.
 */ 
Bubbles::Bubbles(Bubbles *old_bubbles, int lmax, int k) {
    this->is_sub_bubbles = false;
    this->nbub = old_bubbles->nbub;
    this->bubbles = new Bubble*[nbub];
    for (int i = 0; i < old_bubbles->getBubbleCount(); i++) {
        this->bubbles[i] = new Bubble(old_bubbles->bubbles[i], lmax, k);
    }
}

/*
 * Get new bubbles object containing some of the original bubbles.
 * The bubbles selected in the new objects are the ones with 
 * the ibub values matching to those in input parameter 'ibubs'.
 * NOTE: this function makes a shallow copy of the input bubbles 'this',
 *       i.e., the underlying Bubble objects are copied as references only 
 */
Bubbles *Bubbles::getSubBubbles(int *ibubs, int nbub) {
    Bubbles *new_bubbles = new Bubbles(nbub);
    new_bubbles->is_sub_bubbles = true;
    
    // copy the references to the wanted Bubble-objects specified
    // in ibubs
    for (int i = 0; i < new_bubbles->getBubbleCount(); i++) {
        new_bubbles->bubbles[i] = this->getBubble(ibubs[i]);
    }
    return new_bubbles;
}

/*
 * Get the pointer to the Bubble with local order number 'i' equal to
 * input parameter 'i'. If not found NULL is returned.
 * 
 * @param i  - The local order number of the bubble
 */
Bubble *Bubbles::getBubbleWithLocalOrderNumber(int i) {
    if (i < this->nbub) {
        return this->bubbles[i];
    }
    return NULL;
}

/*
 * Get the pointer to the Bubble with global order number 'ibub' equal to
 * input parameter 'ibub'. If not found NULL is returned.
 * 
 * @param ibub  - The global order number of the bubble
 */
Bubble *Bubbles::getBubble(int ibub) {
    for (int i = 0; i < this->getBubbleCount(); i ++) {
        if (this->bubbles[i]->ibub == ibub) {
            return this->bubbles[i];
        }
    }
    return NULL;
}

/*
 * Check if the Bubbles contains a Bubble with global order number 'ibub'.
 * 
 * @param ibub  - The global order number of the bubble
 */
bool Bubbles::containsBubble(int ibub) {
    Bubble *bubble = this->getBubble(ibub);
    return (bubble != NULL);
}

/*
 * Init a bubble with global order number 'ibub' to the 'i':th slot in the
 * internal bubbles array. Contains also the values for the bubble.
 * 
 * @param grid - The grid used in the bubble
 * @param i    - The internal order number of the bubble
 * @param ibub - The global order number of the bubble
 * @param center - The global center point of the bubble
 * @param lmax   - The maximum value of quantum number 'l' for the bubble
 * @param k      - The parameter k for the r^k multiplier of the values
 * @param bf     - The values of the bubble
 * @param charge - The charge of the atom at the center of the bubble
 * @param streaContainer - The container holding the streams used in cuda evaluation of anything
 *                         related to this object 
 */
void Bubbles::initBubble(Grid1D *grid, int i, int ibub, double center[3], int lmax,
                        int k, double *bf, double charge, StreamContainer *streamContainer) {
    this->bubbles[i] = new Bubble(ibub, grid, center, lmax, k, bf, charge, streamContainer);
}

/*
 * Init a bubble with global order number 'ibub' to the 'i':th slot in the
 * internal bubbles array. Contains also the values for the bubble.
 * 
 * @param grid - The grid used in the bubble
 * @param i    - The internal order number of the bubble
 * @param ibub - The global order number of the bubble
 * @param center - The global center point of the bubble
 * @param lmax   - The maximum value of quantum number 'l' for the bubble
 * @param k      - The parameter k for the r^k multiplier of the values
 * @param charge - The charge of the atom at the center of the bubble
 * @param streaContainer - The container holding the streams used in cuda evaluation of anything
 *                         related to this object 
 */
void Bubbles::initBubble(Grid1D *grid, int i, int ibub, double center[3], int lmax,
                        int k, double charge, StreamContainer *streamContainer) {
    
    
    check_errors(__FILE__, __LINE__);
    this->bubbles[i] = new Bubble(ibub, grid, center, lmax, k, charge, streamContainer);
}

void Bubbles::unregister() {
    /*for (int ibub = 0; ibub < this->getBubbleCount(); ibub ++) {
        cudaHostUnregister(this->getBubble(ibub)->f);
        check_errors(__FILE__, __LINE__);
    }*/
}

void Bubbles::waitBubblesUploaded(int device) {
    for (int i = 0; i < this->getBubbleCount(); i ++) {
        this->bubbles[i]->waitBubbleUploaded(device);
    }
}

/*
 * Set MPI-configuration used by the bubble object.
 */ 
void Bubbles::setProcessorConfiguration( int processor_order_number, int number_of_processors) {
    for (int i = 0; i < this->getBubbleCount(); i ++) {
        this->bubbles[i]->setProcessorConfiguration(processor_order_number, number_of_processors);
    }
}

double Bubbles::integrate() {
    double result = 0.0;
    for (int i = 0; i < this->getBubbleCount(); i ++) {
        result += this->getBubbleWithLocalOrderNumber(i)->integrate();
    }
    return result;
}

void Bubbles::download() {
    for (int i = 0; i < this->getBubbleCount(); i ++) {
        this->bubbles[i]->download(this->bubbles[i]->lmax);
    }
}

void Bubbles::add(Bubbles *bubbles) {
    // go through all the Bubble-objects present in this 
    for (int i = 0; i < bubbles->getBubbleCount(); i ++) {
        // get the matching bubble in the added bubbles
        Bubble * bubble = bubbles->getBubble(this->bubbles[i]->ibub);
        
        // if the corresponding Bubble exists in both the Bubbles, do the add 
        if (bubble) {
            this->bubbles[i]->add(bubble);
        }
    }
    check_errors(__FILE__, __LINE__);
}

void Bubbles::destroy() {
    if (!this->is_sub_bubbles) {
        for (int ibub = 0; ibub < this->getBubbleCount(); ibub ++) {
            this->bubbles[ibub]->destroy();
            delete this->bubbles[ibub];
        }
    }
    delete[] this->bubbles;
}


void Bubbles::inject(Grid3D *grid3d, CudaCube *cube, int lmin, CudaCube *gradients_cube_x,
                     CudaCube *gradients_cube_y, CudaCube *gradients_cube_z, bool evaluate_value,
                     bool evaluate_gradients_x, bool evaluate_gradients_y, bool evaluate_gradients_z) {
    check_errors(__FILE__, __LINE__);
    int total_slice_count = cube->getShape(Z_);
    // the minimum l is 0 always in the multiplication
    int device_slice_count;
    
    // get the pointer arrays from the cubes
    double **device_cubes = cube->getDeviceCubes();
    double **device_gradients_x, **device_gradients_y, **device_gradients_z;
    // get the device gradient result pointers
    if (evaluate_gradients_x) device_gradients_x = gradients_cube_x->getDeviceCubes();
    if (evaluate_gradients_y) device_gradients_y = gradients_cube_y->getDeviceCubes();
    if (evaluate_gradients_z) device_gradients_z = gradients_cube_z->getDeviceCubes();  
    
    size_t *device_pitches = cube->getDevicePitches();
    int *device_memory_shape = cube->getDeviceMemoryShape();
    
    int slice_offset = 0;
    Bubble *bubble;
    
    StreamContainer *streamContainer = cube->getStreamContainer();
    // copy the cubes to the device & execute the kernels
    for (int device = 0; device < streamContainer->getNumberOfDevices(); device ++) {
        // set the used device (gpu)
        streamContainer->setDevice(device);
        double *dev_cube  = device_cubes[device];
        double *dev_gradient_x, *dev_gradient_y, *dev_gradient_z;
        
        // get the gradient addresses for the device
        if (evaluate_gradients_x) dev_gradient_x = device_gradients_x[device];
        if (evaluate_gradients_y) dev_gradient_y = device_gradients_y[device];
        if (evaluate_gradients_z) dev_gradient_z = device_gradients_z[device];
        
        // calculate the number of vectors this device handles
        device_slice_count =  total_slice_count / streamContainer->getNumberOfDevices()
                                  + ((total_slice_count % streamContainer->getNumberOfDevices()) > device);                
        for (int stream = 0; stream < streamContainer->getStreamsPerDevice(); stream++) {
            
            // determine the count of vectors handled by this stream
            int slice_count = device_slice_count / streamContainer->getStreamsPerDevice() 
                                + ((device_slice_count % streamContainer->getStreamsPerDevice()) > stream);

            check_errors(__FILE__, __LINE__);
            
            // get the launch configuration for the f1-inject
            dim3 block, grid;
            cube->getLaunchConfiguration(&grid, &block, slice_count, INJECT_BLOCK_SIZE);
            if (slice_count > 0) {
                // inject bubbles to the cube
                for (int i = 0; i < this->getBubbleCount(); i++) {
                    bubble = this->getBubbleWithLocalOrderNumber(i);
                    // wait that the bubble is uploaded to the device before starting
                    if (stream == 0) bubble->waitBubbleUploaded(device);
                    
                    // call the kernel
                    if (lmin == 0) {
                        if (evaluate_gradients_x && evaluate_gradients_y && evaluate_gradients_z) {
                            if (evaluate_value) {
                                Bubbles_evaluate_grid_gradients < true, true, true, true, true>
                                    <<< grid, block, INJECT_BLOCK_SIZE * sizeof(double) * 8, 
                                        *streamContainer->getStream(device, stream) >>>
                                                    (bubble->device_copies[device], 
                                                    dev_cube, 
                                                    dev_gradient_x,
                                                    dev_gradient_y,
                                                    dev_gradient_z,
                                                    grid3d->axis[X_]->device_gridpoints[device],
                                                    grid3d->axis[Y_]->device_gridpoints[device],
                                                    grid3d->axis[Z_]->device_gridpoints[device],
                                                    grid3d->shape[X_],
                                                    grid3d->shape[Y_],
                                                    grid3d->shape[Z_],
                                                    bubble->crd[X_], 
                                                    bubble->crd[Y_],
                                                    bubble->crd[Z_],
                                                    bubble->k,
                                                    slice_offset,
                                                    device_pitches[device],
                                                    device_memory_shape[Y_], 
                                                    slice_count,
                                                    1.0);
                            }
                            else {
                                Bubbles_evaluate_grid_gradients < true, false, true, true, true>
                                    <<< grid, block, INJECT_BLOCK_SIZE * sizeof(double) * 8, 
                                        *streamContainer->getStream(device, stream) >>>
                                                    (bubble->device_copies[device], 
                                                    dev_cube, 
                                                    dev_gradient_x,
                                                    dev_gradient_y,
                                                    dev_gradient_z,
                                                    grid3d->axis[X_]->device_gridpoints[device],
                                                    grid3d->axis[Y_]->device_gridpoints[device],
                                                    grid3d->axis[Z_]->device_gridpoints[device],
                                                    grid3d->shape[X_],
                                                    grid3d->shape[Y_],
                                                    grid3d->shape[Z_],
                                                    bubble->crd[X_], 
                                                    bubble->crd[Y_],
                                                    bubble->crd[Z_],
                                                    bubble->k,
                                                    slice_offset,
                                                    device_pitches[device],
                                                    device_memory_shape[Y_], 
                                                    slice_count,
                                                    1.0);
                            }
                        }
                        else if (evaluate_gradients_x) {
                            Bubbles_evaluate_grid_gradients < true, false, true, false, false>
                                <<< grid, block, INJECT_BLOCK_SIZE * sizeof(double) * 8, 
                                    *streamContainer->getStream(device, stream) >>>
                                                (bubble->device_copies[device], 
                                                dev_cube, 
                                                dev_gradient_x,
                                                dev_gradient_y,
                                                dev_gradient_z,
                                                grid3d->axis[X_]->device_gridpoints[device],
                                                grid3d->axis[Y_]->device_gridpoints[device],
                                                grid3d->axis[Z_]->device_gridpoints[device],
                                                grid3d->shape[X_],
                                                grid3d->shape[Y_],
                                                grid3d->shape[Z_],
                                                bubble->crd[X_], 
                                                bubble->crd[Y_],
                                                bubble->crd[Z_],
                                                bubble->k,
                                                slice_offset,
                                                device_pitches[device],
                                                device_memory_shape[Y_], 
                                                slice_count,
                                                1.0);
                        }
                        else if (evaluate_gradients_y) {
                            Bubbles_evaluate_grid_gradients < true, false, false, true, false>
                                <<< grid, block, INJECT_BLOCK_SIZE * sizeof(double) * 8, 
                                    *streamContainer->getStream(device, stream) >>>
                                                (bubble->device_copies[device], 
                                                dev_cube, 
                                                dev_gradient_x,
                                                dev_gradient_y,
                                                dev_gradient_z,
                                                grid3d->axis[X_]->device_gridpoints[device],
                                                grid3d->axis[Y_]->device_gridpoints[device],
                                                grid3d->axis[Z_]->device_gridpoints[device],
                                                grid3d->shape[X_],
                                                grid3d->shape[Y_],
                                                grid3d->shape[Z_],
                                                bubble->crd[X_], 
                                                bubble->crd[Y_],
                                                bubble->crd[Z_],
                                                bubble->k,
                                                slice_offset,
                                                device_pitches[device],
                                                device_memory_shape[Y_], 
                                                slice_count,
                                                1.0);
                        }
                        else if (evaluate_gradients_z) {
                            Bubbles_evaluate_grid_gradients < true, false, false, false, true>
                                <<< grid, block, INJECT_BLOCK_SIZE * sizeof(double) * 8, 
                                    *streamContainer->getStream(device, stream) >>>
                                                (bubble->device_copies[device], 
                                                dev_cube, 
                                                dev_gradient_x,
                                                dev_gradient_y,
                                                dev_gradient_z,
                                                grid3d->axis[X_]->device_gridpoints[device],
                                                grid3d->axis[Y_]->device_gridpoints[device],
                                                grid3d->axis[Z_]->device_gridpoints[device],
                                                grid3d->shape[X_],
                                                grid3d->shape[Y_],
                                                grid3d->shape[Z_],
                                                bubble->crd[X_], 
                                                bubble->crd[Y_],
                                                bubble->crd[Z_],
                                                bubble->k,
                                                slice_offset,
                                                device_pitches[device],
                                                device_memory_shape[Y_], 
                                                slice_count,
                                                1.0);
                        }
                        else if (evaluate_value) {
                            Bubbles_evaluate_grid_pitched
                                <<< grid, block, INJECT_BLOCK_SIZE * sizeof(double) * 8, 
                                    *streamContainer->getStream(device, stream) >>>
                                                (bubble->device_copies[device], 
                                                dev_cube, 
                                                grid3d->axis[X_]->device_gridpoints[device],
                                                grid3d->axis[Y_]->device_gridpoints[device],
                                                grid3d->axis[Z_]->device_gridpoints[device],
                                                grid3d->shape[X_],
                                                grid3d->shape[Y_],
                                                grid3d->shape[Z_],
                                                bubble->crd[X_], 
                                                bubble->crd[Y_],
                                                bubble->crd[Z_],
                                                bubble->k,
                                                slice_offset,
                                                device_pitches[device],
                                                device_memory_shape[Y_], 
                                                slice_count,
                                                1.0);
                        }
                    }
                    else if (evaluate_value) {
                        
                        Bubbles_evaluate_grid_lmin
                            <<< grid, block, INJECT_BLOCK_SIZE * sizeof(double) * 8, 
                                *streamContainer->getStream(device, stream) >>>
                                            (bubble->device_copies[device], 
                                            dev_cube, 
                                            grid3d->axis[X_]->device_gridpoints[device],
                                            grid3d->axis[Y_]->device_gridpoints[device],
                                            grid3d->axis[Z_]->device_gridpoints[device],
                                            grid3d->shape[X_],
                                            grid3d->shape[Y_],
                                            grid3d->shape[Z_],
                                            bubble->crd[X_], 
                                            bubble->crd[Y_],
                                            bubble->crd[Z_],
                                            bubble->k,
                                            slice_offset,
                                            device_pitches[device],
                                            device_memory_shape[Y_], 
                                            slice_count,
                                            lmin,
                                            1.0);
                    }
                    
                    check_errors(__FILE__, __LINE__);
                }
            }
            
            // increase the address by the number of vectors in this array
            if (evaluate_value) dev_cube += slice_count * device_pitches[device] / sizeof(double) * device_memory_shape[Y_];
            if (evaluate_gradients_x) dev_gradient_x += slice_count * device_pitches[device] / sizeof(double) * device_memory_shape[Y_];
            if (evaluate_gradients_y) dev_gradient_y += slice_count * device_pitches[device] / sizeof(double) * device_memory_shape[Y_];
            if (evaluate_gradients_z) dev_gradient_z += slice_count * device_pitches[device] / sizeof(double) * device_memory_shape[Y_];
            
            slice_offset += slice_count;
        }
    }
}

/**************************************************************
 *  BubblesEvaluator function implementations                 *
 **************************************************************/


/*
 * Evaluate the bubbles at preset points. The results are stored in the device memory.
 * 
 * @param gradient_direction - possible values X_ = 0, Y_ = 1, Z_ = 2, (X_, Y_, Z_) = 3 && this->evaluateGradients
 *                             anything else: no gradients
 */

void BubblesEvaluator::evaluatePoints(Points *result_points, Points *gradient_points_x, Points *gradient_points_y, Points *gradient_points_z, int gradient_direction) {
    
    int warp_size = 32;
    int total_warp_count = result_points->point_coordinates->number_of_points / warp_size + ((result_points->point_coordinates->number_of_points % warp_size) > 0);
    int point_offset = 0;
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        this->streamContainer->setDevice(device);
        
        // allocate space for device results and device points
        int device_warp_count = total_warp_count / this->streamContainer->getNumberOfDevices()
                                  + ((total_warp_count % this->streamContainer->getNumberOfDevices()) > device); 
        int device_point_count = device_warp_count * warp_size;
        int device_point_offset = 0;
        
        check_errors(__FILE__, __LINE__);
        // get the pointers to the device points & results
        double *device_points_ptr =  result_points->point_coordinates->device_coordinates[device];
        double *device_results_ptr = result_points->device_values[device];
        double *device_gradients_x_ptr = NULL;
        double *device_gradients_y_ptr = NULL;
        double *device_gradients_z_ptr = NULL;
        
        if (gradient_direction == 3) {
            device_gradients_x_ptr = gradient_points_x->device_values[device];
            device_gradients_y_ptr = gradient_points_y->device_values[device];
            device_gradients_z_ptr = gradient_points_z->device_values[device];
        }
        else if (gradient_direction < 3 && gradient_direction >= 0) {
            device_gradients_x_ptr = result_points->device_values[device];
            device_gradients_y_ptr = result_points->device_values[device];
            device_gradients_z_ptr = result_points->device_values[device];
        }
        
        for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream ++) {
            // get the number of points that are in the responsibility of this stream
            int stream_warp_count =  device_warp_count / this->streamContainer->getStreamsPerDevice()
                                  + ((device_warp_count % streamContainer->getStreamsPerDevice()) > stream); 
            int stream_point_count = stream_warp_count * warp_size;
            
            // make sure that the last stream does not go over board
            if (stream_point_count + point_offset > result_points->point_coordinates->number_of_points) {
                stream_point_count = result_points->point_coordinates->number_of_points - point_offset;
            }
            
            check_errors(__FILE__, __LINE__);
            
            if (stream_point_count > 0) {
                for (int i = 0; i < this->bubbles->getBubbleCount(); i++) {
                    Bubble *bubble = this->bubbles->getBubbleWithLocalOrderNumber(i);
                    
                    // wait that the bubble is uploaded before calling the kernel
                    if (stream == 0) bubble->waitBubbleUploaded(device);
                    
                    int grid_size = (stream_point_count + INJECT_BLOCK_SIZE - 1) / INJECT_BLOCK_SIZE;
                    //printf("ibub: %d, device: %d, stream: %d, grid_size: %d, block_size: %d, stream_point_count: %d, device_point_offset: %d, device_point_count: %d, point_count: %d\n",
                    //      ibub, device, stream, grid_size, INJECT_BLOCK_SIZE, stream_point_count, device_point_offset, device_point_count, this->point_count);
                    if (gradient_direction == X_) {
                        Bubbles_evaluate_gradient_points
                            <true, false, true, false, false>
                            <<< grid_size, INJECT_BLOCK_SIZE, INJECT_BLOCK_SIZE * sizeof(double) * 7, 
                                *this->streamContainer->getStream(device, stream) >>>
                                    (bubble->device_copies[device],
                                    device_results_ptr,
                                    device_gradients_x_ptr,
                                    device_gradients_y_ptr,
                                    device_gradients_z_ptr,
                                    device_points_ptr,
                                    device_point_count,
                                    bubble->crd[X_], 
                                    bubble->crd[Y_],
                                    bubble->crd[Z_],
                                    bubble->k,
                                    stream_point_count,
                                    device_point_offset,
                                    1.0
                                    );
                    }
                    else if (gradient_direction == Y_) {
                        Bubbles_evaluate_gradient_points
                            <true, false, false, true, false>
                            <<< grid_size, INJECT_BLOCK_SIZE, INJECT_BLOCK_SIZE * sizeof(double) * 7, 
                                *this->streamContainer->getStream(device, stream) >>>
                                    (bubble->device_copies[device],
                                    device_results_ptr,
                                    device_gradients_x_ptr,
                                    device_gradients_y_ptr,
                                    device_gradients_z_ptr,
                                    device_points_ptr,
                                    device_point_count,
                                    bubble->crd[X_], 
                                    bubble->crd[Y_],
                                    bubble->crd[Z_],
                                    bubble->k,
                                    stream_point_count,
                                    device_point_offset,
                                    1.0
                                    );
                    }
                    else if (gradient_direction == Z_) {
                        Bubbles_evaluate_gradient_points
                            <true, false, false, false, true>
                            <<< grid_size, INJECT_BLOCK_SIZE, INJECT_BLOCK_SIZE * sizeof(double) * 7, 
                                *this->streamContainer->getStream(device, stream) >>>
                                    (bubble->device_copies[device],
                                    device_results_ptr,
                                    device_gradients_x_ptr,
                                    device_gradients_y_ptr,
                                    device_gradients_z_ptr,
                                    device_points_ptr,
                                    device_point_count,
                                    bubble->crd[X_], 
                                    bubble->crd[Y_],
                                    bubble->crd[Z_],
                                    bubble->k,
                                    stream_point_count,
                                    device_point_offset,
                                    1.0
                                    );
                    }
                    else if (gradient_direction == 3) {
                        
                        Bubbles_evaluate_gradient_points
                            <true, true, true, true, true>
                            <<< grid_size, INJECT_BLOCK_SIZE, INJECT_BLOCK_SIZE * sizeof(double) * 7, 
                                *this->streamContainer->getStream(device, stream) >>>
                                    (bubble->device_copies[device],
                                    device_results_ptr,
                                    device_gradients_x_ptr,
                                    device_gradients_y_ptr,
                                    device_gradients_z_ptr,
                                    device_points_ptr,
                                    device_point_count,
                                    bubble->crd[X_], 
                                    bubble->crd[Y_],
                                    bubble->crd[Z_],
                                    bubble->k,
                                    stream_point_count,
                                    device_point_offset,
                                    1.0
                                    );
                    }
                    else {
                        Bubbles_evaluate_points_simple
                            <<< grid_size, INJECT_BLOCK_SIZE, INJECT_BLOCK_SIZE * sizeof(double) * 7, 
                                *this->streamContainer->getStream(device, stream) >>>
                                    (bubble->device_copies[device],
                                        device_results_ptr,
                                        device_points_ptr,
                                        device_point_count,
                                        bubble->crd[X_], 
                                        bubble->crd[Y_],
                                        bubble->crd[Z_],
                                        bubble->k,
                                        stream_point_count,
                                        device_point_offset,
                                        1.0
                                    );
                    }
                    check_errors(__FILE__, __LINE__);
                }
            }
            // add the pointers
            point_offset += stream_point_count;
            device_point_offset += stream_point_count;
        }
        check_errors(__FILE__, __LINE__);
    }
}


/**************************************************************
 *  Function3DMultiplier-implementation                           *
 **************************************************************/

/* 
 * Injects the f1_bubbles to this->cube1 and f2_bubbles to this->cube2,
 * multiplies this->cube1 with this->cube2 and de-injects the 'result_bubbles'
 * from 'this->cube1'
 * 
 * @param f1_bubbles
 * @param f2_bubbles
 * @param result_bubbles
 */
void Function3DMultiplier::multiply(Bubbles *f1_bubbles, Bubbles *f2_bubbles, Bubbles *result_bubbles) {
    int total_slice_count = this->cube1->getShape(Z_);
    // the minimum l is 0 always in the multiplication
    int device_slice_count;
    
    // get the pointer arrays from the cubes
    double **f1_device_cubes = this->cube1->getDeviceCubes();
    size_t *f1_device_pitches = this->cube1->getDevicePitches();
    double **f2_device_cubes = this->cube2->getDeviceCubes();
    size_t *f2_device_pitches = this->cube2->getDevicePitches();
    
    int *f1_device_memory_shape = this->cube1->getDeviceMemoryShape();
    int *f2_device_memory_shape = this->cube2->getDeviceMemoryShape();
    int f1_shape[3];
    f1_shape[X_] = this->cube1->getShape(X_);
    f1_shape[Y_] = this->cube1->getShape(Y_);
    f1_shape[Z_] = this->cube1->getShape(Z_);
    
    int f2_shape[3];
    f2_shape[X_] = this->cube2->getShape(X_);
    f2_shape[Y_] = this->cube2->getShape(Y_);
    f2_shape[Z_] = this->cube2->getShape(Z_);
    
    int slice_offset = 0;
    Bubble *bubble;
    
    // copy the cubes to the device & execute the kernels
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        // set the used device (gpu)
        this->streamContainer->setDevice(device);
        
        //cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        
        //int first_block = 0;
        double *dev_f1_cube  = f1_device_cubes[device];
        double *dev_f2_cube  = f2_device_cubes[device];
        
        // calculate the number of vectors this device handles
        device_slice_count =  total_slice_count / this->streamContainer->getNumberOfDevices()
                                  + ((total_slice_count % this->streamContainer->getNumberOfDevices()) > device);                
        for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream++) {
            
            // determine the count of vectors handled by this stream
            int slice_count = device_slice_count / this->streamContainer->getStreamsPerDevice() 
                                + ((device_slice_count % this->streamContainer->getStreamsPerDevice()) > stream);
            if (slice_count > 0) {
                // get the launch configuration for the f1-inject
                dim3 block, grid;
                this->cube1->getLaunchConfiguration(&grid, &block, slice_count, INJECT_BLOCK_SIZE);
                
                check_errors(__FILE__, __LINE__);
                // inject the f1 bubbles to the f1_cube (and sum)
                for (int i = 0; i < f1_bubbles->getBubbleCount(); i++) {
                    bubble = f1_bubbles->getBubbleWithLocalOrderNumber(i);
                    // wait that the bubble is uploaded to the device before starting
                    if (stream == 0) bubble->waitBubbleUploaded(device);
                    
                    Bubbles_evaluate_grid_pitched 
                        <<< grid, block, INJECT_BLOCK_SIZE * sizeof(double) * 7, 
                            *this->streamContainer->getStream(device, stream) >>>
                                        (bubble->device_copies[device], 
                                        dev_f1_cube,
                                        this->grid->axis[X_]->device_gridpoints[device],
                                        this->grid->axis[Y_]->device_gridpoints[device],
                                        this->grid->axis[Z_]->device_gridpoints[device],
                                        f1_shape[X_],
                                        f1_shape[Y_],
                                        f1_shape[Z_],
                                        bubble->crd[X_], 
                                        bubble->crd[Y_],
                                        bubble->crd[Z_],
                                        bubble->k,
                                        slice_offset,
                                        f1_device_pitches[device],
                                        f1_device_memory_shape[Y_], 
                                        slice_count,
                                        1.0);
                                        
                    check_errors(__FILE__, __LINE__);
                }

                check_errors(__FILE__, __LINE__);
                
                // get the launch configuration for the f2-inject
                this->cube2->getLaunchConfiguration(&grid, &block, slice_count, INJECT_BLOCK_SIZE);
                
                // inject the f2 bubbles to the f1_cube (and sum)
                for (int i = 0; i < f2_bubbles->getBubbleCount(); i++) {
                    bubble = f2_bubbles->getBubbleWithLocalOrderNumber(i);
                    
                    // wait that the bubble is uploaded to the device before starting
                    if (stream == 0) bubble->waitBubbleUploaded(device);
                    
                    // call the kernel
                    Bubbles_evaluate_grid_pitched 
                        <<< grid, block, INJECT_BLOCK_SIZE * sizeof(double) * 7, 
                            *this->streamContainer->getStream(device, stream) >>>
                                        (bubble->device_copies[device], 
                                        dev_f2_cube, 
                                        this->grid->axis[X_]->device_gridpoints[device],
                                        this->grid->axis[Y_]->device_gridpoints[device],
                                        this->grid->axis[Z_]->device_gridpoints[device],
                                        f2_shape[X_],
                                        f2_shape[Y_],
                                        f2_shape[Z_],
                                        bubble->crd[X_], 
                                        bubble->crd[Y_],
                                        bubble->crd[Z_],
                                        bubble->k,
                                        slice_offset,
                                        f2_device_pitches[device],
                                        f2_device_memory_shape[Y_],
                                        slice_count, 
                                        1.0);
                    check_errors(__FILE__, __LINE__);
                }
                
                // get the launch configuration for the multiplication and result-inject
                this->cube2->getLaunchConfiguration(&grid, &block, slice_count, INJECT_BLOCK_SIZE);
                
                // multiply dev_f1_cube with dev_f2_cube and store the result to dev_f1_cube
                multiply_3d_cubes(dev_f1_cube, f1_shape[X_], f1_shape[Y_], f1_device_memory_shape[Y_], f1_device_pitches[device],
                                dev_f2_cube, f2_shape[X_], f2_shape[Y_], f2_device_memory_shape[Y_], f2_device_pitches[device],
                                slice_count, &grid, &block, this->streamContainer->getStream(device, stream));
                check_errors(__FILE__, __LINE__);
                
                // de-inject (deduct) the result bubbles from the dev_f1_cube
                for (int i = 0; i < result_bubbles->getBubbleCount(); i++) {
                    bubble = result_bubbles->getBubbleWithLocalOrderNumber(i);
                    
                    // wait that the bubble is uploaded to the device before starting
                    if (stream == 0) bubble->waitBubbleUploaded(device);
                    
                    // call the kernel
                    Bubbles_evaluate_grid_pitched 
                        <<< grid, block, INJECT_BLOCK_SIZE * sizeof(double) * 7, 
                            *this->streamContainer->getStream(device, stream) >>>
                                        (bubble->device_copies[device], 
                                        dev_f1_cube, 
                                        this->grid->axis[X_]->device_gridpoints[device],
                                        this->grid->axis[Y_]->device_gridpoints[device],
                                        this->grid->axis[Z_]->device_gridpoints[device],
                                        f1_shape[X_],
                                        f1_shape[Y_],
                                        f1_shape[Z_],
                                        bubble->crd[X_], 
                                        bubble->crd[Y_],
                                        bubble->crd[Z_],
                                        bubble->k,
                                        slice_offset,
                                        f1_device_pitches[device],
                                        f1_device_memory_shape[Y_], 
                                        slice_count,
                                        -1.0);
                    check_errors(__FILE__, __LINE__);
                }
                
                // increase the address by the number of vectors in this array
                // something else
                dev_f1_cube += slice_count * f1_device_pitches[device] / sizeof(double) * f1_device_memory_shape[Y_];
                dev_f2_cube += slice_count * f2_device_pitches[device] / sizeof(double) * f2_device_memory_shape[Y_];
                slice_offset += slice_count;
            }
        }
    }
}



/********************************************
 *  Fortran interfaces                      *
 ********************************************/

extern "C" void bubbles_add_cuda(Bubbles *bubbles, Bubbles *bubbles1) {
    bubbles->add(bubbles1);
}

extern "C" Bubbles* bubbles_get_sub_bubbles_cuda(Bubbles *bubbles, int *ibubs, int nbub) {
    return bubbles->getSubBubbles(ibubs, nbub);
}

extern "C" Bubbles *bubbles_init_cuda(int nbub) {
    Bubbles *new_bubbles = new Bubbles(nbub);
    check_errors(__FILE__, __LINE__);
    return new_bubbles;
}

/*
 * 
 * @param id - local index of the bubble inited in Fortran format: first index is 1.
 */
extern "C" void bubble_init_cuda(Bubbles *bubbles, Grid1D *grid, int i, int ibub, double center[3], int lmax,
                        int k, double charge, StreamContainer *streamContainer) {
    bubbles->initBubble(grid, i-1, ibub, center, lmax, k, charge, streamContainer);
    check_errors(__FILE__, __LINE__);
}

/*
 * Upload the content ('bf') of the Bubble with global order number 'ibub' to the device.
 * 
 * @param ibub - tHe global order number of the bubble
 */
extern "C" void bubble_upload_all_cuda(Bubbles *bubbles, int ibub, int lmax, int k, double *bf) {
    if (bubbles->containsBubble(ibub)) {
        bubbles->getBubble(ibub)->k = k;
        bubbles->getBubble(ibub)->uploadAll(bf, lmax);
        check_errors(__FILE__, __LINE__);
    }
}

extern "C" void bubble_upload_cuda(Bubbles *bubbles,  int ibub, int lmax, double *bf) {
    if (bubbles->containsBubble(ibub)) {
        bubbles->getBubble(ibub)->upload(bf, lmax);
        check_errors(__FILE__, __LINE__);
    }
}

extern "C" void bubble_add_cuda(Bubbles *bubbles, Bubbles *bubbles1, int ibub) {
    bubbles->getBubble(ibub)->add(bubbles1->getBubble(ibub));
    check_errors(__FILE__, __LINE__);
}


extern "C" void bubbles_destroy_cuda(Bubbles* bubbles){
    if (bubbles) {
        bubbles->destroy();
        delete bubbles;
        check_errors(__FILE__, __LINE__);
    }
}

extern "C" double bubbles_integrate_cuda(Bubbles *bubbles) {
    return bubbles->integrate();
}

extern "C" void bubbles_set_processor_configuration_cuda(Bubbles *bubbles, int processor_order_number, int number_of_processors) {
    bubbles->setProcessorConfiguration(processor_order_number, number_of_processors);
}

extern "C" void bubbles_inject_cuda(Bubbles *bubbles, Grid3D *grid, int lmin, CudaCube *cube) {
    bubbles->inject(grid, cube, lmin);
}

extern "C" void bubbles_inject_to_cuda(Bubbles *bubbles, Grid3D *grid, int lmin, CudaCube *cudaCube, double *cube, int offset, int cube_host_shape[3]) {
    cudaCube->initHost(&cube[offset], cube_host_shape, true);
    cudaCube->upload();
    bubbles->inject(grid, cudaCube, lmin);
}

extern "C" double *bubbles_init_page_locked_f_cuda(int lmax, int shape){
    //allocated += 1;
    double * result_f;
    check_errors(__FILE__, __LINE__);
    cudaHostAlloc((void **)&result_f, 
                  sizeof(double) * (lmax+1) * (lmax+1) * shape,
                  cudaHostAllocPortable);
    check_errors(__FILE__, __LINE__);
    //printf("Allocated 1, Now allocated %d, address: %ld\n", allocated, result_f);
    return result_f;
}

extern "C" void bubbles_destroy_page_locked_f_cuda(double * f){
    //allocated -= 1;
    //printf("Deallocated 1, Now allocated %d, address: %ld\n", allocated, f);
    cudaFreeHost(f);
    check_errors(__FILE__, __LINE__);
}


