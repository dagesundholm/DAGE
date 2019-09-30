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
#include "integrator.h"
#include "streamcontainer.h"
#include "../gbfmm/gbfmm_coulomb3d.h"
#include "../gbfmm/gbfmm_helmholtz3d.h"
#include "cube.h"
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include "memory_leak_operators.h"

#define X_ 0
#define Y_ 1
#define Z_ 2

#define FULL_MASK 0xffffffff

// double precision atomic add function
__device__ double atomicAddD(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;

    unsigned long long int old = *address_as_ull, assumed;

    do{ 
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val +__longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}


__global__ void add_to_result(double *result,
                              double *addition,
                              int number_of_dimensions,
                              int maximum_block_count,
                              int max_number_of_stored_results
                             ) {
    // get the id of the point (We are using only the first )
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (index == 0) {
        for (int dimension = 0; dimension < number_of_dimensions; dimension ++) {
            atomicAddD(&result[dimension * max_number_of_stored_results],
                      addition[dimension * maximum_block_count]);
        }
    }
}


__global__ void multiply_vectors(double * __restrict__ vector_a,
                                 const double * __restrict__ vector_b,
                                 const int size,
                                 const int offset) {
    // get the id of the point (We are using only the first )
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (index < size) {
        vector_a[index + offset] *= vector_b[index+offset];
    }
}

__device__ inline int getBlockId() {
    int blockId = blockIdx.x
                  + blockIdx.y * gridDim.x
                  + gridDim.x * gridDim.y * blockIdx.z;
    return blockId;
}

__device__ inline int getGridSize() {
    int gridSize = gridDim.x * gridDim.y * gridDim.z;
    return gridSize;
}


__device__ inline int getGlobalThreadId() {
    int blockId = blockIdx.x
                  + blockIdx.y * gridDim.x
                  + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                   + (threadIdx.z * (blockDim.x * blockDim.y))
                   + (threadIdx.y * blockDim.x)
                   + threadIdx.x;
    return threadId;
}

__device__  inline int getGlobalThreadId(int blockId) {
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                   + (threadIdx.z * (blockDim.x * blockDim.y))
                   + (threadIdx.y * blockDim.x)
                   + threadIdx.x;
    return threadId;
}

__device__ inline int getLocalThreadId() {
    int threadId = (threadIdx.z * (blockDim.x * blockDim.y))
                   + (threadIdx.y * blockDim.x)
                   + threadIdx.x;
    return threadId;
}

__device__ inline int getMemoryId(int x, int y, int z, int memory_shape_x, int memory_shape_y) {
    int memoryId =   (z * memory_shape_x * memory_shape_y)
                   + (y * memory_shape_x)
                   + x;
    return memoryId;
}

__device__ inline void getXYZCoordinates(int *x, int *y, int *z, int shape_x, int shape_y, int blockId) {
    int blocks_per_vector = shape_x / blockDim.x + ((shape_x % blockDim.x) > 0);
    int blocks_per_slice  = (shape_y / blockDim.y + ((shape_y % blockDim.y) > 0)) * blocks_per_vector;
    *x = (blockId % blocks_per_vector) * blockDim.x  + threadIdx.x;
    *y = (blockId % blocks_per_slice) / blocks_per_vector  * blockDim.y  + threadIdx.y;
    *z = (blockId / blocks_per_slice)  * blockDim.z  + threadIdx.z;
}

__device__ inline void getXYZ3D__(int *x, int *y, int *z) {
    *x = blockIdx.x * blockDim.x + threadIdx.x;
    *y = blockIdx.y * blockDim.y + threadIdx.y;
    *z = blockIdx.z * blockDim.z + threadIdx.z;
}

template <unsigned int blockSize> 
__device__  void contractBlock(const int threadId, double &value, double *shared_data) {
#if (__CUDA_ARCH__ >= 350) && (__CUDA_ARCH__ < 700)
    
    // sum the results warp-wise
    value += __shfl_down(value, 1);
    value += __shfl_down(value, 2);
    value += __shfl_down(value, 4);
    value += __shfl_down(value, 8);
    value += __shfl_down(value, 16);
    
    if (threadId < 32) {
        shared_data[threadId] = 0.0;
    }
    __syncthreads();
    
    if (threadId % 32 == 0) {
        shared_data[threadId / 32] = value;
    }
    __syncthreads();
    
    // sum the warp results together at the first warp
    if (threadId < 32 ) {
        value = shared_data[threadId];
        if (blockSize >= 32)   value += __shfl_down(value, 1);
        if (blockSize >= 64)   value += __shfl_down(value, 2);
        if (blockSize >= 128)  value += __shfl_down(value, 4);
        if (blockSize >= 256)  value += __shfl_down(value, 8);
        if (blockSize >= 512)  value += __shfl_down(value, 16);
    }
    
#elif __CUDA_ARCH__ >= 700
    
    // sum the results warp-wise
    value += __shfl_down_sync(FULL_MASK, value, 1);
    value += __shfl_down_sync(FULL_MASK, value, 2);
    value += __shfl_down_sync(FULL_MASK, value, 4);
    value += __shfl_down_sync(FULL_MASK, value, 8);
    value += __shfl_down_sync(FULL_MASK, value, 16);
    
    if (threadId < 32) {
        shared_data[threadId] = 0.0;
    }
    __syncthreads();
    
    if (threadId % 32 == 0) {
        shared_data[threadId / 32] = value;
    }
    __syncthreads();
    
    // sum the warp results together at the first warp
    if (threadId < 32 ) {
        value = shared_data[threadId];
        if (blockSize >= 32)   value += __shfl_down_sync(FULL_MASK, value, 1);
        if (blockSize >= 64)   value += __shfl_down_sync(FULL_MASK, value, 2);
        if (blockSize >= 128)  value += __shfl_down_sync(FULL_MASK, value, 4);
        if (blockSize >= 256)  value += __shfl_down_sync(FULL_MASK, value, 8);
        if (blockSize >= 512)  value += __shfl_down_sync(FULL_MASK, value, 16);
    }
    
#else
     shared_data[threadId] = value;
     
    // reduce in shared memory
    if (blockSize >= 512) {
        if (threadId < 256) {
             shared_data[threadId] += shared_data[threadId + 256];
        }
        __syncthreads();
    }
     
    if (blockSize >= 256) {
        if (threadId < 128) {
            shared_data[threadId] += shared_data[threadId + 128];
        }
        __syncthreads();
    }
     
    if (blockSize >= 128) {
        if (threadId < 64) {
            shared_data[threadId] += shared_data[threadId + 64];
        }
        __syncthreads();
    }
    
    if (blockSize >= 64) {
        if (threadId < 32) {
            shared_data[threadId] += shared_data[threadId + 32];
        }
        __syncthreads();
    }
    
    if (blockSize >= 32) {
        if (threadId < 16) {
            shared_data[threadId] += shared_data[threadId + 16];
        }
        __syncthreads();
    }
    
    if (blockSize >= 16) {
        if (threadId < 8) {
            shared_data[threadId] += shared_data[threadId + 8];
        }
       // __syncthreads();
    }
    
    if (blockSize >= 8) {
        if (threadId < 4) {
            shared_data[threadId] += shared_data[threadId + 4];
        }
        //__syncthreads();
    }
    
    if (blockSize >= 4) {
        if (threadId < 2) {
            shared_data[threadId] += shared_data[threadId + 2];
        }
        //__syncthreads();
    }
    
    if (blockSize >= 2) {
        if (threadId == 0) {
            shared_data[threadId] += shared_data[threadId + 1];
            // write the result of the block to the output_data
            value = shared_data[0];
        }
    }
#endif
}



/*__global__ void multiply_cube_with_vectors(double * __restrict__ cube,
                                           const size_t pitch,
                                           const int memory_shape_y,
                                           const double * __restrict__ vector_x,
                                           const double * __restrict__ vector_y,
                                           const double * __restrict__ vector_z,
                                           const unsigned int shape_x,
                                           const unsigned int shape_y,
                                           const unsigned int shape_z,
                                           // the slice offset in the entire cube
                                           // used to get the right x, y and z values
                                           const int total_slice_offset
                                           ) {
    int x, y, z;
    getXYZ3D__(&x, &y, &z);
    // get the id of the point (We are using only the first )
    if (x < shape_x && y < shape_y && z < shape_z) {
        const double y_factor = vector_y[y];
        const double z_factor = vector_z[z+total_slice_offset];
        const double x_factor = vector_x[x];
        double multiplier = x_factor * y_factor * z_factor;
        
        
        cube[getMemoryId(x, y, z, pitch / sizeof(double), memory_shape_y)] *= multiplier;
    }
}*/




template <unsigned int blockSize>
__global__ void reduce3d(Grid3D *grid,
                         const double *input,
                         size_t input_pitch,
                         int input_memory_shape_y,
                         const int input_shape_x,
                         const int input_shape_y,
                         const int input_shape_z,
                         const int grid_size,
                         const int slice_offset,
                         double *output
                        ) {
    
    // get the x, y & z coordinates of the first point
    int x, y, z, blockId = blockIdx.x;
    
    // temp variables that include the size of vector and slice in items
    int vector = input_pitch / sizeof(double);
    
    // get the id of the point (We are using only the first )
    const int threadId = getLocalThreadId();
    int id;
#if (__CUDA_ARCH__ >= 350) 
    __shared__ double shared_data[blockSize / 32 + 1];
#else
    __shared__ double shared_data[blockSize];
#endif 
    
    
    double value = 0.0;
    getXYZCoordinates(&x, &y, &z, input_shape_x, input_shape_y, blockId);
    while (z < input_shape_z) {
        if (x < input_shape_x && y < input_shape_y && z < input_shape_z) {
            id = getMemoryId(x, y, z, vector, input_memory_shape_y);
            value += input[id] * grid->axis[X_]->integrals[x]
                               * grid->axis[Y_]->integrals[y]
                               * grid->axis[Z_]->integrals[z+slice_offset];
        }
        blockId += grid_size;
        getXYZCoordinates(&x, &y, &z, input_shape_x, input_shape_y, blockId);
    }
    
    // contract the block
    contractBlock<blockSize>(threadId, value, shared_data);
    
    // and finally set the value to the output array
    if (threadId == 0) {
        output[blockIdx.x] = value;
    }
  
}

template <unsigned int blockSize>
__global__ void reduce(double *input_data,
                       double *output_data,
                       const unsigned int size,
                       const int number_of_dimensions,
                       const int maximum_block_count) {
    
    
    // get the id of the point (We are using only the first )
    unsigned const int threadId = threadIdx.x;
    unsigned int original_id=threadId + blockIdx.x * (blockSize*2);
    unsigned const int gridSize = blockSize * 2 * gridDim.x;
#if (__CUDA_ARCH__ >= 350) 
    __shared__ double shared_data[blockSize / 32 + 1];
#else
    __shared__ double shared_data[blockSize];
#endif

    for (int dimension = 0; dimension < number_of_dimensions; dimension++) {
        double value = 0.0;
        unsigned int id = original_id;
        while (id < size) {
            
            value += input_data[id + dimension * maximum_block_count];
            if ((id + blockSize) < size) {
                value +=  input_data[id + dimension * maximum_block_count + blockSize];
            }
            id += gridSize;
        }
        contractBlock<blockSize>(threadId, value, shared_data);
        
        // and finally set the value to the output array
        if (threadId == 0) {
            output_data[blockIdx.x + dimension * maximum_block_count] = value;
        }
    }

}

///////////////////////////////////////////////////////////
//  GBFMMHelmholtz3DMultipoleEvaluator Implementation    //
///////////////////////////////////////////////////////////

GBFMMHelmholtz3DMultipoleEvaluator::GBFMMHelmholtz3DMultipoleEvaluator(StreamContainer *streamContainer,
                                                                   Grid3D *grid,
                                                                   int lmax, 
                                                                   double center[3]) {
    this->lmax = lmax;
    this->center[X_] = center[X_];
    this->center[Y_] = center[Y_];
    this->center[Z_] = center[Z_];
    initCommons(streamContainer, grid, 1, (lmax+1)*(lmax+1), false);
}

void GBFMMHelmholtz3DMultipoleEvaluator::setKappa(double kappa) {
    this->kappa = kappa;
}

__device__ inline double FirstModSphBessel_evaluate_small(const double z, const int n, const double scaling_factor) {
#define MAX_TERM_COUNT 300
    double reslt;
    double prefactor = z / scaling_factor;
    prefactor = pow(prefactor, n);
    double divider = 1.0;
    int i;
    
    for (i = 1; i <= 2*n+1; i=i+2) {
        divider *=  (double)i;
    }
    
    double a = 0.5 * pow(z, 2.0);
    double b = 1.0;
    double addition = 1.0;
    double factorial_i = 1.0;
    
    i = 1;
    prefactor = prefactor / divider;
    reslt = 1.0;
    divider = 1.0;
    while (addition > reslt*pow(10.0, -15.0) && i < MAX_TERM_COUNT) {
        b = b * a;
        factorial_i *= (double) i;
        divider *= (double) (2*n + 2*i + 1);
        addition = b / (factorial_i * divider);
        reslt += addition;
        i += 1;
    }
    reslt *= prefactor;
    
    return reslt;
}

__device__ inline void FirstModSphBessel_evaluate_recursion(const double z, 
                                                       const double scaling_factor,
                                                       const int lmax, 
                                                       const int min_stored_l,
                                                       const int max_stored_l,
                                                       const double last,
                                                       const double second_last,
                                                       double *result
                                                      ) {
    // evaluate the results
    double previous, second_previous, current;
    
    
    
    if (max_stored_l == lmax) {
        result[max_stored_l-min_stored_l] = last;
        result[max_stored_l-min_stored_l-1] = second_last;
    }
    else if (max_stored_l == lmax-1) {
        result[max_stored_l-min_stored_l] = second_last;
    }
    
    // get the two largest l values that are evaluated and store them to second_previous and previous
    second_previous = last;
    previous = second_last;
    
    // do downward recursion for the rest of the l values
    for (int l = lmax-2; l >= min_stored_l; l--) {
        current = (2*l+3) * previous / z * scaling_factor 
                   + second_previous * scaling_factor * scaling_factor;
        second_previous = previous;
        previous = current;
        
        if (l <= max_stored_l) result[l - min_stored_l] = current;
    }
    
    // check if z is zero, then the first will be 1 and the others are 0,
    // we are assuming that the result array is initialized to 0.
    if (z < 1e-12) {
        for (int i = 0; i <= max_stored_l-min_stored_l; i++) result[i] = 0.0;
        if (min_stored_l == 0) result[0] = 1.0;
    }
}


template <unsigned int blockSize> 
__device__ void GBFMMHelmholtz3DMultipoleEvaluator_evaluateMultipoleMomentsBlock(
                                              const double x,
                                              const double y,
                                              const double z,
                                              const double r,
                                              const double kappa,
                                              const int lmax,
                                              double* result,
                                              const double cube_value,
                                              const int result_x_shape,
                                              const int threadId, 
                                              double* shared_data) {
#define STORED_BESSELS 5
    const double kappa_r = kappa*r;
    double bessel_lmax =    FirstModSphBessel_evaluate_small(kappa_r, lmax, 1.0);
    double bessel_lmax_m1 = FirstModSphBessel_evaluate_small(kappa_r, lmax-1, 1.0);    
    int lm_address =0, address2 = 0;
    int l, m, l2; 
    double top = 0.0, bottom = 0.0, new_bottom = 0.0, prev1 = 0.0, prev2 = 0.0, current = 0.0, one_per_r = 1.0 / r;
    double bessel_values[STORED_BESSELS];
    if (r < 1e-12) one_per_r = 0.0;
    
    // set value for l=0, m=0
    FirstModSphBessel_evaluate_recursion(kappa_r, 1.0, lmax, 0, 1, bessel_lmax, bessel_lmax_m1, bessel_values);
    double value = 1.0 * cube_value * bessel_values[0];
    contractBlock<blockSize>(threadId, value, shared_data);
    if (threadId == 0) result[lm_address] += value;
    
    // set value for l=1, m=-1
    lm_address += result_x_shape;
    value = y * one_per_r * cube_value * bessel_values[1];
    contractBlock<blockSize>(threadId, value, shared_data);
    if (threadId == 0) result[lm_address] += value;
    
    
    // set value for l=1, m=0
    lm_address += result_x_shape;
    value = z * one_per_r * cube_value * bessel_values[1];
    contractBlock<blockSize>(threadId, value, shared_data);
    if (threadId == 0) result[lm_address] += value;
    
    // set value for l=1, m=1
    lm_address += result_x_shape;
    value = x * one_per_r * cube_value * bessel_values[1];
    contractBlock<blockSize>(threadId, value, shared_data);
    if (threadId == 0) result[lm_address] += value;
    
    // set all values where m=-1
    m = -1;
    prev1 = y * one_per_r;
    // the starting address has 1 item before from the l=0, 3 from l=1, and 1 from l=2
    address2 = 5 * result_x_shape;
    for (int min_l = 2; min_l <= lmax; min_l += STORED_BESSELS) {
        int max_l = min(lmax, min_l+STORED_BESSELS-1);
        FirstModSphBessel_evaluate_recursion(kappa_r, 1.0, lmax,
                                             min_l, max_l,
                                             bessel_lmax, bessel_lmax_m1, bessel_values);
        for (l = min_l; l <= max_l; l++) {
            current =   ( 2.0*(double)l-1.0) / sqrt( 1.0*(double)((l+m)*(l-m)) ) * z*prev1 * one_per_r;
            if (l > 2) {
                current -=  sqrt( (double)((l+m-1)*(l-m-1)) /  (double)((l+m)*(l-m)) ) * prev2;
            }
            prev2 = prev1;
            prev1 = current;
            value = current * cube_value * bessel_values[l-min_l];
            contractBlock<blockSize>(threadId, value, shared_data);
            if (threadId == 0) result[address2] += value;
            
            // add the address2 to get to the next item with m=0 
            address2 += (2*l+2) * result_x_shape;
        }
    }
    
    
    
    
    // set all values where m=0
    prev1 = z * one_per_r;
    prev2 = 1.0;
    m = 0;
    // the starting address has 1 item before from the l=0, 3 from l=1, and 2 from l=2
    address2 = 6 * result_x_shape;
    for (int min_l = 2; min_l <= lmax; min_l += STORED_BESSELS) {
        int max_l = min(lmax, min_l+STORED_BESSELS-1);
        FirstModSphBessel_evaluate_recursion(kappa_r, 1.0, lmax,
                                             min_l, max_l,
                                             bessel_lmax, bessel_lmax_m1, bessel_values);
        for (l = min_l; l <= max_l; l++) {
            current =   ( 2.0*(double)l-1.0) / sqrt( 1.0*(double)((l+m)*(l-m)) ) * z * prev1 * one_per_r;
            current -=  sqrt( (double)((l+m-1)*(l-m-1)) /  (double)((l+m)*(l-m)) ) * prev2;
            prev2 = prev1;
            prev1 = current;
            value = current * cube_value * bessel_values[l-min_l];
            contractBlock<blockSize>(threadId, value, shared_data);
            if (threadId == 0) result[address2] += value;
            
            // add the address2 to get to the next item with m=0 
            address2 += (2*l+2) * result_x_shape;
        }
    }
    
    
    // set all values where m=1
    prev1 = x * one_per_r;
    m = 1;
    // the starting address has 1 item before from the l=0, 3 from l=1, and 3 from l=2
    address2 = 7 * result_x_shape;
    for (int min_l = 2; min_l <= lmax; min_l += STORED_BESSELS) {
        int max_l = min(lmax, min_l+STORED_BESSELS-1);
        FirstModSphBessel_evaluate_recursion(kappa_r, 1.0, lmax,
                                             min_l, max_l,
                                             bessel_lmax, bessel_lmax_m1, bessel_values);
        for (l = min_l; l <= max_l; l++) {
            current =   ( 2.0*(double)l-1.0) / sqrt( 1.0*(double)((l+m)*(l-m)) ) * z * prev1 * one_per_r;
            if (l > 2) {
                current -=  sqrt( (double)((l+m-1)*(l-m-1)) /  (double)((l+m)*(l-m)) ) * prev2;
            }
            prev2 = prev1;
            prev1 = current;
            value = current *  cube_value * bessel_values[l-min_l];
            contractBlock<blockSize>(threadId, value, shared_data);
            if (threadId == 0) result[address2] += value;
            
            // add the address2 to get to the next item with m=0 
            address2 += (2*l+2) * result_x_shape;
        }
    }
    
    // go through the rest of the stuff
    bottom = y * one_per_r; // bottom refers to solid harmonics value with l=l-1 and m=-(l-1)
    top = x * one_per_r;    // top    refers to solid harmonics value with l=l-1 and m=l-1
    lm_address += result_x_shape;
    for (l=2; l <= lmax; l++) {
        
        new_bottom = sqrt((2.0*(double)l - 1.0) / (2.0*(double)l)) * 
                      one_per_r * ( y*top + x*bottom);
        FirstModSphBessel_evaluate_recursion(kappa_r, 1.0, lmax, l, l,
                                             bessel_lmax, bessel_lmax_m1, bessel_values);
        value = new_bottom * cube_value * bessel_values[0];
        contractBlock<blockSize>(threadId, value, shared_data);
        if (threadId == 0) result[lm_address] += value;
        // set the first address for the loop below
        address2 = lm_address + (2*l+2) * result_x_shape;
        
        // get value for l=l, m=l. The address is 2*l items away from l=l, m=-l
        lm_address += 2*l * result_x_shape;
        top = sqrt((2.0*(double)l - 1.0) / (2.0*(double)l)) * 
                      one_per_r * ( x*top-y*bottom );
                      
        value = top * cube_value * bessel_values[0];
        contractBlock<blockSize>(threadId, value, shared_data);
        if (threadId == 0) result[lm_address] += value;
        
        
        // store the new bottom: l=l, m=-l (we need the old bottom in calculation of top)
        bottom = new_bottom;
        
        // set all values where m=-l
        m = -l;
        prev1 = bottom;
        for (int min_l = l+1; min_l <= lmax; min_l += STORED_BESSELS) {
            int max_l = min(lmax, min_l+STORED_BESSELS-1);
            FirstModSphBessel_evaluate_recursion(kappa_r, 1.0, lmax,
                                                min_l, max_l,
                                                bessel_lmax, bessel_lmax_m1, bessel_values);
            for (l2 = min_l; l2 <= max_l; l2++) {
                current =   ( 2.0*(double)l2-1.0) / sqrt( 1.0*(double)((l2+m)*(l2-m)) ) * z * prev1 * one_per_r;
                if (l2 > l+1) {
                    current -=  sqrt( (double)((l2+m-1)*(l2-m-1)) /  (double)((l2+m)*(l2-m)) ) * prev2;
                }
                prev2 = prev1;
                prev1 = current;
                value = current * cube_value * bessel_values[l2-min_l];
                contractBlock<blockSize>(threadId, value, shared_data);
                if (threadId == 0) result[address2] += value;
                
                // add the address2 to get to the next item with m=l 
                address2 += (2*l2+2) * result_x_shape;
            }
        }
        
        // set all values where m=l
        m = l;
        prev1 = top;
        address2 = lm_address + (2*l+2) * result_x_shape;
        for (int min_l = l+1; min_l <= lmax; min_l += STORED_BESSELS) {
            int max_l = min(lmax, min_l+STORED_BESSELS-1);
            FirstModSphBessel_evaluate_recursion(kappa_r, 1.0, lmax,
                                                min_l, max_l,
                                                bessel_lmax, bessel_lmax_m1, bessel_values);
            for (l2 = min_l; l2 <= max_l; l2++) {
                current =   ( 2.0*(double)l2-1.0) / sqrt( 1.0*(double)((l2+m)*(l2-m)) ) * z * prev1 * one_per_r;
                if (l2 > l+1) {
                    current -=  sqrt( (double)((l2+m-1)*(l2-m-1)) /  (double)((l2+m)*(l2-m)) ) * prev2;
                }
                prev2 = prev1;
                prev1 = current;
                value = current * cube_value * bessel_values[l2-min_l];
                contractBlock<blockSize>(threadId, value, shared_data);
                if (threadId == 0) result[address2] += value;
                
                // add the address2 to get to the next item with m=l 
                address2 += (2*l2+2) * result_x_shape;
            }
        }
        
        // get next address
        lm_address += result_x_shape;
    }
}



template <unsigned int blockSize>
__global__ void GBFMMHelmholtz3DMultipoleEvaluator_evaluate(
                         Grid3D *grid,
                         const double *cube,
                         size_t input_pitch,
                         int input_memory_shape_y,
                         const int input_shape_x,
                         const int input_shape_y,
                         const int input_shape_z,
                         const int grid_size,
                         const int slice_offset,
                         double *device_multipole_moments,
                         const int lmax,
                         const int maximum_block_count,
                         const double center_x,
                         const double center_y,
                         const double center_z, 
                         const double kappa) {
    
    extern __shared__ double shared_multipole_moments[];
    double *shared_data = &shared_multipole_moments[(lmax+1)*(lmax+1)];
    
    // get the x, y & z coordinates of the first point
    int x, y, z, blockId = blockIdx.x;
    
    // temp variables that include the size of vector and slice in items
    int vector = input_pitch / sizeof(double);
    
    // get the id of the point (We are using only the first )
    const int threadId = getLocalThreadId();
    int id;
    
    // init the shared memory to 0.0
    int n = threadId;
    while (n < (lmax+1)*(lmax+1)) {
        shared_multipole_moments[n] = 0.0;
        n += blockDim.x * blockDim.y * blockDim.z;
    }
    
    getXYZCoordinates(&x, &y, &z, input_shape_x, input_shape_y, blockId);
    while (z < input_shape_z) {
        if (x < input_shape_x && y < input_shape_y && z < input_shape_z) {
            id = getMemoryId(x, y, z, vector, input_memory_shape_y);
            const double cube_value = cube[id]  * grid->axis[X_]->integrals[x]
                                                * grid->axis[Y_]->integrals[y]
                                                * grid->axis[Z_]->integrals[z+slice_offset];
            const double x_coord = grid->axis[X_]->gridpoints[x] - center_x;
            const double y_coord = grid->axis[Y_]->gridpoints[y] - center_y;
            const double z_coord = grid->axis[Z_]->gridpoints[z+slice_offset] - center_z;
            const double r = sqrt(x_coord * x_coord + y_coord * y_coord + z_coord * z_coord);
            // call the evaluation for the block
            GBFMMHelmholtz3DMultipoleEvaluator_evaluateMultipoleMomentsBlock<blockSize>
                                                    (x_coord,
                                                     y_coord,
                                                     z_coord,
                                                     r,
                                                     kappa,
                                                     lmax,
                                                     shared_multipole_moments,
                                                     cube_value,
                                                     1,
                                                     threadId,
                                                     shared_data
                                                    );
        }
        // increase the block
        blockId += grid_size;
        getXYZCoordinates(&x, &y, &z, input_shape_x, input_shape_y, blockId);
    }
    __syncthreads();
    // and finally copy the result value to the output array
    n = threadId;
    while (n < (lmax+1)*(lmax+1)) {
        device_multipole_moments[blockIdx.x + maximum_block_count * n] =
            shared_multipole_moments[n];
        n += blockDim.x * blockDim.y * blockDim.z;
    }
  
}

__host__ inline void GBFMMHelmholtz3DMultipoleEvaluator::reduce3D(
                                      double *input_data, 
                                      size_t device_pitch,
                                      int device_y_shape, 
                                      int shape[3],
                                      int slice_count,
                                      double *output_data,
                                      unsigned int grid_size,
                                      unsigned int output_size,
                                      unsigned int threads_per_block,
                                      cudaStream_t *stream, 
                                      int slice_offset,
                                      int device_order_number) {
    size_t sharedMemorySize = sizeof(double) * ((this->lmax+1) * (this->lmax+1) + threads_per_block); 
    dim3 grid, block;
    this->getLaunchConfiguration(&grid, &block, threads_per_block, output_size);
    check_errors(__FILE__, __LINE__);
    switch (threads_per_block)
    {
        case 512:
            GBFMMHelmholtz3DMultipoleEvaluator_evaluate<512><<< grid, block, sharedMemorySize, *stream>>>(
                this->getGrid()->device_copies[device_order_number], 
                input_data, device_pitch, device_y_shape, 
                shape[0], shape[1], slice_count, 
                output_size, slice_offset, output_data,
                this->lmax, this->maximum_block_count, 
                this->center[X_], this->center[Y_], this->center[Z_],
                this->kappa);
            break;
        case 256:
            GBFMMHelmholtz3DMultipoleEvaluator_evaluate<256><<< grid, block, sharedMemorySize, *stream>>>(
                this->getGrid()->device_copies[device_order_number], 
                input_data, device_pitch, device_y_shape, 
                shape[0], shape[1], slice_count, 
                output_size, slice_offset, output_data,
                this->lmax, this->maximum_block_count, 
                this->center[X_], this->center[Y_], this->center[Z_],
                this->kappa);
            break;
        case 128:
            GBFMMHelmholtz3DMultipoleEvaluator_evaluate<128><<< grid, block, sharedMemorySize, *stream>>>(
                this->getGrid()->device_copies[device_order_number], 
                input_data, device_pitch, device_y_shape, 
                shape[0], shape[1], slice_count, 
                output_size, slice_offset, output_data,
                this->lmax, this->maximum_block_count, 
                this->center[X_], this->center[Y_], this->center[Z_],
                this->kappa);
            break;
        case 64:
            GBFMMHelmholtz3DMultipoleEvaluator_evaluate<64> <<< grid, block, sharedMemorySize, *stream>>>(
                this->getGrid()->device_copies[device_order_number], 
                input_data, device_pitch, device_y_shape, 
                shape[0], shape[1], slice_count, 
                output_size, slice_offset, output_data,
                this->lmax, this->maximum_block_count, 
                this->center[X_], this->center[Y_], this->center[Z_],
                this->kappa);
            break;
        case 32:
            GBFMMHelmholtz3DMultipoleEvaluator_evaluate<32> <<< grid, block, sharedMemorySize, *stream>>>(
                this->getGrid()->device_copies[device_order_number], 
                input_data, device_pitch, device_y_shape, 
                shape[0], shape[1], slice_count, 
                output_size, slice_offset, output_data,
                this->lmax, this->maximum_block_count, 
                this->center[X_], this->center[Y_], this->center[Z_],
                this->kappa);
            break;
        case 16:
            GBFMMHelmholtz3DMultipoleEvaluator_evaluate<16> <<< grid, block, sharedMemorySize, *stream>>>(
                this->getGrid()->device_copies[device_order_number], 
                input_data, device_pitch, device_y_shape, 
                shape[0], shape[1], slice_count, 
                output_size, slice_offset, output_data,
                this->lmax, this->maximum_block_count, 
                this->center[X_], this->center[Y_], this->center[Z_],
                this->kappa);
            break;
        case  8:
            GBFMMHelmholtz3DMultipoleEvaluator_evaluate<8>  <<< grid, block, sharedMemorySize, *stream>>>(
                this->getGrid()->device_copies[device_order_number], 
                input_data, device_pitch, device_y_shape, 
                shape[0], shape[1], slice_count, 
                output_size, slice_offset, output_data,
                this->lmax, this->maximum_block_count, 
                this->center[X_], this->center[Y_], this->center[Z_],
                this->kappa);
            break;
        case  4:
            GBFMMHelmholtz3DMultipoleEvaluator_evaluate<4>  <<< grid, block, sharedMemorySize, *stream>>>(
                this->getGrid()->device_copies[device_order_number],  
                input_data, device_pitch, device_y_shape, 
                shape[0], shape[1], slice_count, 
                output_size, slice_offset, output_data,
                this->lmax, this->maximum_block_count, 
                this->center[X_], this->center[Y_], this->center[Z_],
                this->kappa);
            break;
        case  2:
            GBFMMHelmholtz3DMultipoleEvaluator_evaluate<2>  <<< grid, block, sharedMemorySize, *stream>>>(
                this->getGrid()->device_copies[device_order_number],  
                input_data, device_pitch, device_y_shape, 
                shape[0], shape[1], slice_count, 
                output_size, slice_offset, output_data,
                this->lmax, this->maximum_block_count, 
                this->center[X_], this->center[Y_], this->center[Z_],
                this->kappa);
            break;
        case  1:
            GBFMMHelmholtz3DMultipoleEvaluator_evaluate<1>  <<< grid, block, sharedMemorySize, *stream>>>(
                this->getGrid()->device_copies[device_order_number], 
                input_data, device_pitch, device_y_shape, 
                shape[0], shape[1], slice_count, 
                output_size, slice_offset, output_data,
                this->lmax, this->maximum_block_count, 
                this->center[X_], this->center[Y_], this->center[Z_],
                this->kappa);
            break;
    }
    check_errors(__FILE__, __LINE__);
}



///////////////////////////////////////////////////////////
//  GBFMMCoulomb3DMultipoleEvaluator Implementation      //
///////////////////////////////////////////////////////////

GBFMMCoulomb3DMultipoleEvaluator::GBFMMCoulomb3DMultipoleEvaluator(StreamContainer *streamContainer,
                                                                   Grid3D *grid,
                                                                   int lmax, 
                                                                   double center[3]) {
    this->lmax = lmax;
    this->center[X_] = center[X_];
    this->center[Y_] = center[Y_];
    this->center[Z_] = center[Z_];
    initCommons(streamContainer, grid, 1, (lmax+1)*(lmax+1), false);
}


/* Kernels and crucial device functions */
template <unsigned int blockSize> 
__device__ void GBFMMCoulomb3DMultipoleEvaluator_evaluateMultipoleMomentsBlock(
                                              const double x,
                                              const double y,
                                              const double z,
                                              const int lmax,
                                              double* result,
                                              const double cube_value,
                                              const int result_x_shape,
                                              const int threadId, 
                                              double* shared_data) {
    
    int lm_address =0, address2 = 0;
    int l, m, l2; 
    double top = 0.0, bottom = 0.0, new_bottom = 0.0, prev1 = 0.0, prev2 = 0.0, current = 0.0;
    double r2 = x*x+y*y+z*z;
    // set value for l=0, m=0
    double value = 1.0 * cube_value;
    contractBlock<blockSize>(threadId, value, shared_data);
    if (threadId == 0) result[lm_address] += value;
    
    
    // set value for l=1, m=-1
    lm_address += result_x_shape;
    value = y * cube_value;
    contractBlock<blockSize>(threadId, value, shared_data);
    if (threadId == 0) result[lm_address] += value;
    // set all values where m=-1
    m = -1;
    prev1 = y;
    // the starting address has 1 item before from the l=0, 3 from l=1, and 1 from l=2
    address2 = 5 * result_x_shape;
    for (l = 2; l <= lmax; l++) {
        current =   ( 2.0*(double)l-1.0) / sqrt( 1.0*(double)((l+m)*(l-m)) ) * z*prev1;
        if (l > 2) {
            current -=  sqrt( (double)((l+m-1)*(l-m-1)) /  (double)((l+m)*(l-m)) ) * r2 * prev2;
        }
        prev2 = prev1;
        prev1 = current;
        value = current * cube_value;
        contractBlock<blockSize>(threadId, value, shared_data);
        if (threadId == 0) result[address2] += value;
        
        // add the address2 to get to the next item with m=0 
        address2 += (2*l+2) * result_x_shape;
    }
    
    
    
    // set value for l=1, m=0
    lm_address += result_x_shape;
    value = z * cube_value;
    contractBlock<blockSize>(threadId, value, shared_data);
    if (threadId == 0) result[lm_address] += value;
    
    // set all values where m=0
    prev1 = z;
    prev2 = 1.0;
    m = 0;
    // the starting address has 1 item before from the l=0, 3 from l=1, and 2 from l=2
    address2 = 6 * result_x_shape;
    for (l = 2; l <= lmax; l++) {
        current =   ( 2.0*(double)l-1.0) / sqrt( 1.0*(double)((l+m)*(l-m)) ) * z * prev1;
        current -=  sqrt( (double)((l+m-1)*(l-m-1)) /  (double)((l+m)*(l-m)) ) * r2 * prev2;
        prev2 = prev1;
        prev1 = current;
        value = current * cube_value;
        contractBlock<blockSize>(threadId, value, shared_data);
        if (threadId == 0) result[address2] += value;
        
        // add the address2 to get to the next item with m=0 
        address2 += (2*l+2) * result_x_shape;
    }
    
    // set value for l=1, m=1
    lm_address += result_x_shape;
    value = x * cube_value;
    contractBlock<blockSize>(threadId, value, shared_data);
    if (threadId == 0) result[lm_address] += value;
    // set all values where m=1
    prev1 = x;
    m = 1;
    // the starting address has 1 item before from the l=0, 3 from l=1, and 3 from l=2
    address2 = 7 * result_x_shape;
    for (l = 2; l <= lmax; l++) {
        current =   ( 2.0*(double)l-1.0) / sqrt( 1.0*(double)((l+m)*(l-m)) ) * z*prev1;
        if (l > 2) {
            current -=  sqrt( (double)((l+m-1)*(l-m-1)) /  (double)((l+m)*(l-m)) ) * r2 * prev2;
        }
        prev2 = prev1;
        prev1 = current;
        value = current * cube_value;
        contractBlock<blockSize>(threadId, value, shared_data);
        if (threadId == 0) result[address2] += value;
        
        // add the address2 to get to the next item with m=0 
        address2 += (2*l+2) * result_x_shape;
    }
    
    // go through the rest of the stuff
    bottom = y; // bottom refers to solid harmonics value with l=l-1 and m=-(l-1)
    top = x;    // top    refers to solid harmonics value with l=l-1 and m=l-1
    lm_address += result_x_shape;
    for (l=2; l <= lmax; l++) {
        
        new_bottom = sqrt((2.0*(double)l - 1.0) / (2.0*(double)l)) * 
                       ( y*top + x*bottom);
        value = new_bottom * cube_value;
        contractBlock<blockSize>(threadId, value, shared_data);
        if (threadId == 0) result[lm_address] += value;
        
        // set all values where m=-l
        m = -l;
        prev1 = new_bottom;
        address2 = lm_address + (2*l+2) * result_x_shape;
        for (l2 = l+1; l2 <= lmax; l2++) {
            current =   ( 2.0*(double)l2-1.0) / sqrt( 1.0*(double)((l2+m)*(l2-m)) ) * z*prev1;
            if (l2 > l+1) {
                current -=  sqrt( (double)((l2+m-1)*(l2-m-1)) /  (double)((l2+m)*(l2-m)) ) * r2 * prev2;
            }
            prev2 = prev1;
            prev1 = current;
            value = current * cube_value;
            contractBlock<blockSize>(threadId, value, shared_data);
            if (threadId == 0) result[address2] += value;
            
            // add the address2 to get to the next item with m=l 
            address2 += (2*l2+2) * result_x_shape;
        }
        
        
        // get value for l=l, m=l. The address is 2*l items away from l=l, m=-l
        lm_address += 2*l * result_x_shape;
        top = sqrt((2.0*(double)l - 1.0) / (2.0*(double)l)) * 
                      ( x*top-y*bottom );
                      
        value = top * cube_value;
        contractBlock<blockSize>(threadId, value, shared_data);
        if (threadId == 0) result[lm_address] += value;
                      
        // set all values where m=l
        m = l;
        prev1 = top;
        address2 = lm_address + (2*l+2) * result_x_shape;
        for (l2 = l+1; l2 <= lmax; l2++) {
            current =   ( 2.0*(double)l2-1.0) / sqrt( 1.0*(double)((l2+m)*(l2-m)) ) * z*prev1;
            if (l2 > l+1) {
                current -=  sqrt( (double)((l2+m-1)*(l2-m-1)) /  (double)((l2+m)*(l2-m)) ) * r2 * prev2;
            }
            prev2 = prev1;
            prev1 = current;
            value = current * cube_value;
            contractBlock<blockSize>(threadId, value, shared_data);
            if (threadId == 0) result[address2] += value;
            
            // add the address2 to get to the next item with m=l 
            address2 += (2*l2+2) * result_x_shape;
        }
        // store the new bottom: l=l, m=-l (we need the old bottom in calculation of top)
        bottom = new_bottom;
        
        // get next address
        lm_address += result_x_shape;
    }
}

template <unsigned int blockSize>
__global__ void GBFMMCoulomb3DMultipoleEvaluator_evaluate(
                         Grid3D *grid,
                         const double *cube,
                         size_t input_pitch,
                         int input_memory_shape_y,
                         const int input_shape_x,
                         const int input_shape_y,
                         const int input_shape_z,
                         const int grid_size,
                         const int slice_offset,
                         double *device_multipole_moments,
                         const int lmax,
                         const int maximum_block_count,
                         const double center_x,
                         const double center_y,
                         const double center_z) {
    
    extern __shared__ double shared_multipole_moments[];
    
    // get the x, y & z coordinates of the first point
    int x, y, z, blockId = blockIdx.x;
    
    // temp variables that include the size of vector and slice in items
    int vector = input_pitch / sizeof(double);
    
    // get the id of the point (We are using only the first )
    const int threadId = getLocalThreadId();
    int id;
    
    // init the shared memory to 0.0
    int n = threadId;
    while (n < (lmax+1)*(lmax+1)) {
        shared_multipole_moments[n] = 0.0;
        n += blockDim.x * blockDim.y * blockDim.z;
    }
    
    getXYZCoordinates(&x, &y, &z, input_shape_x, input_shape_y, blockId);
    while (z < input_shape_z) {
        if (x < input_shape_x && y < input_shape_y && z < input_shape_z) {
            id = getMemoryId(x, y, z, vector, input_memory_shape_y);
            const double cube_value = cube[id]  * grid->axis[X_]->integrals[x]
                                                * grid->axis[Y_]->integrals[y]
                                                * grid->axis[Z_]->integrals[z+slice_offset];
            
            // call the evaluation for the block
            GBFMMCoulomb3DMultipoleEvaluator_evaluateMultipoleMomentsBlock<blockSize>
                                                    (grid->axis[X_]->gridpoints[x] - center_x,
                                                     grid->axis[Y_]->gridpoints[y] - center_y,
                                                     grid->axis[Z_]->gridpoints[z+slice_offset] - center_z,
                                                     lmax,
                                                     shared_multipole_moments,
                                                     cube_value,
                                                     1,
                                                     threadId, 
                                                     &shared_multipole_moments[(lmax+1)*(lmax+1)]
                                                    );
        }
        // increase the block
        blockId += grid_size;
        getXYZCoordinates(&x, &y, &z, input_shape_x, input_shape_y, blockId);
    }
    __syncthreads();
    // and finally copy the result value to the output array
    n = threadId;
    while (n < (lmax+1)*(lmax+1)) {
        device_multipole_moments[blockIdx.x + maximum_block_count * n] =
            shared_multipole_moments[n];
        n += blockDim.x * blockDim.y * blockDim.z;
    }
  
}

__host__ inline void GBFMMCoulomb3DMultipoleEvaluator::reduce3D(
                                      double *input_data, 
                                      size_t device_pitch,
                                      int device_y_shape, 
                                      int shape[3],
                                      int slice_count,
                                      double *output_data,
                                      unsigned int grid_size,
                                      unsigned int output_size,
                                      unsigned int threads_per_block,
                                      cudaStream_t *stream, 
                                      int slice_offset,
                                      int device_order_number) {
    size_t sharedMemorySize = sizeof(double) * ((this->lmax+1) * (this->lmax+1) + threads_per_block); 
    dim3 grid, block;
    this->getLaunchConfiguration(&grid, &block, threads_per_block, output_size);
    switch (threads_per_block)
    {
        case 512:
            GBFMMCoulomb3DMultipoleEvaluator_evaluate<512><<< grid, block, sharedMemorySize, *stream>>>(
                this->getGrid()->device_copies[device_order_number], 
                input_data, device_pitch, device_y_shape, 
                shape[0], shape[1], slice_count, 
                output_size, slice_offset, output_data,
                this->lmax, this->maximum_block_count, 
                this->center[X_], this->center[Y_], this->center[Z_]);
            break;
        case 256:
            GBFMMCoulomb3DMultipoleEvaluator_evaluate<256><<< grid, block, sharedMemorySize, *stream>>>(
                this->getGrid()->device_copies[device_order_number], 
                input_data, device_pitch, device_y_shape, 
                shape[0], shape[1], slice_count, 
                output_size, slice_offset, output_data,
                this->lmax, this->maximum_block_count, 
                this->center[X_], this->center[Y_], this->center[Z_]);
            break;
        case 128:
            GBFMMCoulomb3DMultipoleEvaluator_evaluate<128><<< grid, block, sharedMemorySize, *stream>>>(
                this->getGrid()->device_copies[device_order_number], 
                input_data, device_pitch, device_y_shape, 
                shape[0], shape[1], slice_count, 
                output_size, slice_offset, output_data,
                this->lmax, this->maximum_block_count, 
                this->center[X_], this->center[Y_], this->center[Z_]);
            break;
        case 64:
            GBFMMCoulomb3DMultipoleEvaluator_evaluate<64> <<< grid, block, sharedMemorySize, *stream>>>(
                this->getGrid()->device_copies[device_order_number], 
                input_data, device_pitch, device_y_shape, 
                shape[0], shape[1], slice_count, 
                output_size, slice_offset, output_data,
                this->lmax, this->maximum_block_count, 
                this->center[X_], this->center[Y_], this->center[Z_]);
            break;
        case 32:
            GBFMMCoulomb3DMultipoleEvaluator_evaluate<32> <<< grid, block, sharedMemorySize, *stream>>>(
                this->getGrid()->device_copies[device_order_number], 
                input_data, device_pitch, device_y_shape, 
                shape[0], shape[1], slice_count, 
                output_size, slice_offset, output_data,
                this->lmax, this->maximum_block_count, 
                this->center[X_], this->center[Y_], this->center[Z_]);
            break;
        case 16:
            GBFMMCoulomb3DMultipoleEvaluator_evaluate<16> <<< grid, block, sharedMemorySize, *stream>>>(
                this->getGrid()->device_copies[device_order_number], 
                input_data, device_pitch, device_y_shape, 
                shape[0], shape[1], slice_count, 
                output_size, slice_offset, output_data,
                this->lmax, this->maximum_block_count, 
                this->center[X_], this->center[Y_], this->center[Z_]);
            break;
        case  8:
            GBFMMCoulomb3DMultipoleEvaluator_evaluate<8>  <<< grid, block, sharedMemorySize, *stream>>>(
                this->getGrid()->device_copies[device_order_number], 
                input_data, device_pitch, device_y_shape, 
                shape[0], shape[1], slice_count, 
                output_size, slice_offset, output_data,
                this->lmax, this->maximum_block_count, 
                this->center[X_], this->center[Y_], this->center[Z_]);
            break;
        case  4:
            GBFMMCoulomb3DMultipoleEvaluator_evaluate<4>  <<< grid, block, sharedMemorySize, *stream>>>(
                this->getGrid()->device_copies[device_order_number],  
                input_data, device_pitch, device_y_shape, 
                shape[0], shape[1], slice_count, 
                output_size, slice_offset, output_data,
                this->lmax, this->maximum_block_count, 
                this->center[X_], this->center[Y_], this->center[Z_]);
            break;
        case  2:
            GBFMMCoulomb3DMultipoleEvaluator_evaluate<2>  <<< grid, block, sharedMemorySize, *stream>>>(
                this->getGrid()->device_copies[device_order_number],  
                input_data, device_pitch, device_y_shape, 
                shape[0], shape[1], slice_count, 
                output_size, slice_offset, output_data,
                this->lmax, this->maximum_block_count, 
                this->center[X_], this->center[Y_], this->center[Z_]);
            break;
        case  1:
            GBFMMCoulomb3DMultipoleEvaluator_evaluate<1>  <<< grid, block, sharedMemorySize, *stream>>>(
                this->getGrid()->device_copies[device_order_number], 
                input_data, device_pitch, device_y_shape, 
                shape[0], shape[1], slice_count, 
                output_size, slice_offset, output_data,
                this->lmax, this->maximum_block_count, 
                this->center[X_], this->center[Y_], this->center[Z_]);
            break;
    }
}


///////////////////////////////////
//  Integrator Definition      //
///////////////////////////////////



__host__ inline void Integrator::reduce1D(double *input_data, 
                                          double *output_data,
                                          unsigned int inputSize,
                                          unsigned int output_size,
                                          unsigned int threads_per_block,
                                          cudaStream_t *stream,
                                          int device_order_number
                                         ) {
    unsigned int block_count = output_size;
    size_t sharedMemorySize = sizeof(double) * threads_per_block * 2; 
    
    switch (threads_per_block)
    {
        case 512:
            reduce<512><<< block_count, threads_per_block, sharedMemorySize, *stream>>>(
                                                                                        input_data, output_data, inputSize, 
                                                                                        this->number_of_dimensions, this->maximum_block_count); 
            break;
        case 256:
            reduce<256><<< block_count, threads_per_block, sharedMemorySize, *stream>>>(
                                                                                        input_data, output_data, inputSize, 
                                                                                        this->number_of_dimensions, this->maximum_block_count); 
            break;
        case 128:
            reduce<128><<< block_count, threads_per_block, sharedMemorySize, *stream>>>(
                                                                                        input_data, output_data, inputSize, 
                                                                                        this->number_of_dimensions, this->maximum_block_count); 
            break;
        case 64:
            reduce<  64><<< block_count, threads_per_block, sharedMemorySize, *stream>>>(
                                                                                        input_data, output_data, inputSize, 
                                                                                        this->number_of_dimensions, this->maximum_block_count); 
            break;
        case 32:
            reduce<  32><<< block_count, threads_per_block, sharedMemorySize, *stream >>>( 
                                                                                        input_data, output_data, inputSize, 
                                                                                        this->number_of_dimensions, this->maximum_block_count); 
            break;
        case 16:
            reduce<  16><<< block_count, threads_per_block, sharedMemorySize, *stream>>>(
                                                                                        input_data, output_data, inputSize, 
                                                                                        this->number_of_dimensions, this->maximum_block_count); 
            break;
        case  8:
            reduce<    8><<< block_count, threads_per_block, sharedMemorySize, *stream >>>(
                                                                                        input_data, output_data, inputSize, 
                                                                                        this->number_of_dimensions, this->maximum_block_count); 
            break;
        case  4:
            reduce<    4><<< block_count, threads_per_block, sharedMemorySize, *stream >>>(
                                                                                        input_data, output_data, inputSize, 
                                                                                        this->number_of_dimensions, this->maximum_block_count); 
            break;
        case  2:
            reduce<    2><<< block_count, threads_per_block, sharedMemorySize, *stream >>>(
                                                                                        input_data, output_data, inputSize, 
                                                                                        this->number_of_dimensions, this->maximum_block_count); 
            break;
        case  1:
            reduce<    1><<< block_count, threads_per_block, sharedMemorySize, *stream >>>(
                                                                                        input_data, output_data, inputSize, 
                                                                                        this->number_of_dimensions, this->maximum_block_count); 
            break;
    }
}

void Integrator::downloadResult(double * result) {
    if (this->first_result_counter == this->max_number_of_stored_results -1) {
        this->first_result_counter = 0;
    }
    else {
        this->first_result_counter ++;
    }
    check_errors(__FILE__, __LINE__);
    // sum the device results together
    if (this->max_number_of_stored_results == 1) {
        for (int dimension = 0; dimension < this->number_of_dimensions; dimension++) {
            result[dimension] = 0.0;
        }
        double *single_device_result = new double[this->number_of_dimensions];
        for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
            this->streamContainer->setDevice(device);
            cudaMemcpyAsync(single_device_result, this->device_results[device],
                        this->number_of_dimensions * sizeof(double), cudaMemcpyDeviceToHost,
                        *this->streamContainer->getStream(device, 0));
            cudaDeviceSynchronize();
            // and sum it to the final result
            for (int dimension = 0; dimension < this->number_of_dimensions; dimension++) {
                result[dimension] += single_device_result[dimension];
            }
                
            // set the cudaResult to zero
            cudaMemsetAsync(this->device_results[device],
                            0, this->number_of_dimensions * sizeof(double),
                            *this->streamContainer->getStream(device, 0));
        }
        delete[] single_device_result;
    }
    else {
        for (int dimension = 0; dimension < this->number_of_dimensions; dimension ++) {
            result[dimension] = 0.0;
            for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
                // synchronize before to assure that each stream has completed
                this->streamContainer->setDevice(device);
                cudaDeviceSynchronize();
                
                // then download the single device result
                double single_device_result = 0.0;
                cudaMemcpyAsync(&single_device_result, &this->device_results[device][this->first_result_counter + dimension * this->max_number_of_stored_results],
                                sizeof(double), cudaMemcpyDeviceToHost, *this->streamContainer->getStream(device, 0));
                cudaDeviceSynchronize();
                // and sum it to the final result
                result[dimension] += single_device_result;
                
                // set the cudaResult to zero
                cudaMemsetAsync(&this->device_results[device][this->first_result_counter + dimension * this->max_number_of_stored_results],
                                0, sizeof(double), *this->streamContainer->getStream(device, 0));
            }
        }
    }
}

///////////////////////////////////
//  Integrator3D Definition      //
///////////////////////////////////

Integrator3D::Integrator3D() {
}

Integrator3D::Integrator3D(StreamContainer *streamContainer, Grid3D *grid, int max_number_of_stored_results, bool init_cube) {
    initCommons(streamContainer, grid, max_number_of_stored_results, /*dimensions=*/1, init_cube);
}

void Integrator3D::initCommons(StreamContainer *streamContainer, Grid3D *grid, int max_number_of_stored_results, int number_of_dimensions, bool init_cube) {
    this->streamContainer = streamContainer;
    check_errors(__FILE__, __LINE__);
    
    // init cube where memory is split between all devices in slices 
    bool all_memory_at_all_devices = false;
    bool sliced = true;
    if (init_cube) {
        this->cube = new CudaCube(streamContainer, grid->getShape(), all_memory_at_all_devices, sliced);
    }
    else {
        this->cube = NULL;
    }
    this->last_result_counter = -1;
    this->first_result_counter = -1;
    this->number_of_dimensions = number_of_dimensions;
    this->grid = grid;
    this->maximum_block_count = 1024;
    this->max_number_of_stored_results = max_number_of_stored_results;
    check_errors(__FILE__, __LINE__);
    
    // allocate space for stream pointers
    cudaHostAlloc((void **)&this->device_results, 
                  sizeof(double *) * streamContainer->getNumberOfDevices(),
                  cudaHostAllocPortable);
    // allocate space for stream pointers
    cudaHostAlloc((void **)&this->device_copies, 
                  sizeof(this) * streamContainer->getNumberOfDevices(),
                  cudaHostAllocPortable);
    cudaHostAlloc((void **)&this->integration_space, 
                  sizeof(double *) * streamContainer->getNumberOfDevices() * streamContainer->getStreamsPerDevice(),
                  cudaHostAllocPortable);
    
    for (int device = 0; device < streamContainer->getNumberOfDevices(); device ++) {
        streamContainer->setDevice(device);
        cudaMalloc(&this->device_results[device], sizeof(double)*this->max_number_of_stored_results * this->number_of_dimensions);
        check_errors(__FILE__, __LINE__);
        cudaMemset(this->device_results[device], 0, sizeof(double)*this->max_number_of_stored_results * this->number_of_dimensions);
        check_errors(__FILE__, __LINE__);
        
        for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream ++) {
            cudaMalloc(&this->integration_space[device*this->streamContainer->getStreamsPerDevice() + stream],
                       sizeof(double) * this->maximum_block_count * this->number_of_dimensions);
            check_errors(__FILE__, __LINE__);
        }
        
        // make copy to the device
        this->grid = grid->device_copies[device];
        cudaMalloc(&this->device_copies[device], sizeof(*this));
        cudaMemcpy(this->device_copies[device], this, sizeof(*this), cudaMemcpyHostToDevice);
        
    }
    this->grid = grid;
}




__host__ inline void Integrator3D::getLaunchConfiguration(dim3 *grid, dim3 *block, int block_size, int grid_size) {
    block->x = 32;
    block->y = block_size/32;
    block->z = 1;
    grid->x = grid_size;
    grid->y = 1;
    grid->z = 1;
    //printf("Grid: %d, %d, %d, Block: %d, %d, %d\n", grid->x, grid->y, grid->z, block->x, block->y, block->z);
}

__host__  void Integrator3D::reduce3D(double *input_data, 
                                            size_t device_pitch,
                                            int device_y_shape, 
                                            int shape[3],
                                            int slice_count,
                                            double *output_data,
                                            unsigned int grid_size,
                                            unsigned int output_size,
                                            unsigned int threads_per_block,
                                            cudaStream_t *stream,
                                            int slice_offset,
                                            int device_order_number
                                           ) {
    size_t sharedMemorySize = sizeof(double) * threads_per_block * 2; 
    dim3 grid, block;
    this->getLaunchConfiguration(&grid, &block, threads_per_block, output_size);
    check_errors(__FILE__, __LINE__);
    
    switch (threads_per_block)
    {
        case 512:
            reduce3d<512><<< grid, block, sharedMemorySize, *stream>>>(this->getGrid()->device_copies[device_order_number], 
                                                                       input_data, device_pitch, device_y_shape, 
                                                                       shape[0], shape[1], slice_count, 
                                                                       output_size, slice_offset, output_data); 
            break;
        case 256:
            reduce3d<256><<< grid, block, sharedMemorySize, *stream>>>(this->getGrid()->device_copies[device_order_number], 
                                                                       input_data, device_pitch, device_y_shape, 
                                                                       shape[0], shape[1], slice_count, 
                                                                       output_size, slice_offset, output_data);
            break;
        case 128:
            reduce3d<128><<< grid, block, sharedMemorySize, *stream>>>(this->getGrid()->device_copies[device_order_number], 
                                                                       input_data, device_pitch, device_y_shape, 
                                                                       shape[0], shape[1], slice_count, 
                                                                       output_size, slice_offset, output_data);
            break;
        case 64:
            reduce3d<64> <<< grid, block, sharedMemorySize, *stream>>>(this->getGrid()->device_copies[device_order_number], 
                                                                       input_data, device_pitch, device_y_shape, 
                                                                       shape[0], shape[1], slice_count, 
                                                                       output_size, slice_offset, output_data); 
            break;
        case 32:
            reduce3d<32> <<< grid, block, sharedMemorySize, *stream>>>(this->getGrid()->device_copies[device_order_number], 
                                                                       input_data, device_pitch, device_y_shape, 
                                                                       shape[0], shape[1], slice_count, 
                                                                       output_size, slice_offset, output_data);
            break;
        case 16:
            reduce3d<16> <<< grid, block, sharedMemorySize, *stream>>>(this->getGrid()->device_copies[device_order_number], 
                                                                       input_data, device_pitch, device_y_shape, 
                                                                       shape[0], shape[1], slice_count, 
                                                                       output_size, slice_offset, output_data);
            break;
        case  8:
            reduce3d<8>  <<< grid, block, sharedMemorySize, *stream>>>(this->getGrid()->device_copies[device_order_number], 
                                                                       input_data, device_pitch, device_y_shape, 
                                                                       shape[0], shape[1], slice_count, 
                                                                       output_size, slice_offset, output_data);
            break;
        case  4:
            reduce3d<4>  <<< grid, block, sharedMemorySize, *stream>>>(this->getGrid()->device_copies[device_order_number], 
                                                                       input_data, device_pitch, device_y_shape, 
                                                                       shape[0], shape[1], slice_count, 
                                                                       output_size, slice_offset, output_data);
            break;
        case  2:
            reduce3d<2>  <<< grid, block, sharedMemorySize, *stream>>>(this->getGrid()->device_copies[device_order_number], 
                                                                       input_data, device_pitch, device_y_shape, 
                                                                       shape[0], shape[1], slice_count, 
                                                                       output_size, slice_offset, output_data);
            break;
        case  1:
            reduce3d<1>  <<< grid, block, sharedMemorySize, *stream>>>(this->getGrid()->device_copies[device_order_number], 
                                                                       input_data, device_pitch, device_y_shape, 
                                                                       shape[0], shape[1], slice_count, 
                                                                       output_size, slice_offset, output_data);
            break;
    }
    check_errors(__FILE__, __LINE__);
}

/*
 * Contract cube 
 */
void Integrator3D::contractCube(double *device_cube, 
                                size_t device_pitch,
                                int device_y_shape,
                                int shape[3],
                                int slice_count,
                                double *extra_data,
                                cudaStream_t *stream,
                                double *device_result,
                                int slice_offset,
                                int device_order_number) {
    
    check_errors(__FILE__, __LINE__);
    double *input_data = device_cube;
    double *output_data = extra_data;
    unsigned int current_size = shape[0] * shape[1] * slice_count;
    unsigned int optimal_threads_per_block = 256;
    unsigned int threads_per_block;
    unsigned int optimal_loops_per_thread = 8;
    unsigned int minimum_block_count = 128;
    unsigned int loops_per_thread;
    unsigned int block_count = 1;
    
    loops_per_thread = optimal_loops_per_thread;
    threads_per_block = optimal_threads_per_block;
    
    // the kernel divides the number of data by threads_per_block*loops_per_thread 
    block_count = current_size / (threads_per_block * loops_per_thread);
    while (block_count < minimum_block_count) {
        loops_per_thread /= 2;
        block_count = current_size / (threads_per_block * loops_per_thread);
        if (loops_per_thread == 1) break;
    }
    while (block_count < minimum_block_count) {
        threads_per_block /= 2;
        block_count = current_size / (threads_per_block * loops_per_thread);
        if (threads_per_block == 32) break;
    }
    if (block_count < minimum_block_count) {
        threads_per_block = optimal_threads_per_block;
        block_count = 1;
    }
    // do the 3D-reduce
    this->reduce3D(input_data,
                   device_pitch,
                   device_y_shape,
                   shape,
                   slice_count,
                   output_data,
                   current_size,
                   block_count,
                   threads_per_block,
                   stream,
                   slice_offset, 
                   device_order_number
                       );
    check_errors(__FILE__, __LINE__);
    // set the output_data to be input for the next loop
    current_size = block_count;
    input_data = output_data;
    output_data = &input_data[block_count];
    
    // then do the 1d-reduce for the rest
    while (current_size > 1) {
        loops_per_thread = optimal_loops_per_thread;
        threads_per_block = optimal_threads_per_block;
        // the kernel divides the number of data by 2*threads_per_block*loops_per_thread 
        block_count = current_size / (2 * threads_per_block * loops_per_thread);
        while (block_count < minimum_block_count) {
            loops_per_thread /= 2;
            block_count = current_size / (2* threads_per_block * loops_per_thread);
            if (loops_per_thread == 1) break;
        }
        while (block_count < minimum_block_count) {
            threads_per_block /= 2;
            block_count = current_size / (2* threads_per_block * loops_per_thread);
            if (threads_per_block == 32) break;
        }
        if (block_count < minimum_block_count) {
            threads_per_block = optimal_threads_per_block;
            block_count = 1;
        }
        
        // do the kernel call via athe switch-case method
        this->reduce1D(input_data, output_data, current_size, block_count, threads_per_block, stream, device_order_number);
        check_errors(__FILE__, __LINE__);

        
        // set the output_data to be input for the next loop
        current_size = block_count;
        input_data = output_data;
        output_data = &input_data[block_count];
    } 
    check_errors(__FILE__, __LINE__);
    add_to_result<<<1, 1, 0, *stream>>> (
        device_result,
        input_data,
        this->number_of_dimensions,
        this->maximum_block_count,
        this->max_number_of_stored_results);
    check_errors(__FILE__, __LINE__);
}


/*
 * Does everything related to integration for a cube, namely uploads the cube
 * to the device, does the integration on devices and downloads the result.
 * Returns result of the integration.
 * 
 * NOTE: this function is blocking for cuda devices. Thus, if you intend
 * to perform more than one integration in a loop, separate use of 
 * integrateOnDevices and downloadResult is adviced.
 */
double Integrator3D::integrate(double *cube, int host_cube_shape[3]) {
    check_errors(__FILE__, __LINE__);
    // init cudacube with host pitch equal with the host x-length in bytes
    this->cube->initHost(cube, host_cube_shape, false);
    check_errors(__FILE__, __LINE__);
    // upload the cube to device
    this->cube->setAllMemoryToZero();
    check_errors(__FILE__, __LINE__);
    this->cube->uploadSliced();
    check_errors(__FILE__, __LINE__);
    
    // get the device memory shape
    int *device_memory_shape = this->cube->getDeviceMemoryShape();
    
    // do the integration
    this->integrateOnDevices(this->cube->getDeviceCubes(), this->cube->getDevicePitches(), device_memory_shape[1], this->grid->getShape());
    
    check_errors(__FILE__, __LINE__);
    // download the result
    double result;
    this->downloadResult(&result);
    check_errors(__FILE__, __LINE__);
    
    check_errors(__FILE__, __LINE__);
    this->cube->unsetHostCube();
    check_errors(__FILE__, __LINE__);
    // destroy the cube
    return result;
}

/*
 * Does integration at the device for the internal CudaCube, does not download result.
 * 
 * NOTE: this function is not blocking for cuda devices.
 */
void Integrator3D::integrate() {
    
    // get the device memory shape
    int *device_memory_shape = this->cube->getDeviceMemoryShape();
    
    // do the integration
    this->integrateOnDevices(this->cube->getDeviceCubes(), this->cube->getDevicePitches(), device_memory_shape[1], this->grid->getShape());
    
    check_errors(__FILE__, __LINE__);
}

/*
 * This function does the integration at all devices. Returns the integration value.
 */
void Integrator3D::integrateOnDevices(double **device_cubes, size_t *device_pitches, int device_memory_shape_y, int shape[3]) {
    // add the counter the follows the number of integrations
    if (this->last_result_counter == this->max_number_of_stored_results -1) {
        this->last_result_counter = 0;
    }
    else {
        this->last_result_counter ++;
    }
    int slice_offset = 0;
    for (int device = 0; device < streamContainer->getNumberOfDevices(); device ++) {
        this->integrateSingleDevice(device_cubes[device], 
                                    device_pitches[device], 
                                    device_memory_shape_y,
                                    shape,
                                    device, slice_offset, 
                                    &this->device_results[device][this->last_result_counter]);
        check_errors(__FILE__, __LINE__);
    }
    
}

void Integrator3D::integrateSingleDevice(double *device_cube, const size_t device_pitch, const int device_memory_shape_y, int shape[3],
                                         const int device, int &slice_offset, double *device_result) {
    

    int device_slice_count =  (this->grid->getShape(Z_)) / this->streamContainer->getNumberOfDevices()
                                  + ((this->grid->getShape(Z_) % this->streamContainer->getNumberOfDevices()) > device);
    this->streamContainer->setDevice(device);
    for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream ++) {
        int slice_count = device_slice_count / this->streamContainer->getStreamsPerDevice() 
                                + ((device_slice_count % this->streamContainer->getStreamsPerDevice()) > stream);
        
        // multiply the cube with the integration coefficients coming from lagrange interpolation polynomials (LIP)
        dim3 grid, block;
        check_errors(__FILE__, __LINE__);
        getCubeLaunchConfiguration(&grid, &block, shape, slice_count, 512);
        
        /*multiply_cube_with_vectors <<< grid, block, 0, *this->streamContainer->getStream(device, stream) >>>
                                        (device_cube,
                                         device_pitch, 
                                         device_memory_shape_y,
                                         this->grid->axis[X_]->getDeviceIntegrals(device),
                                         this->grid->axis[Y_]->getDeviceIntegrals(device),
                                         this->grid->axis[Z_]->getDeviceIntegrals(device),
                                         shape[0],
                                         shape[1],
                                         slice_count,
                                         slice_offset
                                        );
        check_errors(__FILE__, __LINE__);*/
        
        // sum the cube together to result array
        this->contractCube(device_cube, 
                      device_pitch,
                      device_memory_shape_y,
                      shape,
                      slice_count,
                      this->integration_space[device*this->streamContainer->getStreamsPerDevice() + stream],
                      this->streamContainer->getStream(device, stream),
                      device_result, 
                      slice_offset,
                      device);
        check_errors(__FILE__, __LINE__);
        
        cudaDeviceSynchronize();
        slice_offset += slice_count;
        device_cube = &device_cube[device_pitch * device_memory_shape_y * slice_count / sizeof(double)] ;

    }
}


void Integrator3D::setIntegrationCube(CudaCube *cube) {
    this->cube = cube;
}

__device__ __host__
Grid3D *Integrator3D::getGrid() {
    return this->grid;
}

CudaCube *Integrator3D::getIntegrationCube() {
    return this->cube;
}

void Integrator3D::destroy() {
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        streamContainer->setDevice(device);
        
        for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream ++) {
            cudaFree(this->integration_space[device*this->streamContainer->getStreamsPerDevice() + stream]);
        }
        cudaFree(this->device_results[device]);
        cudaFree(this->device_copies[device]);
    }
    cudaFreeHost(this->device_results);
    cudaFreeHost(this->integration_space);
    cudaFreeHost(this->device_copies);
    
    if (this->cube) {
        this->cube->destroy();
        delete this->cube;
    }
}

///////////////////////////////////
//  Integrator1D Definition      //
///////////////////////////////////

Integrator1D::Integrator1D(StreamContainer *streamContainer, Grid1D *grid, int processor_order_number, int number_of_processors, int number_of_dimensions) {
    this->streamContainer = streamContainer;
    this->processor_order_number = processor_order_number;
    this->number_of_processors = number_of_processors;
    check_errors(__FILE__, __LINE__);
    this->last_result_counter = -1;
    this->first_result_counter = -1;
    this->grid = grid;
    this->max_number_of_stored_results = 50;
    this->number_of_dimensions = number_of_dimensions;
    this->maximum_block_count = 1024;
    
    // allocate space for stream pointers
    this->device_vectors = new double*[streamContainer->getNumberOfDevices() * 3];
    cudaHostAlloc((void **)&this->device_results, 
                  sizeof(double *) * streamContainer->getNumberOfDevices(),
                  cudaHostAllocPortable);
    cudaHostAlloc((void **)&this->integration_space, 
                  sizeof(double *) * streamContainer->getNumberOfDevices() * streamContainer->getStreamsPerDevice(),
                  cudaHostAllocPortable);
    // allocate space for device copies
    cudaHostAlloc((void **)&this->device_copies, 
                  sizeof(this) * streamContainer->getNumberOfDevices(),
                  cudaHostAllocPortable);
    check_errors(__FILE__, __LINE__);
    
    // determine how many of the points belong to the current mpi-node
    int processor_point_count = this->grid->getShape() / this->number_of_processors 
                                + ((this->grid->getShape() % this->number_of_processors) > this->processor_order_number);
                                
    // get the offset to the integrals-array caused by other processors
    int remainder = this->grid->getShape() % this->number_of_processors;
    int offset = this->processor_order_number * this->grid->getShape() / this->number_of_processors 
                  + ((remainder < this->processor_order_number) ? remainder : this->processor_order_number); 
   
    for (int device = 0; device < streamContainer->getNumberOfDevices(); device ++) {
        streamContainer->setDevice(device);
        int device_point_count =  processor_point_count / streamContainer->getNumberOfDevices()
                                  + ((processor_point_count % streamContainer->getNumberOfDevices()) > device);
        
        // allocate space for device results and set the results to 0
        cudaMalloc(&this->device_results[device], sizeof(double)*this->max_number_of_stored_results * this->number_of_dimensions);
        cudaMemset(this->device_results[device], 0, sizeof(double)*this->max_number_of_stored_results * this->number_of_dimensions);
        
        // allocate space for the vectors
        cudaMalloc(&this->device_vectors[device], sizeof(double) * device_point_count );
        check_errors(__FILE__, __LINE__);
        
        
        for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream ++) {
            cudaMalloc(&this->integration_space[device*this->streamContainer->getStreamsPerDevice() + stream], sizeof(double) * this->maximum_block_count * this->number_of_dimensions);
            cudaMemset(this->integration_space[device*this->streamContainer->getStreamsPerDevice() + stream], 0, sizeof(double) * this->maximum_block_count * this->number_of_dimensions);
            check_errors(__FILE__, __LINE__);
        }
        
        // copy the device stuff to the device reference
        this->grid = grid->device_copies[device];
        cudaMalloc(&this->device_copies[device], sizeof(*this));
        cudaMemcpy(this->device_copies[device], this, sizeof(*this), cudaMemcpyHostToDevice);
    }
    this->grid = grid;
}

/*
 * Contract vector
 */
void Integrator1D::contractVector(double *device_vector, 
                     double *extra_data,
                     unsigned int vector_length, 
                     cudaStream_t *stream,
                     double *device_result,
                     int device_order_number) {

    double *input_data = device_vector;
    double *output_data = extra_data;
    unsigned int current_size = vector_length;
    unsigned int optimal_threads_per_block = 256;
    unsigned int threads_per_block;
    unsigned int optimal_loops_per_thread = 8;
    unsigned int minimum_block_count = 128;
    unsigned int loops_per_thread;
    unsigned int block_count = 1;
    while (current_size > 1) {
        loops_per_thread = optimal_loops_per_thread;
        threads_per_block = optimal_threads_per_block;
        // the kernel divides the number of data by 2*threads_per_block*loops_per_thread 
        block_count = current_size / (2 * threads_per_block * loops_per_thread);
        while (block_count < minimum_block_count) {
            loops_per_thread /= 2;
            block_count = current_size / (2* threads_per_block * loops_per_thread);
            if (loops_per_thread == 1) break;
        }
        while (block_count < minimum_block_count) {
            threads_per_block /= 2;
            block_count = current_size / (2* threads_per_block * loops_per_thread);
            if (threads_per_block == 32) break;
        }
        if (block_count < minimum_block_count) {
            threads_per_block = optimal_threads_per_block;
            block_count = 1;
        }
        // do the kernel call via athe switch-case method
        this->reduce1D(input_data, output_data, current_size, block_count, threads_per_block, stream, device_order_number);
        check_errors(__FILE__, __LINE__);

        
        // set the output_data to be input for the next loop
        current_size = block_count;
        input_data = output_data;
        output_data = &input_data[block_count];
    }
    // copy the final result to the result
    add_to_result<<<1, 1, 0, *stream>>> (
        device_result,
        input_data,
        this->number_of_dimensions,
        this->maximum_block_count,
        this->max_number_of_stored_results);
    check_errors(__FILE__, __LINE__);
}

/* 
 * This method does the integration for a host vector and returns the
 * correct value.
 * 
 * NOTE: to use this method, the input vector must be at least of lenth this->grid->getShape()
 */
double Integrator1D::integrate(double *vector) {
    // register the host vector as pinned memory
    cudaHostRegister(vector, sizeof(double)*this->grid->getShape(), cudaHostRegisterPortable);
    check_errors(__FILE__, __LINE__);
    
    // upload the host vector to devices
    this->upload(vector);
    check_errors(__FILE__, __LINE__);
    
    // do the integration
    this->integrateOnDevices(this->device_vectors);
    check_errors(__FILE__, __LINE__);
    
    // download the result
    double result;
    this->downloadResult(&result);
    
    //unregister the host vector
    cudaHostUnregister(vector);
    check_errors(__FILE__, __LINE__);
    return result;
}

/*
 * get the pointers to the device vectors
 */
double **Integrator1D::getDeviceVectors() {
    return this->device_vectors;
}

/*
 * This method does the integration with the internal device vector and returns the correct
 * value.
 */
double Integrator1D::integrate() {
    this->integrateOnDevices(this->device_vectors);
    double result;
    this->downloadResult(&result);
    return result;
}

/* 
 * This method uploads the vector to the devices, using the split this class uses
 *  
 * NOTE: to use this method, the input vector must be at least of lenth this->grid->getShape()
 */
void Integrator1D::upload(double *vector) {
    check_errors(__FILE__, __LINE__);
    // determine how many of the points belong to the current mpi-node
    int processor_point_count = this->grid->getShape() / this->number_of_processors 
                                + ((this->grid->getShape() % this->number_of_processors) > this->processor_order_number);
    // get the offset to the integrals-array caused by other processors
    int remainder = this->grid->getShape() % this->number_of_processors;
    int offset = this->processor_order_number * this->grid->getShape() / this->number_of_processors 
                  + ((remainder < this->processor_order_number) ? remainder : this->processor_order_number); 
    vector = &vector[offset];
    double *device_vector;
    for (int device = 0; device < streamContainer->getNumberOfDevices(); device ++) {
        streamContainer->setDevice(device);
        device_vector = this->device_vectors[device];
        int device_point_count =  processor_point_count / streamContainer->getNumberOfDevices()
                                  + ((processor_point_count % streamContainer->getNumberOfDevices()) > device);
        //for (int j = 0; j < device_point_count; j++) {
        //    vector[j] = 1.0;
        //}
        for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream ++) {
            int point_count = device_point_count / this->streamContainer->getStreamsPerDevice() 
                                    + ((device_point_count % this->streamContainer->getStreamsPerDevice()) > stream);
            
            double intsum = 0.0;
            for (int i = 0; i < point_count; i++)  {
                intsum += vector[i];
            }
            cudaMemcpyAsync(device_vector, vector, sizeof(double)*point_count, cudaMemcpyHostToDevice, *this->streamContainer->getStream(device, stream));
            check_errors(__FILE__, __LINE__);
            device_vector += point_count;
            vector += point_count;
        }
    }
}

/*
 * This function does the integration at all devices. Returns the integration value.
 */
void Integrator1D::integrateOnDevices(double **device_vectors) {
    if (this->last_result_counter == this->max_number_of_stored_results -1) {
        this->last_result_counter = 0;
    }
    else {
        this->last_result_counter ++;
    }
    check_errors(__FILE__, __LINE__);
    int offset = 0;
    
    // get the number of points integrated by this node
    int processor_point_count = this->grid->getShape() / this->number_of_processors 
                                + ((this->grid->getShape() % this->number_of_processors) > this->processor_order_number);
    // get the offset in points for this node
    int point_remainder = (this->grid->getShape() % this->number_of_processors);
    int processor_point_offset = this->processor_order_number * (this->grid->getShape() / this->number_of_processors)
                                 + ((point_remainder < this->processor_order_number) ? point_remainder : this->processor_order_number);
                                
    // do the device-wise integration
    for (int device = 0; device < streamContainer->getNumberOfDevices(); device ++) {
        this->integrateSingleDevice(device_vectors[device], 
                                    device, offset, 
                                    &this->device_results[device][this->last_result_counter], 
                                    processor_point_count,
                                    processor_point_offset
                                   );
        check_errors(__FILE__, __LINE__);
    }
}

/*
 * This function does the integration at one device
 */
void Integrator1D::integrateSingleDevice(double *device_vector, const int device, int &offset, double *device_result, int processor_point_count, int processor_point_offset) {

    int device_point_count =  processor_point_count / this->streamContainer->getNumberOfDevices()
                                  + ((processor_point_count % this->streamContainer->getNumberOfDevices()) > device);
    this->streamContainer->setDevice(device);
    int block_size = 256;
    double *device_integrals = &this->grid->getDeviceIntegrals(device)[processor_point_offset+offset];
    for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream ++) {
        int point_count = device_point_count / this->streamContainer->getStreamsPerDevice() 
                                + ((device_point_count % this->streamContainer->getStreamsPerDevice()) > stream);
        
        int grid_size = (point_count + block_size - 1) / block_size;
        //result_array[device * this->streamContainer->getStreamsPerDevice() + stream] = 0.0;
        // multiply the cube with the coefficients coming from lagrange interpolation polynomials (LIP)
        multiply_vectors <<< grid_size, block_size, 0, *this->streamContainer->getStream(device, stream) >>>
                                        (device_vector, device_integrals, point_count, 0);
        check_errors(__FILE__, __LINE__);
        
        // sum the array together to result array item
        this->contractVector(
            device_vector, 
            this->integration_space[device*this->streamContainer->getStreamsPerDevice() + stream],
            point_count, 
            this->streamContainer->getStream(device, stream),
            device_result,
            device);
        check_errors(__FILE__, __LINE__);
        offset += point_count;
        device_vector = &device_vector[point_count];
        device_integrals = &device_integrals[point_count];

    }
}

void Integrator1D::destroy() {
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        this->streamContainer->setDevice(device);
        check_errors(__FILE__, __LINE__);
        cudaFree(this->device_vectors[device]);
        cudaFree(this->device_results[device]);
        
        for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream ++) {
            cudaFree(this->integration_space[device*this->streamContainer->getStreamsPerDevice() + stream]);
            check_errors(__FILE__, __LINE__);
        }
        
        cudaFree(this->device_results[device]);
    }
    cudaFreeHost(this->device_results);
    cudaFreeHost(this->integration_space);
    delete[] device_vectors;
}

///////////////////////////////////
//  Fortran Interfaces           //
///////////////////////////////////

extern "C" Integrator3D *integrator3d_init(StreamContainer *streamContainer, Grid3D *grid) {
    return new Integrator3D(streamContainer, grid);
}
extern "C" void integrator3d_destroy(Integrator3D *integrator) {
    integrator->destroy();
    delete integrator;
}

extern "C" void integrator3d_integrate(Integrator3D *integrator, double *cube, int offset, int host_cube_shape[3], double *result) {
    *result = integrator->integrate(&cube[offset], host_cube_shape);
}



