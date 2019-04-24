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
/*! @file  spherical_harmonics_cuda.cu
 *! @brief CUDA implementation of the spherical harmonics evaluation.
 */


#include "streamcontainer.h"
#include "integrator.h"
#include "spherical_harmonics_cuda.h"
#include "bubbles_cuda.h"
#include "grid.h"
#include "cube.h"
#include "memory_leak_operators.h"
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>


#define X_ 0
#define Y_ 1
#define Z_ 2
#define R_ 3

#define TERM_COUNT 10
#define MAX_TERM_COUNT 300

/** \brief Size of the CUDA blocks in the X dimension */
#define BLOCKDIMX 8
/** \brief Size of the CUDA blocks in the Y dimension */
#define BLOCKDIMY 4
/** \brief Size of the CUDA blocks in the Z dimension */
#define BLOCKDIMZ 4
#define BLOCKSZ 64

cudaError_t cudaErrorStat;


__constant__ int shape_x_, shape_y_, shape_z_, lmax_, ilmmin_, lmin_, ilmmax_, first_term_, normalization_, ijk_max_;



inline __device__ void calc_distance(double *dist_vec_x, double *dist_vec_y, double *dist_vec_z, double *dist,
                              const double reference_point_x, const double reference_point_y, const double reference_point_z, 
                              const double x, const double y, const double z){
    // calculate the vector relative to reference_point
    *dist_vec_x=x-reference_point_x;
    *dist_vec_y=y-reference_point_y;
    *dist_vec_z=z-reference_point_z;
    
    // evaluate the length of the dist_vector, i.e., the distance between dist_vec and reference_point
    *dist=sqrt((*dist_vec_x) * (*dist_vec_x)+
               (*dist_vec_y) * (*dist_vec_y)+
               (*dist_vec_z) * (*dist_vec_z));
    return;
}


__device__ void RealSphericalHarmonics_evaluate_point_simple(const double x,
                                                             const double y,
                                                             const double z,
                                                             const double r,
                                                             const int lmax,
                                                             const int lm_address_difference,
                                                             double *result) {
    
    int lm_address =0, address2 = 0;
    int l, m, l2; 
    double top = 0.0, bottom = 0.0, new_bottom = 0.0, prev1 = 0.0, prev2 = 0.0, current = 0.0;
    //double r2 = x*x+y*y+z*z;
    // set value for l=0, m=0
    result[lm_address] = 1.0;
    
    // set value for l=1, m=-1
    lm_address += lm_address_difference;
    result[lm_address] = y / r;
      
   
    // set all values where m=-1
    m = -1;
    prev1 = y / r;
    // the starting address has 1 item before from the l=0, 3 from l=1, and 1 from l=2
    address2 = 5 * lm_address_difference;
    for (l = 2; l <= lmax; l++) {
        current =   ( 2.0*(double)l-1.0) / sqrt( 1.0*(double)((l+m)*(l-m)) ) * z*prev1 / r;
        if (l > 2) {
            current -=  sqrt( (double)((l+m-1)*(l-m-1)) /  (double)((l+m)*(l-m)) ) * prev2;
        }
        prev2 = prev1;
        prev1 = current;
        result[address2] = current;
        
        // add the address2 to get to the next item with m=0 
        address2 += lm_address_difference * (2*l+2);
    }
    
    // set value for l=1, m=0
    lm_address += lm_address_difference;
    result[lm_address] = z / r;
    
    // set all values where m=0
    prev1 = z / r;
    prev2 = 1.0;
    m = 0;
    // the starting address has 1 item before from the l=0, 3 from l=1, and 2 from l=2
    address2 = 6 * lm_address_difference;
    for (l = 2; l <= lmax; l++) {
        current =   ( 2.0*(double)l-1.0) / sqrt( 1.0*(double)((l+m)*(l-m)) ) * z * prev1 / r;
        current -=  sqrt( (double)((l+m-1)*(l-m-1)) /  (double)((l+m)*(l-m)) ) * prev2;
        prev2 = prev1;
        prev1 = current;
        result[address2] = current;
        
        // add the address2 to get to the next item with m=0 
        address2 += lm_address_difference * (2*l+2);
    }
    
    // set value for l=1, m=1
    lm_address += lm_address_difference;
    result[lm_address] = x / r;
    // set all values where m=1
    prev1 = x / r;
    m = 1;
    // the starting address has 1 item before from the l=0, 3 from l=1, and 3 from l=2
    address2 = 7 * lm_address_difference;
    for (l = 2; l <= lmax; l++) {
        current =   ( 2.0*(double)l-1.0) / sqrt( 1.0*(double)((l+m)*(l-m)) ) * z*prev1 / r;
        if (l > 2) {
            current -=  sqrt( (double)((l+m-1)*(l-m-1)) /  (double)((l+m)*(l-m)) ) * prev2;
        }
        prev2 = prev1;
        prev1 = current;
        result[address2] = current;
        
        // add the address2 to get to the next item with m=0 
        address2 += lm_address_difference * (2*l+2);
    }
    
    // go through the rest of the stuff
    bottom = y / r; // bottom refers to real spherical harmonics value with l=l-1 and m=-(l-1)
    top = x / r;    // top    refers to real spherical harmonics value with l=l-1 and m=l-1
    lm_address += lm_address_difference;
    for (l=2; l <= lmax; l++) {
        
        new_bottom = sqrt((2.0*(double)l - 1.0) / (2.0*(double)l)) * 
                       ( y*top + x*bottom) / r;
        result[lm_address] = new_bottom;
        
        // set all values where m=-l
        m = -l;
        prev1 = new_bottom;
        address2 = lm_address + (2*l+2) * lm_address_difference;
        for (l2 = l+1; l2 <= lmax; l2++) {
            current =   ( 2.0*(double)l2-1.0) / sqrt( 1.0*(double)((l2+m)*(l2-m)) ) * z*prev1 / r;
            if (l2 > l+1) {
                current -=  sqrt( (double)((l2+m-1)*(l2-m-1)) /  (double)((l2+m)*(l2-m)) ) * prev2;
            }
            prev2 = prev1;
            prev1 = current;
            result[address2] = current;
            
            // add the address2 to get to the next item with m=l 
            address2 += lm_address_difference * (2*l2+2);
        }
        
        
        // get value for l=l, m=l. The address is 2*l items away from l=l, m=-l
        lm_address += 2*l*lm_address_difference;
        top = sqrt((2.0*(double)l - 1.0) / (2.0*(double)l)) * 
                      ( x*top-y*bottom ) / r;
        // set all values where m=l
        m = l;
        prev1 = top;
        address2 = lm_address + (2*l+2) * lm_address_difference;
        for (l2 = l+1; l2 <= lmax; l2++) {
            current =   ( 2.0*(double)l2-1.0) / sqrt( 1.0*(double)((l2+m)*(l2-m)) ) * z*prev1 / r;
            if (l2 > l+1) {
                current -=  sqrt( (double)((l2+m-1)*(l2-m-1)) /  (double)((l2+m)*(l2-m)) ) * prev2;
            }
            prev2 = prev1;
            prev1 = current;
            result[address2] = current;
            
            // add the address2 to get to the next item with m=l 
            address2 += lm_address_difference * (2*l2+2);
        }
        // store the new bottom: l=l, m=-l (we need the old bottom in calculation of top)
        bottom = new_bottom;
        result[lm_address] = top;
        
        // get next address
        lm_address += lm_address_difference;
    }
}

__device__ void RealRegularSolidHarmonics_evaluate_point_simple(const double x,
                                                                const double y,
                                                                const double z,
                                                                const int lmax,
                                                                const int lm_address_difference,
                                                                double *result) {
    
    int lm_address =0, address2 = 0;
    int l, m, l2; 
    double top = 0.0, bottom = 0.0, new_bottom = 0.0, prev1 = 0.0, prev2 = 0.0, current = 0.0;
    double r2 = x*x+y*y+z*z;
    // set value for l=0, m=0
    result[lm_address] = 1.0;
    
    // set value for l=1, m=-1
    lm_address += lm_address_difference;
    result[lm_address] = y;
      
   
    // set all values where m=-1
    m = -1;
    prev1 = y;
    // the starting address has 1 item before from the l=0, 3 from l=1, and 1 from l=2
    address2 = 5 * lm_address_difference;
    for (l = 2; l <= lmax; l++) {
        current =   ( 2.0*(double)l-1.0) / sqrt( 1.0*(double)((l+m)*(l-m)) ) * z*prev1;
        if (l > 2) {
            current -=  sqrt( (double)((l+m-1)*(l-m-1)) /  (double)((l+m)*(l-m)) ) * r2 * prev2;
        }
        prev2 = prev1;
        prev1 = current;
        result[address2] = current;
        
        // add the address2 to get to the next item with m=0 
        address2 += lm_address_difference * (2*l+2);
    }
    
    // set value for l=1, m=0
    lm_address += lm_address_difference;
    result[lm_address] = z;
    
    // set all values where m=0
    prev1 = z;
    prev2 = 1.0;
    m = 0;
    // the starting address has 1 item before from the l=0, 3 from l=1, and 2 from l=2
    address2 = 6 * lm_address_difference;
    for (l = 2; l <= lmax; l++) {
        current =   ( 2.0*(double)l-1.0) / sqrt( 1.0*(double)((l+m)*(l-m)) ) * z * prev1;
        current -=  sqrt( (double)((l+m-1)*(l-m-1)) /  (double)((l+m)*(l-m)) ) * r2 * prev2;
        prev2 = prev1;
        prev1 = current;
        result[address2] = current;
        
        // add the address2 to get to the next item with m=0 
        address2 += lm_address_difference * (2*l+2);
    }
    
    // set value for l=1, m=1
    lm_address += lm_address_difference;
    result[lm_address] = x;
    // set all values where m=1
    prev1 = x;
    m = 1;
    // the starting address has 1 item before from the l=0, 3 from l=1, and 3 from l=2
    address2 = 7 * lm_address_difference;
    for (l = 2; l <= lmax; l++) {
        current =   ( 2.0*(double)l-1.0) / sqrt( 1.0*(double)((l+m)*(l-m)) ) * z*prev1;
        if (l > 2) {
            current -=  sqrt( (double)((l+m-1)*(l-m-1)) /  (double)((l+m)*(l-m)) ) * r2 * prev2;
        }
        prev2 = prev1;
        prev1 = current;
        result[address2] = current;
        
        // add the address2 to get to the next item with m=0 
        address2 += lm_address_difference * (2*l+2);
    }
    
    // go through the rest of the stuff
    bottom = y; // bottom refers to solid harmonics value with l=l-1 and m=-(l-1)
    top = x;    // top    refers to solid harmonics value with l=l-1 and m=l-1
    lm_address += lm_address_difference;
    for (l=2; l <= lmax; l++) {
        
        new_bottom = sqrt((2.0*(double)l - 1.0) / (2.0*(double)l)) * 
                       ( y*top + x*bottom);
        result[lm_address] = new_bottom;
        
        // set all values where m=-l
        m = -l;
        prev1 = new_bottom;
        address2 = lm_address + (2*l+2) * lm_address_difference;
        for (l2 = l+1; l2 <= lmax; l2++) {
            current =   ( 2.0*(double)l2-1.0) / sqrt( 1.0*(double)((l2+m)*(l2-m)) ) * z*prev1;
            if (l2 > l+1) {
                current -=  sqrt( (double)((l2+m-1)*(l2-m-1)) /  (double)((l2+m)*(l2-m)) ) * r2 * prev2;
            }
            prev2 = prev1;
            prev1 = current;
            result[address2] = current;
            
            // add the address2 to get to the next item with m=l 
            address2 += lm_address_difference * (2*l2+2);
        }
        
        
        // get value for l=l, m=l. The address is 2*l items away from l=l, m=-l
        lm_address += 2*l*lm_address_difference;
        top = sqrt((2.0*(double)l - 1.0) / (2.0*(double)l)) * 
                      ( x*top-y*bottom );
        // set all values where m=l
        m = l;
        prev1 = top;
        address2 = lm_address + (2*l+2) * lm_address_difference;
        for (l2 = l+1; l2 <= lmax; l2++) {
            current =   ( 2.0*(double)l2-1.0) / sqrt( 1.0*(double)((l2+m)*(l2-m)) ) * z*prev1;
            if (l2 > l+1) {
                current -=  sqrt( (double)((l2+m-1)*(l2-m-1)) /  (double)((l2+m)*(l2-m)) ) * r2 * prev2;
            }
            prev2 = prev1;
            prev1 = current;
            result[address2] = current;
            
            // add the address2 to get to the next item with m=l 
            address2 += lm_address_difference * (2*l2+2);
        }
        // store the new bottom: l=l, m=-l (we need the old bottom in calculation of top)
        bottom = new_bottom;
        result[lm_address] = top;
        
        // get next address
        lm_address += lm_address_difference;
    }
}


__device__ inline void getXYZ3D(int *x, int *y, int *z) {
    *x = blockIdx.x * blockDim.x + threadIdx.x;
    *y = blockIdx.y * blockDim.y + threadIdx.y;
    *z = blockIdx.z * blockDim.z + threadIdx.z;
}

/*__global__ void RealRegularSolidHarmonics_evaluate_3d( int shape_x, int shape_y, shape_z,
                                                       int slice_offset, int slice_count,
                                                       int pitch, int memory_shape_y, int memory_shape_z,
                                                       double *cubes, const double zero_point_x, const double zero_point_y, const double zero_point_z,
                                                       const int lmin, const int lmax) {    
    // get the id of the point (We are using only the first )
    int x, y, z;
    getXYZ3D(&x, &y, &z);
    z += slice_offset;
    
                        
    // Check that the point is within the block & is within the handled slices
    if (x < shape_x && y < shape_y && z < shape_z && z < slice_count) {
                  
        double relative_position_x, relative_position_y, relative_position_z, distance = 0.0;
        // calculate relative position to the zero-point and distance to it 
        calc_distance(&relative_position_x,
                      &relative_position_y,
                      &relative_position_z,
                      &distance, 
                      zero_point_x, 
                      zero_point_y,
                      zero_point_z,
                      grid->grid_points_x[x],
                      grid->grid_points_y[y],
                      grid->grid_points_z[z]);
            
        // calculate the solid harmonic value for the point
        RealRegularSolidHarmonics_evaluate_point_simple(relative_position_x,
                                                        relative_position_y,
                                                        relative_position_z,
                                                        lmax,
                                                        (int) pitch * memory_shape_y * memory_shape_z,
                                                        &cubes[id]);
    } 
    return;

}*/


__global__ void RealRegularSolidCubeHarmonics_evaluate_grid( const int shape_x, const int shape_y, const int shape_z,
                                                             const double *gridpoints_x, const double *gridpoints_y, const double *gridpoints_z,
                                                             const int lmax,
                                                             double *cubes, const size_t pitch,
                                                             const double zero_point_x, const double zero_point_y, const double zero_point_z,
                                                             const int slice_offset,
                                                             // the number of slices handled by this kernel call
                                                             const int slice_count,
                                                             // the number of slices that resides in the memory of this device
                                                             const int device_slice_count,
                                                             // order number of device used in this evaluation
                                                             const int device_order_number
                                                           ) {  
    //const int shape_x = grid->shape[X_], shape_y = grid->shape[Y_], shape_z = grid->shape[Z_];
    /*const int shape_x = grid->axis[X_]->ncell * (grid->axis[X_]->nlip - 1) + 1;
    const int shape_y = grid->axis[Y_]->ncell * (grid->axis[Y_]->nlip - 1) + 1;
    const int shape_z = grid->axis[Z_]->ncell * (grid->axis[Z_]->nlip - 1) + 1;*/
    
    // The result array will be in fortran with indices l, x, y, z. 
    // This means that the x index will be the fastest to change.
    int x, y, z;
    getXYZ3D(&x, &y, &z);
    
                        
    // Check that the point is within the block 
    if (x < shape_x &&
        y < shape_y &&
        z+slice_offset < shape_z &&
        z < slice_count) {
        
        // get the id of the point in the result array
        int id = + z * shape_y * pitch / sizeof(double)
                 + y * pitch / sizeof(double)
                 + x;
        double relative_position_x, relative_position_y, relative_position_z, distance = 0.0;
        // calculate relative position to the zero-point and distance to it 
        calc_distance(&relative_position_x,
                      &relative_position_y,
                      &relative_position_z,
                      &distance, 
                      zero_point_x, 
                      zero_point_y,
                      zero_point_z,
                      gridpoints_x[x],
                      gridpoints_y[y],
                      gridpoints_z[z+slice_offset]);
        
        
            
        // calculate the solid harmonic value for the point
        RealRegularSolidHarmonics_evaluate_point_simple(relative_position_x,
                                                        relative_position_y,
                                                        relative_position_z,
                                                        lmax,
                                                        (int) pitch / sizeof(double) * shape_y * device_slice_count,
                                                        &cubes[id]);
    } 
    return;

}

__global__ void RealRegularSolidHarmonics_evaluate_grid_kernel_fast( Grid3D *grid,
                                                               double *cubes, const double zero_point_x, const double zero_point_y, const double zero_point_z,
                                                               const int lmin, const int lmax){    
    // get the id of the point (We are using only the first )
    const int id=threadIdx.x + blockIdx.x * blockDim.x;
    
    const int shape_x = grid->axis[X_]->ncell * (grid->axis[X_]->nlip - 1) + 1;
    const int shape_y = grid->axis[Y_]->ncell * (grid->axis[Y_]->nlip - 1) + 1;
    const int shape_z = grid->axis[Z_]->ncell * (grid->axis[Z_]->nlip - 1) + 1;
    
    // The result array will be in fortran with indices l, x, y, z. 
    // This means that the x index will be the fastest to change.
    const short z = id / (shape_x * shape_y);
    const short y = (id - z * shape_x * shape_y) / (shape_x);
    const short x = (id - z * shape_x * shape_y - y * shape_x); 
    
                        
    // Check that the point is within the block 
    if (x < shape_x &&
        y < shape_y &&
        z < shape_z) {
                  
        double relative_position_x, relative_position_y, relative_position_z, distance = 0.0;
        // calculate relative position to the zero-point and distance to it 
        calc_distance(&relative_position_x,
                      &relative_position_y,
                      &relative_position_z,
                      &distance, 
                      zero_point_x, 
                      zero_point_y,
                      zero_point_z,
                      grid->axis[X_]->gridpoints[x],
                      grid->axis[Y_]->gridpoints[y],
                      grid->axis[Z_]->gridpoints[z]);
            
        // calculate the solid harmonic value for the point
        RealRegularSolidHarmonics_evaluate_point_simple(relative_position_x,
                                                        relative_position_y,
                                                        relative_position_z,
                                                        lmax,
                                                        (int) shape_x * shape_y * shape_z,
                                                        &cubes[id]);
    } 
    return;

}


__global__ void RealSphericalCubeHarmonics_evaluate_grid( const int shape_x, const int shape_y, const int shape_z,
                                                          const double *gridpoints_x, const double *gridpoints_y, const double *gridpoints_z,
                                                          const int lmin, const int lmax, const int normalization,
                                                          double *cubes, size_t pitch,
                                                          const double zero_point_x, const double zero_point_y, const double zero_point_z,
                                                          const int slice_offset,
                                                          // the number of slices handled by this kernel call
                                                          const int slice_count,
                                                          // the number of slices that resides in the memory of this device
                                                          const int device_slice_count) {    
    
    // The result array will be in fortran with indices l, x, y, z. 
    // This means that the x index will be the fastest to change.
    int x, y, z, i, l, m;
    double normalization_factor;
    getXYZ3D(&x, &y, &z);
    
                        
    // Check that the point is within the block 
    if (x < shape_x &&
        y < shape_y &&
        z+slice_offset < shape_z &&
        z < slice_count) {
        
        // get the id of the point in the result array
        int id = + z * shape_y * pitch / sizeof(double)
                 + y * pitch / sizeof(double)
                 + x;
                  
        double relative_position_x, relative_position_y, relative_position_z, distance = 0.0;
        // calculate relative position to the zero-point and distance to it 
        calc_distance(&relative_position_x,
                      &relative_position_y,
                      &relative_position_z,
                      &distance, 
                      zero_point_x, 
                      zero_point_y,
                      zero_point_z,
                      gridpoints_x[x],
                      gridpoints_y[y],
                      gridpoints_z[z+slice_offset]);
            
        
        // calculate the real harmonics values for the point
        if (distance > 1e-12) {
            RealSphericalHarmonics_evaluate_point_simple(relative_position_x,
                                                         relative_position_y,
                                                         relative_position_z,
                                                         distance,
                                                         lmax,
                                                         (int) pitch / sizeof(double) * shape_y * device_slice_count,
                                                         &cubes[id]);
        }
        else {
            i = 0;
            for (l = lmin; l <= lmax; l++) {
                for (m = -l; m <= l; m++) {
                    cubes[id+i] = 0.0;
                    i += pitch / sizeof(double) * shape_y * device_slice_count;
                }
            }
            if (lmin == 0) cubes[id] = 1.0;
        }
        
        // Multiply with normalization factor sqrt((2*l+1) / (4 * pi)), if 
        // we are using conventional normalization
        if (normalization == 2) {
            i = 0;
            normalization_factor = 1.0;
            for (l = lmin; l <= lmax; l++) {
                normalization_factor = sqrt((2.0*(double)l+1.0)/(4.0*M_PI));
                for (m = -l; m <= l; m++) {
                    cubes[id+i] *= normalization_factor;
                    i += pitch / sizeof(double) * shape_y * device_slice_count;
                }
            }
        
        }
    } 
    return;

}

__global__ void RealSphericalHarmonics_evaluate_grid_kernel_fast(Grid3D *grid,
                                                               double *cubes, const double zero_point_x, const double zero_point_y, const double zero_point_z, 
                                                               const int lmin, const int lmax, const int normalization
                                                                ) {
    
    
    int i = 0;
    // load the number_of_lm_terms to shared memory
    
    /* extern __shared__ int shared_memory[]; 
    int *number_of_lm_terms = shared_memory;
    
    i = threadIdx.x;
    while (i < ijk_max_) {
        number_of_lm_terms[i] = harmonics->number_of_lm_terms[i];
        i += blockDim.x;
    }
    __syncthreads();
    
    
    i = threadIdx.x;
    int *lm_indices = &shared_memory[ijk_max_];
    while (i < number_of_lm_terms[ijk_max_-1]) {
        lm_indices[i] = harmonics->lm_indices[i];
        i += blockDim.x;
    }
    __syncthreads();
    
    
    // load the coefficients to shared memory
    i = threadIdx.x;
    double *coefficients = (double * )&shared_memory[ijk_max_+number_of_lm_terms[ijk_max_-1]];
    while (i < number_of_lm_terms[ijk_max_-1]) {
        coefficients[i] = harmonics->new_coefficients[i];
        i += blockDim.x;
    }
    __syncthreads();*/
    
    // get the id of the point (We are using only the first )
    const int id=threadIdx.x + blockIdx.x * blockDim.x;
    
    const int shape_x = grid->axis[X_]->ncell * (grid->axis[X_]->nlip - 1) + 1;
    const int shape_y = grid->axis[Y_]->ncell * (grid->axis[Y_]->nlip - 1) + 1;
    const int shape_z = grid->axis[Z_]->ncell * (grid->axis[Z_]->nlip - 1) + 1;
    
    // The result array will be in fortran with indices l, x, y, z. 
    // This means that the x index will be the fastest to change.
    const int z = id / (shape_x * shape_y);
    const int y = (id - z * shape_x * shape_y) / (shape_x);
    const int x = (id - z * shape_x * shape_y - y * shape_x); 
    
    int l = 0;
    int m = 0;
    
    double normalization_factor = 0.0;                    
    // Check that the point is within the block 
    if (x < shape_x &&
        y < shape_y &&
        z < shape_z) {
        // get pointer to the result array value we are evaluating
        // first get the number of lm-pairs
        //const int address =   z * (shape_x * shape_y)
        //              + y * (shape_x)
        //              + x;
        
                  
        double relative_position_x, relative_position_y, relative_position_z, distance = 0.0;
        // calculate relative position to the zero-point and distance to it 
        calc_distance(&relative_position_x,
                      &relative_position_y,
                      &relative_position_z,
                      &distance, 
                      zero_point_x, 
                      zero_point_y,
                      zero_point_z,
                      grid->axis[X_]->gridpoints[x],
                      grid->axis[Y_]->gridpoints[y],
                      grid->axis[Z_]->gridpoints[z]);
        
            
        // calculate the solid harmonic values for the point
        
        //RealRegularSolidHarmonics_evaluate_point_new(harmonics, relative_position_x, relative_position_y,
        //                                                          relative_position_z, (int) shape_x * shape_y * shape_z,
        //                                                          number_of_lm_terms, coefficients, lm_indices, &cubes[address]);
        if (distance > 1e-4) {
            RealSphericalHarmonics_evaluate_point_simple(relative_position_x,
                                                        relative_position_y,
                                                        relative_position_z,
                                                        distance,
                                                        lmax,
                                                        (int) shape_x * shape_y * shape_z,
                                                        &cubes[id]);
        }
        else {
            cubes[id] = 1.0;
        }
        
        // Multiply with normalization factor sqrt((2*l+1) / (4 * pi)), if 
        // we are using conventional normalization
        if (normalization == 2) {
            i = 0;
            normalization_factor = 1.0;
            for (l = lmin; l <= lmax; l++) {
                normalization_factor = sqrt((2.0*(double)l+1.0)/(4.0*M_PI));
                for (m = -l; m <= l; m++) {
                    cubes[id+i] *= normalization_factor;
                    i += shape_x*shape_y*shape_z;
                }
            }
        
        }
    } 
    return;

}







__host__ inline void check_cuda_errors(const char *filename, const int line_number) {
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







/*
 * Set cube values withing width, height and depth to zero
 */
__global__ void set_cube_to_zero(double *cube, long pitch, long row_count, const int width, const int height, const int depth, const int size) {
    
    // get the id of the point (We are using only the first )
    const int id=threadIdx.x + blockIdx.x * blockDim.x;
    
    const int shape_x = width;
    const int shape_y = height;
    
    // The result array will be in fortran with indices l, x, y, z. 
    // This means that the x index will be the fastest to change.
    const int z = id / (shape_x * shape_y);
    const int y = (id - z * shape_x * shape_y) / (shape_x);
    const int x = (id - z * shape_x * shape_y - y * shape_x); 
    
    long cube_pointer;
    double *cube_value;
    
    if (x < width && y < height && z < depth) {
        cube_pointer = (long)cube + x * size + y * pitch  + z * row_count * pitch;
        cube_value = (double *)cube_pointer;
        *cube_value = 0.0;
    }
}

extern "C" void cube_set_to_zero_(int *, int *, int *, int *,
               long *, long *, long *, long *, int *, int *);

extern "C" void cube_set_to_zero_(int *deviceID, int *width, int *height, int *depth,
               long *devPtr, long *pitch, long *x_size, long *y_size, int *size, int *number_of_devices) {
    // Allocate cube
    // create extent for the cube 
    
    int device = (*deviceID-1)%(*number_of_devices);
    cudaSetDevice(device);
    
    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the
                        // maximum occupancy for a full device
                        // launch
    int gridSize;       // The actual grid size needed, based on input
                        // size

    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)set_cube_to_zero,
        0,
        (*width) * (*height) * (*depth));

    // Round up according to array size
    gridSize = ((*width) * (*height) * (*depth) + blockSize - 1) / blockSize;
    //printf("device: %d, gridsize: %d, blocksize:%d, dev cube: %ld, pitch: %ld, y_size: %ld, width: %d, height: %d, depth: %d", 
    //    device, gridSize, blockSize, *devPtr, *pitch, *y_size, *width, *height, *depth
    //);
    set_cube_to_zero<<<gridSize, blockSize>>> ((double *)*devPtr, *pitch, *y_size, *width, *height, *depth, *size);
    
//     cudaExtent extent = make_cudaExtent(*width * sizeof(RP), *height, *depth);
//     cudaPitchedPtr devPitchedPtr = make_cudaPitchedPtr((void *)*devPtr, *pitch, *x_size, *y_size);
//     printf("setting zero ptr %ld, xsize: %ld, ysize: %ld, pitch: %ld", *devPtr, devPitchedPtr.xsize, devPitchedPtr.ysize, devPitchedPtr.pitch);
//     set_cube_to_zero<<<>>>
    check_cuda_errors(__FILE__, __LINE__);
    return;
}
    
/*************************************************
 * Host RealHarmonics functionality              *
 * ***********************************************/

void RealHarmonics::initRealHarmonics(int lmin, int lmax, int normalization, StreamContainer *streamContainer) {
    this->lmin = lmin;
    this->lmax = lmax;
    this->normalization = normalization;
    this->streamContainer = streamContainer;
}
    
/*************************************************
 * Host RealCubeHarmonics functionality          *
 * ***********************************************/

int *RealCubeHarmonics::getShape() {
    return this->shape;
}

double **RealCubeHarmonics::getDeviceResults() {
    return this->device_results;
}

double *RealCubeHarmonics::getDeviceResults(int device) {
    return this->device_results[device];
}

size_t *RealCubeHarmonics::getDevicePitches() {
    return this->device_pitches;   
}

size_t RealCubeHarmonics::getDevicePitch(int device) {
    return this->device_pitches[device];
}

void RealCubeHarmonics::initCubeHarmonics(int lmin, int lmax, int normalization, int shape[3], StreamContainer *streamContainer) {
    this->initRealHarmonics(lmin, lmax, normalization, streamContainer);
    // allocate space for device cube pointers
    this->device_results = new double*[this->streamContainer->getNumberOfDevices()];
    this->device_pitches = new size_t[this->streamContainer->getNumberOfDevices()];
    this->device_copies = new RealHarmonics*[this->streamContainer->getNumberOfDevices()];
    
    // copy the shape
    this->shape[X_] = shape[X_];
    this->shape[Y_] = shape[Y_];
    this->shape[Z_] = shape[Z_];
    
    // the limits of the lmax array
    int ilmmax = (this->lmax+1)*(this->lmax+1);
    //int ilmmin = (this->lmin)*(this->lmin);
    
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        // set the correct GPU
        this->streamContainer->setDevice(device);
        cudaPitchedPtr pointer;
        
        // get the portion that is handled by device with order number 'device'
        int device_slice_count =  shape[Z_] / this->streamContainer->getNumberOfDevices()
                                  + ((shape[Z_] % this->streamContainer->getNumberOfDevices()) > device);
        
        // allocate memory for entire shape for the main pointers
        cudaExtent extent = make_cudaExtent(shape[X_] * sizeof(double), shape[Y_], device_slice_count * ilmmax );
        cudaMalloc3D (&pointer, extent);
        
        check_cuda_errors(__FILE__, __LINE__);
        this->device_pitches[device] = pointer.pitch;
        this->device_results[device] = (double *) pointer.ptr;
        
        // allocate the device memory and copy
        cudaMalloc(&this->device_copies[device], sizeof(*this));
        cudaMemcpy(this->device_copies[device], this, sizeof(*this), cudaMemcpyHostToDevice);
        check_cuda_errors(__FILE__, __LINE__);
    }
}

void RealCubeHarmonics::evaluate(Grid3D *grid, double center[3]) {
    check_cuda_errors(__FILE__, __LINE__);       
    int slice_offset = 0;
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        // set the correct GPU
        this->streamContainer->setDevice(device);
        size_t device_pitch = this->getDevicePitch(device);
        double *device_results = this->getDeviceResults(device);
        
        // get the portion that is handled by device with order number 'device'
        int device_slice_count =  shape[Z_] / this->streamContainer->getNumberOfDevices()
                                  + ((shape[Z_] % this->streamContainer->getNumberOfDevices()) > device);
                                  
        for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream ++) {
            // get the portion that is handled by stream with order number 'stream'
            int slice_count = device_slice_count / this->streamContainer->getStreamsPerDevice() 
                                + ((device_slice_count % this->streamContainer->getStreamsPerDevice()) > stream);
                    
            // do the per-stream evaluation (NOTE: the functions hide the kernel calls to the spherical harmonics / solid harmonics)
            this->evaluateSingleStream(device_results, device_pitch, device, grid, center, 
                                       slice_count, device_slice_count, slice_offset, 
                                       this->streamContainer->getStream(device, stream));
            check_cuda_errors(__FILE__, __LINE__);                   
            
            // add to the slice_offset
            slice_offset += slice_count;
            
            // add to the cube pointer
            device_results += device_pitch / sizeof(double) * this->shape[Y_] * slice_count;
        }
    }
}

/*
 * Note: works best if the host_results is registered/inited as pinned before using this method
 * 
 * @param host_results pointer to a four dimensional array of shape (shape[X_], shape[Y_], shape[Z_], lmax)
 * @param host_results_shape (x, y, z, l)
 */
void RealCubeHarmonics::download(double *host_results, int host_results_shape[4]) {
    check_cuda_errors(__FILE__, __LINE__);
    
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        // set the correct GPU
        this->streamContainer->setDevice(device);
        
        // and get the corresponding cube pointer and pitch
        size_t device_pitch = this->getDevicePitch(device);
        double *device_results = this->getDeviceResults(device);
        
        // get the portion that is handled by device with order number 'device'
        int device_slice_count =  this->shape[Z_]  / this->streamContainer->getNumberOfDevices()
                                  + ((this->shape[Z_] * (this->lmax+1) % this->streamContainer->getNumberOfDevices()) > device);
                                  
        for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream ++) {
            // get the portion that is handled by stream with order number 'stream'
            int slice_count = device_slice_count / this->streamContainer->getStreamsPerDevice() 
                                + ((device_slice_count % this->streamContainer->getStreamsPerDevice()) > stream);
            int lm_offset = 0;
            int lm_device_offset = 0;
            if (slice_count > 0) {
                for (int n = this->lmin*this->lmin; n < (this->lmax +1) * (this->lmax +1); n++) {
                    cudaMemcpy3DParms memCopyParameters = {0};
                    memCopyParameters.dstPtr = make_cudaPitchedPtr(&host_results[lm_offset],          host_results_shape[X_]*sizeof(double),  host_results_shape[X_],  host_results_shape[Y_]);
                    memCopyParameters.srcPtr = make_cudaPitchedPtr(&device_results[lm_device_offset], device_pitch, shape[X_], shape[Y_]);
                    memCopyParameters.extent = make_cudaExtent(this->shape[X_] * sizeof(double), this->shape[Y_], slice_count);
                    memCopyParameters.kind   = cudaMemcpyDeviceToHost;
                    
                    // copy the f1 cube to device: 3D
                    cudaMemcpy3DAsync(&memCopyParameters, 
                                    *this->streamContainer->getStream(device, stream));
                    check_cuda_errors(__FILE__, __LINE__);       
                    
                    // add to the offsets caused by the l
                    lm_offset        += this->shape[X_] * this->shape[Y_] * this->shape[Z_];
                    lm_device_offset += device_pitch / sizeof(double) * this->shape[Y_] * device_slice_count;
                    
                }
                
                // add to the result pointers    
                host_results += slice_count *  host_results_shape[X_] * host_results_shape[Y_];
                device_results += device_pitch / sizeof(double) * this->shape[Y_] * slice_count;
            }
        }
    }
}

void RealCubeHarmonics::registerHostResultArray(double *host_results, int host_results_shape[4]) {
    
    // register host memory for download
    cudaHostRegister(host_results, host_results_shape[0]*host_results_shape[1]*host_results_shape[2]*host_results_shape[3] * sizeof(double), cudaHostRegisterPortable);
    check_cuda_errors(__FILE__, __LINE__);       
}

void RealCubeHarmonics::unregisterHostResultArray(double *host_results) {
    // unregister host memory
    cudaHostUnregister(host_results);
    check_cuda_errors(__FILE__, __LINE__); 
}

void RealCubeHarmonics::destroy() {
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        cudaFree(this->device_results[device]);
        cudaFree(this->device_copies[device]);
    }
    
    delete[] this->device_results;
    delete[] this->device_pitches;
    delete[] this->device_copies;
}

/****************************************************
 * Host RealRegularSolidCubeHarmonics functionality *
 ****************************************************/

RealRegularSolidCubeHarmonics::RealRegularSolidCubeHarmonics(int lmin, int lmax, int normalization, int shape[3], StreamContainer *streamContainer) {
    this->initCubeHarmonics(lmin, lmax, normalization, shape, streamContainer);
}

void RealRegularSolidCubeHarmonics::evaluateSingleStream(double *device_results, size_t device_pitch, int device,
                                                         Grid3D *grid3d, double center[3], 
                                                         int slice_count, int device_slice_count, int slice_offset, cudaStream_t *stream) {
    if (slice_count > 0) {
        // get the launch configuration
        dim3 grid, block;
        getCubeLaunchConfiguration(&grid, &block, this->shape, slice_count, 256);
        // call the kernel
        RealRegularSolidCubeHarmonics_evaluate_grid
        <<< grid, block, 0, *stream >>>
        ( grid3d->getShape(X_), grid3d->getShape(Y_), grid3d->getShape(Z_),
            grid3d->axis[X_]->device_gridpoints[device],
            grid3d->axis[Y_]->device_gridpoints[device],
            grid3d->axis[Z_]->device_gridpoints[device],
            lmax,
            device_results, device_pitch,
            center[X_], center[Y_], center[Z_],
            slice_offset, slice_count, device_slice_count, device);
        check_cuda_errors(__FILE__, __LINE__);
    }
    
}

/****************************************************
 * Host RealSphericalCubeHarmonics functionality    *
 ****************************************************/

RealSphericalCubeHarmonics::RealSphericalCubeHarmonics(int lmin, int lmax, int normalization, int shape[3], StreamContainer *streamContainer) {
    this->initCubeHarmonics(lmin, lmax, normalization, shape, streamContainer);
}

void RealSphericalCubeHarmonics::evaluateSingleStream(double *device_results, size_t device_pitch, int device,
                                                      Grid3D *grid3d, double center[3], 
                                                      int slice_count, int device_slice_count, int slice_offset, cudaStream_t *stream) {
    

    if (slice_count > 0) {
        // get the launch configuration
        dim3 grid, block;
        getCubeLaunchConfiguration(&grid, &block, this->shape, slice_count, 256);
        // call the kernel
        RealSphericalCubeHarmonics_evaluate_grid
        <<< grid, block, 0, *stream >>>
        ( grid3d->getShape(X_), grid3d->getShape(Y_), grid3d->getShape(Z_),
            grid3d->axis[X_]->device_gridpoints[device],
            grid3d->axis[Y_]->device_gridpoints[device],
            grid3d->axis[Z_]->device_gridpoints[device],
            this->lmin, this->lmax, this->normalization,
            device_results, device_pitch,
            center[X_], center[Y_], center[Z_],
            slice_offset, slice_count, device_slice_count
        );
        check_cuda_errors(__FILE__, __LINE__);
    }
}

/*******************************************************
 * Fortran interfaces - RealRegularSolidCubeHarmonics  *
 *******************************************************/

extern "C" RealRegularSolidCubeHarmonics *realregularsolidcubeharmonics_init_cuda(int lmin, int lmax, int normalization, int shape[3], StreamContainer *streamContainer) {
    return new RealRegularSolidCubeHarmonics(lmin, lmax, normalization, shape, streamContainer);
}

extern "C" void realregularsolidcubeharmonics_destroy_cuda(RealRegularSolidCubeHarmonics *harmonics) {
    harmonics->destroy();
}

extern "C" void realregularsolidcubeharmonics_evaluate_cuda(RealRegularSolidCubeHarmonics *harmonics, Grid3D *grid, double center[3]) {
    harmonics->evaluate(grid, center);
}

/****************************************************
 * Fortran interfaces - RealSphericalCubeHarmonics  *
 ****************************************************/

extern "C" RealSphericalCubeHarmonics *realsphericalcubeharmonics_init_cuda(int lmin, int lmax, int normalization, int shape[3], StreamContainer *streamContainer) {
    return new RealSphericalCubeHarmonics(lmin, lmax, normalization, shape, streamContainer);
}

extern "C" void realsphericalcubeharmonics_download_cuda(RealSphericalCubeHarmonics *harmonics, double *host_results, int host_results_shape[4]) {
    harmonics->download(host_results, host_results_shape);
}

extern "C" void realsphericalcubeharmonics_destroy_cuda(RealSphericalCubeHarmonics *harmonics) {
    harmonics->destroy();
}

extern "C" void realsphericalcubeharmonics_evaluate_cuda(RealSphericalCubeHarmonics *harmonics, Grid3D *grid, double center[3]) {
    harmonics->evaluate(grid, center);
}

/****************************************************
 * Fortran interfaces - RealCubeHarmonics           *
 ****************************************************/

extern "C" void realcubeharmonics_register_result_array_cuda(RealCubeHarmonics *harmonics, double *host_results, int host_results_shape[4]) {
    harmonics->registerHostResultArray(host_results, host_results_shape);
}

extern "C" void realcubeharmonics_unregister_result_array_cuda(RealCubeHarmonics *harmonics, double *host_results, int host_results_shape[4]) {
    harmonics->unregisterHostResultArray(host_results);
}




