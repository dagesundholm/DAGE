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
#include <math.h>
#include <stdlib.h>
#include "../bubbles/grid.h"
#include "../bubbles/streamcontainer.h"
#include "../bubbles/cube.h"
#include "../bubbles/integrator.h"
#include "../bubbles/spherical_harmonics_cuda.h"
#include "gbfmm_helmholtz3d.h"
#include "gbfmm_potential_operator.h"
#include "gbfmm_coulomb3d.h"
#include "../bubbles/memory_leak_operators.h"

#define X_ 0
#define Y_ 1
#define Z_ 2
#define BLOCK_SIZE 512

__host__ inline void check_helmholtz3d_errors(const char *filename, const int line_number) {
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

__device__ inline void getXYZ3D(int *x, int *y, int *z) {
    *x = blockIdx.x * blockDim.x + threadIdx.x;
    *y = blockIdx.y * blockDim.y + threadIdx.y;
    *z = blockIdx.z * blockDim.z + threadIdx.z;
}

/*
 * Returns the cube pointer offset caused by the x, y, z coordinates with given pitch and memory shape in y-direction
 */
__device__ inline int getCubeOffset(const int x, const int y, const int z, const size_t pitch, int memory_y_shape) {
    return    z * memory_y_shape * pitch / sizeof(double)
            + y * pitch / sizeof(double)
            + x;
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

template <int normalization>
__device__ inline void FirstModSphBessel_multiply_result_with_normalization_factor(const int min_stored_l,
                                                                                   const int max_stored_l,
                                                                                   double *result) {
    if (normalization != 1) {
        for (int l = min_stored_l; l <= max_stored_l; l++) {
            result[l-min_stored_l] *= sqrt((2.0*(double)l+1.0)/(4.0*M_PI));
        }
    }
}

template <int normalization>
__device__ double GBFMMHelmholtz3D_evaluate_le_point(const double x,
                                                     const double y,
                                                     const double z,
                                                     const double r,
                                                     const double kappa,
                                                     const int lmax,
                                                     const double* __restrict__ local_expansion) {
#define STORED_BESSELS 5
    const double kappa_r = kappa*r;
    double bessel_lmax =    FirstModSphBessel_evaluate_small(kappa_r, lmax, 1.0);
    double bessel_lmax_m1 = FirstModSphBessel_evaluate_small(kappa_r, lmax-1, 1.0);    
    
    int lm_address =0, address2 = 0;
    int l, m, l2; 
    double top = 0.0, bottom = 0.0, new_bottom = 0.0, prev1 = 0.0, prev2 = 0.0, current = 0.0, one_per_r = 1.0 / r;
    
    // set value for l=0, m=0
    double bessel_values[STORED_BESSELS];
    FirstModSphBessel_evaluate_recursion(kappa_r, 1.0, lmax, 0, 1, bessel_lmax, bessel_lmax_m1, bessel_values);
    FirstModSphBessel_multiply_result_with_normalization_factor<normalization>(0, 1, bessel_values);
    
    // set value for l=0, m=0
    double result = 1.0 * bessel_values[0] * local_expansion[lm_address];
    if (r < 1e-12) one_per_r = 0.0;
    
 
    
    
    // set value for l=1, m=-1
    lm_address += 1;
    result += y * one_per_r * bessel_values[1] * local_expansion[lm_address];
    
    // set value for l=1, m=0
    lm_address += 1;
    result += z * one_per_r * bessel_values[1] * local_expansion[lm_address];
    
    // set value for l=1, m=1
    lm_address += 1;
    result += x * one_per_r * bessel_values[1] * local_expansion[lm_address];
   
    // set all values where m=-1
    m = -1;
    prev1 = y * one_per_r;
    // the starting address has 1 item before from the l=0, 3 from l=1, and 1 from l=2
    address2 = 5;
    for (int min_l = 2; min_l <= lmax; min_l += STORED_BESSELS) {
        int max_l = min(lmax, min_l+STORED_BESSELS-1);
        FirstModSphBessel_evaluate_recursion(kappa_r, 1.0, lmax,
                                             min_l, max_l,
                                             bessel_lmax, bessel_lmax_m1, bessel_values);
        FirstModSphBessel_multiply_result_with_normalization_factor<normalization>(min_l, max_l, bessel_values);
        for (l = min_l; l <= max_l; l++) {
            current =   ( 2.0*(double)l-1.0) / sqrt( 1.0*(double)((l+m)*(l-m)) ) * z * prev1 * one_per_r;
            if (l > 2) {
                current -=  sqrt( (double)((l+m-1)*(l-m-1)) /  (double)((l+m)*(l-m)) ) * prev2;
            }
            prev2 = prev1;
            prev1 = current;
            result += current * local_expansion[address2] * bessel_values[l-min_l];
            
            // add the address2 to get to the next item with m=0 
            address2 += (2*l+2);
        }
    }
    
    // set all values where m=0
    prev1 = z * one_per_r;
    prev2 = 1.0;
    m = 0;
    // the starting address has 1 item before from the l=0, 3 from l=1, and 2 from l=2
    address2 = 6;
    for (int min_l = 2; min_l <= lmax; min_l += STORED_BESSELS) {
        int max_l = min(lmax, min_l+STORED_BESSELS-1);
        FirstModSphBessel_evaluate_recursion(kappa_r, 1.0, lmax,
                                             min_l, max_l,
                                             bessel_lmax, bessel_lmax_m1, bessel_values);
        FirstModSphBessel_multiply_result_with_normalization_factor<normalization>(min_l, max_l, bessel_values);
        for (l = min_l; l <= max_l; l++) {
            current =   ( 2.0*(double)l-1.0) / sqrt( 1.0*(double)((l+m)*(l-m)) ) * z * prev1 * one_per_r;
            current -=  sqrt( (double)((l+m-1)*(l-m-1)) /  (double)((l+m)*(l-m)) ) * prev2;
            prev2 = prev1;
            prev1 = current;
            result += current * local_expansion[address2] * bessel_values[l-min_l];;
            
            // add the address2 to get to the next item with m=0 
            address2 += (2*l+2);
        }
    }
    
    // set all values where m=1
    prev1 = x * one_per_r;
    m = 1;
    // the starting address has 1 item before from the l=0, 3 from l=1, and 3 from l=2
    address2 = 7;
    for (int min_l = 2; min_l <= lmax; min_l += STORED_BESSELS) {
        int max_l = min(lmax, min_l+STORED_BESSELS-1);
        FirstModSphBessel_evaluate_recursion(kappa_r, 1.0, lmax,
                                             min_l, max_l,
                                             bessel_lmax, bessel_lmax_m1, bessel_values);
        FirstModSphBessel_multiply_result_with_normalization_factor<normalization>(min_l, max_l, bessel_values);
        for (l = min_l; l <= max_l; l++) {
            current =   ( 2.0*(double)l-1.0) / sqrt( 1.0*(double)((l+m)*(l-m)) ) * z*prev1 * one_per_r;
            if (l > 2) {
                current -=  sqrt( (double)((l+m-1)*(l-m-1)) /  (double)((l+m)*(l-m)) ) * prev2;
            }
            prev2 = prev1;
            prev1 = current;
            result += current * local_expansion[address2] * bessel_values[l-min_l];;
            
            // add the address2 to get to the next item with m=0 
            address2 += (2*l+2);
        }
    }
    
    // go through the rest of the stuff
    bottom = y * one_per_r; // bottom refers to solid harmonics value with l=l-1 and m=-(l-1)
    top = x * one_per_r;    // top    refers to solid harmonics value with l=l-1 and m=l-1
    lm_address += 1;
    for (l=2; l <= lmax; l++) {
        // get the bessel values
        FirstModSphBessel_evaluate_recursion(kappa_r, 1.0, lmax,
                                             l, l,
                                             bessel_lmax, bessel_lmax_m1, bessel_values);
        FirstModSphBessel_multiply_result_with_normalization_factor<normalization>(l, l, bessel_values);
        
        new_bottom = sqrt((2.0*(double)l - 1.0) / (2.0*(double)l)) * 
                       one_per_r * ( y*top + x*bottom);
        result += new_bottom * local_expansion[lm_address] * bessel_values[0];
        
        // set the first address for the loop below
        address2 = lm_address + (2*l+2);
        
        // get value for l=l, m=l. The address is 2*l items away from l=l, m=-l
        lm_address += 2*l;
        top = sqrt((2.0*(double)l - 1.0) / (2.0*(double)l)) * 
                      one_per_r * ( x*top-y*bottom );
        result += top * local_expansion[lm_address] * bessel_values[0];
        
        
        // store the new bottom: l=l, m=-l (we need the old bottom in calculation of top)
        bottom = new_bottom;
        
        // set all values where m=-l
        m = -l;
        prev1 = new_bottom;
        for (int min_l = l+1; min_l <= lmax; min_l += STORED_BESSELS) {
            int max_l = min(lmax, min_l+STORED_BESSELS-1);
            FirstModSphBessel_evaluate_recursion(kappa_r, 1.0, lmax,
                                                min_l, max_l,
                                                bessel_lmax, bessel_lmax_m1, bessel_values);
            FirstModSphBessel_multiply_result_with_normalization_factor<normalization>(min_l, max_l, bessel_values);
            for (l2 = min_l; l2 <= max_l; l2++) {
                current =   ( 2.0*(double)l2-1.0) / sqrt( 1.0*(double)((l2+m)*(l2-m)) ) * z * prev1 * one_per_r;
                if (l2 > l+1) {
                    current -=  sqrt( (double)((l2+m-1)*(l2-m-1)) /  (double)((l2+m)*(l2-m)) ) *  prev2;
                }
                prev2 = prev1;
                prev1 = current;
                result += current * local_expansion[address2] * bessel_values[l2-min_l];
                
                // add the address2 to get to the next item with m=l 
                address2 += (2*l2+2);
            }
        }
        
        
        
        // set all values where m=l
        m = l;
        prev1 = top;
        address2 = lm_address + (2*l+2);
        for (int min_l = l+1; min_l <= lmax; min_l += STORED_BESSELS) {
            int max_l = min(lmax, min_l+STORED_BESSELS-1);
            FirstModSphBessel_evaluate_recursion(kappa_r, 1.0, lmax,
                                                min_l, max_l,
                                                bessel_lmax, bessel_lmax_m1, bessel_values);
            FirstModSphBessel_multiply_result_with_normalization_factor<normalization>(min_l, max_l, bessel_values);
            for (l2 = min_l; l2 <= max_l; l2++) {
                current =   ( 2.0*(double)l2-1.0) / sqrt( 1.0*(double)((l2+m)*(l2-m)) ) * z * prev1 * one_per_r;
                if (l2 > l+1) {
                    current -=  sqrt( (double)((l2+m-1)*(l2-m-1)) /  (double)((l2+m)*(l2-m)) ) * prev2;
                }
                prev2 = prev1;
                prev1 = current;
                result += current * local_expansion[address2] * bessel_values[l2-min_l];
                
                // add the address2 to get to the next item with m=l 
                address2 += (2*l2+2);
            }
        }
        
        // get next address
        lm_address += 1;
    }
    return result;
}

/* 
 * Evaluate Local expansion on a grid
 */
template <int normalization>
__global__ void GBFMMHelmholtz3D_evaluate_le_grid(
                              double* __restrict__ cube,
                              const int lmax,
                              const double kappa,
                              const double* __restrict__ local_expansion,
                              const double* __restrict__ grid_points_x,
                              const double* __restrict__ grid_points_y,
                              const double* __restrict__ grid_points_z,
                              const int shape_x,
                              const int shape_y,
                              const int shape_z,
                              const double zero_point_x,
                              const double zero_point_y,
                              const double zero_point_z, 
                              const int slice_offset,
                              const size_t pitch,
                              const int memory_y_shape,
                              const int slice_count) {    
    
    // The x index will be the fastest to change.
    int x, y, z;
    getXYZ3D(&x, &y, &z);
    
    // get the offset from the input cube pointer
    const int id = getCubeOffset(x, y, z, pitch, memory_y_shape);
    
    double value;
    double relative_position_x, relative_position_y, relative_position_z, r;
                
    //printf("X: %f, cell_spacing: %f, ncell: %d", distance, bubble->cell_spacing, ncell);
    // Check that the point is within the block 
    if (x < shape_x && y < shape_y && z+slice_offset < shape_z && z < slice_count) {
        // calculate relative position to the zero-point and distance to it 
        
            relative_position_x = grid_points_x[x] - zero_point_x;
            relative_position_y = grid_points_y[y] - zero_point_y;
            relative_position_z = grid_points_z[z+slice_offset] - zero_point_z;
            r = sqrt(relative_position_x * relative_position_x + relative_position_y * relative_position_y + relative_position_z * relative_position_z);
        
    } 
    
    // calculate the value for local expansion value multiplied with real solid harmonics in Racah's normalization
    value = GBFMMHelmholtz3D_evaluate_le_point<normalization>
        (relative_position_x,
         relative_position_y,
         relative_position_z,
         r,
         kappa,
         lmax,
         local_expansion);
    
    // if the point resides within the cube, add the value calculated above to the current value
    if (x < shape_x && y < shape_y && z+slice_offset < shape_z && z < slice_count) {
        if (normalization == 2) cube[id] += 8.0 * kappa * value;
    }
    return;

}


/*
 * Evaluate Helmholtz-local expansion at grid 
 */
/*__global__ void GBFMMHelmholtz3D_evaluate_le_grid(double*  __restrict__ spherical_harmonics_cube, 
                                                  const int spherical_harmonics_memory_shape_y, int spherical_harmonics_memory_shape_z,
                                                  const size_t spherical_harmonics_pitch,
                                                  double* __restrict__ bessels_cube, 
                                                  const int bessels_memory_shape_y, int bessels_memory_shape_z,
                                                  const size_t bessels_pitch,
                                                  double* __restrict__ result_cube,
                                                  const int result_cube_shape_x, const int result_cube_shape_y, const int result_cube_shape_z,
                                                  const int result_cube_memory_shape_y, const size_t result_cube_pitch,
                                                  const int slice_count,
                                                  const int spherical_harmonics_slice_offset,
                                                  const int bessels_slice_offset,
                                                  const int output_slice_offset,
                                                  double* __restrict__ local_expansion,
                                                  double kappa,
                                                  const int lmax) {
    // get the x, y, z coordinates
    int x, y, z;
    getXYZ3D(&x, &y, &z);
    // check that we are within range of the cube
    if (z < slice_count &&
        x < result_cube_shape_x && y < result_cube_shape_y && z+output_slice_offset < result_cube_shape_z) {
 
        // get the offsets from the input addresses caused by the x, y, z coordinates
        int cube1_offset = getCubeOffset(x, y, z+spherical_harmonics_slice_offset, spherical_harmonics_pitch, spherical_harmonics_memory_shape_y);
        int cube2_offset = getCubeOffset(x, y, z+bessels_slice_offset, bessels_pitch, bessels_memory_shape_y);
        int result_cube_offset = getCubeOffset(x, y, z+output_slice_offset, result_cube_pitch, result_cube_memory_shape_y);
    
        // get the address offset between two following spherical harmonics l,m values for one point
        int spherical_harmonics_address_offset = spherical_harmonics_pitch / sizeof(double) * spherical_harmonics_memory_shape_y * spherical_harmonics_memory_shape_z;
         // get the address offset between two following bessels l values for one point
        int bessels_address_offset = bessels_pitch / sizeof(double) * bessels_memory_shape_y * bessels_memory_shape_z;
    
        
        double *spherical_harmonics = &spherical_harmonics_cube[cube1_offset];
        double *bessels = &bessels_cube[cube2_offset];
      
        
        // do the evaluation
        double result = 0.0;
        double lesum = 0.0;
        for (int l = 0; l <= lmax; l++) {
            // add to the result value
            for (int m = -l; m <= l; m++) {
                // add to the result value
                result += (*spherical_harmonics) * (*bessels) *  (*local_expansion);
                
                // add the spherical harmonics pointer to get the next l, m value
                spherical_harmonics = &spherical_harmonics[spherical_harmonics_address_offset];
                
                // add the local expansion pointer to get the next l, m value
                lesum += *local_expansion;
                local_expansion++;
            }
            // add the bessels pointer to get the next l value
            bessels = &bessels[bessels_address_offset];
        }
        // and finally add to the result cube the local expansion result
        result_cube[result_cube_offset] +=  8.0 * kappa * result;
    }
}*/

/*************************************************** 
 *       GBFMMHelmholtz3D implementation             *
 *                                                 *
 ***************************************************/

GBFMMHelmholtz3D::GBFMMHelmholtz3D(
               // the grid from which the subgrids are extracted from (should represent the entire input domain) needed 
               // to evaluate coulomb potential for using gbfmm
               Grid3D *grid_in,
               // the grid from which the subgrids are extracted from (should represent the entire output domain) needed 
               // to evaluate coulomb potential for using gbfmm
               Grid3D *grid_out,
               // the maximum angular momentum quantum number 'l' value
               int lmax,
               // the box indices for which the evaluation of multipoles and eventually potential is performed by
               // this node
               int domain[2],
               // the first and last cell index in x-direction for each box in domain 
               int *input_start_indices_x, int *input_end_indices_x,
               // the first and last cell index in y-direction for each box in domain 
               int *input_start_indices_y, int *input_end_indices_y,
               // the first and last cell index in z-direction for each box in domain 
               int *input_start_indices_z, int *input_end_indices_z, 
               // the first and last cell index in x-direction for each box in domain 
               int *output_start_indices_x, int *output_end_indices_x,
               // the first and last cell index in y-direction for each box in domain 
               int *output_start_indices_y, int *output_end_indices_y,
               // the first and last cell index in z-direction for each box in domain 
               int *output_start_indices_z, int *output_end_indices_z, 
               // the main streamcontainer used to extract the boxwise streamcontainers from 
               StreamContainer *streamContainer) {
    
    // init the common things using the function defined in gbfmm_potential_operator.cu
    initGBFMMPotentialOperator(grid_in, grid_out, lmax, domain,
                               input_start_indices_x, input_end_indices_x, 
                               input_start_indices_y, input_end_indices_y,
                               input_start_indices_z, input_end_indices_z,
                               output_start_indices_x, output_end_indices_x, 
                               output_start_indices_y, output_end_indices_y,
                               output_start_indices_z, output_end_indices_z,
                               streamContainer);
    

    StreamContainer **device_containers =  new StreamContainer*[this->streamContainer->getNumberOfDevices()];
    check_helmholtz3d_errors(__FILE__, __LINE__);
    
    this->normalization = 2;
    
}

GBFMMHelmholtz3D::GBFMMHelmholtz3D(
               // the grid from which the subgrids are extracted from (should represent the entire domain) needed 
               // to evaluate coulomb potential for using gbfmm
               GBFMMCoulomb3D *parent_operator,
               // the maximum angular momentum quantum number 'l' value
               int lmax) {
    // init the common things using the function defined in gbfmm_potential_operator.cu
    initGBFMMPotentialOperator(parent_operator, lmax);
    
    this->normalization = 2;
}

void GBFMMHelmholtz3D::initIntegrators() {
    // init the subgrids and the streamcontainers for each domain box
    for (int i = 0; i <= this->domain[1]-this->domain[0]; i++) {
        // initialize the Integrator needed for multipole evaluation with a buffer for (this->lmax+1)*(this->lmax+1) results 
        this->integrators[i] = new GBFMMHelmholtz3DMultipoleEvaluator(this->streamContainers[i],
                                                                    this->input_grids[i],
                                                                    this->lmax, 
                                                                    &this->centers[i*3]);
    }
}

void GBFMMHelmholtz3D::initHarmonics() {
    //this->bessels = new FirstModifiedSphericalCubeBessels *[streamContainer->getNumberOfDevices()];
    //this->harmonics = new RealCubeHarmonics * [this->streamContainer->getNumberOfDevices()];
    
    // initialize the solid-harmonics evaluators.
    // NOTE: this assumes that each of the boxes have the same shape and that the multipole center is at the center of the
    // box. If the cube-grid is changed to be non-equidistant at some point, this must be changed to be box-wise.
    //for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        // initialize the spherical harmonics
    //    this->harmonics[device] = new RealSphericalCubeHarmonics(/*lmin=*/0,
    //                                                             this->lmax, 
    //                                                             /*normalization=conventional*/2,
    //                                                             this->input_grids[0]->getShape(), 
    //                                                             this->device_containers[device]);
        
        // initialize the first modified spherical bessels values
    //    this->bessels[device] = new FirstModifiedSphericalCubeBessels(/*lmin=*/0, this->lmax, this->input_grids[0]->getShape(), this->device_containers[device]);

    //}
    //check_helmholtz3d_errors(__FILE__, __LINE__);
    
    // evaluate the spherical harmonics on all devices
    //for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
    //    this->harmonics[device]->evaluate(this->device_grids[device], this->centers); 
    //}
    //check_helmholtz3d_errors(__FILE__, __LINE__);
}

void GBFMMHelmholtz3D::destroyHarmonics() {
    // destroy the solid harmonics and bessels from all devices
    /*for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        this->harmonics[device]->destroy(); 
        delete this->harmonics[device];
        this->bessels[device]->destroy();
        delete this->bessels[device];
    }
    delete[] this->bessels;
    delete[] this->harmonics;*/
}


/*
 * Evaluates multipole moments between l:0-this->lmax for box 'i'.
 * To use this function, the box 'i' must belong to the domain of this node.
 * 
 * NOTE: this function IS NOT BLOCKING with regards to CUDA
 */
void GBFMMHelmholtz3D::calculateMultipoleMomentsBox(int i, CudaCube *cube) {
    this->integrators[i]->setIntegrationCube(cube);
    this->integrators[i]->integrate();
    this->integrators[i]->setIntegrationCube(NULL);
    /*int *cube_device_memory_shape = cube->getDeviceMemoryShape();
    int *integration_device_memory_shape = this->integrators[i]->getIntegrationCube()->getDeviceMemoryShape();
    int lm_cube_offset = 0, l_cube_offset = 0;
    check_helmholtz3d_errors(__FILE__, __LINE__);
    int * spherical_harmonics_memory_shape = this->harmonics[0]->getShape();
    size_t spherical_harmonics_pitch = this->harmonics[0]->getDevicePitch(0); 
            
    int * bessels_device_memory_shape = this->bessels[0]->getShape();
    size_t bessels_device_pitch = this->bessels[0]->getDevicePitch(0); 
    for (int l = 0; l <= this->lmax ; l++) {
        for (int m = -l; m <= l; m++) {
            
            
            // loop over the devices of the box "i"'s StreamContainer
            for (int device = 0; device < this->streamContainers[i]->getNumberOfDevices(); device++) {
                this->streamContainers[i]->setDevice(device);
                
                // get the device number in the box "i"'s StreamContainer
                int device_number = this->streamContainers[i]->getDeviceNumber(device);
                // get the device order number in global StreamContainer
                int device_order_number = this->streamContainer->getDeviceOrderNumber(device_number);
                
                
                // get the device order number in cube's streamcontainer
                int device_cube_order_number = cube->getStreamContainer()->getDeviceOrderNumber(device_number);
                double *device_cube = cube->getDevicePointer(device_cube_order_number);
                size_t device_cube_pitch = cube->getDevicePitch(device_cube_order_number);
                
                // get the pointer to the temp cube used in integration
                // NOTE: the integrator has the same streamcontainer as the one looped above 
                double *device_temp_cube = this->integrators[i]->getIntegrationCube()->getDevicePointer(device);
                size_t device_temp_pitch = this->integrators[i]->getIntegrationCube()->getDevicePitch(device);
                
                // get pointer to the first item of the spherical harmonics for this device
                // NOTE: the zeros comes from the fact that there is only one device per streamcontainer of the
                // SolidHarmonics
                double *device_spherical_harmonics = this->harmonics[device_order_number]->getDeviceResults(0); 
                
                // get pointer to the first item of the first modified spherical Bessel function values for this device
                // NOTE: the zeros comes from the fact that there is only one device per streamcontainer of the
                // SolidHarmonics
                double *device_bessels = this->bessels[device_order_number]->getDeviceResults(0); 
                
                // get the number of slices handled by this device
                int device_slice_count = cube->getShape(Z_) / this->streamContainers[i]->getNumberOfDevices()
                                            + ((cube->getShape(Z_) % this->streamContainers[i]->getNumberOfDevices()) > device);
                
                for (int stream = 0; stream < this->streamContainers[i]->getStreamsPerDevice(); stream++) {
                    cudaStream_t *streamObject = this->streamContainers[i]->getStream(device, stream);
                        
                    // get the number of slices handled by this stream
                    int slice_count = device_slice_count / this->streamContainers[i]->getStreamsPerDevice() 
                                        + ((device_slice_count % this->streamContainers[i]->getStreamsPerDevice()) > stream);
                                        
                                        
                    // get the launch configuration for the multiplication and integration
                    dim3 block, grid;
                    
                    getCubeLaunchConfiguration(&grid, &block, cube->getShape(), slice_count, 256);
                    // multiply the spherical harmonics with the cube and store to device_temp_cube, i.e.,  "this->integrators[i]->getIntegrationCube()"
                    multiply_3d_cubes(&device_spherical_harmonics[lm_cube_offset], cube->getShape(X_), cube->getShape(Y_), spherical_harmonics_memory_shape[Y_], spherical_harmonics_pitch,
                                    device_cube, cube->getShape(X_), cube->getShape(Y_), cube_device_memory_shape[Y_], device_cube_pitch,
                                    device_temp_cube,  this->integrators[i]->getIntegrationCube()->getShape(X_), this->integrators[i]->getIntegrationCube()->getShape(Y_),
                                    integration_device_memory_shape[Y_], device_temp_pitch,
                                    slice_count, &grid, &block, streamObject);
                    check_helmholtz3d_errors(__FILE__, __LINE__);
                    
                    // multiply the bessel values with the device_temp_cube, i.e.,  "this->integrators[i]->getIntegrationCube()" in place (store to device_temp_cube)
                    multiply_3d_cubes(device_temp_cube,  this->integrators[i]->getIntegrationCube()->getShape(X_), this->integrators[i]->getIntegrationCube()->getShape(Y_),
                                    integration_device_memory_shape[Y_], device_temp_pitch,
                                    &device_bessels[l_cube_offset], cube->getShape(X_), cube->getShape(Y_), bessels_device_memory_shape[Y_], bessels_device_pitch,
                                    slice_count, &grid, &block, streamObject);
                    check_helmholtz3d_errors(__FILE__, __LINE__);
                            
                    
                    // add to the pointers
                    device_spherical_harmonics += slice_count * spherical_harmonics_memory_shape[Y_] * spherical_harmonics_pitch / sizeof(double); 
                    device_cube            += slice_count * cube_device_memory_shape[Y_] * device_cube_pitch / sizeof(double); 
                    device_temp_cube       += slice_count * integration_device_memory_shape[Y_] * device_temp_pitch / sizeof(double);
                    device_bessels         += slice_count * bessels_device_memory_shape[Y_] * bessels_device_pitch / sizeof(double); 
                }
                
            }
        
        
            // add to the offset to the spherical harmonics caused by the l, m-cubes
            lm_cube_offset += spherical_harmonics_memory_shape[Y_] * spherical_harmonics_memory_shape[Z_] * spherical_harmonics_pitch / sizeof(double);
            
            // start the integration process at the GPUs
            // NOTE: this is not blocking
            // NOTE: the results are stored to the buffer of the integrator 
            this->integrators[i]->integrate();
            check_helmholtz3d_errors(__FILE__, __LINE__);
            
        }
        // add to the offset to the bessels caused by the l, m-cubes
        l_cube_offset += bessels_device_memory_shape[Y_] * bessels_device_memory_shape[Z_] * bessels_device_pitch / sizeof(double);
    }*/
    
    
}

/*
 * Downloads multipole moments between l:0-this->lmax for each box belonging
 * to the domain of this node.
 * 
 * NOTE: this functions IS BLOCKING with regards to CUDA
 */
void GBFMMHelmholtz3D::downloadMultipoleMoments(double *host_multipole_moments) {
    // do the evaluation for the boxes
    host_multipole_moments = &host_multipole_moments[(domain[0]-1) * (this->lmax+1)*(this->lmax+1)];
    for (int i = 0; i <= this->domain[1]-this->domain[0]; i++) {
        this->integrators[i]->downloadResult(host_multipole_moments);
        
        // multiply with factor, if the normalization != 1
        if (this->normalization != 1) {
            for (int l = 0; l <= this->lmax; l++) {
                double normalization_factor = sqrt((2.0*(double)l+1.0)/(4.0*M_PI));
                for (int n = 0; n <= 2*l; n++) {
                    host_multipole_moments[l*l + n] *= normalization_factor;
                }
            }
        }
        
        // add the pointer to be prepared for the next box
        host_multipole_moments += (this->lmax+1)*(this->lmax+1);
    }
}

/*
 * Sets the energy and evaluates the First modified bessel function values.
 */
void GBFMMHelmholtz3D::setEnergy(double energy) {
    this->energy = energy;
    
    // evaluate the spherical harmonics on all devices
    //for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
    //    this->bessels[device]->setKappa(sqrt(-2.0 * energy));
    //    this->bessels[device]->evaluate(this->device_grids[device], this->centers); 
    //}
    //check_helmholtz3d_errors(__FILE__, __LINE__);
    double kappa = sqrt(-2.0 * energy);
    // init the subgrids and the streamcontainers for each domain box
    for (int i = 0; i <= this->domain[1]-this->domain[0]; i++) {
        // initialize the Integrator needed for multipole evaluation with a buffer for (this->lmax+1)*(this->lmax+1) results 
        ((GBFMMHelmholtz3DMultipoleEvaluator*)this->integrators[i])->setKappa(kappa);
    }
}

/*
 * Evaluates the potential within space of a single box. 
 */
void GBFMMHelmholtz3D::evaluatePotentialLEBox(
                                            // order number of the box within the domain of 
                                            // this node (the indexing should start from 0)
                                            int i,
                                            // pointer to the HOST memory of the local expansion
                                            // for box
                                            double *local_expansion,
                                            double zero_point[3], 
                                            Grid3D *grid3d,
                                            CudaCube *output_cube,
                                            StreamContainer *streamContainer) {
    double **device_cubes  = output_cube->getDeviceCubes();
    size_t *device_pitches = output_cube->getDevicePitches();
    
    int shape_x = output_cube->getShape(X_);
    int shape_y = output_cube->getShape(Y_);
    int shape_z = output_cube->getShape(Z_);
    double kappa = sqrt(-2.0 * this->energy);
    int *device_memory_shape = output_cube->getDeviceMemoryShape();
    int slice_offset = 0;
    
    
    check_helmholtz3d_errors(__FILE__, __LINE__);

    // check if we are at the borders of the boxes, that are not the 
    // borders of the global grid
    if (this->output_end_indices_x[i] != this->grid_out->axis[X_]->ncell) {
        shape_x -= 1;
    }
    
    if (this->output_end_indices_y[i] != this->grid_out->axis[Y_]->ncell) {
        shape_y -= 1;
    }
    
    if (this->output_end_indices_z[i] != this->grid_out->axis[Z_]->ncell) {
        shape_z -= 1;
    }
    for (int device = 0; device < streamContainer->getNumberOfDevices(); device ++) {
        
        // get global device order number
        int device_number = streamContainer->getDeviceNumber(device);
        int global_device_order_number = this->streamContainer->getDeviceOrderNumber(device_number);
        
        // set the correct device
        streamContainer->setDevice(device);   
        
        // get the device cube pointer & pitch
        double *device_cube = device_cubes[device];
        size_t device_pitch = device_pitches[device];
        
        // set the device expansion pointer
        double *device_expansion = &this->device_expansions[global_device_order_number][i * (this->lmax+1)*(this->lmax+1)];
        
        
        int device_slice_count = shape_z / streamContainer->getNumberOfDevices()
                                    + ((shape_z % streamContainer->getNumberOfDevices()) > device);
        for (int stream = 0; stream < streamContainer->getStreamsPerDevice(); stream ++) {
 
            // upload the expansion
            cudaMemcpyAsync(device_expansion, local_expansion, sizeof(double)*(this->lmax + 1)*(this->lmax +1),
                            cudaMemcpyHostToDevice, *streamContainer->getStream(device, stream));
            //check_coulomb_errors(__FILE__, __LINE__);
            
            int slice_count = device_slice_count / streamContainer->getStreamsPerDevice() 
                                        + ((device_slice_count % streamContainer->getStreamsPerDevice()) > stream);
                                        
            // get the launch configuration
            dim3 block, grid;
            output_cube->getLaunchConfiguration(&grid, &block, slice_count, BLOCK_SIZE);
                                        
            // evaluate the real regular solid harmonics multiplied with the local expansion values
            // for the slices belonging for this stream
            GBFMMHelmholtz3D_evaluate_le_grid<2>
                <<< grid, block, 0, *streamContainer->getStream(device, stream) >>>
                             (device_cube,
                              this->lmax,
                              kappa,
                              device_expansion,
                              grid3d->axis[X_]->device_gridpoints[device],
                              grid3d->axis[Y_]->device_gridpoints[device],
                              grid3d->axis[Z_]->device_gridpoints[device],
                              shape_x,
                              shape_y,
                              shape_z,
                              zero_point[X_],
                              zero_point[Y_],
                              zero_point[Z_], 
                              slice_offset,
                              device_pitch,
                              device_memory_shape[Y_],
                              slice_count);
            //check_coulomb_errors(__FILE__, __LINE__);
            
            // add the counter with the number of slices handled so far
            slice_offset += slice_count;
            device_cube = &device_cube[slice_count * device_memory_shape[Y_] * device_pitch /  sizeof(double)];
        }
    }
    /*
    for (int device = 0; device < streamContainer->getNumberOfDevices(); device ++) {
        
        // get global device order number
        int device_number = streamContainer->getDeviceNumber(device);
        int global_device_order_number = this->streamContainer->getDeviceOrderNumber(device_number);
        
        // set the correct device
        streamContainer->setDevice(device);   
        
        // get the device cube pointer, device memory shap & pitch
        double *device_cube      = device_cubes[device];
        int *device_memory_shape = output_cube->getDeviceMemoryShape();
        size_t device_pitch      = device_pitches[device];
        
        
        // set the device expansion pointer
        double *device_expansion = &this->device_expansions[global_device_order_number][i * (this->lmax+1)*(this->lmax+1)];
        

        // get pointer to the first item of the spherical harmonics for this device
        // NOTE: the zeros comes from the fact that there is only one device per streamcontainer of the
        // SolidHarmonics
        double *device_spherical_harmonics      = this->harmonics[global_device_order_number]->getDeviceResults(0); 
        int *device_spherical_harmonics_shape   = this->harmonics[global_device_order_number]->getShape();
        size_t device_spherical_harmonics_pitch = this->harmonics[global_device_order_number]->getDevicePitch(0);
        int spherical_harmonics_slice_offset = 0;
        
        // get pointer to the first item of the first modified besself functions for this device
        // NOTE: the zeros comes from the fact that there is only one device per streamcontainer of the
        // SolidHarmonics
        double *device_bessels             = this->bessels[global_device_order_number]->getDeviceResults(0); 
        int *bessels_device_memory_shape   = this->harmonics[global_device_order_number]->getShape();
        size_t bessels_device_pitch        = this->harmonics[global_device_order_number]->getDevicePitch(0);
        int bessels_slice_offset = 0;
        
        
        int device_slice_count = shape_z / streamContainer->getNumberOfDevices()
                                    + ((shape_z % streamContainer->getNumberOfDevices()) > device);
                              
        for (int stream = 0; stream < streamContainer->getStreamsPerDevice(); stream ++) {
 
            // upload the expansion
            cudaMemcpyAsync(device_expansion, local_expansion, sizeof(double)*(this->lmax + 1)*(this->lmax +1),
                            cudaMemcpyHostToDevice, *streamContainer->getStream(device, stream));

            check_helmholtz3d_errors(__FILE__, __LINE__);
            
            int slice_count = device_slice_count / streamContainer->getStreamsPerDevice() 
                                        + ((device_slice_count % streamContainer->getStreamsPerDevice()) > stream);
                                        
            // get the launch configuration
            dim3 block, grid;
            output_cube->getLaunchConfiguration(&grid, &block, slice_count, BLOCK_SIZE);
                                      
            // evaluate the real spherical harmonics multiplied with first modified bessel functions multiplied with the local expansion values
            // for the slices belonging for this stream
            GBFMMHelmholtz3D_evaluate_le_grid<<< grid, block, 0, 
                        *streamContainer->getStream(device, stream) >>>
                                             (device_spherical_harmonics, 
                                              device_spherical_harmonics_shape[Y_], device_spherical_harmonics_shape[Z_],
                                              device_spherical_harmonics_pitch,
                                              device_bessels, 
                                              bessels_device_memory_shape[Y_], bessels_device_memory_shape[Z_],
                                              bessels_device_pitch,
                                              device_cube,
                                              shape_x, shape_y, shape_z,
                                              device_memory_shape[Y_], device_pitch,
                                              slice_count,
                                              spherical_harmonics_slice_offset,
                                              bessels_slice_offset,
                                              output_slice_offset,
                                              device_expansion,
                                              kappa,
                                              this->lmax);
            
            check_helmholtz3d_errors(__FILE__, __LINE__);
            
            // add the counter with the number of slices handled so far
            output_slice_offset += slice_count;
            spherical_harmonics_slice_offset += slice_count;
            bessels_slice_offset += slice_count;

        }
    }*/
}

/*************************************************** 
 *              Fortran interfaces                 *
 *                                                 *
 ***************************************************/

extern "C" GBFMMHelmholtz3D *gbfmmhelmholtz3d_init_cuda(
               // the grid from which the subgrids are extracted from (should represent the entire input domain) needed 
               // to evaluate coulomb potential for using gbfmm
               Grid3D *grid_in, 
               // the grid from which the subgrids are extracted from (should represent the entire output domain) needed 
               // to evaluate coulomb potential for using gbfmm
               Grid3D *grid_out, 
               // the maximum angular momentum quantum number 'l' value
               int lmax,
               // the box indices for which the evaluation of multipoles and eventually potential is performed by
               // this node
               int domain[2],
               // the first and last cell index in x-direction for each box in domain 
               int *input_start_indices_x, int *input_end_indices_x,
               // the first and last cell index in y-direction for each box in domain 
               int *input_start_indices_y, int *input_end_indices_y,
               // the first and last cell index in z-direction for each box in domain 
               int *input_start_indices_z, int *input_end_indices_z, 
               // the first and last cell index in x-direction for each box in domain 
               int *output_start_indices_x, int *output_end_indices_x,
               // the first and last cell index in y-direction for each box in domain 
               int *output_start_indices_y, int *output_end_indices_y,
               // the first and last cell index in z-direction for each box in domain 
               int *output_start_indices_z, int *output_end_indices_z, 
               // the main streamcontainer used to extract the boxwise streamcontainers from 
               StreamContainer *streamContainer) {
    GBFMMHelmholtz3D *new_gbfmmhelmholtz3d = new GBFMMHelmholtz3D(grid_in, grid_out, lmax, domain,
                                                                  input_start_indices_x, input_end_indices_x, 
                                                                  input_start_indices_y, input_end_indices_y,
                                                                  input_start_indices_z, input_end_indices_z,
                                                                  output_start_indices_x, output_end_indices_x, 
                                                                  output_start_indices_y, output_end_indices_y,
                                                                  output_start_indices_z, output_end_indices_z,
                                                                  streamContainer);
    return new_gbfmmhelmholtz3d;
}

extern "C" GBFMMHelmholtz3D *gbfmmhelmholtz3d_init_child_operator_cuda(
               // the operator used as a parent for the inited helmholtz-operator
               GBFMMCoulomb3D *gbfmm_coulomb3d, 
               // the maximum angular momentum quantum number 'l' value
               int lmax) {
    GBFMMHelmholtz3D *new_gbfmmhelmholtz3d = new GBFMMHelmholtz3D(gbfmm_coulomb3d, lmax);
    return new_gbfmmhelmholtz3d;
}

extern "C" void gbfmmhelmholtz3d_init_harmonics_cuda(
               // a pointer to the pre-inited gbfmm coulomb3d operator
               GBFMMHelmholtz3D *gbfmmhelmholtz3d) {
    gbfmmhelmholtz3d->initHarmonics();
}

extern "C" void gbfmmhelmholtz3d_destroy_harmonics_cuda(
               // a pointer to the pre-inited gbfmm coulomb3d operator
               GBFMMHelmholtz3D *gbfmmhelmholtz3d) {
    gbfmmhelmholtz3d->destroyHarmonics();
}

extern "C" void gbfmmhelmholtz3d_calculate_multipole_moments_cuda(
               // a pointer to the pre-inited gbfmm coulomb3d operator
               GBFMMHelmholtz3D *gbfmmhelmholtz3d,
               // a pointer to the cube for which the multipole moments are evaluated
               // the boxes needed for multipole are uploaded for 
               CudaCube *input_cube) {
    gbfmmhelmholtz3d->calculateMultipoleMoments(input_cube);
}

extern "C" void gbfmmhelmholtz3d_download_multipole_moments_cuda(
               // a pointer to the pre-inited gbfmm helmholtz3d operator, with which
               // the multipole moments are calculated
               GBFMMHelmholtz3D *gbfmmhelmholtz3d,
               // a pointer to the 2-dimensional array residing in host memory in which the multipole moments are stored
               double *host_multipole_moments) {
    gbfmmhelmholtz3d->downloadMultipoleMoments(host_multipole_moments);
}

extern "C" void gbfmmhelmholtz3d_upload_domain_boxes_cuda(
               // a pointer to the pre-inited gbfmm coulomb3d operator, with which
               // the multipole moments are calculated
               GBFMMHelmholtz3D *gbfmmhelmholtz3d,
               // a pointer to the cube for which the multipole moments are evaluated
               CudaCube *input_cube) { 
    gbfmmhelmholtz3d->uploadDomainBoxes(input_cube);
}

extern "C" void gbfmmhelmholtz3d_set_energy_cuda(
               // a pointer to the pre-inited gbfmm helmholtz3d operator, with which
               // the multipole moments are calculated
               GBFMMHelmholtz3D *gbfmmhelmholtz3d,
               // a pointer to the cube for which the multipole moments are evaluated
               double energy) { 
    gbfmmhelmholtz3d->setEnergy(energy);
}

extern "C" void gbfmmhelmholtz3d_evaluate_potential_le_cuda(
               GBFMMHelmholtz3D *gbfmmhelmholtz3d, 
               double *local_expansion,
               CudaCube *output_cube) {
    gbfmmhelmholtz3d->evaluatePotentialLE(local_expansion, output_cube);
}

extern "C" StreamContainer* gbfmmhelmholtz3d_get_box_stream_container_cuda(
               GBFMMHelmholtz3D *gbfmmhelmholtz3d, 
               int ibox) {
    return gbfmmhelmholtz3d->getBoxStreamContainer(ibox);
}

extern "C" void gbfmmhelmholtz3d_destroy_cuda(
               // the destroyed gbfmm coulomb3d operator
               GBFMMHelmholtz3D *gbfmmhelmholtz3d) {
    gbfmmhelmholtz3d->destroy();
}
