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
#include "streamcontainer.h"
#include "memory_leak_operators.h"
#include "evaluators.h"
#include "cube.h"
#include "bubbles_cuda.h"
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

#define X_ 0
#define Y_ 1
#define Z_ 2

#define NLIP 7
#define BLOCK_SIZE 256

#define FULL_MASK 0xffffffff

extern __shared__ double shared_memory[];

/*
 * NOTE: this method assumes that the grid is equidistant (in the sense that all cells have equal length)
 */
template <int nlip>
__device__ inline
void calculate_icell_equidistant(const Grid1D *grid,
                                 const double coordinate,
                                 int &icell,
                                 double &in_cell_coordinate,
                                 double &one_per_grid_step
                                ) {
    double grid_step = grid->h[0];
    one_per_grid_step = 1.0 / grid_step;
    double start = grid->d[0];
    double cell_length = (nlip-1) * grid_step;
    icell = (int)((coordinate - start) / (cell_length));
    if (coordinate - start < 0.0) icell = -1;
    double cell_center =  start + ((double)icell + 0.5) * cell_length;
    in_cell_coordinate = (coordinate - cell_center) * one_per_grid_step;
}

/*
 * Read Lagrange interpolation polynomials into the shared memory.
 */
template <int nlip, int result_length>
__device__ inline
void read_lip(double *device_lip, int thread_id, double *shared_memory_lip) {
    __syncthreads();
    if (thread_id < nlip * result_length) {
        shared_memory_lip[thread_id] = device_lip[thread_id];
    }
    __syncthreads();
}

/*
 *
 * NOTE: polynomials must be an array of size 8
 */
template <int nlip, int result_length>
__device__ inline
void evaluate_polynomials(double *lip, double x, double *polynomials) {


    for (int i = 0; i < result_length; i++) {
        // init the polynomial as the first value of the lip
        polynomials[i] = lip[i*nlip];

        for (int k = 1; k < nlip; k++) {
            polynomials[i] = lip[i*nlip + k] + x*polynomials[i];
        }
    }
    if (result_length < 1) polynomials[0] = 0.0;
    if (result_length < 2) polynomials[1] = 0.0;
    if (result_length < 3) polynomials[2] = 0.0;
    if (result_length < 4) polynomials[3] = 0.0;
    if (result_length < 5) polynomials[4] = 0.0;
    if (result_length < 6) polynomials[5] = 0.0;
    if (result_length < 7) polynomials[6] = 0.0;
    if (result_length < 8) polynomials[7] = 0.0;
}


/*
 * Evaluates sum of 'coefficients[i]' times 'polynomials[i]' for 'coefficients'
 * that reside in device memory.
 *
 * NOTE: polynomials must be an array of size 8
 */
template <int nlip>
 __device__ inline
double evaluate_coefficients(double *polynomials, const double* __restrict__ c, int address, int thread_id) {

    const int EVALUATE_BLOCK_SIZE = 8;

    // get the thread rank within its little block of size 'EVALUATE_BLOCK_SIZE'
    int thread_rank =  thread_id%EVALUATE_BLOCK_SIZE;

    // let us use the results array as temp array
    double temp_results[EVALUATE_BLOCK_SIZE];

    // TODO: make this more generic
    int addresses[EVALUATE_BLOCK_SIZE];
#if (__CUDA_ARCH__ >= 350) && (__CUDA_ARCH__ < 700)
    addresses[0] =__shfl(address, 0, EVALUATE_BLOCK_SIZE)  + thread_rank;
    addresses[1] =__shfl(address, 1, EVALUATE_BLOCK_SIZE)  + thread_rank;
    addresses[2] =__shfl(address, 2, EVALUATE_BLOCK_SIZE)  + thread_rank;
    addresses[3] =__shfl(address, 3, EVALUATE_BLOCK_SIZE)  + thread_rank;
    addresses[4] =__shfl(address, 4, EVALUATE_BLOCK_SIZE)  + thread_rank;
    addresses[5] =__shfl(address, 5, EVALUATE_BLOCK_SIZE)  + thread_rank;
    addresses[6] =__shfl(address, 6, EVALUATE_BLOCK_SIZE)  + thread_rank;
    addresses[7] =__shfl(address, 7, EVALUATE_BLOCK_SIZE)  + thread_rank;
#elif (__CUDA_ARCH__ >= 700)
    addresses[0] =__shfl_sync(FULL_MASK, address, 0, EVALUATE_BLOCK_SIZE)  + thread_rank;
    addresses[1] =__shfl_sync(FULL_MASK, address, 1, EVALUATE_BLOCK_SIZE)  + thread_rank;
    addresses[2] =__shfl_sync(FULL_MASK, address, 2, EVALUATE_BLOCK_SIZE)  + thread_rank;
    addresses[3] =__shfl_sync(FULL_MASK, address, 3, EVALUATE_BLOCK_SIZE)  + thread_rank;
    addresses[4] =__shfl_sync(FULL_MASK, address, 4, EVALUATE_BLOCK_SIZE)  + thread_rank;
    addresses[5] =__shfl_sync(FULL_MASK, address, 5, EVALUATE_BLOCK_SIZE)  + thread_rank;
    addresses[6] =__shfl_sync(FULL_MASK, address, 6, EVALUATE_BLOCK_SIZE)  + thread_rank;
    addresses[7] =__shfl_sync(FULL_MASK, address, 7, EVALUATE_BLOCK_SIZE)  + thread_rank;
#endif

    int reg = thread_rank;
    if (thread_rank < nlip) {
        temp_results[reg] = __ldg(&c[addresses[0]]);
        reg = ((reg == 0) ? EVALUATE_BLOCK_SIZE-1 : reg - 1);
        temp_results[reg] = __ldg(&c[addresses[1]]);
        reg = ((reg == 0) ? EVALUATE_BLOCK_SIZE-1 : reg - 1);
        temp_results[reg] = __ldg(&c[addresses[2]]);
        reg = ((reg == 0) ? EVALUATE_BLOCK_SIZE-1 : reg - 1);
        temp_results[reg] = __ldg(&c[addresses[3]]);
        reg = ((reg == 0) ? EVALUATE_BLOCK_SIZE-1 : reg - 1);
        temp_results[reg] = __ldg(&c[addresses[4]]);
        reg = ((reg == 0) ? EVALUATE_BLOCK_SIZE-1 : reg - 1);
        temp_results[reg] = __ldg(&c[addresses[5]]);
        reg = ((reg == 0) ? EVALUATE_BLOCK_SIZE-1 : reg - 1);
        temp_results[reg] = __ldg(&c[addresses[6]]);
        reg = ((reg == 0) ? EVALUATE_BLOCK_SIZE-1 : reg - 1);
        temp_results[reg] = __ldg(&c[addresses[7]]);
    }
    else {
        temp_results[0] = 0.0;
        temp_results[1] = 0.0;
        temp_results[2] = 0.0;
        temp_results[3] = 0.0;
        temp_results[4] = 0.0;
        temp_results[5] = 0.0;
        temp_results[6] = 0.0;
        temp_results[7] = 0.0;
    }

    reg = thread_rank;
    double result =  temp_results[0]                                      * polynomials[reg];
#if (__CUDA_ARCH__ >= 350) && (__CUDA_ARCH__ < 700)
    reg = ((reg == EVALUATE_BLOCK_SIZE-1) ? 0 : reg + 1);
    result += __shfl(temp_results[1], thread_rank+1, EVALUATE_BLOCK_SIZE) * polynomials[reg];
    reg = ((reg == EVALUATE_BLOCK_SIZE-1) ? 0 : reg + 1);
    result += __shfl(temp_results[2], thread_rank+2, EVALUATE_BLOCK_SIZE) * polynomials[reg];
    reg = ((reg == EVALUATE_BLOCK_SIZE-1) ? 0 : reg + 1);
    result += __shfl(temp_results[3], thread_rank+3, EVALUATE_BLOCK_SIZE) * polynomials[reg];
    reg = ((reg == EVALUATE_BLOCK_SIZE-1) ? 0 : reg + 1);
    result += __shfl(temp_results[4], thread_rank+4, EVALUATE_BLOCK_SIZE) * polynomials[reg];
    reg = ((reg == EVALUATE_BLOCK_SIZE-1) ? 0 : reg + 1);
    result += __shfl(temp_results[5], thread_rank+5, EVALUATE_BLOCK_SIZE) * polynomials[reg];
    reg = ((reg == EVALUATE_BLOCK_SIZE-1) ? 0 : reg + 1);
    result += __shfl(temp_results[6], thread_rank+6, EVALUATE_BLOCK_SIZE) * polynomials[reg];
    reg = ((reg == EVALUATE_BLOCK_SIZE-1) ? 0 : reg + 1);
    result += __shfl(temp_results[7], thread_rank+7, EVALUATE_BLOCK_SIZE) * polynomials[reg];
#elif (__CUDA_ARCH__ >= 700)
    reg = ((reg == EVALUATE_BLOCK_SIZE-1) ? 0 : reg + 1);
    result += __shfl_sync(FULL_MASK, temp_results[1], thread_rank+1, EVALUATE_BLOCK_SIZE) * polynomials[reg];
    reg = ((reg == EVALUATE_BLOCK_SIZE-1) ? 0 : reg + 1);
    result += __shfl_sync(FULL_MASK, temp_results[2], thread_rank+2, EVALUATE_BLOCK_SIZE) * polynomials[reg];
    reg = ((reg == EVALUATE_BLOCK_SIZE-1) ? 0 : reg + 1);
    result += __shfl_sync(FULL_MASK, temp_results[3], thread_rank+3, EVALUATE_BLOCK_SIZE) * polynomials[reg];
    reg = ((reg == EVALUATE_BLOCK_SIZE-1) ? 0 : reg + 1);
    result += __shfl_sync(FULL_MASK, temp_results[4], thread_rank+4, EVALUATE_BLOCK_SIZE) * polynomials[reg];
    reg = ((reg == EVALUATE_BLOCK_SIZE-1) ? 0 : reg + 1);
    result += __shfl_sync(FULL_MASK, temp_results[5], thread_rank+5, EVALUATE_BLOCK_SIZE) * polynomials[reg];
    reg = ((reg == EVALUATE_BLOCK_SIZE-1) ? 0 : reg + 1);
    result += __shfl_sync(FULL_MASK, temp_results[6], thread_rank+6, EVALUATE_BLOCK_SIZE) * polynomials[reg];
    reg = ((reg == EVALUATE_BLOCK_SIZE-1) ? 0 : reg + 1);
    result += __shfl_sync(FULL_MASK, temp_results[7], thread_rank+7, EVALUATE_BLOCK_SIZE) * polynomials[reg];
#endif

    return result;
}

/*
 * Evaluates sum of 'coefficients[i]' times 'polynomials[i]' for 'coefficients'
 * that reside in registers.
 *
 * NOTE: 'polynomials' and 'c' must be arrays of size nlip
 */
template <int nlip>
 __device__ inline
double evaluate_coefficients_register(double *polynomials, double *c) {
    double result = 0.0;
    for (int ilip = 0; ilip < nlip; ilip ++) {
        result += polynomials[ilip] * c[ilip];
    }
    return result;
}

/*
 * Evaluates sum of 'coefficients[i]' times 'polynomials[i]' for 'coefficients'
 * that reside in registers that are spread within the neighbouring threads.
 * Also the 'polynomials should lie in registers'
 *
 * NOTE: 'polynomials' and 'c' must be arrays of size nlip
 */
template <int nlip>
 __device__ inline
double evaluate_coefficients_shuffle(double *polynomials, double coefficient, int thread_order_number, int x_modulo) {
    double result = 0.0;
    if (nlip == 7) {
        // get the number of thread having the first coefficient
        int first_of_cell = thread_order_number - x_modulo;

        // do not take the 32:nd thread in to the games because each warp is handling
        // 5 cells, i.e., 6*5 + 1 points
        if (thread_order_number < 31) {
            result =  __shfl(coefficient, first_of_cell  , 32) * polynomials[0];
            result += __shfl(coefficient, first_of_cell+1, 32) * polynomials[1];
            result += __shfl(coefficient, first_of_cell+2, 32) * polynomials[2];
            result += __shfl(coefficient, first_of_cell+3, 32) * polynomials[3];
            result += __shfl(coefficient, first_of_cell+4, 32) * polynomials[4];
            result += __shfl(coefficient, first_of_cell+5, 32) * polynomials[5];
            result += __shfl(coefficient, first_of_cell+6, 32) * polynomials[6];
        }
    }
    return result;
}

/*
 * Evaluate cube at 'points'
 *
 * if calling to the version with evaluate_gradients=true, we are also evaluating the
 * gradients and storing the results to 'device_gradients'
 */
template <bool evaluate_value, bool evaluate_gradients_x, bool evaluate_gradients_y, bool evaluate_gradients_z>
__global__ void
#if (__CUDA_ARCH__ <= 350)
__launch_bounds__(BLOCK_SIZE)
#else
__launch_bounds__(BLOCK_SIZE)
#endif
CubeEvaluator_evaluate_points(const double* __restrict__ device_cube,
                              const size_t device_pitch,
                              const size_t device_shape_y,
                              const Grid3D* __restrict__ grid,
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
                              // number of points in this kernel call
                              const int point_count,
                              // device_point_offset
                              const int device_point_offset,
                              const double multiplier
                       ) {

    // Get the point order number within this kernel call
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    double value, gradient[3];
    double in_cell_coordinate_x = 0.0, in_cell_coordinate_y = 0.0, in_cell_coordinate_z = 0.0;
    double one_per_grid_step_x = 0.0, one_per_grid_step_y = 0.0, one_per_grid_step_z = 0.0;
    int icell_x = 0, icell_y = 0, icell_z = 0, ncell_x = 0, ncell_y= 0, ncell_z= 0;
    bool valid_point = true;


    // get the number of cells
    ncell_x = grid->axis[X_]->ncell;
    ncell_y = grid->axis[Y_]->ncell;
    ncell_z = grid->axis[Z_]->ncell;

    if (id + device_point_offset < device_number_of_points && id < point_count ) {
        // get the cell indices and coordinates within cell in grid steps
        calculate_icell_equidistant<NLIP>(
            grid->axis[X_], points[id + device_point_offset],                             icell_x, in_cell_coordinate_x, one_per_grid_step_x);
        calculate_icell_equidistant<NLIP>(
            grid->axis[Y_], points[id + device_point_offset + device_number_of_points],   icell_y, in_cell_coordinate_y, one_per_grid_step_y);
        calculate_icell_equidistant<NLIP>(
            grid->axis[Z_], points[id + device_point_offset + device_number_of_points*2], icell_z, in_cell_coordinate_z, one_per_grid_step_z);
    }
    else {
        valid_point = false;
    }


    // if the result is not within the grid, set the icells to 0 and mark the point to be non-valid
    if (icell_x < 0 || icell_x >= ncell_x || icell_y < 0 || icell_y >= ncell_y || icell_z < 0 || icell_z >= ncell_z) {
        icell_x = 0;
        icell_y = 0;
        icell_z = 0;
        valid_point = false;
    }

    // read the LIPs in the shared memory
    __shared__ double lip[NLIP * NLIP];
    read_lip<NLIP, NLIP>(grid->axis[X_]->lip, threadIdx.x, lip);

    // evaluate the polynomials in x and y directions
    double x_polynomials[8];
    evaluate_polynomials<NLIP, NLIP>(lip, in_cell_coordinate_x, x_polynomials);
    double polynomials[8];
    evaluate_polynomials<NLIP, NLIP>(lip, in_cell_coordinate_y, polynomials);

    double x_values[NLIP], y_values[NLIP];

    // get the address to the first grid point of icell_x, icell_y and icell_z
    int address =   icell_x * (NLIP-1)
                  + icell_y * device_pitch / sizeof(double) * (NLIP-1)
                  + icell_z * device_pitch / sizeof(double) * device_shape_y * (NLIP-1);

    if (evaluate_value || evaluate_gradients_z) {

        for (int j = 0; j < NLIP; j++) {
            // add the address by 'j' slices
            int z_address = address + j * device_pitch / sizeof(double) * device_shape_y;
            for (int k = 0; k < NLIP; k++) {
                // add the address by 'k' rows
                int y_address = z_address + k * device_pitch / sizeof(double);
                x_values[k] = evaluate_coefficients<NLIP>(x_polynomials, device_cube, y_address, threadIdx.x);
            }
            y_values[j] = evaluate_coefficients_register<NLIP>(polynomials, x_values);
        }
    }

    if (evaluate_value) {

        // evaluate the polynomials in z-direction.
        // NOTE: reusing the y-direction polynomial registers
        evaluate_polynomials<NLIP, NLIP>(lip, in_cell_coordinate_z, polynomials);

        // evaluate the coefficients
        value = evaluate_coefficients_register<NLIP>(polynomials, y_values);

        // if the point handled is valid, let's add it to the results
        if (valid_point) {
            result_array[id+device_point_offset] += multiplier * value;
        }
    }

    // if we are evaluating the gradients, it is done within the brackets below
    if (evaluate_gradients_x || evaluate_gradients_y || evaluate_gradients_z) {
        __shared__ double derivative_lip[(NLIP-1) * NLIP];
        read_lip<NLIP-1, NLIP>(grid->axis[X_]->derivative_lip, threadIdx.x, derivative_lip);

        if (evaluate_gradients_z) {
            // evaluate the gradient polynomials in z-direction.
            evaluate_polynomials<NLIP-1, NLIP>(derivative_lip, in_cell_coordinate_z, polynomials);

            // multiply the polynomials with 1 / grid_step
            for (int j = 0; j < NLIP; j++) {
                polynomials[j] *= one_per_grid_step_z;
            }

            // evaluate the derivative coefficients
            // we can reuse the previous y_values, which are the same for this case
            gradient[Z_] = evaluate_coefficients_register<NLIP>(polynomials, y_values);
        }

        // NOTE: we now have the derivatives in x-direction, but for the rest y- and z- directions,
        //       we have to recalculate everything else, as we need to save some registers.
        //       If we would have 49*2 extra registers the next loop would be futile.

        if (evaluate_gradients_y) {
            // let's calculate the y-axis derivative polynomials.
            // Note that we are still using the same x-direction polynomials
            evaluate_polynomials<NLIP-1, NLIP>(derivative_lip, in_cell_coordinate_y, polynomials);

            // multiply the polynomials with 1 / grid_step
            for (int j = 0; j < NLIP; j++) {
                polynomials[j] *= one_per_grid_step_y;
            }

            // and let's do the looping again
            for (int j = 0; j < NLIP; j++) {
                // add the address by 'j' slices
                int z_address = address + j * device_pitch / sizeof(double) * device_shape_y;
                for (int k = 0; k < NLIP; k++) {
                    // add the address by 'k' rows
                    int y_address = z_address + k * device_pitch / sizeof(double);
                    x_values[k] = evaluate_coefficients<NLIP>(x_polynomials, device_cube, y_address, threadIdx.x);
                }
                y_values[j] = evaluate_coefficients_register<NLIP>(polynomials, x_values);
            }

            // evaluate the polynomials in z-direction.
            // reusing the y-direction polynomial registers
            evaluate_polynomials<NLIP, NLIP>(lip, in_cell_coordinate_z, polynomials);

            // finally, we can get the derivative in y-direction
            gradient[Y_] = evaluate_coefficients_register<NLIP>(polynomials, y_values);

        }

        if (evaluate_gradients_x) {
            // evaluate the normal polynomials in y-direction
            evaluate_polynomials<NLIP, NLIP>(lip, in_cell_coordinate_y, polynomials);

            // and evaluate the derivative polynomials in x-direction
            evaluate_polynomials<NLIP-1, NLIP>(derivative_lip, in_cell_coordinate_x, x_polynomials);

            // multiply the polynomials with 1 / grid_step
            for (int j = 0; j < NLIP; j++) {
                x_polynomials[j] *= one_per_grid_step_x;
            }

            // and let's do the looping again
            for (int j = 0; j < NLIP; j++) {
                // add the address by 'j' slices
                int z_address = address + j * device_pitch / sizeof(double) * device_shape_y;
                for (int k = 0; k < NLIP; k++) {
                    // add the address by 'k' rows
                    int y_address = z_address + k * device_pitch / sizeof(double);
                    x_values[k] = evaluate_coefficients<NLIP>(x_polynomials, device_cube, y_address, threadIdx.x);
                }
                y_values[j] = evaluate_coefficients_register<NLIP>(polynomials, x_values);
            }

            // evaluate the polynomials in z-direction.
            // reusing the y-direction polynomial registers
            evaluate_polynomials<NLIP, NLIP>(lip, in_cell_coordinate_z, polynomials);

            // finally, we are ready and can get the derivative in z-direction
            gradient[X_] = evaluate_coefficients_register<NLIP>(polynomials, y_values);
        }

        // if the point handled is valid, let's store the gradient to the device_gradients
        if (valid_point) {
            if (evaluate_gradients_x) device_gradients_x[id+device_point_offset] += multiplier * gradient[X_];
            if (evaluate_gradients_y) device_gradients_y[id+device_point_offset] += multiplier * gradient[Y_];
            if (evaluate_gradients_z) device_gradients_z[id+device_point_offset] += multiplier * gradient[Z_];
        }
    }

    return;

}

/*
 * Evaluate cube gradients at grid points.  The results are stored to 'device_gradients'.
 */
template <int nlip, bool evaluate_gradients_x, bool evaluate_gradients_y, bool evaluate_gradients_z>
__global__ void
#if (__CUDA_ARCH__ <= 350)
__launch_bounds__(BLOCK_SIZE)
#else
__launch_bounds__(BLOCK_SIZE)
#endif
CubeEvaluator_evaluate_grid_gradients(const double* __restrict__ device_cube,
                                      const size_t device_pitch,
                                      const size_t device_shape_y,
                                      const Grid3D* __restrict__ grid,
                                      double* __restrict__ device_gradients_x,
                                      double* __restrict__ device_gradients_y,
                                      double* __restrict__ device_gradients_z,
                                      // number of slices handled by this device
                                      // in previous calls
                                      int device_slice_offset,
                                      // number of slices handled by all devices
                                      // in previous calls
                                      int slice_offset,
                                      // number of slices handled by this call
                                      int slice_count,
                                      // number of slices handled by this
                                      // number of warps in a x-axis row
                                      int warps_per_string,
                                      const double multiplier
                                      ) {

    // Get the point order number within this kernel call
    int global_warp_id, thread_order_number, cells_per_warp;
    int WARP_SIZE = 32;

    bool valid_point = true;

    // if nlip is 7, each warp of 32 handles 5 cells
    if (nlip == 7) {
        // get the global warp order number
        global_warp_id =   blockIdx.x * blockDim.x / WARP_SIZE
                             + threadIdx. x / WARP_SIZE;
        cells_per_warp = 5;

        // get the order number of thread within the warp
        thread_order_number = threadIdx.x % WARP_SIZE;

        if (thread_order_number == 31) valid_point = false;
    }


    // get the number of cells
    int ncell_x = grid->axis[X_]->ncell;
    int ncell_y = grid->axis[Y_]->ncell;
    int ncell_z = grid->axis[Z_]->ncell;
    int y_shape = ncell_y * (nlip-1) + 1;

    // get the z and y coordinates
    int z = global_warp_id / (warps_per_string * y_shape);
    int y = (global_warp_id - z * warps_per_string * y_shape) / warps_per_string;

    // get the warp id withing the x-axis string
    int string_warp_id = (global_warp_id
                         - z * warps_per_string * y_shape
                         - y * warps_per_string);
    int icell_x = string_warp_id * cells_per_warp + thread_order_number / (nlip-1);
    int x_modulo = thread_order_number % (nlip-1);
    int x = icell_x * (nlip-1) + x_modulo;


    // get the order numbers of cells within this device
    int icell_z =  (z + slice_offset) / (nlip-1);
    int icell_y =  y / (nlip-1);

    // and get the remainders of the y and z coordinates
    int y_modulo = y % (nlip-1);
    int z_modulo = (z + slice_offset) % (nlip-1);

    // if this thread handles the last cell of the x-axis
    // set the correct icell
    if (x_modulo == 0 && icell_x > 0) {
        icell_x -= 1;
        x_modulo = 6;
    }

    // if this thread handles data in the last index of the y-axis
    if (y_modulo == 0 && icell_y > 0) {
        icell_y -= 1;
        y_modulo = 6;
    }

    // if this thread handles data in the last index of the z-axis
    if (z_modulo == 0 && icell_z > 0) {
        icell_z -= 1;
        z_modulo = 6;
    }




    // if the result is not within the grid, mark the point to be non-valid
    if (   icell_x  < 0 || x >= ncell_x * (nlip-1) + 1
        || icell_y  < 0 || y >= y_shape
        || icell_z  < 0 || z + slice_offset >= ncell_z * (nlip-1) + 1
        || z >= slice_count) {
        valid_point = false;
        icell_x = 0;
        icell_y = 0;
        icell_z = 0;
        x = 0;
        y = 0;
        z = 0;
        thread_order_number = 32;
    }

    if (thread_order_number == 0 && x_modulo != 0) valid_point = false;

    // calculate the 1 / grid steps for all axis
    double one_per_grid_step_x = 1.0 / grid->axis[X_]->h[icell_x];
    double one_per_grid_step_y = 1.0 / grid->axis[Y_]->h[icell_y];
    double one_per_grid_step_z = 1.0 / grid->axis[Z_]->h[icell_z];

    // get the in cell coordinate of x
    double in_cell_coordinate_x = (double)(x_modulo - 3);
    double in_cell_coordinate_y = (double)(y_modulo - 3);
    double in_cell_coordinate_z = (double)(z_modulo - 3);


    // read the LIPs in the shared memory
    __shared__ double lip[nlip * nlip];
    read_lip<nlip, nlip>(grid->axis[X_]->lip, threadIdx.x, lip);
    __shared__ double derivative_lip[(nlip-1) * nlip];
    read_lip<nlip-1, nlip>(grid->axis[X_]->derivative_lip, threadIdx.x, derivative_lip);

    // init the polynomials in x direction
    double x_polynomials[8];

    // init the polynomials in y/z direction
    double polynomials[8];

    double x_values[nlip], y_values[nlip];
    int address;
    double gradient[3];

    // evaluate gradient to x direction
    if (evaluate_gradients_z) {
        address =   x
                + y * device_pitch / sizeof(double)
                + icell_z  * (nlip-1) * device_pitch / sizeof(double) * device_shape_y;
        for (int j = 0; j < nlip; j++) {
            // add the address by 'j' slices
            int z_address = address + j * device_pitch / sizeof(double) * device_shape_y;

            // read the value
            y_values[j] = device_cube[z_address];
        }

        // evaluate the polynomials in z-direction.
        evaluate_polynomials<nlip-1, nlip>(derivative_lip, in_cell_coordinate_z, polynomials);

        // multiply the polynomials with 1 / grid_step
        for (int j = 0; j < nlip; j++) {
            polynomials[j] *= one_per_grid_step_z;
        }

        // Now we have all to evaluate the gradients in z direction. Let's do it.
        gradient[Z_] = evaluate_coefficients_register<nlip>(polynomials, y_values);
    }

    // evaluate gradient to y direction
    if (evaluate_gradients_y) {
        // evaluate the derivative polynomials in y direction
        evaluate_polynomials<nlip-1, nlip>(derivative_lip, in_cell_coordinate_y, polynomials);

        // multiply the polynomials with 1 / grid_step
        for (int j = 0; j < nlip; j++) {
            polynomials[j] *= one_per_grid_step_y;
        }

        // get the address to the first grid point of icell_y and icell_z and to the point x
        address =   x
                + icell_y * device_pitch / sizeof(double) * (nlip-1)
                + icell_z * device_pitch / sizeof(double) * device_shape_y * (nlip-1);
        for (int j = 0; j < nlip; j++) {
            // add the address by 'j' slices
            int z_address = address + j * device_pitch / sizeof(double) * device_shape_y;

            for (int k = 0; k < nlip; k++) {
                // add the address by 'k' rows
                int y_address = z_address + k * device_pitch / sizeof(double);

                // read in the x value
                x_values[k] = device_cube[y_address];

            }
            y_values[j] = evaluate_coefficients_register<nlip>(polynomials, x_values);
        }

        // evaluate the polynomials in z-direction.
        // NOTE: reusing the y-direction polynomial registers
        evaluate_polynomials<nlip, nlip>(lip, in_cell_coordinate_z, polynomials);

        // Now we have all to evaluate the gradients in y direction. Let's do it.
        gradient[Y_] = evaluate_coefficients_register<nlip>(polynomials, y_values);
    }

    // evaluate gradient to z direction
    if (evaluate_gradients_x) {
        // evaluate the polynomials in y-direction.
        evaluate_polynomials<nlip, nlip>(lip, in_cell_coordinate_y, polynomials);

        // evaluate the derivative polynomials in x direction
        evaluate_polynomials<nlip-1, nlip>(derivative_lip, in_cell_coordinate_x, x_polynomials);

        // multiply the polynomials with 1 / grid_step
        for (int j = 0; j < nlip; j++) {
            x_polynomials[j] *= one_per_grid_step_x;
        }

        address = x
                  + icell_y * device_pitch / sizeof(double) * (nlip-1)
                  + icell_z * device_pitch / sizeof(double) * device_shape_y * (nlip-1);
        for (int j = 0; j < nlip; j++) {
            // add the address by 'j' slices
            int z_address = address + j * device_pitch / sizeof(double) * device_shape_y;
            for (int k = 0; k < nlip; k++) {
                // add the address by 'k' rows
                int y_address = z_address + k * device_pitch / sizeof(double);

                // read in the x value
                double x_value = device_cube[y_address];

                // evaluate the derivative value
                x_values[k] = evaluate_coefficients_shuffle<nlip>(x_polynomials, x_value, thread_order_number, x_modulo);
            }
            y_values[j] = evaluate_coefficients_register<nlip>(polynomials, x_values);
        }

        // evaluate the polynomials in z-direction.
        // NOTE: reusing the y-direction polynomial registers
        evaluate_polynomials<nlip, nlip>(lip, in_cell_coordinate_z, polynomials);

        // evaluate the coefficients
        gradient[X_] = evaluate_coefficients_register<nlip>(polynomials, y_values);
    }

    address = x
              + y * device_pitch / sizeof(double)
              + (z + device_slice_offset) * device_pitch / sizeof(double) * device_shape_y;

    // if the point handled is valid, let's add it to the results
    if (valid_point) {
        /*if (x >= ncell_x * (nlip-1) +1 || y >= ncell_y * (nlip-1) + 1 || z + slice_offset >= ncell_z * (nlip-1) + 1 ||  z >= slice_count) {
            printf("over bounds x: %d/%d, y: %d/%d, z: %d / %d\n", x, ncell_x * (nlip-1) +1, y, ncell_y * (nlip-1) + 1, z, slice_count);
        }*/
        //int max_address = (device_slice_offset + slice_count) * device_shape_y *  device_pitch / sizeof(double);
        //if (address >= max_address || address < 0 ) printf("address over bounds: %d / %d", address, max_address);
        if (evaluate_gradients_x) device_gradients_x[address] += multiplier * gradient[X_];
        if (evaluate_gradients_y) device_gradients_y[address] += multiplier * gradient[Y_];
        if (evaluate_gradients_z) device_gradients_z[address] += multiplier * gradient[Z_];
    }


    return;

}


// this is a bit ugly ... ideally fin_diff_order and grid_type should both be
// tempate parameters, but they should also be user-input ...


template<int finite_diff_order>
__device__ __forceinline__
double evaluate_derivative(const int curr_id,
                           const int prev_id1, const int prev_id2, const int prev_id3, const int prev_id4, const int prev_id5, const int prev_id6, const int prev_id7, const int prev_id8, 
                           const int next_id1, const int next_id2, const int next_id3, const int next_id4, const int next_id5, const int next_id6, const int next_id7, const int next_id8,
                           const double* __restrict__ device_cube, const int grid_type, const int local_pos, const double h){
    if (curr_id == -1) return 0.0;

    // printf("xy: %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i\n",  prev_id6, prev_id5, prev_id4, prev_id3, prev_id2, prev_id1, next_id1, next_id2, next_id3, next_id4, next_id5, next_id6);

    double curr_value,
           prev_value1, prev_value2, prev_value3, prev_value4, prev_value5, prev_value6, prev_value7, prev_value8, 
           next_value1, next_value2, next_value3, next_value4, next_value5, next_value6, next_value7, next_value8;
    if (curr_id > -1)      curr_value  = __ldg(&device_cube[curr_id]);
    if (prev_id1 > -1)     prev_value1 = __ldg(&device_cube[prev_id1]);
    if (next_id1 > -1)     next_value1 = __ldg(&device_cube[next_id1]);
    if (finite_diff_order >= 3){
        if (prev_id2 > -1) prev_value2 = __ldg(&device_cube[prev_id2]);
        if (next_id2 > -1) next_value2 = __ldg(&device_cube[next_id2]);
    }
    if (finite_diff_order >= 4){
        if (prev_id3 > -1) prev_value3 = __ldg(&device_cube[prev_id3]);
        if (next_id3 > -1) next_value3 = __ldg(&device_cube[next_id3]);
    }
    if (finite_diff_order >= 5){
        if (prev_id4 > -1) prev_value4 = __ldg(&device_cube[prev_id4]);
        if (next_id4 > -1) next_value4 = __ldg(&device_cube[next_id4]);
    }
    if (finite_diff_order >= 6){
        if (prev_id5 > -1) prev_value5 = __ldg(&device_cube[prev_id5]);
        if (next_id5 > -1) next_value5 = __ldg(&device_cube[next_id5]);
    }
    if (finite_diff_order >= 7){
        if (prev_id6 > -1) prev_value6 = __ldg(&device_cube[prev_id6]);
        if (next_id6 > -1) next_value6 = __ldg(&device_cube[next_id6]);
    }
    if (finite_diff_order >= 8){
        if (prev_id7 > -1) prev_value7 = __ldg(&device_cube[prev_id7]);
        if (next_id7 > -1) next_value7 = __ldg(&device_cube[next_id7]);
    }
    if (finite_diff_order >= 9){
        if (prev_id8 > -1) prev_value8 = __ldg(&device_cube[prev_id8]);
        if (next_id8 > -1) next_value8 = __ldg(&device_cube[next_id8]);
    }

    if(grid_type == 1){ // equidistant
        if ( finite_diff_order >= 11 && prev_id1 > -1 && next_id1 > -1 && prev_id2 > -1 && next_id2 > -1 && prev_id3 > -1 && next_id3 > -1 && prev_id4 > -1 && next_id4 > -1 && prev_id5 > -1 && next_id5 > -1 ) { // x x x x x o x x x x x
            return (-1.0 * prev_value5 + 12.5 * prev_value4 - 75.0 * prev_value3 + 300.0 * prev_value2 - 1050.0 * prev_value1 + 1050.0 * next_value1 - 300.0 * next_value2 + 75.0 * next_value3 - 12.5 * next_value4 + 1.0 * next_value5) / (1260.0*h);
        }
        else if ( finite_diff_order >= 9 && prev_id1 > -1 && next_id1 > -1 && prev_id2 > -1 && next_id2 > -1 && prev_id3 > -1 && next_id3 > -1 && prev_id4 > -1 && next_id4 > -1 ) { // x x x x o x x x x
            return (3.0 * prev_value4 - 32.0 * prev_value3 + 168.0 * prev_value2 - 672.0 * prev_value1 + 672.0 * next_value1 - 168.0 * next_value2 + 32.0 * next_value3 - 3.0 * next_value4 ) / (840.0*h);
        }
        else if ( finite_diff_order >= 7 && prev_id1 > -1 && next_id1 > -1 && prev_id2 > -1 && next_id2 > -1 && prev_id3 > -1 && next_id3 > -1 ) { // x x x o x x x
            return (-1.0 * prev_value3 + 9.0 * prev_value2 - 45.0 * prev_value1 + 45.0 * next_value1 - 9.0 * next_value2 + 1.0 * next_value3) / (60.0*h);
        }
        else if ( finite_diff_order >= 5 && prev_id1 > -1 && next_id1 > -1 && prev_id2 > -1 && next_id2 > -1 ) { // x x o x x
            return (-1.0 * next_value2 + 8.0 * next_value1 - 8.0 * prev_value1 + prev_value2) / (12.0*h);
        }
        else if ( finite_diff_order >= 3 && prev_id1 > -1 && next_id1 > -1 ) { // x o x
            return (next_value1 - prev_value1) / (2.0 * h);
        }
        else if ( finite_diff_order >= 3 && next_id1 > -1 && next_id2 > -1 ) { // o x x
            return (-1.0 * next_value2 + 4.0 * next_value1 - 3.0 * curr_value) / (2.0 * h);
        }
        else if ( finite_diff_order >= 3 && prev_id1 > -1 && prev_id2 > -1) { // x x o
            return (1.0 * prev_value2 - 4.0 * prev_value2 + 3.0 * curr_value) / (2.0 * h);
        }
        else if ( finite_diff_order >= 2 && next_id1 > -1 ) { // o x
            return (next_value1 - curr_value) / h;
        }
        else if ( finite_diff_order >= 2 && prev_id1 > -1 ) { // x o
            return (curr_value - prev_value1) / h;
        }
    }
    else if(grid_type == 2){ // lobatto

        if (finite_diff_order == 9){

            if ( prev_id4 > -1 && prev_id3 > -1 && prev_id2 > -1 && prev_id1 > -1 && next_id1 > -1 && next_id2 > -1 && next_id3 > -1 && next_id4 > -1){

                // centre at pos 0: {0.00019967699748295, -0.0036233997312734, 0.057221995168223, -1.1410916482317,  0,                 1.1410916482317,  -0.057221995168223, 0.0036233997312734, -0.00019967699748295}
                // centre at pos 1: {0.00089464933543953, -0.030021565652814,  0.64613210399319,  -2.7472765818353,  1.9243274545541,   0.22575990586671, -0.022628075171264, 0.0031999844358336, -0.000387875525842}
                // centre at pos 2: {0.014381546028833,   -0.46942683846151,   1.7209311289771,   -2.310743566852,   0.78934433020611,  0.31410347917631, -0.080765809769078, 0.032393740023427,  -0.010218009329259}
                // centre at pos 3: {0.075131403109164,   -0.38693923899442,   0.6095587888754,   -0.79702587611059, 0,                 0.79702587611059, -0.6095587888754,   0.38693923899442,   -0.075131403109164}
                // centre at pos 4: {0.010218009329259,   -0.032393740023427,  0.080765809769078, -0.31410347917631, -0.78934433020611, 2.310743566852,   -1.7209311289771,   0.46942683846151,   -0.014381546028833}
                // centre at pos 5: {0.000387875525842,   -0.0031999844358336, 0.022628075171264, -0.22575990586671, -1.9243274545541,  2.7472765818353,  -0.64613210399319,  0.030021565652814,  -0.00089464933543953}
                // centre at pos 6: {0.00019967699748295, -0.0036233997312734, 0.057221995168223, -1.1410916482317,  0,                 1.1410916482317,  -0.057221995168223, 0.0036233997312734, -0.00019967699748295}
                switch(local_pos){
                    case(0):
                        return ( 0.00019967699748295 * prev_value4 + -0.0036233997312734 * prev_value3 + 0.057221995168223 * prev_value2 + -1.1410916482317  * prev_value1 + 0                 * curr_value + 1.1410916482317  * next_value1 + -0.057221995168223 * next_value2 + 0.0036233997312734 * next_value3  + -0.00019967699748295 * next_value4 )/h;
                    case(1):
                        return ( 0.00089464933543953 * prev_value4 + -0.030021565652814  * prev_value3 + 0.64613210399319  * prev_value2 + -2.7472765818353  * prev_value1 + 1.9243274545541   * curr_value + 0.22575990586671 * next_value1 + -0.022628075171264 * next_value2 + 0.0031999844358336 * next_value3  + -0.000387875525842   * next_value4 )/h;
                    case(2):
                        return ( 0.014381546028833   * prev_value4 + -0.46942683846151   * prev_value3 + 1.7209311289771   * prev_value2 + -2.310743566852   * prev_value1 + 0.78934433020611  * curr_value + 0.31410347917631 * next_value1 + -0.080765809769078 * next_value2 + 0.032393740023427  * next_value3  + -0.010218009329259   * next_value4 )/h;
                    case(3):
                        return ( 0.075131403109164   * prev_value4 + -0.38693923899442   * prev_value3 + 0.6095587888754   * prev_value2 + -0.79702587611059 * prev_value1 + 0                 * curr_value + 0.79702587611059 * next_value1 + -0.6095587888754   * next_value2 + 0.38693923899442   * next_value3  + -0.075131403109164   * next_value4 )/h;
                    case(4):
                        return ( 0.010218009329259   * prev_value4 + -0.032393740023427  * prev_value3 + 0.080765809769078 * prev_value2 + -0.31410347917631 * prev_value1 + -0.78934433020611 * curr_value + 2.310743566852   * next_value1 + -1.7209311289771   * next_value2 + 0.46942683846151   * next_value3  + -0.014381546028833   * next_value4 )/h;
                    case(5):
                        return ( 0.000387875525842   * prev_value4 + -0.0031999844358336 * prev_value3 + 0.022628075171264 * prev_value2 + -0.22575990586671 * prev_value1 + -1.9243274545541  * curr_value + 2.7472765818353  * next_value1 + -0.64613210399319  * next_value2 + 0.030021565652814  * next_value3  + -0.00089464933543953 * next_value4 )/h;
                }

            }

            // centre at pos 0: {-3.7853180645098,   5.504949536976,    -3.1667044864103,  3.2707014929785,   -5.0400751724465, 10.133071777992,  -10.15049044192,  3.3679166521506,  -0.13405129481064}
            // centre at pos 1: {-0.70024662338511,  -0.30782735432573, 1.6600314728931,   -1.4050588677272,  2.0324950315796,  -3.9835379214689, 3.9560488107718,  -1.3031520813358,  0.051247532998119}
            // centre at pos 2: {0.12436941987383,   -0.5125363975973,  -0.37008926442197, 1.3824872733712,   -1.5646036607053,  2.8292082985006, -2.7390058952044, 0.88379134357785,  -0.033621117394515}
            // centre at pos 3: {-0.033971645333468, 0.11472881590315,  -0.36562067241713, -0.50265593560168, 1.6092337716972,  -2.276606371715,  2.0689731675221,  -0.63666554571692, 0.022584415661807}
            else if (prev_id1 == -1)
                return ( -3.7853180645098   * curr_value  + 5.504949536976    * next_value1 + -3.1667044864103  * next_value2 + 3.2707014929785   * next_value3 + -5.0400751724465 * next_value4 + 10.133071777992  * next_value5 + -10.15049044192  * next_value6 + 3.3679166521506   * next_value7 + -0.13405129481064   * next_value8 )/h;
            else if (prev_id2 == -1)
                return ( -0.70024662338511  * prev_value1 + -0.30782735432573 * curr_value  + 1.6600314728931   * next_value1 + -1.4050588677272  * next_value2 + 2.0324950315796  * next_value3 + -3.9835379214689 * next_value4 + 3.9560488107718  * next_value5 + -1.3031520813358  * next_value6 +  0.051247532998119  * next_value7 )/h;
            else if (prev_id3 == -1)
                return ( 0.12436941987383   * prev_value2 + -0.5125363975973  * prev_value1 + -0.37008926442197 * curr_value  + 1.3824872733712   * next_value1 + -1.5646036607053 * next_value2 +  2.8292082985006 * next_value3 + -2.7390058952044 * next_value4 + 0.88379134357785  * next_value5 +  -0.033621117394515 * next_value6 )/h;
            else if (prev_id4 == -1)
                return ( -0.033971645333468 * prev_value3 + 0.11472881590315  * prev_value2 + -0.36562067241713 * prev_value1 + -0.50265593560168 * curr_value  + 1.6092337716972  * next_value1 + -2.276606371715  * next_value2 + 2.0689731675221  * next_value3 + -0.63666554571692 * next_value4 +  0.022584415661807  * next_value5 )/h;
            // centre at pos 6: {0.13405129481064,   -3.3679166521506,  10.15049044192,   -10.133071777992, 5.0400751724464,  -3.2707014929785, 3.1667044864103,  -5.504949536976,   3.7853180645098}
            // centre at pos 5: {-0.051247532998119, 1.3031520813358,   -3.9560488107718, 3.9835379214689,  -2.0324950315796, 1.4050588677272,  -1.6600314728931, 0.30782735432572,  0.70024662338511}
            // centre at pos 4: {0.033621117394515,  -0.88379134357785, 2.7390058952044,  -2.8292082985006, 1.5646036607053,  -1.3824872733712, 0.37008926442197, 0.5125363975973,   -0.12436941987383}
            // centre at pos 3: {-0.022584415661807, 0.63666554571692,  -2.0689731675221, 2.276606371715,   -1.6092337716972, 0.50265593560168, 0.36562067241713, -0.11472881590315, 0.033971645333468}
            else if (next_id1 == -1) 
                return ( 0.13405129481064   * prev_value8 + -3.3679166521506  * prev_value7 + 10.15049044192   * prev_value6 + -10.133071777992 * prev_value5 + 5.0400751724464  * prev_value4 + -3.2707014929785 * prev_value3 + 3.1667044864103  * prev_value2 + -5.504949536976   * prev_value1 + 3.7853180645098   * curr_value  )/h;
            else if (next_id2 == -1)
                return ( -0.051247532998119 * prev_value7 + 1.3031520813358   * prev_value6 + -3.9560488107718 * prev_value5 + 3.9835379214689  * prev_value4 + -2.0324950315796 * prev_value3 + 1.4050588677272  * prev_value2 + -1.6600314728931 * prev_value1 + 0.30782735432572  * curr_value  + 0.70024662338511  * next_value1 )/h;
            else if (next_id3 == -1)
                return ( 0.033621117394515  * prev_value6 + -0.88379134357785 * prev_value5 + 2.7390058952044  * prev_value4 + -2.8292082985006 * prev_value3 + 1.5646036607053  * prev_value2 + -1.3824872733712 * prev_value1 + 0.37008926442197 * curr_value  + 0.5125363975973   * next_value1 + -0.12436941987383 * next_value2 )/h;
            else if (next_id4 == -1)
                return ( -0.022584415661807 * prev_value5 + 0.63666554571692  * prev_value4 + -2.0689731675221 * prev_value3 + 2.276606371715   * prev_value2 + -1.6092337716972 * prev_value1 + 0.50265593560168 * curr_value  + 0.36562067241713 * next_value1 + -0.11472881590315 * next_value2 + 0.033971645333468 * next_value3 )/h;

        }
        else if (finite_diff_order == 7){

            if ( prev_id3 > -1 && prev_id2 > -1 && prev_id1 > -1 && next_id1 > -1 && next_id2 > -1 && next_id3 > -1 ){
                // some of the following 7 rules are slightly subobtimal in the
                // sense that an asymmetric choice of points would have a smaller
                // error, but the problem is small and this is easier
                // generated using https://github.com/lnw/finite-diff-weights
                // assuming a the lobatto grid with points {-3,-2.4906716888357,-1.40654638041214,0,1.40654638041214,2.4906716888357,3}
                // centre at pos 0: {-0.0019439691154442, 0.049739522152831, -1.1258469276013,  0,                  1.1258469276013,  -0.049739522152831, 0.0019439691154442}
                // centre at pos 1: {-0.017112093697958,  0.55235536265242,  -2.588681589899,   1.8401216993421,    0.23119070988819, -0.019343936928851, 0.0014698486430997}
                // centre at pos 2: {-0.23589340925937,   1.1716189825849,   -1.899781686814,   0.70249557758477,   0.30822337312933, -0.054985788958057, 0.0083229517325248}
                // centre at pos 3: {-0.10416666666667,   0.30251482375627,  -0.66898974686292, 0,                  0.66898974686292, -0.30251482375627,  0.10416666666667}
                // centre at pos 4: {-0.0083229517325248, 0.054985788958057, -0.30822337312933, -0.70249557758477,  1.899781686814,   -1.1716189825849,   0.23589340925937}
                // centre at pos 5: {-0.0014698486430997, 0.019343936928851, -0.23119070988819, -1.8401216993421,   2.588681589899,   -0.55235536265242,  0.017112093697958}
                // centre at pos 6: {-0.0019439691154442, 0.049739522152831, -1.1258469276013,  0,                  1.1258469276013,  -0.049739522152831, 0.0019439691154442}
                switch(local_pos){
                    case(0):
                        return ( -0.00194396911544 * prev_value3 + 0.0497395221528 * prev_value2 + -1.1258469276   * prev_value1 + 0               * curr_value + 1.1258469276   * next_value1 + -0.0497395221528 * next_value2 + 0.00194396911544 * next_value3 )/h;
                    case(1):
                        return ( -0.017112093698   * prev_value3 + 0.552355362652  * prev_value2 + -2.5886815899   * prev_value1 + 1.84012169934   * curr_value + 0.231190709888 * next_value1 + -0.0193439369289 * next_value2 + 0.0014698486431  * next_value3 )/h;
                    case(2):
                        return ( -0.235893409259   * prev_value3 + 1.17161898258   * prev_value2 + -1.89978168681  * prev_value1 + 0.702495577585  * curr_value + 0.308223373129 * next_value1 + -0.0549857889581 * next_value2 + 0.00832295173252 * next_value3 )/h;
                    case(3):
                        return ( -0.104166666667   * prev_value3 + 0.302514823756  * prev_value2 + -0.668989746863 * prev_value1 + 0               * curr_value + 0.668989746863 * next_value1 + -0.302514823756  * next_value2 + 0.104166666667   * next_value3 )/h;
                    case(4):
                        return ( -0.00832295173252 * prev_value3 + 0.0549857889581 * prev_value2 + -0.308223373129 * prev_value1 + -0.702495577585 * curr_value + 1.89978168681  * next_value1 + -1.17161898258   * next_value2 + 0.235893409259   * next_value3 )/h;
                    case(5):
                        return ( -0.0014698486431  * prev_value3 + 0.0193439369289 * prev_value2 + -0.231190709888 * prev_value1 + -1.84012169934  * curr_value + 2.5886815899   * next_value1 + -0.552355362652  * next_value2 + 0.017112093698   * next_value3 )/h;
                }
            }

            // generated using https://github.com/lnw/finite-diff-weights
            // assuming a the lobatto grid with points {-3,-2.4906716888357,-1.40654638041214,0,1.40654638041214,2.4906716888357,3}
            // centre at pos 0: {-3.5,              4.7338588676399,   -1.8896617418485, 1.0666666666667,   -0.68332160435892, 0.43912447856748, -0.16666666666667 }
            // centre at pos 1: {-0.81430867141476, 0,                 1.1519427380981,  -0.53286889603279, 0.32044659909624,  -0.2007490598786, 0.075537290131815 }
            // centre at pos 2: {0.20841888850511,  -0.73860142772332, 0,                0.75556602902867,  -0.35548063466879, 0.20546361183919, -0.075366466980858}
            else if (prev_id1 == -1)
                return ( -3.5              * curr_value  + 4.7338588676399   * next_value1 + -1.8896617418485 * next_value2 + 1.0666666666667   * next_value3 + -0.68332160435892 * next_value4 + 0.43912447856748 * next_value5 + -0.16666666666667  * next_value6)/h;
            else if (prev_id2 == -1)
                return ( -0.81430867141476 * prev_value1 + 0                 * curr_value  + 1.1519427380981  * next_value1 + -0.53286889603279 * next_value2 + 0.32044659909624  * next_value3 + -0.2007490598786 * next_value4 + 0.075537290131815  * next_value5)/h;
            else if (prev_id3 == -1)
                return ( 0.20841888850511  * prev_value2 + -0.73860142772332 * prev_value1 + 0                * curr_value  + 0.75556602902867  * next_value1 + -0.35548063466879 * next_value2 + 0.20546361183919 * next_value3 + -0.075366466980858 * next_value4)/h;
            // generated using https://github.com/lnw/finite-diff-weights
            // assuming a the lobatto grid with points {-3,-2.4906716888357,-1.40654638041214,0,1.40654638041214,2.4906716888357,3}
            // centre at pos 6: {0.16666666666667,   -0.43912447856748, 0.68332160435892,  -1.0666666666667,  1.8896617418485,  -4.7338588676399, 3.5              }
            // centre at pos 5: {-0.075537290131815, 0.2007490598786,   -0.32044659909624, 0.53286889603279,  -1.1519427380981, 0,                0.81430867141476 }
            // centre at pos 4: {0.075366466980858,  -0.20546361183919, 0.35548063466879,  -0.75556602902867, 0,                0.73860142772332, -0.20841888850511}
            else if (next_id1 == -1)
                return ( 0.16666666666667   * prev_value6 + -0.43912447856748 * prev_value5 + 0.68332160435892  * prev_value4 + -1.0666666666667  * prev_value3 + 1.8896617418485  * prev_value2 + -4.7338588676399 * prev_value1 + 3.5               * curr_value )/h;
            else if (next_id2 == -1)
                return ( -0.075537290131815 * prev_value5 + 0.2007490598786   * prev_value4 + -0.32044659909624 * prev_value3 + 0.53286889603279  * prev_value2 + -1.1519427380981 * prev_value1 + 0                * curr_value  + 0.81430867141476  * next_value1)/h;
            else if (next_id3 == -1)
                return ( 0.075366466980858  * prev_value4 + -0.20546361183919 * prev_value3 + 0.35548063466879  * prev_value2 + -0.75556602902867 * prev_value1 + 0                * curr_value  + 0.73860142772332 * next_value1 + -0.20841888850511 * next_value2)/h;

        }
        else if (finite_diff_order == 5){
            if ( prev_id2 > -1 && prev_id1 > -1 && next_id1 > -1 && next_id2 > -1 ){

                // generated using https://github.com/lnw/finite-diff-weights
                // assuming a the lobatto grid with points {-3,-2.4906716888357,-1.40654638041214,0,1.40654638041214,2.4906716888357,3}
                // centre at pos 0: {0.035706928371056, -1.0933955997541,  0,                 1.0933955997541,  -0.035706928371056}
                // centre at pos 1: {0.35921124012101,  -2.2180302446983,  1.6211545002871,   0.2529151880879,  -0.015250683797656}
                // centre at pos 2: {0.3998165330722,   -1.1763296553934,  0.48352837852975,  0.32874345055196, -0.035758706760461}
                // centre at pos 3: {0.093999911479851, -0.52193296182726, 0,                 0.52193296182726, -0.093999911479851}
                // centre at pos 4: {0.035758706760461, -0.32874345055196, -0.48352837852975, 1.1763296553934,  -0.3998165330722}
                // centre at pos 5: {0.015250683797656, -0.2529151880879,  -1.6211545002871,  2.2180302446983,  -0.35921124012101}
                switch(local_pos){
                    case(0):
                        return ( 0.035706928371056 * prev_value2 + -1.0933955997541  * prev_value1 + 0                 * curr_value + 1.0933955997541  * next_value1 + -0.035706928371056 * next_value2 )/h;
                    case(1):
                        return ( 0.35921124012101  * prev_value2 + -2.2180302446983  * prev_value1 + 1.6211545002871   * curr_value + 0.2529151880879  * next_value1 + -0.015250683797656 * next_value2 )/h;
                    case(2):
                        return ( 0.3998165330722   * prev_value2 + -1.1763296553934  * prev_value1 + 0.48352837852975  * curr_value + 0.32874345055196 * next_value1 + -0.035758706760461 * next_value2 )/h;
                    case(3):
                        return ( 0.093999911479851 * prev_value2 + -0.52193296182726 * prev_value1 + 0                 * curr_value + 0.52193296182726 * next_value1 + -0.093999911479851 * next_value2 )/h;
                    case(4):
                        return ( 0.035758706760461 * prev_value2 + -0.32874345055196 * prev_value1 + -0.48352837852975 * curr_value + 1.1763296553934  * next_value1 + -0.3998165330722   * next_value2 )/h;
                    case(5):
                        return ( 0.015250683797656 * prev_value2 + -0.2529151880879  * prev_value1 + -1.6211545002871  * curr_value + 2.2180302446983  * next_value1 + -0.35921124012101  * next_value2 )/h;
                }

            }

                // generated using https://github.com/lnw/finite-diff-weights
                // assuming a the lobatto grid with points {-3,-2.4906716888357,-1.40654638041214,0,1.40654638041214,2.4906716888357,3}
                // centre at pos 0: {-3.1512062536842,  3.9301627535249,  -0.98505481216702, 0.24193000589467,  -0.03583169356836}
                // centre at pos 1: {-0.98083020142501, 0.38287613952776, 0.72328921328621,  -0.14557478380549, 0.020239632416528}
            else if (prev_id1 == -1)
                return ( -3.1512062536842  * curr_value  + 3.9301627535249  * next_value1 + -0.98505481216702 * next_value2 + 0.24193000589467  * next_value3 + -0.03583169356836 * next_value4 )/h;
            else if (prev_id2 == -1)
                return ( -0.98083020142501 * prev_value1 + 0.38287613952776 * curr_value  + 0.72328921328621  * next_value1 + -0.14557478380549 * next_value2 + 0.020239632416528 * next_value3 )/h;
                // generated using https://github.com/lnw/finite-diff-weights
                // assuming a the lobatto grid with points {-3,-2.4906716888357,-1.40654638041214,0,1.40654638041214,2.4906716888357,3}
                // centre at pos 6: {0.03583169356836,   -0.24193000589467, 0.98505481216702,  -3.9301627535249,  3.1512062536842}
                // centre at pos 5: {-0.020239632416528, 0.14557478380549,  -0.72328921328621, -0.38287613952776, 0.98083020142501}
            else if (next_id1 == -1)
                return ( 0.03583169356836   * prev_value4 + -0.24193000589467 * prev_value3 + 0.98505481216702  * prev_value2 + -3.9301627535249  * prev_value1 + 3.1512062536842  * curr_value  )/h;
            else if (next_id2 == -1)
                return ( -0.020239632416528 * prev_value3 + 0.14557478380549  * prev_value2 + -0.72328921328621 * prev_value1 + -0.38287613952776 * curr_value  + 0.98083020142501 * next_value1 )/h;

        }
    }

    return 0.0;
}

/*
 * Evaluate cube gradients at grid points for simple equidistant grid.  The results are stored to 'device_gradients'.
 *
 */

template <int finite_diff_order, bool evaluate_gradients_x, bool evaluate_gradients_y, bool evaluate_gradients_z>
__global__ void
CubeEvaluator_evaluate_simple_grid_gradients(
                                             const double* __restrict__ device_cube,
                                             const size_t device_pitch,
                                             const size_t device_shape_y,
                                             const Grid3D* __restrict__ grid,
                                             double* __restrict__ device_gradients_x,
                                             double* __restrict__ device_gradients_y,
                                             double* __restrict__ device_gradients_z,
                                             // number of slices handled by this device
                                             // in previous calls
                                             int device_slice_offset,
                                             // number of slices handled by all devices
                                             // in previous calls
                                             int slice_offset,
                                             // number of slices handled by this call
                                             int slice_count,
                                             const double multiplier
                                             ) {


    // The result array will be in fortran with indices x, y, z.
    // This means that the x index will be the fastest to change.
    int x, y, z;
    getXYZ(&x, &y, &z);

    const int grid_type_x = grid->axis[X_]->grid_type,
              grid_type_y = grid->axis[Y_]->grid_type,
              grid_type_z = grid->axis[Z_]->grid_type;

    const double h_x = grid->axis[X_]->h[0];
    const double h_y = grid->axis[Y_]->h[0];
    const double h_z = grid->axis[Z_]->h[0];

    // get the offset from the input cube pointer
    int id = getCubeOffset3D(x, y, z+slice_offset, device_pitch, device_shape_y);
    int local_id = getCubeOffset3D(x, y, z+device_slice_offset, device_pitch, device_shape_y);

    bool valid_point =     x >= 0
                        && y >= 0
                        && z+slice_offset >= 0
                        && z < slice_count
                        && x < grid->shape[X_]
                        && y < grid->shape[Y_]
                        && z+slice_offset < grid->shape[Z_];
    if (!valid_point) id = -1;

	// position within a cell.  This is required because there is no
	// translational symmetry by fractions of a cell, and this is relevant for
	// finite diff weights.
    int local_pos_x, local_pos_y, local_pos_z;
    if(grid_type_x == 2 || grid_type_y == 2 || grid_type_z == 2){ // only read in case of lobatto
      local_pos_x = x%(NLIP-1);
      local_pos_y = y%(NLIP-1);
      local_pos_z = (z+slice_offset)%(NLIP-1);
    }


    // evaluate gradient to z direction
    if (evaluate_gradients_z) {
        int prev_id1 = -1, prev_id2 = -1, prev_id3 = -1, prev_id4 = -1, prev_id5 = -1, prev_id6 = -1, prev_id7 = -1, prev_id8 = -1,
            next_id1 = -1, next_id2 = -1, next_id3 = -1, next_id4 = -1, next_id5 = -1, next_id6 = -1, next_id7 = -1, next_id8 = -1;
        if (finite_diff_order >= 2 && z + slice_offset -1 >= 0) {
            prev_id1 = getCubeOffset3D(x, y, z+slice_offset-1, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 2 && z + slice_offset +1 < grid->shape[Z_]) {
            next_id1 = getCubeOffset3D(x, y, z+slice_offset+1, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 3 && z + slice_offset -2 >= 0) {
            prev_id2 = getCubeOffset3D(x, y, z+slice_offset-2, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 3 && z + slice_offset +2 < grid->shape[Z_]) {
            next_id2 = getCubeOffset3D(x, y, z+slice_offset+2, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 4 && z + slice_offset -3 >= 0) {
            prev_id3 = getCubeOffset3D(x, y, z+slice_offset-3, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 4 && z + slice_offset +3 < grid->shape[Z_]) {
            next_id3 = getCubeOffset3D(x, y, z+slice_offset+3, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 5 && z + slice_offset -4 >= 0) {
            prev_id4 = getCubeOffset3D(x, y, z+slice_offset-4, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 5 && z + slice_offset +4 < grid->shape[Z_]) {
            next_id4 = getCubeOffset3D(x, y, z+slice_offset+4, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 6 && z + slice_offset -5 >= 0) {
            prev_id5 = getCubeOffset3D(x, y, z+slice_offset-5, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 6 && z + slice_offset +5 < grid->shape[Z_]) {
            next_id5 = getCubeOffset3D(x, y, z+slice_offset+5, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 7 && z + slice_offset -6 >= 0) {
            prev_id6 = getCubeOffset3D(x, y, z+slice_offset-6, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 7 && z + slice_offset +6 < grid->shape[Z_]) {
            next_id6 = getCubeOffset3D(x, y, z+slice_offset+6, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 8 && z + slice_offset -7 >= 0) {
            prev_id7 = getCubeOffset3D(x, y, z+slice_offset-7, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 8 && z + slice_offset +7 < grid->shape[Z_]) {
            next_id7 = getCubeOffset3D(x, y, z+slice_offset+7, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 9 && z + slice_offset -8 >= 0) {
            prev_id8 = getCubeOffset3D(x, y, z+slice_offset-8, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 9 && z + slice_offset +8 < grid->shape[Z_]) {
            next_id8 = getCubeOffset3D(x, y, z+slice_offset+8, device_pitch, device_shape_y);
        }
        const double value = evaluate_derivative<finite_diff_order>(id, prev_id1, prev_id2, prev_id3, prev_id4, prev_id5, prev_id6, prev_id7, prev_id8,
                                                 next_id1, next_id2, next_id3, next_id4, next_id5, next_id6, next_id7, next_id8,
                                                 device_cube, grid_type_z, local_pos_z, h_z);
        if (valid_point) device_gradients_z[local_id] = multiplier * value;
    }

    // evaluate gradient to y direction
    if (evaluate_gradients_y) {
        int prev_id1 = -1, prev_id2 = -1, prev_id3 = -1, prev_id4 = -1, prev_id5 = -1, prev_id6 = -1, prev_id7 = -1, prev_id8 = -1, 
            next_id1 = -1, next_id2 = -1, next_id3 = -1, next_id4 = -1, next_id5 = -1, next_id6 = -1, next_id7 = -1, next_id8 = -1;
        if (finite_diff_order >= 2 && y -1 >= 0) {
            prev_id1 = getCubeOffset3D(x, y-1, z+slice_offset, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 2 && y + 1 < grid->shape[Y_]) {
            next_id1 = getCubeOffset3D(x, y+1, z+slice_offset, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 3 && y - 2 >= 0) {
            prev_id2 = getCubeOffset3D(x, y-2, z+slice_offset, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 3 && y + 2 < grid->shape[Y_]) {
            next_id2 = getCubeOffset3D(x, y+2, z+slice_offset, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 4 && y - 3 >= 0) {
            prev_id3 = getCubeOffset3D(x, y-3, z+slice_offset, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 4 && y + 3 < grid->shape[Y_]) {
            next_id3 = getCubeOffset3D(x, y+3, z+slice_offset, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 5 && y - 4 >= 0) {
            prev_id4 = getCubeOffset3D(x, y-4, z+slice_offset, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 5 && y + 4 < grid->shape[Y_]) {
            next_id4 = getCubeOffset3D(x, y+4, z+slice_offset, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 6 && y - 5 >= 0) {
            prev_id5 = getCubeOffset3D(x, y-5, z+slice_offset, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 6 && y + 5 < grid->shape[Y_]) {
            next_id5 = getCubeOffset3D(x, y+5, z+slice_offset, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 7 && y - 6 >= 0) {
            prev_id6 = getCubeOffset3D(x, y-6, z+slice_offset, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 7 && y + 6 < grid->shape[Y_]) {
            next_id6 = getCubeOffset3D(x, y+6, z+slice_offset, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 8 && y - 7 >= 0) {
            prev_id7 = getCubeOffset3D(x, y-7, z+slice_offset, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 8 && y + 7 < grid->shape[Y_]) {
            next_id7 = getCubeOffset3D(x, y+7, z+slice_offset, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 9 && y - 8 >= 0) {
            prev_id8 = getCubeOffset3D(x, y-8, z+slice_offset, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 9 && y + 8 < grid->shape[Y_]) {
            next_id8 = getCubeOffset3D(x, y+8, z+slice_offset, device_pitch, device_shape_y);
        }
        const double value = evaluate_derivative<finite_diff_order>(id, prev_id1, prev_id2, prev_id3, prev_id4, prev_id5, prev_id6, prev_id7, prev_id8,
                                                 next_id1, next_id2, next_id3, next_id4, next_id5, next_id6, next_id7, next_id8,
                                                 device_cube, grid_type_y, local_pos_y, h_y);
        if (valid_point) device_gradients_y[local_id] = multiplier * value;
    }

    // evaluate gradient to x direction
    if (evaluate_gradients_x) {
        int prev_id1 = -1, prev_id2 = -1, prev_id3 = -1, prev_id4 = -1, prev_id5 = -1, prev_id6 = -1, prev_id7 = -1, prev_id8 = -1, 
            next_id1 = -1, next_id2 = -1, next_id3 = -1, next_id4 = -1, next_id5 = -1, next_id6 = -1, next_id7 = -1, next_id8 = -1;
        if (finite_diff_order >= 2 && x - 1 >= 0) {
            prev_id1 = getCubeOffset3D(x-1, y, z+slice_offset, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 2 && x + 1 < grid->shape[X_]) {
            next_id1 = getCubeOffset3D(x+1, y, z+slice_offset, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 3 && x - 2 >= 0) {
            prev_id2 = getCubeOffset3D(x-2, y, z+slice_offset, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 3 && x + 2 < grid->shape[X_]) {
            next_id2 = getCubeOffset3D(x+2, y, z+slice_offset, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 4 && x - 3 >= 0) {
            prev_id3 = getCubeOffset3D(x-3, y, z+slice_offset, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 4 && x + 3 < grid->shape[X_]) {
            next_id3 = getCubeOffset3D(x+3, y, z+slice_offset, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 5 && x - 4 >= 0) {
            prev_id4 = getCubeOffset3D(x-4, y, z+slice_offset, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 5 && x + 4 < grid->shape[X_]) {
            next_id4 = getCubeOffset3D(x+4, y, z+slice_offset, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 6 && x - 5 >= 0) {
            prev_id5 = getCubeOffset3D(x-5, y, z+slice_offset, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 6 && x + 5 < grid->shape[X_]) {
            next_id5 = getCubeOffset3D(x+5, y, z+slice_offset, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 7 && x - 6 >= 0) {
            prev_id6 = getCubeOffset3D(x-6, y, z+slice_offset, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 7 && x + 6 < grid->shape[X_]) {
            next_id6 = getCubeOffset3D(x+6, y, z+slice_offset, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 8 && x - 7 >= 0) {
            prev_id7 = getCubeOffset3D(x-7, y, z+slice_offset, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 8 && x + 7 < grid->shape[X_]) {
            next_id7 = getCubeOffset3D(x+7, y, z+slice_offset, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 9 && x - 8 >= 0) {
            prev_id8 = getCubeOffset3D(x-8, y, z+slice_offset, device_pitch, device_shape_y);
        }

        if (finite_diff_order >= 9 && x + 8 < grid->shape[X_]) {
            next_id8 = getCubeOffset3D(x+8, y, z+slice_offset, device_pitch, device_shape_y);
        }
        const double value = evaluate_derivative<finite_diff_order>(id, prev_id1, prev_id2, prev_id3, prev_id4, prev_id5, prev_id6, prev_id7, prev_id8,
                                                 next_id1, next_id2, next_id3, next_id4, next_id5, next_id6, next_id7, next_id8,
                                                 device_cube, grid_type_x, local_pos_x, h_x);
        if (valid_point) device_gradients_x[local_id] = multiplier * value;
    }

    return;
}

/*
 * Evaluate values of the radial gradients at bubbles, i.e.,
 * the radial gradients of input bubbles are evaluated to the
 * result bubbles values.
 *
 * @param nlip - number of lagrange integration polyniomials per
 *               cell, i.e., the number of grid points per cell
 */
template <int nlip>
__device__ inline void BubblesEvaluator_evaluate_radial_gradients(
                                         const Grid1D* __restrict__ grid,
                                         // maximum quantum number 'l'
                                         const int lmax,
                                         // k value for the bubble
                                         const int &k,
                                         // constant pointer to a double array representing the
                                         // input bubbles f_Alm coefficients
                                         const double* __restrict__ f,
                                         // pointer to a variable double array representing the
                                         // output bubbles g_Alm coefficients
                                         double* result,
                                         // global offset in warps
                                         const int warp_offset
                                        ) {
    int global_warp_id, thread_order_number, cells_per_warp;
    bool valid_point = true;
    const int WARP_SIZE = 32;

    // order number of handled point
    const int id = threadIdx.x + blockIdx.x * blockDim.x;

    // if nlip is 7, each warp of 32 handles 5 cells
    if (nlip == 7) {
        // get the global warp order number
        global_warp_id =   id / WARP_SIZE + warp_offset;
        cells_per_warp = 5;

        // get the order number of thread within the warp
        thread_order_number = threadIdx.x % WARP_SIZE;

        if (thread_order_number == 31) valid_point = false;
    }


    // number of cells
    const int ncell = grid->ncell;

    // get the order number of cell
    int icell = global_warp_id * cells_per_warp + thread_order_number / (nlip-1);
    // order number of point in cell
    int in_cell_point = thread_order_number % (nlip-1);

    // let's set it up so that the nlip:th point in cell belongs to the previous cell
    if (in_cell_point == 0 && icell > 0) {
        icell -= 1;
        in_cell_point = nlip;
    }
    if (thread_order_number == 0 && in_cell_point != 0) valid_point = false;

    // if the cell number is not within the evaluated range, we do not evaluate the
    // values
    bool participant = true;
    if (icell >= ncell ) {
        participant = false;
    }

    double in_cell_coordinate = (double)(in_cell_point-3);

    // read the LIPs in the shared memory
    __shared__ double lip[nlip * nlip];
    read_lip<nlip, nlip>(grid->lip, threadIdx.x, lip);
    __shared__ double derivative_lip[(nlip-1) * nlip];
    read_lip<nlip-1, nlip>(grid->derivative_lip, threadIdx.x, derivative_lip);


    if (participant) {
        // evaluate the derivative polynomials
        double derivative_polynomials[nlip];
        evaluate_polynomials<nlip-1, nlip>(derivative_lip, in_cell_coordinate, derivative_polynomials);
        double one_per_grid_step = 1.0 / grid->h[icell];
        // finally, multiply the derivative polynomials with 1 / grid_step
        for (int j = 0; j < nlip; j++) {
            derivative_polynomials[j] *= one_per_grid_step;
        }

        // get the initial address:
        int address = icell * (nlip-1) + in_cell_point;

        for (int n = 0; n < (lmax+1) * (lmax+1); n++) {
            // get the input function values
            double value = f[address];
            // and evaluate the radial coefficients
            double temp = evaluate_coefficients_shuffle<nlip>(derivative_polynomials, value, thread_order_number, in_cell_point);
            // if the point is valid, stored the result
            if (valid_point) result[address] = temp;

            // add the address by one n index:
            address += ncell * nlip;
        }
    }
}


/**************************************************************
 *  Error checking                                            *
 **************************************************************/
__host__ inline void check_eval_errors(const char *filename, const int line_number) {
//#ifdef DEBUG_CUDA
    cudaDeviceSynchronize();
//#endif

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      printf("CUDA error at %s:%i: %s\n", filename, line_number, cudaGetErrorString(error));
      exit(-1);
    }
}




/**************************************************************
 *  BubblesEvaluator-implementation                           *
 **************************************************************/



BubblesEvaluator::BubblesEvaluator(StreamContainer *streamContainer) {
    this->streamContainer = streamContainer;
}


void BubblesEvaluator::setBubbles(Bubbles *bubbles) {
    this->bubbles = bubbles;
}

/*
 * Evaluate the bubbles at grid points.
 *
 * @param bubbles - The bubbles that are evaluated in to the grid
 * @param grid - The grid associated with all the output cubes
 */
void BubblesEvaluator::evaluateGrid(Grid3D *grid, CudaCube *result_cube, CudaCube *gradient_cube_x, CudaCube *gradient_cube_y, CudaCube *gradient_cube_z, int gradient_direction, int fin_diff_ord) {

    if (gradient_direction == X_) {
        this->bubbles->inject(grid, result_cube, 0, gradient_cube_x,
                            gradient_cube_y, gradient_cube_z, false, true, false, false);
    }
    else if (gradient_direction == Y_) {
        this->bubbles->inject(grid, result_cube, 0, gradient_cube_x,
                            gradient_cube_y, gradient_cube_z, false, false, true, false);
    }
    else if (gradient_direction == Z_) {
        this->bubbles->inject(grid, result_cube, 0, gradient_cube_x,
                            gradient_cube_y, gradient_cube_z, false, false, false, true);
    }
    else if (gradient_direction == 3) {
        this->bubbles->inject(grid, result_cube, 0, gradient_cube_x,
                            gradient_cube_y, gradient_cube_z, true, true, true, true);
    }
    else {
        this->bubbles->inject(grid, result_cube);
    }

}

/*
 * Deallocate the device and host memory allocated for this object.
 */
void BubblesEvaluator::destroy() {
    this->streamContainer = NULL;
    this->bubbles = NULL;
}

/**************************************************************
 *  CubeEvaluator-implementation                           *
 **************************************************************/



CubeEvaluator::CubeEvaluator(StreamContainer *streamContainer) {
    this->streamContainer = streamContainer;
}

/*
 * Deallocate the device and host memory allocated for this object.
 */
void CubeEvaluator::destroy() {
    this->streamContainer = NULL;
    this->input_cube = NULL;
    this->grid = NULL;
}


/*
 * Set the input cube from which the evaluation is performed.
 *
 * @param input_cube - CudaCube object from which the evaluation is performed. The shape
 *                     of the data should be according to the given grid
 */
void CubeEvaluator::setInputCube(CudaCube *input_cube) {
    this->input_cube = input_cube;
}

/*
 * Set the input grid from which the evaluation is performed.
 *
 * @param input_grid - Grid3D object defining the shape of the cube for which the evaluation is performed.
 */
void CubeEvaluator::setInputGrid(Grid3D *input_grid) {
    this->grid = input_grid;
}


/*
 * Evaluate the cube at preset points. The results are stored in the device memory.
 * @param result_points      - Points-object in which the results are stored, if gradient_direction=0-2, the results are stored here
 * @param gradient_points_x  - Points-object in which the gradiends in x-direction are stored, if gradient_direction=3
 * @param gradient_points_y  - Points-object in which the gradiends in y-direction are stored, if gradient_direction=3
 * @param gradient_points_z  - Points-object in which the gradiends in z-direction are stored, if gradient_direction=3
 * @param gradient_direction - possible values X_ = 0, Y_ = 1, Z_ = 2, (X_, Y_, Z_) = 3 && this->evaluateGradients
 *                             anything else: no gradients
 */
void CubeEvaluator::evaluatePoints(Points *result_points,
                                   Points *gradient_points_x,
                                   Points *gradient_points_y,
                                   Points *gradient_points_z,
                                   int gradient_direction) {
    int warp_size = 32;
    int total_warp_count = result_points->point_coordinates->number_of_points / warp_size + ((result_points->point_coordinates->number_of_points % warp_size) > 0);
    int point_offset = 0;
    int *cube_memory_shape = this->input_cube->getDeviceMemoryShape();
    check_eval_errors(__FILE__, __LINE__);
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        this->streamContainer->setDevice(device);

        // allocate space for device results and device points
        int device_warp_count = total_warp_count / this->streamContainer->getNumberOfDevices()
                                  + ((total_warp_count % this->streamContainer->getNumberOfDevices()) > device);
        int device_point_count = device_warp_count * warp_size;
        int device_point_offset = 0;
        // get the order number of 'device' in cube's streamcontainer
        int cube_device = this->input_cube->getStreamContainer()->getDeviceOrderNumber(this->streamContainer->getDeviceNumber(device));
        check_eval_errors(__FILE__, __LINE__);

        // get the pointers to the device points & results
        double *device_points_ptr =  result_points->point_coordinates->device_coordinates[device];
        double *device_results_ptr = result_points->device_values[device];
        double *device_gradients_x_ptr;
        double *device_gradients_y_ptr;
        double *device_gradients_z_ptr;

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

            if (stream_point_count > 0) {
                // set the result to zero
                check_eval_errors(__FILE__, __LINE__);

                int grid_size = (stream_point_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                if (gradient_direction == X_) {
                    CubeEvaluator_evaluate_points <false, true, false, false>
                        <<< grid_size, BLOCK_SIZE, 0,
                            *this->streamContainer->getStream(device, stream) >>>
                                (this->input_cube->getDevicePointer(cube_device),
                                    this->input_cube->getDevicePitch(cube_device),
                                    cube_memory_shape[Y_],
                                    // TODO: replace the cube_device with grid_device in below line, probably does not
                                    //       matter but we have to be careful.
                                    grid->device_copies[cube_device],
                                    device_results_ptr,
                                    device_gradients_x_ptr,
                                    device_gradients_y_ptr,
                                    device_gradients_z_ptr,
                                    device_points_ptr,
                                    device_point_count,
                                    stream_point_count,
                                    device_point_offset,
                                    1.0
                                );
                }
                else if (gradient_direction == Y_) {
                    CubeEvaluator_evaluate_points <false, false, true, false>
                        <<< grid_size, BLOCK_SIZE, 0,
                            *this->streamContainer->getStream(device, stream) >>>
                                (this->input_cube->getDevicePointer(cube_device),
                                    this->input_cube->getDevicePitch(cube_device),
                                    cube_memory_shape[Y_],
                                    // TODO: replace the cube_device with grid_device in below line, probably does not
                                    //       matter but we have to be careful.
                                    this->grid->device_copies[cube_device],
                                    device_results_ptr,
                                    device_gradients_x_ptr,
                                    device_gradients_y_ptr,
                                    device_gradients_z_ptr,
                                    device_points_ptr,
                                    device_point_count,
                                    stream_point_count,
                                    device_point_offset,
                                    1.0
                                );
                }
                else if (gradient_direction == Z_) {
                    CubeEvaluator_evaluate_points <false, false, false, true>
                        <<< grid_size, BLOCK_SIZE, 0,
                            *this->streamContainer->getStream(device, stream) >>>
                                (this->input_cube->getDevicePointer(cube_device),
                                    this->input_cube->getDevicePitch(cube_device),
                                    cube_memory_shape[Y_],
                                    // TODO: replace the cube_device with grid_device in below line, probably does not
                                    //       matter but we have to be careful.
                                    this->grid->device_copies[cube_device],
                                    device_results_ptr,
                                    device_gradients_x_ptr,
                                    device_gradients_y_ptr,
                                    device_gradients_z_ptr,
                                    device_points_ptr,
                                    device_point_count,
                                    stream_point_count,
                                    device_point_offset,
                                    1.0
                                );
                }
                else if (gradient_direction == 3) {
                    CubeEvaluator_evaluate_points <true, true, true, true>
                        <<< grid_size, BLOCK_SIZE, 0,
                            *this->streamContainer->getStream(device, stream) >>>
                                (this->input_cube->getDevicePointer(cube_device),
                                    this->input_cube->getDevicePitch(cube_device),
                                    cube_memory_shape[Y_],
                                    // TODO: replace the cube_device with grid_device in below line, probably does not
                                    //       matter but we have to be careful.
                                    this->grid->device_copies[cube_device],
                                    device_results_ptr,
                                    device_gradients_x_ptr,
                                    device_gradients_y_ptr,
                                    device_gradients_z_ptr,
                                    device_points_ptr,
                                    device_point_count,
                                    stream_point_count,
                                    device_point_offset,
                                    1.0
                                );
                }
                else {
                    CubeEvaluator_evaluate_points <true, false, false, false>
                        <<< grid_size, BLOCK_SIZE, 0,
                            *this->streamContainer->getStream(device, stream) >>>
                                (this->input_cube->getDevicePointer(cube_device),
                                    this->input_cube->getDevicePitch(cube_device),
                                    cube_memory_shape[Y_],
                                    // TODO: replace the cube_device with grid_device in below line, probably does not
                                    //       matter but we have to be careful.
                                    this->grid->device_copies[cube_device],
                                    device_results_ptr,
                                    device_gradients_x_ptr,
                                    device_gradients_y_ptr,
                                    device_gradients_z_ptr,
                                    device_points_ptr,
                                    device_point_count,
                                    stream_point_count,
                                    device_point_offset,
                                    1.0
                                );
                }
            }
            check_eval_errors(__FILE__, __LINE__);

            // add the pointers
            point_offset += stream_point_count;
            device_point_offset += stream_point_count;
        }
        check_eval_errors(__FILE__, __LINE__);
    }
}


/*
 * Evaluate the cube at the points of grid. The results are stored in the device memory
 * in the result_cube and gradient_cubes. The latter only occurs if gradient_direction == 3.
 * true.
 *
 * @param grid               - The grid associated with all the input and output cubes
 * @param results_cube       - CudaCube where the results are stored, if gradient direction is 0-2, the gradients will be stored here
 * @param gradients_cube_x   - CudaCube where the x-gradients are stored if the gradient_direction=3
 * @param gradients_cube_y   - CudaCube where the y-gradients are stored if the gradient_direction=3
 * @param gradients_cube_z   - CudaCube where the z-gradients are stored if the gradient_direction=3
 * @param gradient_direction - possible values X_ = 0, Y_ = 1, Z_ = 2, (X_, Y_, Z_) = 3 && this->evaluateGradients
 *                             anything else: no gradients
 *
 */
void CubeEvaluator::evaluateGrid(Grid3D *grid,
                                 CudaCube *result_cube,
                                 CudaCube *gradient_cube_x,
                                 CudaCube *gradient_cube_y,
                                 CudaCube *gradient_cube_z,
                                 const int gradient_direction,
                                 const int finite_diff_order) {
    check_eval_errors(__FILE__, __LINE__);

    // printf("fin diff order in evaluateGrid: %i, %i \n", finite_diff_order, gradient_direction);

    int total_slice_count = result_cube->getShape(Z_);
    // the minimum l is 0 always in the multiplication
    int device_slice_count;

    // get the input cube pointer
    // TODO: we are assuming here, that the input and output cubes have the same
    // memory shapes, this is probably not the case in all occasions in the future
    double **device_input_cubes = this->input_cube->getDeviceCubes();

    // get the pointer arrays from the cubes
    double **device_cubes = result_cube->getDeviceCubes();
    double **device_gradients_x;
    double **device_gradients_y;
    double **device_gradients_z;

    // get the device gradient result pointers
    if (gradient_direction < 3) {
        device_gradients_x = result_cube->getDeviceCubes();
        device_gradients_y = result_cube->getDeviceCubes();
        device_gradients_z = result_cube->getDeviceCubes();
    }
    else {
        device_gradients_x = gradient_cube_x->getDeviceCubes();
        device_gradients_y = gradient_cube_y->getDeviceCubes();
        device_gradients_z = gradient_cube_z->getDeviceCubes();
    }

    size_t *device_pitches = result_cube->getDevicePitches();
    int *device_memory_shape = result_cube->getDeviceMemoryShape();

    // init some stuff to help calculate the launch parameters
    // NOTE: these are for nlip: 7
    //int cells_per_block =  BLOCK_SIZE / 32 * 5;
    int warps_per_string = grid->axis[X_]->ncell / 5 + 1;
    int warps_per_slice = grid->axis[Y_]->ncell * warps_per_string;
    int warps_per_block = BLOCK_SIZE / 32;
    int slice_offset = 0;

    // copy the cubes to the device & execute the kernels
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        // set the used device (gpu)
        this->streamContainer->setDevice(device);

        //double *dev_cube  = device_cubes[device];
        double *dev_input_cube = device_input_cubes[device];
        double *dev_gradient_x = device_gradients_x[device];
        double *dev_gradient_y = device_gradients_y[device];
        double *dev_gradient_z = device_gradients_z[device];
        int device_slice_offset = 0;

        // calculate the number of vectors this device handles
        device_slice_count =  total_slice_count / this->streamContainer->getNumberOfDevices()
                                  + ((total_slice_count % this->streamContainer->getNumberOfDevices()) > device);
        for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream++) {

            // determine the count of vectors handled by this stream
            int slice_count = device_slice_count / this->streamContainer->getStreamsPerDevice()
                                + ((device_slice_count % this->streamContainer->getStreamsPerDevice()) > stream);

            check_eval_errors(__FILE__, __LINE__);

            if (slice_count > 0) {
                // calculate the launch configuration for the f1-inject
                //int grid_size = warps_per_slice * slice_count * warps_per_block + 1;
                dim3 block, launch_grid;
                result_cube->getLaunchConfiguration(&launch_grid, &block, slice_count, BLOCK_SIZE);

                // call the kernel
                if (gradient_direction == X_ && finite_diff_order==7) {
                    CubeEvaluator_evaluate_simple_grid_gradients <7, true, false, false>
                        <<< launch_grid, block, 0,
                            *this->streamContainer->getStream(device, stream) >>>
                                        (dev_input_cube,
                                        device_pitches[device],
                                        device_memory_shape[Y_],
                                        grid->device_copies[device],
                                        dev_gradient_x,
                                        dev_gradient_y,
                                        dev_gradient_z,
                                        // number of slices handled by this device
                                        // in previous calls
                                        device_slice_offset,
                                        // number of slices handled by all devices
                                        // in previous calls
                                        slice_offset,
                                        // number of slices handled by this call
                                        slice_count,
                                        //warps_per_string,
                                        1.0);
                }
                if (gradient_direction == X_ && finite_diff_order==9) {
                    CubeEvaluator_evaluate_simple_grid_gradients <9, true, false, false>
                        <<< launch_grid, block, 0,
                            *this->streamContainer->getStream(device, stream) >>>
                                        (dev_input_cube,
                                        device_pitches[device],
                                        device_memory_shape[Y_],
                                        grid->device_copies[device],
                                        dev_gradient_x,
                                        dev_gradient_y,
                                        dev_gradient_z,
                                        // number of slices handled by this device
                                        // in previous calls
                                        device_slice_offset,
                                        // number of slices handled by all devices
                                        // in previous calls
                                        slice_offset,
                                        // number of slices handled by this call
                                        slice_count,
                                        //warps_per_string,
                                        1.0);
                }
                else if (gradient_direction == Y_ && finite_diff_order==7) {
                    CubeEvaluator_evaluate_simple_grid_gradients <7, false, true, false>
                        <<< launch_grid, block, 0,
                            *this->streamContainer->getStream(device, stream) >>>
                                        (dev_input_cube,
                                        device_pitches[device],
                                        device_memory_shape[Y_],
                                        grid->device_copies[device],
                                        dev_gradient_x,
                                        dev_gradient_y,
                                        dev_gradient_z,
                                        // number of slices handled by this device
                                        // in previous calls
                                        device_slice_offset,
                                        // number of slices handled by all devices
                                        // in previous calls
                                        slice_offset,
                                        // number of slices handled by this call
                                        slice_count,
                                        //warps_per_string,
                                        1.0);
                }
                else if (gradient_direction == Y_ && finite_diff_order==9) {
                    CubeEvaluator_evaluate_simple_grid_gradients <9, false, true, false>
                        <<< launch_grid, block, 0,
                            *this->streamContainer->getStream(device, stream) >>>
                                        (dev_input_cube,
                                        device_pitches[device],
                                        device_memory_shape[Y_],
                                        grid->device_copies[device],
                                        dev_gradient_x,
                                        dev_gradient_y,
                                        dev_gradient_z,
                                        // number of slices handled by this device
                                        // in previous calls
                                        device_slice_offset,
                                        // number of slices handled by all devices
                                        // in previous calls
                                        slice_offset,
                                        // number of slices handled by this call
                                        slice_count,
                                        //warps_per_string,
                                        1.0);
                }
                else if (gradient_direction == Z_ && finite_diff_order==7) {
                    CubeEvaluator_evaluate_simple_grid_gradients <7, false, false, true>
                        <<< launch_grid, block, 0,
                            *this->streamContainer->getStream(device, stream) >>>
                                        (dev_input_cube,
                                        device_pitches[device],
                                        device_memory_shape[Y_],
                                        grid->device_copies[device],
                                        dev_gradient_x,
                                        dev_gradient_y,
                                        dev_gradient_z,
                                        // number of slices handled by this device
                                        // in previous calls
                                        device_slice_offset,
                                        // number of slices handled by all devices
                                        // in previous calls
                                        slice_offset,
                                        // number of slices handled by this call
                                        slice_count,
                                        //warps_per_string,
                                        1.0);
                }
                else if (gradient_direction == Z_ && finite_diff_order==9) {
                    CubeEvaluator_evaluate_simple_grid_gradients <9, false, false, true>
                        <<< launch_grid, block, 0,
                            *this->streamContainer->getStream(device, stream) >>>
                                        (dev_input_cube,
                                        device_pitches[device],
                                        device_memory_shape[Y_],
                                        grid->device_copies[device],
                                        dev_gradient_x,
                                        dev_gradient_y,
                                        dev_gradient_z,
                                        // number of slices handled by this device
                                        // in previous calls
                                        device_slice_offset,
                                        // number of slices handled by all devices
                                        // in previous calls
                                        slice_offset,
                                        // number of slices handled by this call
                                        slice_count,
                                        //warps_per_string,
                                        1.0);
                }
                else if (gradient_direction == 3 && finite_diff_order==7) {
                    CubeEvaluator_evaluate_simple_grid_gradients <7, true, true, true>
                        <<< launch_grid, block, 0,
                            *this->streamContainer->getStream(device, stream) >>>
                                        (dev_input_cube,
                                        device_pitches[device],
                                        device_memory_shape[Y_],
                                        grid->device_copies[device],
                                        dev_gradient_x,
                                        dev_gradient_y,
                                        dev_gradient_z,
                                        // number of slices handled by this device
                                        // in previous calls
                                        device_slice_offset,
                                        // number of slices handled by all devices
                                        // in previous calls
                                        slice_offset,
                                        // number of slices handled by this call
                                        slice_count,
                                        //warps_per_string,
                                        1.0);
                }
                else if (gradient_direction == 3 && finite_diff_order==9) {
                    CubeEvaluator_evaluate_simple_grid_gradients <9, true, true, true>
                        <<< launch_grid, block, 0,
                            *this->streamContainer->getStream(device, stream) >>>
                                        (dev_input_cube,
                                        device_pitches[device],
                                        device_memory_shape[Y_],
                                        grid->device_copies[device],
                                        dev_gradient_x,
                                        dev_gradient_y,
                                        dev_gradient_z,
                                        // number of slices handled by this device
                                        // in previous calls
                                        device_slice_offset,
                                        // number of slices handled by all devices
                                        // in previous calls
                                        slice_offset,
                                        // number of slices handled by this call
                                        slice_count,
                                        //warps_per_string,
                                        1.0);
                }

                check_eval_errors(__FILE__, __LINE__);
                // increase the address by the number of vectors in this array
                device_slice_offset += slice_count;
                slice_offset += slice_count;
            }
        }
    }
}

/********************************************
 *  Fortran interfaces for Evaluator        *
 ********************************************/

extern "C" void evaluator_evaluate_grid_cuda(Evaluator *evaluator, Grid3D *grid, CudaCube *result_cube, CudaCube *gradient_cube_x, CudaCube *gradient_cube_y, CudaCube *gradient_cube_z, int gradient_direction, int fin_diff_ord) {
    evaluator->evaluateGrid(grid, result_cube, gradient_cube_x, gradient_cube_y, gradient_cube_z, gradient_direction, fin_diff_ord);
}

extern "C" void evaluator_evaluate_grid_without_gradients_cuda(Evaluator *evaluator, Grid3D *grid, CudaCube *result_cube, int fin_diff_ord) {
    evaluator->evaluateGrid(grid, result_cube, NULL, NULL, NULL, -1, fin_diff_ord);
}

extern "C" void evaluator_evaluate_grid_x_gradients_cuda(Evaluator *evaluator, Grid3D *grid, CudaCube *result_cube, int fin_diff_ord) {
    evaluator->evaluateGrid(grid, result_cube, NULL, NULL, NULL, X_, fin_diff_ord);
}

extern "C" void evaluator_evaluate_grid_y_gradients_cuda(Evaluator *evaluator, Grid3D *grid, CudaCube *result_cube, int fin_diff_ord) {
    evaluator->evaluateGrid(grid, result_cube, NULL, NULL, NULL, Y_, fin_diff_ord);
}

extern "C" void evaluator_evaluate_grid_z_gradients_cuda(Evaluator *evaluator, Grid3D *grid, CudaCube *result_cube, int fin_diff_ord) {
    evaluator->evaluateGrid(grid, result_cube, NULL, NULL, NULL, Z_, fin_diff_ord);
}

extern "C" void evaluator_evaluate_points_cuda(Evaluator *evaluator, Points *result_points, Points *gradient_points_x, Points *gradient_points_y, Points *gradient_points_z, int gradient_direction) {
    evaluator->evaluatePoints(result_points, gradient_points_x, gradient_points_y, gradient_points_z, gradient_direction);
}

extern "C" void evaluator_evaluate_points_without_gradients_cuda(Evaluator *evaluator, Points *result_points) {
    evaluator->evaluatePoints(result_points, NULL, NULL, NULL, -1);
}

extern "C" void evaluator_evaluate_points_x_gradients_cuda(Evaluator *evaluator, Points *result_points) {
    evaluator->evaluatePoints(result_points, NULL, NULL, NULL, X_);
}

extern "C" void evaluator_evaluate_points_y_gradients_cuda(Evaluator *evaluator, Points *result_points) {
    evaluator->evaluatePoints(result_points, NULL, NULL, NULL, Y_);
}

extern "C" void evaluator_evaluate_points_z_gradients_cuda(Evaluator *evaluator, Points *result_points) {
    evaluator->evaluatePoints(result_points, NULL, NULL, NULL, Z_);
}

extern "C" void evaluator_destroy_cuda(Evaluator *evaluator) {
    evaluator->destroy();
}

/********************************************
 *  Fortran interfaces for BubblesEvaluator *
 ********************************************/


extern "C" BubblesEvaluator *bubblesevaluator_init_cuda(StreamContainer *streamContainer) {
    BubblesEvaluator *new_bubbles_evaluator = new BubblesEvaluator(streamContainer);
    return new_bubbles_evaluator;
}

extern "C" void bubblesevaluator_set_bubbles_cuda(BubblesEvaluator *bubbles_evaluator, Bubbles *bubbles) {
    bubbles_evaluator->setBubbles(bubbles);
}


/********************************************
 *  Fortran interfaces for CubeEvaluator *
 ********************************************/



extern "C" CubeEvaluator *cubeevaluator_init_cuda(StreamContainer *streamContainer) {
    CubeEvaluator *new_cube_evaluator = new CubeEvaluator(streamContainer);
    return new_cube_evaluator;
}

extern "C" void cubeevaluator_set_input_cube_cuda(CubeEvaluator *cube_evaluator, CudaCube *cube) {
    cube_evaluator->setInputCube(cube);
}

extern "C" void cubeevaluator_set_input_grid_cuda(CubeEvaluator *cube_evaluator, Grid3D *grid) {
    cube_evaluator->setInputGrid(grid);
}




