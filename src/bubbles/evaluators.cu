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
    addresses[0] =__shfl(address, 0, EVALUATE_BLOCK_SIZE)  + thread_rank;
    addresses[1] =__shfl(address, 1, EVALUATE_BLOCK_SIZE)  + thread_rank;
    addresses[2] =__shfl(address, 2, EVALUATE_BLOCK_SIZE)  + thread_rank;
    addresses[3] =__shfl(address, 3, EVALUATE_BLOCK_SIZE)  + thread_rank;
    addresses[4] =__shfl(address, 4, EVALUATE_BLOCK_SIZE)  + thread_rank;
    addresses[5] =__shfl(address, 5, EVALUATE_BLOCK_SIZE)  + thread_rank;
    addresses[6] =__shfl(address, 6, EVALUATE_BLOCK_SIZE)  + thread_rank;
    addresses[7] =__shfl(address, 7, EVALUATE_BLOCK_SIZE)  + thread_rank;

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



// fixme lnw
// fixme: why is there evaluate_derivative{xy/z}?  Why should they be treated independently? lnw
// nlip tests outcommented, because nlip=7 is hardcoded all over the place anyway (lnw)
__device__ __forceinline__
double evaluate_derivative(const int current_id,
                           const int previous_id1, const int previous_id2, const int previous_id3, const int previous_id4, const int previous_id5, const int previous_id6,
                           const int next_id1, const int next_id2, const int next_id3, const int next_id4, const int next_id5, const int next_id6,
                           const double* __restrict__ device_cube, const int fin_diff_order, const int grid_type, const int local_pos, const double h){
    if (current_id == -1) return 0.0;

    // printf("xy: %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i\n",  previous_id6, previous_id5, previous_id4, previous_id3, previous_id2, previous_id1, next_id1, next_id2, next_id3, next_id4, next_id5, next_id6);

    double current_value,
           previous_value1, previous_value2, previous_value3, previous_value4, previous_value5, previous_value6,
           next_value1, next_value2, next_value3, next_value4, next_value5, next_value6;
    if (previous_id1 > -1) previous_value1 = __ldg(&device_cube[previous_id1]);
    if (current_id > -1)   current_value   = __ldg(&device_cube[current_id]);
    if (next_id1 > -1)     next_value1     = __ldg(&device_cube[next_id1]);
    if (fin_diff_order >= 3){
        if (previous_id2 > -1) previous_value2 = __ldg(&device_cube[previous_id2]);
        if (next_id2 > -1)     next_value2     = __ldg(&device_cube[next_id2]);
    }
    if (fin_diff_order >= 4){
        if (previous_id3 > -1) previous_value3 = __ldg(&device_cube[previous_id3]);
        if (next_id3 > -1)     next_value3     = __ldg(&device_cube[next_id3]);
    }
    if (fin_diff_order >= 5){
        if (previous_id4 > -1) previous_value4 = __ldg(&device_cube[previous_id4]);
        if (next_id4 > -1)     next_value4     = __ldg(&device_cube[next_id4]);
    }
    if (fin_diff_order >= 6){
        if (next_id5 > -1)     next_value5     = __ldg(&device_cube[next_id5]);
        if (previous_id5 > -1) previous_value5 = __ldg(&device_cube[previous_id5]);
    }
    if (fin_diff_order >= 7){
        if (previous_id6 > -1) previous_value6 = __ldg(&device_cube[previous_id6]);
        if (next_id6 > -1)     next_value6     = __ldg(&device_cube[next_id6]);
    }

    if(grid_type == 1){ // equidistant
#if 1
        if ( fin_diff_order >= 11 && previous_id1 > -1 && next_id1 > -1 && previous_id2 > -1 && next_id2 > -1 && previous_id3 > -1 && next_id3 > -1 && previous_id4 > -1 && next_id4 > -1 && previous_id5 > -1 && next_id5 > -1 ) { // x x x x x o x x x x x
            return (-1.0 * previous_value5 + 12.5 * previous_value4 - 75.0 * previous_value3 + 300.0 * previous_value2 - 1050.0 * previous_value1 + 1050.0 * next_value1 - 300.0 * next_value2 + 75.0 * next_value3 - 12.5 * next_value4 + 1.0 * next_value5) / (1260.0*h);
        }
        else if ( fin_diff_order >= 9 && previous_id1 > -1 && next_id1 > -1 && previous_id2 > -1 && next_id2 > -1 && previous_id3 > -1 && next_id3 > -1 && previous_id4 > -1 && next_id4 > -1 ) { // x x x x o x x x x
            return (3.0 * previous_value4 - 32.0 * previous_value3 + 168.0 * previous_value2 - 672.0 * previous_value1 + 672.0 * next_value1 - 168.0 * next_value2 + 32.0 * next_value3 - 3.0 * next_value4 ) / (840.0*h);
        }
        else if ( fin_diff_order >= 7 && previous_id1 > -1 && next_id1 > -1 && previous_id2 > -1 && next_id2 > -1 && previous_id3 > -1 && next_id3 > -1 ) { // x x x o x x x
            return (-1.0 * previous_value3 + 9.0 * previous_value2 - 45.0 * previous_value1 + 45.0 * next_value1 - 9.0 * next_value2 + 1.0 * next_value3) / (60.0*h);
        }
        else if ( fin_diff_order >= 5 && previous_id1 > -1 && next_id1 > -1 && previous_id2 > -1 && next_id2 > -1 ) { // x x o x x
            return (-1.0 * next_value2 + 8.0 * next_value1 - 8.0 * previous_value1 + previous_value2) / (12.0*h);
        }
        else if ( previous_id1 > -1 && next_id1 > -1 ) { // x o x
            return (next_value1 - previous_value1) / (2.0 * h);
        }
        else if ( next_id1 > -1 && next_id2 > -1 ) { // o x x
            return (-1.0 * next_value2 + 4.0 * next_value1 - 3.0 * current_value) / (2.0 * h);
        }
        else if ( previous_id1 > -1 && previous_id2 > -1) { // x x o
            return (1.0 * previous_value2 - 4.0 * previous_value2 + 3.0 * current_value) / (2.0 * h);
        }
        else if ( previous_id1 == -1 && next_id1 > -1 ) { // - o x
            return (next_value1 - current_value) / h;
        }
        else if ( previous_id1 > -1 && next_id1 == -1 ) { // x o -
            return (current_value - previous_value1) / h;
        }

#else

    // {3., -32., 168., -672., 0, 672., -168., 32., -3.}
        if ( previous_id1 > -1 && next_id1 > -1 && previous_id2 > -1 && next_id2 > -1 && previous_id3 > -1 && next_id3 > -1 && previous_id4 > -1 && next_id4 > -1 ) { // x x x x o x x x x
            return (3.0 * previous_value4 - 32.0 * previous_value3 + 168.0 * previous_value2 - 672.0 * previous_value1 + 672.0 * next_value1 - 168.0 * next_value2 + 32.0 * next_value3 - 3.0 * next_value4 ) / (840.0*h);
        }
    // equidistant, using 7 points:
    // accuracy order 6: {-2.45           ,6              ,-7.5           ,6.66666666667 ,-3.75         ,1.2           ,-0.166666666667}      {-147., 360 , -450., 400. , -225., 72. ,-10.} / 60
    // accuracy order 6: {-0.166666666667 ,-1.28333333333 ,2.5            ,-1.66666666667,0.833333333333,-0.25         ,0.0333333333333}      {-10. , -77., 150. , -100., 50.  , -15.,  2.} / 60
    // accuracy order 6: {0.0333333333333 ,-0.4           ,-0.583333333333,1.33333333333 ,-0.5          ,0.133333333333,-0.0166666666667}     {2.   , -24., -35. , 80.  , -30. , 8.  , -1.} / 60
    // accuracy order 6: {-0.0166666666667,0.15           ,-0.75          ,0             ,0.75          ,-0.15         ,0.0166666666667}   =  {-1   , 9   , -45  , 0    , 45   , -9  ,   1} / 60
    // accuracy order 6: {0.0166666666667 ,-0.133333333333,0.5            ,-1.33333333333,0.583333333333,0.4           ,-0.0333333333333}     {1.   , -8. , 30.  , -80. , 35.  , 24. , -2.} / 60
    // accuracy order 6: {-0.0333333333333,0.25           ,-0.833333333333,1.66666666667 ,-2.5          ,1.28333333333 ,0.166666666667}       {-2.  , 15. , -50. , 100. , -150., 77. , 10.} / 60
    // accuracy order 6: {0.166666666667  ,-1.2           ,3.75           ,-6.66666666667,7.5           ,-6            ,2.45}                 {10.  , -72., 225. , -400., 450. , -360, 147.} / 60

        else if ( previous_id3 > -1 && previous_id2 > -1 && previous_id1 > -1 && next_id1 > -1     && next_id2 > -1     && next_id3 > -1     )
            return ( -1.0  * previous_value3 + 9.0  * previous_value2  - 45.  * previous_value1                          + 45.  * next_value1     - 9.  * next_value2     + 1.   * next_value3 ) / (60.0*h);
        else if ( previous_id2 > -1 && previous_id1 > -1 && next_id1 > -1     && next_id2 > -1     && next_id3 > -1     && next_id4 > -1     )
            return ( 2.0   * previous_value2 - 24.  * previous_value1  - 35.  * current_value   + 80.  * next_value1     - 30.  * next_value2     + 8.  * next_value3     - 1.   * next_value4 ) / (60.0*h);
        else if ( previous_id4 > -1 && previous_id3 > -1 && previous_id2 > -1 && previous_id1 > -1 && next_id1 > -1     && next_id2 > -1     )
            return ( 1.0   * previous_value4 - 8.   * previous_value3  + 30.  * previous_value2 - 80.  * previous_value1 + 35.  * current_value   + 24. * next_value1     - 2.   * next_value2 ) / (60.0*h);
        else if ( previous_id1 > -1 && next_id1 > -1     && next_id2 > -1     && next_id3 > -1     && next_id4 > -1     && next_id5 > -1     )
            return ( -10.0 * previous_value1 - 77.  * current_value    + 150. * next_value1     - 100. * next_value2     + 50.  * next_value3     - 15. * next_value4     + 2.   * next_value5 ) / (60.0*h);
        else if ( previous_id5 > -1 && previous_id4 > -1 && previous_id3 > -1 && previous_id2 > -1 && previous_id1 > -1 && next_id1 > -1     )
            return ( -2.0  * previous_value5 + 15.  * previous_value4  - 50.  * previous_value3 + 100. * previous_value2 - 150. * previous_value1 + 77. * current_value   + 10.  * next_value1 ) / (60.0*h);
        else if ( next_id1 > -1     && next_id2 > -1     && next_id3 > -1     && next_id4 > -1     && next_id5 > -1     && next_id6 > -1     )
            return ( -147. * current_value   + 360. * next_value1      - 450. * next_value2     + 400. * next_value3     - 225. * next_value4     + 72. * next_value5     - 10.  * next_value6 ) / (60.0*h);
        else if ( previous_id6 > -1 && previous_id5 > -1 && previous_id4 > -1 && previous_id3 > -1 && previous_id2 > -1 && previous_id1 > -1 )
            return ( 10.0  * previous_value6 - 72.  * previous_value5  + 225. * previous_value4 - 400. * previous_value3 + 450. * previous_value2 - 360 * previous_value1 + 147. * current_value ) / (60.0*h);

    // equidistant, using 5 points
    // {-2.08333333333,4,-3,1.33333333333,-0.25}                               {-25.,  48, -36, 16., -3.} / 12
    // {-0.25,-0.833333333333,1.5,-0.5,0.0833333333333}                        { -3.,-10., 18., -6.,  1.} / 12
    // {0.0833333333333,-0.666666666667,0,0.666666666667,-0.0833333333333} =   {  1., -8.,   0,  8., -1.} / 12
    // {-0.0833333333333,0.5,-1.5,0.833333333333,0.25}                         { -1.,  6.,-18., 10.,  3.} / 12
    // {0.25,-1.33333333333,3,-4,2.08333333333}                                {  3.,-16.,  36, -48, 25.} / 12

        else if ( previous_id2 > -1 && previous_id1 > -1 && next_id1 > -1     && next_id2 > -1     )
            return (   1. * previous_value2 - 8.  * previous_value1                         +  8. * next_value1     - 1.  * next_value2   ) / (12.0*h);
        else if ( previous_id1 > -1 && next_id1 > -1     && next_id2 > -1     && next_id3 > -1     )
            return (  -3. * previous_value1 - 10. * current_value   + 18. * next_value1     - 6.  * next_value2     + 1.  * next_value3   ) / (12.0*h);
        else if ( previous_id3 > -1 && previous_id2 > -1 && previous_id1 > -1 && next_id1 > -1     )
            return (  -1. * previous_value3 +  6. * previous_value2 - 18. * previous_value1 + 10. * current_value   + 3.  * next_value1   ) / (12.0*h);
        else if ( next_id1 > -1     && next_id2 > -1     && next_id3 > -1     && next_id4 > -1     )
            return ( -25. * current_value   + 48. * next_value1     - 36. * next_value2     + 16. * next_value3     - 3.  * next_value4   ) / (12.0*h);
        else if ( previous_id4 > -1 && previous_id3 > -1 && previous_id2 > -1 && previous_id1 > -1 )
            return ( 3.0  * previous_value4 - 16. * previous_value3 + 36. * previous_value2 - 48. * previous_value1 + 25. * current_value ) / (12.0*h);

#endif

    }
    else if(grid_type == 2){ // lobatto

//        static_assert(NLIP==7, "nlip should be 7");

        if (fin_diff_order == 9){
        
 // NLIP x central
 // 9 x aligned
 // -1 overlap

        }
        else if (fin_diff_order == 7){
        
            if ( previous_id3 > -1 && previous_id2 > -1 && previous_id1 > -1 && next_id1 > -1 && next_id2 > -1 && next_id3 > -1 ){
                // some of the following 7 rules are slightly subobtimal in the
                // sense that an asymmetric choice of points would have a smaller
                // error, but the problem is small and this is easier
                // generated using https://github.com/lnw/finite-diff-weights
                // assuming a the lobatto grid with points {-3,-2.4906716888357,-1.40654638041214,0,1.40654638041214,2.4906716888357,3}
                // centre at pos 0: {-0.00194396911544, 0.0497395221528, -1.1258469276,   0,               1.1258469276,   -0.0497395221528, 0.00194396911544}
                // centre at pos 1: {-0.017112093698,   0.552355362652,  -2.5886815899,   1.84012169934,   0.231190709888, -0.0193439369289, 0.0014698486431}
                // centre at pos 2: {-0.235893409259,   1.17161898258,   -1.89978168681,  0.702495577585,  0.308223373129, -0.0549857889581, 0.00832295173252}
                // centre at pos 3: {-0.104166666667,   0.302514823756,  -0.668989746863, 0,               0.668989746863, -0.302514823756,  0.104166666667}
                // centre at pos 4: {-0.00832295173252, 0.0549857889581, -0.308223373129, -0.702495577585, 1.89978168681,  -1.17161898258,   0.235893409259}
                // centre at pos 5: {-0.0014698486431,  0.0193439369289, -0.231190709888, -1.84012169934,  2.5886815899,   -0.552355362652,  0.017112093698}
                // centre at pos 6: {-0.00194396911544, 0.0497395221528, -1.1258469276,   0,               1.1258469276,   -0.0497395221528, 0.00194396911544}

                switch(local_pos){
                    case(0):
                        return ( -0.00194396911544 * previous_value3 + 0.0497395221528 * previous_value2 + -1.1258469276   * previous_value1 + 0               * current_value + 1.1258469276   * next_value1 + -0.0497395221528 * next_value2 + 0.00194396911544 * next_value3 )/h;
                    case(1):
                        return ( -0.017112093698   * previous_value3 + 0.552355362652  * previous_value2 + -2.5886815899   * previous_value1 + 1.84012169934   * current_value + 0.231190709888 * next_value1 + -0.0193439369289 * next_value2 + 0.0014698486431  * next_value3 )/h;
                    case(2):
                        return ( -0.235893409259   * previous_value3 + 1.17161898258   * previous_value2 + -1.89978168681  * previous_value1 + 0.702495577585  * current_value + 0.308223373129 * next_value1 + -0.0549857889581 * next_value2 + 0.00832295173252 * next_value3 )/h;
                    case(3):
                        return ( -0.104166666667   * previous_value3 + 0.302514823756  * previous_value2 + -0.668989746863 * previous_value1 + 0               * current_value + 0.668989746863 * next_value1 + -0.302514823756  * next_value2 + 0.104166666667   * next_value3 )/h;
                    case(4):
                        return ( -0.00832295173252 * previous_value3 + 0.0549857889581 * previous_value2 + -0.308223373129 * previous_value1 + -0.702495577585 * current_value + 1.89978168681  * next_value1 + -1.17161898258   * next_value2 + 0.235893409259   * next_value3 )/h;
                    case(5):
                        return ( -0.0014698486431  * previous_value3 + 0.0193439369289 * previous_value2 + -0.231190709888 * previous_value1 + -1.84012169934  * current_value + 2.5886815899   * next_value1 + -0.552355362652  * next_value2 + 0.017112093698   * next_value3 )/h;
                }
            }
            // generated using https://github.com/lnw/finite-diff-weights
            // assuming a the lobatto grid with points {-3,-2.4906716888357,-1.40654638041214,0,1.40654638041214,2.4906716888357,3}
            // centre at pos 0: {-3.5,            4.73385886764,   -1.88966174185, 1.06666666667,   -0.683321604359, 0.439124478567,  -0.166666666667}
            // centre at pos 1: {-0.814308671415, 0,               1.1519427381,   -0.532868896033, 0.320446599096,  -0.200749059879, 0.0755372901318}
            // centre at pos 2: {0.208418888505,  -0.738601427723, 0,              0.755566029029,  -0.355480634669, 0.205463611839,  -0.0753664669809}
            else if (previous_id1 == -1)
                return ( -3.5            * current_value   + 4.73385886764   * next_value1     + -1.88966174185 * next_value2   + 1.06666666667   * next_value3 + -0.683321604359 * next_value4 + 0.439124478567  * next_value5 + -0.166666666667  * next_value6)/h;
            else if (previous_id2 == -1)
                return ( -0.814308671415 * previous_value1 + 0               * current_value   + 1.1519427381   * next_value1   + -0.532868896033 * next_value2 + 0.320446599096  * next_value3 + -0.200749059879 * next_value4 +  0.0755372901318 * next_value5)/h;
            else if (previous_id3 == -1)
                return ( 0.208418888505  * previous_value2 + -0.738601427723 * previous_value1 + 0              * current_value + 0.755566029029  * next_value1 + -0.355480634669 * next_value2 + 0.205463611839  * next_value3 + -0.0753664669809 * next_value4)/h;
            // generated using https://github.com/lnw/finite-diff-weights
            // assuming a the lobatto grid with points {-3,-2.4906716888357,-1.40654638041214,0,1.40654638041214,2.4906716888357,3}
            // centre at pos 4: {0.0753664669809,  -0.205463611839, 0.355480634669,  -0.755566029029, 0,             0.738601427723, -0.208418888505}
            // centre at pos 5: {-0.0755372901318, 0.200749059879,  -0.320446599096, 0.532868896033,  -1.1519427381, 0,              0.814308671415}
            // centre at pos 6: {0.166666666667,   -0.439124478567, 0.683321604359,  -1.06666666667,  1.88966174185, -4.73385886764, 3.5}
            else if (next_id1 == -1)
                return ( 0.166666666667   * previous_value6 + -0.439124478567 * previous_value5 + 0.683321604359  * previous_value4 + -1.06666666667  * previous_value3 + 1.88966174185 * previous_value2 + -4.73385886764 * previous_value1 + 3.5             * current_value)/h;
            else if (next_id2 == -1)
                return ( -0.0755372901318 * previous_value5 + 0.200749059879  * previous_value4 + -0.320446599096 * previous_value3 +  0.532868896033 * previous_value2 + -1.1519427381 * previous_value1 + 0              * current_value   + 0.814308671415  * next_value1)/h;
            else if (next_id3 == -1)
                return (  0.0753664669809 * previous_value4 + -0.205463611839 * previous_value3 + 0.355480634669  * previous_value2 + -0.755566029029 * previous_value1 + 0             * current_value   + 0.738601427723 * next_value1     + -0.208418888505 * next_value2)/h;

        }

    }

    return 0.0;
}

/*
 * Evaluate cube gradients at grid points for simple equidistant grid.  The results are stored to 'device_gradients'.
 *
 */

template <int fin_diff_order, bool evaluate_gradients_x, bool evaluate_gradients_y, bool evaluate_gradients_z>
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

    const int gridtype_x = grid->axis[X_]->grid_type,
              gridtype_y = grid->axis[Y_]->grid_type,
              gridtype_z = grid->axis[Z_]->grid_type;

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
    const int local_pos_x = x%(NLIP-1);
    const int local_pos_y = y%(NLIP-1);
    const int local_pos_z = (z+slice_offset)%(NLIP-1);


    // evaluate gradient to z direction
    if (evaluate_gradients_z) {
        int previous_id1 = -1, previous_id2 = -1, previous_id3 = -1, previous_id4 = -1, previous_id5 = -1, previous_id6 = -1,
            next_id1 = -1, next_id2 = -1, next_id3 = -1, next_id4 = -1, next_id5 = -1, next_id6 = -1;
        if (/*fin_diff_order >= 3 &&*/ z + slice_offset -1 >= 0) {
            previous_id1 = getCubeOffset3D(x, y, z+slice_offset-1, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 3 &&*/ z + slice_offset +1 < grid->shape[Z_]) {
            next_id1 = getCubeOffset3D(x, y, z+slice_offset+1, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 3 &&*/ z + slice_offset -2 >= 0) {
            previous_id2 = getCubeOffset3D(x, y, z+slice_offset-2, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 3 &&*/ z + slice_offset +2 < grid->shape[Z_]) {
            next_id2 = getCubeOffset3D(x, y, z+slice_offset+2, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 5 &&*/ z + slice_offset -3 >= 0) {
            previous_id3 = getCubeOffset3D(x, y, z+slice_offset-3, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 5 &&*/ z + slice_offset +3 < grid->shape[Z_]) {
            next_id3 = getCubeOffset3D(x, y, z+slice_offset+3, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 7 &&*/ z + slice_offset -4 >= 0) {
            previous_id4 = getCubeOffset3D(x, y, z+slice_offset-4, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 7 &&*/ z + slice_offset +4 < grid->shape[Z_]) {
            next_id4 = getCubeOffset3D(x, y, z+slice_offset+4, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 7 &&*/ z + slice_offset -5 >= 0) {
            previous_id5 = getCubeOffset3D(x, y, z+slice_offset-5, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 7 &&*/ z + slice_offset +5 < grid->shape[Z_]) {
            next_id5 = getCubeOffset3D(x, y, z+slice_offset+5, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 7 &&*/ z + slice_offset -6 >= 0) {
            previous_id6 = getCubeOffset3D(x, y, z+slice_offset-6, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 7 &&*/ z + slice_offset +6 < grid->shape[Z_]) {
            next_id6 = getCubeOffset3D(x, y, z+slice_offset+6, device_pitch, device_shape_y);
        }
        const double value = evaluate_derivative(id, previous_id1, previous_id2, previous_id3, previous_id4, previous_id5, previous_id6,
                                                 next_id1, next_id2, next_id3, next_id4, next_id5, next_id6,
                                                 device_cube, fin_diff_order, gridtype_z, local_pos_z, h_z);
        if (valid_point) device_gradients_z[local_id] = multiplier * value;
    }

    // evaluate gradient to y direction
    if (evaluate_gradients_y) {
        int previous_id1 = -1, previous_id2 = -1, previous_id3 = -1, previous_id4 = -1, previous_id5 = -1, previous_id6 = -1,
            next_id1 = -1, next_id2 = -1, next_id3 = -1, next_id4 = -1, next_id5 = -1, next_id6 = -1;
        if (/*fin_diff_order >= 3 &&*/ y -1 >= 0) {
            previous_id1 = getCubeOffset3D(x, y-1, z+slice_offset, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 3 &&*/ y + 1 < grid->shape[Y_]) {
            next_id1 = getCubeOffset3D(x, y+1, z+slice_offset, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 5 &&*/ y - 2 >= 0) {
            previous_id2 = getCubeOffset3D(x, y-2, z+slice_offset, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 5 &&*/ y + 2 < grid->shape[Y_]) {
            next_id2 = getCubeOffset3D(x, y+2, z+slice_offset, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 7 &&*/ y - 3 >= 0) {
            previous_id3 = getCubeOffset3D(x, y-3, z+slice_offset, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 5 &&*/ y + 3 < grid->shape[Y_]) {
            next_id3 = getCubeOffset3D(x, y+3, z+slice_offset, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 7 &&*/ y - 4 >= 0) {
            previous_id4 = getCubeOffset3D(x, y-4, z+slice_offset, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 7 &&*/ y + 4 < grid->shape[Y_]) {
            next_id4 = getCubeOffset3D(x, y+4, z+slice_offset, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 7 &&*/ y - 5 >= 0) {
            previous_id5 = getCubeOffset3D(x, y-5, z+slice_offset, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 7 &&*/ y + 5 < grid->shape[Y_]) {
            next_id5 = getCubeOffset3D(x, y+5, z+slice_offset, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 7 &&*/ y - 6 >= 0) {
            previous_id6 = getCubeOffset3D(x, y-6, z+slice_offset, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 7 &&*/ y + 6 < grid->shape[Y_]) {
            next_id6 = getCubeOffset3D(x, y+6, z+slice_offset, device_pitch, device_shape_y);
        }
        const double value = evaluate_derivative(id, previous_id1, previous_id2, previous_id3, previous_id4, previous_id5, previous_id6,
                                                 next_id1, next_id2, next_id3, next_id4, next_id5, next_id6,
                                                 device_cube, fin_diff_order, gridtype_y, local_pos_y, h_y);
        if (valid_point) device_gradients_y[local_id] = multiplier * value;
    }

    // evaluate gradient to x direction
    // fin_diff_order tests outcommented, because fin_diff_order=7 is hardcoded all over the place anyway (lnw)
    if (evaluate_gradients_x) {
        int previous_id1 = -1, previous_id2 = -1, previous_id3 = -1, previous_id4 = -1, previous_id5 = -1, previous_id6 = -1,
            next_id1 = -1, next_id2 = -1, next_id3 = -1, next_id4 = -1, next_id5 = -1, next_id6 = -1;
        if (/*fin_diff_order >= 3 &&*/ x -1 >= 0) {
            previous_id1 = getCubeOffset3D(x-1, y, z+slice_offset, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 3 &&*/ x + 1 < grid->shape[X_]) {
            next_id1 = getCubeOffset3D(x+1, y, z+slice_offset, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 5 &&*/ x - 2 >= 0) {
            previous_id2 = getCubeOffset3D(x-2, y, z+slice_offset, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 5 &&*/ x + 2 < grid->shape[X_]) {
            next_id2 = getCubeOffset3D(x+2, y, z+slice_offset, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 5 &&*/ x - 3 >= 0) {
            previous_id3 = getCubeOffset3D(x-3, y, z+slice_offset, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 5 &&*/ x + 3 < grid->shape[X_]) {
            next_id3 = getCubeOffset3D(x+3, y, z+slice_offset, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 7 &&*/ x - 4 >= 0) {
            previous_id4 = getCubeOffset3D(x-4, y, z+slice_offset, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 7 &&*/ x + 4 < grid->shape[X_]) {
            next_id4 = getCubeOffset3D(x+4, y, z+slice_offset, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 7 &&*/ x - 5 >= 0) {
            previous_id5 = getCubeOffset3D(x-5, y, z+slice_offset, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 7 &&*/ x + 5 < grid->shape[X_]) {
            next_id5 = getCubeOffset3D(x+5, y, z+slice_offset, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 7 &&*/ x - 6 >= 0) {
            previous_id6 = getCubeOffset3D(x-6, y, z+slice_offset, device_pitch, device_shape_y);
        }

        if (/*fin_diff_order >= 7 &&*/ x + 6 < grid->shape[X_]) {
            next_id6 = getCubeOffset3D(x+6, y, z+slice_offset, device_pitch, device_shape_y);
        }
        const double value = evaluate_derivative(id, previous_id1, previous_id2, previous_id3, previous_id4, previous_id5, previous_id6,
                                                 next_id1, next_id2, next_id3, next_id4, next_id5, next_id6,
                                                 device_cube, fin_diff_order, gridtype_x, local_pos_x, h_x);
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
    cudaThreadSynchronize();
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
void BubblesEvaluator::evaluateGrid(Grid3D *grid, CudaCube *result_cube, CudaCube *gradient_cube_x, CudaCube *gradient_cube_y, CudaCube *gradient_cube_z, int gradient_direction) {

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
                                 int gradient_direction) {
    check_eval_errors(__FILE__, __LINE__);
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
                const int fin_diff_order = 7;
                result_cube->getLaunchConfiguration(&launch_grid, &block, slice_count, BLOCK_SIZE);

                // call the kernel
                if (gradient_direction == X_) {
                    CubeEvaluator_evaluate_simple_grid_gradients <fin_diff_order, true, false, false>
                        <<< launch_grid, block, 0,
                            *this->streamContainer->getStream(device, stream) >>>
                    //CubeEvaluator_evaluate_grid_gradients <NLIP, true, false, false>
                    //    <<< grid_size, BLOCK_SIZE, 0,
                    //        *this->streamContainer->getStream(device, stream) >>>
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
                else if (gradient_direction == Y_) {
                    CubeEvaluator_evaluate_simple_grid_gradients <fin_diff_order, false, true, false>
                        <<< launch_grid, block, 0,
                            *this->streamContainer->getStream(device, stream) >>>
                    //CubeEvaluator_evaluate_grid_gradients <NLIP, false, true, false>
                    //    <<< grid_size, BLOCK_SIZE, 0,
                    //        *this->streamContainer->getStream(device, stream) >>>
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
                else if (gradient_direction == Z_) {
                    CubeEvaluator_evaluate_simple_grid_gradients <fin_diff_order, false, false, true>
                        <<< launch_grid, block, 0,
                            *this->streamContainer->getStream(device, stream) >>>
                    //CubeEvaluator_evaluate_grid_gradients <NLIP, false, false, true>
                    //    <<< grid_size, BLOCK_SIZE, 0,
                    //        *this->streamContainer->getStream(device, stream) >>>
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
                else if (gradient_direction == 3) {
                    CubeEvaluator_evaluate_simple_grid_gradients <fin_diff_order, true, true, true>
                        <<< launch_grid, block, 0,
                            *this->streamContainer->getStream(device, stream) >>>
                    //CubeEvaluator_evaluate_grid_gradients <NLIP, true, true, true>
                    //    <<< grid_size, BLOCK_SIZE, 0,
                    //        *this->streamContainer->getStream(device, stream) >>>
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

extern "C" void evaluator_evaluate_grid_cuda(Evaluator *evaluator, Grid3D *grid, CudaCube *result_cube, CudaCube *gradient_cube_x, CudaCube *gradient_cube_y, CudaCube *gradient_cube_z, int gradient_direction) {
    evaluator->evaluateGrid(grid, result_cube, gradient_cube_x, gradient_cube_y, gradient_cube_z, gradient_direction);
}

extern "C" void evaluator_evaluate_grid_without_gradients_cuda(Evaluator *evaluator, Grid3D *grid, CudaCube *result_cube) {
    evaluator->evaluateGrid(grid, result_cube, NULL, NULL, NULL, -1);
}

extern "C" void evaluator_evaluate_grid_x_gradients_cuda(Evaluator *evaluator, Grid3D *grid, CudaCube *result_cube) {
    evaluator->evaluateGrid(grid, result_cube, NULL, NULL, NULL, X_);
}

extern "C" void evaluator_evaluate_grid_y_gradients_cuda(Evaluator *evaluator, Grid3D *grid, CudaCube *result_cube) {
    evaluator->evaluateGrid(grid, result_cube, NULL, NULL, NULL, Y_);
}

extern "C" void evaluator_evaluate_grid_z_gradients_cuda(Evaluator *evaluator, Grid3D *grid, CudaCube *result_cube) {
    evaluator->evaluateGrid(grid, result_cube, NULL, NULL, NULL, Z_);
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




