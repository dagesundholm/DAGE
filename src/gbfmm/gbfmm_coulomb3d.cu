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
#include <stdlib.h>
#include <stdio.h>
#include "../bubbles/grid.h"
#include "../bubbles/streamcontainer.h"
#include "gbfmm_coulomb3d.h"
#include "../bubbles/cube.h"
#include "../bubbles/integrator.h"
#include "../bubbles/spherical_harmonics_cuda.h"
#include "gbfmm_potential_operator.h"
#include "../bubbles/memory_leak_operators.h"
#define X_ 0
#define Y_ 1
#define Z_ 2
#define BLOCK_SIZE 512

__host__ inline void check_coulomb_errors(const char *filename, const int line_number) {
#ifdef DEBUG_CUDA
    cudaDeviceSynchronize();
#endif
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      printf("CUDA error at %s:%i: %s\n", filename, line_number, cudaGetErrorString(error));
      exit(-1);
    }
    
}

/* MISC device functions */

inline __device__ void calculate_distance_vector(double &dist_vec_x, double &dist_vec_y, double &dist_vec_z, 
                              const double reference_point_x, const double reference_point_y, const double reference_point_z, 
                              const double x, const double y, const double z){
    // calculate the vector relative to reference_point
    dist_vec_x=x-reference_point_x;
    dist_vec_y=y-reference_point_y;
    dist_vec_z=z-reference_point_z;
    return;
}

__device__ inline void getXYZ_(int *x, int *y, int *z) {
    *x = blockIdx.x * blockDim.x + threadIdx.x;
    *y = blockIdx.y * blockDim.y + threadIdx.y;
    *z = blockIdx.z * blockDim.z + threadIdx.z;
}

/*
 * Returns the cube pointer offset caused by the x, y, z coordinates with given pitch and memory shape in y-direction
 */
__device__ inline int getCubeOffset3D_(const int x, const int y, const int z, const size_t pitch, int memory_y_shape) {
    return    z * memory_y_shape * pitch / sizeof(double)
            + y * pitch / sizeof(double)
            + x;
}


/* Kernels and crucial device functions */

__device__ double GBFMMCoulomb3D_evaluate_le_point(const double x,
                                                   const double y,
                                                   const double z,
                                                   const int lmax,
                                                   const double* __restrict__ local_expansion) {
    
    int lm_address =0, address2 = 0;
    int l, m, l2; 
    double top = 0.0, bottom = 0.0, new_bottom = 0.0, prev1 = 0.0, prev2 = 0.0, current = 0.0;
    double r2 = x*x+y*y+z*z;
    // set value for l=0, m=0
    double result = 1.0 * local_expansion[lm_address];
    
    
    // set value for l=1, m=-1
    lm_address += 1;
    result += y * local_expansion[lm_address];
   
    // set all values where m=-1
    m = -1;
    prev1 = y;
    // the starting address has 1 item before from the l=0, 3 from l=1, and 1 from l=2
    address2 = 5;
    for (l = 2; l <= lmax; l++) {
        current =   ( 2.0*(double)l-1.0) / sqrt( 1.0*(double)((l+m)*(l-m)) ) * z*prev1;
        if (l > 2) {
            current -=  sqrt( (double)((l+m-1)*(l-m-1)) /  (double)((l+m)*(l-m)) ) * r2 * prev2;
        }
        prev2 = prev1;
        prev1 = current;
        result += current * local_expansion[address2];
        
        // add the address2 to get to the next item with m=0 
        address2 += (2*l+2);
    }
    
    
    // set value for l=1, m=0
    lm_address += 1;
    result += z * local_expansion[lm_address];
    
    // set all values where m=0
    prev1 = z;
    prev2 = 1.0;
    m = 0;
    // the starting address has 1 item before from the l=0, 3 from l=1, and 2 from l=2
    address2 = 6;
    for (l = 2; l <= lmax; l++) {
        current =   ( 2.0*(double)l-1.0) / sqrt( 1.0*(double)((l+m)*(l-m)) ) * z * prev1;
        current -=  sqrt( (double)((l+m-1)*(l-m-1)) /  (double)((l+m)*(l-m)) ) * r2 * prev2;
        prev2 = prev1;
        prev1 = current;
        result += current * local_expansion[address2];
        
        // add the address2 to get to the next item with m=0 
        address2 += (2*l+2);
    }
    
    // set value for l=1, m=1
    lm_address += 1;
    result += x * local_expansion[lm_address];
    // set all values where m=1
    prev1 = x;
    m = 1;
    // the starting address has 1 item before from the l=0, 3 from l=1, and 3 from l=2
    address2 = 7;
    for (l = 2; l <= lmax; l++) {
        current =   ( 2.0*(double)l-1.0) / sqrt( 1.0*(double)((l+m)*(l-m)) ) * z*prev1;
        if (l > 2) {
            current -=  sqrt( (double)((l+m-1)*(l-m-1)) /  (double)((l+m)*(l-m)) ) * r2 * prev2;
        }
        prev2 = prev1;
        prev1 = current;
        result += current * local_expansion[address2];
        
        // add the address2 to get to the next item with m=0 
        address2 += (2*l+2);
    }
    
    // go through the rest of the stuff
    bottom = y; // bottom refers to solid harmonics value with l=l-1 and m=-(l-1)
    top = x;    // top    refers to solid harmonics value with l=l-1 and m=l-1
    lm_address += 1;
    for (l=2; l <= lmax; l++) {
        
        new_bottom = sqrt((2.0*(double)l - 1.0) / (2.0*(double)l)) * 
                       ( y*top + x*bottom);
        result += new_bottom * local_expansion[lm_address];
        
        // set all values where m=-l
        m = -l;
        prev1 = new_bottom;
        address2 = lm_address + (2*l+2);
        for (l2 = l+1; l2 <= lmax; l2++) {
            current =   ( 2.0*(double)l2-1.0) / sqrt( 1.0*(double)((l2+m)*(l2-m)) ) * z*prev1;
            if (l2 > l+1) {
                current -=  sqrt( (double)((l2+m-1)*(l2-m-1)) /  (double)((l2+m)*(l2-m)) ) * r2 * prev2;
            }
            prev2 = prev1;
            prev1 = current;
            result += current * local_expansion[address2];
            
            // add the address2 to get to the next item with m=l 
            address2 += (2*l2+2);
        }
        
        
        // get value for l=l, m=l. The address is 2*l items away from l=l, m=-l
        lm_address += 2*l;
        top = sqrt((2.0*(double)l - 1.0) / (2.0*(double)l)) * 
                      ( x*top-y*bottom );
        // set all values where m=l
        m = l;
        prev1 = top;
        address2 = lm_address + (2*l+2);
        for (l2 = l+1; l2 <= lmax; l2++) {
            current =   ( 2.0*(double)l2-1.0) / sqrt( 1.0*(double)((l2+m)*(l2-m)) ) * z*prev1;
            if (l2 > l+1) {
                current -=  sqrt( (double)((l2+m-1)*(l2-m-1)) /  (double)((l2+m)*(l2-m)) ) * r2 * prev2;
            }
            prev2 = prev1;
            prev1 = current;
            result += current * local_expansion[address2];
            
            // add the address2 to get to the next item with m=l 
            address2 += (2*l2+2);
        }
        // store the new bottom: l=l, m=-l (we need the old bottom in calculation of top)
        bottom = new_bottom;
        result += top * local_expansion[lm_address];
        
        // get next address
        lm_address += 1;
    }
    return result;
}



/* 
 * Evaluate Local expansion on a grid
 */
__global__ void GBFMMCoulomb3D_evaluate_le_grid(
                              double* __restrict__ cube,
                              int lmax,
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
    getXYZ_(&x, &y, &z);
    
    
    // get the offset from the input cube pointer
    const int id = getCubeOffset3D_(x, y, z, pitch, memory_y_shape);
    
    double value;
    double relative_position_x, relative_position_y, relative_position_z;
                
    //printf("X: %f, cell_spacing: %f, ncell: %d", distance, bubble->cell_spacing, ncell);
    // Check that the point is within the block 
    if (x < shape_x && y < shape_y && z+slice_offset < shape_z && z < slice_count) {
        // calculate relative position to the zero-point and distance to it 
        calculate_distance_vector(relative_position_x,
                                  relative_position_y,
                                  relative_position_z,
                                  zero_point_x, 
                                  zero_point_y,
                                  zero_point_z,
                                  grid_points_x[x],
                                  grid_points_y[y],
                                  grid_points_z[z+slice_offset]);
        
    } 
    
    // calculate the value for local expansion value multiplied with real solid harmonics in Racah's normalization
    value = GBFMMCoulomb3D_evaluate_le_point(relative_position_x,
                                             relative_position_y,
                                             relative_position_z,
                                             lmax,
                                             local_expansion);
    
    // if the point resides within the cube, add the value calculated above to the current value
    if (x < shape_x && y < shape_y && z+slice_offset < shape_z && z < slice_count) {
        cube[id] += value;
    }
    return;

}

/*************************************************** 
 *       GBFMMCoulomb3D implementation             *
 *                                                 *
 ***************************************************/

GBFMMCoulomb3D::GBFMMCoulomb3D(
               // the grid from which the subgrids are extracted from (should represent the entire domain) needed 
               // to evaluate coulomb potential for using gbfmm
               Grid3D *grid_in,
               // the grid from which the subgrids are extracted from (should represent the entire domain) needed 
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
                               input_start_indices_x, input_end_indices_x, input_start_indices_y,
                               input_end_indices_y, input_start_indices_z, input_end_indices_z,
                               output_start_indices_x, output_end_indices_x, output_start_indices_y,
                               output_end_indices_y, output_start_indices_z, output_end_indices_z,
                               streamContainer);
    
    
}

void GBFMMCoulomb3D::initHarmonics() {
    //this->harmonics = new RealCubeHarmonics *[this->streamContainer->getNumberOfDevices()];
    // initialize the solid-harmonics evaluators.
    // NOTE: this assumes that each of the boxes have the same shape and that the multipole center is at the center of the
    // box. If the cube-grid is changed to be non-equidistant at some point, this must be changed to be box-wise.
    //for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        // initialize the solid harmonics
    //    this->harmonics[device] = new RealRegularSolidCubeHarmonics(/*lmin=*/0, this->lmax, /*normalization=Racah's*/1, this->input_grids[0]->getShape(), this->device_containers[device]);

    //}
    //check_coulomb_errors(__FILE__, __LINE__);
    
    // evaluate the solid harmonics on all devices
    //for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
    //    this->harmonics[device]->evaluate(this->device_grids[device], this->centers); 
    //}
    //check_coulomb_errors(__FILE__, __LINE__);
}

void GBFMMCoulomb3D::destroyHarmonics() {
    // destroy the solid harmonics and bessels from all devices
    //for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
    //    this->harmonics[device]->destroy(); 
    //    delete this->harmonics[device];
    //}
    //delete[] this->harmonics;
}

void GBFMMCoulomb3D::initIntegrators() {
    // init the subgrids and the streamcontainers for each domain box
    for (int i = 0; i <= this->domain[1]-this->domain[0]; i++) {
        // initialize the Integrator needed for multipole evaluation with a buffer for (this->lmax+1)*(this->lmax+1) results 
        this->integrators[i] = new GBFMMCoulomb3DMultipoleEvaluator(this->streamContainers[i],
                                                                    this->input_grids[i],
                                                                    this->lmax, 
                                                                    &this->centers[i*3]);
    }
}



/*
 * Downloads multipole moments between l:0-this->lmax for each box belonging
 * to the domain of this node.
 * 
 * NOTE: this functions IS BLOCKING with regards to CUDA
 */
void GBFMMCoulomb3D::downloadMultipoleMoments(double *host_multipole_moments) {
    // do the evaluation for the boxes
    host_multipole_moments = &host_multipole_moments[(domain[0]-1) * (this->lmax+1)*(this->lmax+1)];
    for (int i = 0; i <= this->domain[1]-this->domain[0]; i++) {
        this->integrators[i]->downloadResult(host_multipole_moments);
        host_multipole_moments += (this->lmax+1)*(this->lmax+1);
    }
}


/*
 * Evaluates multipole moments between l:0-this->lmax for box 'i'.
 * To use this function, the box 'i' must belong to the domain of this node.
 * 
 * NOTE: this function IS NOT BLOCKING with regards to CUDA
 */
void GBFMMCoulomb3D::calculateMultipoleMomentsBox(int i, CudaCube *cube) {
    this->integrators[i]->setIntegrationCube(cube);
    this->integrators[i]->integrate();
    this->integrators[i]->setIntegrationCube(NULL);
    //int *cube_device_memory_shape = cube->getDeviceMemoryShape();
    //int *integration_device_memory_shape = this->integrators[i]->getIntegrationCube()->getDeviceMemoryShape();
    //int lm_cube_offset = 0;
    //check_coulomb_errors(__FILE__, __LINE__);
    /*for (int l = 0; l <= this->lmax ; l++) {
        for (int m = -l; m <= l; m++) {
            int * solid_harmonics_memory_shape = this->harmonics[0]->getShape();
            size_t solid_harmonics_pitch = this->harmonics[0]->getDevicePitch(0); 
            
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
                
                // get pointer to the first item of the solid harmonics for this device
                // NOTE: the zeros comes from the fact that there is only one device per streamcontainer of the
                // SolidHarmonics
                double *device_solid_harmonics = this->harmonics[device_order_number]->getDeviceResults(0); 
                
                
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
                    
                    // multiply the solid harmonics with the cube and store to device_temp_cube, i.e.,  "this->integrators[i]->getIntegrationCube()"
                    multiply_3d_cubes(&device_solid_harmonics[lm_cube_offset], cube->getShape(X_), cube->getShape(Y_), solid_harmonics_memory_shape[Y_], solid_harmonics_pitch,
                                    device_cube, cube->getShape(X_), cube->getShape(Y_), cube_device_memory_shape[Y_], device_cube_pitch,
                                    device_temp_cube,  this->integrators[i]->getIntegrationCube()->getShape(X_), this->integrators[i]->getIntegrationCube()->getShape(Y_),
                                    integration_device_memory_shape[Y_], device_temp_pitch,
                                    slice_count, &grid, &block, streamObject);
                    check_coulomb_errors(__FILE__, __LINE__);
                            
                    
                    // add to the pointers
                    device_solid_harmonics += slice_count * solid_harmonics_memory_shape[Y_] * solid_harmonics_pitch / sizeof(double); 
                    device_cube            += slice_count * cube_device_memory_shape[Y_] * device_cube_pitch / sizeof(double); 
                    device_temp_cube       += slice_count * integration_device_memory_shape[Y_] * device_temp_pitch / sizeof(double);
                }
            }
        
        
            // add to the offset caused by the l, m-cubes
            lm_cube_offset += solid_harmonics_memory_shape[Y_] * solid_harmonics_memory_shape[Z_] * solid_harmonics_pitch / sizeof(double);
            
            // start the integration process at the GPUs
            // NOTE: this is not blocking
            // NOTE: the results are stored to the buffer of the integrator 
            this->integrators[i]->integrate();
            
        }
    }*/
    
    
}


/*
 * Evaluates the potential within space of a single box. 
 */
void GBFMMCoulomb3D::evaluatePotentialLEBox(
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
    int *device_memory_shape = output_cube->getDeviceMemoryShape();
    int slice_offset = 0;
    //check_coulomb_errors(__FILE__, __LINE__);
    
    
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
            GBFMMCoulomb3D_evaluate_le_grid<<< grid, block, 0, 
                        *streamContainer->getStream(device, stream) >>>
                             (device_cube,
                              this->lmax,
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
}




/*************************************************** 
 *              Fortran interfaces                 *
 *                                                 *
 ***************************************************/

extern "C" GBFMMCoulomb3D *gbfmmcoulomb3d_init_cuda(
               // the grid from which the subgrids are extracted from (should represent the entire domain) needed 
               // to evaluate coulomb potential for using gbfmm
               Grid3D *grid_in, 
               // the grid from which the subgrids are extracted from (should represent the entire domain) needed 
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
    GBFMMCoulomb3D *new_gbfmm_coulomb3d = new GBFMMCoulomb3D(grid_in, grid_out, lmax, domain,
                                                             input_start_indices_x, input_end_indices_x, 
                                                             input_start_indices_y, input_end_indices_y,
                                                             input_start_indices_z, input_end_indices_z,
                                                             output_start_indices_x, output_end_indices_x, 
                                                             output_start_indices_y, output_end_indices_y,
                                                             output_start_indices_z, output_end_indices_z,
                                                             streamContainer);
    return new_gbfmm_coulomb3d;
}

extern "C" void gbfmmcoulomb3d_init_harmonics_cuda(
               // a pointer to the pre-inited gbfmm coulomb3d operator
               GBFMMCoulomb3D *gbfmm_coulomb3d) {
    gbfmm_coulomb3d->initHarmonics();
}

extern "C" void gbfmmcoulomb3d_destroy_harmonics_cuda(
               // a pointer to the pre-inited gbfmm coulomb3d operator
               GBFMMCoulomb3D *gbfmm_coulomb3d) {
    gbfmm_coulomb3d->destroyHarmonics();
}

extern "C" void gbfmmcoulomb3d_calculate_multipole_moments_cuda(
               // a pointer to the pre-inited gbfmm coulomb3d operator
               GBFMMCoulomb3D *gbfmm_coulomb3d,
               // a pointer to the cube for which the multipole moments are evaluated
               // the boxes needed for multipole are uploaded for 
               CudaCube *input_cube) {
    gbfmm_coulomb3d->calculateMultipoleMoments(input_cube);
}

extern "C" void gbfmmcoulomb3d_download_multipole_moments_cuda(
               // a pointer to the pre-inited gbfmm coulomb3d operator, for which
               // the multipole moments are calculated
               GBFMMCoulomb3D *gbfmm_coulomb3d,
               // a pointer to the 2-dimensional array residing in host memory in which the multipole moments are stored
               double *host_multipole_moments) {
    gbfmm_coulomb3d->downloadMultipoleMoments(host_multipole_moments);
}

extern "C" void gbfmmcoulomb3d_upload_domain_boxes_cuda(
               // a pointer to the pre-inited gbfmm coulomb3d operator, for which
               // the multipole moments are calculated
               GBFMMCoulomb3D *gbfmm_coulomb3d,
               // a pointer to the cube for which the multipole moments are evaluated
               CudaCube *input_cube) { 
    gbfmm_coulomb3d->uploadDomainBoxes(input_cube);
}

extern "C" void gbfmmcoulomb3d_evaluate_potential_le_cuda(
               GBFMMCoulomb3D *gbfmm_coulomb3d, 
               double *local_expansion,
               CudaCube *output_cube) {
    gbfmm_coulomb3d->evaluatePotentialLE(local_expansion, output_cube);
}

extern "C" StreamContainer* gbfmmcoulomb3d_get_box_stream_container_cuda(
               GBFMMCoulomb3D *gbfmm_coulomb3d, 
               int ibox) {
    return gbfmm_coulomb3d->getBoxStreamContainer(ibox);
}

extern "C" void gbfmmcoulomb3d_destroy_cuda(
               // the destroyed gbfmm coulomb3d operator
               GBFMMCoulomb3D *gbfmm_coulomb3d) {
    gbfmm_coulomb3d->destroy();
}
