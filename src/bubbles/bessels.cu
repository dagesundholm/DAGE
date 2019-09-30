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
#include "grid.h"
#include "cube.h"
#include "bessels.h"
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


__host__ inline void check_bessels_errors(const char *filename, const int line_number) {
#ifdef CUDA_DEBUG
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      printf("CUDA error at %s:%i: %s\n", filename, line_number, cudaGetErrorString(error));
      exit(-1);
    }
#endif
}

__device__ inline void getXYZ3D(int *x, int *y, int *z) {
    *x = blockIdx.x * blockDim.x + threadIdx.x;
    *y = blockIdx.y * blockDim.y + threadIdx.y;
    *z = blockIdx.z * blockDim.z + threadIdx.z;
}

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

inline __device__ double factorial(int x) {
    int i = 1;
    double reslt = 1;
    
    for (i = 2; i <= x; i++) {
        reslt *= (double)i;
    }
    return reslt;
}

__device__ double SecondModSphBesselCollection_evaluate_small(const double z, const int n, const double scaling_factor) {

    double reslt;
    double prefactor1 = pow(scaling_factor, n) * pow(z, n);
    double prefactor2 = pow(scaling_factor, n) / (pow(-1.0, n) * pow(z, n+1));
    double fact_i;
    long divider = 1, divider2 = 1;
    int i;
    
    for (i = 1; i <= 2*n-1; i=i+2) {
        divider = divider * i;
    }
    
    prefactor1 = prefactor1 / ((double) divider * (2.0*n+1.0));
    prefactor2 = prefactor2 * (double) divider;
    reslt = prefactor1 - prefactor2;
    
    divider = 1;
    divider2 = 1;
    fact_i = 1;
    for (i = 1; i <= TERM_COUNT; i++) {
        divider *=  (2*n + 2*i + 1);
        divider2 *= (-2*n + 2*i - 1);
        fact_i *= (double)i;
        reslt += pow((0.5 * pow(z, 2)), i) / ( fact_i ) *
                     (prefactor1 / ((double)divider) - prefactor2 / ( (double)divider2 ) );
    }
    reslt *= M_PI/2.0 * pow(-1.0, n+1);
    return reslt;
}

__device__ double FirstModSphBesselCollection_evaluate_small(const double z, const int n, const double scaling_factor) {
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

__device__ void FirstModSphBesselCollection_evaluate_recursion(const double z, const int address_offset, 
                                                               const double scaling_factor, const int lmax, 
                                                               double *result) {
    // evaluate the results
    int offset = address_offset * lmax;
    int l = 0;
    double previous = 0.0, second_previous = 0.0, current = 0.0;
    
    // check if z is zero, then the first will be 1 and the others are 0,
    // we are assuming that the result array is initialized to 0.
    if (z < 1e-12) {
        result[0] = 1.0;
        return;
    }
    
    // get the two largest l values that are evaluated and store them to second_previous and previous
    second_previous = FirstModSphBesselCollection_evaluate_small(z, lmax, scaling_factor);
    result[offset] = second_previous;
    offset -= address_offset;
    previous = FirstModSphBesselCollection_evaluate_small(z, lmax-1, scaling_factor);
    result[offset] = previous;
    offset -= address_offset;
    
    // do downward recursion for the rest of the l values
    for (l = lmax-2; l >= 0; l--) {
        current = (2*l+3) * previous / z * scaling_factor 
                   + second_previous * scaling_factor * scaling_factor;
        second_previous = previous;
        previous = current;
        result[offset] = current;
        offset -= address_offset;
    }
    return;
}

__global__ void FirstModSphBesselCollection_evaluate_grid(const int shape_x, const int shape_y, const int shape_z,
                                                          const double *gridpoints_x, const double *gridpoints_y, const double *gridpoints_z,
                                                          const int lmin, const int lmax,
                                                          const double kappa, double scaling_factor,
                                                          double *cubes, size_t pitch,
                                                          const double zero_point_x, const double zero_point_y, const double zero_point_z,
                                                          const int slice_offset,
                                                          // the number of slices handled by this kernel call
                                                          const int slice_count,
                                                          // the number of slices that resides in the memory of this device
                                                          const int device_slice_count) {    
    
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
            
        

        FirstModSphBesselCollection_evaluate_recursion(kappa*distance, 
                                                       (int) pitch / sizeof(double) * shape_y * device_slice_count,
                                                       scaling_factor,
                                                       lmax,
                                                       &cubes[id]);
        
    } 
    return;
}

__global__ void SecondModSphBesselCollection_evaluate_grid_kernel(Grid3D *grid,
                                                               double *cubes, 
                                                               int lmax,
                                                               const double kappa,
                                                               const double scaling_factor,
                                                               const double zero_point_x,
                                                               const double zero_point_y,
                                                               const double zero_point_z) {
    
    // get the id of the point (We are using only the first )
    const int id=threadIdx.x + blockIdx.x * blockDim.x;
    
    const int shape_x = grid->shape[X_];
    const int shape_y = grid->shape[Y_];
    const int shape_z = grid->shape[Z_];
    
    // The result array will be in fortran with indices l, x, y, z. 
    // This means that the x index will be the fastest to change.
    const int z = id / (shape_x * shape_y);
    const int y = (id - z * shape_x * shape_y) / (shape_x);
    const int x = (id - z * shape_x * shape_y - y * shape_x); 
    
    int l = 0;
    int i = 0;
                        
    // Check that the point is within the block 
    if (x < shape_x &&
        y < shape_y &&
        z < shape_z) {
        // get pointer to the result array value we are evaluating
        const int address =   z * (shape_x * shape_y)
                      + y * (shape_x)
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
                      grid->axis[X_]->gridpoints[x],
                      grid->axis[Y_]->gridpoints[y],
                      grid->axis[Z_]->gridpoints[z]);
        
        i = 0;
        if (distance > 1e-12) {
            // evaluate the results
            for (l = 0; l <= lmax; l++) {
                cubes[address+i] = SecondModSphBesselCollection_evaluate_small(kappa * distance, l, scaling_factor);
                i = i+shape_x*shape_y*shape_z;
            }
            
        }
        else {
            // evaluate the results
            cubes[address] = 0.0;
            i = shape_x*shape_y*shape_z;
            for (l = 1; l <= lmax; l++) {
                cubes[address+i] = 0.0;
                i = i+shape_x*shape_y*shape_z;
            }
        }
    } 
    return;

}



/*************************************************
 * Host Bessels functionality              *
 * ***********************************************/

void Bessels::initBessels(int lmin, int lmax, StreamContainer *streamContainer) {
    this->lmin = lmin;
    this->lmax = lmax;
    this->streamContainer = streamContainer;
}
    
/*************************************************
 * Host ModifiedSphericalBessels functionality   *
 * ***********************************************/


void ModifiedSphericalBessels::setKappa(double kappa) {
    this->kappa = kappa;
}


void ModifiedSphericalBessels::initModifiedSphericalBessels(int lmin, int lmax, StreamContainer *streamContainer) {
    this->initBessels(lmin, lmax, streamContainer);
    this->scaling_factor = 1.0;
}


/*************************************************
 * Host ModifiedSphericalCubeBessels functionality   *
 * ***********************************************/

void ModifiedSphericalCubeBessels::initModifiedSphericalCubeBessels(int lmin, int lmax, int shape[3], StreamContainer *streamContainer) {
    this->initModifiedSphericalBessels(lmin, lmax, streamContainer);
    // allocate space for device cube pointers
    this->device_results = new double*[this->streamContainer->getNumberOfDevices()];
    this->device_pitches = new size_t[this->streamContainer->getNumberOfDevices()];
    this->device_copies = new Bessels*[this->streamContainer->getNumberOfDevices()];
    
    // copy the shape
    this->shape[X_] = shape[X_];
    this->shape[Y_] = shape[Y_];
    this->shape[Z_] = shape[Z_];
    
    // the limits of the lmax array
    
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        // set the correct GPU
        this->streamContainer->setDevice(device);
        cudaPitchedPtr pointer;
        
        // get the portion that is handled by device with order number 'device'
        int device_slice_count =  shape[Z_] / this->streamContainer->getNumberOfDevices()
                                  + ((shape[Z_] % this->streamContainer->getNumberOfDevices()) > device);
        
        // allocate memory for entire shape for the main pointers
        cudaExtent extent = make_cudaExtent(shape[X_] * sizeof(double), shape[Y_], device_slice_count * (lmax+1) );
        cudaMalloc3D (&pointer, extent);
        
        check_bessels_errors(__FILE__, __LINE__);
        this->device_pitches[device] = pointer.pitch;
        this->device_results[device] = (double *) pointer.ptr;
        
        // allocate the device memory and copy
        cudaMalloc(&this->device_copies[device], sizeof(*this));
        cudaMemcpy(this->device_copies[device], this, sizeof(*this), cudaMemcpyHostToDevice);
        check_bessels_errors(__FILE__, __LINE__);
    }
}


void ModifiedSphericalCubeBessels::evaluate(Grid3D *grid, double center[3]) {
    int slice_offset = 0;
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        // set the correct GPU
        this->streamContainer->setDevice(device);
        
        // and get the corresponding cube pointer and pitch
        size_t device_pitch = this->getDevicePitch(device);
        double *device_results = this->getDeviceResults(device);
        
        // get the portion that is handled by device with order number 'device'
        int device_slice_count =  this->shape[Z_] / this->streamContainer->getNumberOfDevices()
                                  + ((this->shape[Z_] % this->streamContainer->getNumberOfDevices()) > device);
                                  
        for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream ++) {
            // get the portion that is handled by stream with order number 'stream'
            int slice_count = device_slice_count / this->streamContainer->getStreamsPerDevice() 
                                + ((device_slice_count % this->streamContainer->getStreamsPerDevice()) > stream);
                    
            // do the per-stream evaluation (NOTE: the functions hide the kernel calls to the spherical harmonics / solid harmonics)
            this->evaluateSingleStream(device_results, device_pitch, device, grid, center, 
                                       slice_count, device_slice_count, slice_offset, 
                                       this->streamContainer->getStream(device, stream));
            check_bessels_errors(__FILE__, __LINE__);                   
            
            // add to the slice_offset
            slice_offset += slice_count;
            
            // add to the cube pointer
            device_results += device_pitch / sizeof(double) * this->shape[Y_] * slice_count;
        }
    }
}

void ModifiedSphericalCubeBessels::destroy() {
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        cudaFree(this->device_results[device]);
        cudaFree(this->device_copies[device]);
    }
    
    delete[] this->device_results;
    delete[] this->device_pitches;
    delete[] this->device_copies;
}


void ModifiedSphericalCubeBessels::registerHostResultArray(double *host_results, int host_results_shape[4]) {
    
    // register host memory for download
    cudaHostRegister(host_results, host_results_shape[0]*host_results_shape[1]*host_results_shape[2]*host_results_shape[3] * sizeof(double), cudaHostRegisterPortable);
    check_bessels_errors(__FILE__, __LINE__); 
}

void ModifiedSphericalCubeBessels::unregisterHostResultArray(double *host_results) {
    // unregister host memory
    cudaHostUnregister(host_results);
    check_bessels_errors(__FILE__, __LINE__); 
}

/*
 * Note: works best if the host_results is registered before using this method
 * 
 * @param host_results pointer to a four dimensional array of shape (shape[X_], shape[Y_], shape[Z_], lmax)
 * @param host_results_shape (x, y, z, l)
 */
void ModifiedSphericalCubeBessels::download(double *host_results, int host_results_shape[4]) {
    
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
            int l_offset = 0;
            int l_device_offset = 0;
            
            for (int l = this->lmin; l <= this->lmax; l++) {
                cudaMemcpy3DParms memCopyParameters = {0};
                memCopyParameters.dstPtr = make_cudaPitchedPtr(&host_results[l_offset],          host_results_shape[X_]*sizeof(double),  host_results_shape[X_],  host_results_shape[Y_]);
                memCopyParameters.srcPtr = make_cudaPitchedPtr(&device_results[l_device_offset], device_pitch, shape[X_], shape[Y_]);
                memCopyParameters.extent = make_cudaExtent(this->shape[X_] * sizeof(double), this->shape[Y_], slice_count);
                memCopyParameters.kind   = cudaMemcpyDeviceToHost;
                
                // copy the f1 cube to device: 3D
                cudaMemcpy3DAsync(&memCopyParameters, 
                                *this->streamContainer->getStream(device, stream));
                check_bessels_errors(__FILE__, __LINE__); 
                
                // add to the offsets caused by the l
                l_offset        += this->shape[X_] * this->shape[Y_] * this->shape[Z_];
                l_device_offset += device_pitch / sizeof(double) * this->shape[1] * device_slice_count;
                
            }
            
            // add to the result pointers    
            host_results += slice_count *  host_results_shape[X_] * host_results_shape[Y_];
            device_results += device_pitch / sizeof(double) * this->shape[Y_] * slice_count;
        }
    }
}

int *ModifiedSphericalCubeBessels::getShape() {
    return this->shape;
}

double **ModifiedSphericalCubeBessels::getDeviceResults() {
    return this->device_results;
}

double *ModifiedSphericalCubeBessels::getDeviceResults(int device) {
    return this->device_results[device];
}

size_t *ModifiedSphericalCubeBessels::getDevicePitches() {
    return this->device_pitches;   
}

size_t ModifiedSphericalCubeBessels::getDevicePitch(int device) {
    return this->device_pitches[device];
}

/********************************************************
 * Host FirstModifiedSphericalCubeBessels functionality *
 ********************************************************/

FirstModifiedSphericalCubeBessels::FirstModifiedSphericalCubeBessels(int lmin, int lmax, int shape[3], StreamContainer *streamContainer) {
    this->initModifiedSphericalCubeBessels(lmin, lmax, shape, streamContainer);
}

void FirstModifiedSphericalCubeBessels::evaluateSingleStream(double *device_results, size_t device_pitch, int device,
                                                         Grid3D *grid3d, double center[3], 
                                                         int slice_count, int device_slice_count, int slice_offset, cudaStream_t *stream) {
    // get the launch configuration
    dim3 grid, block;
    getCubeLaunchConfiguration(&grid, &block, this->shape, slice_count, 256);
    
    // call the kernel
    FirstModSphBesselCollection_evaluate_grid
       <<< grid, block, 0, *stream >>>
       ( grid3d->getShape(X_), grid3d->getShape(Y_), grid3d->getShape(Z_),
         grid3d->axis[X_]->device_gridpoints[device],
         grid3d->axis[Y_]->device_gridpoints[device],
         grid3d->axis[Z_]->device_gridpoints[device],
         this->lmin, this->lmax,
         this->kappa, this->scaling_factor,
         device_results, device_pitch,
         center[X_], center[Y_], center[Z_],
         slice_offset, slice_count, device_slice_count);
    check_bessels_errors(__FILE__, __LINE__);
    
}
/***********************************************************
 * Fortran interfaces - FirstModifiedSphericalCubeBessels  *
 ***********************************************************/

extern "C" FirstModifiedSphericalCubeBessels *firstmodifiedsphericalcubebessels_init_cuda(int lmin, int lmax, int shape[3], StreamContainer *streamContainer) {
    return new FirstModifiedSphericalCubeBessels(lmin, lmax, shape, streamContainer);
}

extern "C" void firstmodifiedsphericalcubebessels_destroy_cuda(FirstModifiedSphericalCubeBessels *bessels) {
    bessels->destroy();
}

extern "C" void firstmodifiedsphericalcubebessels_set_kappa_cuda(FirstModifiedSphericalCubeBessels *bessels, double kappa) {
    bessels->setKappa(kappa);
}


extern "C" void firstmodifiedsphericalcubebessels_evaluate_cuda(FirstModifiedSphericalCubeBessels *bessels, Grid3D *grid, double center[3]) {
    bessels->evaluate(grid, center);
}

extern "C" void firstmodifiedsphericalcubebessels_download_cuda(FirstModifiedSphericalCubeBessels *bessels, double *host_results, int host_results_shape[4]) {
    bessels->download(host_results, host_results_shape);
}

extern "C" void firstmodifiedsphericalcubebessels_register_result_array_cuda(FirstModifiedSphericalCubeBessels *bessels, double *host_results, int host_results_shape[4]) {
    bessels->registerHostResultArray(host_results, host_results_shape);
}

extern "C" void firstmodifiedsphericalcubebessels_unregister_result_array_cuda(FirstModifiedSphericalCubeBessels *bessels, double *host_results, int host_results_shape[4]) {
    bessels->unregisterHostResultArray(host_results);
}
