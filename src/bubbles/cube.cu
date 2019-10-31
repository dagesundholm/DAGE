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
#include "cube.h"
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

#define X_ 0
#define Y_ 1
#define Z_ 2


/*************************************************
 * Kernels  & device functions                   *
 *************************************************/
__device__ inline 
int getGlobalIdx_3D_3D()
{
    int blockId = blockIdx.x
        + blockIdx.y * gridDim.x
        + gridDim.x * gridDim.y * blockIdx.z;
    int threadId
        = blockId * (blockDim.x * blockDim.y * blockDim.z)
        + (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x)
        + threadIdx.x;
    return threadId;
}

__device__ inline
int getGlobalIdx_1D_3D()
{
    return blockIdx.x * blockDim.x * blockDim.y * blockDim.z
        + threadIdx.z * 
        blockDim.y * blockDim.x 
        + threadIdx.y * blockDim.x + threadIdx.x;
}

__device__ inline
void getXYZ3D(int *x, int *y, int *z) {
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

__global__ void sum_3d(double*       __restrict__ cube1, const int cube1_shape_x, const int cube1_shape_y, const int cube1_memory_shape_y, const size_t cube1_pitch,
                       const double* __restrict__ cube2, const int cube2_shape_x, const int cube2_shape_y, const int cube2_memory_shape_y, const size_t cube2_pitch,
                       const int slice_count
                      ) {
    // get the x, y, z coordinates
    int x, y, z;
    getXYZ3D(&x, &y, &z);
    // check that we are within range of the cube
    if (x < cube1_shape_x && y < cube1_shape_y && z < slice_count &&
        x < cube2_shape_x && y < cube2_shape_y && z < slice_count) {
 
        // get the offsets from the input addresses caused by the x, y, z coordinates
        int cube1_offset = getCubeOffset(x, y, z, cube1_pitch, cube1_memory_shape_y);
        int cube2_offset = getCubeOffset(x, y, z, cube2_pitch, cube2_memory_shape_y);
          
        // do the sum
        cube1[cube1_offset] += cube2[cube2_offset];
    }
}

__global__ void sum_3d_with_factor(double* __restrict__ cube1, const int cube1_shape_x, const int cube1_shape_y, const int cube1_memory_shape_y, const size_t cube1_pitch,
                       const double* __restrict__ cube2, const int cube2_shape_x, const int cube2_shape_y, const int cube2_memory_shape_y, const size_t cube2_pitch,
                       const int slice_count, const double factor
                      ) {
    // get the x, y, z coordinates
    int x, y, z;
    getXYZ3D(&x, &y, &z);
    // check that we are within range of the cube
    if (x < cube1_shape_x && y < cube1_shape_y && z < slice_count &&
        x < cube2_shape_x && y < cube2_shape_y && z < slice_count) {
 
        // get the offsets from the input addresses caused by the x, y, z coordinates
        int cube1_offset = getCubeOffset(x, y, z, cube1_pitch, cube1_memory_shape_y);
        int cube2_offset = getCubeOffset(x, y, z, cube2_pitch, cube2_memory_shape_y);
          
        // do the sum
        cube1[cube1_offset] += factor * cube2[cube2_offset];
    }
}

/*
 * Multiply pitched 3d-cubes and store the result to the first input cube 'cube1'.
 */
__global__ void multiply_3d_inplace(double*  __restrict__ cube1, const int cube1_shape_x, const int cube1_shape_y, const int cube1_memory_shape_y, const size_t cube1_pitch,
                                    const double* __restrict__ cube2, const int cube2_shape_x, const int cube2_shape_y, const int cube2_memory_shape_y, const size_t cube2_pitch,
                                    const int slice_count
                                    ) {
    // get the x, y, z coordinates
    int x, y, z;
    getXYZ3D(&x, &y, &z);
    // check that we are within range of the cube
    if (x < cube1_shape_x && y < cube1_shape_y && z < slice_count &&
        x < cube2_shape_x && y < cube2_shape_y && z < slice_count) {
 
        // get the offsets from the input addresses caused by the x, y, z coordinates
        int cube1_offset = getCubeOffset(x, y, z, cube1_pitch, cube1_memory_shape_y);
        int cube2_offset = getCubeOffset(x, y, z, cube2_pitch, cube2_memory_shape_y);
          
        // do the product
        cube1[cube1_offset] *= cube2[cube2_offset];
    }
}

/*
 * Multiply pitched 3d-cubes and store the result to the first input cube 'cube1'.
 */
__global__ void multiply_3d(const double*  __restrict__ cube1, const int cube1_shape_x, const int cube1_shape_y, const int cube1_memory_shape_y, const size_t cube1_pitch,
                       const double* __restrict__ cube2, const int cube2_shape_x, const int cube2_shape_y, const int cube2_memory_shape_y, const size_t cube2_pitch,
                       double* __restrict__ result_cube, const int result_cube_shape_x, const int result_cube_shape_y, const int result_cube_memory_shape_y, const size_t result_cube_pitch,
                       const int slice_count
                      ) {
    // get the x, y, z coordinates
    int x, y, z;
    getXYZ3D(&x, &y, &z);
    // check that we are within range of the cube
    if (x < cube1_shape_x && y < cube1_shape_y && z < slice_count &&
        x < cube2_shape_x && y < cube2_shape_y && z < slice_count) {
 
        // get the offsets from the input addresses caused by the x, y, z coordinates
        int cube1_offset = getCubeOffset(x, y, z, cube1_pitch, cube1_memory_shape_y);
        int cube2_offset = getCubeOffset(x, y, z, cube2_pitch, cube2_memory_shape_y);
        int result_cube_offset = getCubeOffset(x, y, z, result_cube_pitch, result_cube_memory_shape_y);
          
        // do the sum
        result_cube[result_cube_offset] = cube1[cube1_offset] * cube2[cube2_offset];
    }
}


/*************************************************
 * Host Global functionality                     *
 * ***********************************************/


void getCubeLaunchConfiguration(dim3 *grid, dim3 *block, int shape[3], int slice_count, int block_size) {
    block->x = 32;
    block->y = block_size/32;
    block->z = 1;
    grid->x = shape[X_]/32 + ((shape[X_] % 32) > 0);
    grid->y = shape[Y_]/block->y + ((shape[Y_] % block->y) > 0);
    grid->z = slice_count;
    //printf("Grid: %d, %d, %d, Block: %d, %d, %d\n", grid->x, grid->y, grid->z, block->x, block->y, block->z);
}


void multiply_3d_cubes(double*  __restrict__ cube1, const int cube1_shape_x, const int cube1_shape_y, const int cube1_memory_shape_y, const size_t cube1_pitch,
                       const double* __restrict__ cube2, const int cube2_shape_x, const int cube2_shape_y, const int cube2_memory_shape_y, const size_t cube2_pitch,
                       const int slice_count, dim3 *grid, dim3 *block, cudaStream_t *stream) {
    multiply_3d_inplace <<<*grid, *block, 0, *stream>>>
        (cube1, cube1_shape_x, cube1_shape_y, cube1_memory_shape_y, cube1_pitch,
         cube2, cube2_shape_x, cube2_shape_y, cube2_memory_shape_y, cube2_pitch, slice_count);
}

void multiply_3d_cubes(const double*  __restrict__ cube1, const int cube1_shape_x, const int cube1_shape_y, const int cube1_memory_shape_y, const size_t cube1_pitch,
                       const double* __restrict__ cube2, const int cube2_shape_x, const int cube2_shape_y, const int cube2_memory_shape_y, const size_t cube2_pitch,
                       double* __restrict__ result_cube, const int result_cube_shape_x, const int result_cube_shape_y, const int result_cube_memory_shape_y, const size_t result_cube_pitch,
                       const int slice_count, dim3 *grid, dim3 *block, cudaStream_t *stream) {
    multiply_3d <<<*grid, *block, 0, *stream>>>
        (cube1, cube1_shape_x, cube1_shape_y, cube1_memory_shape_y, cube1_pitch,
         cube2, cube2_shape_x, cube2_shape_y, cube2_memory_shape_y, cube2_pitch,
         result_cube, result_cube_shape_x, result_cube_shape_y, result_cube_memory_shape_y, result_cube_pitch,
         slice_count);
}

void sum_3d_cubes_with_factor(double*  __restrict__ cube1, const int cube1_shape_x, const int cube1_shape_y, const int cube1_memory_shape_y, const size_t cube1_pitch,
                       const double* __restrict__ cube2, const int cube2_shape_x, const int cube2_shape_y, const int cube2_memory_shape_y, const size_t cube2_pitch,
                       const int slice_count, dim3 *grid, dim3 *block, double factor, cudaStream_t *stream) {
    sum_3d_with_factor <<<*grid, *block, 0, *stream>>>
        (cube1, cube1_shape_x, cube1_shape_y, cube1_memory_shape_y, cube1_pitch, cube2, cube2_shape_x, cube2_shape_y, cube2_memory_shape_y, cube2_pitch, slice_count, factor);
}

void registerHostCube(double *host_cube, int host_cube_shape[3]) {
    check_errors(__FILE__, __LINE__);
    cudaHostRegister(host_cube, host_cube_shape[0]*host_cube_shape[1]*host_cube_shape[2]*sizeof(double), cudaHostRegisterPortable);
    check_errors(__FILE__, __LINE__);
}

void unregisterHostCube(double *host_cube) {
    cudaHostUnregister(host_cube);
    check_errors(__FILE__, __LINE__);
}

/*************************************************
 * Host Class functionality                      *
 * ***********************************************/

void CudaCube::getLaunchConfiguration(dim3 *grid, dim3 *block, int slice_count, int block_size) {
    block->x = 32;
    block->y = block_size/32;
    block->z = 1;
    grid->x = this->getShape(0)/32 + ((this->getShape(0) % 32) > 0);
    grid->y = this->getShape(1)/block->y + ((this->getShape(1) % block->y) > 0);
    grid->z = slice_count;
    // printf("Grid: %d, %d, %d, Block: %d, %d, %d\n", grid->x, grid->y, grid->z, block->x, block->y, block->z);
}


void CudaCube::initMemorySliced(StreamContainer *streamContainer, int shape[3], bool all_memory_at_all_devices) {
    this->shape[0] = shape[0];
    this->shape[1] = shape[1];
    this->shape[2] = shape[2];
    this->parentCube = NULL;
    this->streamContainer = streamContainer;
    this->sliced = true;
    this->is_slice = false;
    this->is_subcube = false;
    this->slice_dimension = -1;
    this->all_memory_at_all_devices = all_memory_at_all_devices;
    this->downloaded_events = new cudaEvent_t*[this->streamContainer->getNumberOfDevices()];

    // allocate space for device cube pointers
    this->device_cubes = new double*[this->streamContainer->getNumberOfDevices()];
    this->device_pitches = new size_t[this->streamContainer->getNumberOfDevices()];
    check_errors(__FILE__, __LINE__);
    
    if (all_memory_at_all_devices) {
        // allocate space for device cube pointers
        this->device_gather_cubes = new double*[this->streamContainer->getNumberOfDevices()];
        this->device_gather_pitches = new size_t[this->streamContainer->getNumberOfDevices()];
        check_errors(__FILE__, __LINE__);
    }

    for (int device = 0; device < streamContainer->getNumberOfDevices(); device ++) {
        this->streamContainer->setDevice(device);

        int device_slice_count =  this->shape[2]  / streamContainer->getNumberOfDevices()
                                  + ((this->shape[2] % streamContainer->getNumberOfDevices()) > device);
        
        cudaPitchedPtr pointer;

        cudaExtent extent = make_cudaExtent(shape[0] * sizeof(double), shape[1], device_slice_count);
        
        // if all memory is initialized at all devices, we allocate two cubes: one for all 
        // handling and one to allow summation upon download
        if (all_memory_at_all_devices) {
            // allocate memory for entire shape for the main pointers
            cudaExtent device_memory_extent = make_cudaExtent(shape[0] * sizeof(double), shape[1], shape[2]);
            cudaMalloc3D (&pointer, device_memory_extent);
            check_errors(__FILE__, __LINE__);
            this->device_pitches[device] = pointer.pitch;
            this->device_cubes[device] = (double *) pointer.ptr;
            
            // and sliced memory for the gathering
            cudaPitchedPtr pointer2;
            cudaMalloc3D(&pointer2, device_memory_extent);
            this->device_gather_pitches[device] = pointer2.pitch;
            this->device_gather_cubes[device] = (double *) pointer2.ptr;
            check_errors(__FILE__, __LINE__);
            
        }
        else {
            cudaMalloc3D (&pointer, extent);
            check_errors(__FILE__, __LINE__);
            this->device_pitches[device] = pointer.pitch;
            this->device_cubes[device] = (double *) pointer.ptr;
        }
    }
}

void CudaCube::initMemory(StreamContainer *streamContainer, int shape[3], bool all_memory_at_all_devices) {
    this->shape[0] = shape[0];
    this->shape[1] = shape[1];
    this->shape[2] = shape[2];
    this->sliced = false;
    this->is_slice = false;
    this->is_subcube = false;
    this->slice_dimension = -1;
    this->parentCube = NULL;
    this->streamContainer = streamContainer;
    this->all_memory_at_all_devices = all_memory_at_all_devices;

    // allocate space for device cube pointers
    this->device_cubes = new double*[this->streamContainer->getNumberOfDevices()];
    this->device_pitches = new size_t[this->streamContainer->getNumberOfDevices()];

    for (int device = 0; device < streamContainer->getNumberOfDevices(); device ++) {
        this->streamContainer->setDevice(device);

        int device_vector_count =  (this->shape[2] * this->shape[1]) / streamContainer->getNumberOfDevices()
                                  + (((this->shape[2] * this->shape[1]) % streamContainer->getNumberOfDevices()) > device);
        cudaMallocPitch((void**)&device_cubes[device], &device_pitches[device], 
                    shape[0] * sizeof(double), device_vector_count); 
    }
}

/*
 * Sets memory within this cube's shape to zero. 
 * 
 * NOTE: does not set all memory to zero, only the memory residing within
 *       the cube's area. For instance if the pitch of the cube is larger than
 *       occupied by the useful data, the extra data remains unchanged. Also
 *       if this cube is a subcube, the data outside the subcubes shape remains
 *       unchanged.
 */
void CudaCube::setToZero() {
    double *device_cube;
    size_t device_pitch;
    // get the shape of the memory
    int *device_memory_shape = this->getDeviceMemoryShape();
    
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        this->streamContainer->setDevice(device);
        device_cube = this->device_cubes[device];
        device_pitch = this->device_pitches[device];

        // get the number of slices set to zero by this device
        int device_slice_count;
        if (this->all_memory_at_all_devices) {
            device_slice_count = this->shape[2];
        }
        else {
            device_slice_count = this->shape[2] / this->streamContainer->getNumberOfDevices()
                                    + ((this->shape[2] % this->streamContainer->getNumberOfDevices()) > device);
        }
        
        // do the upload in streams
        for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream ++) {
            
            int slice_count = device_slice_count / this->streamContainer->getStreamsPerDevice() 
                                + ((device_slice_count % this->streamContainer->getStreamsPerDevice()) > stream);
                                
            cudaPitchedPtr pitchedPointer = make_cudaPitchedPtr(device_cube, device_pitch, device_memory_shape[0], device_memory_shape[1]);
            cudaExtent extent = make_cudaExtent(this->shape[0] * sizeof(double), this->shape[1], slice_count);
            
            // set the memory to zero: 3D
            cudaMemset3DAsync(pitchedPointer, 0, extent, 
                              *this->streamContainer->getStream(device, stream));
            
            device_cube += slice_count * device_pitch / sizeof(double) * device_memory_shape[1];
        }
    }
}

void CudaCube::setAllMemoryToZero() {
    double *device_cube;
    size_t device_pitch;
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        this->streamContainer->setDevice(device);
        device_cube = this->device_cubes[device];
        device_pitch = this->device_pitches[device];

        // get the number of slices uploaded by the device
        int device_slice_count;
        if (this->all_memory_at_all_devices) {
            device_slice_count = this->shape[2];
        }
        else {
            device_slice_count =  this->shape[2] / this->streamContainer->getNumberOfDevices()
                                    + ((this->shape[2] % this->streamContainer->getNumberOfDevices()) > device);
        }
        
        for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream ++) {
            int slice_count = device_slice_count / this->streamContainer->getStreamsPerDevice() 
                                + ((device_slice_count % this->streamContainer->getStreamsPerDevice()) > stream);
            // copy the f1 cube to device: 3D
            cudaMemsetAsync((void *) device_cube,
                              0,
                              device_pitch * this->shape[1] * slice_count, 
                              *this->streamContainer->getStream(device, stream)
                             );
            check_errors(__FILE__, __LINE__);
            device_cube += slice_count * device_pitch * this->shape[1] / sizeof(double);
        }
    }
}

void CudaCube::initHost(double *host_cube, int host_cube_shape[3], bool register_host) {
    // check if the cube is registered, if is, unregister before 
    // registering new cube
    unsetHostCube();
    check_errors(__FILE__, __LINE__);
    
    this->host_cube = host_cube;
    this->host_pitch = host_cube_shape[0]*sizeof(double);
    this->host_shape[0] = host_cube_shape[0];
    this->host_shape[1] = host_cube_shape[1];
    this->host_shape[2] = host_cube_shape[2];
    
    // register the host cube as pinned memory, if so desired
    if (register_host) {
        int registered_shape[3];
        registered_shape[0] = host_cube_shape[0];
        registered_shape[1] = host_cube_shape[1];
        registered_shape[2] = min(host_cube_shape[2], this->shape[2]);
        registerHostCube(host_cube, registered_shape);
        check_errors(__FILE__, __LINE__);
        this->host_registered = true;
    }
    
    // mark that this cube has a host counterpart
    this->host_initialized = true;
}



/*
 * This method cuts the connection between the device and host memory.
 * 
 * Unregisters the memory, if the cube  is not subcube and the memory has been
 * previously initialized by this cube.
 */
void CudaCube::unsetHostCube() {
    if (this->host_initialized) {
        if (this->host_registered && !this->isSubCube() && !this->isSlice()) {
            check_errors(__FILE__, __LINE__);
            unregisterHostCube(this->host_cube);
            check_errors(__FILE__, __LINE__);
            this->host_registered = false;
        }
        this->host_initialized = false;
        this->host_cube = NULL;
    }
}


/********************************************
 *  Constructors & Destructor               *
 ********************************************/

CudaCube::CudaCube() {
}

CudaCube::CudaCube(StreamContainer *streamContainer, int shape[3], bool all_memory_at_all_devices, bool sliced) {
    if (sliced) {
        this->initMemorySliced(streamContainer, shape, all_memory_at_all_devices);
    }
    else {
        this->initMemory(streamContainer, shape, all_memory_at_all_devices);
    }
    this->host_initialized = false;
    this->host_registered = false;
}

CudaCube::CudaCube(StreamContainer *streamContainer, int shape[3], double *host_cube, int host_cube_shape[3], bool all_memory_at_all_devices, bool register_host, bool sliced) {
    if (sliced) {
        this->initMemorySliced(streamContainer, shape, all_memory_at_all_devices);
    }
    else {
        this->initMemory(streamContainer, shape, all_memory_at_all_devices);
    }
    this->host_initialized = false;
    this->host_registered = false;
    this->initHost(host_cube, host_cube_shape, register_host);
}   


void CudaCube::destroy() {
    if (!this->isSubCube() && !this->isSlice()) {
        for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
            this->streamContainer->setDevice(device);
            // free the device cube
            cudaFree(device_cubes[device]); 
            check_errors(__FILE__, __LINE__);
            
            // and the gather cube, if any
            if (this->all_memory_at_all_devices) {
                cudaFree(device_gather_cubes[device]);
                check_errors(__FILE__, __LINE__);
            }
        }
    }
    // if the host has been registed, then unregister the pointer
    // NOTE: unregister has check for subcube so this call is safe
    this->unsetHostCube();
    check_errors(__FILE__, __LINE__);
    
    delete[] this->downloaded_events;
    // free the device pointer and pitch arrays  
    delete[] this->device_cubes;
    //cudaFreeHost(this->device_cubes);
    delete[] this->device_pitches;
    //cudaFreeHost(this->device_pitches);
    check_errors(__FILE__, __LINE__);
    
     // and the gather arrays, if any
     if (this->all_memory_at_all_devices) {
         //cudaFreeHost(device_gather_cubes);
         delete[] this->device_gather_cubes;
         //cudaFreeHost(device_gather_pitches);
         delete[] this->device_gather_pitches;
         check_errors(__FILE__, __LINE__);
     }
}

/********************************************
 *  Accessors                               *
 ********************************************/

/*
 * Returns the shape of the device cube, taking into
 * account contributions from all devices.
 */
int CudaCube::getShape(int axis) {
    // check if we are handling a slice, if we are,
    // we are omitting the slice dimension
    if (this->isSlice()) {
        if (this->slice_dimension <= axis) {
            return this->shape[axis-1];
        }
        else {
            return this->shape[axis];
        }
    }
    else {
        return this->shape[axis];
    }
}

int *CudaCube::getShape() {
    return this->shape;
}

void CudaCube::setStreamContainer(StreamContainer *streamContainer) {
    this->streamContainer = streamContainer;
}

StreamContainer *CudaCube::getStreamContainer() {
    return this->streamContainer;
}

double ** CudaCube::getDeviceCubes() {
    return this->device_cubes;
}

size_t * CudaCube::getDevicePitches() {
    return this->device_pitches;
}

size_t CudaCube::getDevicePitch(int device) {
    return this->device_pitches[device];
}

size_t CudaCube::getHostPitch() {
    return this->host_pitch;
}

double * CudaCube::getHostCube() {
    return this->host_cube;
}

bool CudaCube::isHostInitialized() {
    return this->host_initialized;
}

bool CudaCube::isSubCube() {
    return this->is_subcube;
}

bool CudaCube::isSlice() {
    return this->is_slice;
}

double *CudaCube::getDevicePointer(int device) {
    return this->device_cubes[device];
}

/*
 * Returns the leading dimension of the slice, which can be
 * given as input to the CuBLAS methods.
 */
int CudaCube::getDeviceSliceLeadingDimension(int device) {
    if (this->isSlice()) {
        // get the shape of the underlying device memory
        int *device_memory_shape = this->getDeviceMemoryShape();
        // the slice_dimension == 0 case might be incorrect as it is, at least for now,
        // never used
        if (this->slice_dimension == 0) {
            return 1;
        }
        else if (this->slice_dimension == 1) {
            return this->getDevicePitch(device) / sizeof(double) * device_memory_shape[1];
        }
        else {
            return this->getDevicePitch(device) / sizeof(double);
        }
    }
    else {
        printf("Attempting to get a slice leading dimension for a non-slice cube. This is not allowed.");
        exit(-1);
        return 0;
    }
    
}

/*
 * Gets the shape of the memory of the host cube. 
 */
int *CudaCube::getHostMemoryShape() {
    // check if parent cube is not NULL and has inited host memory
    if (this->parentCube && this->parentCube->isHostInitialized()) {
        return this->parentCube->getHostMemoryShape();
    }
    else {
        if (this->isHostInitialized()) {
            return this->host_shape;
        }
        else {
            return NULL;
        }
    }
}

/*
 * Gets the initialization shape of the memory of the device cube. 
 * 
 * NOTE: the actual shape is different, as there the memory is pitched.
 *       Also, if all_memory_at_all_devices is false, the device 
 *       memory is split between devices.
 */
int *CudaCube::getDeviceMemoryShape() {
    // check if parent cube is not NULL
    if (this->parentCube) {
        return this->parentCube->getDeviceMemoryShape();
    }
    else {
        return this->shape;
    }
}

/********************************************
 *  Upload methods                          *
 ********************************************/

/*
 * Uploads the data from host to device. To use this method, the host cube has to have
 * the same Y-shape as the device-cube (i.e., device-shape). If this is not the case
 * use the other method.
 * 
 * NOTE: this method assumes that host_cube is already registered as pinned memory
 */
void CudaCube::uploadFrom(double *host_cube, size_t host_pitch) {
    double *device_cube;
    size_t device_pitch;
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        this->streamContainer->setDevice(device);
        device_cube = this->device_cubes[device];
        device_pitch = this->device_pitches[device];

        int device_vector_count =  (this->shape[2] * this->shape[1]) / this->streamContainer->getNumberOfDevices()
                                  + (((this->shape[2] * this->shape[1]) % this->streamContainer->getNumberOfDevices()) > device);
        for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream ++) {
            
            int vector_count = device_vector_count / this->streamContainer->getStreamsPerDevice() 
                                + ((device_vector_count % this->streamContainer->getStreamsPerDevice()) > stream);
            // copy the f1 cube to device: 3D
            cudaMemcpy2DAsync((void *) device_cube, device_pitch,
                              (void *) host_cube, host_pitch,
                              this->shape[0] * sizeof(double),
                              vector_count, 
                              cudaMemcpyHostToDevice,
                              *this->streamContainer->getStream(device, stream)
                             );
            
            host_cube += vector_count * host_pitch / sizeof(double);
            device_cube += vector_count * device_pitch / sizeof(double);
        }
    }
}

/*
 * Uploads the data from host to device. 
 * 
 * NOTE: this method assumes that host_cube is already registered as pinned memory
 * @param host_cube is the pointer to the first element of the uploaded host cube
 * @param host_shape is the shape of the memory at the host
 * @param upload_shape is the shape of the uploaded memory
 */
void CudaCube::uploadSlicedFrom(double *host_cube, int host_shape[3], size_t host_pitch, int upload_shape[3]) {
    double *device_cube;
    size_t device_pitch;
    double *host_cube_pointer = host_cube;
    int *device_memory_shape = this->getDeviceMemoryShape();
    int *host_memory_shape = this->getHostMemoryShape();
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        this->streamContainer->setDevice(device);
        device_cube = this->device_cubes[device];
        device_pitch = this->device_pitches[device];

        // get the number of slices uploaded by the device
        int device_slice_count;
        if (this->all_memory_at_all_devices) {
            device_slice_count = upload_shape[2];
            host_cube_pointer = host_cube;
        }
        else {
            device_slice_count =  upload_shape[2] / this->streamContainer->getNumberOfDevices()
                                    + ((upload_shape[2] % this->streamContainer->getNumberOfDevices()) > device);
        }
        // do the upload in streams
        for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream ++) {
            
            int slice_count = device_slice_count / this->streamContainer->getStreamsPerDevice() 
                                + ((device_slice_count % this->streamContainer->getStreamsPerDevice()) > stream);
            cudaMemcpy3DParms memCopyParameters = {0};
            memCopyParameters.srcPtr = make_cudaPitchedPtr(host_cube_pointer,   host_pitch,  host_memory_shape[0],  host_memory_shape[1]);
            memCopyParameters.dstPtr = make_cudaPitchedPtr(device_cube, device_pitch, device_memory_shape[0], device_memory_shape[1]);
            memCopyParameters.extent = make_cudaExtent(upload_shape[0] * sizeof(double), upload_shape[1], slice_count);
            memCopyParameters.kind   = cudaMemcpyHostToDevice;
            // copy the f1 cube to device: 3D
            cudaMemcpy3DAsync(&memCopyParameters, 
                              *this->streamContainer->getStream(device, stream));
            check_errors(__FILE__, __LINE__);
            
            host_cube_pointer += slice_count * host_memory_shape[0] * host_memory_shape[1];
            device_cube += slice_count * device_pitch / sizeof(double) * device_memory_shape[1];
        }
    }
}

void CudaCube::upload() {
    if (this->sliced) {
        this->uploadSliced();
    }
    else {
        this->uploadFrom(this->host_cube, this->host_pitch);
    }
}

void CudaCube::uploadSliced() {
    int upload_shape[3];
    upload_shape[0] = min(this->host_shape[0], this->shape[0]);
    upload_shape[1] = min(this->host_shape[1], this->shape[1]);
    upload_shape[2] = min(this->host_shape[2], this->shape[2]);
    this->uploadSlicedFrom(this->host_cube, this->host_shape, this->host_pitch, upload_shape);
}

void CudaCube::uploadSliced(int upload_shape[3]) {
    this->uploadSlicedFrom(this->host_cube, this->host_shape, this->host_pitch, upload_shape);
}


/********************************************
 *  Download methods                        *
 ********************************************/

/*
 * NOTE: this method assumes that host_cube is already registered as pinned memory.
 * To use this method, the uploaded host cube has to have
 * the same Y-shape as the device-cube (i.e., device-shape). If this is not the case
 * use the sliced method.
 */
void CudaCube::downloadTo(double *host_cube, size_t host_pitch) {
    double *device_cube;
    size_t device_pitch;
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        this->streamContainer->setDevice(device);
        device_cube = this->device_cubes[device];
        device_pitch = this->device_pitches[device];

        int device_vector_count =  (this->shape[2] * this->shape[1]) / this->streamContainer->getNumberOfDevices()
                                  + (((this->shape[2] * this->shape[1]) % this->streamContainer->getNumberOfDevices()) > device);
        
        for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream ++) {
            int vector_count = device_vector_count / this->streamContainer->getStreamsPerDevice() 
                                + ((device_vector_count % this->streamContainer->getStreamsPerDevice()) > stream);
            // copy the f1 cube to device: 3D
            cudaMemcpy2DAsync((void *) host_cube, host_pitch,
                              (void *) device_cube, device_pitch,
                              this->shape[0] * sizeof(double), 
                              vector_count, 
                              cudaMemcpyDeviceToHost,
                              *this->streamContainer->getStream(device, stream)
                             );
            check_errors(__FILE__, __LINE__);
            host_cube += vector_count * host_pitch / sizeof(double);
            device_cube += vector_count * device_pitch / sizeof(double);
        }
    }
    
    // synchronize the devices
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        this->streamContainer->setDevice(device);
        cudaDeviceSynchronize();
    }
}

/*
 * Downloads the data from device to host. 
 * 
 * NOTE: this method assumes that host_cube is already registered as pinned memory
 * @param host_cube is the pointer to the first element of the uploaded host cube
 * @param host_shape is the shape of the memory at the host
 * @param upload_shape is the shape of the uploaded memory
 */
void CudaCube::downloadSlicedTo(double *host_cube, int host_shape[3], size_t host_pitch, int download_shape[3]) {
    double *device_cube;
    int *device_memory_shape = this->getDeviceMemoryShape();
    int *host_memory_shape = this->getHostMemoryShape();
    size_t device_pitch;
    
    int device_offset = 0;
    
    // if we have all memory at all devices, we must do the gathering
    // of the data first, i.e., download from peer-devices the contributions
    // in the area where this device is resposible
    if (this->all_memory_at_all_devices && this->streamContainer->getNumberOfDevices() > 1) {
        this->gatherDeviceContributions(download_shape);
    }
    
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        this->streamContainer->setDevice(device);
        device_cube = this->device_cubes[device];
        device_pitch = this->device_pitches[device];
        
        // get the number of slices downloaded by the device
        if (!this->all_memory_at_all_devices) {
            // because of the gathering, we only need to download approximately 1/device_count part of 
            // all slices, however, because all of the memory is at all devices, the offset is not 
            // set to zero for the new device
            device_offset = 0;
        }
        
        // get the portion that is downloaded from device with order number 'device'
        int device_slice_count =  download_shape[2] / this->streamContainer->getNumberOfDevices()
                                  + ((download_shape[2] % this->streamContainer->getNumberOfDevices()) > device);
        
        for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream ++) {
            
            int slice_count = device_slice_count / this->streamContainer->getStreamsPerDevice() 
                                + ((device_slice_count % this->streamContainer->getStreamsPerDevice()) > stream);
            cudaMemcpy3DParms memCopyParameters = {0};
            memCopyParameters.dstPtr = make_cudaPitchedPtr(host_cube,   host_pitch,  host_memory_shape[0],  host_memory_shape[1]);
            memCopyParameters.srcPtr = make_cudaPitchedPtr(&device_cube[device_offset], device_pitch, device_memory_shape[0], device_memory_shape[1]);
            memCopyParameters.extent = make_cudaExtent(download_shape[0] * sizeof(double), download_shape[1], slice_count);
            memCopyParameters.kind   = cudaMemcpyDeviceToHost;
            // copy the f1 cube to device: 3D
            cudaMemcpy3DAsync(&memCopyParameters, 
                              *this->streamContainer->getStream(device, stream));
            check_errors(__FILE__, __LINE__);
            
            host_cube += slice_count *  host_pitch / sizeof(double) * host_memory_shape[1];
            device_offset += slice_count * device_pitch / sizeof(double) * device_memory_shape[1];
        }
    }
    
    // synchronize the devices
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        this->streamContainer->setDevice(device);
        this->downloaded_events[device] = this->streamContainer->recordDeviceEvent(device);
    }
}

/*
 * Sums together contributions from all devices residing at each devices download area.
 */
void CudaCube::gatherDeviceContributions(int gather_shape[3]) {
    double *device1_cube, *device1_gather_cube, *device2_cube;
    int *device_memory_shape = this->getDeviceMemoryShape();
    size_t device1_pitch, device2_pitch, device1_gather_pitch;
    
    int device1 = 0;
    this->streamContainer->setDevice(device1);
    device1_gather_cube  = this->device_gather_cubes[device1];
    device1_gather_pitch = this->device_gather_pitches[device1];
    device1_cube         = this->device_cubes[device1];
    device1_pitch        = this->device_pitches[device1];
      
    int device1_offset = 0;
    int global_offset = 0;
    // get the number of slices that this 'device1' is responsible for
    int device_slice_count =  gather_shape[2]; //device_slice_count / this->streamContainer->getStreamsPerDevice() 
                           // + ((device_slice_count % this->streamContainer->getStreamsPerDevice()) > stream);
   
                  
    for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream ++) {
        
        
        int slice_count = device_slice_count / this->streamContainer->getStreamsPerDevice() 
                            + ((device_slice_count % this->streamContainer->getStreamsPerDevice()) > stream);
        if (slice_count > 0) {
                            
            // get the launch configuration for the sum
            dim3 block, grid;
            this->getLaunchConfiguration(&grid, &block, slice_count, 256);
            // get the data at the are that this 'device1' is responsible for from all other devices and sum
            for (int device2 = 1; device2 < this->streamContainer->getNumberOfDevices(); device2 ++) {
                int can_access_peer;
                cudaDeviceCanAccessPeer(&can_access_peer, device1, device2);
                //printf("stream: %d/%d, device1: %d, device2: %d, slice_count: %d\n", stream, this->streamContainer->getStreamsPerDevice(), device1, device2, slice_count);
                device2_cube = this->device_cubes[device2];
                device2_pitch = this->device_pitches[device2];
            
            
                        
                if (can_access_peer) {
                    // do the sum
                    sum_3d <<<grid, block, 0, *this->streamContainer->getStream(device1, stream)>>>
                        (&device1_cube[global_offset], this->shape[0], this->shape[1], device_memory_shape[1], device1_pitch,
                        &device2_cube[device1_offset], this->shape[0], this->shape[1], device_memory_shape[1], device2_pitch,
                        slice_count);
                    check_errors(__FILE__, __LINE__);
                }
                else {
                    // prepare for the peer-to-peer memory copy
                    cudaMemcpy3DPeerParms memCopyParameters = {0};
                    memCopyParameters.dstPtr = make_cudaPitchedPtr(&device1_gather_cube[device1_offset],  device1_gather_pitch, device_memory_shape[0], device_memory_shape[1]);
                    memCopyParameters.dstDevice = device1;
                    memCopyParameters.srcPtr = make_cudaPitchedPtr(&device2_cube[global_offset],  device2_pitch, device_memory_shape[0], device_memory_shape[1]);
                    memCopyParameters.srcDevice = device2;
                    memCopyParameters.extent = make_cudaExtent(gather_shape[0] * sizeof(double), gather_shape[1], slice_count);
                    // do the peer-to-peer copy
                    cudaMemcpy3DPeerAsync(&memCopyParameters, 
                                        *this->streamContainer->getStream(device1, stream));
                    check_errors(__FILE__, __LINE__);
                        
                    // do the sum
                    sum_3d <<<grid, block, 0, *this->streamContainer->getStream(device1, stream)>>>
                        (&device1_cube[global_offset],        this->shape[0], this->shape[1], device_memory_shape[1], device1_pitch,
                        &device1_gather_cube[device1_offset], this->shape[0], this->shape[1], device_memory_shape[1], device1_gather_pitch,
                        slice_count
                        );
                    check_errors(__FILE__, __LINE__);
                }
                        
            }
            
            // copy the data from device1 to all others
            for (int device2 = 1; device2 < this->streamContainer->getNumberOfDevices(); device2 ++) {
                int can_access_peer;
                cudaDeviceCanAccessPeer(&can_access_peer, device1, device2);
                
                device2_cube = this->device_cubes[device2];
                device2_pitch = this->device_pitches[device2];
                if (can_access_peer) {
                    cudaMemcpy3DParms memCopyParameters = {0};
                    memCopyParameters.srcPtr = make_cudaPitchedPtr(&device1_cube[device1_offset],  device1_pitch, device_memory_shape[0], device_memory_shape[1]);
                    memCopyParameters.dstPtr = make_cudaPitchedPtr(&device2_cube[global_offset],   device2_pitch, device_memory_shape[0], device_memory_shape[1]);
                    memCopyParameters.extent = make_cudaExtent(gather_shape[0] * sizeof(double), gather_shape[1], slice_count);
                    memCopyParameters.kind   = cudaMemcpyDefault;
                    
                    // do the copy
                    cudaMemcpy3DAsync(&memCopyParameters, 
                                    *this->streamContainer->getStream(device1, stream));
                    check_errors(__FILE__, __LINE__);
                }
                else {
                    // prepare for the peer-to-peer memory copy
                    cudaMemcpy3DPeerParms memCopyParameters = {0};
                    memCopyParameters.dstPtr = make_cudaPitchedPtr(&device2_cube[global_offset],  device2_pitch, device_memory_shape[0], device_memory_shape[1]);
                    memCopyParameters.dstDevice = device1;
                    memCopyParameters.srcPtr = make_cudaPitchedPtr(&device1_cube[device1_offset],  device1_pitch, device_memory_shape[0], device_memory_shape[1]);
                    memCopyParameters.srcDevice = device2;
                    memCopyParameters.extent = make_cudaExtent(gather_shape[0] * sizeof(double), gather_shape[1], slice_count);
                    
                    // do the peer-to-peer copy
                    cudaMemcpy3DPeerAsync(&memCopyParameters, 
                                        *this->streamContainer->getStream(device1, stream));
                    check_errors(__FILE__, __LINE__);
                        
                }
            }
                            
            // add to the offsets
            device1_offset += slice_count * device1_gather_pitch / sizeof(double) * device_memory_shape[1];
            global_offset  += slice_count * device1_pitch / sizeof(double) * device_memory_shape[1];
        }
        
    }
    
    // synchronize the devices
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        this->streamContainer->setDevice(device);
        cudaDeviceSynchronize();
    }
    check_errors(__FILE__, __LINE__);
}




void CudaCube::download() {
    if (sliced) {
        this->downloadSliced();
    }
    else {
        double *host_cube = this->host_cube;
        size_t host_pitch = this->host_pitch;
        this->downloadTo(host_cube, host_pitch);
    }
}

void CudaCube::downloadSliced() {
    int download_shape[3];
    download_shape[0] = min(this->host_shape[0], this->shape[0]);
    download_shape[1] = min(this->host_shape[1], this->shape[1]);
    download_shape[2] = min(this->host_shape[2], this->shape[2]);
    this->downloadSlicedTo(this->host_cube, this->host_shape, this->host_pitch, download_shape);
}

void CudaCube::downloadSliced(int download_shape[3]) {
    this->downloadSlicedTo(this->host_cube, this->getHostMemoryShape(), this->host_pitch, download_shape);
}


/********************************************
 *  Partitioning methods                    *
 ********************************************/

/*
 * Gets a portion of the CudaCube and creates a new object from it. The copy is 
 * shallow, i.e., any operation done on the subcube also affects the original cube.
 * 
 * NOTE: the CudaCube used must be sliced in order to use this method.
 * NOTE: the CudaCube used must have all_memory_at_all_devices=true in order to use this method
 */
CudaCube *CudaCube::getSubCube(int start_indices[3], int end_indices[3], StreamContainer *streamContainer) {
    CudaCube *result = new CudaCube();
    // check if we have a correct input cube
    if (!this->all_memory_at_all_devices) {
        printf("Attempting to get a subcube of a CudaCube that has all_memory_at_all_devices=false. This is not allowed.");
        exit(-1);
    }
    
    // check if we have a correct input cube
    if (!this->sliced) {
        printf("Attempting to get a subcube of a CudaCube that has sliced=false. This is not allowed.");
        exit(-1);
    }
    
    check_errors(__FILE__, __LINE__);
    
    // copy the common parameters that remain the same
    result->streamContainer = streamContainer;
    result->host_initialized = this->host_initialized;
    result->host_registered = this->host_registered;
    result->all_memory_at_all_devices = this->all_memory_at_all_devices;
    result->is_subcube = true;
    result->parentCube = this;
    result->is_slice = false;
    result->slice_dimension = -1;
    result->sliced = this->sliced;
    result->downloaded_events = new cudaEvent_t*[result->streamContainer->getNumberOfDevices()];
    
    // get the host cube parameters if there are any
    if (this->isHostInitialized()) {
        // the pitch remains the same, as the memory space is still the same
        result->host_pitch = this->host_pitch;
        int *host_memory_shape = this->getHostMemoryShape();
        int host_offset =   start_indices[2] * host_memory_shape[1] * host_memory_shape[0] 
                          + start_indices[1] * host_memory_shape[0]
                          + start_indices[0];
        
        // the shape of the cube is changed
        result->host_shape[0] = end_indices[0] - start_indices[0];
        result->host_shape[1] = end_indices[1] - start_indices[1];
        result->host_shape[2] = end_indices[2] - start_indices[2];
        
        // as well as the pointer
        result->host_cube = &this->host_cube[host_offset];
    }
    
    //get the device cube parameters
    // the shape of the cube is changed
    result->shape[0] = end_indices[0] - start_indices[0];
    result->shape[1] = end_indices[1] - start_indices[1];
    result->shape[2] = end_indices[2] - start_indices[2];
    
    result->device_cubes = new double*[result->streamContainer->getNumberOfDevices()];
    result->device_pitches = new size_t[result->streamContainer->getNumberOfDevices()];
    // allocate space for device cube pointers
    //cudaHostAlloc((void **)&result->device_cubes, 
    //              sizeof(double *) * this->streamContainer->getNumberOfDevices(),
    //              cudaHostAllocPortable);
    // allocate space for device cube pointers
    //cudaHostAlloc((void **)&result->device_pitches, 
    //              sizeof(double) * this->streamContainer->getNumberOfDevices(),
    //              cudaHostAllocPortable);
    
    // if all memory at all devices, do the allocation for gather cubes also
    if (this->all_memory_at_all_devices) {
        // allocate space for device cube pointers
        result->device_gather_cubes = new double*[result->streamContainer->getNumberOfDevices()];
        //cudaHostAlloc((void **)&result->device_gather_cubes, 
        //            sizeof(double *) * this->streamContainer->getNumberOfDevices(),
        //            cudaHostAllocPortable);
        // allocate space for device cube pointers
        result->device_gather_pitches = new size_t[result->streamContainer->getNumberOfDevices()];
        //cudaHostAlloc((void **)&result->device_gather_pitches, 
        //            sizeof(double) * this->streamContainer->getNumberOfDevices(),
        //            cudaHostAllocPortable);
        //check_errors(__FILE__, __LINE__);
    }
    
    //check_errors(__FILE__, __LINE__);
    
    result->host_pitch = host_pitch;
    
    // get the shape of the device memory
    int *device_memory_shape = this->getDeviceMemoryShape();
        
    // set the device pointers and the pitches
    for (int device = 0; device < result->getStreamContainer()->getNumberOfDevices(); device ++) {
        // get the actual device number
        int device_number = result->getStreamContainer()->getDeviceNumber(device);
        // and get the corresponding order number in the streamContainer of 'this'
        int this_order_number = this->getStreamContainer()->getDeviceOrderNumber(device_number);
        
        // get the offset caused by the slicing in device memory (in elements) 
        int device_offset =   start_indices[2] * device_memory_shape[1] * this->device_pitches[this_order_number] / sizeof(double)
                            + start_indices[1] * this->device_pitches[this_order_number] / sizeof(double)
                            + start_indices[0];
        // the pitch remains the same
        result->device_pitches[device] = this->device_pitches[this_order_number];
            
        // but the device cube is offset by 'device_offset'
        double *device_cube = this->device_cubes[this_order_number];
        result->device_cubes[device] = &device_cube[device_offset];
        
        // if all memory at all devices, set the offsetted pointers for gather cubes also
        if (this->all_memory_at_all_devices) {
            result->device_gather_pitches[device] = this->device_gather_pitches[this_order_number];
            
        }
    }
    
    return result;
}

/*
 * Gets a slice of the CudaCube and creates a new object from it. The copy is very
 * shallow, i.e., any operation done on the subcube also affects the original cube.
 */
CudaCube *CudaCube::getSlice(int slice_index, int slice_dimension) {
    // get the starting indices of the subcube
    int start_indices[3] = {0, 0, 0};
    start_indices[slice_dimension] = slice_index;
    
    // get the ending indices of the subcube
    int end_indices[3];
    end_indices[0] = this->shape[0];
    end_indices[1] = this->shape[1];
    end_indices[2] = this->shape[2];
    end_indices[slice_dimension] = slice_index + 1;
    
    CudaCube *slice = this->getSubCube(start_indices, end_indices, this->getStreamContainer());
    slice->slice_dimension = slice_dimension;
    slice->is_slice = true;
    return slice;
}


/********************************************
 *  Fortran interfaces                      *
 ********************************************/

extern "C" CudaCube *cudacube_init_and_set_cuda(StreamContainer *streamContainer, int shape[3], 
                                        double *host_cube, int host_cube_offset, int host_cube_shape[3],
                                        int all_memory_at_all_devices, int register_host, int sliced) {
    CudaCube *cudaCube = new CudaCube(streamContainer, shape, &host_cube[host_cube_offset], host_cube_shape,
                                      all_memory_at_all_devices != 0, register_host != 0, sliced != 0); 
    return cudaCube;
}

extern "C" CudaCube *cudacube_init_cuda(StreamContainer *streamContainer, int shape[3], int all_memory_at_all_devices, int sliced) {
    CudaCube *cudaCube = new CudaCube(streamContainer, shape, all_memory_at_all_devices != 0, sliced != 0); 
    return cudaCube;
}

extern "C" CudaCube *cudacube_get_slice_cuda(CudaCube *cudaCube, int slice_index, int slice_dimension) {
    // remove one from both to get to C-indexing from Fortran indexing
    return cudaCube->getSlice(slice_index-1, slice_dimension-1);
}

extern "C" StreamContainer *cudacube_get_stream_container_cuda(CudaCube *cudaCube) {
    // remove one from both to get to C-indexing from Fortran indexing
    return cudaCube->getStreamContainer();
}


extern "C" CudaCube *cudacube_get_subcube_cuda(CudaCube *cudaCube, int start_indices[3], int end_indices[3], StreamContainer *streamContainer) {
    int start_indices_c[3];
    // remove one from start indices to get to C-indexing from Fortran indexing
    start_indices_c[0] = start_indices[0]-1; 
    start_indices_c[1] = start_indices[1]-1;
    start_indices_c[2] = start_indices[2]-1;
    
    return cudaCube->getSubCube(start_indices_c, end_indices, streamContainer);
}

extern "C" void cudacube_destroy_cuda(CudaCube *cudaCube){
    
    cudaCube->destroy();
    delete cudaCube;
}

extern "C" void cudacube_download_cuda(CudaCube *cudaCube){
    cudaCube->download();
}

extern "C" void cudacube_upload_cuda(CudaCube *cudaCube){
    cudaCube->upload();
}

extern "C" void cudacube_set_to_zero_cuda(CudaCube *cudaCube){
    cudaCube->setToZero();
}

extern "C" void cudacube_set_host_cube_cuda(CudaCube *cudaCube, double *host_cube, int host_cube_shape[3], int register_host){
    cudaCube->initHost(host_cube, host_cube_shape, register_host != 0);
}

extern "C" double* cudacube_get_host_cube_cuda(CudaCube *cudaCube){
    return cudaCube->getHostCube();
}

extern "C" void cudacube_unset_host_cube_cuda(CudaCube *cudaCube){
    cudaCube->unsetHostCube();
}

extern "C" void register_host_cube_cuda(double *host_cube, int host_cube_shape[3]){
    registerHostCube(host_cube, host_cube_shape);
}

extern "C" void unregister_host_cube_cuda(double *host_cube){
    unregisterHostCube(host_cube);
}



extern "C" double *cudacube_init_page_locked_cube_cuda(int shape[3]){
    double * result_cube;
    cudaHostAlloc((void **)&result_cube, 
                  sizeof(double) * shape[0] * shape[1] * shape[2],
                  cudaHostAllocPortable);
    check_errors(__FILE__, __LINE__);
    return result_cube;
}

extern "C" void cudacube_destroy_page_locked_cube_cuda(double * cube){
    cudaFreeHost(cube);
    check_errors(__FILE__, __LINE__);
}


