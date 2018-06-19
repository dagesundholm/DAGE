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
#ifndef INCLUDE_CUBE
#define INCLUDE_CUBE
#include "streamcontainer.h"
using namespace std;

class CudaCube {
    private:
        // shape of the original device array
        // NOTE: in case of the sub-cube this is not the same as the
        // shape of the data handled by this object
        int shape[3];
        // shape of the host memory
        // NOTE: in case of the sub-cube this is not the same as the
        // shape of the data handled by this object
        int host_shape[3];
        // if the host array is set to this object
        bool host_initialized;
        // if the host array is registered as pinned memory
        bool host_registered;
        // if all memory of the cube resides at all devices
        bool all_memory_at_all_devices;
        // if the cube is a part of a larger cuda cube
        bool is_subcube;
        // if the cube memory is uploaded and downloaded in slices
        bool sliced;
        // if the cube is a slice of a larger cuda cube
        bool is_slice;
        // the dimension of slicing
        int slice_dimension;
        // pitch in the host cube
        size_t host_pitch;
        // pitch in the device cubes 
        size_t *device_pitches;
        // pitch in the gathering device cubes 
        size_t *device_gather_pitches;
        // pointer to the host cube
        double *host_cube;
        // pointers to device parts of cube 
        double **device_cubes;
        // pointers to device parts of cube 
        double **device_gather_cubes;
        // pointer to the cube this cube is extracted from (should be NULL if is_subcube is false)
        CudaCube *parentCube;
        // pointer to streamcontainer used in the integrator
        StreamContainer *streamContainer;
        
        void initMemory(StreamContainer *streamContainer, int shape[3], bool all_memory_at_all_devices);
        void initMemorySliced(StreamContainer *streamContainer, int shape[3], bool all_memory_at_all_devices);
        cudaEvent_t ** uploaded_events;
        cudaEvent_t ** downloaded_events;
        //void setToZeroSliced();
        //void setAllMemoryToZeroSliced();
    public:
        CudaCube();
        CudaCube(StreamContainer *streamContainer, int shape[3], double *host_cube, int host_cube_shape[3], bool all_memory_at_all_devices, bool register_host = true, bool sliced = false);
        CudaCube(StreamContainer *streamContainer, int shape[3], bool all_memory_at_all_devices, bool sliced = false);
        
        void setToZero();
        void setAllMemoryToZero();
        void unsetHostCube();
        
        // accessors
        size_t getDevicePitch(int device);
        size_t *getDevicePitches();
        double **getDeviceCubes();
        double *getHostCube();
        int getShape(int axis);
        int *getShape();
        size_t getHostPitch();
        bool isHostInitialized();
        bool isSubCube();
        int *getHostMemoryShape();
        int *getDeviceMemoryShape();
        double *getDevicePointer(int device);
        int getDeviceSliceLeadingDimension(int device);
        bool isSlice();
        
        // launch configuration methods
        void getLaunchConfiguration(dim3 *grid, dim3 *block, int slice_count, int block_size);
        
        // sub-structure methods
        CudaCube *getSubCube(int start_indices[3], int end_indices[3], StreamContainer *streamContainer);
        CudaCube *getSlice(int slice_index, int slice_dimension);
        
        // upload the cube to devices
        void upload();
        void uploadFrom(double *host_cube, size_t host_pitch);
        void uploadSliced();
        void uploadSliced(int upload_shape[3]);
        void uploadSlicedFrom(double *host_cube, int host_shape[3], size_t host_pitch, int upload_shape[3]);
        
        // download cube from devices
        void download();
        void downloadTo(double *host_cube, size_t host_pitch);
        void downloadSliced();
        void downloadSliced(int download_shape[3]);
        void downloadSlicedTo(double *host_cube, int host_shape[3], size_t host_pitch, int download_shape[3]);
        void initHost(double *host_cube, int host_cube_shape[3], bool register_host);
        void gatherDeviceContributions(int gather_shape[3]);
        
        // streamcontainer accessors
        StreamContainer *getStreamContainer();
        void setStreamContainer(StreamContainer *streamContainer);
        
        // destroy all cuda related objects
        void destroy();
};


void multiply_3d_cubes(double*  __restrict__ cube1, const int cube1_shape_x, const int cube1_shape_y, const int cube1_memory_shape_y, const size_t cube1_pitch,
                       const double* __restrict__ cube2, const int cube2_shape_x, const int cube2_shape_y, const int cube2_memory_shape_y, const size_t cube2_pitch,
                       const int slice_count, dim3 *grid, dim3 *block, cudaStream_t *stream);
void multiply_3d_cubes(const double*  __restrict__ cube1, const int cube1_shape_x, const int cube1_shape_y, const int cube1_memory_shape_y, const size_t cube1_pitch,
                       const double* __restrict__ cube2, const int cube2_shape_x, const int cube2_shape_y, const int cube2_memory_shape_y, const size_t cube2_pitch,
                       double* __restrict__ result_cube, const int result_cube_shape_x, const int result_cube_shape_y, const int result_cube_memory_shape_y, const size_t result_cube_pitch,
                       const int slice_count, dim3 *grid, dim3 *block, cudaStream_t *stream);
void getCubeLaunchConfiguration(dim3 *grid, dim3 *block, int shape[3], int slice_count, int block_size);

#endif