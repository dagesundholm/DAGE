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
#ifndef INCLUDE_INTEGRATOR
#define INCLUDE_INTEGRATOR
#include "streamcontainer.h"
#include "cube.h"
#include "grid.h"
using namespace std;

class Integrator {
    protected:
        // result counter, follows the index of the first undownloaded result
        int first_result_counter;
        // result counter, follows the index of the latest calculated result
        int last_result_counter;
        // maximum number of stored results
        int max_number_of_stored_results;
        // pointer to streamcontainer used in the integrator
        StreamContainer *streamContainer;
        // free space needed in contraction 
        double **integration_space;
        // free space needed for results
        double **device_results;
        // the copies of the integrator containing the device data
        Integrator **device_copies;
    public:
        
        // the number of results each call to the integrator produces
        int number_of_dimensions;
        // maximum_block count in reduce call
        int maximum_block_count;
        // do the contraction within block
        template <unsigned int blockSize> 
        __device__ 
        void contractBlock(const int threadId, double &value);
        __host__ inline void reduce1D(double *input_data, 
                                      double *output_data,
                                      unsigned int inputSize,
                                      unsigned int output_size,
                                      unsigned int threads_per_block,
                                      cudaStream_t *stream,
                                      int device_order_number);
        void downloadResult(double *);
};


class Integrator1D : public Integrator {
    private:
        Grid1D *grid;
        int shape;
        // the number of processors used in the integration
        int number_of_processors;
        // the order number of the current processor amongst the 'number_of_processors' 
        int processor_order_number;
        // array of pointers to device vectors 
        double **device_vectors;
    public:
        Integrator1D(StreamContainer *streamContainer, Grid1D *grid, int processor_order_number=0, int number_of_processors=1, int number_of_dimensions = 1);
        double integrate(double *vector);
        double integrate();
        void contractVector(double *device_vector, 
                       double *extra_data,
                       unsigned int vector_length, 
                       cudaStream_t *stream,
                       double *device_result,
                       int device_order_number);
        double **getDeviceVectors();
        void upload(double *vector);
        void integrateOnDevices(double **device_vectors);
        void integrateSingleDevice(double *device_vector, const int device, int &offset, double *result_array, int processor_point_count, int processor_point_offset);
        void destroy();
        
};



class Integrator3D : public Integrator {
    protected:
        CudaCube *cube;
        Grid3D *grid;
        
        // private worker functions 
        void initCommons(StreamContainer *streamContainer, Grid3D *grid, int max_number_of_stored_results = 50, int dimensions = 1, bool init_cube = true);
        void contractCube(double *device_cube, 
                          size_t device_pitch,
                          int device_y_shape,
                          int shape[3],
                          int slice_count,
                          double *extra_data,
                          cudaStream_t *stream,
                          double *device_result, 
                          int slice_offset,
                          int device_order_number
                         );
         virtual void reduce3D(double *input_data, 
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
                       int device_order_number);
    public:
        Integrator3D();
        Integrator3D(StreamContainer *streamContainer, Grid3D *grid, int max_number_of_stored_results = 50, bool init_cube = true);
        void integrateSingleDevice(double *device_cube, const size_t device_pitch, const int device_memory_shape_y, int shape[3],
                                         const int device, int &slice_offset, double *device_result);
        void integrateOnDevices(double **device_cubes, size_t *device_pitches, int device_memory_shape_y, int shape[3]);
        double integrate(double *cube, int host_cube_shape[3]);
        void integrate();
        __device__ __host__ Grid3D *getGrid();
        __host__  inline void getLaunchConfiguration(dim3 *grid, dim3 *block, int block_size, int grid_size);
        void setIntegrationCube(CudaCube *cube);
        CudaCube *getIntegrationCube();
        void destroy();
};

void contract_cube(double *device_cube, 
                   size_t device_pitch,
                   int device_y_shape,
                   int shape[3],
                   int slice_count,
                   double *extra_data,
                   cudaStream_t *stream,
                   double *device_result);

#endif