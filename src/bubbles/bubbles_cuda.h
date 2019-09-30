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
#ifndef INCLUDE_BUBBLES
#define INCLUDE_BUBBLES

#include "streamcontainer.h"
#include "grid.h"
#include "integrator.h"
#include "cube.h"
#include "spherical_harmonics_cuda.h"
using namespace std;

class Bubble {
    private:
        void initDeviceMemory(int ibub, Grid1D *grid, double center[3], int lmax,
               int k, double charge, StreamContainer *streamContainer);
        Integrator1D *integrator;
        cudaEvent_t ** uploaded_events;
    public:
        // pointer to streamcontainer used in the bubble functions
        StreamContainer *streamContainer;
        double crd[3];
        // the order number of the bubble within the global bubbles.
        // NOTE: this can be different from the local bubble order number
        int ibub;
        short lmin;
        short lmax;
        // the lmax of the memory allocated on device, should be changed only at allocation
        short device_memory_lmax;
        // the number of processors used in the integration
        int number_of_processors;
        // the order number of the current processor amongst the 'number_of_processors' 
        int processor_order_number;
        int k;
        int charge;
        Grid1D *grid;
        double * f;
        double * cf;
        double * df;
        size_t f_pitch;
        
        // device pointer arrays
        double ** device_f;
        size_t *  device_f_pitch;
        double ** device_cf;
        double ** device_df;
        Grid1D ** device_grids;
        Bubble ** device_copies; 
        
        // constructors
        Bubble(int ibub, Grid1D *grid, double center[3], int lmax, int k, double *bf,
               double charge, StreamContainer *streamContainer);

        Bubble(int ibub, Grid1D *grid, double center[3], int lmax, int k, double charge, StreamContainer *streamContainer);
        Bubble(Bubble *old_bubble, int lmax, int k);

        
        // functions
        void waitBubbleUploaded(int device);
        void waitBubbleUploaded(int device, cudaStream_t *stream);
        double integrate();
        void setProcessorConfiguration( int processor_order_number, int number_of_processors);
        void decreaseK(int k_decrease);
        void calculateCf();
        void registerHost(double *f);
        void setToZero();
        void uploadAll(double *f, int lmax);
        void upload(double *f, int lmax, bool register_host=true);
        void add(Bubble *bubble);
        void download(int lmax);
        void destroy();
};

class Bubbles {
    private:
        int nbub;
        bool is_sub_bubbles;
        Bubble ** bubbles;
    public:
        Bubbles(int nbub);
        Bubbles(Bubbles *old_bubbles, int lmax, int k);
        Bubble *getBubble(int ibub);
        bool    containsBubble(int ibub);
        Bubble *getBubbleWithLocalOrderNumber(int i);
        Bubbles *getSubBubbles(int *ibubs, int nbub);
        void add(Bubbles *bubbles);
        void setProcessorConfiguration( int processor_order_number, int number_of_processors);
        void initBubble(Grid1D *grid, int i, int ibub, double center[3], int lmax,
                            int k, double *bf, double charge, StreamContainer *streamContainer);
        void initBubble(Grid1D *grid, int i, int ibub, double center[3], int lmax,
                            int k, double charge, StreamContainer *streamContainer);
        void inject(Grid3D *grid, CudaCube *cube, int lmin = 0, CudaCube *gradients_cube_x = NULL,
                    CudaCube *gradients_cube_y = NULL, CudaCube *gradients_cube_z = NULL, bool evaluate_value = true,
                     bool evaluate_gradients_x = false, bool evaluate_gradients_y = false, bool evaluate_gradients_z = false);
        int getBubbleCount();
        void waitBubblesUploaded(int device);
        
        double integrate();
        void unregister();
        void download();
        void destroy();

};


__device__ __forceinline__ 
void getXYZ(int *x, int *y, int *z) {
    *x = blockIdx.x * blockDim.x + threadIdx.x;
    *y = blockIdx.y * blockDim.y + threadIdx.y;
    *z = blockIdx.z * blockDim.z + threadIdx.z;
    // printf("z: %i, bi: %i, bd: %i, ti: %i \n", *z, blockIdx.z, blockDim.z, threadIdx.z);
}

/*
 * Returns the cube pointer offset caused by the x, y, z coordinates with given pitch and memory shape in y-direction
 */
__device__ __forceinline__
int getCubeOffset3D(const int x, const int y, const int z, const size_t pitch, const int memory_y_shape) {
    return    z * memory_y_shape * pitch / sizeof(double)
            + y * pitch / sizeof(double)
            + x;
}



typedef struct DeviceCube {
    double *cube;
} DeviceCube_t;

typedef struct Injector{
    int gdims[3];
    double *gr[3];
    double *cube_h;
    int cuda_gdims[3]; /* Number of CUDA blocks in each dimension */
    int cuda_nblocks_exec;  /* Number of executed CUDA blocks */
    int cuda_nblocks_total;  /* Total number of CUDA blocks needed */
    cudaPitchedPtr cube_d;
    double k;
    int lmax;
    int lmin;
    int numy;
    int *nt;
    double *coeffs;
    int *expos;
    int nlip;
    struct Injector *copy_dev;
} Injector_t;
#endif
