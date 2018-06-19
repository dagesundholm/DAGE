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
#ifndef INCLUDE_SPH_HARM
#define INCLUDE_SPH_HARM
#include <cuda.h>
#include "grid.h"
#include "streamcontainer.h"




class RealHarmonics {
    public:
        /** Minimum quantum number 'l' value evaluated */
        int lmin;
        /** Maximum quantum number 'l' value evaluated */
        int lmax;
        /** Used normalization 1: Racah normalization, 2: Conventional normalization */
        int normalization;
        RealHarmonics **device_copies;
        StreamContainer *streamContainer;
        void initRealHarmonics(int lmin, int lmax, int normalization, StreamContainer *streamContainer);
        /* Destroy all the cuda stuff and deallocate host memory also */
        virtual void destroy() = 0;
        
};

class RealCubeHarmonics : public RealHarmonics {   
    protected:
        /** The device result arrays. Have 4 dimensions, x, y, z, and lm_index. **/ 
        double **device_results;
        /** The pitches of device result arrays.  **/
        size_t *device_pitches;
        /** The underlying cube shape, The reason why we are not using Grid3D is that 
            we do not want to limit the results to a position, which would be the case 
            with a Grid3D (has coordinates instead of relative coordinates)**/
        int shape[3];
        void initCubeHarmonics(int lmin, int lmax, int normalization, int shape[3], StreamContainer *streamContainer);
    public:
        /* 
         * Evaluates the grid at the grid (grid must be of shape 'this->shape')
         */
        void evaluate(Grid3D *grid, double center[3]);
        virtual void evaluateSingleStream(double *device_results, size_t device_pitch, int device,
                                          Grid3D *grid, double center[3], int slice_count,
                                          int device_slice_count, int slice_offset, cudaStream_t *stream) = 0;
        void download(double *host_results, int host_results_shape[4]);
        void registerHostResultArray(double *host_results, int host_results_shape[4]);
        void unregisterHostResultArray(double *host_results);
        double **getDeviceResults();
        double *getDeviceResults(int device);
        size_t *getDevicePitches();
        size_t getDevicePitch(int device);
        int *getShape();
        void destroy();
};


class RealRegularSolidCubeHarmonics : public RealCubeHarmonics {
    public:
        RealRegularSolidCubeHarmonics(int lmin, int lmax, int normalization, int shape[3], StreamContainer *streamContainer);
        void evaluateSingleStream(double *device_results, size_t device_pitch, int device,
                               Grid3D *grid3d, double center[3], int slice_count,
                               int device_slice_count, int slice_offset, cudaStream_t *stream);
};

class RealSphericalCubeHarmonics : public RealCubeHarmonics {
    public:
        RealSphericalCubeHarmonics(int lmin, int lmax, int normalization, int shape[3], StreamContainer *streamContainer);
        void evaluateSingleStream(double *device_results, size_t device_pitch, int device,
                               Grid3D *grid, double center[3], int slice_count,
                               int device_slice_count, int slice_offset, cudaStream_t *stream);
};

typedef struct MultipoleEvaluator {
    double **deviceHarmonics;
    double **deviceIntegralsX;
    double **deviceIntegralsY;
    double **deviceIntegralsZ;
    double **deviceBessels;
    double **deviceCubes;
    double **extraData;
    Grid3D **grids;
    //StreamContainer_t *streamContainer;
    int streams_per_device;
    int number_of_devices;
    cudaStream_t **streams;
    int is_helmholtz;
} MultipoleEvaluator_t;


#endif 