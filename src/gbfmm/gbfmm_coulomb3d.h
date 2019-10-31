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
#ifndef INCLUDE_COULOMB3D
#define INCLUDE_COULOMB3D
#include "../bubbles/grid.h"
#include "../bubbles/cube.h"
#include "../bubbles/streamcontainer.h"
#include "../bubbles/integrator.h"
#include "../bubbles/spherical_harmonics_cuda.h"
#include "gbfmm_potential_operator.h"
using namespace std;

class GBFMMCoulomb3DMultipoleEvaluator : public Integrator3D {
    protected:
       void reduce3D(double *input_data, 
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
       int lmax;
       double center[3];
       
       // constructor
       GBFMMCoulomb3DMultipoleEvaluator(StreamContainer *streamContainer, Grid3D *grid, int lmax, double center[3]);
       
};

class GBFMMCoulomb3D : public GBFMMPotentialOperator {        
    public:
        GBFMMCoulomb3D(
               // the grid from which the subgrids are extracted from (should represent the input domain) needed 
               // to evaluate coulomb potential for using gbfmm
               Grid3D *grid_in,
               // the grid from which the subgrids are extracted from (should represent the output domain) needed 
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
               StreamContainer *streamContainer);
        void evaluatePotentialLEBox(int i,
                                    double *local_expansion,
                                    double zero_point[3], 
                                    Grid3D *grid,
                                    CudaCube *output_cube,
                                    StreamContainer *streamContainer);
        void calculateMultipoleMomentsBox(int i, CudaCube *cube);
        void downloadMultipoleMoments(double *host_multipole_moments);
        void initHarmonics();
        void initIntegrators();
        void destroyHarmonics();
};
#endif
