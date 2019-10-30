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
#ifndef INCLUDE_GBFMM_POTENTIAL_OPERATOR
#define INCLUDE_GBFMM_POTENTIAL_OPERATOR
#include "../bubbles/grid.h"
#include "../bubbles/cube.h"
#include "../bubbles/streamcontainer.h"
#include "../bubbles/integrator.h"
#include "../bubbles/spherical_harmonics_cuda.h"

using namespace std;


class GBFMMPotentialOperator {
    protected:
        // starting and ending box indices of the domain belonging to this node 
        int domain[2];
        // centers of the boxes
        double *centers;
        // the maximum angular momentum quantum number 'l' value used in evaluation
        int lmax;
        // the normalization of spherical harmonics used in the operator
        int normalization;
        // the first and last cell index in x, y, and z -directions for each box in domain for input cube 
        int *input_start_indices_x, *input_start_indices_y, *input_start_indices_z, *input_end_indices_x, *input_end_indices_y, *input_end_indices_z;
        // the first and last cell index in x, y, and z -directions for each box in domain for output cube
        int *output_start_indices_x, *output_start_indices_y, *output_start_indices_z, *output_end_indices_x, *output_end_indices_y, *output_end_indices_z;
        // space used in evaluation of local expansion
        double **device_expansions;
        // if the operator uses some objects of another operator
        bool is_child_operator;
        // parent operator of the operator
        GBFMMPotentialOperator *parent_operator;
        // main grid of the evaluation domain for this node
        Grid3D *grid_in;
        // main grid of the output domain for this node
        Grid3D *grid_out;
        // subgrids of the domain boxes of the input cube
        Grid3D **input_grids;
        // subgrids of the domain boxes of the output cube
        Grid3D **output_grids;
        // grids used in evaluation of harmonics (and bessels)
        Grid3D **device_grids;
        // stream container split to single device containers
        StreamContainer **device_containers;
        // integrators for multipole evaluation
        Integrator3D **integrators;
        // sub-streamcontainers for the domain boxes
        StreamContainer **streamContainers;
        // solid/spherical harmonics
        RealCubeHarmonics **harmonics;
        // main (global) streamcontainer from which the sub-containers are extracted from
        StreamContainer *streamContainer;
        
    public:
        void initGBFMMPotentialOperator(
               // the grid from which the subgrids are extracted from (should represent the entire input domain) needed 
               // to evaluate coulomb potential for using gbfmm
               Grid3D *grid_in,
               // the grid from which the subgrids are extracted from (should represent the entire output domain) needed 
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
        void initGBFMMPotentialOperator(
               // the operator used as the basis of the new operator
               GBFMMPotentialOperator *parent_operator,
               // the maximum angular momentum quantum number 'l' value
               int lmax);
        void evaluatePotentialLE(double *local_expansion, CudaCube *output_cube);
        virtual void evaluatePotentialLEBox(int i,
                                    double *local_expansion,
                                    double zero_point[3], 
                                    Grid3D *grid,
                                    CudaCube *output_cube,
                                    StreamContainer *streamContainer) = 0;
        virtual void downloadMultipoleMoments(double *host_multipole_moments);
        void calculateMultipoleMoments(CudaCube *input_cube);
        virtual void calculateMultipoleMomentsBox(int i, CudaCube *cube) = 0;
        virtual void destroyHarmonics() = 0;
        virtual void initHarmonics() = 0;
        virtual void initIntegrators();
        void uploadDomainBoxes(CudaCube *input_cube);
        StreamContainer *getBoxStreamContainer(int ibox);
        void destroy();
};
#endif
