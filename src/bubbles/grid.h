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
#ifndef INCLUDE_GRID
#define INCLUDE_GRID
#include "streamcontainer.h"
using namespace std;

class Grid1D {
    private:
        StreamContainer *streamContainer;
        bool is_subgrid;
    public:
        int ncell;
        int nlip;
        double r_max;
        double *h;
        double *d;
        double *lip;
        double *derivative_lip;
        double *lower_derivative_lip;
        double *base_integrals;
        double *gridpoints;
        double *integrals;
        double ** device_h;
        double ** device_d;
        double ** device_lip;
        double ** device_derivative_lip;
        double ** device_lower_derivative_lip;
        double ** device_gridpoints;
        double ** device_integrals;
        Grid1D ** device_copies;
        
        // constructor
        Grid1D(int ncell, int nlip, double r_max, double *h, double *d, double *gridpoints, double *lip, double *derivative_lip, double *lower_derivative_lip, double *base_integrals, StreamContainer *streamContainer, bool init_device_memory = true);
        Grid1D();
        int getShape();
        Grid1D *getSubGrid(int start_index, int end_index, StreamContainer *streamContainer);
        double **getDeviceIntegrals();
        double *getDeviceIntegrals(int device);
        double *getIntegrals(int first_cell = 0);
        double *calculateIntegrals(int first_cell = 0, int last_cell = -1);
        void upload();
        // destroy all cuda related objects
        void destroy();
};
// TODO: Note, this is the new version, as soon as you are ready, replace the Grid3D_t stuff with this
class Grid3D {
    private:
        StreamContainer *streamContainer;
        bool is_subgrid;
    public:
        int shape[3];
        int *getShape();
        int getShape(int axis);
        Grid1D **axis;
        Grid1D ***device_axis;
        Grid3D **device_copies;
        
        Grid3D(Grid1D **axis, StreamContainer *streamContainer);
        Grid3D();
        Grid3D *getSubGrid(int start_indices[3], int end_indices[3], StreamContainer *streamContainer);
        void destroy();
};


#endif