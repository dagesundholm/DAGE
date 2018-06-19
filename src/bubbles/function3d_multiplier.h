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
#ifndef INCLUDE_FUNCTION3DMULTIPLIER
#define INCLUDE_FUNCTION3DMULTIPLIER
#include "bubbles_cuda.h"
#include "spherical_harmonics_cuda.h"
#include "bubbles_multiplier.h"
#include "streamcontainer.h"
#include "grid.h"
#include "cube.h"

class Function3DMultiplier {
    private:
        CudaCube *cube1;
        CudaCube *cube2;
        CudaCube *result_cube;
        Grid3D *grid;
        StreamContainer *streamContainer;
    public:
        Function3DMultiplier(Grid3D *grid, StreamContainer *streamContainer);
        void uploadCubes(double *cube1, int cube1_offset, int cube1_host_shape[3], int cube1_lmax,
                         double *cube2, int cube2_offset, int cube2_host_shape[3], int cube2_lmax);
        void setHostResultCube(double *host_cube, int cube_offset, int cube_host_shape[3]);
        void multiply(Bubbles *f1_bubbles, Bubbles *f2_bubbles, Bubbles *result_bubbles);
        void downloadResult();
        void destroy();
};
#endif