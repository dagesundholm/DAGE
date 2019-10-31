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
#ifndef INCLUDE_EVALUATORS
#define INCLUDE_EVALUATORS

#include "streamcontainer.h"
#include "bubbles_cuda.h"
#include "cube.h"
#include "points.h"

class Evaluator  {
    protected:
        StreamContainer *streamContainer;
    public:
        virtual void evaluatePoints(Points *result_points, Points *gradient_points_x  = NULL, Points *gradient_points_y  = NULL, Points *gradient_points_z = NULL, int gradient_direction =  3) = 0;
        virtual void evaluateGrid(Grid3D *grid, CudaCube *result_cube, CudaCube *gradient_cube_x = NULL, CudaCube *gradient_cube_y = NULL, CudaCube *gradient_cube_z = NULL, int gradient_direction = 3, int fin_diff_ord = 0) = 0;
        virtual void destroy() = 0;
};

class BubblesEvaluator : public Evaluator {
    private:
        Bubbles * bubbles;
    public:
        BubblesEvaluator(StreamContainer *streamContainer);
        void setBubbles(Bubbles *bubbles);
        void evaluatePoints(Points *result_points, Points *gradient_points_x  = NULL, Points *gradient_points_y  = NULL, Points *gradient_points_z = NULL, int gradient_direction =  3);
		void evaluateGrid(Grid3D *grid, CudaCube *result_cube, CudaCube *gradient_cube_x = NULL, CudaCube *gradient_cube_y = NULL, CudaCube *gradient_cube_z = NULL, int gradient_direction = 3, int fin_diff_ord = 0);
        void destroy();
};

class CubeEvaluator : public Evaluator {
    private:
        Grid3D *grid;
        CudaCube * input_cube;
    public:
        CubeEvaluator(StreamContainer *streamContainer);
        void setInputCube(CudaCube *input_cube);
        void setInputGrid(Grid3D *grid);
        void evaluatePoints(Points *result_points, Points *gradient_points_x  = NULL, Points *gradient_points_y  = NULL, Points *gradient_points_z = NULL, int gradient_direction =  3);
		void evaluateGrid(Grid3D *grid, CudaCube *result_cube, CudaCube *gradient_cube_x = NULL, CudaCube *gradient_cube_y = NULL, CudaCube *gradient_cube_z = NULL, int gradient_direction = 3, int fin_diff_ord = 0);
        void destroy();
};
#endif
