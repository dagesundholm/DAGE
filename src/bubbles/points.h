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
#ifndef INCLUDE_POINTS
#define INCLUDE_POINTS
#include "streamcontainer.h"
using namespace std;

class PointCoordinates {
    public:
        PointCoordinates **device_copies;
        StreamContainer *stream_container;
        int number_of_points;
        double *coordinates;
        double **device_coordinates;
       
        
        PointCoordinates(double *coordinates, int number_of_points, StreamContainer *stream_container);
        double *initHostPoints();
        void destroyHostPoints(double *host_points);
        // destroy all cuda related objects
        void destroy();
};

class Points {
    public:
        PointCoordinates *point_coordinates;
        double ** device_values;
        Points ** device_copies;
        StreamContainer *stream_container;
        
        Points(PointCoordinates *point_coordinates, StreamContainer *stream_container);
        double *getHostValues();
        void download(double *host_values);
        void setToZero();
        // destroy all cuda related objects and data
        void destroy();
        
};


#endif