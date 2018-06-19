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
#ifndef INCLUDE_BUBBLESMULTIPLIER
#define INCLUDE_BUBBLESMULTIPLIER
#include "bubbles_cuda.h"
#include "streamcontainer.h"

class BubblesMultiplier {
    private:
        Bubbles *bubbles1;
        Bubbles *bubbles2;
        Bubbles *taylor_series_bubbles1;
        Bubbles *taylor_series_bubbles2;
        Bubbles *result_bubbles;
        int processor_order_number;
        int number_of_processors;
        int ** device_number_of_terms;
        double ** device_coefficients;
        int ** device_result_lm;
        int ** device_positions;
        bool complex_multiplication;
        StreamContainer *streamContainer;
    public:
        BubblesMultiplier(Bubbles *bubbles1, Bubbles *bubbles2, Bubbles *result_bubbles, 
                          Bubbles *taylor_series_bubbles1, Bubbles *taylor_series_bubbles2, int lmax,
                          double *coefficients, int *number_of_terms, int *result_lm, int *positions,
                          int result_lm_size, int processor_order_number,
                          int number_of_processors, StreamContainer *streamContainer);
        void multiplyBubble(int ibub, Bubbles* bubbles1, Bubbles* bubbles2, Bubbles* result_bubbles, double factor, int first_cell = -1, int last_cell = -1);
        void multiplyBubble(int ibub, double *bubble1_bf, double *bubble2_bf, double *result_bubble_bf,
                            double *taylor_series_bubble1_bf, double *taylor_series_bubble2_bf,
                            int lmax1, int lmax2, int tlmax1, int tlmax2);
        void setK(int bubble1_k, int bubble2_k, int result_bubble_k, int taylor_series_bubble1_k, int taylor_series_bubble2_k);
        Bubbles *getBubbles1();
        Bubbles *getBubbles2();
        Bubbles *getResultBubbles();
        void downloadResult(int lmax, int *ibubs, int nbub);
        void destroy();
};
#endif