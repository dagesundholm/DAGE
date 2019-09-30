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
#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

extern "C" void matrix_elemental_multiply(const size_t * row_count, 
                                          const size_t * column_count, 
                                          const size_t * pitch,
                                          const double *matrix_1, 
                                          const double *matrix_2, 
                                          double * result_matrix
                                          const size_t * matrix_1_pitch,
                                          const size_t * matrix_2_pitch,
                                          const size_t * result_matrix_pitch) {
    
    int num_elements = (*row_count) * (*column_count)
    int block_size = 512
    int grid_size = num_elements / block_size;

    multiply <<<grid_size,block_size>>> (
        row_count, column_count,  
        matrix_1, matrix_2, result_matrix, 
        matrix_1_pitch, matrix_2_pitch, result_matrix_pitch
    )
    
}

__global__ void multiply(const size_t * row_count, 
                         const size_t * column_count, 
                         const double * matrix_1,
                         const double * matrix_2,
                         double * result_matrix,
                         const size_t * matrix_1_pitch,
                         const size_t * matrix_2_pitch,
                         const size_t * result_matrix_pitch) {
    
    for (int row = 0; row < row_count; row++)  {  
        // update the pointer to point to the beginning of the next row  
        float* m1RowData = (float*)(((char*)matrix_1) + (row * matrix_1_pitch));  
        float* m2RowData = (float*)(((char*)matrix_2) + (row * matrix_2_pitch));  
        float* resultRowData = (float*)(((char*)result_matrix) + (row * result_matrix_pitch));  
        
        for (int column = 0; column < column_count; column++)  {  
            resultRowData[column] = m1RowData[column] * m2RowData[column];  
        }  
    }  
}