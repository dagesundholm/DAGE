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
//#include <algorithm> *std::max_element(result_cube, result_cube + totalPointCount)
#include "streamcontainer.h"
#include "spherical_harmonics_cuda.h"
#include "function3d_multiplier.h"
#include "memory_leak_operators.h"
#include "cube.h"
//#include "sys/types.h"
//#include "sys/sysinfo.h"

__host__ inline void check_errs(const char *filename, const int line_number) {
#ifdef DEBUG_CUDA
    cudaThreadSynchronize();
#endif
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      printf("CUDA error at %s:%i: %s\n", filename, line_number, cudaGetErrorString(error));
      exit(-1);
    }
}

/*void check_used_memory_cpu() {

    struct sysinfo memInfo;

    sysinfo (&memInfo);
    long long totalVirtualMem = memInfo.totalram;
    //Add other values in next statement to avoid int overflow on right hand side...
    totalVirtualMem += memInfo.totalswap;
    totalVirtualMem *= memInfo.mem_unit;
    
    long long virtualMemUsed = memInfo.totalram - memInfo.freeram;
    //Add other values in next statement to avoid int overflow on right hand side...
    virtualMemUsed += memInfo.totalswap - memInfo.freeswap;
    virtualMemUsed *= memInfo.mem_unit;
    
    printf("Used Virtual CPU memory: %ld / %ld\n", virtualMemUsed, totalVirtualMem);
}*/


Function3DMultiplier::Function3DMultiplier(Grid3D *grid, StreamContainer *streamContainer) {
    this->grid         = grid;
    bool all_memory_at_all_devices = false;
    bool sliced = true;
    this->cube1        = new CudaCube(streamContainer, grid->shape, all_memory_at_all_devices, sliced); 
    check_errs(__FILE__, __LINE__);
    this->cube2        = new CudaCube(streamContainer, grid->shape, all_memory_at_all_devices, sliced); 
    check_errs(__FILE__, __LINE__);
    this->streamContainer = streamContainer;
}

void Function3DMultiplier::uploadCubes(double *cube1, int cube1_offset, int cube1_host_shape[3], int cube1_lmax,
                                       double *cube2, int cube2_offset, int cube2_host_shape[3], int cube2_lmax) {

    
    if (cube1_lmax <= cube2_lmax) {
        this->cube1->initHost(&cube1[cube1_offset], cube1_host_shape, false);
        check_errs(__FILE__, __LINE__);
        this->cube2->initHost(&cube2[cube2_offset], cube2_host_shape, false);
        check_errs(__FILE__, __LINE__);
        this->result_cube = this->cube2;
    }
    else {
        this->cube1->initHost(&cube2[cube2_offset], cube2_host_shape, false);
        check_errs(__FILE__, __LINE__);
        this->cube2->initHost(&cube1[cube1_offset], cube1_host_shape, false);
        check_errs(__FILE__, __LINE__);
        this->result_cube = this->cube1;
    }
    check_errs(__FILE__, __LINE__);
    this->cube1->upload();
    check_errs(__FILE__, __LINE__);
    this->cube2->upload();
    check_errs(__FILE__, __LINE__);
}

void Function3DMultiplier::setHostResultCube(double *host_cube, int cube_offset, int cube_host_shape[3]) {
    // The multiply function stores the host cube array address to this->result_cube.
    this->result_cube->initHost(&host_cube[cube_offset], cube_host_shape, false);
}

void Function3DMultiplier::downloadResult() {
    // The multiply function stores the result to this->cube1. The result is downloaded to the host 
    // address of this->result_cube
    int download_shape[3];
    download_shape[0] = this->result_cube->getShape(0);
    download_shape[1] = this->result_cube->getShape(1);
    download_shape[2] = this->result_cube->getShape(2);
    this->cube1->downloadSlicedTo(this->result_cube->getHostCube(), download_shape, this->result_cube->getHostPitch(), download_shape);

    check_errs(__FILE__, __LINE__);
    this->cube1->unsetHostCube();
    check_errs(__FILE__, __LINE__);
    this->cube2->unsetHostCube();
    check_errs(__FILE__, __LINE__);
}

void Function3DMultiplier::destroy() {
    this->cube1->destroy();
    delete this->cube1;
    check_errs(__FILE__, __LINE__);
    this->cube2->destroy();
    delete this->cube2;
    check_errs(__FILE__, __LINE__);
    //this->result_cube->destroy();
    //delete this->result_cube;
    //check_errs(__FILE__, __LINE__);
}

/***********************************************************
 *             The Fortran Interfaces                      *
 ***********************************************************/

extern "C" void function3dmultiplier_destroy_cuda(Function3DMultiplier *multiplier) {
    multiplier->destroy();
    delete multiplier;
}

extern "C" Function3DMultiplier *function3dmultiplier_init_cuda(Grid3D *grid, StreamContainer *streamContainer) {
    Function3DMultiplier *new_multiplier = new Function3DMultiplier(grid, streamContainer);
    return new_multiplier;
}

extern "C" void function3dmultiplier_download_result_cuda(Function3DMultiplier *multiplier) {
    multiplier->downloadResult();
}

extern "C" void function3dmultiplier_multiply_cuda(Function3DMultiplier *multiplier, Bubbles *bubbles1, Bubbles *bubbles2, Bubbles *result_bubbles) {
    multiplier->multiply(bubbles1, bubbles2, result_bubbles);
}

extern "C" void function3dmultiplier_upload_cubes_cuda(Function3DMultiplier *multiplier, 
                                                       double *cube1, int cube1_offset, int cube1_host_shape[3], int cube1_lmax,
                                                       double *cube2, int cube2_offset, int cube2_host_shape[3], int cube2_lmax) {
    multiplier->uploadCubes(cube1, cube1_offset, cube1_host_shape, cube1_lmax,
                            cube2, cube2_offset, cube2_host_shape, cube2_lmax);
}

extern "C" void function3dmultiplier_set_host_result_cube_cuda(Function3DMultiplier *multiplier, 
                                                       double *cube, int cube_offset, int cube_host_shape[3]) {
    multiplier->setHostResultCube(cube, cube_offset, cube_host_shape);
}