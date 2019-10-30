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
#include <stdlib.h>
#include <stdio.h>
#include "../bubbles/grid.h"
#include "../bubbles/streamcontainer.h"
#include "gbfmm_potential_operator.h"
#include "../bubbles/cube.h"
#include "../bubbles/integrator.h"
#include "../bubbles/spherical_harmonics_cuda.h"
#include "../bubbles/memory_leak_operators.h"
#include "../bubbles/cuda_profiling.h"
#define X_ 0
#define Y_ 1
#define Z_ 2
#define BLOCK_SIZE 512

/*************************************************** 
 *       GBFMMPotentialOperator abstract class     *
 *         function implementations                *
 *                                                 *
 ***************************************************/

/* 
 * Inits the things common in gbfmm potential operators
 */
void GBFMMPotentialOperator::initGBFMMPotentialOperator(
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
               StreamContainer *streamContainer) {
    this->lmax = lmax;
    this->grid_in = grid_in;
    this->grid_out = grid_out;
    this->domain[0] = domain[0];
    this->domain[1] = domain[1];
    this->is_child_operator = false;

    // init the arrays of grids and streamcontainers 
    this->input_grids = new Grid3D*[domain[1]-domain[0]+1];
    this->output_grids = new Grid3D*[domain[1]-domain[0]+1];
    this->device_grids = new Grid3D*[streamContainer->getNumberOfDevices()];
    this->integrators = new Integrator3D*[domain[1]-domain[0]+1];
    this->streamContainers = new StreamContainer*[domain[1]-domain[0]+1];
    this->device_containers = new StreamContainer*[streamContainer->getNumberOfDevices()];
    this->harmonics = new RealCubeHarmonics *[streamContainer->getNumberOfDevices()]; 
    this->device_expansions = new double*[streamContainer->getNumberOfDevices()];
    
    this->streamContainer = streamContainer;
    
    // init arrays for end and start cell-indices of the main grid
    this->input_start_indices_x = new int[domain[1]-domain[0]+1];
    this->input_end_indices_x = new int[domain[1]-domain[0]+1];
    this->input_start_indices_y = new int[domain[1]-domain[0]+1];
    this->input_end_indices_y = new int[domain[1]-domain[0]+1];
    this->input_start_indices_z = new int[domain[1]-domain[0]+1];
    this->input_end_indices_z = new int[domain[1]-domain[0]+1];
    
    
    this->output_start_indices_x = new int[domain[1]-domain[0]+1];
    this->output_end_indices_x = new int[domain[1]-domain[0]+1];
    this->output_start_indices_y = new int[domain[1]-domain[0]+1];
    this->output_end_indices_y = new int[domain[1]-domain[0]+1];
    this->output_start_indices_z = new int[domain[1]-domain[0]+1];
    this->output_end_indices_z = new int[domain[1]-domain[0]+1];
    
    this->centers = new double[3 * domain[1]-domain[0]+1];
    
    // init the subgrids and the streamcontainers for each domain box
    for (int i = 0; i <= domain[1]-domain[0]; i++) {
        // get a sub-streamcontainer 
        this->streamContainers[i] = new StreamContainer(this->streamContainer, i, domain[1]-domain[0]+1);
        
        // get the subgrids
        int input_start_indices[3] = {input_start_indices_x[i], input_start_indices_y[i], input_start_indices_z[i]};
        int input_end_indices[3]   = {input_end_indices_x[i],   input_end_indices_y[i],   input_end_indices_z[i]};
        int output_start_indices[3] = {output_start_indices_x[i], output_start_indices_y[i], output_start_indices_z[i]};
        int output_end_indices[3]   = {output_end_indices_x[i],   output_end_indices_y[i],   output_end_indices_z[i]};
        
        this->input_grids[i] = this->grid_in->getSubGrid(input_start_indices, input_end_indices, this->streamContainers[i]);
        this->output_grids[i] = this->grid_out->getSubGrid(output_start_indices, output_end_indices, this->streamContainers[i]);
        
        
        // store the starting and ending indices of input grid boxes in x, y and z direction
        this->input_start_indices_x[i] = input_start_indices_x[i];
        this->input_start_indices_y[i] = input_start_indices_y[i];
        this->input_start_indices_z[i] = input_start_indices_z[i];
        this->input_end_indices_x[i] = input_end_indices_x[i];
        this->input_end_indices_y[i] = input_end_indices_y[i];
        this->input_end_indices_z[i] = input_end_indices_z[i];
        
        // store the starting and ending indices of output grid boxes in x, y and z direction
        this->output_start_indices_x[i] = output_start_indices_x[i];
        this->output_start_indices_y[i] = output_start_indices_y[i];
        this->output_start_indices_z[i] = output_start_indices_z[i];
        this->output_end_indices_x[i] = output_end_indices_x[i];
        this->output_end_indices_y[i] = output_end_indices_y[i];
        this->output_end_indices_z[i] = output_end_indices_z[i];
        
        // calculate the center of the box
        this->centers[i*3 + X_] = (this->input_grids[i]->axis[X_]->gridpoints
                                    [this->input_grids[i]->axis[X_]->ncell * (this->input_grids[i]->axis[X_]->nlip - 1)]
                                 + this->input_grids[i]->axis[X_]->gridpoints[0]) / 2.0;
        this->centers[i*3 + Y_] = (this->input_grids[i]->axis[Y_]->gridpoints
                                    [this->input_grids[i]->axis[Y_]->ncell * (this->input_grids[i]->axis[Y_]->nlip - 1)]
                                 + this->input_grids[i]->axis[Y_]->gridpoints[0]) / 2.0;
        this->centers[i*3 + Z_] = (this->input_grids[i]->axis[Z_]->gridpoints
                                    [this->input_grids[i]->axis[Z_]->ncell * (this->input_grids[i]->axis[Z_]->nlip - 1)]
                                 + this->input_grids[i]->axis[Z_]->gridpoints[0]) / 2.0;
                                 
                                 
 
    }
    
    this->initIntegrators();
    
    // get the solid harmonics evaluation subgrid
    int start_indices[3] = {input_start_indices_x[0], input_start_indices_y[0], input_start_indices_z[0]};
    int end_indices[3]   = {input_end_indices_x[0],   input_end_indices_y[0],   input_end_indices_z[0]};
    
    check_errors(__FILE__, __LINE__);
    // initialize the solid-harmonics evaluators.
    // NOTE: this assumes that each of the boxes have the same shape and that the multipole center is at the center of the
    // box. If the cube-grid is changed to be non-equidistant at some point, this must be changed to be box-wise.
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        // get a streamcontainer that contains only the streams of this device
        this->device_containers[device] = this->streamContainer->getSingleDeviceContainer(device);
        
        // allocate space for device local expansions 
        this->streamContainer->setDevice(device);
        cudaMalloc(&device_expansions[device], (domain[1]-domain[0]+1) * (this->lmax +1) * (this->lmax+1) * sizeof(double));

    }
    check_errors(__FILE__, __LINE__);
    
    // init the device grids
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        this->device_grids[device] = this->grid_in->getSubGrid(start_indices, end_indices, this->device_containers[device]);
    }
    check_errors(__FILE__, __LINE__);
}

void GBFMMPotentialOperator::initIntegrators() {
    // init the subgrids and the streamcontainers for each domain box
    for (int i = 0; i <= domain[1]-domain[0]; i++) {
        // initialize the Integrator needed for multipole evaluation with a buffer for (this->lmax+1)*(this->lmax+1) results 
        this->integrators[i] = new Integrator3D(this->streamContainers[i], this->input_grids[i], (51)*(51));
    }
}


 /* 
 * Inits the things common in gbfmm potential operators from an exising operator
 */
void GBFMMPotentialOperator::initGBFMMPotentialOperator(
               // the operator used as the basis of the new operator
               GBFMMPotentialOperator *parent_operator,
               // the maximum angular momentum quantum number 'l' value
               int lmax) {
    this->lmax = lmax;
    this->grid_in = parent_operator->grid_in;
    this->grid_out = parent_operator->grid_out;
    this->domain[0] = parent_operator->domain[0];
    this->domain[1] = parent_operator->domain[1];
    this->is_child_operator = true;
    this->parent_operator = parent_operator;

    // reuse the parent_operator's common stuff to avoid extra memory usage
    this->input_grids = parent_operator->input_grids;
    this->output_grids = parent_operator->output_grids;
    this->device_grids = parent_operator->device_grids;
    this->device_containers = parent_operator->device_containers;
    this->streamContainer = parent_operator->streamContainer;
    this->streamContainers = parent_operator->streamContainers;
    this->centers = parent_operator->centers;
    
    // init the arrays for harmonics and device expansions
    this->device_expansions = new double*[this->streamContainer->getNumberOfDevices()];
    check_errors(__FILE__, __LINE__);
    
    // init arrays for end and start cell-indices of the main grid
    this->input_start_indices_x = parent_operator->input_start_indices_x;
    this->input_end_indices_x = parent_operator->input_end_indices_x;
    this->input_start_indices_y =  parent_operator->input_start_indices_y;
    this->input_end_indices_y = parent_operator->input_end_indices_y;
    this->input_start_indices_z = parent_operator->input_start_indices_z;
    this->input_end_indices_z = parent_operator->input_end_indices_z;
    this->output_start_indices_x = parent_operator->output_start_indices_x;
    this->output_end_indices_x = parent_operator->output_end_indices_x;
    this->output_start_indices_y =  parent_operator->output_start_indices_y;
    this->output_end_indices_y = parent_operator->output_end_indices_y;
    this->output_start_indices_z = parent_operator->output_start_indices_z;
    this->output_end_indices_z = parent_operator->output_end_indices_z;
    
    
    this->integrators = new Integrator3D*[domain[1]-domain[0]+1];
    this->initIntegrators();
    
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        // allocate space for device expansions 
        this->streamContainer->setDevice(device);
        cudaMalloc(&device_expansions[device], (domain[1]-domain[0]+1) * (this->lmax +1) * (this->lmax+1) * sizeof(double));

    }
    check_errors(__FILE__, __LINE__);
    
}


/*
 * Downloads multipole moments between l:0-this->lmax for each box belonging
 * to the domain of this node.
 * 
 * NOTE: this functions IS BLOCKING with regards to CUDA
 */
void GBFMMPotentialOperator::downloadMultipoleMoments(double *host_multipole_moments) {
    // do the evaluation for the boxes
    host_multipole_moments = &host_multipole_moments[(domain[0]-1) * (this->lmax+1)*(this->lmax+1)];
    for (int i = 0; i <= this->domain[1]-this->domain[0]; i++) {
        for (int n = 0; n < (this->lmax+1)*(this->lmax+1); n++) {
            // download the result for box i and lm-index: n
            this->integrators[i]->downloadResult(host_multipole_moments);
            host_multipole_moments++;
        }
    }
}

/*
 * Uploads input cube for boxes between to the correct device
 * 
 * NOTE: this function IS NOT BLOCKING with regards to CUDA
 */
void GBFMMPotentialOperator::uploadDomainBoxes(CudaCube *input_cube) {
    
    // init the subcubes representing boxes
    CudaCube **sub_cubes = new CudaCube*[this->domain[1]-this->domain[0]+1];
    for (int i = 0; i <= this->domain[1]-this->domain[0]; i++) {
        int start_indices[3] = {this->input_start_indices_x[i], this->input_start_indices_y[i], this->input_start_indices_z[i]};
        int end_indices[3] =   {this->input_end_indices_x[i],   this->input_end_indices_y[i],   this->input_end_indices_z[i]  };
        sub_cubes[i] = input_cube->getSubCube(start_indices, end_indices, this->streamContainers[i]);
    }
    
    // dupload the input_cube for the boxes
    for (int i = 0; i <= this->domain[1]-this->domain[0]; i++) {
        // upload the box to the GPU's
        sub_cubes[i]->upload();
    }
    
    // delete the subcubes
    for (int i = 0; i <= this->domain[1]-this->domain[0]; i++) {
        sub_cubes[i]->destroy();
        delete sub_cubes[i];
    }
    delete[] sub_cubes;
}


/*
 * Evaluates multipole moments between l:0-this->lmax for each box belonging
 * to the domain of this node.
 * 
 * NOTE: this function IS NOT BLOCKING with regards to CUDA
 */
void GBFMMPotentialOperator::calculateMultipoleMoments(CudaCube *input_cube) {
    // init the subcubes representing boxes
    CudaCube **sub_cubes = new CudaCube*[this->domain[1]-this->domain[0]+1];
    for (int i = 0; i <= this->domain[1]-this->domain[0]; i++) {
        int start_indices[3] = {this->input_start_indices_x[i] * (this->input_grids[i]->axis[X_]->nlip-1),
                                this->input_start_indices_y[i] * (this->input_grids[i]->axis[Y_]->nlip-1),
                                this->input_start_indices_z[i] * (this->input_grids[i]->axis[Z_]->nlip-1)    };
        int end_indices[3] =   {this->input_end_indices_x[i] * (this->input_grids[i]->axis[X_]->nlip-1) + 1, 
                                this->input_end_indices_y[i] * (this->input_grids[i]->axis[Y_]->nlip-1) + 1,
                                this->input_end_indices_z[i] * (this->input_grids[i]->axis[Z_]->nlip-1) + 1  };
        sub_cubes[i] = input_cube->getSubCube(start_indices, end_indices, this->streamContainers[i]);
    }
    
    // do the evaluation for the boxes
    for (int i = 0; i <= this->domain[1]-this->domain[0]; i++) {
        // evaluate the potential within the box
        this->calculateMultipoleMomentsBox(i, sub_cubes[i]);
    }
    
    // delete the subcubes
    for (int i = 0; i <= this->domain[1]-this->domain[0]; i++) {
        sub_cubes[i]->destroy();
        delete sub_cubes[i];
    }
    delete[] sub_cubes;
}


/*
 * Evaluates local expansion at output_cube's points for each of the boxes belonging
 * to the domain of this node.
 * 
 * NOTE: the input pointer 'local_expansion' must point to the l=0, m=0, box=0 value of the
 * local expansion in the HOST memory.
 */
void GBFMMPotentialOperator::evaluatePotentialLE(double *local_expansion, CudaCube *output_cube) {
    // init the subcubes representing boxes
    CudaCube **sub_cubes = new CudaCube*[this->domain[1]-this->domain[0]+1];
    
    // set the pointer to the start of the domain belonging to this node
    local_expansion = &local_expansion[(this->domain[0]-1) * (this->lmax+1) * (this->lmax+1)];
    
    for (int i = 0; i <= this->domain[1]-this->domain[0]; i++) {
        int start_indices[3] = {this->output_start_indices_x[i] * (this->output_grids[i]->axis[X_]->nlip - 1),
                                this->output_start_indices_y[i] * (this->output_grids[i]->axis[Y_]->nlip - 1),
                                this->output_start_indices_z[i] * (this->output_grids[i]->axis[Z_]->nlip - 1)};
        int end_indices[3] =   {this->output_end_indices_x[i] * (this->output_grids[i]->axis[X_]->nlip - 1) + 1,
                                this->output_end_indices_y[i] * (this->output_grids[i]->axis[Y_]->nlip - 1) + 1,
                                this->output_end_indices_z[i] * (this->output_grids[i]->axis[Z_]->nlip - 1) + 1 };
        sub_cubes[i] = output_cube->getSubCube(start_indices, end_indices, this->streamContainers[i]);
    }
    
    check_errors(__FILE__, __LINE__);
    // do the evaluation for the boxes
    for (int i = 0; i <= this->domain[1]-this->domain[0]; i++) {
        // evaluate the potential within the box
        evaluatePotentialLEBox(i, 
                               local_expansion, 
                               &this->centers[i*3],
                               this->output_grids[i],
                               sub_cubes[i],
                               this->streamContainers[i]);
        // add the local_expansion pointer to be ready to handle the next box
        local_expansion = &local_expansion[(this->lmax+1)*(this->lmax+1)];
        // check_errors(__FILE__, __LINE__);
        
    }
    // delete the subcubes
    for (int i = 0; i <= this->domain[1]-this->domain[0]; i++) {
        sub_cubes[i]->destroy();
        delete[] sub_cubes[i];
    }
    delete[] sub_cubes;
}

/*
 * Destroys the common objects of the GBFMMPotentialOperator object
 */
void GBFMMPotentialOperator::destroy() {
    // check if this operator is independent or child. If is independent
    // delete all stuff.
    if (!this->is_child_operator) {
        delete[] this->input_start_indices_x;
        delete[] this->input_end_indices_x;
        delete[] this->input_start_indices_y;
        delete[] this->input_end_indices_y;
        delete[] this->input_start_indices_z;
        delete[] this->input_end_indices_z;
        delete[] this->output_start_indices_x;
        delete[] this->output_end_indices_x;
        delete[] this->output_start_indices_y;
        delete[] this->output_end_indices_y;
        delete[] this->output_start_indices_z;
        delete[] this->output_end_indices_z;
        delete[] this->centers;
        // destroy the grids, streamscontainers 
        for (int i = 0; i <= this->domain[1]-this->domain[0]; i++) {
            this->input_grids[i]->destroy();
            this->output_grids[i]->destroy();
            this->streamContainers[i]->destroy();
        }
        
        // destroy the device containers and grids from all devices
        for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
            this->streamContainer->setDevice(device);
            this->device_containers[device]->destroy();
            this->device_grids[device]->destroy();
        }
        
        
        delete[] this->input_grids;
        delete[] this->output_grids;
        delete[] this->device_grids;
        delete[] this->streamContainers;
        delete[] this->device_containers;
    }
    // destroy the integrators
    for (int i = 0; i <= this->domain[1]-this->domain[0]; i++) {
        this->integrators[i]->destroy();
    }
    delete[] this->integrators;
    
    // destroy the harmonics 
    //this->destroyHarmonics();
    
    // destroy the device expansions from all devices
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        this->streamContainer->setDevice(device);
        cudaFree(this->device_expansions[device]);
    }
    
    delete[] this->device_expansions;
}

/*
 * Returns the streamContainer associated with box ibox.
 * 
 * NOTE: box indexing starts from 1
 */
StreamContainer *GBFMMPotentialOperator::getBoxStreamContainer(int ibox) {
    int i = ibox-this->domain[0];
    return this->streamContainers[i];
}
