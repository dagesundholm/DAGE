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
#include "grid.h"
#include "streamcontainer.h"
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include "memory_leak_operators.h"

#define X_ 0
#define Y_ 1
#define Z_ 2



__host__ inline void check_grid_errors(const char *filename, const int line_number) {
#ifdef DEBUG_CUDA
    cudaThreadSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      printf("CUDA error at %s:%i: %s\n", filename, line_number, cudaGetErrorString(error));
      exit(-1);
    }
#endif
}

/*************************************************** 
 *                   Grid1D implementation         *
 *                                                 *
 ***************************************************/

Grid1D::Grid1D() {
}


Grid1D::Grid1D(int ncell, int nlip, double r_max, double *h, double *d, int grid_type, double *gridpoints, double *lip, double *derivative_lip, double *lower_derivative_lip, double *base_integrals, StreamContainer *streamContainer, bool init_device_memory) {
    this->ncell = ncell;
    this->nlip = nlip;
    this->r_max = r_max;
    this->grid_type = grid_type;
    this->streamContainer = streamContainer;
    // allocate space for device pointers
    this->device_h = new double*[streamContainer->getNumberOfDevices()];
    this->device_d = new double*[streamContainer->getNumberOfDevices()]; 
    this->device_lip = new double*[streamContainer->getNumberOfDevices()];
    this->device_derivative_lip = new double*[streamContainer->getNumberOfDevices()];
    this->device_lower_derivative_lip = new double*[streamContainer->getNumberOfDevices()];
    this->device_gridpoints = new double*[streamContainer->getNumberOfDevices()];
    this->device_integrals = new double*[streamContainer->getNumberOfDevices()];
    this->device_copies = new Grid1D *[streamContainer->getNumberOfDevices()];
    
    // allocate the memory at device
    if (init_device_memory) {
        for (int device = 0; device < streamContainer->getNumberOfDevices(); device ++) {
            streamContainer->setDevice(device);
            size_t sz = sizeof(double)*this->ncell;
            cudaMalloc(&this->device_h[device], sz);
            cudaMalloc(&this->device_d[device], sz);

            
            sz = sizeof(double)*(this->ncell * (this->nlip-1) +1);
            cudaMalloc(&this->device_gridpoints[device], sz);
            cudaMalloc(&this->device_integrals[device], sz);
            
            sz=sizeof(double)*(nlip)*(nlip);
            cudaMalloc(&this->device_lip[device], sz);
            sz=sizeof(double)*(nlip)*(nlip-1);
            cudaMalloc(&this->device_derivative_lip[device], sz);
            sz=sizeof(double)*(nlip-2)*(nlip-1);
            cudaMalloc(&this->device_lower_derivative_lip[device], sz);

            this->h              = this->device_h[device];
            this->d              = this->device_d[device];
            this->gridpoints     = this->device_gridpoints[device];
            this->integrals      = this->device_integrals[device];
            this->lip            = this->device_lip[device];
            this->derivative_lip = this->device_derivative_lip[device];
            this->lower_derivative_lip = this->device_lower_derivative_lip[device];

            cudaMalloc(&this->device_copies[device], sizeof(Grid1D));
            cudaMemcpy(this->device_copies[device], this, sizeof(Grid1D), cudaMemcpyHostToDevice);
        }
        
        // set the host variables and register them for faster data transfer
        cudaHostAlloc((void **)&this->h, sizeof(double)*this->ncell, cudaHostAllocPortable);
        cudaHostAlloc((void **)&this->d, sizeof(double)*this->ncell, cudaHostAllocPortable);
        cudaHostAlloc((void **)&this->lip, sizeof(double)*(nlip)*(nlip), cudaHostAllocPortable);
        cudaHostAlloc((void **)&this->derivative_lip, sizeof(double)*(nlip)*(nlip-1), cudaHostAllocPortable);
        cudaHostAlloc((void **)&this->lower_derivative_lip, sizeof(double)*(nlip-1)*(nlip-2), cudaHostAllocPortable);
        cudaHostAlloc((void **)&this->base_integrals, sizeof(double)*(nlip), cudaHostAllocPortable);
        cudaHostAlloc((void **)&this->gridpoints, sizeof(double)*((nlip-1)*(ncell)+1), cudaHostAllocPortable);
        for (int i = 0; i < this->ncell; i++) {
            this->h[i] = h[i];
            this->d[i] = d[i];
        }
        for (int i = 0; i < nlip*nlip; i++) {
            this->lip[i] = lip[i];
        }
        for (int i = 0; i < (nlip)*(nlip-1); i++) {
            this->derivative_lip[i] = derivative_lip[i];
        }
        for (int i = 0; i < (nlip-1)*(nlip-2); i++) {
            this->lower_derivative_lip[i] = lower_derivative_lip[i];
        }
        for (int i = 0; i < nlip; i++) {
            this->base_integrals[i] = base_integrals[i];
        }
      
        for (int i = 0; i < (nlip-1)*(ncell)+1; i++) {   
            this->gridpoints[i] = gridpoints[i];
        }
        
    }
    else {
        this->h = h;
        this->d = d;
        this->lip = lip;
        this->derivative_lip = derivative_lip;
        this->lower_derivative_lip = lower_derivative_lip;
        this->base_integrals = base_integrals;
        this->gridpoints = gridpoints;
        //this->integrals = this->calculateIntegrals();
    }
    
    this->integrals = this->calculateIntegrals();
    
    // upload the memory to device, if there is any memory allocated
    if (init_device_memory) {
        this->upload();
    }

}

void Grid1D::upload() {
    double *device_h, *device_d, *device_gridpoints, *device_integrals, *host_h, *host_d, *host_gridpoints, *host_integrals;
    int cells_per_stream, gridpoints_per_stream;
    
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        streamContainer->setDevice(device);
        
        // get the preallocated device pointers
        device_h = this->device_h[device];
        device_d = this->device_d[device];
        device_gridpoints = this->device_gridpoints[device];
        device_integrals = this->device_integrals[device];
        
        // NOTE: for all devices the first pointer points to  the first value of each array
        host_h = this->h;
        host_d = this->d;
        host_gridpoints = this->gridpoints;
        host_integrals = this->integrals;
        
        // upload the lip to the device
        cudaMemcpyAsync(this->device_lip[device], this->lip, sizeof(double)*(this->nlip)*(this->nlip), cudaMemcpyHostToDevice, *this->streamContainer->getStream(device, 0));
        cudaMemcpyAsync(this->device_derivative_lip[device], this->derivative_lip, sizeof(double)*(this->nlip)*(this->nlip-1), cudaMemcpyHostToDevice, *this->streamContainer->getStream(device, 0));
        cudaMemcpyAsync(this->device_lower_derivative_lip[device], this->lower_derivative_lip, sizeof(double)*(this->nlip-2)*(this->nlip-1), cudaMemcpyHostToDevice, *this->streamContainer->getStream(device, 0));

        for (int stream = 0; stream < this->streamContainer->getStreamsPerDevice(); stream++) {
            cells_per_stream = (this->ncell /  this->streamContainer->getStreamsPerDevice()) + 
                               ((this->ncell %  this->streamContainer->getStreamsPerDevice()) > stream);
            gridpoints_per_stream = ((this->ncell*(this->nlip-1)+1) /  this->streamContainer->getStreamsPerDevice()) + 
                               (((this->ncell*(this->nlip-1)+1) %  this->streamContainer->getStreamsPerDevice()) > stream);
            // upload the data to device
            cudaMemcpyAsync(device_h, host_h, sizeof(double)*cells_per_stream, cudaMemcpyHostToDevice, *this->streamContainer->getStream(device, stream));
            cudaMemcpyAsync(device_d, host_d, sizeof(double)*cells_per_stream, cudaMemcpyHostToDevice, *this->streamContainer->getStream(device, stream));
            cudaMemcpyAsync(device_gridpoints, host_gridpoints, sizeof(double)*gridpoints_per_stream, cudaMemcpyHostToDevice, *this->streamContainer->getStream(device, stream));
            cudaMemcpyAsync(device_integrals, host_integrals, sizeof(double)*gridpoints_per_stream, cudaMemcpyHostToDevice, *this->streamContainer->getStream(device, stream));
            
            // add to the pointers
            device_h += cells_per_stream;
            device_d += cells_per_stream;
            device_gridpoints += gridpoints_per_stream;
            device_integrals  += gridpoints_per_stream;
            host_h += cells_per_stream;
            host_d += cells_per_stream;
            host_gridpoints += gridpoints_per_stream;
            host_integrals  += gridpoints_per_stream;
            
        }
        
    }
    
    // synchronize the host with both devices
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        this->streamContainer->setDevice(device);
        cudaDeviceSynchronize();
    }
    
}

Grid1D *Grid1D::getSubGrid(int start_cell_index, int end_cell_index, StreamContainer *streamContainer) {
    Grid1D *subgrid = new Grid1D(end_cell_index-start_cell_index, this->nlip, this->r_max, &this->h[start_cell_index],
                 &this->d[start_cell_index], this->grid_type, &this->gridpoints[start_cell_index*(this->nlip-1)],
                 this->lip, this->derivative_lip, this->lower_derivative_lip, this->base_integrals, streamContainer, false);
    subgrid->is_subgrid = true;
    double *host_integrals = subgrid->integrals;
    
    for (int device = 0; device < streamContainer->getNumberOfDevices(); device ++) {
        streamContainer->setDevice(device);
        
        // get the order number of current device in the context of the streamContainer of 'this'
        int device_number = streamContainer->getDeviceNumber(device);
        int device_order_number = this->streamContainer->getDeviceOrderNumber(device_number);
        
        // get the pointers to the device arrays
        subgrid->device_h[device]          = &this->device_h[device_order_number][start_cell_index];
        subgrid->device_d[device]          = &this->device_d[device_order_number][start_cell_index];
        subgrid->device_gridpoints[device] = &this->device_gridpoints[device_order_number][start_cell_index*(subgrid->nlip-1)];
        subgrid->device_lip[device]        = this->device_lip[device_order_number];
        subgrid->device_derivative_lip[device] = this->device_derivative_lip[device_order_number];
        subgrid->device_lower_derivative_lip[device] = this->device_lower_derivative_lip[device_order_number];
        
        // allocate & upload the integrals (cannot be taken from this)
        size_t sz = sizeof(double)*(subgrid->ncell * (subgrid->nlip-1) +1);
        cudaMalloc(&subgrid->device_integrals[device], sz);
        cudaMemcpy(subgrid->device_integrals[device], host_integrals, sz, cudaMemcpyHostToDevice);
        check_grid_errors(__FILE__, __LINE__);
        
        // set the pointers to the device copy
        subgrid->gridpoints = subgrid->device_gridpoints[device];
        subgrid->integrals = subgrid->device_integrals[device];
        subgrid->h = subgrid->device_h[device];
        subgrid->d = subgrid->device_d[device];
        subgrid->lip = subgrid->device_lip[device];
        subgrid->derivative_lip = subgrid->device_derivative_lip[device];
        subgrid->lower_derivative_lip = subgrid->device_lower_derivative_lip[device];
        
        cudaMalloc(&subgrid->device_copies[device], sizeof(Grid1D));
        cudaMemcpy(subgrid->device_copies[device], subgrid, sizeof(Grid1D), cudaMemcpyHostToDevice);
    }
    
    // set the pointers to the host arrays
    subgrid->h          = &this->h[start_cell_index];
    subgrid->d          = &this->d[start_cell_index];
    subgrid->gridpoints = &this->gridpoints[start_cell_index*(subgrid->nlip-1)];
    subgrid->integrals  = host_integrals;
    subgrid->lip        =  this->lip;
    return subgrid;
}

double **Grid1D::getDeviceIntegrals() {
    return this->device_integrals;
}

double *Grid1D::getDeviceIntegrals(int device) {
    return this->device_integrals[device];
}

int Grid1D::getShape() {
    return this->ncell * (this->nlip-1) + 1;
}

double *Grid1D::getIntegrals(int first_cell) {
    return &this->integrals[first_cell*(this->nlip-1)];
}


/*
 * Calculates the values needed for integration of vector of with shape of this grid.
 * 
 * NOTE: this is a host function, meaning that it does not use anything at gpus
 * NOTE: first_cell and last_cell must be in C indexing (starting from 0)
 */
double *Grid1D::calculateIntegrals(int first_cell, int last_cell) {
    if (last_cell == -1) last_cell = this->ncell-1;
    
    // init the result array;
    double *result = new double[(last_cell-first_cell+1)*(this->nlip-1) +1];
    
    // init it to zero
    for (int i = 0; i < (last_cell-first_cell+1)*(this->nlip-1) +1; i++) {
        result[i] = 0.0;
    }
    
    // calculate the values
    for (int i = 0; i < last_cell-first_cell+1; i++) {
        int icell = first_cell +i;
        for (int ilip = 0; ilip < this->nlip; ilip ++) {
            result[icell*(this->nlip-1) + ilip] += this->base_integrals[ilip] * this->h[icell];
        }
    } 
    return result;
}

// destroy all cuda related objects
void Grid1D::destroy() {
    check_errors_and_lock(__FILE__, __LINE__);
    // determine whether this is a subgrid, if not delete everything normally
    if (!this->is_subgrid) {
        for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device++) {
            this->streamContainer->setDevice(device);
            cudaFree(this->device_h[device]);
            cudaFree(this->device_d[device]);
            cudaFree(this->device_gridpoints[device]);
            cudaFree(this->device_lip[device]);
            cudaFree(this->device_derivative_lip[device]);
            cudaFree(this->device_lower_derivative_lip[device]);
        }      
        cudaFreeHost(this->h);
        cudaFreeHost(this->d);
        cudaFreeHost(this->lip);
        cudaFreeHost(this->derivative_lip);
        cudaFreeHost(this->lower_derivative_lip);
        cudaFreeHost(this->base_integrals);
        cudaFreeHost(this->gridpoints); 
    }
    
    // if is a subgrid, delete only the device copy and integrals   
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device++) {
        this->streamContainer->setDevice(device);
        cudaFree(this->device_integrals[device]);
        cudaFree(this->device_copies[device]);
        check_errors_and_lock(__FILE__, __LINE__);
    } 
    
    delete[] this->device_integrals;
    delete[] this->device_gridpoints;
    delete[] this->device_copies;
    delete[] this->device_h;
    delete[] this->device_d;
    delete[] this->device_lip;
    delete[] this->device_derivative_lip;
    delete[] this->device_lower_derivative_lip;
    delete[] this->integrals;
    check_grid_errors(__FILE__, __LINE__);
    check_errors_and_lock(__FILE__, __LINE__);
}

/*************************************************** 
 *                   Grid3D implementation         *
 *                                                 *
 ***************************************************/

Grid3D::Grid3D() {
}


Grid3D::Grid3D(Grid1D **axis, StreamContainer *streamContainer) {
    this->streamContainer = streamContainer;
    Grid1D **temp_axis;
    
    // set the shape parameter
    this->shape[0] = axis[0]->ncell * (axis[0]->nlip-1) +1;
    this->shape[1] = axis[1]->ncell * (axis[0]->nlip-1) +1;
    this->shape[2] = axis[2]->ncell * (axis[0]->nlip-1) +1;
    
    // allocate memory for the pointers of arrays
    this->device_copies = new Grid3D *[streamContainer->getNumberOfDevices()];
    temp_axis = new Grid1D *[3];
    this->device_axis = new Grid1D **[streamContainer->getNumberOfDevices()];
    
    for (int device = 0; device < streamContainer->getNumberOfDevices(); device ++) {
        streamContainer->setDevice(device);
        
        // set the device axis
        temp_axis[0] = axis[0]->device_copies[device];
        temp_axis[1] = axis[1]->device_copies[device];
        temp_axis[2] = axis[2]->device_copies[device];
        
        // copy the device axis to device
        cudaMalloc(&this->device_axis[device], sizeof(Grid1D *)*3);
        cudaMemcpy(this->device_axis[device], temp_axis, sizeof(Grid1D *) * 3, cudaMemcpyHostToDevice);
        
        this->axis = this->device_axis[device];
        
        // allocate the device memory and copy
        cudaMalloc(&this->device_copies[device], sizeof(Grid3D));
        cudaMemcpy(this->device_copies[device], this, sizeof(Grid3D), cudaMemcpyHostToDevice);
        
    }
    temp_axis[0] = axis[0];
    temp_axis[1] = axis[1];
    temp_axis[2] = axis[2];
    // set the host pointers to the returned object
    this->axis = temp_axis;
}


void Grid3D::destroy() {
    // destroy the Grid1D objects owned by this object
    this->axis[0]->destroy();
    check_errors_and_lock(__FILE__, __LINE__);
    delete this->axis[0];
    this->axis[1]->destroy();
    check_errors_and_lock(__FILE__, __LINE__);
    delete this->axis[1];
    this->axis[2]->destroy();
    check_errors_and_lock(__FILE__, __LINE__);
    delete this->axis[2];
    // free the device_copies
    for (int device = 0; device < this->streamContainer->getNumberOfDevices(); device ++) {
        this->streamContainer->setDevice(device);
        check_errors_and_lock(__FILE__, __LINE__);
        cudaFree(this->device_copies[device]);
        check_errors_and_lock(__FILE__, __LINE__);
        cudaFree(this->device_axis[device]);
        check_errors_and_lock(__FILE__, __LINE__);
    }
    check_errors_and_lock(__FILE__, __LINE__);
    // free the host parameters
    delete[] this->device_copies;
    delete[] this->device_axis;
    delete[] this->axis;
    check_grid_errors(__FILE__, __LINE__);
}

Grid3D *Grid3D::getSubGrid(int start_cell_indices[3], int end_cell_indices[3], StreamContainer *streamContainer) {
    Grid3D *subgrid = new Grid3D();
    subgrid->streamContainer = streamContainer;
    Grid1D **temp_axis;
    // allocate memory for the pointers of arrays
    subgrid->device_copies = new Grid3D*[streamContainer->getNumberOfDevices()];
    temp_axis = new Grid1D*[3];
    subgrid->device_axis = new Grid1D **[streamContainer->getNumberOfDevices()];
    
    // mark this grid to be a subgrid
    subgrid->is_subgrid = true;
    
    Grid1D **axis;
    axis = new Grid1D *[3]; 
    
   
    // init the axis
    axis[0] = this->axis[0]->getSubGrid(start_cell_indices[0], end_cell_indices[0], streamContainer);
    axis[1] = this->axis[1]->getSubGrid(start_cell_indices[1], end_cell_indices[1], streamContainer);
    axis[2] = this->axis[2]->getSubGrid(start_cell_indices[2], end_cell_indices[2], streamContainer);
    
    // set the shape parameter
    subgrid->shape[0] = axis[0]->ncell * (axis[0]->nlip-1) +1;
    subgrid->shape[1] = axis[1]->ncell * (axis[1]->nlip-1) +1;
    subgrid->shape[2] = axis[2]->ncell * (axis[2]->nlip-1) +1;
    
    for (int device = 0; device < streamContainer->getNumberOfDevices(); device ++) {
        streamContainer->setDevice(device);
        
        // set the device copies to the axis pointers
        temp_axis[0] = axis[0]->device_copies[device];
        temp_axis[1] = axis[1]->device_copies[device];
        temp_axis[2] = axis[2]->device_copies[device];
        
        // copy the device axis to device
        cudaMalloc(&subgrid->device_axis[device], sizeof(Grid1D *)*3);
        cudaMemcpy( subgrid->device_axis[device], temp_axis, sizeof(Grid1D *) * 3, cudaMemcpyHostToDevice);
        
        subgrid->axis = subgrid->device_axis[device];
        
        // allocate the device memory and copy
        cudaMalloc(&subgrid->device_copies[device], sizeof(Grid3D));
        check_grid_errors(__FILE__, __LINE__);
        cudaMemcpy(subgrid->device_copies[device], subgrid, sizeof(Grid3D), cudaMemcpyHostToDevice);
        check_grid_errors(__FILE__, __LINE__);
    }
    
    temp_axis[0] = axis[0];
    temp_axis[1] = axis[1];
    temp_axis[2] = axis[2];
    
    // set the host pointers to the returned object
    subgrid->axis = temp_axis;
    delete[] axis;
    return subgrid;
}

int *Grid3D::getShape() {
    return this->shape;
}

int Grid3D::getShape(int axis) {
    return this->shape[axis];
}

/*************************************************** 
 *              Fortran interfaces                 *
 *                                                 *
 ***************************************************/

extern "C" Grid1D *grid1d_init_cuda(int ncell, int nlip, double r_max, double *h, double *d, int grid_type, double *gridpoints, double *lip, double *derivative_lip, double *lower_derivative_lip, double *base_integrals, StreamContainer *streamContainer) {
    Grid1D *new_grid = new Grid1D(ncell, nlip, r_max, h, d, grid_type, gridpoints, lip, derivative_lip, lower_derivative_lip, base_integrals, streamContainer);
    return new_grid;
}

extern "C" void grid1d_upload_cuda(Grid1D *grid) {
    grid->upload();
}

extern "C" void grid1d_destroy_cuda(Grid1D *grid) {
    grid->destroy();
    delete grid;
}


extern "C" Grid3D *grid3d_init_cuda(Grid1D **axis, StreamContainer *streamContainer) {
    Grid3D *new_grid = new Grid3D(axis, streamContainer);
    return new_grid;
}

extern "C" void grid3d_destroy_cuda(Grid3D *grid) {
    grid->destroy();
    delete grid;
}
