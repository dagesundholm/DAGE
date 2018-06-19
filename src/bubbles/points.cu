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

#include "points.h"
#include "streamcontainer.h"
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

/*************************************************** 
 *         PointCoordinates implementation         *
 *                                                 *
 ***************************************************/


PointCoordinates::PointCoordinates(double *coordinates, int number_of_points, StreamContainer *stream_container) {

    this->stream_container = stream_container;
    this->number_of_points = number_of_points;
    double *host_coordinates = coordinates;
    
    // allocate memory for the pointers of arrays
    this->device_coordinates = new double*[stream_container->getNumberOfDevices()];
    
    
    // calculate the number of warps calculated
    int warp_size = 32;
    int total_warp_count = this->number_of_points / warp_size + ((this->number_of_points % warp_size) > 0);
    int point_offset = 0;
    for (int device = 0; device < this->stream_container->getNumberOfDevices(); device ++) {
        this->stream_container->setDevice(device);
        
        // allocate space for device results and device points
        int device_warp_count =  total_warp_count / this->stream_container->getNumberOfDevices()
                                  + ((total_warp_count % stream_container->getNumberOfDevices()) > device); 
        int device_point_count = device_warp_count * warp_size;
        int device_point_offset = 0;
        
        cudaMalloc(&this->device_coordinates[device], 3 * device_point_count * sizeof(double));
        
        
        double  *device_coordinates_ptr = this->device_coordinates[device];
        for (int stream = 0; stream < this->stream_container->getStreamsPerDevice(); stream ++) {
            // get the number of points that are in the responsibility of this stream
            int stream_warp_count =  device_warp_count / this->stream_container->getStreamsPerDevice()
                                  + ((device_warp_count % stream_container->getStreamsPerDevice()) > stream); 
            int stream_point_count = stream_warp_count * 32;
            
            // make sure that the last stream does not go over board
            if (stream_point_count + point_offset > number_of_points) {
                stream_point_count = number_of_points - point_offset;
            }
           
                                  
            // copy the x coordinates
            cudaMemcpy2DAsync  (device_coordinates_ptr, sizeof(double),
                host_coordinates, sizeof(double) * 3, sizeof(double),
                stream_point_count, cudaMemcpyHostToDevice,
                *this->stream_container->getStream(device, stream));     
            
            // copy the y coordinates
            cudaMemcpy2DAsync  (&device_coordinates_ptr[device_point_count], sizeof(double),
                &host_coordinates[1], sizeof(double) * 3, sizeof(double),
                stream_point_count, cudaMemcpyHostToDevice,
                *this->stream_container->getStream(device, stream));
            
            // copy the z coordinates
            cudaMemcpy2DAsync  (&device_coordinates_ptr[device_point_count*2], sizeof(double),
                &host_coordinates[2], sizeof(double) * 3, sizeof(double),
                stream_point_count, cudaMemcpyHostToDevice,
                *this->stream_container->getStream(device, stream));
            
            // add the pointers
            device_coordinates_ptr += stream_point_count;
            host_coordinates += stream_point_count * 3;
            point_offset += stream_point_count;
            device_point_offset += stream_point_count;
        }
    }
    
}

double *PointCoordinates::initHostPoints() {
    double *host_values;
    cudaHostAlloc((void **)&host_values, 
                  sizeof(double) * this->number_of_points,
                  cudaHostAllocPortable);
    return host_values;
}

void PointCoordinates::destroyHostPoints(double *host_values) {
   check_errors(__FILE__, __LINE__);
   cudaFreeHost(host_values);
   check_errors(__FILE__, __LINE__);
}

void PointCoordinates::destroy() {
    // free the device_copies
    for (int device = 0; device < this->stream_container->getNumberOfDevices(); device ++) {
        this->stream_container->setDevice(device);
        cudaFree(this->device_coordinates[device]);
    }
    // free the host parameters
    delete[] this->device_coordinates;
}

/*************************************************** 
 *         Points implementation                   *
 *                                                 *
 ***************************************************/

Points::Points(PointCoordinates *point_coordinates, StreamContainer *stream_container) {
    this->stream_container = stream_container;
    this->point_coordinates = point_coordinates;
    
    // allocate memory for the pointers of arrays
    this->device_copies = new Points*[stream_container->getNumberOfDevices()];
    this->device_values = new double*[stream_container->getNumberOfDevices()];
    
    for (int device = 0; device < stream_container->getNumberOfDevices(); device ++) {
        stream_container->setDevice(device);
        
        // copy the coordinates to device
        cudaMalloc(&this->device_values[device], sizeof(double)*this->point_coordinates->number_of_points);
        
        // allocate the device memory and copy
        cudaMalloc(&this->device_copies[device], sizeof(Points));
        cudaMemcpy(this->device_copies[device], this, sizeof(Points), cudaMemcpyHostToDevice);
        
        check_errors(__FILE__, __LINE__);
    }
    check_errors(__FILE__, __LINE__);
}

void Points::setToZero() {
    int warp_size = 32;
    int total_warp_count = this->point_coordinates->number_of_points / warp_size + ((this->point_coordinates->number_of_points % warp_size) > 0);
    int point_offset = 0;
    for (int device = 0; device < this->stream_container->getNumberOfDevices(); device ++) {
        this->stream_container->setDevice(device);
        
        // allocate space for device results and device points
        int device_warp_count = total_warp_count / this->stream_container->getNumberOfDevices()
                                  + ((total_warp_count % this->stream_container->getNumberOfDevices()) > device); 
        int device_point_offset = 0;
        
        // get the pointers to the device points & results
        double  *device_values_ptr = this->device_values[device];
        
        for (int stream = 0; stream < this->stream_container->getStreamsPerDevice(); stream ++) {
            // get the number of points that are in the responsibility of this stream
            int stream_warp_count =  device_warp_count / this->stream_container->getStreamsPerDevice()
                                  + ((device_warp_count % stream_container->getStreamsPerDevice()) > stream); 
            int stream_point_count = stream_warp_count * warp_size;
            
            // make sure that the last stream does not go over board
            if (stream_point_count + point_offset > this->point_coordinates->number_of_points) {
                stream_point_count = this->point_coordinates->number_of_points - point_offset;
            }
            
            if (stream_point_count > 0) {
                // set the result to zero
                cudaMemsetAsync(&device_values_ptr[device_point_offset], 0, stream_point_count * sizeof(double),
                                *this->stream_container->getStream(device, stream) );
            }
            // add the pointers
            point_offset += stream_point_count;
            device_point_offset += stream_point_count;
        }
        check_errors(__FILE__, __LINE__);
    }
    check_errors(__FILE__, __LINE__);
}



/*
 * Starts the downloading of the results of points values from this->device_values to host_values.
 * NOTE: Does not synchronize the cpus with the gpus.
 */
void Points::download(double *host_values) {
    int warp_size = 32;
    int total_warp_count = this->point_coordinates->number_of_points / warp_size + ((this->point_coordinates->number_of_points % warp_size) > 0);
    int point_offset = 0;
    check_errors(__FILE__, __LINE__);
    for (int device = 0; device < this->stream_container->getNumberOfDevices(); device ++) {
        this->stream_container->setDevice(device);
        
        // allocate space for device results and device points
        int device_warp_count =  total_warp_count / this->stream_container->getNumberOfDevices()
                                  + ((total_warp_count % stream_container->getNumberOfDevices()) > device); 
        
        // get the pointers to the device points & results
        double *device_points_ptr = this->device_values[device];
        
        for (int stream = 0; stream < this->stream_container->getStreamsPerDevice(); stream ++) {
            // get the number of points that are in the responsibility of this stream
            int stream_warp_count =  device_warp_count / this->stream_container->getStreamsPerDevice()
                                  + ((device_warp_count % stream_container->getStreamsPerDevice()) > stream); 
            int stream_point_count = stream_warp_count * warp_size;
            
            // make sure that the last stream does not go over board
            if (stream_point_count + point_offset > this->point_coordinates->number_of_points) {
                stream_point_count = this->point_coordinates->number_of_points - point_offset;
            }
            if (stream_point_count > 0) {
                // do the asynchronous copy
                cudaMemcpyAsync(&host_values[point_offset],
                    device_points_ptr,
                    sizeof(double) * stream_point_count,
                    cudaMemcpyDeviceToHost,
                    *this->stream_container->getStream(device, stream)); 
            }
            // add the pointer
            device_points_ptr += stream_point_count;
            point_offset += stream_point_count;
        }
        check_errors(__FILE__, __LINE__);
    }
    check_errors(__FILE__, __LINE__);
    
}




void Points::destroy() {
    check_errors(__FILE__, __LINE__);
    
    // free the device_copies
    for (int device = 0; device < this->stream_container->getNumberOfDevices(); device ++) {
        this->stream_container->setDevice(device);
        cudaFree(this->device_copies[device]);
        cudaFree(this->device_values[device]);
    }
    // free the host parameters
    delete[] this->device_copies;
    delete[] this->device_values;
    check_errors(__FILE__, __LINE__);
}


/*************************************************** 
 *   Fortran interfaces for PointCoordinates       *
 *                                                 *
 ***************************************************/

extern "C" PointCoordinates *pointcoordinates_init_cuda(double *coordinates, int number_of_points, StreamContainer *stream_container) {
    PointCoordinates *new_point_coordinates = new PointCoordinates(coordinates, number_of_points, stream_container);
    return new_point_coordinates;
}


extern "C" void pointcoordinates_destroy_cuda(PointCoordinates *point_coordinates) {
    point_coordinates->destroy();
    delete point_coordinates;
}

extern "C" double *pointcoordinates_init_host_points_cuda(PointCoordinates *point_coordinates) {
    return point_coordinates->initHostPoints();
}

extern "C" void pointcoordinates_destroy_host_points_cuda(double *host_points) {
   check_errors(__FILE__, __LINE__);
   cudaFreeHost(host_points);
   check_errors(__FILE__, __LINE__);
}

/*************************************************** 
 *   Fortran interfaces for PointCoordinates       *
 *                                                 *
 ***************************************************/

extern "C" Points *points_init_cuda(PointCoordinates *point_coordinates, StreamContainer *stream_container) {
    Points *new_points = new Points(point_coordinates, stream_container);
    return new_points;
}

extern "C" void points_destroy_cuda(Points *points) {
    points->destroy();
    delete points;
}


extern "C" void points_set_to_zero_cuda(Points *points) {
    points->setToZero();
}

extern "C" void points_download_cuda(Points *points, double *host_values) {
    points->download(host_values);
}




