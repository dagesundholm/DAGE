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
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include "streamcontainer.h"
#include "memory_leak_operators.h"
#define EVENTS_PER_STREAM 5
#define EVENTS_PER_DEVICE 50

  ///////////////////////////////////
 // StreamContainer Definition    //
///////////////////////////////////

StreamContainer::StreamContainer(int streams_per_device) {
    cudaGetDeviceCount(&this->number_of_devices);
    this->streams_per_device = streams_per_device;
    this->subcontainer = false;
    this->device_event_counter = new int[this->number_of_devices];
    this->stream_event_counter = new int*[this->number_of_devices];
    // allocate space for stream pointers
    cudaHostAlloc((void**)&this->streams, 
                  sizeof(cudaStream_t *) * this->number_of_devices * streams_per_device,
                  cudaHostAllocPortable);
    cudaHostAlloc((void**)&this->streamEvents, 
                  sizeof(cudaEvent_t *) * this->number_of_devices * streams_per_device * EVENTS_PER_DEVICE,
                  cudaHostAllocPortable);
    cudaHostAlloc((void**)&this->deviceEvents, 
                  sizeof(cudaEvent_t *) * this->number_of_devices * streams_per_device * EVENTS_PER_DEVICE,
                  cudaHostAllocPortable);
    this->device_numbers = new int[this->number_of_devices];
    for (int device = 0; device < this->number_of_devices; device ++) {
        this->device_numbers[device] = device;
        cudaSetDevice(device);
        this->device_event_counter[device] = 0;
        this->stream_event_counter[device] = new int[this->streams_per_device];
        // create device events
        for (int event = 0; event < EVENTS_PER_DEVICE; event ++) {
            this->deviceEvents[device * EVENTS_PER_DEVICE + event] = 
                new cudaEvent_t;
            cudaEventCreate(this->deviceEvents[device * EVENTS_PER_DEVICE + event]);
        }
        
        for (int stream = 0; stream < streams_per_device; stream++ ) {
            this->streams[device*streams_per_device + stream] = new cudaStream_t;
            cudaStreamCreate(this->streams[device*streams_per_device + stream]);
            this->stream_event_counter[device][stream] = 0;
            /*for (int event = 0; event < EVENTS_PER_STREAM; event ++) {
                this->streamEvents[device * streams_per_device * EVENTS_PER_STREAM + 
                                             stream * EVENTS_PER_STREAM +
                                             event] = new cudaEvent_t;
                cudaEventCreate(this->streamEvents[device * streams_per_device * EVENTS_PER_STREAM + 
                                             stream * EVENTS_PER_STREAM +
                                             event]);
            }*/
        }
    }
}

/*
 * Init a new streamcontainer by getting the streams of devices with given order numbers ('device_order_numbers').
 */
StreamContainer::StreamContainer(StreamContainer *parentContainer, int *device_order_numbers, int number_of_devices) {
    this->streams_per_device = parentContainer->streams_per_device;
    this->number_of_devices = number_of_devices;
    this->device_numbers = new int[number_of_devices];
    this->subcontainer = true;
    
    // allocate space for stream pointers
    cudaHostAlloc((void**)&this->streams, 
                  sizeof(cudaStream_t *) * this->number_of_devices * this->streams_per_device,
                  cudaHostAllocPortable);
    for (int device = 0; device < this->number_of_devices; device++) {
        this->device_numbers[device] = parentContainer->device_numbers[device_order_numbers[device]];
        for (int stream = 0; stream < this->streams_per_device; stream++ ) {
            this->streams[device*streams_per_device + stream] = parentContainer->getStream(device_order_numbers[device], stream);
        }
    }
}

/*
 * Init a new streamcontainer by getting a part of the parent StreamContainer's streams. the number of devices and 
 * streams used in this streamcontainer is determined by the two input parameters: 'subcontainer_order_number' and
 * 'total_subcontainers' by splitting the streams as evenly as possible between all subcontainers.
 */
StreamContainer::StreamContainer(StreamContainer *parentContainer, int subcontainer_order_number, int total_subcontainers) {
    this->subcontainer = true;
    int devices_per_subcontainer = parentContainer->number_of_devices / total_subcontainers;
    int devices_modulus = parentContainer->number_of_devices % total_subcontainers;
    int subcontainers_per_device = total_subcontainers / parentContainer->getNumberOfDevices();
    int subcontainers_modulus = total_subcontainers % parentContainer->getNumberOfDevices();
    int streams_per_subcontainer, streams_modulus;
    
    if (devices_modulus == 0 && parentContainer->number_of_devices != 0) {
        this->streams_per_device = parentContainer->streams_per_device;
        this->number_of_devices = devices_per_subcontainer;
        this->device_numbers = new int[this->number_of_devices];
        int device_offset = subcontainer_order_number * devices_per_subcontainer;
    
        // allocate space for stream pointers
        cudaHostAlloc((void**)&this->streams, 
                    sizeof(cudaStream_t *) * this->number_of_devices * this->streams_per_device,
                    cudaHostAllocPortable);
        
        this->device_event_counter = &parentContainer->device_event_counter[device_offset];
        this->stream_event_counter = new int*[this->number_of_devices];
        this->deviceEvents = &parentContainer->streamEvents[device_offset * EVENTS_PER_DEVICE];
        this->streamEvents = &parentContainer->streamEvents[device_offset * streams_per_device * EVENTS_PER_STREAM];
        // copy the stream pointers
        for (int device = 0; device < this->number_of_devices; device++) {
            this->stream_event_counter[device] = parentContainer->stream_event_counter[device + device_offset];
            this->device_numbers[device] = parentContainer->device_numbers[device+device_offset];
            for (int stream = 0; stream < this->streams_per_device; stream++ ) {
                this->streams[device*streams_per_device + stream] = parentContainer->getStream(device+device_offset, stream);
            }
        }
        
    }
    // if device_modulus is equal to the number of devices, one device handles more than one subcontainer
    // consequence: only one device per subcontainer
    else if (devices_modulus == parentContainer->number_of_devices) {
        // determine the order number of the device the subcontainer belongs to
        int device_order_number = 0;
        int max_subcontainers = (device_order_number + 1) * subcontainers_per_device + (subcontainers_modulus > device_order_number);
        int device_subcontainer_count = max_subcontainers;
        while (subcontainer_order_number >= max_subcontainers) {
            device_order_number += 1;
            device_subcontainer_count = subcontainers_per_device + (subcontainers_modulus > device_order_number);
            max_subcontainers += device_subcontainer_count;
        }
        
        // set the correct device number
        this->number_of_devices = 1;
        this->device_numbers = new int[1];
        this->device_numbers[0] = parentContainer->device_numbers[device_order_number];
        
        // determine the order number of the stream the subcontainer belongs to
        streams_per_subcontainer = parentContainer->getStreamsPerDevice() / device_subcontainer_count;
        streams_modulus = parentContainer->streams_per_device % device_subcontainer_count;
        
        
                
        // get the order number of subcontainer within device
        int order_number_within_device = subcontainer_order_number - (max_subcontainers - device_subcontainer_count);
        int stream_offset;
        
        if (streams_per_subcontainer == 0 && streams_modulus == 0) {
            printf("ERROR: Invalid number of streams per device in the input StreamContainer: %d.", parentContainer->streams_per_device);
        }
        else if (streams_per_subcontainer == 0 && streams_modulus != 0) {
            int subcontainers_per_stream = device_subcontainer_count / parentContainer->getStreamsPerDevice();
            int subcontainers_per_stream_modulus = device_subcontainer_count % parentContainer->getStreamsPerDevice();
            if (subcontainers_per_stream_modulus == 0) {
                this->streams_per_device = 1;
                stream_offset = order_number_within_device / subcontainers_per_stream;
            }
            else {
                printf("ERROR: unfavourable stream-configuration, more subcontainers for device (%d), than available streams (%d) and subcontainers per device is not equal.\n", 
                    device_subcontainer_count, parentContainer->streams_per_device);

            }
        }
        else if (streams_per_subcontainer > 0) {
            // set the number of streams per device
            this->streams_per_device = streams_per_subcontainer +
                    ((order_number_within_device > streams_modulus || streams_modulus == 0) ? 0 : 1);
            stream_offset = order_number_within_device * streams_per_subcontainer +  
                    ((order_number_within_device > streams_modulus) ? streams_modulus : order_number_within_device);
                
            
            
        }
        else {
            printf("ERROR: unfavourable stream-configuration, more subcontainers for device (%d), than available streams (%d) and subcontainers per device is not equal.\n", 
                    device_subcontainer_count, parentContainer->streams_per_device);
        }
        // allocate space for stream pointers
        cudaHostAlloc((void**)&this->streams, 
                        sizeof(cudaStream_t *) * this->number_of_devices * this->streams_per_device,
                        cudaHostAllocPortable);
        
        this->device_event_counter = &parentContainer->device_event_counter[device_order_number];
        this->stream_event_counter = new int *[this->number_of_devices];
        this->stream_event_counter[0] = &parentContainer->stream_event_counter[device_order_number][stream_offset];
        this->deviceEvents = &parentContainer->streamEvents[device_order_number * EVENTS_PER_DEVICE];
        this->streamEvents = &parentContainer->streamEvents[device_order_number * streams_per_device * EVENTS_PER_STREAM];
            
        // copy the stream pointers
        for (int stream = 0; stream < this->streams_per_device; stream ++) {
            this->streams[stream] = parentContainer->getStream(device_order_number, stream + stream_offset);
        }
    }
    else {
        printf("ERROR: Mixed configuration is not supported, yet!\n");
    }
    
    
}

StreamContainer *StreamContainer::getSingleDeviceContainer(int device_order_number) {
    StreamContainer *result = new StreamContainer(this, &device_order_number, 1);
    return result;
}



cudaStream_t * StreamContainer::getStream(int device_order_number, int stream) {
    return this->streams[device_order_number*this->streams_per_device + stream];
}

/*cudaStream_t * StreamContainer::selectSequentialStream(int order_number) {
    int device = order_number % this->getNumberOfDevices();
    int stream = order_number / this->getStreamsPerDevice();
    cudaSetDevice(device);
    return this->streams[device*this->getStreamsPerDevice() + stream];
}

cudaStream_t * StreamContainer::selectSplitStream(int order_number) {
    int device = order_number / this->getStreamsPerDevice();
    int stream = order_number % this->getStreamsPerDevice();
    cudaSetDevice(device);
    return this->streams[device*this->getStreamsPerDevice() + stream];
}*/

int StreamContainer::getDeviceNumber(int device_order_number) {
    return this->device_numbers[device_order_number];
}

int StreamContainer::getNumberOfDevices() {
    return this->number_of_devices;
}

int StreamContainer::getStreamsPerDevice() {
    return this->streams_per_device;
}

void StreamContainer::setDevice(int device_order_number) {
    cudaSetDevice(this->device_numbers[device_order_number]);
}

int StreamContainer::getDeviceOrderNumber(int device_number) {
    for (int order_number = 0; order_number < this->number_of_devices; order_number++) {
        if (device_number == this->device_numbers[order_number]) return order_number;
    }
    return -1;
}

void StreamContainer::synchronizeAllDevices() {
    // sync the devices
    for (int device = 0; device < this->getNumberOfDevices(); device ++) {
        this->setDevice(device);
        cudaDeviceSynchronize();
    }
}

void StreamContainer::recordAndWaitEvent(int device) {
    this->setDevice(device);
    cudaEvent_t * event = this->recordDeviceEvent(device);
    cudaStreamWaitEvent(0, *event, 0);
    
}

void StreamContainer::recordAndWaitEvent() {
    for (int device; device < this->getNumberOfDevices(); device ++) {
        recordAndWaitEvent(device);
    }
}

void StreamContainer::recordAndWaitEvent(int device, int stream) {
    this->setDevice(device);
    cudaEvent_t * event = this->recordStreamEvent(device, stream);
    cudaStreamWaitEvent(*this->getStream(device, stream), *event, 0);
}

cudaEvent_t *StreamContainer::recordStreamEvent(int device, int stream) {
    cudaEvent_t * event = this->streamEvents[device * streams_per_device * EVENTS_PER_STREAM + 
                                             stream * EVENTS_PER_STREAM +
                                             this->stream_event_counter[device][stream]];
    this->setDevice(device);
    cudaEventRecord(*event);
    
    this->stream_event_counter[device][stream] ++;
    if (this->stream_event_counter[device][stream] == EVENTS_PER_STREAM) {
        this->stream_event_counter[device][stream] = 0;
    }
    return event;
}

cudaEvent_t *StreamContainer::recordDeviceEvent(int device) {
    cudaEvent_t * event = this->deviceEvents[device * EVENTS_PER_DEVICE + this->device_event_counter[device]];
    check_errors(__FILE__, __LINE__);
    this->setDevice(device);
    check_errors(__FILE__, __LINE__);
    cudaEventRecord(*event);
    check_errors(__FILE__, __LINE__);
    
    this->device_event_counter[device] ++;
    if (this->device_event_counter[device] == EVENTS_PER_DEVICE) {
        this->device_event_counter[device] = 0;
    }
    return event;
}

cudaEvent_t **StreamContainer::recordDeviceEvents() {
    cudaEvent_t ** events = new cudaEvent_t*[this->getNumberOfDevices()];
    for (int device = 0; device < this->getNumberOfDevices(); device++) {
        events[device] = this->recordDeviceEvent(device);
    }
    return events;
}

void StreamContainer::synchronizeDevice(int device_order_number) {
    this->setDevice(device_order_number);
    cudaDeviceSynchronize();
}

void StreamContainer::destroy() {
    if (!this->subcontainer) {
        for (int device = 0; device < this->number_of_devices; device ++) {
            this->setDevice(device);
            // destroy the device events
            for (int event = 0; event < EVENTS_PER_DEVICE; event ++) {
                cudaEventDestroy(*this->deviceEvents
                                    [device * EVENTS_PER_DEVICE +
                                     event]);
            }
            
            // destroy the streams and stream events
            for (int stream = 0; stream < this->streams_per_device; stream++ ) {
                cudaStreamDestroy(*(this->streams[device*this->streams_per_device + stream]));
                
                // destroy the stream events
                //for (int event = 0; event < EVENTS_PER_STREAM; event ++) {
                //    cudaEventDestroy(*this->streamEvents
                //                           [device * streams_per_device * EVENTS_PER_STREAM + 
                //                             stream * EVENTS_PER_STREAM +
                //                             event]);
                //}
            }
            delete[] this->stream_event_counter[device];
        }
        delete[] this->device_event_counter;
    }
    cudaFreeHost(this->streams);
    if (!this->subcontainer) {
        cudaFreeHost(this->streamEvents);
        cudaFreeHost(this->deviceEvents);
        delete[] this->stream_event_counter;
        delete[] this->device_numbers;
    }
}

/*************************************************** 
 *              Fortran interfaces                 *
 *                                                 *
 ***************************************************/

extern "C" void streamcontainer_init(StreamContainer **streamContainer, int streams_per_device) {
    *streamContainer = new StreamContainer(streams_per_device);
}

extern "C" void streamcontainer_enable_peer_to_peer_(StreamContainer **streamContainer) {
    for (int device1 = 0; device1 < (*streamContainer)->getNumberOfDevices(); device1 ++) {
        (*streamContainer)->setDevice(device1);
        for (int device2 = 0; device2 < (*streamContainer)->getNumberOfDevices(); device2 ++) {
            int can_access_peer;
            
            // check if it is possible for a 'device1' to access 'device2' 
            cudaDeviceCanAccessPeer(&can_access_peer, device1, device2);
            if (can_access_peer && device1 != device2) {
                cudaDeviceEnablePeerAccess(device2, 0);
                // get the last error
                cudaError_t error = cudaGetLastError();
                
                // if the peer to peer access is already enabled, 
                // do nothing, otherwise use the default error handling
                // to check if there are any other errors
                if (error != cudaErrorPeerAccessAlreadyEnabled) {
                    handleLastError(error, __FILE__, __LINE__);
                }
            }
        }
    }
}

extern "C" StreamContainer* streamcontainer_get_subcontainer(StreamContainer *streamContainer, int subcontainer_order_number, int total_subcontainers) {
    return new StreamContainer(streamContainer, subcontainer_order_number, total_subcontainers);
}

extern "C" int streamcontainer_get_number_of_devices(StreamContainer *streamContainer) {
    return streamContainer->getNumberOfDevices();
}

extern "C" int streamcontainer_get_streams_per_device(StreamContainer *streamContainer) {
    return streamContainer->getStreamsPerDevice();
}

extern "C" cudaEvent_t* streamcontainer_record_device_event(StreamContainer *streamContainer, int device_order_number) {
    return streamContainer->recordDeviceEvent(device_order_number-1);
}

extern "C" cudaEvent_t** streamcontainer_record_device_events(StreamContainer *streamContainer) {
    return streamContainer->recordDeviceEvents();
}

extern "C" void streamcontainer_destroy(StreamContainer *streamContainer) {
    streamContainer->destroy();
    delete streamContainer;
}

extern "C" void check_cuda_errors_from_fortran_(bool lock) {
    if (lock) {
        cudaThreadSynchronize();
    }

    cudaError_t error = cudaGetLastError();
    handleLastError(error, "Fortran", 0);
}
