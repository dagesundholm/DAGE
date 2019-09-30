/*! @file gpu_info.cpp
 *! @brief print gpu info
 */
#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>


void print_device_props_short()
{
    const size_t kb = 1024;
    const size_t mb = kb * kb;

    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("Found the following GPUs:\n");

    for(int i = 0; i < devCount; ++i)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        printf("  | %d: %s\n", i, props.name);
        printf("  | arch version: %d.%d\n", props.major, props.minor);
        printf("  | Global memory: %d MB\n", props.totalGlobalMem / mb);
    }
}

void print_device_props_complete()
{
    const int kb = 1024;
    const int mb = kb * kb;

    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("Found the following GPUs:\n");

    for(int i = 0; i < devCount; ++i)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        printf("  | %d: %s\n", i, props.name);
        printf("  | Arch version: %d.%d\n", props.major, props.minor);
        printf("  | Global memory: %d MB\n", props.totalGlobalMem / mb);
        printf("  | Shared memory: %d KB\n", props.sharedMemPerBlock / kb);
        printf("  | Constant memory: %d KB\n", props.totalConstMem / kb);
        printf("  | Block registers: %d\n", props.regsPerBlock);
        printf("  | Warp size: %d\n", props.warpSize );
        printf("  | Threads per block: %d\n", props.maxThreadsPerBlock );
        printf("  | Max block dimensions: %d, %d, %d\n", props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2] );
        printf("  | Max grid dimensions: %d, %d, %d\n", props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2] );
    }
}


extern "C" void gpu_print_info_short() {
    print_device_props_short();
}

extern "C" void gpu_print_info_long() {
    print_device_props_complete();
}


