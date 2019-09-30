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
//#include <cublas.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdint.h>

#include <stdio.h>

#define RP double
#define GEMM cublasDgemm
#define GEMM_BATCHED cublasDgemmBatched
#define GEMV cublasDgemv
//#define RP float
//#define FUNC cublasSgemm

// It seems that more that 16 streams doesn't help, but doesn't harm either...
#define NUMSTREAMS 300

extern "C" void cube_alloc_(int *, int * ,int * ,int *, long *, long *, int *);

extern "C" void matrix_alloc_(int *, int * ,int * ,long *, long *, int *);
extern "C" void vector_alloc_(int *, int * ,long *, int *);
extern "C" void destroy_(int *, long *);
extern "C" void cube_upload_(int *, RP *, int *,int *,int *, long *, long *);
extern "C" void cube_download_(int *, RP *, int *,int *,int *, long *, long *);
extern "C" void vector_upload_(int *, RP *, int *, long *);
extern "C" void vector_download_(int *, RP *, int *, long *);
extern "C" void my_cublas_dgemm_(int *,  int *, int *, int *, int *,
                                  RP *, 
                                  long *, long *,
                                  long *, long *,
                                  RP *,
                                  long *, long *);
extern "C" void my_cublas_dgemm_batched_( int *, int *, int *, int *,
                                  RP *, 
                                  long *, long *,
                                  long *, long *,
                                  RP *,
                                  long *, long *, 
                                  long *);
extern "C" void my_cublas_dgemv_t_(int *,  int *, 
                                  int *, int *,
                                  RP *, 
                                  long *, long *,
                                  long *, int  *,
                                  RP *,
                                  long *, int *);
extern "C" void my_cublas_dgemv_n_(int *,  int *, 
                                  int *, int *,
                                  RP *, 
                                  long *, long *,
                                  long *, int  *,
                                  RP *,
                                  long *, int *);
extern "C" void my_cublas_daxpy_(int *, int *, RP *, long *, long *);
extern "C" void my_cublas_create_(void);
extern "C" void my_cublas_destroy_(void);
extern "C" void cuda_sync_(int *);
extern "C" void cuda_sync_all_(void);
extern "C" void get_number_of_devices_(int *);



int size=sizeof(RP);
cudaError_t cudastatus;



// This contains the cublas handle and streams.
typedef struct cublasEnv_t{
    cublasHandle_t handle;
    cudaStream_t *streams;
    double **d_A, **d_B, **d_C;
} cublasEnv_t;
cublasEnv_t *cublasEnv;
int number_of_devices;

inline void check_cublas_errors(const char *filename, const int line_number) {
    /*cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      printf("CUDA error at %s:%i: %s\n", filename, line_number, cudaGetErrorString(error));
      exit(-1);
    }*/
    
}

extern "C" void cube_alloc_(int *deviceID, int *width ,int *height ,int *depth,
               long *devPtr, long *pitch, int *size) {
    // Allocate cube
    // create extent for the cube 
    
    int device = (*deviceID-1)%number_of_devices;
    cudaSetDevice(device);
    
    cudaExtent extent = make_cudaExtent(*width * sizeof(RP), *height,
            *depth);
    cudaPitchedPtr devPitchedPtr;
    
    //size_t mem_tot_0 = 0;
    //size_t mem_free_0 = 0;
    //size_t mem_tot_1 = 0;
    //size_t mem_free_1 = 0;
    
    //cudaMemGetInfo  (&mem_free_1, &mem_tot_1);
    cudastatus=cudaMalloc3D(&devPitchedPtr, extent);
    check_cublas_errors(__FILE__, __LINE__);
//    Dangerous casting here!
    *devPtr=(intptr_t)devPitchedPtr.ptr;
    *pitch=devPitchedPtr.pitch;
    *size=sizeof(RP);
    cudastatus=cudaMemset3D( devPitchedPtr, 0, extent);

    //cudaMemGetInfo  (&mem_free_0, &mem_tot_0);
    //printf("Cuda free memory before allocation: %ld, after allocation:%ld\n", mem_free_1, mem_free_0);
    return;
}



extern "C" void matrix_alloc_(int *deviceID,int *width ,int *height ,
               long *devPtr, long *pitch, int *size) {
    
    int device = (*deviceID-1)%number_of_devices;
    cudaSetDevice(device);
//    Allocate cube
    size_t pitche;
    cudastatus=cudaMallocPitch((void **)devPtr, &pitche, *width * sizeof(RP), *height);
    check_cublas_errors(__FILE__, __LINE__);
//    Dangerous casting here!
    *pitch = pitche;
    *size=sizeof(RP);
    return;
}

extern "C" void vector_alloc_(int *deviceID, int *length,
               long *devPtr, int *size) {
    
    int device = (*deviceID-1)%number_of_devices;
    cudaSetDevice(device);
//    Allocate cube
    cudastatus = cudaMalloc((void **)devPtr, *length * sizeof(RP));
    check_cublas_errors(__FILE__, __LINE__);
//    Dangerous casting here!
    *size=sizeof(RP);
    return;
}

extern "C" void destroy_(int *deviceID, long *devPtr){
    int device = (*deviceID-1)%number_of_devices;
    cudaSetDevice(device);
    //size_t mem_tot_0 = 0;
    //size_t mem_free_0 = 0;
    //size_t mem_tot_1 = 0;
    //size_t mem_free_1 = 0;
    //cudaMemGetInfo  (&mem_free_1, &mem_tot_1);
    cudastatus=cudaFree((void *) *devPtr);
    check_cublas_errors(__FILE__, __LINE__);
    
    //cudaMemGetInfo  (&mem_free_0, &mem_tot_0);
    //printf("Cuda free memory before free: %ld, after free:%ld\n", mem_free_1, mem_free_0);
    return;
}

extern "C" void cube_upload_(int *deviceID, RP *hstPtr, int *width ,int *height ,int *depth,
               long *devPtr, long *pitch) {
    int device = (*deviceID-1)%number_of_devices;
    cudaSetDevice(device);
//    Define copy "from host to device" parameters
    cudaMemcpy3DParms h2d={0};
    h2d.srcPtr = make_cudaPitchedPtr((void *)hstPtr,
        *width*sizeof(RP),*width,*height);
    h2d.dstPtr =  make_cudaPitchedPtr((void *)*devPtr,
        *pitch,*width,*height);
    h2d.extent = make_cudaExtent(*width * sizeof(RP), *height,
            *depth);
    h2d.kind   = cudaMemcpyHostToDevice;

//    Copy to device 
    cudastatus = cudaMemcpy3D( &h2d );
    check_cublas_errors(__FILE__, __LINE__);
    return;
}

extern "C" void cube_upload_slice_(int *deviceID, RP *cubeHostPointer, int *width, int *height, int *z,
               long *cubeDevicePointer, long *pitch) {
    int device = (*deviceID-1)%number_of_devices;
    cudaSetDevice(device);
    // Get the starting pointers at the host and at the device
    void *devicePointer = (void*)(*cubeDevicePointer+(*z * (*height) * (*pitch)));
    void *hostPointer = (void*)((long)cubeHostPointer+(*z * (*height) * (*width) * sizeof(RP)));
    cudastatus = cudaMemcpy2D( devicePointer, *pitch, 
                               hostPointer, *width*sizeof(RP),
                               *width * sizeof(RP), *height,
                               cudaMemcpyHostToDevice );
    check_cublas_errors(__FILE__, __LINE__);
    return;
}

extern "C" void matrix_upload_(int *deviceID, RP *hstPtr, int *width, int *height,
               long *devPtr, long *pitch) {
    int device = (*deviceID-1)%number_of_devices;
    cudaSetDevice(device);
    //    Copy to device 
    cudastatus = cudaMemcpy2D( (void*)*devPtr, *pitch, 
                               hstPtr, *width*sizeof(RP),
                               *width * sizeof(RP), *height,
                               cudaMemcpyHostToDevice );
    check_cublas_errors(__FILE__, __LINE__);
    return;
}

extern "C" void vector_upload_(int *deviceID, RP *hstPtr, int *length, long *devPtr) {
    int device = (*deviceID-1)%number_of_devices;
    cudaSetDevice(device);
    //    Copy to device 
    cudastatus = cudaMemcpy( (void*)*devPtr, hstPtr, 
                               *length* sizeof(RP), cudaMemcpyHostToDevice );
    check_cublas_errors(__FILE__, __LINE__);
    return;
}

extern "C" void cube_download_(int *deviceID, RP *hstPtr, int *width ,int *height ,int *depth,
               long *devPtr, long *pitch) {
    int device = (*deviceID-1)%number_of_devices;
    cudaSetDevice(device);
//    Define copy "from device to host" parameters
    cudaMemcpy3DParms d2h={0};
    d2h.srcPtr =  make_cudaPitchedPtr((void *)*devPtr,
        *pitch,*width,*height);
    d2h.dstPtr = make_cudaPitchedPtr((void *)hstPtr,
        *width*size,*width,*height);
    d2h.extent = make_cudaExtent(*width * size, *height,
            *depth);
    d2h.kind   = cudaMemcpyDeviceToHost;

//    cudastatus=cudaMemset3D( d2h.srcPtr, 0, d2h.extent);

//    Copy to host 
    cudastatus = cudaMemcpy3D( &d2h );
    check_cublas_errors(__FILE__, __LINE__);

    return;
}

extern "C" void matrix_download_(int *deviceID, RP *hstPtr, int *width, int *height,
               long *devPtr, long *pitch) {
    int device = (*deviceID-1)%number_of_devices;
    cudaSetDevice(device);
    //    Copy to host 
    cudastatus = cudaMemcpy2D( hstPtr, *width*sizeof(RP), 
                               (void*)*devPtr, *pitch,
                               *width * sizeof(RP), *height,
                               cudaMemcpyDeviceToHost );
    check_cublas_errors(__FILE__, __LINE__);
    
    return;
}

extern "C" void vector_download_(int *deviceID, RP *hstPtr, int *length, long *devPtr) {
    int device = (*deviceID-1)%number_of_devices;
    cudaSetDevice(device);
    //    Copy to host 
    cudastatus = cudaMemcpy( hstPtr, (void*)*devPtr, 
                             *length* sizeof(RP), cudaMemcpyDeviceToHost );
    check_cublas_errors(__FILE__, __LINE__);
    return;

    return;
}

/* Creat CuBLAS environment */
extern "C" void my_cublas_create_(void){
    int device, i;
    cudaGetDeviceCount(&number_of_devices);
    cublasEnv = new cublasEnv_t[number_of_devices];
    for (device = 0; device < number_of_devices; device++) {
        cudaSetDevice(device);
        cublasCreate(&(cublasEnv[device].handle));
        cublasEnv[device].streams=(cudaStream_t*) malloc(sizeof(cudaStream_t)*NUMSTREAMS);
        cudaMalloc((void**)&cublasEnv[device].d_A, 500*sizeof(double*));
        cudaMalloc((void**)&cublasEnv[device].d_B, 500*sizeof(double*));
        cudaMalloc((void**)&cublasEnv[device].d_C, 500*sizeof(double*));
        for (i=0;i<NUMSTREAMS;i++)
            cudaStreamCreate( &(cublasEnv[device].streams[i]) );
    }
    return;
}
extern "C" void my_cublas_destroy_(void){

    int device, i;
    for (device = 0; device < number_of_devices; device++) {
        cudaSetDevice(device);
        cublasDestroy(cublasEnv[device].handle);
        for (i=0;i<NUMSTREAMS;i++) {
            cudaStreamDestroy( cublasEnv[device].streams[i] );
        }
        
        cudaFree(cublasEnv[device].d_A);
        cudaFree(cublasEnv[device].d_B);
        cudaFree(cublasEnv[device].d_C);
        free(cublasEnv[device].streams);
    }
    return;
}


//extern "C" void cublassscal_(
//                       long *aPtr, int *N ,long *lda,RP *alpha){
//    cublasStatus_t custat;
//    custat=cublasDscal(cublasEnv.handle,*N,
//                         alpha,
//                         (double *)*aPtr,*lda);
//    return;
//}

extern "C" void my_cublas_daxpy_(int *deviceID, int *N, RP *alpha, 
                       long *aPtr, long *bPtr){
    int device = (*deviceID-1)%number_of_devices;
    cudaSetDevice(device);
    
    cublasStatus_t custat;
    cudaSetDevice(0);
    custat=cublasDaxpy(cublasEnv[device].handle,
                         *N,alpha,
                         (double *)*aPtr,1,
                         (double *)*bPtr,1);
    return;
}

static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

extern "C" void my_cublas_dgemm_( int *deviceID, int *strID, int *m, int *n, int *k,
                                  RP *alpha, 
                                  long *aPtr, long *lda,
                                  long *bPtr, long *ldb,
                                  RP *beta,
                                  long *cPtr, long *ldc){
    cublasStatus_t custat;
    int device = (*deviceID-1)%number_of_devices;
    cudaSetDevice(device);
    
    cublasSetStream(cublasEnv[device].handle,cublasEnv[device].streams[(*strID-1)%NUMSTREAMS]);

    custat=GEMM(cublasEnv[device].handle,CUBLAS_OP_N, CUBLAS_OP_N,
                         *m,*n,*k,alpha,
                         (double *)*aPtr,*lda,
                         (double *)*bPtr,*ldb,
                         beta,
                         (double *)*cPtr,*ldc);
    check_cublas_errors(__FILE__, __LINE__);
    return;
}

    
extern "C" void my_cublas_dgemm_batched_( int *deviceID, int *m, int *n, int *k,
                                  RP *alpha, 
                                  long *aPtr, long *lda,
                                  long *bPtr, long *ldb,
                                  RP *beta,
                                  long *cPtr, long *ldc, 
                                  long *batchCount) {
    cublasStatus_t custat;
    int device = (*deviceID-1)%number_of_devices;
    cudaSetDevice(device);
    
    // Copy the host array of device pointers to the device
    cudaMemcpy(cublasEnv[device].d_A, (double**)aPtr, *batchCount*sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(cublasEnv[device].d_B, (double**)bPtr, *batchCount*sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(cublasEnv[device].d_C, (double**)cPtr, *batchCount*sizeof(double*), cudaMemcpyHostToDevice);

    custat=GEMM_BATCHED(cublasEnv[device].handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         *m,*n,*k,alpha,
                         (const double **)cublasEnv[device].d_A, *lda,
                         (const double **)cublasEnv[device].d_B, *ldb,
                         beta,
                         (double **)cublasEnv[device].d_C, *ldc, 
                         *batchCount
               );
    check_cublas_errors(__FILE__, __LINE__);
    return;
}


/* 
 * Cublas dgemv: double precision general matrix vector product
 *   Does operation y = alpha * op(A) * x + beta * y
 * Parameters: 
 *   m: number of rows of matrix A
 *   n: number of columns of matrix A
 *   alpha: scalar used for multiplication
 *   aPtr: pointer to matrix A
 *   lda: leading dimension of two-dimensional array used to store matrix A
 *   xPtr: vector with n elements
 *   incx: 
 *   beta: scalar used for multiplication of y
 *   yPtr: vector with m elements
 */
extern "C" void my_cublas_dgemv_n_(int *deviceID, int *strID, 
                                  int *m, int *n,
                                  RP *alpha, 
                                  long *aPtr, long *lda,
                                  long *xPtr, int  *incx,
                                  RP *beta,
                                  long *yPtr, int *incy) {
    cublasStatus_t custat;
    int device = (*deviceID-1)%number_of_devices;
    cudaSetDevice(device);

    cublasSetStream(cublasEnv[device].handle,cublasEnv[device].streams[(*strID-1)%NUMSTREAMS]);

    custat=GEMV(cublasEnv[device].handle, CUBLAS_OP_N, 
                         *m,*n,alpha,
                         (double *)*aPtr,*lda,
                         (double *)*xPtr,*incx,
                         beta,
                         (double *)*yPtr,*incy);
    return;
}

/* 
 * Cublas dgemv: double precision general matrix vector product
 *   Does operation y = alpha * op(A) * x + beta * y
 * Parameters: 
 *   m: number of rows of matrix A
 *   n: number of columns of matrix A
 *   alpha: scalar used for multiplication
 *   aPtr: pointer to matrix A
 *   lda: leading dimension of two-dimensional array used to store matrix A
 *   xPtr: vector with n elements
 *   incx: 
 *   beta: scalar used for multiplication of y
 *   yPtr: vector with m elements
 */
extern "C" void my_cublas_dgemv_t_(int *deviceID, int *strID, 
                                  int *m, int *n,
                                  RP *alpha, 
                                  long *aPtr, long *lda,
                                  long *xPtr, int  *incx,
                                  RP *beta,
                                  long *yPtr, int *incy) {
    cublasStatus_t custat;
    int device = (*deviceID-1)%number_of_devices;
    cudaSetDevice(device);

    cublasSetStream(cublasEnv[device].handle,cublasEnv[device].streams[(*strID-1)%NUMSTREAMS]);

    custat=GEMV(cublasEnv[device].handle, CUBLAS_OP_T, 
                         *m,*n,alpha,
                         (double *)*aPtr,*lda,
                         (double *)*xPtr,*incx,
                         beta,
                         (double *)*yPtr,*incy);
    return;
}


extern "C" void my_cublas_ddot_(  int *deviceID, int *strID, int *n, 
                                  long *xPtr, int *incx,
                                  long *yPtr, int *incy,
                                  double *result
                               ) {
    int device = (*deviceID-1)%number_of_devices;
    cudaSetDevice(device);
    cublasStatus_t custat;

    cublasSetStream(cublasEnv[device].handle,cublasEnv[device].streams[(*strID-1)%NUMSTREAMS]);

    custat=cublasDdot(cublasEnv[device].handle, *n, 
                         (double *)*xPtr,*incx,
                         (double *)*yPtr,*incy,
                         result);
    return;
}

/*extern "C" void* allocateDeviceMemory(int size) {
    void * memory;
}*/

extern "C" void cuda_sync_(int *deviceID) {
    int device = (*deviceID-1)%number_of_devices;
    cudaSetDevice(device);
    cudaDeviceSynchronize( );
    return;
}
extern "C" void cuda_sync_all_() {
    int nof_devices; 
    get_number_of_devices_(&nof_devices);
    for (int device = 0; device < nof_devices; device++) {
        cudaSetDevice(device);
        cudaDeviceSynchronize( );
    }
    return;
}

extern "C" void cuda_enable_peer_comm_() {
    int nof_devices; 
    get_number_of_devices_(&nof_devices);
    for (int device = 0; device < nof_devices; device++) {
        cudaSetDevice(device);
        for (int device2 = 0; device2 < nof_devices; device2++) {
            if (device != device2) {
                //int can_access_peer;
                //cudaDeviceCanAccessPeer(&can_access_peer, device, device2);
                //if (can_access_peer == 1) {
                    //cudaDeviceEnablePeerAccess(device2, 0);
                    //cudaDeviceSynchronize();
                    //cudaError_t result = cudaGetLastError();
                    //if (result == cudaErrorPeerAccessAlreadyEnabled) {
                    //    printf("Peer access was already enabled between devices: %d and %d", device, device2);
                    //}
                //}
                //else {
                //    printf("ERROR: Device %d cannot access device %d.", device, device2);
                //}
            }
        }
    }
    return;
}





extern "C" void get_number_of_devices_(int *device_count) {
    cudaGetDeviceCount(device_count);
    return;
}
