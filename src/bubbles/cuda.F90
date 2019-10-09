!----------------------------------------------------------------------------------!
!    Copyright (c) 2010-2018 Pauli Parkkinen, Eelis Solala, Wen-Hua Xu,            !
!                            Sergio Losilla, Elias Toivanen, Jonas Juselius        !
!                                                                                  !
!    Permission is hereby granted, free of charge, to any person obtaining a copy  !
!    of this software and associated documentation files (the "Software"), to deal !
!    in the Software without restriction, including without limitation the rights  !
!    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell     !
!    copies of the Software, and to permit persons to whom the Software is         !
!    furnished to do so, subject to the following conditions:                      !
!                                                                                  !
!    The above copyright notice and this permission notice shall be included in all!
!    copies or substantial portions of the Software.                               !
!                                                                                  !
!    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    !
!    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      !
!    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE   !
!    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        !
!    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, !
!    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE !
!    SOFTWARE.                                                                     !
!----------------------------------------------------------------------------------!
!> @file cuda.F90
!! The CUDA backend

!> CUDA backend
!!
module cuda_m
    use globals_m
    use ISO_C_BINDING
    implicit none
    public CUDACube, CUDAMatrix, CUDAVector, CUDABlas
    public CUDAsync, CUDAsync_all
    public CUBLAS_dgemv_n, CUBLAS_dgemv_t, CUBLAS_cube_sum
    public cudacube_init_cuda, cudacube_destroy_cuda, cudacube_download_cuda
    public register_host_cube_cuda, unregister_host_cube_cuda, CUDACube_init_page_locked_cube, &
           CUDACube_destroy_page_locked_cube, CUDACube_init_page_locked_1D_cube, &
           CUDACube_destroy_page_locked_1D_cube
    public CUDA_get_number_of_devices, CUDA_enable_peer_communication

    private

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                   CudaCube definition                                   %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    type CUDACube
        ! str(:) contains element distances between the same element in both
        ! adjacent slices
        ! str(X_) is the distance between cube(i,j,k) and cube(i+1,j,k)
        ! str(Y_) is the distance between cube(i,j,k) and cube(i,j+1,k)
        ! str(Z_) is the distance between cube(i,j,k) and cube(i,j,k+1)
        ! str(X_) should not be used for slicing, because elements are never
        ! adjacent
        type(C_PTR), allocatable    :: cuda_interface
        logical                     :: is_subcube
        integer(INT32)              :: dims(3)
        integer(INT32), allocatable :: sz(:)
        integer(INT32)              :: number_of_devices
        integer(INT64), allocatable :: devPtr(:),pitch(:),str(:, :)
        real(REAL64), pointer :: hstPtr(:,:,:)=>NULL()
        integer(INT32) :: parent_dims(3)
    contains
        procedure, public   :: upload        => CUDACube_upload
        procedure, public   :: download      => CUDACube_download
        procedure, public   :: destroy       => CUDACube_destroy
        procedure, public   :: slice         => CUDACube_slice
        procedure, public   :: get_subcube   => CUDAcube_get_subcube
        procedure, public   :: set_to_zero   => CUDAcube_set_to_zero
        procedure, public   :: init_host     => CudaCube_init_host_cube
        procedure, private  :: CudaCube_set_host_cube, CudaCube_set_host_1d
        generic,   public   :: set_host      => CudaCube_set_host_cube, CudaCube_set_host_1d
        procedure, public   :: get_host_cube => CudaCube_get_host_cube
        procedure, public   :: unset_host    => CudaCube_unset_host_cube
    end type
    
    interface CUDACube
        module procedure :: CUDACube_init
        module procedure :: CUDACube_init_remote
    end interface

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                   CudaCube Cuda-interfaces                              %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#ifdef HAVE_CUDA

    interface
        type(C_PTR) function CudaCube_init_and_set_cuda(streamContainer, shape, &
                                 host_cube, host_cube_offset, host_cube_shape, &
                                 all_memory_at_all_devices, &
                                 register_host, sliced) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: streamContainer
            integer(C_INT)        :: shape(3)
            real(C_DOUBLE)        :: host_cube(*)
            integer(C_INT), value :: host_cube_offset
            integer(C_INT)        :: host_cube_shape(3)
            integer(C_INT), value :: all_memory_at_all_devices
            integer(C_INT), value :: register_host
            integer(C_INT), value :: sliced
        end function
    end interface

    interface
        type(C_PTR) function CudaCube_init_page_locked_cube_cuda(shape) bind(C)
            use ISO_C_BINDING
            integer(C_INT)        :: shape(3)
        end function
    end interface

    interface
        subroutine CudaCube_destroy_page_locked_cube_cuda(cube) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: cube
        end subroutine
    end interface

    interface
        type(C_PTR) function CudaCube_init_cuda(streamContainer, shape, &
                              all_memory_at_all_devices, sliced) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: streamContainer
            integer(C_INT)        :: shape(3)
            integer(C_INT), value :: all_memory_at_all_devices
            integer(C_INT), value :: sliced
        end function
    end interface

    interface
        type(C_PTR) function CudaCube_get_slice_cuda(cuda_cube, &
                              slice_index, slice_dimension) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: cuda_cube
            integer(C_INT), value :: slice_index
            integer(C_INT), value :: slice_dimension
        end function
    end interface

    interface
        type(C_PTR) function CudaCube_get_stream_container_cuda(cuda_cube) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: cuda_cube
        end function
    end interface

     interface
        type(C_PTR) function CudaCube_get_subcube_cuda(cuda_cube, &
                              start_indices, end_indices, streamContainer) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: cuda_cube
            integer(C_INT)        :: start_indices(3)
            integer(C_INT)        :: end_indices(3)
            type(C_PTR),    value :: streamContainer
        end function
    end interface


    interface
        subroutine CudaCube_destroy_cuda(cudacube) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: cudacube
        end subroutine
    end interface

    interface
        subroutine CudaCube_download_cuda(cudacube) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: cudacube
        end subroutine
    end interface

    interface
        subroutine CudaCube_upload_cuda(cudacube) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: cudacube
        end subroutine
    end interface

    interface
        subroutine CudaCube_set_to_zero_cuda(cudacube) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: cudacube
        end subroutine
    end interface

    interface
        subroutine CudaCube_set_host_cube_cuda(cudacube, host_cube, host_cube_shape, register_host)  bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: cudacube
            real(C_DOUBLE)        :: host_cube(*)
            integer(C_INT)        :: host_cube_shape(3)
            integer(C_INT), value :: register_host
        end subroutine
    end interface

    interface
         type(C_PTR) function CudaCube_get_host_cube_cuda(cudacube)  bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: cudacube
        end function
    end interface

    interface
        subroutine CudaCube_unset_host_cube_cuda(cudacube)  bind(C)
            use ISO_C_BINDING
            type(C_PTR), value :: cudacube

        end subroutine
    end interface


    interface
        subroutine register_host_cube_cuda(host_cube, host_cube_shape)  bind(C)
            use ISO_C_BINDING
            real(C_DOUBLE)    :: host_cube(*)
            integer(C_INT)    :: host_cube_shape(3)
        end subroutine
    end interface

    interface
        subroutine unregister_host_cube_cuda(host_cube)  bind(C)
            use ISO_C_BINDING
            real(C_DOUBLE)    :: host_cube(*)

        end subroutine
    end interface

    
#endif

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                   CudaMatrix definition                                 %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    type CUDAMatrix
        integer(INT32) :: dims(2)
        integer(INT32), allocatable :: sz(:)
        integer(INT32) :: number_of_devices
        logical        :: is_slice
        integer(INT64), allocatable :: devPtr(:),pitch(:),str(:, :)
        real(REAL64), pointer :: hstPtr(:,:)=>NULL()
        type(C_PTR), allocatable :: cuda_interface
    contains
        procedure, public   :: upload => CUDAmatrix_upload
        procedure, public   :: download => CUDAmatrix_download
        procedure, public   :: destroy => CUDAmatrix_destroy
        procedure, public   :: slice => CUDAmatrix_slice
    end type
    
    interface CUDAMatrix
        module procedure :: CUDAmatrix_init
        module procedure :: CUDAmatrix_init_remote
    end interface

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                   CudaVector definition                                 %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    type CUDAVector
        logical        :: is_slice
        integer(INT32) :: dims
        integer(INT32), allocatable :: sz(:)
        integer(INT32) :: number_of_devices
        integer(INT64), allocatable :: devPtr(:), pitch(:), str(:)
        real(REAL64), pointer :: hstPtr(:)=>NULL()
    contains
        procedure, public   :: upload => CUDAvector_upload
        procedure, public   :: download => CUDAvector_download
        procedure, public   :: destroy => CUDAvector_destroy

    end type
    
    interface CUDAVector
        module procedure :: CUDAvector_init
        module procedure :: CUDAvector_init_remote
    end interface


    !interface operator(.dot.)
    !    module procedure :: CUDAvector_dot
    !end interface 
    

    integer :: cublas_inits=0

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                   CudaBlas definition                                   %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    type CUDABlas
        type(C_PTR), allocatable :: cuda_interface
    contains
        procedure, private  :: CUDABlas_dgemm_batched_cube_matrix
        procedure, private  :: CUDABlas_dgemm_batched_matrix_cube
        procedure, public   :: get_number_of_devices     => CUDABlas_get_number_of_devices
        procedure, public   :: get_streams_per_device     => CUDABlas_get_streams_per_device
        generic,   public   :: mm_multiplication_batched => CUDABlas_dgemm_batched_cube_matrix, &
                                                            CUDABlas_dgemm_batched_matrix_cube
        procedure, public   :: mm_multiplication         => CUDABlas_dgemm
        procedure, public   :: destroy                   => CUDABlas_destroy
    end type

    interface CUDABlas
        module procedure :: CUDABlas_init
    end interface
     

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                   CudaBLAS Cuda-interfaces                              %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#ifdef HAVE_CUDA

    interface
        type(C_PTR) function CUDABlas_init_cuda(streamContainer, pointers_per_stream) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: streamContainer
            integer(C_INT), value :: pointers_per_stream
        end function
    end interface

    interface
        integer(C_INT) function CUDABlas_get_number_of_devices_cuda(cudablas)  bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value   :: cudablas

        end function
    end interface

    interface
        integer(C_INT) function CUDABlas_get_streams_per_device_cuda(cudablas)  bind(C)
            use ISO_C_BINDING
            type(C_PTR), value      :: cudablas

        end function
    end interface

    interface
        subroutine CUDABlas_destroy_cuda(cudablas) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: cudablas
        end subroutine
    end interface
    
    interface
        subroutine CUDABlas_mm_multiplication_batched_cuda( &
                       cudablas, device, stream, cube1, cube2, result_cube, &
                       slice_dimension, alpha, beta, waited_event) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: cudablas
            integer(C_INT), value :: device
            integer(C_INT), value :: stream
            type(C_PTR),    value :: cube1
            type(C_PTR),    value :: cube2
            type(C_PTR),    value :: result_cube
            integer(C_INT), value :: slice_dimension
            real(C_DOUBLE), value :: alpha
            real(C_DOUBLE), value :: beta
            type(C_PTR),    value :: waited_event
        end subroutine
    end interface

    interface
        subroutine CUDABlas_mm_multiplication_cuda( &
                       cudablas, device, stream, slice1, slice2, result_slice, &
                       alpha, beta) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: cudablas
            integer(C_INT), value :: device
            integer(C_INT), value :: stream
            type(C_PTR),    value :: slice1
            type(C_PTR),    value :: slice2
            type(C_PTR),    value :: result_slice
            real(C_DOUBLE), value :: alpha
            real(C_DOUBLE), value :: beta
        end subroutine
    end interface
    
#endif

contains

    function CUDACube_init(cube, all_memory_at_all_devices, container) result(new)
        real(REAL64), intent(in)             :: cube(:,:,:)
        logical,      intent(in), optional   :: all_memory_at_all_devices
        type(C_PTR),  intent(in), optional   :: container
        type(CUDACube)                       :: new
        !integer                    :: device
        !integer(INT64)             :: device_pointer
        integer                              :: all_memory_at_all_devices_int, sliced_int

        new%dims = shape(cube)

        ! convert all_memory_at_all_devices to integer
        if (present(all_memory_at_all_devices)) then
            if (all_memory_at_all_devices) then
                all_memory_at_all_devices_int = 1
            else
                all_memory_at_all_devices_int = 0
            end if
        else
            all_memory_at_all_devices_int = 1
        end if

        ! always do not register host and use sliced memory uploads and download
        if (present(container)) then
            ! always do not register host use sliced memory uploads and downloads
            new%cuda_interface = CudaCube_init_and_set_cuda( &
                                    container, shape(cube), cube, 0, &
                                    shape(cube), all_memory_at_all_devices_int, 0, 1)

        else
            ! use global container
            new%cuda_interface = CudaCube_init_and_set_cuda( &
                                    stream_container, shape(cube), cube, 0, &
                                    shape(cube), all_memory_at_all_devices_int, 0, 1)
        end if

       
    end function

    function CUDAcube_init_remote(dims, all_memory_at_all_devices, container) result(new)
        integer(INT32), intent(in)           :: dims(3)
        logical,        intent(in), optional :: all_memory_at_all_devices
        type(C_PTR),    intent(in), optional :: container
        type(CUDACube)                       :: new
        integer                              :: all_memory_at_all_devices_int, sliced_int

        new%dims = dims
        ! convert all_memory_at_all_devices to integer
        if (present(all_memory_at_all_devices)) then
            if (all_memory_at_all_devices) then
                all_memory_at_all_devices_int = 1
            else
                all_memory_at_all_devices_int = 0
            end if
        else
            all_memory_at_all_devices_int = 1
        end if

        ! always register host and use sliced memory uploads and download
        if (present(container)) then
            ! always register host and use sliced memory uploads and downloads
            new%cuda_interface = cudacube_init_cuda(container, dims, &
                                                    all_memory_at_all_devices_int, 1)

        else
            ! call the corresponding c++ method
            new%cuda_interface = cudacube_init_cuda(stream_container, dims, &
                                                    all_memory_at_all_devices_int, 1)
        end if
    end function

    function CUDAmatrix_init(matrix) result(new)
        real(REAL64), target :: matrix(:, :)
        type(CUDAMatrix)   :: new
        integer              :: device

        new%hstPtr => matrix
        new%dims=shape(new%hstPtr)
        new%is_slice = .FALSE.
        call get_number_of_devices(new%number_of_devices)
        allocate(new%str(2, new%number_of_devices))
        allocate(new%sz(new%number_of_devices))
        allocate(new%pitch(new%number_of_devices))
        do device = 1, new%number_of_devices
            call matrix_alloc(device, new%dims(X_), new%dims(Y_),&
                            new%devPtr(device), new%pitch(device), new%sz(device))
            new%str(X_, device)=1
            new%str(Y_, device)=new%pitch(device)/new%sz(device)
        end do

        
    end function

    function CUDAmatrix_init_remote(dims) result(new) 
        ! Initialize an in-device array, without reference to anything local
        integer(INT32),intent(in) :: dims(2)
        type(CUDAMatrix)        :: new
        integer                   :: device

        new%dims=dims
        new%hstPtr=>NULL()
        new%is_slice = .FALSE.
        call get_number_of_devices(new%number_of_devices)
        allocate(new%str(2, new%number_of_devices))
        allocate(new%sz(new%number_of_devices))
        allocate(new%pitch(new%number_of_devices))
        do device = 1, new%number_of_devices
            call matrix_alloc(device, new%dims(X_), new%dims(Y_),&
                            new%devPtr(device), new%pitch(device), new%sz(device))
            new%str(X_, device)=1
            new%str(Y_, device)=new%pitch(device)/new%sz(device)
        end do
    end function

    function CUDAvector_init(vector) result(new)
        real(REAL64),target :: vector(:)
        type(CUDAVector)  :: new
        integer             :: device

        new%hstPtr => vector
        new%dims=size(new%hstPtr)
        new%is_slice = .FALSE.
        call get_number_of_devices(new%number_of_devices)
        allocate(new%str(new%number_of_devices))
        allocate(new%sz(new%number_of_devices))
        allocate(new%pitch(new%number_of_devices))
        
        do device = 1, new%number_of_devices
            call vector_alloc(new%dims,&
                            new%devPtr(device), new%sz(device))
            new%str(device)=1
        end do
        
    end function

    
    

    function CUDAvector_init_remote(dims) result(new)
        integer(INT32), intent(in) :: dims
        type(CUDAVector)         :: new
        integer                    :: device

        new%hstPtr=>NULL()
        new%dims=dims
        new%is_slice = .FALSE.
        
        call get_number_of_devices(new%number_of_devices)
        allocate(new%str(new%number_of_devices))
        allocate(new%sz(new%number_of_devices))
        allocate(new%pitch(new%number_of_devices))
        
        do device = 1, new%number_of_devices
            call vector_alloc(new%dims,&
                            new%devPtr(device), new%sz(device))
            new%str(device)=1
        end do
    end function

    function CudaCube_get_host_cube(self) result(cube)
        class(CUDACube), intent(in)    :: self
        real(REAL64),    pointer       :: cube(:, :, :)
        type(C_PTR)                    :: c_pointer
        c_pointer = CudaCube_get_host_cube_cuda(self%cuda_interface)
        call c_f_pointer(c_pointer, cube, self%dims)
    end function

    subroutine CudaCube_set_host_cube(self, cube)
        class(CUDACube), intent(inout) :: self
        real(REAL64),    intent(in)    :: cube(:, :, :)

        call CudaCube_set_host_cube_cuda(self%cuda_interface, cube, shape(cube), 0)
    end subroutine

    subroutine CudaCube_set_host_1d(self, cube, host_cube_shape)
        class(CUDACube), intent(inout) :: self
        real(REAL64),    intent(in)    :: cube(:)
        integer,         intent(in)    :: host_cube_shape(3)

        call CudaCube_set_host_cube_cuda(self%cuda_interface, cube, host_cube_shape, 0)
    end subroutine

    subroutine CudaCube_init_host_cube(self)
        class(CUDACube), intent(inout) :: self
        real(REAL64),    pointer       :: cube(:, :, :)

        cube => CUDACube_init_page_locked_cube(self%dims)

        call self%set_host(cube)
    end subroutine

    subroutine CudaCube_unset_host_cube(self)
        class(CUDACube), intent(inout) :: self

        call CudaCube_unset_host_cube_cuda(self%cuda_interface)
    end subroutine

    subroutine CUDAcube_destroy(self)
        class(CUDACube) :: self
        integer           :: device
        if (allocated(self%cuda_interface)) then
            call CUDACube_destroy_cuda(self%cuda_interface)
            deallocate(self%cuda_interface)
        end if
        if (allocated(self%devPtr)) then  
            if (.not. self%is_subcube) then
                do device = 1, self%number_of_devices
                    call destroy(device, self%devPtr(device))
                end do
            end if
            deallocate(self%devPtr)
        end if
        if (allocated(self%str)) deallocate(self%str)
        if (allocated(self%sz)) deallocate(self%sz)
        if (allocated(self%pitch)) deallocate(self%pitch)
        if (associated(self%hstPtr)) nullify(self%hstPtr)
    end subroutine

    subroutine CUDAvector_destroy(self)
        class(CUDAVector) :: self
        integer           :: device
        if (allocated(self%devPtr)) then
            if (.not. self%is_slice) then
                do device = 1, self%number_of_devices
                    call destroy(device, self%devPtr(device))
                end do
            end if
            deallocate(self%devPtr)
        end if
        if (allocated(self%str)) deallocate(self%str)
        if (allocated(self%sz)) deallocate(self%sz)
        if (allocated(self%pitch)) deallocate(self%pitch)
        if (associated(self%hstPtr)) nullify(self%hstPtr)
    end subroutine

    subroutine CUDAcube_set_to_zero(self)
        class(CUDACube)   :: self

        call CudaCube_set_to_zero_cuda(self%cuda_interface)
        !integer             :: device
        !integer(INT64)      :: xsize, ysize
        !if (self%is_subcube) then
        !    do device = 1, self%number_of_devices
        !        xsize = self%parent_dims(X_) * self%sz(device)
        !        ysize = self%parent_dims(Y_)
        !        call cube_set_to_zero(device, self%dims(X_), self%dims(Y_), self%dims(Z_), &
        !                        self%devPtr(device), self%pitch(device),          &
        !                        xsize, ysize, self%sz(device), self%number_of_devices)
        !    end do
        !else
        !    do device = 1, self%number_of_devices
        !        xsize = self%dims(X_) * self%sz(device)
        !        ysize = self%dims(Y_)
        !        call cube_set_to_zero(device, self%dims(X_), self%dims(Y_), self%dims(Z_), &
        !                        self%devPtr(device),  self%pitch(device),          &
        !                        xsize, ysize, self%sz(device), self%number_of_devices)
        !    end do
        !end if
    end subroutine

    subroutine CUDAcube_upload(self)
        class(CUDACube)     :: self
        integer             :: device
        call CudaCube_upload_cuda(self%cuda_interface)
        !do device = 1, self%number_of_devices
        !    call cube_upload(device, self%hstPtr,self%dims(X_),self%dims(Y_),self%dims(Z_),&
        !                 self%devPtr(device),self%pitch(device))
        !end do

    end subroutine
 
    subroutine CUDAcube_download(self)
        class(CUDACube) :: self
        !real(REAL64)      :: cubes(self%dims(X_), self%dims(Y_), &
        !                           self%dims(Z_), self%number_of_devices)
        !integer           :: device

        call CudaCube_download_cuda(self%cuda_interface)
        !if (self%number_of_devices == 1) then
        !    call cube_download(1, self%hstPtr, &
        !                 self%dims(X_),self%dims(Y_),self%dims(Z_),&
        !                 self%devPtr(1),self%pitch(1))
        !else
        !    cubes = 0.0d0
        !    self%hstPtr = 0.0d0
        !    do device = 1, self%number_of_devices
        !        call cube_download(device, cubes(:, :, :, device), &
        !                 self%dims(X_),self%dims(Y_),self%dims(Z_),&
        !                 self%devPtr(device),self%pitch(device))
        !        self%hstPtr = self%hstPtr + cubes(:, :, :, device)
        !    end do
        !end if
    end subroutine

    subroutine CUDAmatrix_upload(self)
        class(CUDAMatrix) :: self
        !integer             :: device
        
        call CudaCube_upload_cuda(self%cuda_interface)
        !do device = 1, self%number_of_devices
        !    call matrix_upload(device, self%hstPtr,self%dims(X_),self%dims(Y_),&
        !                    self%devPtr(device),self%pitch(device))
        !end do
    end subroutine

    subroutine CUDAmatrix_download(self)
        class(CUDAMatrix) :: self

        !real(REAL64)        :: matrices(self%dims(X_), self%dims(Y_), self%number_of_devices)
        !integer             :: device

        call CudaCube_download_cuda(self%cuda_interface)
        !if (self%number_of_devices == 1) then
        !    call matrix_download(1, self%hstPtr, self%dims(X_),self%dims(Y_), &
        !                 self%devPtr(1),self%pitch(1))
        !else
        !    matrices = 0.0d0
        !    do device = 1, self%number_of_devices
        !        call matrix_download(device, matrices(:, :, device), self%dims(X_),self%dims(Y_), &
        !                 self%devPtr(device), self%pitch(device))
        !        self%hstPtr = self%hstPtr + matrices(:, :, device)
        !    end do
        !end if
        
    end subroutine

    subroutine CUDAvector_upload(self)
        class(CUDAVector) :: self
        integer             :: device

        do device = 1, self%number_of_devices
            call vector_upload(device, self%hstPtr,self%dims,&
                            self%devPtr(device))
        end do
    end subroutine

    subroutine CUDAvector_download(self)
        class(CUDAVector) :: self
        real(REAL64)        :: vectors(self%dims, self%number_of_devices)
        integer             :: device

        if (self%number_of_devices == 1) then
            call vector_download(1, self%hstPtr,self%dims, &
                            self%devPtr(1))
        else
            do device = 1, self%number_of_devices
                call vector_download(device, vectors(:, device), self%dims, self%devPtr(device))
                self%hstPtr = self%hstPtr + vectors(:, device)
            end do
        end if
            
    end subroutine


    subroutine CUDAmatrix_destroy(self)
        class(CUDAMatrix), intent(inout) :: self
        integer                          :: device

        if (allocated(self%cuda_interface)) then
            call CUDACube_destroy_cuda(self%cuda_interface)
            deallocate(self%cuda_interface)
        end if
        if (allocated(self%devPtr)) then
            if (.not. self%is_slice) then
                do device=1, self%number_of_devices
                    call destroy(device, self%devPtr(device))
                end do
            end if
            deallocate(self%devPtr)
        end if
        if (allocated(self%str)) deallocate(self%str)
        if (allocated(self%sz)) deallocate(self%sz)
        if (allocated(self%pitch)) deallocate(self%pitch)
        if (allocated(self%cuda_interface)) deallocate(self%cuda_interface)
        if (associated(self%hstPtr)) nullify(self%hstPtr)
    end subroutine

    function CUDACube_slice(self, dir, idx) result(slice)
        ! Pick a slice (matrix) from a cube allocated in the GPGPU, not linked
        ! to any local array
        ! Examples:
        !    CUDAcube_slice(a,Y_,3) refers to a%hstPtr(:,3,:)
        ! dir is the slicing direction -> 2 for y, 3 for z (DON'T USE X)
        ! idx is the slicing index
        class(CUDACube), intent(in)   :: self
        !> Direction of slicing and the order number of slice
        integer(INT32),    intent(in)   :: dir, idx
        !> the result CudaMatrix object
        type(CUDAMatrix)              :: slice
        !integer                        :: device

        slice%cuda_interface = cudacube_get_slice_cuda(self%cuda_interface, idx, dir)
    end function

    function CUDAcube_get_subcube(self, cube_indices, container) result(subcube)
        ! Pick a subcube of cuda cube
        class(CUDACube),   intent(in)            :: self
        integer(INT32),    intent(in)            :: cube_indices(2, 3)
        type(C_PTR),       intent(in), optional  :: container
        type(CUDACube)                           :: subcube
        integer(INT32)                           :: start_indices(3), end_indices(3)

        start_indices = cube_indices(1, :)
        end_indices = cube_indices(2, :)

        if (present(container)) then
            subcube%cuda_interface = cudacube_get_subcube_cuda(self%cuda_interface, start_indices, end_indices, container)
        else
            ! use the current stream container
            subcube%cuda_interface = cudacube_get_subcube_cuda(self%cuda_interface, start_indices, end_indices, &
                     CudaCube_get_stream_container_cuda(self%cuda_interface))
        end if
    end function

    function CUDAmatrix_slice(self,dir,idx) result(slice)
        ! Pick a slice (vector) from a matrix allocated in the GPGPU, not linked
        ! to any local array
        ! Examples:
        !    CUDAmatrix_slice(a,Y_,1) refers to a%hstPtr(:,3)
        ! dir is the slicing direction -> 2 for y_ (DON'T USE X)
        ! idx is the slicing index
        class(CUDAMatrix),intent(in) :: self
        integer(INT32),intent(in)   :: dir, idx
        type(CUDAVector)          :: slice
        
        integer                     :: device
        
        slice%is_slice = .TRUE.
        slice%number_of_devices = self%number_of_devices
        allocate(slice%devPtr(self%number_of_devices))
        allocate(slice%str(self%number_of_devices))
        allocate(slice%sz(self%number_of_devices))
        slice%dims=self%dims(X_)
        slice%hstPtr=>NULL()
        do device = 1, self%number_of_devices
            slice%devPtr(device)=self%devPtr(device)+(idx-1)*self%str(dir, device)*self%sz(device)
            slice%str(device)=1_INT64
        end do
    end function

    function CUDABlas_init(pointers_per_stream, container) result(new)
        type(CUDABlas)                     :: new
        integer, intent(in)                :: pointers_per_stream
        type(C_PTR), intent(in), optional  :: container

        if (present(container)) then
            new%cuda_interface = CudaBlas_init_cuda(container, pointers_per_stream)
        else
            ! if the input container is not given, we are using the global one
            new%cuda_interface = CudaBlas_init_cuda(stream_container, pointers_per_stream)
        end if
    end function

    function CUDABlas_get_number_of_devices(self) result(number_of_devices)
        class(CUDABlas)                     :: self
        integer                            :: number_of_devices

        number_of_devices = CudaBlas_get_number_of_devices_cuda(self%cuda_interface)
    end function

    function CUDABlas_get_streams_per_device(self) result(streams_per_device)
        class(CUDABlas)                     :: self
        integer                            :: streams_per_device

        streams_per_device = CudaBlas_get_streams_per_device_cuda(self%cuda_interface)
    end function

    subroutine CUDABlas_destroy(self)
        class(CUDABlas), intent(inout) :: self
        
        if (allocated(self%cuda_interface)) then
            call CUDABlas_destroy_cuda(self%cuda_interface) 
        end if
    end subroutine

    

    function CUDAvector_dot(self, vector2, threadId, parentThreadId) result(reslt)
        type(CUDAVector), intent(in) :: self, vector2
        real(REAL64)                   :: reslt
        integer(INT32) :: parentThreadId
        integer(INT32) :: threadID
        integer(INT32) :: device

        device = mod(parentThreadId-1, self%number_of_devices)+1
        call my_cublas_ddot(parentThreadId, threadID, self%dims,      &
                              self%devPtr(device), 1,    &
                              vector2%devPtr(device), 1, &
                              reslt)
 
    end function

    subroutine CUDABlas_dgemm(self, parentThreadId, threadID, a, b, c, alpha, beta)
        class(CUDABlas), intent(in) :: self
        ! Perform matrix multiplication (gemm) with matrices on the GPGPU
        integer(INT32)              :: parentThreadId, threadID, device
        type(CUDAMatrix)            :: a, b, c
        real(REAL64)                :: alpha, beta
        !  Pointers to  Matrix   Matrix    Strides
        !   starts      heights  widths  (ld* in gemm)

        call cudablas_mm_multiplication_cuda( &
                       self%cuda_interface, parentThreadId, threadId, &
                       a%cuda_interface, b%cuda_interface, c%cuda_interface, &
                       alpha, beta)
        !device = mod(parentThreadId-1, a%number_of_devices)+1
        !call my_cublas_dgemm(parentThreadId, threadID,a%dims(1), b%dims(2), a%dims(2),&
        !                      alpha, &
        !                      a%devPtr(device),a%str(2, device),&
        !                      b%devPtr(device),b%str(2, device),&
        !                      beta,&
        !                      c%devPtr(device),c%str(2, device))
            
    end subroutine

    subroutine CUDABlas_dgemm_batched_cube_matrix(self, device_id, stream_id, a, b, c, alpha, &
                                                  beta, slice_dimension, waited_event)
        class(CUDABlas), intent(in)      :: self
        ! Perform matrix multiplication (gemm) with matrices on the GPGPU
        integer(INT32), intent(in)       :: device_id, stream_id, slice_dimension
        type(CUDACube),  intent(in)      :: a
        type(CUDAMatrix), intent(in)     :: b
        type(CUDACube),  intent(inout)   :: c
        real(REAL64),      intent(in)    :: alpha, beta
        type(C_PTR),     intent(in)      :: waited_event

        !type(CUDAMatrix)                :: a_slice, c_slice
        !integer(INT32)                    :: b_row_count, a_row_count, b_column_count, i, device
        !integer(INT64)                    :: a_pointers(a%dims(slice_dimension)), &
        !                                     b_pointers(a%dims(slice_dimension)), &
        !                                     c_pointers(a%dims(slice_dimension))
        ! leading dimensions of arrays a, b and c 
        !integer(INT64)                    :: lda, ldb, ldc, batch_count

        call cudablas_mm_multiplication_batched_cuda(self%cuda_interface, device_id, stream_id, &
                                                     a%cuda_interface, b%cuda_interface, c%cuda_interface, &
                                                     slice_dimension, alpha, beta, waited_event)
    end subroutine
    
    subroutine CUDABlas_dgemm_batched_matrix_cube(self, device_id, stream_id, a, b, c, alpha, &
                                                  beta, slice_dimension, waited_event)
        class(CUDABlas), intent(in)      :: self
        ! Perform matrix multiplication (gemm) with matrices on the GPGPU
        integer(INT32), intent(in)       :: device_id, stream_id, slice_dimension
        type(CUDAMatrix), intent(in)     :: a
        type(CUDACube),  intent(in)      :: b
        type(CUDACube),  intent(inout)   :: c
        real(REAL64),      intent(in)    :: alpha, beta
        type(C_PTR),     intent(in)      :: waited_event

        !type(CUDAMatrix)                :: b_slice, c_slice
        !integer                           :: b_row_count, a_row_count, b_column_count, i, device
        !integer(INT64)                    :: a_pointers(b%dims(slice_dimension)), &
        !                                     b_pointers(b%dims(slice_dimension)), &
        !                                     c_pointers(b%dims(slice_dimension))
        ! leading dimensions of arrays a, b and c 
        !integer(INT64)                    :: lda, ldb, ldc, batch_count
        call cudablas_mm_multiplication_batched_cuda(self%cuda_interface, device_id, stream_id, &
                                                     a%cuda_interface, b%cuda_interface, c%cuda_interface, &
                                                     slice_dimension, alpha, beta, waited_event)
    end subroutine


    subroutine CUBLAS_dgemv_n(parentThreadId, threadID, a, x, y, alpha, beta, incx, incy)
        ! Perform matrix-vector multiplication (gemv) with matrices on the GPGPU
        integer(INT32),     intent(in)   :: parentThreadID
        integer(INT32),     intent(in)   :: threadID
        type(CUDAMatrix), intent(in)   :: a
        type(CUDAVector), intent(in)   :: x
        type(CUDAVector)               :: y
        real(REAL64)                     :: alpha, beta
        integer(INT32)                   :: incx, incy, device
        
        device = mod(parentThreadId-1, a%number_of_devices)+1
        call my_cublas_dgemv_n(threadID,a%dims(1), a%dims(2),&
                              alpha, &
                              a%devPtr(device),a%str(2, device),&
                              x%devPtr(device),incx,&
                              beta,&
                              y%devPtr(device),incy)
            
    end subroutine

    subroutine CUBLAS_dgemv_t(parentThreadId, threadID, a, x, y, alpha, beta, incx, incy)
        ! Perform matrix-vector multiplication (gemv) with matrices on the GPGPU
        ! Do transpose for input matrix before multiplication
        integer(INT32),     intent(in)   :: parentThreadID
        integer(INT32),     intent(in)   :: threadID
        type(CUDAMatrix), intent(in)   :: a
        type(CUDAVector), intent(in)   :: x
        type(CUDAVector), intent(inout):: y
        real(REAL64),       intent(in)   :: alpha, beta
        integer(INT32),     intent(in)   :: incx, incy
        integer(INT32)                   :: device
        device = mod(parentThreadId-1, a%number_of_devices)+1
        call my_cublas_dgemv_t(parentThreadId, threadID, a%dims(1), a%dims(2),&
                              alpha, &
                              a%devPtr(device),a%str(2, device),&
                              x%devPtr(device),incx,&
                              beta,&
                              y%devPtr(device),incy)
            
    end subroutine
        
    subroutine CUBLAS_cube_sum(parentThreadId, a, b, alpha)
        ! b <- alpha * a + b
        ! Based on daxpy
        integer(INT32),     intent(in)   :: parentThreadID
        type(CUDACube) :: a, b
        real(REAL64) :: alpha
        integer(INT32) :: i
        integer(INT32)                   :: device
!        do i=0,a%dims(Y_)*a%dims(Z_)-1
!                call my_cublas_daxpy(handle,a%dims(X_),alpha,&
!                                            a%devPtr+ i * a%str(Y_) * a%sz,&
!                                            b%devPtr+ i * b%str(Y_) * b%sz)
!        end do
!        print*,a%str(X_),a%dims(Y_),a%dims(Z_)
        device = mod(parentThreadId-1, a%number_of_devices)+1
        call my_cublas_daxpy(parentThreadId, a%str(Y_, device)*a%dims(Y_)*a%dims(Z_),alpha,&
                             a%devPtr(device),&
                             b%devPtr(device))
    end subroutine

    function CUDACube_init_page_locked_1D_cube(cube_shape) result(result_pointer)
        integer, intent(in)   :: cube_shape(3)
        type(C_PTR)           :: c_pointer
        real(REAL64), pointer :: result_pointer(:)
        c_pointer = CUDACube_init_page_locked_cube_cuda(cube_shape)

        call c_f_pointer(c_pointer, result_pointer, [product(cube_shape)])
    end function

    function CUDACube_init_page_locked_cube(cube_shape) result(result_pointer)
        integer, intent(in)   :: cube_shape(3)
        type(C_PTR)           :: c_pointer
        real(REAL64), contiguous, pointer :: result_pointer(:, :, :)
        c_pointer = CUDACube_init_page_locked_cube_cuda(cube_shape)

        call c_f_pointer(c_pointer, result_pointer, cube_shape)
    end function

    subroutine CUDACube_destroy_page_locked_1D_cube(f_pointer)
        real(REAL64), pointer, intent(inout) :: f_pointer(:)
        call CUDACube_destroy_page_locked_cube_cuda(c_loc(f_pointer))
    end subroutine

    subroutine CUDACube_destroy_page_locked_cube(f_pointer)
        real(REAL64), pointer, intent(inout) :: f_pointer(:, :, :)
        call CUDACube_destroy_page_locked_cube_cuda(c_loc(f_pointer))
    end subroutine
    
    function CUDA_get_number_of_devices() result(number_of_devices)
        integer :: number_of_devices

        call get_number_of_devices(number_of_devices)
    end function
        
    subroutine CUDAsync(parentThreadId)
        integer(INT32),     intent(in)   :: parentThreadID
        call cuda_sync(parentThreadId)
    end subroutine

    subroutine CUDAsync_all()
        call cuda_sync_all()
    end subroutine

    subroutine CUDA_enable_peer_communication()
        call cuda_enable_peer_comm()
    end subroutine
    

    subroutine CUDA_check_memory()

    end subroutine


end module
