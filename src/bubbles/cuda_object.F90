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
module CudaObject_class
#ifdef HAVE_CUDA
    use ISO_C_BINDING
#endif
    implicit none

    type, abstract :: CudaObject
#ifdef HAVE_CUDA
        type(C_PTR), allocatable :: stream_container
        type(C_PTR), allocatable :: cuda_interface
    contains
        procedure(cuda_destroy), public, deferred        :: cuda_destroy
        procedure(cuda_init), public, deferred           :: cuda_init
        procedure, public                                :: is_cuda_inited       => &
                                                                CudaObject_is_inited
        procedure, public                                :: cuda_upload        => &
                                                                CudaObject_cuda_upload
        procedure, public                                :: cuda_download      => &
                                                                CudaObject_cuda_download
        procedure, public                                :: set_cuda_interface => &
                                                                CudaObject_set_cuda_interface
        procedure, public                                :: get_cuda_interface => &
                                                                CudaObject_get_cuda_interface
        procedure, public                                :: set_stream_container => &
                                                                CudaObject_set_stream_container
        procedure, public                                :: get_stream_container => &
                                                                CudaObject_get_stream_container
        procedure, public                                :: dereference_cuda_interface => &
                                                                CudaObject_dereference_cuda_interface
#endif
    end type

#ifdef HAVE_CUDA
    abstract interface
       subroutine cuda_destroy(self)
           import CudaObject
           class(CudaObject), intent(inout)       :: self
       end subroutine

       subroutine cuda_init(self) 
           import CudaObject
           class(CudaObject), intent(inout)       :: self
       end subroutine
    end interface
#endif
contains

#ifdef HAVE_CUDA

    
    pure function CudaObject_is_inited(self) result(inited)
        class(CudaObject), intent(in) :: self
        logical                       :: inited

        inited = allocated(self%cuda_interface)
    end function

    pure subroutine CudaObject_set_stream_container(self, input_stream_container)
        class(CudaObject), intent(inout) :: self
        type(C_PTR),    intent(in)       :: input_stream_container

        self%stream_container = input_stream_container
    end subroutine

    pure function CudaObject_get_stream_container(self) result(result_stream_container)
        class(CudaObject), intent(in) :: self
        type(C_PTR)                   :: result_stream_container

        result_stream_container = self%stream_container
    end function 

    subroutine CudaObject_set_cuda_interface(self, cuda_interface)
        class(CudaObject), intent(inout) :: self
        type(C_PTR),    intent(in)       :: cuda_interface

        self%cuda_interface = cuda_interface
    end subroutine

    function CudaObject_get_cuda_interface(self) result(cuda_interface)
        class(CudaObject), intent(inout) :: self
        type(C_PTR)                      :: cuda_interface
        if (.not. allocated(self%cuda_interface)) then
            call self%cuda_init()
        end if
        cuda_interface = self%cuda_interface
    end function 

    subroutine CudaObject_dereference_cuda_interface(self)
        class(CudaObject), intent(inout) :: self

        if (allocated(self%cuda_interface)) deallocate(self%cuda_interface)
    end subroutine

    subroutine CudaObject_cuda_download(self) 
        class(CudaObject), intent(inout)       :: self
    end subroutine

    subroutine CudaObject_cuda_upload(self, cuda_interface) 
        class(CudaObject),           intent(in) :: self
        type(C_PTR),       optional, intent(in) :: cuda_interface
    end subroutine
#endif
end module
