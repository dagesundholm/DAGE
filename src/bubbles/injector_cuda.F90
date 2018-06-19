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
module CUDAInjector_class
    use globals_m
    use ISO_C_BINDING
    use grid_class
    use harmonic_class
    use cuda_m
    implicit none

    public :: CUDAInjector_init, CUDAInjector

    private

    type :: CUDAInjector
        private
        type(CudaCube)        :: cube
        type(Grid3D), pointer :: grid
    contains
        procedure, private :: CUDAInjector_download
        procedure, public  :: inject_to => CUDAInjector_inject_to
        generic :: download => CUDAInjector_download
        procedure :: destroy => CUDAInjector_destroy
    end type

    interface CUDAInjector
        module procedure :: CUDAInjector_init
    end interface

    interface
        subroutine bubbles_inject_to_cuda(bubbles, grid, lmin, cudacube, &
                                          cube, offset, cube_host_shape) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: bubbles
            type(C_PTR), value    :: grid
            integer(C_INT), value :: lmin
            type(C_PTR), value    :: cudacube
            real(C_DOUBLE)        :: cube(*)
            integer(C_INT), value :: offset
            integer(C_INT)        :: cube_host_shape(3)
        end subroutine
    end interface
contains

    function CUDAInjector_init(grid) result(new)
        !> The grid the injection happens to, the input grid must be cuda-inited
        type(Grid3D),      intent(in), target :: grid
        type(CUDAInjector)                    :: new
  
        new%grid => grid
        ! init cube where the memory is sliced and all memory is not allocated on every device
        new%cube = CudaCube(grid%get_shape())
    end function 


    subroutine CUDAInjector_inject_to( self, bubbls, lmin, cube, cube_offset, cube_host_shape)
        class(CUDAInjector),   intent(inout)  :: self
        type(C_PTR),           intent(in)     :: bubbls
        integer,               intent(in)     :: lmin
        real(REAL64),          intent(inout)  :: cube(:, :, :)
        integer,               intent(in)     :: cube_offset
        integer,               intent(in)     :: cube_host_shape(3)
    
        call bubbles_inject_to_cuda(bubbls, self%grid%cuda_interface, lmin, self%cube%cuda_interface, &
                                    cube, cube_offset, cube_host_shape)
    end subroutine

    
    subroutine CUDAInjector_download( self )
        class(CUDAInjector) :: self
        call self%cube%download()
    end subroutine

    subroutine CUDAInjector_destroy( self )
        class(CUDAInjector) :: self
        call self%cube%destroy()
    end subroutine


end module

