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
!> @file operator.F90
!! Linear transformation of functions

!> Linear transformation of functions
!!
!! An operator is an object that transforms functions into new
!! functions. A given operator may either act on the `cube` or `bubbles`
!! attributes of a function or both. See operator3d description.
!!
!! @todo: Documentation
! This module implements
! * operator type declarations
! * operator constuctors
! * bubble operators when applicable
module Operators_class
    use Function3D_types_m
    use expo_m
    use globals_m
    use GaussQuad_class
    use grid_class
    use ISO_FORTRAN_ENV
    use timer_m
    use xmatrix_m
!     use LIPBasis_class
    use mpi_m
    use Evaluators_class
    use Bubbles_class
#ifdef HAVE_CUDA
    use cuda_m
#endif

    implicit none

    public :: Operator3D
    public :: operator(.apply.)

    private

    !> Operators acting on Function3D objects.

    !> The (intended future) usage is
    !! ~~~~~~~~~~~~~~~~~~~~~~~{.f90}
    !! myop=MyOperator3D(...)
    !!
    !! g= myop.apply.f
    !! ~~~~~~~~~~~~~~~~~~~~~~~
    !! now the last line reads
    !! ~~~~~~~~~~~~~~~~~~~~~~~{.f90}
    !!  g = myop .apply. f
    !! ~~~~~~~~~~~~~~~~~~~~~~~
    !! Available operators are \ref projector3d, \ref laplacian3d, \ref coulomb3d.
    !! 
    !!  New types of operators should be implemented as types extending
    !!  Operator3D (or any type which extends Operator3D, in which case
    !!  the `transform_bubbles` method does not need to be defined). See
    !!  coulomb3d_init as example. Example of Operator3D that can be defined in
    !!  a new module:
    !! ~~~~~~~~~~~~~~~~~~~~~~~{.f90}
    !! type, extends(Operator3D) :: Crushinator3D
    !! contains
    !!     procedure :: transform_bubbles => Crushinator3D_transform_bubbles
    !! end type
    !!
    !! interface Crushinator3D
    !!     module procedure :: Crushinator3D_init
    !! end interface
    !! contains
    !!
    !! function Crushinator3D_init(gridin, gridout, ...) result(new)
    !!     type(Crushinator3D)                        :: new
    !!     type(Grid3D), intent(in), target           :: gridin
    !!     type(Grid3D), intent(in), target, optional :: gridout
    !! !   ...
    !! !   (Other input parameters)
    !! !   ...
    !!
    !!     type(Grid3D), pointer                      :: gridout_p
    !!
    !!     if(present(gridout)) then
    !!         gridout_p=>gridout
    !!     else
    !!         gridout_p=>gridin
    !!     end if
    !!
    !!     ! WARNING! This type of constructor seems to lead to all kinds of
    !!     ! nasty bugs with gfortran 4.9.
    !!     ! See http://gcc.gnu.org/bugzilla/show_bug.cgi?id=57957
    !!     new=Crushinator3D(&
    !!         gridin  = gridin, &
    !!         gridout = gridout_p, &
    !!         w       = ( w(:) computed from input parameters ) , &
    !!         coda    = ( coda computed from input parameters ), &
    !!         f       = ( f computed from input parameters, gridin, gridout ) )
    !! end function
    !!~~~~~~~~~~~~~~~~~~~~~~~
    !! ## The `f` matrices
    !!
    !! Their dimensions are
    !! ~~~~~~~~~~~~~~~~~~~~~~{.F90}
    !!  self%f(X_)%p( self%gridout%axis(X_)%get_shape(), self%gridin% axis(X_)%get_shape(), size(self%w) )
    !!  self%f(Y_)%p( self%gridin% axis(Y_)%get_shape(), self%gridout%axis(Y_)%get_shape(), size(self%w) ) ! TRANSPOSE
    !!  self%f(Z_)%p( self%gridin% axis(Z_)%get_shape(), self%gridout%axis(Z_)%get_shape(), size(self%w) ) ! TRANSPOSE
    !! ~~~~~~~~~~~~~~~~~~~~~~
    !! \sa grid_class::alloc_transformation_matrices
    type, abstract :: Operator3D
        private
        !> Input grid
        type(Grid3D), public                :: gridin
        !> Output grid
        type(Grid3D), public                :: gridout
        !> Cube transformation matrices
        type(REAL64_3D), public             :: f(3)
        !> Cube transformation weights
        real(REAL64), allocatable, public   :: w(:)
        !> Coda constant (coefficient of identity operator in expansion)
        real(REAL64), public                :: coda = 0.0_REAL64
        !> Type of the resulting function (see \ref function3dtype_m
        !! "Function3D_types_m")
        integer, public                     :: result_type = F3D_TYPE_NONE
        contains
            procedure  :: get_dims          => operator3d_get_dims
            procedure  :: get_result_type   => operator3d_get_result_type
            procedure  :: destroy           => operator3d_destroy
            procedure(cube_operator),   deferred  :: transform_cube 
            procedure(bubble_operator), deferred  :: transform_bubbles       

    end type

    

    !> Interface for new Bubbles transformation functions

    !> @todo Change to function returning Bubbles
    abstract interface
        function bubble_operator(self, bubsin) result(new)
            import
            class(Operator3D), intent(in) :: self
            type(Bubbles), intent(in)     :: bubsin
            type(Bubbles)                 :: new
        end function


        recursive function cube_operator(self, cubein) result(cubeout)
            import
            class(Operator3D), intent(in) :: self
            !> Input cube
            real(REAL64), intent(in)      :: cubein(:,:,:)
            real(REAL64), allocatable     :: cubeout(:,:,:)
        end function

        
    end interface
   
    integer, parameter :: IN_=1
    integer, parameter :: OUT_=2
contains

    pure function Operator3D_get_dims(self) result(dims)
        class(Operator3D), intent(in)  :: self

        integer                        :: dims(X_:Z_, IN_:OUT_) !(3,2)

        dims(:,IN_) =self%gridin% get_shape()
        dims(:,OUT_)=self%gridout%get_shape()
    end function

    function Operator3D_get_result_type(self) result(result_type)
        class(Operator3D), intent(in) :: self
        integer                       :: result_type

        result_type=self%result_type
    end function

    !> Destructor
    pure subroutine Operator3D_destroy(self)
            class(Operator3D), intent(inout) :: self

            if (allocated(self%f(X_)%p)) deallocate(self%f(X_)%p)
            if (allocated(self%f(Y_)%p)) deallocate(self%f(Y_)%p)
            if (allocated(self%f(Z_)%p)) deallocate(self%f(Z_)%p)

            if (allocated(self%w))  deallocate(self%w)

            call self%gridin%destroy()
            call self%gridout%destroy()
    end subroutine



end module

