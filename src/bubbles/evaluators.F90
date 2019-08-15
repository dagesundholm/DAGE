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
!> @file evaluators.F90
!! Classes for interpolation and integration of scalar functions
module Evaluators_class
    use globals_m
    use grid_class
    use LIPBasis_Class
    use xmatrix_m
    use CartIter_class
    use ISO_FORTRAN_ENV
    use CudaObject_class
    use Points_class
#ifdef HAVE_CUDA
    use cuda_m
#endif
#ifdef HAVE_OMP
    use omp_lib
#endif
    implicit none

    public :: Interpolator, Interpolator1D
    public :: SimpleInterpolator1D
    public :: Integrator, Integrator1D
    public :: InOutIntegrator1D
    public :: Evaluator
    public :: extrapolate_first_nlip7
    public :: Evaluator_destroy
#ifdef HAVE_CUDA
    public :: Integrator3D_destroy, Integrator3D_init, Integrator3D_integrate
    public :: Evaluator_destroy_cuda, Evaluator_evaluate_grid_cuda,  &
              Evaluator_evaluate_grid_x_gradients_cuda, Evaluator_evaluate_grid_y_gradients_cuda, &
              Evaluator_evaluate_grid_z_gradients_cuda, Evaluator_evaluate_grid_without_gradients_cuda, &
              Evaluator_evaluate_points_cuda,  &
              Evaluator_evaluate_points_x_gradients_cuda, Evaluator_evaluate_points_y_gradients_cuda, &
              Evaluator_evaluate_points_z_gradients_cuda, Evaluator_evaluate_points_without_gradients_cuda
#endif

    private

    !> Class for interpolation of scalar functions
    
    !> Interpolator instantiation requires only an array of Grid1D. These
    !! grids form the N-dimensional Cartesian grid in which we will
    !! interpolate.
    !!
    !!     foo=Interpolator(grids(:))
    !!
    !! Then, we interpolate by passing the coefficients and the points at
    !! which we want to interpolate:
    !!
    !!     interpolated_values=foo%eval(cube, points)
    !!
    !! If a derivative order was passed at instantiation
    !!
    !!     foo=Interpolator(grids(:), der_order=2)
    !!
    !! \c foo%%eval will return *all* possible directional derivatives up to the
    !! order requested. The derivatives are computed in the order provided by
    !! cartiter_class::cartiter.
    !!
    !! Example usage:
    !! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.f90}
    !! program inter_test
    !!     use interpolator_class
    !!     use grid_class
    !!     use globals_m
    !!     implicit none
    !!     type(Interpolator) :: foo
    !!     type(Grid3D),pointer :: my_grid3d
    !! 
    !!     integer, parameter :: NLIP=5, NP=500
    !!     integer, parameter :: NCELL3D(3)=[20,20,20], NDIM3D(3)=NCELL3D*(NLIP-1)+1
    !!     real(REAL64), parameter :: A=2.d0
    !! 
    !!     integer :: i, j, k
    !! 
    !!     real(REAL64) :: cube(NDIM3D(1),NDIM3D(2),NDIM3D(3)), x, y, z
    !! 
    !!     real(REAL64) :: pts(3,NP+1), outvalues(1,NP+1)
    !! 
    !!     ! New grid  
    !!     my_grid3d => Grid3D(reshape([-A,A,-A,A,-A,A],[2,3]), NDIM3D, NLIP)
    !!     ! Create Interpolator
    !!     foo=Interpolator(my_grid3d%axis,0)
    !!     
    !!     ! Give some values to the cube
    !!     do i=1,NDIM3D(1)
    !!         x=-A+2.d0*A/(NDIM3D(1)-1)*(i-1)
    !!         do j=1,NDIM3D(2)
    !!             y=-A+2.d0*A/(NDIM3D(2)-1)*(j-1)
    !!             do k=1,NDIM3D(3)
    !!                 cube(i,j,k)=sin(x*x+y*y)
    !!             end do
    !!         end do
    !!     end do
    !! 
    !!     ! Interpolation points mesh
    !!     pts(1,:)=[(-A+2.d0*A/NP*(j-1),j=1,NP+1)]
    !!     pts(2:3,:)=0.d0
    !! 
    !!     ! Perform interpolation
    !!     outvalues=foo%eval(cube,pts)
    !!
    !!     ! Print values
    !!     print'(4f16.8)', (pts(:,i),outvalues(1,:),  i=1,NP+1)
    !! end program
    !! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    type :: Interpolator
        private
        !> Number of dimensions (1D, 2D, 3D, etc)
        integer(INT32) :: ndim
        !> Order of the LIP's +1
        integer(INT32) :: nlip
        !> Derivative order
        integer(INT32) :: der=0
        !> Number of derivatives
        integer(INT32) :: num_der=0
        !> The grids for each dimension (size \c ndim)
        type(Grid1DPointer), allocatable  :: gr(:)
        !> The number of points in each dimension (size \c ndim)
        integer(INT32), allocatable :: gdim(:)
        !> Size of the coefficients block (\c product(gdim))
        integer(INT32) :: blocksz=1
        !> Polynomial coefficients
        !! \f$\partial_{k}p_{i}(x)=\sum_{j=0}^{N-1}\f$ 
        !!      \c pcoeffs(k)%%p(i,j) \f$x^{N-j}\f$
        type(REAL64_2D), allocatable :: pcoeffs(:)
        !> Possible combinations for derivatives, shape: (ndim, num_der)
        integer(INT32),  allocatable :: derivative_possibilities(:, :)
        !> The changed dimension in derivative, shape: (num_der)
        integer(INT32),  allocatable :: changed_dimensions(:)
    contains
        procedure :: eval => Interpolator_eval
        procedure :: destroy => Interpolator_destroy
    end type

    !> Simplification of Interpolator to only 1D grids
    type :: Interpolator1D
        private
        !> Order of the LIP's +1
        integer(INT32)               :: nlip
        !> Derivative order
        integer(INT32)               :: maximum_derivative_order=0
        !> The grids for each dimension (size \c ndim)
        type(Grid1D), pointer        :: grid
        !> Number of grid points on the grid
        integer(INT32)               :: grid_point_count=1
        !> If the first point is ignored in the evaluation of derivatives etc
        logical                      :: ignore_first
        !> Polynomial coefficients
        !! \f$\partial_{k}p_{i}(x)=\sum_{j=0}^{N-1}\f$ 
        !!      \c pcoeffs(k)%%p(i,j) \f$x^{N-j}\f$
        type(REAL64_2D), allocatable :: polynomial_coefficients(:)
        !> Polynomial coefficients for nlip=nlip-1
        !! \f$\partial_{k}p_{i}(x)=\sum_{j=0}^{N-1}\f$ 
        !!      \c pcoeffs(k)%%p(i,j) \f$x^{N-j}\f$
        type(REAL64_2D), allocatable :: lower_polynomial_coefficients(:)
    contains
        procedure, private :: Interpolator1D_eval, Interpolator1D_eval_array
        generic :: eval => Interpolator1D_eval, Interpolator1D_eval_array
        procedure :: eval_point_cells_array => Interpolator1D_eval_point_cells_array
        procedure :: destroy => Interpolator1D_destroy
    end type

    !> Simplification of Interpolator1D to only interpolation, i.e.,
    !! derivation is not possible with this class
    type :: SimpleInterpolator1D
        private
        !> Order of the LIP's +1
        integer(INT32) :: nlip
        !> The grid on which interpolation happens 
        type(Grid1D), pointer     :: gr
        !> Number of grid points on the grid
        integer(INT32)            :: grid_point_count=1
        !> Polynomial coefficients, size: (nlip, nlip)
        real(REAL64), allocatable :: polynomial_coefficients(:, :)
    contains
        procedure, private :: SimpleInterpolator1D_eval, SimpleInterpolator1D_eval_array
        generic   :: eval    => SimpleInterpolator1D_eval, SimpleInterpolator1D_eval_array
        procedure :: destroy => SimpleInterpolator1D_destroy
    end type
        
    type :: Integrator
        private
        !> Number of dimensions (1D, 2D, 3D, etc)
        integer(INT32) :: ndim
        !> Order of the LIP's +1
        integer(INT32) :: nlip
        !> Number of cells
        integer(INT32), allocatable :: ncell(:)
        !> Size of the coefficients block (\c product(gdim))
        integer(INT32), allocatable :: blocksz(:)
        !> Integrals
        real(REAL64), allocatable :: base_ints(:)
        !> Steps
        type(REAL64_1D), pointer :: h(:)
    contains
        procedure :: eval => Integrator_eval
        procedure :: destroy => Integrator_destroy
    end type

    type :: Integrator1D
        private
        !> Order of the LIP's +1
        integer(INT32)              :: nlip
        !> Number of cells
        integer(INT32)              :: ncell
        !> Integrals
        real(REAL64), allocatable   :: base_ints(:)
        !> Steps
        real(REAL64), allocatable   :: h(:)
    contains
        procedure :: eval => Integrator1D_eval
        procedure :: destroy => Integrator1D_destroy
    end type

    !> Helper object to iterate over slices of arbitrary dimensionality arrays
    type :: ArrayHopper
        integer :: ndim
        integer :: n
        integer :: start
        integer,allocatable :: index(:)
        integer,allocatable :: gdim(:)
        integer,allocatable :: stride(:)
        integer :: idim
        procedure(ArrayHopper_get_stride), pointer :: get_stride
        procedure(ArrayHopper_next_pos), pointer :: next_pos
    contains
        procedure :: set_start => ArrayHopper_set_start
        procedure :: destroy => ArrayHopper_destroy
    end type

    type :: InOutIntegrator1D
        integer                   :: n
        integer                   :: ncell
        integer                   :: sz
        real(REAL64), pointer     :: h(:)
        real(REAL64), allocatable :: ints_in (:,:)
        real(REAL64), allocatable :: ints_out(:,:)
    contains
        procedure :: destroy =>  InOutIntegrator1D_destroy
        procedure :: inwards =>  InOutIntegrator1D_inwards
        procedure :: outwards => InOutIntegrator1D_outwards
    end type

    interface Interpolator
        module procedure :: Interpolator_init
    end interface

    interface Interpolator1D
        module procedure :: Interpolator1D_init
    end interface

   interface SimpleInterpolator1D
        module procedure :: SimpleInterpolator1D_init
    end interface

    interface Integrator
        module procedure :: Integrator_init
    end interface

    interface Integrator1D
        module procedure :: Integrator1D_init
    end interface

    interface InOutIntegrator1D
        module procedure :: InOutIntegrator1D_init
    end interface

    interface ArrayHopper
        module procedure :: ArrayHopper_init
    end interface

#ifdef HAVE_CUDA
    interface
        type(C_PTR) function Integrator3D_init(streamContainer, grid)  bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: streamContainer
            type(C_PTR), value    :: grid

        end function
    end interface

    
    interface
        subroutine Integrator3D_destroy(integrator)  bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: integrator

        end subroutine
    end interface

    interface
        subroutine Integrator3D_integrate(integrator, cube, offset, cube_host_shape, res)  bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: integrator
            real(C_DOUBLE)        :: cube(*)
            integer(C_INT), value :: offset
            integer(C_INT)        :: cube_host_shape(3)
            real(C_DOUBLE)        :: res

        end subroutine
    end interface
#endif

!%%%%%%%%%%%%%%% Evaluator abstract type definition %%%%%%%%%%%%%%%%%%

    type, abstract, extends(CudaObject) :: Evaluator
        !> Determines if this evaluator allocates (gpu) memory for the gradients
        logical                   :: high_memory_profile
        !> The output grid of the evaluator, i.e., result points
        type(Grid3D), pointer     :: grid
#ifdef HAVE_CUDA
        !> The result cuda cube object, needed only if the points are replaced by a grid
        type(CudaCube), allocatable :: result_cuda_cube
        !> The gradients x direction cuda cube object, needed only if the points are replaced by a grid
        type(CudaCube), allocatable :: gradient_cuda_cube_x
        !> The gradients y direction cuda cube object, needed only if the points are replaced by a grid
        type(CudaCube), allocatable :: gradient_cuda_cube_y
        !> The gradients y direction cuda cube object, needed only if the points are replaced by a grid
        type(CudaCube), allocatable :: gradient_cuda_cube_z
        !> The object to store gradients in x-direction for point-evaluations
        type(Points),   allocatable ::  output_derivative_x_points
        !> The object to store gradients in y-direction for point-evaluations
        type(Points),   allocatable ::  output_derivative_y_points
        !> The object to store gradients in z-direction for point-evaluations
        type(Points),   allocatable ::  output_derivative_z_points
        !> The object to store the results for point-evaluations
        type(Points),   allocatable ::  result_points
    
        !> Flag to indicate whether the gradient cuda cubes were preinited before passing to this object 
        logical                   :: gradient_cuda_cube_preinited
        !> Flag to indicate whether the result cuda cube was preinited before passing to this object 
        logical                   :: result_cuda_cube_preinited
        !> Flag to indicate whether the cuda grid was preinited before passing to this object 
        logical                   :: cuda_grid_preinited
        
#endif
    contains
        procedure :: set_output_grid                        =>  Evaluator_set_output_grid
        procedure :: set_output_points                      =>  Evaluator_set_output_points
        procedure :: destroy_output_cubes                   =>  Evaluator_destroy_output_cubes
        procedure :: destroy_output_points                  =>  Evaluator_destroy_output_points
#ifdef HAVE_CUDA
        procedure :: cuda_evaluate_grid_derivative          =>  Evaluator_cuda_evaluate_grid_derivative
        procedure :: cuda_evaluate_points_derivative        =>  Evaluator_cuda_evaluate_points_derivative
        procedure :: cuda_evaluate_grid_without_gradients   =>  Evaluator_cuda_evaluate_grid_without_gradients
        procedure :: cuda_evaluate_points_without_gradients =>  Evaluator_cuda_evaluate_points_without_gradients
#endif
    end type

!%%%%%%%%%%%%%%% Evaluator_cuda_interfaces %%%%%%%%%%%%%%%%%%

#ifdef HAVE_CUDA

    interface
        pure subroutine Evaluator_evaluate_grid_cuda(evaluator, grid, result_cube, &
                gradient_cube_x, gradient_cube_y, gradient_cube_z, gradient_direction, finite_diff_order) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: evaluator
            type(C_PTR),    value :: grid
            type(C_PTR),    value :: result_cube
            type(C_PTR),    value :: gradient_cube_x
            type(C_PTR),    value :: gradient_cube_y
            type(C_PTR),    value :: gradient_cube_z
            integer(C_INT), value :: gradient_direction
            integer(C_INT), value :: finite_diff_order
        end subroutine
    end interface

    interface
        pure subroutine Evaluator_evaluate_grid_without_gradients_cuda(evaluator, grid, result_cube) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: evaluator
            type(C_PTR),    value :: grid
            type(C_PTR),    value :: result_cube
        end subroutine
    end interface

    interface
        pure subroutine Evaluator_evaluate_grid_x_gradients_cuda(evaluator, grid, result_cube, finite_diff_order) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: evaluator
            type(C_PTR),    value :: grid
            type(C_PTR),    value :: result_cube
            integer(C_INT), value :: finite_diff_order
        end subroutine
    end interface

    interface
        pure subroutine Evaluator_evaluate_grid_y_gradients_cuda(evaluator, grid, result_cube, finite_diff_order) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: evaluator
            type(C_PTR),    value :: grid
            type(C_PTR),    value :: result_cube
            integer(C_INT), value :: finite_diff_order
        end subroutine
    end interface

    interface
        pure subroutine Evaluator_evaluate_grid_z_gradients_cuda(evaluator, grid, result_cube, finite_diff_order) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: evaluator
            type(C_PTR),    value :: grid
            type(C_PTR),    value :: result_cube
            integer(C_INT), value :: finite_diff_order
        end subroutine
    end interface

    interface
        pure subroutine Evaluator_evaluate_points_cuda(evaluator, result_points,  &
             gradient_points_x, gradient_points_y, gradient_points_z, gradient_direction) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: evaluator
            type(C_PTR),    value :: result_points
            type(C_PTR),    value :: gradient_points_x
            type(C_PTR),    value :: gradient_points_y
            type(C_PTR),    value :: gradient_points_z
            integer(C_INT), value :: gradient_direction
        end subroutine
    end interface

    interface
        pure subroutine Evaluator_evaluate_points_without_gradients_cuda(evaluator, result_points) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: evaluator
            type(C_PTR),    value :: result_points
        end subroutine
    end interface

    interface
        pure subroutine Evaluator_evaluate_points_x_gradients_cuda(evaluator, result_points) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: evaluator
            type(C_PTR),    value :: result_points
        end subroutine
    end interface

    interface
        pure subroutine Evaluator_evaluate_points_y_gradients_cuda(evaluator, result_points) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: evaluator
            type(C_PTR),    value :: result_points
        end subroutine
    end interface

    interface
        pure subroutine Evaluator_evaluate_points_z_gradients_cuda(evaluator, result_points) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: evaluator
            type(C_PTR),    value :: result_points
        end subroutine
    end interface

    interface
        pure subroutine Evaluator_destroy_cuda(evaluator) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: evaluator
        end subroutine
    end interface
#endif

contains
    !%%%%%%%%%%%%%%% SimpleInterpolator %%%%%%%%%%%%%%%%%%

    !> SimpleInterpolator1D constructor. Note, this interpolator is 
    !! not cabaple of calculating derivatives. Thus, it is Simple
    function SimpleInterpolator1D_init(grid) result(self)
        !> 1-dimensional grids
        type(Grid1D), target         :: grid
        type(SimpleInterpolator1D)   :: self
        type(REAL64_2D), allocatable :: pol_coefficients(:)


        self%gr=>grid
        self%grid_point_count = self%gr%get_shape()
        self%nlip=self%gr%get_nlip()

        ! evalueate polynomial coefficients 
        pol_coefficients = self%gr%lip%coeffs(0)
        self%polynomial_coefficients= pol_coefficients(1)%p(:, :)
        deallocate(pol_coefficients(1)%p)
        deallocate(pol_coefficients)
    end function

     !> Interpolate at 'points', for 
    function SimpleInterpolator1D_eval(self, f_vals, points) result(res)
        class(SimpleInterpolator1D), intent(in)  :: self
        real(REAL64), intent(in), target         :: f_vals(self%grid_point_count)
        !> x,y,z coordinates of points where f_vals is evaluated 
        !! Dimensions self%ndim, number of points
        real(REAL64), intent(in)                 :: points(:)
        real(REAL64)                             :: res(size(points))
 
        integer(INT32) :: icell, start, npoints, ipoint
        real(REAL64)   :: crd_cell, polys(self%nlip)

        npoints=size(points)

        ! The first temporary array is the full cube

        ! Iterate over points
#ifdef HAVE_OMP
        !$OMP PARALLEL DO IF (npoints > 1000)&
        !$OMP& PRIVATE(icell, crd_cell, polys, start) 
#endif 
        do ipoint=1, npoints
            ! Find the cell of the ith point at dimension idim
            icell = self%gr%get_icell(points(ipoint))

            if (icell > 0) then
                ! Compute local cell coordinates
                crd_cell=self%gr%x2cell(points(ipoint),icell)

                ! Evaluate polynomials in cell coordinates
                polys=eval_polys( self%polynomial_coefficients, crd_cell )

                start = (icell - 1) * (self%nlip - 1) + 1
                res(ipoint) = sum( f_vals(start:start+self%nlip-1) * polys )
            else
                res(ipoint) = 0.0d0
            end if
        end do
#ifdef HAVE_OMP
        !$OMP END PARALLEL DO
#endif

        return
    end function

    function SimpleInterpolator1D_eval_array(self, f_value_arrays, points) result(res)
        class(SimpleInterpolator1D), intent(in)  :: self
        real(REAL64), intent(in)                 :: f_value_arrays(:, :)
        !> x,y,z coordinates of points where f_vals is evaluated 
        !! Dimensions self%ndim, number of points
        real(REAL64), intent(in)                 :: points(:)
        real(REAL64)                             :: res(size(points), size(f_value_arrays, 2))
 
        integer(INT32) :: icell(size(points)), start(size(points)), npoints, ipoint, iarray, narrays
        real(REAL64)   :: crd_cell, polys(self%nlip, size(points))

        call bigben%split("SimpleInterpolator - eval array")
        npoints = size(points)
        narrays = size(f_value_arrays, 2)

        ! Iterate over points
#ifdef HAVE_OMP
        !$OMP PARALLEL DO IF (npoints > 1000) PRIVATE(iarray)
#endif 
        do ipoint=1,npoints
            ! Find the cell of the ith point
            icell(ipoint) = self%gr%get_icell(points(ipoint))
            res(ipoint, :) = 0.0d0
            if (icell(ipoint) > 0) then
                ! Evaluate polynomials in cell coordinates
                polys(:, ipoint) = eval_polys( self%polynomial_coefficients, self%gr%x2cell(points(ipoint),icell(ipoint)))

                start(ipoint) = (icell(ipoint) - 1) * (self%nlip - 1) + 1
                forall (iarray = 1:narrays)
                    res(ipoint, iarray) = &
                        sum( f_value_arrays(start(ipoint):start(ipoint)+self%nlip-1, iarray) * polys(:, ipoint) )
                end forall
            end if
        end do
#ifdef HAVE_OMP
        !$OMP END PARALLEL DO
#endif
        call bigben%stop()
    end function

    subroutine SimpleInterpolator1D_destroy(self)
        class(SimpleInterpolator1D) :: self

        nullify(self%gr)
        deallocate(self%polynomial_coefficients)
    end subroutine

    !%%%%%%%%%%%%%%% Interpolator1D %%%%%%%%%%%%%%%%%%

    !> Interpolator1D constructor
    function Interpolator1D_init(grid, maximum_derivative_order, ignore_first) result(new)
        !> 1-dimensional grid on which the interpolation will occur
        type(Grid1D), target                 :: grid
        !> Maximum derivative order required
        integer(INT32), optional, intent(in) :: maximum_derivative_order
        !> If the first point is ignored when evaluating derivatives etc
        logical,        optional, intent(in) :: ignore_first
        type(Interpolator1D)                 :: new

        integer(INT32)                       :: ncell
        integer(INT32)                       :: i ! lnw: remove

        new%grid => grid
        new%nlip = new%grid%get_nlip()
        new%grid_point_count = new%grid%get_shape()

        if(present(maximum_derivative_order)) then
            new%maximum_derivative_order = maximum_derivative_order
        else
            new%maximum_derivative_order = 0
        end if
        
        if (present(ignore_first)) then
            new%ignore_first = ignore_first
        else
            new%ignore_first = .FALSE.
        end if
        new%lower_polynomial_coefficients = new%grid%lower_lip%coeffs(new%maximum_derivative_order)
        new%polynomial_coefficients = new%grid%lip%coeffs(new%maximum_derivative_order)

    end function

    !> Interpolate at 'points', for 
    function Interpolator1D_eval_array(self, f_value_arrays, points) result(res)
        class(Interpolator1D), intent(in)  :: self
        !> coordinates of points where f_vals is evaluated
        real(REAL64), intent(in)           :: f_value_arrays(:, :)
        real(REAL64), intent(in)           :: points(:)
        real(REAL64)                       :: res(size(points), self%maximum_derivative_order+1, &
                                                  size(f_value_arrays, 2))
        integer(INT32)                     :: icell, start
        real(REAL64)                       :: crd_cell, h
        !! loop counters and size variables
        integer(INT32)                     :: ipoint, npoints, derivative_order, iarray


        npoints=size(points)

        ! Iterate over points
#ifdef HAVE_OMP
        !$OMP PARALLEL DO IF (npoints > 1000)
#endif
        do ipoint=1, npoints
            ! Compute local cell coordinate
            ! Find the cell of the ith point
            icell = self%grid%get_icell(points(ipoint))

            if (icell == 1 .and. self%ignore_first) then
                ! Change of variable to cell coordinates
                crd_cell = self%grid%x2cell(points(ipoint), icell)

                ! get the cell step of the cell we are handling
                h        = self%grid%get_cell_scale(icell)

                ! get the starting grid point index for the selected cell
                start    = (icell - 1) * (self%nlip - 1) + 2 !  = 2

                ! evaluate the derivatives for point ipoint
                forall (derivative_order = 0 : self%maximum_derivative_order, iarray = 1 : size(f_value_arrays, 2))
                    ! Evaluate polynomials in cell coordinates
                    res(ipoint, derivative_order + 1, iarray) = sum(   &
                            f_value_arrays(start: start + (self%nlip - 2), iarray) &
                            * eval_polys(self%lower_polynomial_coefficients(derivative_order+1)%p(:,:), crd_cell) ) &
                            * h**(-derivative_order)
                end forall

            else if (icell > 0) then
                ! Change of variable to cell coordinates
                crd_cell = self%grid%x2cell(points(ipoint), icell)

                ! get the cell step of the cell we are handling
                h        = self%grid%get_cell_scale(icell)

                ! get the starting grid point index for the selected cell
                start    = (icell - 1) * (self%nlip - 1) + 1

                ! evaluate the derivatives for point ipoint
                forall (derivative_order = 0 : self%maximum_derivative_order, iarray = 1 : size(f_value_arrays, 2))
                    ! Evaluate polynomials in cell coordinates
                    res(ipoint, derivative_order + 1, iarray) = sum(   &
                            f_value_arrays(start: start + (self%nlip - 1), iarray) &
                            * eval_polys(self%polynomial_coefficients(derivative_order+1)%p(:,:), crd_cell) ) &
                            * h**(-derivative_order)
                end forall
            else
                res(ipoint, :, :) = 0
            end if
        end do
#ifdef HAVE_OMP
        !$OMP END PARALLEL DO
#endif

        return
    end function


    !> Interpolate at gridpoints belonging to cells that contain 'points' for all f_value_arrays
    !!
    function Interpolator1D_eval_point_cells_array(self, f_value_arrays, points) result(res)
        class(Interpolator1D), intent(in), target  :: self
        !> coordinates of points where f_vals is evaluated
        real(REAL64), intent(in)                   :: f_value_arrays(:, :)
        real(REAL64), intent(in)                   :: points(:)
        real(REAL64)                               :: res(size(f_value_arrays, 1), self%maximum_derivative_order+1, &
                                                          size(f_value_arrays, 2))
        integer(INT32)                             :: icell(size(points)), start(size(points))
        real(REAL64)                               :: h(size(points))
        real(REAL64), allocatable                  :: coordinates(:, :)
        !! loop counters and size variables
        integer(INT32)                             :: ipoint, npoints, derivative_order, iarray, ilip


        npoints=size(points)
        res = 0.0d0
        allocate(coordinates(0:self%nlip-1, size(points)))
#ifdef HAVE_OMP
        !$OMP PARALLEL DO IF (npoints > 1000)
#endif 
        ! Iterate over points
        do ipoint=1, npoints
            ! Compute local cell coordinate
            ! Find the cell of the ith point
            icell(ipoint) = self%grid%get_icell(points(ipoint))

            ! get the starting grid point index for the selected cell 
            start(ipoint) = (icell(ipoint) - 1) * (self%nlip - 1) + 1

            ! Get cell coordinates
            coordinates(:, ipoint) = self%grid%get_coordinates([icell(ipoint), icell(ipoint)])

            ! get the cell step of the cell we are handling
            h(ipoint)        = self%grid%get_cell_scale(icell(ipoint))

            ! evaluate the derivatives for point ipoint
            forall (derivative_order = 0 : self%maximum_derivative_order, &
                    iarray = 1 : size(f_value_arrays, 2), ilip = 0 : self%nlip -1)
                ! Evaluate polynomials in cell coordinates
                res(start(ipoint)+ilip, derivative_order + 1, iarray) = sum(   &
                        f_value_arrays(start(ipoint): start(ipoint) + (self%nlip - 1), iarray) &
                        * eval_polys(self%polynomial_coefficients(derivative_order+1)%p(:,:), &
                          self%grid%x2cell( coordinates(ilip, ipoint), icell(ipoint)) ) ) &
                        * h(ipoint)**(-derivative_order)

            end forall
        end do
#ifdef HAVE_OMP
        !$OMP END PARALLEL DO
#endif
        
        deallocate(coordinates)

        return
    end function


    
    !> Interpolate at 'points', for 
    function Interpolator1D_eval(self, f_vals, points) result(res)
        class(Interpolator1D), intent(in)  :: self
        !> coordinates of points where f_vals is evaluated 
        real(REAL64), intent(in)           :: points(:)
        real(REAL64), intent(in)           :: f_vals(self%grid_point_count)
        real(REAL64)                       :: res(size(points), self%maximum_derivative_order+1)
        integer(INT32)                     :: icell, start
        real(REAL64)                       :: crd_cell, h
        !! loop counters and size variables
        integer(INT32)                     :: ipoint, npoints, derivative_order


        npoints=size(points)

        ! Iterate over points
#ifdef HAVE_OMP
        !$OMP PARALLEL DO IF (npoints > 1000)
#endif 
        do ipoint=1, npoints
            ! Compute local cell coordinate
            ! Find the cell of the ith point
            icell = self%grid%get_icell(points(ipoint))
            
            if (icell > 0 .and. self%ignore_first) then

                ! Change of variable to cell coordinates
                crd_cell = self%grid%x2cell(points(ipoint), icell)

                ! get the cell step of the cell we are handling
                h        = self%grid%get_cell_scale(icell)

                ! get the starting grid point index for the selected cell 
                start    = (icell - 1) * (self%nlip - 1) + 2

                ! evaluate the derivatives for point ipoint
                forall (derivative_order = 0 : self%maximum_derivative_order)
                    ! Evaluate polynomials in cell coordinates
                    res(ipoint, derivative_order + 1) = sum(   &
                            f_vals(start: start + (self%nlip - 2)) &
                            * eval_polys(self%lower_polynomial_coefficients(derivative_order+1)%p(:,:), crd_cell) ) &
                            * h**(-derivative_order)
                end forall

            else if (icell > 0) then

                ! Change of variable to cell coordinates
                crd_cell = self%grid%x2cell(points(ipoint), icell)

                ! get the cell step of the cell we are handling
                h        = self%grid%get_cell_scale(icell)

                ! get the starting grid point index for the selected cell 
                start    = (icell - 1) * (self%nlip - 1) + 1

                ! evaluate the derivatives for point ipoint
                forall (derivative_order = 0 : self%maximum_derivative_order)
                    ! Evaluate polynomials in cell coordinates
                    res(ipoint, derivative_order + 1) = sum(   &
                            f_vals(start: start + (self%nlip - 1)) &
                            * eval_polys(self%polynomial_coefficients(derivative_order+1)%p(:,:), crd_cell) ) &
                            * h**(-derivative_order)
                end forall
            else
                res(ipoint, :) = 0.0d0
            end if
        end do
#ifdef HAVE_OMP
        !$OMP END PARALLEL DO
#endif

        return
    end function

    subroutine Interpolator1D_destroy(self)
        class(Interpolator1D) :: self
        integer               :: k

        ! dereference the grid
        nullify(self%grid)

        ! deallocate the polynomial coefficients for each derivative order
        do k=0, self%maximum_derivative_order
            deallocate(self%polynomial_coefficients(k+1)%p)
        end do
        deallocate(self%polynomial_coefficients)
    end subroutine

    !%%%%%%%%%%%%%%% Interpolator %%%%%%%%%%%%%%%%%%

    !> Interpolator constructor
    function Interpolator_init(grids,deriv_order) result(self)
        !> Array of \c ndim 1-dimensional grids
        type(Grid1DPointer)                  :: grids(:)
        !> Maximum derivative order required
        integer(INT32), optional, intent(in) :: deriv_order
        type(Interpolator)                   :: self

        integer(INT32) :: idim, ncell
        type(CartIter) :: cart

        self%ndim=size(grids)   
        self%gr = grids
        self%nlip=grids(1)%p%get_nlip()

        if(present(deriv_order)) then
            self%der=deriv_order
        else
            self%der=0
        end if
        
        allocate(self%gdim(self%ndim))
        self%pcoeffs=self%gr(1)%p%lip%coeffs(self%der)
        do idim=1,self%ndim
            ncell=self%gr(idim)%p%get_ncell()
            self%gdim(idim)=ncell*(self%nlip-1)+1

            self%blocksz=self%blocksz*self%gdim(idim)
        end do

        cart=CartIter(self%ndim, self%der)
        self%num_der=cart%get_num_iter()
        allocate(self%derivative_possibilities(self%ndim, self%num_der))
        allocate(self%changed_dimensions(self%num_der))
        call cart%get_possible_values(self%derivative_possibilities, self%changed_dimensions)
        call cart%destroy()
    end function

    subroutine Interpolator_destroy(self)
        class(Interpolator) :: self
        integer :: k, idim

        deallocate(self%gdim)
        deallocate(self%derivative_possibilities)
        deallocate(self%changed_dimensions)
        do k=0,self%der
            deallocate(self%pcoeffs(k+1)%p)
        end do
        deallocate(self%gr)
        deallocate(self%pcoeffs)
    end subroutine

    !> Interpolate at 'points', for 
    function Interpolator_eval(self, f_vals, points) result(res)
        class(Interpolator), intent(in)  :: self
        !> x,y,z coordinates of points where f_vals is evaluated 
        !! Dimensions 3, number of points
        real(REAL64), intent(in)         :: points(:,:)
        real(REAL64), intent(in), target :: f_vals(self%blocksz)
        real(REAL64)                     :: res(size(points,dim=2), self%num_der)
 
        integer(INT32)                   :: icell(self%ndim)
        integer(INT32)                   :: idim, start, k, sz, npoints
        integer(INT32)                   :: ipoint, dimflip, ider
        real(REAL64)                     :: crd_cell(self%ndim), polys(self%nlip)

        real(REAL64)                     :: h(self%ndim)
        ! real(REAL64), pointer            :: cellh(:), celld(:)

        type(REAL64_1D), allocatable, target :: tmp_array(:)
        type(REAL64_1D_Pointer), allocatable :: tmp(:)

        type(ArrayHopper) :: hopper
        type(CartIter) :: deriter
        integer :: kappa(self%ndim)
        logical :: continue_iteration, continue_hopping, zero_cell

        
        npoints = size(points, dim=2)
#ifdef HAVE_OMP
        !$OMP PARALLEL IF (npoints > 1000) &
        !$OMP& PRIVATE(ider, start, sz, idim, icell, ipoint, h, polys, hopper, continue_hopping, k, tmp, tmp_array, crd_cell)
#endif 
        hopper=ArrayHopper(self)

        ! Allocate temporary arrays
        allocate(tmp(0:self%ndim))
        allocate(tmp_array(0:self%ndim))
        sz=1
        do idim=self%ndim,1,-1
            allocate(tmp_array(idim)%p(sz), source = 0.0d0)
            tmp(idim)%p => tmp_array(idim)%p
            sz=sz*self%nlip
        end do

        ! The first temporary array is the full cube/bubbles
        tmp(0)%p=>f_vals

        ! Iterate over points
#ifdef HAVE_OMP
        !$OMP DO 
#endif
        pointloop: do ipoint= 1, npoints
            ! Compute local cell coordinate
            do idim=1,self%ndim
                ! Find the cell of the ith point at dimension idim
                icell(idim) = self%gr(idim)%p%get_icell(points(idim, ipoint))
                
                if (icell(idim)==0) then
                    ! If no valid cell, exit
                    res(ipoint, :) = 0.d0
                    print *, "POINT OUT OF BOUNDS"
                    cycle pointloop
                else
                    ! Otherwise, do the coordinate transform

                    ! Change of variable to cell coordinates
                    crd_cell(idim)=self%gr(idim)%p%x2cell(points(idim, ipoint),&
                                                        icell(idim))
                    h(idim)=self%gr(idim)%p%get_cell_scale(icell(idim))
                end if
            end do

            do ider = 1, size(self%derivative_possibilities, 2) 
                ! Iterate over dimensions
                do idim= self%changed_dimensions(ider), self%ndim
                    ! Evaluate polynomials in cell coordinates
                    polys=eval_polys(  self%pcoeffs(self%derivative_possibilities(idim, ider)+1)%p(:,:),&
                                    crd_cell(idim) ) * h(idim)**(-self%derivative_possibilities(idim, ider))
                    ! Contract cube with polys in this dimension
                    call hopper%set_start(idim,icell)
                    k=0
                    call hopper%next_pos(start, continue_hopping)
                    
                    do while(continue_hopping)
                        k=k+1
                        tmp(idim)%p(k) = &
                            sum( tmp(idim-1)%p(start:start+self%nlip-1) * polys )
                        call hopper%next_pos(start, continue_hopping)
                    end do

                end do
                res(ipoint, ider) = tmp(self%ndim)%p(1)
            end do
            
        end do pointloop
#ifdef HAVE_OMP
        !$OMP END DO
#endif

        call hopper%destroy()
        do idim=1,self%ndim
            deallocate(tmp_array(idim)%p)
            nullify(tmp(idim)%p)
        end do
        nullify(tmp(0)%p)
        deallocate(tmp, tmp_array)
#ifdef HAVE_OMP
        !$OMP END PARALLEL
#endif
        return

    end function

    

    !%%%%%%%%%%%%%%% Integrator %%%%%%%%%%%%%%%%%%
    !> Integrator constructor
    function Integrator_init(grids) result(self)
        type(Integrator) :: self
        !> Array of \c ndim 1-dimensional grids
        type(Grid1D), target :: grids(:)

        integer(INT32) :: idim

        self%ndim=size(grids)
        self%nlip=grids(1)%get_nlip()

        allocate(self%ncell(self%ndim))
        allocate(self%blocksz(0:self%ndim))
        allocate(self%h(self%ndim))

        self%base_ints=grids(1)%lip%integrals()

        self%blocksz(self%ndim)=1
        do idim=self%ndim,1,-1
            self%ncell(idim)=grids(idim)%get_ncell()
            self%blocksz(idim-1)=self%blocksz(idim)*grids(idim)%get_shape()
            self%h(idim)%p=grids(idim)%get_cell_scales()
        end do
    end function

    !> Destructor
    pure subroutine Integrator_destroy(self)
        class(Integrator), intent(inout) :: self
        integer                          :: idim

        if (allocated(self%ncell)) deallocate(self%ncell)
        if (allocated(self%blocksz)) deallocate(self%blocksz)
        if (allocated(self%base_ints)) deallocate(self%base_ints)
        if (associated(self%h)) then
            do idim=1,self%ndim
                deallocate(self%h(idim)%p)
            end do
            deallocate(self%h)
        end if
    end subroutine

    !> Integrate over all space
    pure function Integrator_eval(self,f_vals) result(res)
        class(Integrator), intent(in)   :: self
        real(REAL64),target, intent(in) :: f_vals(self%blocksz(0))
        real(REAL64)                    :: res
 
        integer(INT32)                  :: icell
        integer(INT32)                  :: idim,a,k

        type(REAL64_1D), allocatable :: tmp(:)

        ! Allocate temporary arrays
        allocate(tmp(0:1))
        ! The first temporary array is the full cube
        allocate(tmp(0)%p(self%blocksz(0)), source=f_vals)

        ! Iterate over dimensions
        do idim=1,self%ndim
            allocate(tmp(1)%p(self%blocksz(idim)), source = 0.0d0)
            !tmp(1)%p=[ (0.d0, k=1, self%blocksz(idim) ) ]
            a=0
            do k=1, self%blocksz(idim)
                a=a+1
                ! Contract cube with polys in this dimension
                do icell=1,self%ncell(idim)
                    tmp(1)%p(k)=tmp(1)%p(k)+&
                      sum( tmp(0)%p(a:a+self%nlip-1) * self%base_ints ) * &
                          self%h(idim)%p(icell)
                    a=a+self%nlip-1
                end do
            end do
            deallocate(tmp(0)%p)
            tmp(0)%p=tmp(1)%p
            deallocate(tmp(1)%p)
        end do
        res=tmp(0)%p(1)
        deallocate(tmp(0)%p)
        deallocate(tmp)
        return
    end function

    !%%%%%%%%%%%%%%% Integrator1D %%%%%%%%%%%%%%%%%%
    !> Integrator constructor
    function Integrator1D_init(grid) result(self)
        type(Integrator1D) :: self
        !> Integration grid
        type(Grid1D), target :: grid

        integer(INT32) :: idim

        self%nlip=grid%get_nlip()
        self%base_ints=grid%lip%integrals()
       
        self%ncell = grid%get_ncell()
        self%h = grid%get_cell_scales()
    end function

    !> Destructor
    pure subroutine Integrator1D_destroy(self)
        class(Integrator1D), intent(inout) :: self
        integer                          :: idim

        if (allocated(self%base_ints)) deallocate(self%base_ints)
        if (allocated(self%h)) deallocate(self%h)
    end subroutine

    !> Integrate over all space
    pure function Integrator1D_eval(self,f_vals) result(res)
        class(Integrator1D), intent(in) :: self
        real(REAL64),        intent(in) :: f_vals(:)
        real(REAL64)                    :: res
        integer(INT32)                  :: icell
        real(REAL64)                    :: tmp(self%ncell)
        
        ! Contract array with base integrals
        forall (icell=1 : self%ncell)
            tmp(icell) = &
                sum( f_vals((icell-1)*(self%nlip-1)+1:(icell)*(self%nlip-1)+1) * self%base_ints ) * &
                self%h(icell)
        end forall
        res = sum(tmp)
        return
    end function

    !%%%%%%%%%% ArrayHopper %%%%%%%%%%%
    pure function ArrayHopper_init(interpol) result(self)
        class(Interpolator), intent(in) :: interpol
        type(ArrayHopper) :: self
        integer :: idim, jdim

        self%ndim=interpol%ndim
        self%n=interpol%nlip-1
        allocate(self%gdim(self%ndim))
        self%gdim(:)=interpol%gdim(:)
        allocate(self%stride(self%ndim-1))
        allocate(self%index(self%ndim-1))
        ! Setup strides
        self%stride(:)=self%gdim(:self%ndim-1)
        do idim=1,self%ndim-1
            do jdim=idim-1,1,-1
                self%stride(idim)=(self%stride(idim)-self%n)*&
                                  interpol%gdim(jdim)
            end do
        end do
    end function

    pure subroutine ArrayHopper_destroy(self)
        class(ArrayHopper), intent(inout) :: self

        deallocate(self%gdim)
        deallocate(self%index)
        deallocate(self%stride)     
    end subroutine

    pure subroutine ArrayHopper_set_start(self,idim,icell)
        class(ArrayHopper), intent(inout) :: self
        integer,            intent(in) :: idim, icell(3)
        integer                        :: jdim
        
        self%index(:)=1
        if(self%ndim>1) self%index(1)=0
        self%idim=idim
        if (idim==1) then
            ! Position of first block
            self%start=0
            do jdim=self%ndim,2,-1
               self%start=(self%start+(icell(jdim)-1)*self%n)*&
                                self%gdim(jdim-1)
            end do
            self%start=self%start+(icell(1)-1)*self%n+1
            self%get_stride=>ArrayHopper_get_stride
        else
            ! Position of first block, always one for idim>1
            self%start=1
            self%get_stride=>ArrayHopper_get_n
        end if
        if(idim==self%ndim) then
            self%next_pos=>ArrayHopper_next_pos_scalar
        else
            self%start=self%start-self%get_stride(1)
            self%next_pos=>ArrayHopper_next_pos
        end if
    end subroutine

    !> Returns \c .TRUE. if there are more elements to iterate, and sets
    !! \c start to the position of the next element
    pure subroutine ArrayHopper_next_pos(self, start, continue)
        class(ArrayHopper), intent(inout) :: self
        integer,            intent(out)   :: start
        logical,            intent(out)   :: continue

        integer :: idim

        continue=.TRUE.
        do idim=1,self%ndim-self%idim
            if(self%index(idim)<=self%n) then
                self%index(idim)=self%index(idim)+1
                self%start=self%start+self%get_stride(idim)
                start=self%start
                return
            else
                self%index(idim)=1
            end if
        end do
        continue=.FALSE.
    end subroutine

    pure subroutine ArrayHopper_next_pos_scalar(self,start, continue)
        class(ArrayHopper), intent(inout)  :: self
        integer, intent(out)               :: start
        logical, intent(out)               :: continue
        if(self%start>0) then
            start=self%start
            self%start=-1
            continue=.TRUE.
        else
            continue=.FALSE.
        end if
        return
    end subroutine

    pure function ArrayHopper_get_stride(self,idim) result(stride)
        class(ArrayHopper), intent(in) :: self
        integer,            intent(in) :: idim
        integer                        :: stride

        stride=self%stride(idim)
        return
    end function

    pure function ArrayHopper_get_n(self,idim)
        class(ArrayHopper), intent(in) :: self
        integer,            intent(in) :: idim
        integer                        :: ArrayHopper_get_n
        ArrayHopper_get_n=self%n+1
        return
    end function

    !%%%%%%%%%%%% InOutIntegrator1D %%%%%%%%%%%%%%%%%%

    function InOutIntegrator1D_init(gr) result(new)
        type(InOutIntegrator1D) :: new
        type(Grid1D)            :: gr

        new%n        = gr%get_nlip() - 1
        new%ncell    = gr%get_ncell()
        new%sz       = gr%get_shape()
        new%h        =>gr%get_cell_scales()
        new%ints_in  = gr%lip%inward_integrals()
        new%ints_out = gr%lip%outward_integrals()
    end function

    subroutine InOutIntegrator1D_destroy(self)
        class(InOutIntegrator1D) :: self

        deallocate(self%ints_in)
        deallocate(self%ints_out)
        nullify(self%h)
    end subroutine
    
    function InOutIntegrator1D_inwards(self, f_vals) result(res)
        class(InOutIntegrator1D), intent(in) :: self
        real(REAL64),             intent(in) :: f_vals(:)
        real(REAL64)                         :: res(self%sz)

        integer :: i, start

        res(self%sz)=0.d0 !Last element should be 0
        ! self%sz is the number of grid points in input/output grid
        start=self%sz
        ! go through all cells starting from the last
        do i= self%ncell, 1, -1
            ! evaluate the first grid point of the cell
            ! self%n is the number of LIPs in the input grid - 1
            start=start-self%n
            res(start:start+self%n-1) = &
                xmatmul( f_vals(start:start+self%n),self%ints_in ) * & 
                    self%h(i) + res(start+self%n)
        end do
    end function

    function InOutIntegrator1D_outwards(self, f_vals) result(res)
        class(InOutIntegrator1D) :: self
        real(REAL64)             :: f_vals(:)
        real(REAL64)             :: res(self%sz)

        integer :: i, start

        res(1)=0.d0 ! First element should be 0
        start=1
        ! go through all cell starting from first cell
        do i= 1, self%ncell
            res(start+1:start+self%n) = &
                xmatmul( f_vals(start:start+self%n),self%ints_out ) * &
                    self%h(i) + res(start)
            ! evaluate the first grid point of the next cell
            start=start+self%n
        end do
    end function

    !%%%%%%%%%%%%%%% Evaluator %%%%%%%%%%%%%%%%%%


#ifdef HAVE_CUDA

    !> Evaluate the derivative in the requested direction and store it to 'results'.
    !! To use this method, the 'input_cube' or 'bubbles' has to have been
    !! uploaded and set before hand. Additionally, the 'grid' and 'result_cuda_cube'
    !! have to be set.
    subroutine Evaluator_cuda_evaluate_grid_derivative(self, direction, results, set_to_zero, download, finite_diff_order)
        !> evaluator object
        class(Evaluator),          intent(inout)    :: self
        integer,                   intent(in)       :: direction
        real(REAL64),              intent(inout)    :: results(:, :, :)
        logical,       optional,   intent(in)       :: set_to_zero,  download
        logical                                     :: set_to_zero_, download_
        integer,                   intent(in)       :: finite_diff_order

        download_    = .TRUE.
        if (present(download))    download_ = download
        set_to_zero_ = .TRUE.
        if (present(set_to_zero)) set_to_zero_ = set_to_zero
        if (set_to_zero_) call self%result_cuda_cube%set_to_zero()
        call self%result_cuda_cube%set_host(results)

        if (direction == X_) then
            call Evaluator_evaluate_grid_x_gradients_cuda(self%cuda_interface, self%grid%get_cuda_interface(), &
                                                          self%result_cuda_cube%cuda_interface, finite_diff_order)
        else if (direction == Y_) then
            call Evaluator_evaluate_grid_y_gradients_cuda(self%cuda_interface, self%grid%get_cuda_interface(), &
                                                          self%result_cuda_cube%cuda_interface, finite_diff_order)
        else if (direction == Z_) then
            call Evaluator_evaluate_grid_z_gradients_cuda(self%cuda_interface, self%grid%get_cuda_interface(), &
                                                          self%result_cuda_cube%cuda_interface, finite_diff_order)
        end if
        if (download_) call self%result_cuda_cube%download()

    end subroutine

    !> Evaluate the values at the grid points and store them to 'results'.
    !! To use this method, the 'input_cube' or 'bubbles' has to have been
    !! uploaded and set before hand. Additionally, the 'grid' and 'result_cuda_cube'
    !! have to be set.
    subroutine Evaluator_cuda_evaluate_grid_without_gradients(self, results, set_to_zero, download)
        !> evaluator object
        class(Evaluator),          intent(inout)    :: self
        real(REAL64),              intent(inout)    :: results(:, :, :)
        logical,       optional,   intent(in)       :: set_to_zero,  download
        logical                                     :: set_to_zero_, download_

        download_    = .TRUE.
        if (present(download))    download_ = download
        set_to_zero_ = .TRUE.
        if (present(set_to_zero)) set_to_zero_ = set_to_zero

        if (set_to_zero_) call self%result_cuda_cube%set_to_zero()
        call self%result_cuda_cube%set_host(results)

        call Evaluator_evaluate_grid_without_gradients_cuda(self%cuda_interface, self%grid%get_cuda_interface(), &
                                                            self%result_cuda_cube%cuda_interface)
        if (download_) call self%result_cuda_cube%download()

    end subroutine

    !> Evaluate the derivative in the requested direction and store it to 'results'.
    !! To use this method, the 'input_cube' or 'bubbles' has to have been
    !! uploaded and set before hand. Additionally, the 'grid' and 'result_cuda_cube'
    !! have to be set.
    subroutine Evaluator_cuda_evaluate_points_derivative(self, direction, results, set_to_zero, download)
        !> evaluator object
        class(Evaluator),          intent(inout)    :: self
        integer,                   intent(in)       :: direction
        type(Points),              intent(inout)    :: results
        logical,       optional,   intent(in)       :: set_to_zero,  download
        logical                                     :: set_to_zero_, download_

        download_    = .TRUE.
        if (present(download))    download_ = download
        set_to_zero_ = .TRUE.
        if (present(set_to_zero)) set_to_zero_ = set_to_zero

        if (set_to_zero_) call self%result_points%cuda_set_to_zero()
        call results%set_cuda_interface(self%result_points%get_cuda_interface())

        if (direction == X_) then
            call Evaluator_evaluate_points_x_gradients_cuda(self%cuda_interface, results%get_cuda_interface())
        else if (direction == Y_) then
            call Evaluator_evaluate_points_y_gradients_cuda(self%cuda_interface, results%get_cuda_interface())
        else if (direction == Z_) then
            call Evaluator_evaluate_points_z_gradients_cuda(self%cuda_interface, results%get_cuda_interface())
        end if
        if (download_) call results%cuda_download()
        call results%dereference_cuda_interface()

    end subroutine

        !> Evaluate the derivative in the requested direction and store it to 'results'.
    !! To use this method, the 'input_cube' or 'bubbles' has to have been
    !! uploaded and set before hand. Additionally, the 'grid' and 'result_cuda_cube'
    !! have to be set.
    subroutine Evaluator_cuda_evaluate_points_without_gradients(self, results, set_to_zero, download)
        !> evaluator object
        class(Evaluator),          intent(inout)    :: self
        type(Points),              intent(inout)    :: results
        logical,       optional,   intent(in)       :: set_to_zero,  download
        logical                                     :: set_to_zero_, download_
        download_    = .TRUE.
        if (present(download))    download_ = download
        set_to_zero_ = .TRUE.
        if (present(set_to_zero)) set_to_zero_ = set_to_zero

        if (set_to_zero_) call self%result_points%cuda_set_to_zero()
        call results%set_cuda_interface(self%result_points%get_cuda_interface())

        call Evaluator_evaluate_points_without_gradients_cuda(self%cuda_interface, results%get_cuda_interface())
        
        if (download_) call results%cuda_download()
        call results%dereference_cuda_interface()

    end subroutine
#endif

    subroutine Evaluator_destroy_output_cubes(self)
        !> Evaluator object
        class(Evaluator), intent(inout)      :: self
#ifdef HAVE_CUDA
        if (allocated(self%result_cuda_cube)) then
            call self%result_cuda_cube%destroy()
            deallocate(self%result_cuda_cube)
        end if

        if (allocated(self%gradient_cuda_cube_x)) then
            call self%gradient_cuda_cube_x%destroy()
            deallocate(self%gradient_cuda_cube_x)
        end if 

        if (allocated(self%gradient_cuda_cube_y)) then
            call self%gradient_cuda_cube_y%destroy()
            deallocate(self%gradient_cuda_cube_y)
        end if 

        if (allocated(self%gradient_cuda_cube_z)) then
            call self%gradient_cuda_cube_z%destroy()
            deallocate(self%gradient_cuda_cube_z)
        end if 
#endif
    end subroutine

    subroutine Evaluator_set_output_grid(self, grid)
        !> CubeEvaluator object
        class(Evaluator), intent(inout)      :: self
        !> The output grid for the evaluator
        type(Grid3D), target, intent(in)     :: grid
        logical                              :: all_memory_at_all_devices, reallocate
#ifdef HAVE_CUDA
        reallocate = .not. allocated(self%result_cuda_cube) .or. .not. associated(self%grid)
#else
        reallocate = .not. associated(self%grid)
#endif
        if (.not. reallocate ) then
            if (.not. self%grid%is_equal(grid)) reallocate = .TRUE.
        end if

        if (reallocate) then
            self%grid => grid
#ifdef HAVE_CUDA
            call self%destroy_output_cubes()

            ! for cube grid, we must not have all memory at all devices .TRUE. for the output cubes
            all_memory_at_all_devices = .FALSE.
            self%result_cuda_cube = CudaCube(self%grid%get_shape(), all_memory_at_all_devices)

            if (self%high_memory_profile) then
                self%gradient_cuda_cube_x = CudaCube(self%grid%get_shape(), all_memory_at_all_devices)
                self%gradient_cuda_cube_y = CudaCube(self%grid%get_shape(), all_memory_at_all_devices)
                self%gradient_cuda_cube_z = CudaCube(self%grid%get_shape(), all_memory_at_all_devices)
            end if
#endif
        end if
    end subroutine

    subroutine Evaluator_destroy_output_points(self)
        !> Evaluator object
        class(Evaluator), intent(inout)      :: self
#ifdef HAVE_CUDA
        ! deallocate the memory if it is allocated
        if (allocated(self%result_points)) then
            call self%result_points%cuda_destroy()
            call self%result_points%destroy()
            deallocate(self%result_points)
        end if

        if (allocated(self%output_derivative_x_points)) then
            call self%output_derivative_x_points%cuda_destroy()
            call self%output_derivative_x_points%destroy()
            deallocate(self%output_derivative_x_points)
        end if

        if (allocated(self%output_derivative_y_points)) then
            call self%output_derivative_y_points%cuda_destroy()
            call self%output_derivative_y_points%destroy()
            deallocate(self%output_derivative_y_points)
        end if

        if (allocated(self%output_derivative_z_points)) then
            call self%output_derivative_z_points%cuda_destroy()
            call self%output_derivative_z_points%destroy()
            deallocate(self%output_derivative_z_points)
        end if 
#endif
    end subroutine
    
    
    subroutine Evaluator_destroy(self)
        !> Evaluator object
        class(Evaluator), intent(inout)      :: self

#ifdef HAVE_CUDA
        call Evaluator_destroy_cuda(self%cuda_interface)
#endif
        call self%destroy_output_points()
        call self%destroy_output_cubes()
    end subroutine

    subroutine Evaluator_set_output_points(self, result_points, force_reallocate)
        !> CubeEvaluator object
        class(Evaluator), intent(inout)      :: self
        !> The output grid for the evaluator
        type(Points),         intent(in)     :: result_points
        !> If the reallocation of the poins is forced
        logical,  optional,   intent(in)     :: force_reallocate
        logical                              :: all_memory_at_all_devices, reallocate
#ifdef HAVE_CUDA

        if (present(force_reallocate)) reallocate = force_reallocate
        if (.not. reallocate) reallocate = .not. allocated(self%result_points)
        if (.not. reallocate) then
            if (.not. self%result_points%point_coordinates%are_equal(result_points%point_coordinates)) &
                reallocate = .TRUE.
        end if
        if (reallocate) then

            call self%destroy_output_points()

            ! then do the reallocation with the new shape
            self%result_points = Points(result_points)
            call self%result_points%cuda_init()
            if (self%high_memory_profile) then
                self%output_derivative_x_points = Points(result_points)
                call self%output_derivative_x_points%cuda_init()
                self%output_derivative_y_points = Points(result_points)
                call self%output_derivative_y_points%cuda_init()
                self%output_derivative_z_points = Points(result_points)
                call self%output_derivative_z_points%cuda_init()
            end if
        end if
            
#endif
    end subroutine

   


    !> Extrapolate and replace the first values of 'values'
    !! using Lagrange interpolation polynomial of order 'order'.
    pure subroutine extrapolate_first_nlip7(order, grid_type, values)
        integer(INT32),      intent(in)    :: order
        integer(INT32),      intent(in)    :: grid_type
        real(REAL64), intent(inout) :: values(:)

        if (grid_type == 1) then ! equidistant
            if (order == 2) then
                values(1) = &
                       2.0d0 * values(2) &
                    -  1.0d0 * values(3) 
                            
            else if (order == 3) then
                values(1) = &
                       3.0d0 * values(2) &
                    -  3.0d0 * values(3) &
                    +  1.0d0 * values(4)
                    
            else if (order == 6) then
                values(1) = &
                       6.0d0 * values(2) &
                    - 15.0d0 * values(3) &
                    + 20.0d0 * values(4) &
                    - 15.0d0 * values(5) &
                    +  6.0d0 * values(6) &
                    -  1.0d0 * values(7)
            end if
       else if (grid_type == 2) then ! lobatto
            if (order == 2) then
                values(1) = &
                      1.46980575696  * values(2) &
                    - 0.469805756961 * values(3) 
                            
            else if (order == 3) then
                values(1) = &
                      1.77037274348  * values(2) &
                    - 1.00204109193  * values(3) &
                    + 0.231668348449 * values(4)
                    
            else if (order == 6) then
                values(1) = &
                      2.41108834235  * values(2) &
                    - 3.01108834235  * values(3) &
                    + 3.2d0          * values(4) &
                    - 3.01108834235  * values(5) &
                    + 2.41108834235  * values(6) &
                    - 1.d0           * values(7)
            end if
        end if
    end subroutine

end module

