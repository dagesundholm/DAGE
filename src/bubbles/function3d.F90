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
!> @file function3d.F90
!! Defines Function3D

!> Implements Function3D, the central object of libbubbles.
module Function3D_class
    use Function3D_types_m
    use Globals_m
    use xmatrix_m
    use io_m
    use Bubbles_class
    use Grid_class
    use MultiPoleTools_m
    use Evaluators_class
    use CartIter_class
    use Harmonic_class
    use RealSphericalHarmonics_class
    use ParallelInfo_class
    use mpi_m
    use timer_m
    use expo_m
    use CudaObject_class
    use Action_class
! debugging
    use MemoryLeakChecker_m
    use Points_class
#ifdef HAVE_CUDA
    use cuda_m
    use ISO_C_BINDING
#endif
    implicit none

    public :: Function3D
    public :: Function3DEvaluator
    public :: Projector3D
    public :: Operator3D
    public :: Operator3DArray
    public :: cube_project, contract_cube, contract_complex_cube, bubbles_multipliers
    public :: assignment(=)
    public :: operator(*)
    public :: operator(/)
    public :: operator(.apply.)
    public :: operator(.dot.)
   

    private

    character(len=3), parameter   :: BINARY_EXTENSION = 'fun'
    real(REAL64),     parameter   :: DEFAULT_STEP     = 0.3
    integer,          parameter   :: LABEL_LEN        = 20

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                   Function3D definition                                 %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    !> Bubbles representation of a three-dimensional scalar function.

    !> Three dimensional scalar functions are given as
    !!  \f[
    !!  f(x,y,z)=
    !!   \underbrace{f^\Delta(x,y,z)}_\text{cube} +
    !!   \underbrace{\sum_{Alm}{r^k_A f^A_{lm}(r_A)
    !!               Y^A_{lm} (\theta_A, \phi_A)}}_\text{bubbles}
    !!  \f]
    type, public :: Function3D
        private
        !> Descriptive label for a function.
        character(LABEL_LEN), public      :: label = "Function"
        !> Cube coefficients
        real(REAL64), pointer, public      :: cube(:,:,:) => null()
        !> The real underlying cube
        real(REAL64), allocatable, private :: cube_data(:, :, :)  
        !> Cube contaminants in the multiplication
        real(REAL64), allocatable, public  :: cube_contaminants(:,:)
        !> Cube contaminants in the multiplication
        real(REAL64), allocatable, public  :: bubbles_contaminants(:,:)
        !> Cube grid
        type(Grid3D), public, pointer      :: grid
        !> Bubbles
        type(Bubbles), public     :: bubbles
        !> Taylor series bubbles representing the off-diagonal terms
        class(Bubbles), public, allocatable       :: taylor_series_bubbles
        !> Parallel info object to handle parallelisation of the function3d
        class(ParallelInfo), public, pointer      :: parallelization_info
        !> General characteristics of the function. Used for function3d_product.
        integer, public                 :: type=F3D_TYPE_NONE
        !> Largest order of cutoff in taylor series
        integer, public                 :: taylor_order = 2
    contains
        ! Constructor subroutine
        procedure :: init_explicit     => Function3D_init_explicit_sub
        procedure :: init_copy         => Function3D_init_copy_sub
        procedure :: init_cube         => Function3D_init_cube
        ! Destructor
        ! TODO: final
        procedure :: destroy           => Function3D_destroy
        procedure :: destroy_cube      => Function3D_destroy_cube

        ! accessors
        procedure :: get_label         => Function3D_get_label
        procedure :: set_label         => Function3D_set_label
        procedure :: get_cube          => Function3D_get_cube
        procedure :: set_cube          => Function3D_set_cube
        procedure :: get_type          => Function3D_get_type
        procedure :: set_type          => Function3D_set_type

        ! IO
        procedure :: write             => Function3D_write
        procedure :: dump              => Function3D_dump
        procedure :: read              => Function3D_read
        procedure :: load              => Function3D_load
        procedure :: combine           => Function3D_combine
        procedure :: info              => Function3D_info

        ! Workers
        procedure :: copy_content      => Function3D_copy_content
        procedure :: evaluate          => Function3D_evaluate
        procedure :: integrate         => Function3D_integrate
        procedure :: inject_bubbles    => Function3D_inject_bubbles
        procedure :: project_onto      => Function3D_project_onto
        procedure :: cube_multipoles   => Function3D_cube_multipoles
        procedure :: bubble_multipoles => Function3D_bubble_multipoles
        procedure :: left_dot_product  => Function3D_dot_product
        procedure :: right_dot_product => Function3D_dot_product
        procedure :: multiply          => Function3D_multiply
        procedure :: multiply_sub      => Function3D_multiply_sub
        procedure :: multiply_bubbles  => Function3D_multiply_bubbles
        procedure :: get_nuclear_potential => Function3D_get_nuclear_potential
        procedure :: get_cube_contaminants => function3d_get_cube_contaminants
        procedure :: get_taylor_series_bubbles => Function3D_get_taylor_series_bubbles
        procedure :: precalculate_taylor_series_bubbles => Function3D_precalculate_taylor_series_bubbles
        procedure  :: dispose_extra_bubbles      => Function3D_dispose_extra_bubbles
        procedure  :: inject_extra_bubbles       => Function3D_inject_extra_bubbles
        procedure  :: inject_bubbles_to_cube     => Function3D_inject_bubbles_to_cube

        ! Communication
        procedure :: communicate_cube          => Function3D_communicate_cube
        procedure :: communicate_cube_borders  => Function3D_communicate_cube_borders

        ! xwh, (4*N), N: number of points
        ! |\nabla f|^2
        ! (\nabla f)_x
        ! (\nabla f)_y
        ! (\nabla f)_z
        procedure :: square_gradient_norm  => Function3D_square_gradient_norm
        procedure :: add_in_place          => Function3D_add_in_place
        procedure :: subtract_in_place     => Function3D_subtract_in_place
        procedure :: product_in_place_REAL64 => Function3D_product_in_place_REAL64
        procedure :: print_out_cube_at_point => Function3D_print_out_cube_at_point
        procedure :: print_out_centers => Function3D_print_out_centers

        ! Binary operators
        procedure, private :: Function3D_add
        generic, public :: operator(+) => Function3D_add

        procedure, private :: function3d_subtract
        generic, public :: operator(-) => Function3D_subtract

        procedure, private :: Function3D_product
        generic, public :: operator(*) => Function3D_product


    end type


!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                   Function3D pointer definition                         %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    type, public :: Function3DPointer
        type(Function3D), pointer  :: p
    end type

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                   CubeEvaluator definition                              %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    type, extends(Evaluator), public :: CubeEvaluator
        private
        type(Grid3D), pointer               :: input_grid
#ifdef HAVE_CUDA
        type(CudaCube), allocatable         :: input_cuda_cube
        logical                             :: input_cuda_cube_preinited
#else
        type(Interpolator)        :: interpolator
#endif
    contains
        ! Destructor
        procedure            :: destroy             => CubeEvaluator_destroy
        ! Evaluation functions
        procedure            :: evaluate_points            => CubeEvaluator_evaluate_points
        procedure            :: evaluate_points_from_cubes => CubeEvaluator_evaluate_points_from_cubes
        procedure            :: evaluate_divergence_points => CubeEvaluator_evaluate_divergence_points
        procedure            :: evaluate_divergence_grid   => CubeEvaluator_evaluate_divergence_grid
        procedure            :: evaluate_grid              => CubeEvaluator_evaluate_grid
        procedure            :: set_input_cube             => CubeEvaluator_set_input_cube
        procedure            :: destroy_input_cube         => CubeEvaluator_destroy_input_cube
#ifdef HAVE_CUDA
        procedure            :: cuda_init                  => CubeEvaluator_cuda_init
        procedure            :: cuda_destroy               => CubeEvaluator_cuda_destroy
#endif
    end type

    interface CubeEvaluator
        module procedure :: CubeEvaluator_init
    end interface

!--------------------------- CUDA/C++ interfaces ---------------------------------!
!  See implementations at evaluators.cu                                           !
!---------------------------------------------------------------------------------!

#ifdef HAVE_CUDA
    interface
        type(C_PTR) function CubeEvaluator_init_cuda(streamContainer) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: streamContainer
        end function
    end interface

    interface
        subroutine CubeEvaluator_set_input_grid_cuda(cube_evaluator, input_grid) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: cube_evaluator
            type(C_PTR), value    :: input_grid
        end subroutine
    end interface

    interface
        subroutine CubeEvaluator_set_input_cube_cuda(cube_evaluator, input_cube) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: cube_evaluator
            type(C_PTR), value    :: input_cube
        end subroutine
    end interface
#endif

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                   Function3D evaluator definition                       %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    type, extends(Evaluator), public :: Function3DEvaluator
        private
        type(CubeEvaluator)       :: cube_evaluator
        type(BubblesEvaluator)    :: bubbles_evaluator
    contains
        procedure            :: get_cube_evaluator    => Function3DEvaluator_get_cube_evaluator
        procedure            :: get_bubbles_evaluator => Function3DEvaluator_get_bubbles_evaluator
        procedure            :: cuda_init             => Function3DEvaluator_cuda_init
        procedure            :: cuda_destroy          => Function3DEvaluator_cuda_destroy
        ! Destructor
        procedure            :: destroy               => Function3DEvaluator_destroy
        ! Evaluation functions
        procedure            :: evaluate_grid         => Function3DEvaluator_evaluate_grid
        procedure            :: evaluate_points       => Function3DEvaluator_evaluate_points
        procedure            :: evaluate_divergence_as_Function3D &
                                                      => Function3DEvaluator_evaluate_divergence_as_Function3D
        procedure            :: evaluate_gradients_as_Function3Ds &
                                                      => Function3DEvaluator_evaluate_gradients_as_Function3Ds
        procedure            :: set_output_points     => Function3DEvaluator_set_output_points
        procedure            :: destroy_stored_objects=> Function3DEvaluator_destroy_stored_objects
    end type

    interface Function3DEvaluator
        module procedure :: Function3DEvaluator_init
    end interface



!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                   Operator3D definition                                 %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    type, extends(CudaObject), abstract :: Operator3D
        !> Input grid
        type(Grid3D),                 public, pointer :: gridin
        !> Output grid
        type(Grid3D),                 public, pointer :: gridout
        !> Cube transformation matrices
        type(REAL64_3D), allocatable, public          :: f(:)
        !> Cube transformation weights
        real(REAL64),    allocatable, public          :: w(:)
        !> Coda constant (coefficient of identity operator in expansion)
        real(REAL64), public                          :: coda = 0.0_REAL64
        !> Type of the resulting function (see \ref function3dtype_m
        !! "Function3D_types_m")
        integer, public                               :: result_type = F3D_TYPE_NONE
        !> The parallel info of the result objects, if not specified the parallel 
        !! info of the input object is used
        class(ParallelInfo),                 pointer  :: result_parallelization_info
        !> Determines if operator is a part of larger operator
        logical                                       :: suboperator = .FALSE.
#ifdef HAVE_CUDA
        type(CUDACube)                                :: cuda_fx, cuda_fy, cuda_fz, input_cuda_cube, &
                                                         output_cuda_cube, cuda_tmp1, cuda_tmp2
        type(CUDABlas)                                :: cuda_blas
        logical, public                               :: cuda_inited = .FALSE.
#endif
    contains
        procedure  :: operate_on                 => Operator3D_operate_on_Function3D
        procedure, private  :: operator_bubbles  => Operator3D_operator_bubbles
        procedure, private  :: operator_cube     => Operator3D_operator_cube
        procedure  :: get_dims                   => operator3D_get_dims
        procedure  :: get_result_type            => operator3D_get_result_type
        procedure  :: destroy                    => operator3D_destroy
        procedure  :: set_transformation_weights => Operator3D_set_transformation_weights 
        procedure :: transform_cube              => Operator3D_transform_cube
        procedure(bubble_operator),    deferred  :: transform_bubbles 
#ifdef HAVE_CUDA
        procedure :: transform_cuda_cube   => Operator3D_transform_cuda_cube
        procedure :: cuda_init             => Operator3D_cuda_init
        procedure :: cuda_destroy          => Operator3D_cuda_destroy
#endif

    end type

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                   Operator3D array definition                           %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!   This object is a dummy in order to be able to define arrays of         %
!   Operator3D objects                                                     %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    type Operator3DArray
        class(Operator3D), allocatable :: op
    end type

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                   Projector3D definition & interfaces                   %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    !> Projects the Function3D into the output grid. 
    type, extends(Operator3D) :: Projector3D
        !> The grids used in interpolation of the bubbles
        type(Grid1DPointer), allocatable               :: bubble_grids_in(:)
        !> The output grids used in interpolation of the bubbles
        type(Grid1DPointer), allocatable               :: bubble_grids_out(:)
        !> The interpolators used for bubbles
        type(SimpleInterpolator1D), allocatable        :: interpolators(:)
    contains
        procedure, private  :: operator_cube         => Projector3D_operator_cube
        procedure           :: destroy               => Projector3D_destroy
        procedure           :: transform_bubbles     => Projector3D_transform_bubbles
        procedure           :: transform_bubbles_sub => Projector3D_transform_bubbles_sub
    end type

    interface Projector3D
        module procedure :: Projector3D_init
    end interface

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                   Function3D interfaces                                 %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    interface assignment(=)
        module procedure :: function3d_assign_real64
        module procedure :: Function3D_assign_Function3D
!        module procedure :: Function3D_assign_poly
    end interface

    interface operator(*)
        module procedure :: function3d_times_real64
        module procedure :: REAL64_times_Function3D_poly
    end interface

    interface operator(/)
        module procedure :: function3d_divided_by_real64
    end interface

   interface operator(.dot.)
        module procedure :: Function3D_dot
   end interface 

    interface Function3D
        module procedure :: function3d_init_copy
        module procedure :: function3d_init_explicit
        module procedure :: function3d_init_file
    end interface

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                           Operator3D interfaces                         %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
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
#ifdef HAVE_CUDA
        recursive subroutine cuda_cube_operator(self, cubein, cubeout)
            import 
            class(Operator3D), intent(in)    :: self
            !> Input cube
            type(CUDACube),  intent(in)    :: cubein
            type(CUDACube),  intent(inout) :: cubeout
        end subroutine
#endif

        
    end interface

    interface operator(.apply.)
        module procedure :: Function3D_operate_with
        module procedure :: Operator3D_operate_on
    end interface
   
    integer, parameter :: IN_=1
    integer, parameter :: OUT_=2



contains

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                                Constructors                            %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    !> Creates a function from another function.
    subroutine Function3D_init_copy_sub(self, orig, copy_content, label, type, parallelization_info, lmax)
        class(Function3D),           intent(inout)    :: self
        class(Function3D),           intent(in)       :: orig
        logical,          optional                    :: copy_content
        character(len=*), optional, intent(in)        :: label
        integer,          optional, intent(in)        :: type
        class(ParallelInfo), optional, intent(in)     :: parallelization_info
        integer,          optional, intent(in)        :: lmax
        integer                                       :: gdims(3)
        ! use the parallelization_info of the orig, if 'parallellization_info' is
        ! not given. If there is no parallelization_info in the orig, use none.
        if (present(parallelization_info)) then
            call self%init_explicit(parallelization_info,&
                        taylor_series_order = orig%taylor_order)
        else 
            call self%init_explicit( orig%parallelization_info,  &
                        taylor_series_order = orig%taylor_order)
        end if

        if (orig%bubbles%get_nbub_global() > 0) then
            self%bubbles = Bubbles(orig%bubbles, lmax = lmax)
        end if 
        
        ! Do not copy content unless explicitly requested
        if(present(copy_content) .and. copy_content) then
            call self%copy_content(orig)
        end if

        if(present(label)) self%label = label

        if(present(type)) then
            self%type  = type
        else
            self%type = orig%type
        end if

    end subroutine

    !> Copy the content (bubbles and cube from 'input_function' to 'self')
    subroutine Function3D_copy_content(self, input_function)
        class(Function3D),           intent(inout)       :: self
        class(Function3D),           intent(in), target  :: input_function
        integer(INT32)                                   :: self_shape(3), input_function_shape(3)
        integer(INT32)                                   :: self_limits(2, 3), in_limits(2, 3) 
        class(ParallelInfo), pointer                     :: in_parallel_info

        ! do plain copy of the bubbles for now
        call self%bubbles%copy_content(input_function%bubbles)

        ! also for the taylor series bubbles
        if (allocated(input_function%taylor_series_bubbles)) then
            if (.not. allocated(self%taylor_series_bubbles)) then
                allocate(self%taylor_series_bubbles, &
                         source = Bubbles(input_function%taylor_series_bubbles, copy_content = .TRUE.))
            else
                call self%taylor_series_bubbles%copy_content(input_function%taylor_series_bubbles)
            end if
        end if
        
        self_shape = self%grid%get_shape()
        input_function_shape = input_function%grid%get_shape()
        if (self_shape(X_) == input_function_shape(X_) .and. &
            self_shape(Y_) == input_function_shape(Y_) .and. &
            self_shape(Z_) == input_function_shape(Z_)) then
            self%cube = input_function%cube
        else
            ! get the domain grid limits of both functions
            in_parallel_info => input_function%parallelization_info
            in_limits = in_parallel_info%get_multiplication_cell_limits()
            in_limits(1, :) = in_limits(1, :) - [1, 1, 1]
            in_limits = in_limits * (input_function%grid%get_nlip()-1)
            in_limits(1, :) = in_limits(1, :) + [1, 1, 1]

            self_limits = self%parallelization_info%get_multiplication_cell_limits() 
            self_limits(1, :) = self_limits(1, :) - [1, 1, 1]
            self_limits       = self_limits * (self%grid%get_nlip()-1)
            self_limits(1, :) = self_limits(1, :) + [1, 1, 1]

            ! and do a corresponding copy
            self%cube(self_limits(1, X_) : self_limits(2, X_)+1, &
                      self_limits(1, Y_) : self_limits(2, Y_)+1, &
                      self_limits(1, Z_) : self_limits(2, Z_)+1) = &
                input_function%cube(in_limits(1, X_) : in_limits(2, X_)+1, &
                                    in_limits(1, Y_) : in_limits(2, Y_)+1, &
                                    in_limits(1, Z_) : in_limits(2, Z_)+1)
        end if

            
        
    end subroutine

    !> Creates a function from another function.
    function Function3D_init_copy(orig, copy_content, label, type, parallelization_info, lmax) result(new)
        type(Function3D),           intent(in)       :: orig
        logical,          optional                   :: copy_content
        character(len=*), optional, intent(in)       :: label
        integer,          optional, intent(in)       :: type
        class(ParallelInfo), optional, intent(in)    :: parallelization_info
        integer,          optional, intent(in)       :: lmax

        type(Function3D)                 :: new
        call new%init_copy(orig, copy_content, label, type, parallelization_info, lmax)

    end function

    subroutine Function3D_init_cube(self)
        class(Function3D), target, intent(inout) :: self
        
#ifdef HAVE_CUDA
        self%cube => CudaCube_init_page_locked_cube(self%grid%get_shape())
#else
        allocate(self%cube_data(self%grid%axis(X_)%get_shape(),&
                                self%grid%axis(Y_)%get_shape(),&
                                self%grid%axis(Z_)%get_shape()))
        self%cube => self%cube_data
#endif
    end subroutine

    subroutine Function3D_destroy_cube(self)
        class(Function3D), target, intent(inout) :: self
        
#ifdef HAVE_CUDA
        if (associated(self%cube)) then
            call CudaCube_destroy_page_locked_cube(self%cube)
            nullify(self%cube)
        end if
#else
        if(allocated(self%cube_data)) then
            deallocate(self%cube_data)
            nullify(self%cube)
        end if
#endif
    end subroutine

    subroutine Function3D_init_explicit_sub(self, parallelization_info, bubs, label, type, &
                                      taylor_series_order)
        class(Function3D),             intent(inout) :: self
        class(ParallelInfo), target,   intent(in)    :: parallelization_info
        type(Bubbles),       optional, intent(in)    :: bubs
        character(len=*),    optional, intent(in)    :: label
        integer,             optional, intent(in)    :: type
        integer,             optional, intent(in)    :: taylor_series_order
        ! init parallel info
        self%parallelization_info => parallelization_info

        self%grid    => self%parallelization_info%get_grid()
        
        call self%init_cube()
        
        if (present(taylor_series_order)) then
            self%taylor_order = taylor_series_order
        end if
        if(present(bubs)) then 
            self%bubbles = Bubbles(bubs, copy_content = .TRUE.)
        end if
        if(present(label)) self%label = label
        if(present(type))  self%type  = type
    end subroutine 

    !> Creates a function from existing grids.
    function Function3D_init_explicit(parallelization_info, bubs, label, type, &
                                      taylor_series_order) result(new)
        class(ParallelInfo), target,   intent(in)   :: parallelization_info
        type(Bubbles),       optional, intent(in)   :: bubs
        character(len=*),    optional, intent(in)   :: label
        integer,             optional, intent(in)   :: type
        integer,             optional, intent(in)   :: taylor_series_order
        type(Function3D)                            :: new

        call new%init_explicit(parallelization_info, bubs, label, type, taylor_series_order)
    end function

    !> Creates a function from a file.

    !> Both binary and formatted files are supported.
    function Function3D_init_file(parallelization_info, folder, filename) result(new)
        class(ParallelInfo), target,   intent(in)   :: parallelization_info
        character(len=*),              intent(in)   :: folder
        character(len=*),              intent(in)   :: filename

        type(Function3D)                            :: new

        character(len(BINARY_EXTENSION))            :: fext

        call pdebug("Function3D_init_file()", 1)
        fext = filext(filename)

        select case(fext)
        case(BINARY_EXTENSION)
            call new%load(trim(folder), trim(filename))
        case default
            call new%read(trim(filename))
        end select
    end function

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                    Function 3D Destructor                              %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    !> Destructor
    subroutine Function3D_destroy(self)
        class(Function3D), intent(inout) :: self

        !call pdebug(trim('Destroying function '//self%label), 1)
        if (allocated(self%cube_contaminants)) deallocate(self%cube_contaminants)
        if (allocated(self%bubbles_contaminants)) deallocate(self%bubbles_contaminants)
        call self%destroy_cube()
        nullify(self%parallelization_info)
        nullify(self%grid)
        call self%bubbles%destroy()
        if(allocated(self%taylor_series_bubbles)) then
            call self%taylor_series_bubbles%destroy()
            deallocate(self%taylor_series_bubbles)
        end if

    end subroutine

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%          Function 3D Accessors (getters and setters)                   %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    !> Get the label.
    function Function3D_get_label(self) result(label)
        class(Function3D) :: self
        character(len=len(trim(self%label))) :: label
        label = trim(self%label)
    end function

    !> Set the label
    subroutine Function3D_set_label(self, label)
        class(Function3D) :: self
        character(len=*), intent(in) :: label
        write(self%label, "(A)") label
    end subroutine

    !> Get the cube coefficients
    function Function3D_get_cube(self) result(cube)
        class(Function3D), target         :: self
        real(REAL64), pointer               :: cube(:,:,:)
        cube => self%cube
    end function

    !> Get the cube coefficients
    subroutine Function3D_set_cube(self, cube)
        class(Function3D), target           :: self
        real(REAL64), intent(in)            :: cube(:,:,:)
        self%cube = cube
    end subroutine

    !> Get function3d::type
    function Function3D_get_type(self) result(type)
        class(Function3D), intent(in) :: self
        integer                       :: type

        type=self%type
    end function

    !> Set function3d::type
    subroutine Function3D_set_type(self,type)
        class(Function3D), intent(inout) :: self
        integer, intent(in)              :: type

        self%type=type
    end subroutine

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                             Function3D IO                              %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    !> Write a function to a formatted text file.
    !!
    !! The file format is determined from the extension of `filename`.
    !!
    !! Here is the format of the binary file (sizes in bytes)
    !!
    !! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    !! - CUBESTART                             8
    !! - PBCSTART                              8
    !! - BUBBLESTART                           8
    !! - LABEL :: CHARACTER(20)               20
    !! - TYPE  :: INTEGER                      4
    !! - GRID
    !!     - NLIP :: INTEGER                   4
    !!     - GRID1D (x3)
    !!         - NCELL :: INTEGER              4
    !!         - NLIP :: INTEGER               4
    !!         - START :: REAL                 8
    !!         - CELLH :: REAL(:) !step        8 x NCELL
    !!     ////// SKIPPED BY NOW //////
    !!     - PBC
    !!         - PBCSTRING?
    !!     ////////////////////////////
    !! - CUBE
    !!     - NDIM :: INTEGER(3)                4 x 3
    !!     - CUBE :: REAL(:,:,:)               8 x NDIM(1) x NDIM(2) x NDIM(3)
    !! - BUBBLESET
    !!     - NBUB :: INTEGER                   4
    !!     - LMAX :: INTEGER                   4
    !!     - K    :: INTEGER                   4
    !!     - BUBBLE (x NBUB)
    !!         - CRD :: REAL(3)                8 x 3
    !!         - Z :: REAL                     8
    !!         - GRID1D
    !!             - NCELL :: INTEGER          4
    !!             - NLIP :: INTEGER           4
    !!             - CELLH :: REAL(:) !step    8 x NCELL
    !!         - F :: REAL(:) (x (LMAX+1)**2)  8 x (NCELL x (NLIP -1)+1)x (LMAX+1)**2)
    !! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    subroutine Function3D_write(self, filename, step, write_bubbles)
        class(Function3D)               :: self
        !> The name of the file to write to
        character(len=*), intent(in)        :: filename
        !> Optional resolution of output data, default is 0.3 a.u.
        real(REAL64), intent(in), optional  :: step
        !> Controls if the bubbles should be injected in the output
        logical, intent(in), optional :: write_bubbles

        type(Function3D), target        :: equidistant_self

        type(Grid3D)                        :: eq_grid

        type(FileFormat)                    :: file_format
        real(REAL64), pointer               :: cube(:,:,:)

        

        ! Project to an equidistant grid
        if (present(step)) then
            eq_grid = Grid3D(self%grid%get_range(),&
                             self%grid%get_nlip(),&
                             step)
        else
            eq_grid = Grid3D(self%grid%get_range(),&
                             self%grid%get_nlip(),&
                             DEFAULT_STEP)
        endif

        equidistant_self=self%project_onto(eq_grid)

        cube => equidistant_self%cube

        ! Inject bubbles if desired
        if (present(write_bubbles) .and. write_bubbles) then
            cube = cube + equidistant_self%inject_bubbles()
        endif

        file_format = FileFormatInit(cube, eq_grid%get_shape(),&
                                     eq_grid%get_range())

        call pinfo("Writing "//self%get_label()//" to `"//filename//"'.")

        if (.not. write_file(filename, file_format)) then
            call perror("Failed to write `"//filename//"'!")
        endif

        call equidistant_self%destroy()

    end subroutine

    !> Saves a function in binary format on disk.
    !!
    !! A file called `funcname.fun` is (re)created for writing in folder 'folder'.
    subroutine Function3D_dump(self, folder, funcname, mode_)
        class(Function3D)                       :: self
        !> Folder for the binary file          
        character(len=*),  intent(in)           :: folder
        !> Basename for the binary file
        character(len=*),  intent(in)           :: funcname
        !> Variable determining which parts of function are stored
        !! 1: all (default), 2: bubbles, 3: cube
        integer, optional, intent(in)           :: mode_ 

        integer(INT64)                          :: cubestart
        integer(INT64)                          :: bubblestart
        integer(INT64)                          :: grid_shape(3)
        integer                                 :: i
        character(len=:), allocatable           :: filename
        real(REAL64)                            :: temp
        integer                                 :: mode

        mode = 1
        if (present(mode_)) mode = mode_ 
        
        if (mode /= DO_NOT_STORE_RESULT_FUNCTIONS) then


            if (filext(funcname)/=BINARY_EXTENSION) &
                        filename = trim(funcname)//'.'//BINARY_EXTENSION

            call pdebug("Dumping a function to `"//trim(folder)//'/'//trim(filename)//"'.", 1)

            open(unit=WRITE_FD, file=trim(folder)//'/'//trim(filename), access='stream')

            write(WRITE_FD, pos=25) self%label

            write(WRITE_FD) self%type

            cubestart = -1
            if (mode /= STORE_ONLY_BUBBLES) then
                ! The cube
                inquire(WRITE_FD, pos=cubestart)
                grid_shape = self%grid%get_shape()

                do i = 1, 3
                    write(WRITE_FD) grid_shape(i)
                end do

                write(WRITE_FD) self%cube
            end if

            bubblestart = -1
            if (mode /= STORE_ONLY_CUBES) then
                inquire(WRITE_FD, pos=bubblestart)

                ! The bubbles
                call self%bubbles%write(WRITE_FD)
            end if
            write(WRITE_FD, pos=1) cubestart
            write(WRITE_FD, pos=17) bubblestart
            deallocate(filename)
            close(unit=WRITE_FD)
        end if
    end subroutine

    !> Loads a function from a binary file. The input Function3D object 'self'
    !! must be preinited according to a fitting mould.
    !!
    subroutine Function3D_load(self, folder, filename)
        class(Function3D),           intent(inout) :: self
        character(len=*),            intent(in)    :: folder
        character(len=*),            intent(in)    :: filename
        integer(INT64)                             :: cubestart
        integer(INT64)                             :: pbcstart
        integer(INT64)                             :: bubblestart
        integer(INT64)                             :: ndim(3)
        logical                                    :: file_exists
        real(REAL64)                            :: temp

        call pdebug("Loading a function from `"//trim(folder)//"/"//trim(filename), 1)

        ! determine if the file we want to load exists, the result is stored to
        ! variable 'file_exists'
        inquire(file=trim(folder)//"/"//trim(filename), exist=file_exists)
        
        if (file_exists) then
            open(unit=READ_FD, file=trim(folder)//"/"//trim(filename), access='stream')

            read(READ_FD) cubestart
            read(READ_FD) pbcstart
            read(READ_FD) bubblestart

            read(READ_FD) self%label
            read(READ_FD) self%type

            ! Read in the cube
            if (cubestart /= -1) then
                read(READ_FD, pos=cubestart) ndim
                read(READ_FD) self%cube
            end if

            ! Read in the Bubbles
            if (bubblestart /= -1) then
                
                ! check if the bubbles is inited, if this is the case, do only
                ! a shallow load of the bubbles (no initialization and reading of grid)
                if (allocated(self%bubbles%bf)) then
                    call self%bubbles%read(READ_FD, bubblestart)
                else
                    call self%bubbles%read_and_init(READ_FD, bubblestart)
                end if
            end if
            close(unit=READ_FD)
        else
            call perror("Cannot load `"//trim(folder)//"/"//trim(filename)//"': File not found.")
            stop
        endif
        
    end subroutine

    !> Reads a function from a formatted text file
    !!
    !! The file format is determined from the extension of `filename`.
    subroutine Function3D_read(self, filename)
        class(Function3D)           :: self
        character(len=*), intent(in)    :: filename

        type(FileFormat), allocatable   :: file_format

        call pdebug("Reading `"//filename//"'.", 1)

        if (read_file(filename, file_format)) then

            ! TODO: it would be cool to be able to read a step from an
            ! input file.

            ! Unpack read data to Function3D attributes
            !self%grid = Grid3D(file_format%ranges, file_format%gdims)
            self%cube = file_format%cube
        else
            call perror("Reading "//filename//"failed.")
            stop
        endif
    end subroutine

    !>Read and combine cube type and bubles type files into Function3D
    subroutine Function3D_combine(self,bublib_name,cube_name)
        class(Function3D) :: self
        character(len=*), intent(in) :: bublib_name,cube_name
        real(REAL64), pointer :: cubetmp(:,:,:)

        call self%read(cube_name)

        open(unit=READ_FD,file=bublib_name,access='stream')

        call self%bubbles%read(READ_FD)

        close(READ_FD)

        cubetmp=self%inject_bubbles()
        self%cube=self%cube-cubetmp
        deallocate(cubetmp)
    end subroutine

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                          Function3D Operations                         %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    !> Sum self and f2 together and store the result to f1. Same cube grid is assumed!
    subroutine Function3D_add_in_place(self, f2, f2_factor)
        class(Function3D),      intent(inout)  :: self
        class(Function3D),      intent(in)     :: f2
        real(REAL64), optional, intent(in)     :: f2_factor 
        type(Bubbles)                          :: temp_bubbles, temp_bubbles2
        real(REAL64)                           :: f2_factor_
        f2_factor_ = 1.0d0
        if (present(f2_factor)) f2_factor_ = f2_factor 

        self%cube = self%cube + f2_factor_ * f2%cube
        if (self%bubbles%get_nbub_global() > 0 .or. f2%bubbles%get_nbub_global() > 0) then
            temp_bubbles2 = (f2_factor_ * f2%bubbles)
            temp_bubbles = self%bubbles + temp_bubbles2
            call self%bubbles%destroy()
            self%bubbles = Bubbles(temp_bubbles, copy_content = .TRUE.)
            call temp_bubbles%destroy()
            call temp_bubbles2%destroy()
        end if
    end subroutine

    !> Sum self and f2 together and store the result to f1. Multiply the f2 
    !! with factor before subtraction. Same cube grid is assumed!
    subroutine Function3D_subtract_in_place(self, f2, f2_factor)
        class(Function3D),      intent(inout)  :: self
        class(Function3D),      intent(in)     :: f2
        real(REAL64), optional, intent(in)     :: f2_factor 
        type(Bubbles)                          :: temp_bubbles, temp_bubbles2
        real(REAL64)                           :: f2_factor_
        f2_factor_ = 1.0d0
        if (present(f2_factor)) f2_factor_ = f2_factor 

        self%cube = self%cube - f2_factor_ * f2%cube
        if (self%bubbles%get_nbub_global() > 0 .or. f2%bubbles%get_nbub_global() > 0) then
            temp_bubbles2 = ((-1.0d0) * f2_factor_ * f2%bubbles)
            temp_bubbles = self%bubbles + temp_bubbles2
            call self%bubbles%destroy()
            self%bubbles = Bubbles(temp_bubbles, copy_content = .TRUE.)
            call temp_bubbles%destroy()
            call temp_bubbles2%destroy()
        end if
    end subroutine

    !> Polymorphic method for real64 multiplication
    subroutine Function3D_product_in_place_REAL64(self, factor)
        class(Function3D), intent(inout)  :: self
        real(REAL64),      intent(in)     :: factor
        integer                           :: ibub
        
        self%cube = self%cube * factor
        if (self%bubbles%get_nbub_global() > 0 ) then
            do ibub = 1, self%bubbles%get_nbub()
                self%bubbles%bf(ibub)%p = factor * self%bubbles%bf(ibub)%p
            end do
        end if

    end subroutine

    !> Sum f1 and f2 using the grid of f1.
    function Function3D_add(f1, f2) result(fout)
        class(Function3D), intent(in)    :: f1
        class(Function3D), intent(in)    :: f2
        type(Function3D) :: fout

        !call pdebug("Function3D_sum()", 1)
        ! Clones f2 to f and projects its cube using the grid of f1
        fout=f2%project_onto(f1%grid)
        fout%parallelization_info => f1%parallelization_info
        fout%cube    = fout%cube    + f1%cube
        if (fout%bubbles%get_nbub_global() > 0 .or. f2%bubbles%get_nbub_global() > 0) then
            fout%bubbles = fout%bubbles + f1%bubbles
        end if

        

        if (f1%type /= f2%type) call pwarn(&
        "Adding Function3D's of different types. &
        &Make sure that's not a problem!")
    end function

    !> Subtract f1 and f2 using the grid of f1.
    function Function3D_subtract(f1, f2) result(fout)
        class(Function3D), intent(in)    :: f1
        class(Function3D), intent(in)    :: f2
        type(Function3D)                 :: temp
        type(Function3D) :: fout
        temp = ((-1.d0)*f2)
        fout= f1 + temp
        call temp%destroy()
    end function

    function Function3D_product(self, f2) result(new)
        class(Function3D), intent(in)    :: self
        class(Function3D), intent(in)    :: f2

        type(Function3D)                 :: new
        new = self%multiply(f2)
    end function

    !> Multiply the cubes of f1 and f2 using the grid of f1
    function Function3D_multiply(self, f2, cell_limits, part_of_dot_product, result_bubbles) result(new)
        class(Function3D), intent(in), target :: self
        class(Function3D), intent(in), target :: f2

        !> Limits in the area in which the operator is applied 
        !! if the f2 contains the entire function
        integer, intent(in), optional         :: cell_limits(2, 3)
        !> If this is set to .true. only cube part's multipole moments are evaluated
        logical, intent(in), optional         :: part_of_dot_product
        type(Bubbles), intent(in), optional, target :: result_bubbles
        type(Function3D), allocatable         :: new

        call self%multiply_sub(f2, new, cell_limits, part_of_dot_product, result_bubbles)
    end function

    !> Multiply the cubes of f1 and f2 using the grid of f1
    subroutine Function3D_multiply_sub(self, f2, new, cell_limits, part_of_dot_product, result_bubbles) 
        class(Function3D), intent(in),    target       :: self
        class(Function3D), intent(in),    target       :: f2
        type(Function3D),  intent(inout), allocatable  :: new

        !> Limits in the area in which the operator is applied 
        !! if the f2 contains the entire function
        integer, intent(in), optional                  :: cell_limits(2, 3)
        !> If this is set to .true., the cube is not downloaded after the multiplication
        logical, intent(in), optional                  :: part_of_dot_product
        type(Bubbles), intent(in), optional, target    :: result_bubbles

 
        integer                                        :: cube_limits(2, 3), multiplication_cell_limits(2, 3)

        ! result Function3D
        real(REAL64), allocatable                      :: f_inject(:,:,:), self_inject(:,:,:), new_inject(:, :, :)
        real(REAL64), pointer                          :: f_cube_pointer(:, :, :)
        type(Bubbles)                                  :: self_bubbles, f2_bubbles
        type(BubblesMultiplier), pointer               :: bubbles_multiplier
        integer                                        :: cube_offset
        logical                                        :: copy_bubbles, dot_product_process
#ifdef HAVE_CUDA
        type(C_PTR)                                    :: function3d_multiplier
        type(C_PTR)                                    :: cuda_bubbles1, cuda_bubbles2, cuda_result_bubbles
#endif
#ifdef HAVE_NVTX   
        call start_nvtx_timing("Function3D multiply. ")   
#endif    
        call bigben%split("Function3D product")
        dot_product_process = .FALSE.
        if (present(part_of_dot_product)) dot_product_process = part_of_dot_product
        
        if (.not. allocated(new)) then
            allocate(Function3D :: new)

            if (self%grid%is_equal(f2%grid) .or. self%grid%is_similar(f2%grid)) then
                f_cube_pointer => f2%get_cube()
                call new%init_explicit(self%parallelization_info, type = self%type, &
                        taylor_series_order = self%taylor_order)
            else 
                call bigben%split("Projecting f2 onto f1's grid")

                ! Project the 'f2' to the grid of 'self'
                new = f2%project_onto(self%grid, cell_limits = cell_limits, &
                        only_cube = .TRUE.)
                f_cube_pointer => new%get_cube() 
            
                new%parallelization_info => self%parallelization_info
                new%taylor_order = self%taylor_order
                call bigben%stop()
            end if
        else 
            ! set the pointer
            f_cube_pointer => f2%get_cube()

            ! we also destroy the taylor series bubbles just in case
            if (allocated(new%taylor_series_bubbles)) then
                call new%taylor_series_bubbles%destroy()
                deallocate(new%taylor_series_bubbles)
            end if
        end if
        ! Bubbles multiplication algorithm
        ! Designed for producing an approximation of the energy density
        if (f2%bubbles%get_nbub_global() > 0 .or. self%bubbles%get_nbub_global() > 0) then
#ifdef HAVE_NVTX       
            call start_nvtx_timing("Bubbles multiplication. ")  
#endif 
            call bigben%split("Bubbles multiplication. ")
            
            ! Throw an error if the types were unspecified
            if( self%type==F3D_TYPE_NONE .or. f2%type==F3D_TYPE_NONE ) then
                write(ppbuf, '("Cannot multiply Function3D objects of &
                              &unspecified type (",i1," and ",i1,").")')&
                              self%type, f2%type
                call perror(ppbuf)
                stop
            else if (f2%bubbles%get_nbub_global() > 0 .and. self%bubbles%get_nbub_global() > 0) then
                call bigben%split("Multiply bubbles")
                if (present(result_bubbles)) then
                    call new%bubbles%copy_content(result_bubbles)
                else
                    call self%multiply_bubbles(f2, new%bubbles)
                end if              
                call bigben%stop()

                ! Set output function type
                if (  self%type==F3D_TYPE_CUSP .or. f2%type==F3D_TYPE_CUSP ) then
                    new%type=F3D_TYPE_CUSP
                end if
            end if
            call bigben%stop()
#ifdef HAVE_CUDA
            call CUDASync_all()
#ifdef HAVE_NVTX  
            call stop_nvtx_timing()
            call start_nvtx_timing("Function3D bubble-uploads. ")
#endif
            call bigben%split("Cuda multiply prepare")

            ! select only the part of the grid that needs to be multiplied by this processor
            multiplication_cell_limits = self%parallelization_info%get_multiplication_cell_limits()
            cube_limits = self%parallelization_info%get_cube_ranges(multiplication_cell_limits)
            cube_offset =     (cube_limits(1, Z_)-1) * self%grid%axis(X_)%get_shape() * self%grid%axis(Y_)%get_shape() &
                            + (cube_limits(1, Y_)-1) * self%grid%axis(X_)%get_shape() &
                            + (cube_limits(1, X_)-1)
            
            ! upload the bubbles to gpus and calculate cf (gpu stuff)
            ! NOTE: injection methods contain all the stuff to make sure 
            !       that bubbles are uploaded properly before any injection occurs.
            !       Thus, there is no need for extra synchonization.
            bubbles_multiplier => bubbles_multipliers%get_multiplier(self%bubbles, f2%bubbles)
            cuda_result_bubbles = new%bubbles%cuda_get_sub_bubbles(bubbles_multiplier%result_bubbles)
            call new%bubbles%cuda_upload_all(cuda_result_bubbles)
            if (self%bubbles%get_lmax() <= f2%bubbles%get_lmax() ) then
                cuda_bubbles1 = self%bubbles%cuda_get_sub_bubbles(bubbles_multiplier%bubbles1)
                call self%bubbles%cuda_upload_all(cuda_bubbles1)
                cuda_bubbles2 = f2%bubbles%cuda_get_sub_bubbles(bubbles_multiplier%bubbles2)
                call f2%bubbles%cuda_upload_all(cuda_bubbles2)
            else
                cuda_bubbles2 = self%bubbles%cuda_get_sub_bubbles(bubbles_multiplier%bubbles2)
                call self%bubbles%cuda_upload_all(cuda_bubbles2)
                cuda_bubbles1 = f2%bubbles%cuda_get_sub_bubbles(bubbles_multiplier%bubbles1)
                call f2%bubbles%cuda_upload_all(cuda_bubbles1)
            end if
            
#ifdef HAVE_NVTX 
            call stop_nvtx_timing()      
            call start_nvtx_timing("Function3DMultiplier multiply. ")
#endif
            ! upload the self%cube and new%cube to gpu
            ! NOTE: we are passing the shape of the original array 
            call Function3DMultiplier_upload_cubes_cuda(self%parallelization_info%function3d_multiplier, &
                     self%cube, cube_offset, self%grid%get_shape(), self%bubbles%get_lmax(), &
                     f_cube_pointer,  cube_offset, self%grid%get_shape(), f2%bubbles%get_lmax())
            call bigben%stop()
            call bigben%split("Cuda multiply")

            ! do the product
            call Function3DMultiplier_multiply_cuda(self%parallelization_info%function3d_multiplier, &
                                                    cuda_bubbles1, cuda_bubbles2, cuda_result_bubbles)


            ! recalculate the multiplication cell limits and offsets for result cube, as it may have 
            ! different gridding with the input functions
            multiplication_cell_limits = new%parallelization_info%get_multiplication_cell_limits()
            cube_limits = new%parallelization_info%get_cube_ranges(multiplication_cell_limits)
            cube_offset =     (cube_limits(1, Z_)-1) * new%grid%axis(X_)%get_shape() * new%grid%axis(Y_)%get_shape() &
                            + (cube_limits(1, Y_)-1) * new%grid%axis(X_)%get_shape() &
                            + (cube_limits(1, X_)-1)
            
  
            ! download the result to cpu, specifically to new%cube
            !if (.not. dot_product_process) then
            call Function3DMultiplier_set_host_result_cube_cuda(self%parallelization_info%function3d_multiplier, &
                    new%cube, cube_offset, new%grid%get_shape())

            call Function3DMultiplier_download_result_cuda(self%parallelization_info%function3d_multiplier)
            !end if

            call CUDASync_all()
#ifdef HAVE_NVTX   
            call stop_nvtx_timing()
#endif
            call bigben%stop()
            

            !if (.not. dot_product_process) then
            call bigben%split("Communicate cube borders")
            ! because we are multiplying only a part of the function, we must communicate afterwards
            call new%communicate_cube_borders(reversed_order = .TRUE.)
            call bigben%stop()

            call Bubbles_destroy_cuda(cuda_bubbles1)
            call Bubbles_destroy_cuda(cuda_bubbles2)
            call Bubbles_destroy_cuda(cuda_result_bubbles)
            !end if
#else      
            call bigben%split("Inject input bubbles")
            self_inject = self%inject_bubbles()
            
            ! f_inject is the injection of bubbles projected on
            ! self's grid
            new%cube(:, :, :) = f_cube_pointer(:, :, :)
            call new%inject_bubbles_to_cube(f2%bubbles, factor = 1.0d0)


            new%cube = new%cube(:, :, :) * (self%cube(:, :, :)+self_inject(:, :, :))
            call bigben%stop()

            ! deallocate the temporary variables used in inject
            deallocate(self_inject)

            ! Figure out how to multiply the bubbles based on the function types

            call bigben%split("Extract result bubbles from cube")

            call new%inject_bubbles_to_cube(new%bubbles, factor = -1.0d0)
            call bigben%stop()
            
#endif
            
        else
            print *, "shapes", shape(f_cube_pointer), shape(self%cube), shape(new%cube)
            new%cube = f_cube_pointer * self%cube
        endif

#ifdef HAVE_CUDA_PROFILING
        call stop_nvtx_timing()
#endif
        call bigben%stop()
    end subroutine

    subroutine Function3D_precalculate_taylor_series_bubbles(self, taylor_order, ignore_cube, &
                                                             ignore_bubbles, non_overlapping)
        class(Function3D), intent(inout)  :: self
        integer, optional, intent(in)  :: taylor_order
        logical, optional, intent(in)     :: ignore_cube
        logical, optional, intent(in)     :: ignore_bubbles
        logical, optional, intent(in)      :: non_overlapping
        type(Bubbles)                     :: taylor_series_bubbles
        taylor_series_bubbles = self%get_taylor_series_bubbles(taylor_order = taylor_order, &
                                                               ignore_cube = ignore_cube, &
                                                               ignore_bubbles = ignore_bubbles, &
                                                               non_overlapping = non_overlapping)
        if(allocated(self%taylor_series_bubbles)) then
            call self%taylor_series_bubbles%destroy()
            call self%taylor_series_bubbles%init_copy(taylor_series_bubbles, copy_content = .TRUE.)
        else
            allocate(Bubbles::self%taylor_series_bubbles)
            call self%taylor_series_bubbles%init_copy(taylor_series_bubbles, copy_content = .TRUE.)
        end if
        call taylor_series_bubbles%destroy()
    end subroutine 

    function Function3D_get_taylor_series_bubbles(self, lmax, taylor_order, &
                 ignore_cube, ignore_bubbles, non_overlapping) &
                 result(taylor_series_bubbles)
        class(Function3D), intent(in)  :: self
        integer, optional, intent(in)  :: lmax
        integer, optional, intent(in)  :: taylor_order
        logical, optional, intent(in)  :: ignore_cube
        logical, optional, intent(in)  :: ignore_bubbles
        logical, optional, intent(in)  :: non_overlapping
        type(Bubbles)                  :: taylor_series_bubbles
        real(REAL64), allocatable      :: bubbles_contaminants(:, :), cube_contaminants(:, :)
        integer                        :: maximum_l, taylor_order_
        logical                        :: ignore_cube_, ignore_bubbles_, non_overlapping_
#ifdef HAVE_CUDA_PROFILING
        call start_nvtx_timing("Get Taylor Series Bubbles")
#endif
        if (present (taylor_order)) then
            taylor_order_ = taylor_order
        else
            taylor_order_ = self%taylor_order
        end if

        if (present (ignore_cube)) then
            ignore_cube_ = ignore_cube
        else
            ignore_cube_ = .FALSE.
        end if
        
        if (present (non_overlapping)) then
            non_overlapping_ = non_overlapping
        else
            non_overlapping_ = .FALSE.
        end if
        
        if (present (ignore_bubbles)) then
            ignore_bubbles_ = ignore_bubbles
        else
            ignore_bubbles_ = .FALSE.
        end if
        
        if (present (lmax)) then
            maximum_l = lmax
        else
            maximum_l = self%bubbles%get_lmax()
        end if
        call bigben%split("Get taylor series bubbles")
        
        if (allocated(self%cube_contaminants)) then
            cube_contaminants = self%cube_contaminants
        else
            !if (self%type /= F3D_TYPE_NUCP) then
                cube_contaminants = self%get_cube_contaminants(taylor_order_)
            !end if
        end if
        
        

        if (allocated(self%bubbles_contaminants)) then
            bubbles_contaminants = self%bubbles_contaminants
        else
            bubbles_contaminants = self%bubbles%get_contaminants(taylor_order_)
        end if

        !if (self%type == F3D_TYPE_NUCP) then
        !    allocate(cube_contaminants((taylor_order_+1)*(taylor_order_+2)*&
        !                          (taylor_order_+3)/6, self%bubbles%get_nbub_global()), source = 0.0d0)
        !end if
        
        if (ignore_cube_)    cube_contaminants = 0.0d0
        if (ignore_bubbles_) bubbles_contaminants = 0.0d0
        
        taylor_series_bubbles = self%bubbles%make_taylor(bubbles_contaminants, cube_contaminants, &
                                                         taylor_order_, non_overlapping_)  
        deallocate(cube_contaminants)
        deallocate(bubbles_contaminants)
        call bigben%stop()
#ifdef HAVE_CUDA_PROFILING
        call stop_nvtx_timing()
#endif
        
    end function

    subroutine Function3D_multiply_bubbles(self, f2, result_bubbles)
        class(Function3D), intent(in), target   :: self
        class(Function3D), intent(in), target   :: f2
        class(Bubbles), intent(inout), target   :: result_bubbles
        !> The result object that will contain the multiple of self and f2's
        !! bubbles
        type(Bubbles),  target                  :: taylor_series_bubbles1, taylor_series_bubbles2
        class(Bubbles), pointer                 :: taylor_pointer2, taylor_pointer1
        type(Bubbles)                           :: bubbles1, bubbles2

        !call pdebug("Multiply bubbles", 1)
        if (self%bubbles%get_nbub_global() > 0 .and. f2%bubbles%get_nbub_global() > 0) then
            

#ifdef HAVE_CUDA_PROFILING
            call start_nvtx_timing("Bubbles Multiplication Fortran Preparation. ")
#endif
            ! If self is cuspy, do Taylor expansion for f2
            if ( self%type==F3D_TYPE_CUSP .or. self%type == F3D_TYPE_NUCP) then
                ! if not pre-calculated, calculate taylor series bubbles for f2
                if (allocated(f2%taylor_series_bubbles)) then
                    taylor_pointer2 => f2%taylor_series_bubbles
                else
                    taylor_series_bubbles2 = f2%get_taylor_series_bubbles()
                    taylor_pointer2 => taylor_series_bubbles2
                end if
            !    bubbles2 = f2%bubbles + taylor_pointer2
            !else
            !    bubbles2 = f2%bubbles
            end if

            ! If f2 is cuspy, do Taylor expansion for self
            if ( f2%type==F3D_TYPE_CUSP .or. f2%type == F3D_TYPE_NUCP)  then 
                ! if not pre-calculated, calculate taylor series bubbles for self
                if (allocated(self%taylor_series_bubbles)) then
                    taylor_pointer1 => self%taylor_series_bubbles
                else
                    taylor_series_bubbles1 = self%get_taylor_series_bubbles()
                    taylor_pointer1 => taylor_series_bubbles1
                end if
            !    bubbles1 = self%bubbles + taylor_pointer1
            !else
            !    bubbles1 = self%bubbles
            end if
#ifdef HAVE_CUDA_PROFILING
            call stop_nvtx_timing()
#endif
            ! do the multiplication
            !result_bubbles = bubbles1 * bubbles2
            call bubbles_multipliers%multiply(self%bubbles, f2%bubbles, &
                          result_bubbles, taylor_pointer1, taylor_pointer2)

            
            call taylor_series_bubbles1%destroy()
            call taylor_series_bubbles2%destroy()
            !call bubbles1%destroy()
            !call bubbles2%destroy()
            nullify(taylor_pointer1)
            nullify(taylor_pointer2)
        end if

    end subroutine

    !> Inner product
    !! \f$ \int_{\mathbb{R}^3} f_1(\mathbf{r}) f_2(\mathbf{r}) \mathrm{d}^3r\f$
    function Function3D_dot(f1, f2) result(dot)
        class(Function3D), intent(in)    :: f1
        class(Function3D), intent(in)    :: f2

        real(REAL64)                     :: dot
        ! determine which input class's method are we calling
        ! if the f1 is a normal Function3D -type and the other 
        ! is not, we are calling the f2's right_dot_product
        ! otherwise, we call the f1's left_dot_product function.
        ! This is done to allow special features of the child
        ! classes. For example GBFMMFunction3D.
        select type(f1)
            type is (Function3D)
               select type(f2)
                   type is (Function3D)
                       dot = f1%right_dot_product(f2)
                   class default
                       dot = f2%left_dot_product(f1)
               end select
           class default
               dot = f1%left_dot_product(f2)
        end select
    end function

    !> Inner product
    !! \f$ \int_{\mathbb{R}^3} f_1(\mathbf{r}) f_2(\mathbf{r}) \mathrm{d}^3r\f$
    function Function3D_dot_product(self, f2) result(dot)
        class(Function3D), intent(in)    :: self
        class(Function3D), intent(in)    :: f2

        real(REAL64)                     :: dot

        type(BubblesMultiplier), pointer :: bubbles_multiplier
        type(Function3D), allocatable    :: temp
        !call bigben%split("Dot product")
        call self%multiply_sub(f2, temp, part_of_dot_product = .TRUE.)
#ifdef HAVE_CUDA_PROFILING
        call start_nvtx_timing("Dot Product. ")
#endif
#ifdef HAVE_CUDA
        bubbles_multiplier => bubbles_multipliers%get_multiplier(self%bubbles, &
                                  f2%bubbles, complex_version = .TRUE.)
        call temp%bubbles%set_cuda_interface(bubbles_multiplier%result_bubbles)
        call temp%bubbles%cuda_upload_all(bubbles_multiplier%result_bubbles)
        call temp%bubbles%cuda_destroy()
        nullify(bubbles_multiplier)
#endif
        
        dot=temp%integrate()
#ifdef HAVE_CUDA_PROFILING
        call stop_nvtx_timing()
#endif
        call temp%destroy()
        !call bigben%stop()
    end function
  

    !> Returns the value of the function at requested points.
    function Function3D_evaluate(self, points, add_bubbles) result(res)
        class(Function3D), intent(in) :: self
        real(REAL64), intent(in)      :: points(:,:)

        real(REAL64)                  :: res(size(points,2))
        real(REAL64)                  :: temp(size(points, 2), 1)

        type(Interpolator)            :: interpol

        logical, optional, intent(in) :: add_bubbles
        logical                       :: add_bubbs
        type(Grid1DPointer)           :: grid1d_pointers(3)

        ! determine whether we are adding contribution from bubbles
        add_bubbs = .TRUE.
        if (present(add_bubbles)) then
            add_bubbs = add_bubbles
        end if
        grid1d_pointers(X_)%p => self%grid%axis(X_)
        grid1d_pointers(Y_)%p => self%grid%axis(Y_)
        grid1d_pointers(Z_)%p => self%grid%axis(Z_)
        interpol=Interpolator(grid1d_pointers)

        temp = interpol%eval(self%cube,points)
        call interpol%destroy()

        ! add the cotribution from bubbles
        if( add_bubbs .and. self%bubbles%get_nbub_global()>0) then
            res = temp(:, 1) + self%bubbles%eval(points)
        else
            res = temp(:, 1)
        end if
       
        return
    end function


    !> Returns the value and the derivatives of the cube at the bubble centers.
    function Function3D_get_cube_contaminants(self, tmax, cube_limits, bubbls) result(contams)
        class(Function3D),intent(in), target :: self
        !> Maximum order in the Taylor expansions
        integer(INT32)                       :: tmax
        !> Limits in the area of the projected cube 
        real(REAL64), intent(in), optional   :: cube_limits(2, 3)
        !> The bubbles used in getting the centers where the contaminants are evaluated
        !! If not given, the bubbles of self, are used
        type(Bubbles), intent(in), optional, target  :: bubbls
        real(REAL64), allocatable            :: domain_bubble_centers(:, :)
        integer,      allocatable            :: domain_ibubs(:)

        real(REAL64), allocatable            :: local_contams(:, :), contams(:,:)

        integer(INT32)                       :: numf, ibub, i
        type(Interpolator)                   :: interpolato
        type(Grid1DPointer)                  :: grid1d_pointers(3)
        type(Bubbles), pointer               :: bubbles_
        
        if (present(bubbls)) then
            bubbles_ => bubbls
        else
            bubbles_ => self%bubbles
        end if 

        ! calculate the number of different derivative terms in the taylor series 
        numf=(tmax+1)*(tmax+2)*(tmax+3)/6
        if (present(cube_limits)) then
            domain_ibubs = bubbles_%get_ibubs_within_range(cube_limits)
            domain_bubble_centers = bubbles_%get_centers(domain_ibubs)
        else
            domain_bubble_centers = self%parallelization_info%get_domain_bubble_centers(bubbles_)
            domain_ibubs = self%parallelization_info%get_domain_bubble_ibubs(bubbles_)
        end if
        allocate(contams(numf, bubbles_%get_nbub_global()), source = 0.0d0)
        ! note: there will be an error if the nbub is 0 and we will go inside the 
        ! if-clause.
        if (size(domain_bubble_centers, 2) > 0) then
            allocate(local_contams(size(domain_bubble_centers, 2), numf), source = 0.0d0)

            grid1d_pointers(X_)%p => self%grid%axis(X_)
            grid1d_pointers(Y_)%p => self%grid%axis(Y_)
            grid1d_pointers(Z_)%p => self%grid%axis(Z_)
            interpolato=Interpolator(grid1d_pointers,tmax)
            local_contams(:,:)=interpolato%eval(self%cube, domain_bubble_centers)
            !end if
            ! deallocate memory, nullify pointer
            call interpolato%destroy()

            forall (i = 1 : size(domain_ibubs))
                contams(:, domain_ibubs(i)) = local_contams(i, :) 
            end forall
            deallocate(local_contams)
        end if 
        deallocate(domain_bubble_centers)
        deallocate(domain_ibubs)
        call self%parallelization_info%sum_matrix(contams)
        return
    end function
    
    


    !> Contract cube with vectors.

    !> Returns
    !! \f[
    !! \sum_{ijk} f^\Delta_{ijk} v^\mathrm{x}_i v^\mathrm{y}_j v^\mathrm{z}_k
    !! \f]
    !!
    !! Used for integration, etc.
    !! (Mappings \f$\mathbb{R}^3\rightarrow\mathbb{R}\f$}
    function contract_cube(incube,vector_x,vector_y,vector_z) result(res)
        real(REAL64), intent(in) :: incube(:, :, :)
        real(REAL64), intent(in) :: vector_x(size(incube, 1)), vector_y(size(incube, 2)), &
                                    vector_z(size(incube, 3))
        real(REAL64) :: res

        ! temporary array variables to hold results
        real(REAL64) :: tmp1(size(vector_y)), tmp3(size(vector_y))
        real(REAL64) :: tmp2(size(vector_z))

        integer(INT32) :: i, k, j

! The CUDA code below works, but is impossibly slow for small arrays, Thus it is
! commented out
!#ifdef HAVE_CUDA
!        type(CUDACube)         :: cuda_cube
!        type(CUDAVector)       :: cuda_vector_x, cuda_vector_y, cuda_tmp1

!        call CUBLAS_init()
!        cuda_cube = CUDACube(incube)
!        cuda_vector_x = CUDAVector(vector_x)
!        cuda_vector_y = CUDAVector(vector_y)
!        cuda_tmp1   = CUDAVector(size(vector_x))
        
!        call cuda_cube%upload()
!        call cuda_vector_x%upload()
!        call cuda_vector_y%upload()

!        do k = 1, size(incube, Z_)
!             call CUBLAS_dgemv_t(k,&
!                    cuda_cube%slice(Z_, k),&
!                    cuda_vector_x,&
!                    cuda_tmp1,&
!                    1.d0, 0.d0, 1, 1)
!             tmp2(k) = cuda_vector_y .dot. cuda_tmp1
!        end do
!        res=sum(tmp2*vector_z)  
 
!        call cuda_vector_x%destroy()
!        call cuda_vector_y%destroy()
!        call cuda_tmp1%destroy()
!        call cuda_cube%destroy()
!        call CUBLAS_destroy()
!#else

        do i=1, size(incube, Z_)
            tmp1(:)=xmatmul(vector_x,incube(:,:,i))
            tmp2(i)=sum(tmp1*vector_y)
        end do
        res=sum(tmp2*vector_z)
!#endif

    end function

    !> Contract cube with vectors.

    !> Returns
    !! \f[
    !! \sum_{ijk} f^\Delta_{ijk} v^\mathrm{x}_i v^\mathrm{y}_j v^\mathrm{z}_k
    !! \f]
    !!
    !! Used for integration, etc.
    !! (Mappings \f$\mathbb{R}^3\rightarrow\mathbb{R}\f$}
    function contract_complex_cube(incube,vecx,vecy,vecz) result(res)
        complex*16,   dimension(:,:,:), intent(in) :: incube
        real(REAL64), dimension(:),     intent(in) :: vecx,vecy,vecz
        complex*16                                 :: res

        complex*16                                 :: tmp1(size(vecy))
        complex*16                                 :: tmp2(size(vecz))

        integer(INT32)                             :: ndimx,ndimy,ndimz
        integer(INT32)                             :: i

        ndimx=size(incube,1)
        ndimy=size(incube,2)
        ndimz=size(incube,3)

        do i=1,ndimz
            tmp1(:)=xmatmul(vecx,incube(:,:,i))
            tmp2(i)=sum(tmp1*vecy)
        end do
        res=sum(tmp2*vecz)

    end function

    !> Get the potential of the cores as a new function
    function Function3D_get_nuclear_potential(self) result(new)
        class(Function3D), intent(in)    :: self
        type(Function3D)                 :: new
        
        real(REAL64),         pointer    :: bubble_values(:, :)
        integer                          :: ibub

        new = Function3D(self, type = F3D_TYPE_NUCP)

        ! init new bubbles with k of 'new' to -1 (values will be multiplied with r^-1)
        ! at evaluation
        new%bubbles = Bubbles(self%bubbles, lmax = 5, k = -1)
        new%bubbles = 0.0d0
        new%cube = 0.0d0

        ! set value to -charge
        do ibub = 1, new%bubbles%get_nbub()
            bubble_values => new%bubbles%get_f(ibub)
            bubble_values(:, 1) = - new%bubbles%get_z(ibub)
            nullify(bubble_values)
        end do
        allocate(new%taylor_series_bubbles, &
                     source = new%get_taylor_series_bubbles())
        
    end function


    !> Returns a cube with the bubbles injected into it
    function Function3D_inject_bubbles(self, ibubs) result(cubeout)
        class(Function3D)             :: self
        integer, intent(in), optional :: ibubs(:)
        real(REAL64), allocatable     :: cubeout(:,:,:)
        

        character(25)                 :: label
        integer                       :: grid_shape(3)

        write(label,'("Bubbles injection (L=",i0,")")') self%bubbles%get_lmax()
        call bigben%split(label)
        !call pdebug("inject_bubbles()", 1)

        if(verbo_g>0) then
            write(ppbuf, '("Injecting bubbles (L=",i0,") into ",a,"...")') &
                self%bubbles%get_lmax(), trim(self%label)
            call pinfo(ppbuf)
        end if

        if (self%bubbles%get_nbub_global() > 0) then
            cubeout = self%bubbles%eval_3dgrid(self%grid, ibubs = ibubs)
        else
            grid_shape = self%grid%get_shape()

            allocate(cubeout(grid_shape(X_),grid_shape(Y_),grid_shape(Z_)), source = 0.d0)
        end if 

        
        call bigben%stop()
        return
    end function

    !> Integrates a function over all space.
    !!
    !! Note that by definition the cube is 0 outside its domain, and the
    !! bubbles might step further out.
    function Function3D_integrate(self, only_bubbles_within_cube) result(val)
        class(Function3D), intent(in) :: self
        !> integrate only the  bubbles which have centers within the cube
        logical, optional, intent(in) :: only_bubbles_within_cube
        real(REAL64)                  :: val, bubbles_value

        real(REAL64)                  :: cube_ranges(2, 3)
        type(Bubbles)                 :: domain_bubbles
        

        call bigben%split("Function3D integrate")
        ! integrate over cube, note does not do communication, only
        ! integrates over the correct domain
        val = 0.0d0
        bubbles_value = 0.0d0
        call bigben%split("Cube integrate")
        val = self%parallelization_info%integrate_cube(self%cube)
        call bigben%stop()
        bubbles_value = 0.0d0
        if (self%bubbles%get_nbub_global() > 0) then 
#ifdef HAVE_CUDA
            bubbles_value = self%bubbles%integrate()
#else
            ! get the bubbles within the domain of this processor
            if (present(only_bubbles_within_cube)) then
                if(only_bubbles_within_cube) then
                    cube_ranges = self%grid%get_range()
                    domain_bubbles = self%bubbles%get_sub_bubbles(cube_ranges, copy_content = .TRUE.)
                else
                    domain_bubbles = self%parallelization_info%get_domain_bubbles(self%bubbles)
                end if
            else
                domain_bubbles = self%parallelization_info%get_domain_bubbles(self%bubbles)
            end if

            ! integrate over bubbles withing the domain of this processor
            bubbles_value = domain_bubbles%integrate()
            
            call domain_bubbles%destroy()
#endif
        end if
        
        call self%parallelization_info%sum_real(val)
        call self%parallelization_info%sum_real(bubbles_value)
        !print '("integrate, cube", f18.14, " bubbles: ", f18.14, " Total:", f18.14)', &
        !    val, bubbles_value, val+bubbles_value
        val = val + bubbles_value
        val = truncate_number(val, 6)
        !print *, "total", val
        call bigben%stop()
        return
    end function


    !> Create a copy of Function3D, projecting the cube onto target_grid.
    function Function3D_project_onto(self, target_grid, cell_limits, only_cube) result(new)
        class(Function3D), intent(in), target    :: self
        !> Grid3D onto which the function is to be projected
        type(Grid3D), intent(in),      target    :: target_grid
        !> Limits in the area of the projected cube 
        integer, intent(in), optional            :: cell_limits(2, 3)
        !> If this is set to .true. only cube part's multipole moments are evaluated
        logical, intent(in), optional            :: only_cube
        type(Function3D)                         :: new

        ! Help variables
        integer                                  :: cube_limits(2, 3)
        real(REAL64), pointer                    :: cube(:, :, :)
        type(Grid3D), target                     :: grid_t
        type(Grid3D), pointer                    :: grid
        type(Projector3D), allocatable           :: projector
        real(REAL64)                             :: qmin(3), qmax(3)
        real(REAL64), allocatable                :: temp_cube(:, :, :)
        integer                                  :: dims_out(3)
        
        call bigben%split("Project Onto start")
        cube => self%get_cube()
        if (present(cell_limits)) then
            ! each cell has nlip grid points and the first and two sequental cells share a grid point
            cube_limits(1, :) = (cell_limits(1, :) - (/1, 1, 1/)) * (self%grid%get_nlip() - 1) + 1
            cube_limits(2, :) = (cell_limits(2, :)) * (self%grid%get_nlip() - 1) + 1
            cube => cube(cube_limits(1, X_) : cube_limits(2, X_), &
                         cube_limits(1, Y_) : cube_limits(2, Y_), &
                         cube_limits(1, Z_) : cube_limits(2, Z_))
            grid_t = self%grid%get_subgrid(cell_limits)
            grid => grid_t
        else 
            grid => self%grid
        end if

        new%grid=>target_grid
        call bigben%stop()

        call bigben%split("Cube project")
        if (.not.associated (new%cube)) call new%init_cube() 
        !call pdebug("function_project()", 1)
        if(grid%is_equal(target_grid) .or. grid%is_similar(target_grid)) then
            ! Source and mould share the same grid
            ! Copy the cube of source
            new%cube = cube
        else if (grid%is_subgrid_of(target_grid)) then 
            ! get the cube coordinates of the input cube at the output cube
            dims_out = target_grid%get_shape()
            ! get start and end coordinates of grid
            qmin = grid%get_qmin()
            qmax = grid%get_qmax()

            ! get the coordinates of the start and end coordinates of grid in the 'target_grid'
            cube_limits(1, :) = (target_grid%get_icell(qmin) - [1, 1, 1]) * (target_grid%get_nlip()-1) + [1, 1, 1]
            cube_limits(2, :) = cube_limits(1, :) - [1, 1, 1]  + grid%get_shape() 
            
            new%cube = 0.0d0
            new%cube(cube_limits(1, X_) : cube_limits(2, X_),  &
                     cube_limits(1, Y_) : cube_limits(2, Y_),  &
                     cube_limits(1, Z_) : cube_limits(2, Z_))  &
                      = cube                                                      
        else
            allocate(projector, source = Projector3D(grid, target_grid))
            temp_cube = projector%transform_cube( cube )
            new%cube = temp_cube
            deallocate(temp_cube)
            call projector%destroy()
            deallocate(projector)
        end if

        call bigben%stop()
        new%taylor_order = self%taylor_order
        
        call bigben%split("Copy bubbles")
        if (.not. present(only_cube)) then
            new%bubbles=self%bubbles!%get_sub_bubbles(grid%get_range())
        else if (.not. only_cube) then
            new%bubbles=self%bubbles
        end if
        call bigben%stop()

        new%label=trim(self%label)//"_project"
        new%type=self%type
        call grid_t%destroy()
        nullify(grid)
        nullify(cube)

        ! as of now, we don't know how addition affects to the off diagonal bubbles
        ! thus, we are setting them to zero for now
        if(allocated(new%taylor_series_bubbles)) then
            call new%taylor_series_bubbles%destroy()
            deallocate(new%taylor_series_bubbles)
        end if
    end function

    function Function3D_square_gradient_norm(self, points, add_bubbles) result(res)
        class(Function3D), intent(in) :: self
        ! 3*N
        real(REAL64), intent(in)      :: points(:,:)

        real(REAL64)                  :: res(4,size(points,2))

        logical, optional, intent(in) :: add_bubbles
        logical                       :: add_bubbs

        type(Interpolator)            :: interpol

        real(REAL64), dimension(:), allocatable :: bubble_dv_x, bubble_dv_y, bubble_dv_z
        real(REAL64), dimension(:,:), allocatable :: cube_dv
        type(Grid1DPointer)                  :: grid1d_pointers(3)

        ! determine whether we are adding contribution from bubbles
        add_bubbs = .TRUE.
        if (present(add_bubbles)) then
            add_bubbs = add_bubbles
        end if

        ! column: 0th order derivative, dz, dy, dx
        allocate(cube_dv(size(points,dim=2),4))

        ! up to 1st order derivative
        
        grid1d_pointers(X_)%p => self%grid%axis(X_)
        grid1d_pointers(Y_)%p => self%grid%axis(Y_)
        grid1d_pointers(Z_)%p => self%grid%axis(Z_)
        interpol=Interpolator(grid1d_pointers,1)
        cube_dv = interpol%eval(self%cube,points)

        res(1,:) = cube_dv(:,4)**2+cube_dv(:,3)**2+cube_dv(:,2)**2
        res(2,:) = cube_dv(:,4)
        res(3,:) = cube_dv(:,3)
        res(4,:) = cube_dv(:,2)

        ! add the cotribution from bubbles
        if( add_bubbs .and. self%bubbles%get_nbub_global()>0) then
            allocate(bubble_dv_x(size(points,dim=2)))
            allocate(bubble_dv_y(size(points,dim=2)))
            allocate(bubble_dv_z(size(points,dim=2)))

            bubble_dv_x = self%bubbles%eval(points,[1,0,0])
            bubble_dv_y = self%bubbles%eval(points,[0,1,0])
            bubble_dv_z = self%bubbles%eval(points,[0,0,1])
            res(1,:)=(cube_dv(:,4)+bubble_dv_x(:))**2+&
                     (cube_dv(:,3)+bubble_dv_y(:))**2+&
                     (cube_dv(:,2)+bubble_dv_z(:))**2
            res(2,:)=cube_dv(:,4)+bubble_dv_x(:)
            res(3,:)=cube_dv(:,3)+bubble_dv_y(:)
            res(4,:)=cube_dv(:,2)+bubble_dv_z(:)
        end if

        call interpol%destroy()
        deallocate(cube_dv)
        if(allocated(bubble_dv_x)) deallocate(bubble_dv_x)
        if(allocated(bubble_dv_y)) deallocate(bubble_dv_y)
        if(allocated(bubble_dv_z)) deallocate(bubble_dv_z)
       
        return
    end function

    !> Polymorphic method for real64 multiplication
    function REAL64_times_Function3D_poly(factor, func) result (new)
        real(REAL64), intent(in)      :: factor
        class(Function3D), intent(in)  :: func
        type(Function3D)               :: new

        select type(func)
            type is (Function3D)
                new = REAL64_times_Function3D(factor, func)
        end select

    end function 

    !> Multiply function by a constant.
    function REAL64_times_Function3D(factor, func) result(new)
        real(REAL64), intent(in)      :: factor
        type(Function3D), intent(in)  :: func

        type(Function3D)              :: new
 
        new%label   = func%label
        new%type    = func%type
        new%grid    => func%grid
        call new%init_cube()
        new%cube    = factor * func%cube
        new%bubbles = factor * func%bubbles
        new%parallelization_info => func%parallelization_info
        new%taylor_order = func%taylor_order 

        if (allocated(new%taylor_series_bubbles)) then
            call new%taylor_series_bubbles%destroy()
            deallocate(new%taylor_series_bubbles)
        end if
    end function

    !> Multiply function by a constant.
    function Function3D_times_REAL64(func, factor) result(new)
        type(Function3D), intent(in)  :: func
        real(REAL64), intent(in)      :: factor

        type(Function3D)              :: new

        new=factor * func
    end function

    !> Divide function by a constant.
    function Function3D_divided_by_REAL64(func, divisor) result(new)
        type(Function3D), intent(in)  :: func
        real(REAL64), intent(in)      :: divisor

        type(Function3D)              :: new

        new=(1.d0/divisor) * func
    end function

    subroutine Function3D_assign_REAL64(func, val)
        type(Function3D), intent(inout) :: func
        real(REAL64),     intent(in)    :: val

        func%cube=val
        func%bubbles=val

        if (allocated(func%taylor_series_bubbles)) then
            call func%taylor_series_bubbles%destroy()
            deallocate(func%taylor_series_bubbles)
        end if
    end subroutine

    subroutine Function3D_assign_Function3D(function1, function2)
        class(Function3D), intent(inout), allocatable :: function1
        type(Function3D),  intent(in)                 :: function2

        if(allocated(function1)) then
            call function1%destroy()
            deallocate(function1)
        end if
        allocate(function1, source = function2)
        nullify(function1%cube)
        call function1%init_cube()
        function1%cube = function2%cube
    end subroutine
    
! the following function seems redundant
!    subroutine Function3D_assign_poly(function1, function2)
!        class(Function3D), intent(inout), allocatable :: function1
!        class(Function3D), intent(in)                :: function2
!        if(allocated(function1)) then
!            call function1%destroy()
!            deallocate(function1)
!        end if
!        allocate(function1, source = function2)
!        nullify(function1%cube)
!        call function1%init_cube()
!        function1%cube = function2%cube
!    end subroutine
    
    subroutine Operator3D_assign(operator1, operator2)
        class(Operator3D), intent(inout), allocatable :: operator1
        class(Operator3D), intent(in)                 :: operator2

        allocate(operator1, source = operator2)
    end subroutine


    !> interface to the method below to allow the operator to be used
    function Function3D_operate_with(self, op) result(new)
        !> Input function
        class(Function3D), intent(in)  :: self
        !> Operator
        class(Operator3D), intent(in)  :: op
        
        class(Function3D), allocatable :: new
        call op%operate_on(self, new)
        
    end function

    subroutine Function3D_communicate_cube(self, sum_borders, reversed_order)
        !> Input function
        class(Function3D), intent(inout), target  :: self
        logical, intent(in), optional             :: sum_borders
        logical, intent(in), optional             :: reversed_order
        real(REAL64), pointer                     :: cube_pointer(:, :, :)
        
        cube_pointer => self%cube
        call self%parallelization_info%communicate_cube(cube_pointer, & 
                 sum_borders = sum_borders, reversed_order = reversed_order)
        nullify(cube_pointer)
    end subroutine

    subroutine Function3D_communicate_cube_borders(self, reversed_order, sum_result)
        !> Input function
        class(Function3D), intent(inout), target  :: self
        logical, intent(in), optional             :: reversed_order
        logical, intent(in), optional             :: sum_result
        real(REAL64), pointer                     :: cube_pointer(:, :, :)
        
        cube_pointer => self%cube
        call self%parallelization_info%communicate_cube_borders(cube_pointer, & 
                  reversed_order = reversed_order, sum_result = sum_result)
        nullify(cube_pointer)
    end subroutine
    
    subroutine Function3D_inject_bubbles_to_cube(self, injected_bubbles, lmin, factor)
        class(Function3D),      intent(inout), target   :: self
        !> The injected bubbles. 
        type(Bubbles),           intent(in),   target   :: injected_bubbles
        !> The maximum angular quantum number of the minimum injected bubbles,
        !! i.e., bubbles with l >= lmin are injected to the cube of self
        integer, optional,       intent(in)             :: lmin
        real(REAL64),  optional                         :: factor
        real(REAL64),  allocatable                      :: injection(:, :, :)
#ifdef HAVE_CUDA
        type(CudaCube)                                  :: cuda_injection
#endif
        type(Bubbles)                                   :: temp_bubbles
        type(Bubbles),    pointer                       :: bubbls
        integer                                         :: lmin_
        lmin_ = 0
        if (present(lmin)) lmin_ = lmin

        bubbls => injected_bubbles
        ! and inject all the extra bubbles to the cube
        if (lmin_ <= bubbls%get_lmax()) then

            allocate(injection(self%grid%axis(X_)%get_shape(), & 
                               self%grid%axis(Y_)%get_shape(), &
                               self%grid%axis(Z_)%get_shape()))
#ifdef HAVE_CUDA
            cuda_injection = CudaCube(injection, all_memory_at_all_devices = .FALSE.)

            call cuda_injection%set_to_zero()
            call bubbls%eval_3dgrid_cuda(self%grid, cuda_injection, lmin=lmin_)

            call cuda_injection%download()

            call CUDASync_all()

            call cuda_injection%destroy()
#else
            injection = bubbls%eval_3dgrid(self%grid, lmin = lmin_)
#endif
            if (present(factor)) injection = injection * factor
            self%cube = self%cube + injection
            deallocate(injection)
        end if

    end subroutine

    !> Injects bubbles with angular quantum number l > 'lmax' of 'self%bubbles' to the cube
    !! and stores the l <= 'lmax' bubbles to 'self%bubbles'.
    subroutine Function3D_inject_extra_bubbles(self, lmax, injected_bubbles)
        class(Function3D),      intent(inout), target   :: self
        !> The maximum angular quantum number of the preserved bubbles,
        !! i.e., bubbles with l > lmax are injected to the cube of self
        integer,                 intent(in)             :: lmax
        !> The injected bubbles. If not present, then this subroutine injects the bubbles
        !! of self to the cube of self.
        type(Bubbles), optional, intent(in),   target   :: injected_bubbles
        type(Bubbles)                                   :: temp_bubbles
        type(Bubbles),    pointer                       :: bubbls

        if (present(injected_bubbles)) then
            bubbls => injected_bubbles
        else
            bubbls => self%bubbles
        end if
        call self%inject_bubbles_to_cube(bubbls, lmin = lmax + 1)

        temp_bubbles = Bubbles(bubbls, copy_content = .TRUE., lmax = min(lmax, bubbls%get_lmax()))
        call self%bubbles%destroy()
        self%bubbles = Bubbles(temp_bubbles, copy_content = .TRUE.)
        call temp_bubbles%destroy()

    end subroutine
    
    subroutine Function3D_dispose_extra_bubbles(self, lmax)
        class(Function3D),      intent(inout), target   :: self
        !> The maximum angular quantum number of the preserved bubbles,
        !! i.e., bubbles with l > lmax are deleted.
        integer,                 intent(in)             :: lmax
        type(Bubbles)                                   :: temp_bubbles
        type(Bubbles),    pointer                       :: bubbls

        if (self%bubbles%get_lmax() > lmax) then
            temp_bubbles = Bubbles(self%bubbles, copy_content = .TRUE., lmax = lmax)
            call self%bubbles%destroy()
            self%bubbles = Bubbles(temp_bubbles, copy_content = .TRUE.)
            call temp_bubbles%destroy()
        end if

    end subroutine
    
    subroutine Function3D_print_out_cube_at_point(self, point, dx, dy, dz)
        class(Function3D), intent(in)       :: self
        real(REAL64),           intent(in)  :: point(3)
        integer, optional, intent(in)       :: dx, dy, dz
        real(REAL64)                        :: c2(3)
        
        integer                             :: ix, iy, iz, x, y, z, nlip, dx_, dy_, dz_, c(3)
        
        dx_ = 1
        dy_ = 3
        dz_ = 3
        if (present(dx)) dx_ = dx
        if (present(dy)) dy_ = dy
        if (present(dz)) dz_ = dz
        
        nlip = self%grid%get_nlip()
        ix =   self%grid%axis(X_)%get_icell(point(X_)) * (nlip-1) +1 &
            + self%grid%lip%get_first()
        iy =   self%grid%axis(Y_)%get_icell(point(Y_)) * (nlip-1) +1 &
            + self%grid%lip%get_first()
        iz = self%grid%axis(Z_)%get_icell(point(Z_)) * (nlip-1) +1 &
            + self%grid%lip%get_first()

        if (ix > 0 .and. iy > 0 .and. iz > 0) then
            c2(X_) = ix + self%grid%axis(X_)%x2cell(point(X_))
            c2(Y_) = iy + self%grid%axis(Y_)%x2cell(point(Y_))
            c2(Z_) = iz + self%grid%axis(Z_)%x2cell(point(Z_))
            c(X_) = nint(c2(X_))
            c(Y_) = nint(c2(Y_))
            c(Z_) = nint(c2(Z_))
            print *, "Point: ", point, "as gridpoints: ", c2
                
            do x = max(1, c(X_)-dx_), min(c(X_)+dx_, size(self%cube, 1))
                write(*, "('----- x:', i3, '-----')", advance = 'no') x
                if (x == c(X_)) then
                    write(*, "('c ')")
                else if (mod(x, nlip-1) == 1) then
                    write(*, "('| ')", advance = 'no')
                else
                    write(*, "('  ')")
                end if
                do y = max(1, c(Y_)-dy_), min(c(Y_)+dy_, size(self%cube, 2)) 
                    if (y == c(Y_)) then
                        write(*, "('c ')", advance = 'no')
                    else if (mod(y, nlip-1) == 1) then
                        write(*, "('| ')", advance = 'no')
                    else
                        write(*, "('  ')", advance = 'no')
                    end if
                    write(*, "('y:', i3, ' .. ')", advance = 'no') y
                    do z = max(1, c(Z_)-dz_), min(c(Z_)+dz_, size(self%cube, 3))
                        write(*, "(f10.6)", advance = 'no') self%cube(x, y, z)
                        if (z == c(Z_) .and. y == c(Y_)) then
                            write(*, "('c ')", advance = 'no')
                        else if (mod(z, nlip-1) == 1) then
                            write(*, "('| ')", advance = 'no')
                        else
                            write(*, "('  ')", advance = 'no')
                        end if
                    end do
                    write(*, *) "" 
                end do
            end do
        end if
    end subroutine
    
    subroutine Function3D_print_out_centers(self, dx, dy, dz)
        class(Function3D), intent(in)  :: self
        integer, optional, intent(in)  :: dx, dy, dz
        integer                        :: i
        real(REAL64)                   :: center(3)

        
    
        do i = 1, self%bubbles%get_nbub()
            center = self%bubbles%get_centers(i)
            call self%print_out_cube_at_point(center, dx, dy, dz)

        end do
    end subroutine

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                          Function3D Multipole tools                    %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    function Function3D_bubble_multipoles(self, lmax, limits, evaluation_point) result(qbub)
        class(Function3D)             :: self
        integer, intent(in), optional :: lmax
        !> The starting and ending points of the multipole evaluation as cell numbers
        integer, intent(in), optional :: limits(2, 3)
        real(REAL64), optional        :: evaluation_point(3)
        

        real(REAL64), allocatable :: qbub(:, :)
        real(REAL64)              :: ranges(2, 3)
        type(Grid3D)              :: subgrid
        type(Bubbles)             :: bubbls


        if (present(limits)) then
            subgrid = self%grid%get_subgrid(limits)
            ranges = subgrid%get_range()
            bubbls = self%bubbles%get_sub_bubbles(ranges, copy_content = .TRUE.)
        else
            bubbls = self%bubbles
        end if 

        qbub = bubbls%get_multipole_moments(lmax = lmax, evaluation_point = evaluation_point)

        call subgrid%destroy()
    end function

    !> Compute the multipole moments up to order `lmax`.

    !> The \f$(l,m)\f$-th multipole moment \f$q_{lm}\f$ is given by
    !! \f[ q_{lm}=
    !!   \int_{\mathbb{R}^3} f^\Delta(x,y,z) Y_{lm}(\theta_A,\phi_A)
    !!   \mathrm{d}^3r \f]
    !! where \f$\theta_A\f$ and \f$\phi_A\f$ are coordinates relative to the
    !! reference point `ref`.
    !!
    !! ### Implementation details
    !!
    !! The integrals are computed in terms of Cartesian monomials
    !! \f$\mathbf{x_A}^\mathbf{\alpha}=
    !!    x_A^{\alpha_1}y_A^{\alpha_2}z_A^{\alpha_3}\f$,
    !! e.g.
    !! \f[ \tilde{q}_{\mathbf{\alpha}}=
    !!   \int_{\mathbb{R}^3} f^\Delta(x,y,z) \mathbf{x}^\mathbf{\alpha}
    !!   \mathrm{d}^3r \f]
    !! which are then added to the relevant momenta. E.g.
    !! \f$\tilde{q}_{200}\f$ contributes to \f$q_{20}\f$ with a factor of
    !! -1/2 and to \f$q_{22}\f$ with a factor of 1/2.
    !!
    !! The algorithm is designed to minimize the amount of operations and to
    !! exploit memory ordering.
    function Function3D_cube_multipoles(self, lmax, ref, limits) result(mp)
        class(Function3D), intent(in) :: self
        !> Maximum multipole order
        integer,           intent(in) :: lmax
        !> Reference point, where the multipoles are evaluated
        real(REAL64),      intent(in) :: ref(3)
        !> Limits of the multipole evaluation in the grid
        integer, optional             :: limits(2, 3)

        ! mp: multipole moments
        real(REAL64)                  :: mp((lmax+1)**2)

        type(CartIter)                :: iter
        integer                       :: kappa(3)
        integer                       :: dimflip

        real(REAL64), allocatable     :: cube  (:,:,:)
        real(REAL64), allocatable     :: matrix(:,:)
        real(REAL64), allocatable     :: vector(:)
        real(REAL64)                  :: cart_mp

        ! integrals
        real(REAL64), allocatable     :: ints_x(:)
        real(REAL64), allocatable     :: ints_y(:)
        real(REAL64), allocatable     :: ints_z(:)

        real(REAL64), allocatable     :: gridpoints_x(:)
        real(REAL64), allocatable     :: gridpoints_y(:)
        real(REAL64), allocatable     :: gridpoints_z(:)

        integer                       :: ndim(3)
        integer                       :: i,j,k, nlip

        type(RealRegularSolidHarmonics) :: harmo
        real(REAL64)                    :: coeff
        integer                         :: l,m
        logical                         :: continue_iteration, continue_lm_iteration

        mp=0.d0
        harmo=RealRegularSolidHarmonics(lmax)
       
        iter=CartIter(ndim=3,maximum_modulus=lmax)

        
        nlip=self%grid%get_nlip()
       
        if (present(limits)) then
            ndim=(limits(2, :) - limits(1, :) + (/1, 1, 1/)) * (nlip-1) + 1
            ! Note: X_ = 1, Y_ = 2, Z_ = 3
            ints_x=self%grid%axis(X_)%get_ints(limits(:, X_))
            ints_y=self%grid%axis(Y_)%get_ints(limits(:, Y_))
            ints_z=self%grid%axis(Z_)%get_ints(limits(:, Z_))

            ! get gridpoints relative to the ref
            gridpoints_x=self%grid%axis(X_)%get_coord(limits(:, X_)) - ref(X_)
            gridpoints_y=self%grid%axis(Y_)%get_coord(limits(:, Y_)) - ref(Y_)
            gridpoints_z=self%grid%axis(Z_)%get_coord(limits(:, Z_)) - ref(Z_)

            ! handle only limited part of the cube
            cube=self%cube((limits(1, X_)-1)*(nlip-1)+1: (limits(2, X_))*(nlip-1)+1,&
                           (limits(1, Y_)-1)*(nlip-1)+1: (limits(2, Y_))*(nlip-1)+1,&
                           (limits(1, Z_)-1)*(nlip-1)+1: (limits(2, Z_))*(nlip-1)+1)
        else
            ndim=self%grid%get_shape()

            ! Note: X_ = 1, Y_ = 2, Z_ = 3
            ints_x=self%grid%axis(X_)%get_ints()
            ints_y=self%grid%axis(Y_)%get_ints()
            ints_z=self%grid%axis(Z_)%get_ints()

            ! get gridpoints relative to the ref
            gridpoints_x=self%grid%axis(X_)%get_coord() - ref(X_)
            gridpoints_y=self%grid%axis(Y_)%get_coord() - ref(Y_)
            gridpoints_z=self%grid%axis(Z_)%get_coord() - ref(Z_)


            ! handle all of the cube
            cube=self%cube
 
        end if

        allocate(matrix(size(cube, 2), size(cube, 3)), source = 0.0d0)
        allocate(vector(size(cube, 3)), source = 0.0d0)
        ! The dimflip<= ... structure is equivalent to a loop like:
        ! do ix=0,lmax
        !     matrix=integrate cube along x
        !     do iy=0,lmax-ix
        !         vector=integrate matrix along y
        !         do iz=0,lmax-ix-iy
        !             cart_mp=integrate vector along z
        !             multiply vector with z_A
        !         end do
        !         multiply matrix columns with y_A
        !     end do
        !    multiply cube x-fibers with x_A
        !end do
        k=0
        call iter%next(kappa, continue_iteration, dimflip)
        do while (continue_iteration)
            ! evaluate if the y-value changed in (x, y, z) kappa matrix
            ! which iterates through different values starting from z
            ! (0, 0, 0), (0, 0, 1), (0, 0, 2) .. (0, 1, 0), (0, 1, 1), ...
            if(dimflip<=Y_) then
                if(dimflip<=X_) then
                    ! Integrate along x
                    matrix=contract_cube(cube, ints_x)
                    ! Multiply with distance's x-component (r_x)
                    if(kappa(X_)<lmax) forall(j=1:ndim(Y_),k=1:ndim(Z_)) &
                        cube(:,j,k) = cube(:,j,k) * gridpoints_x
                end if
                ! Integrate along y
                vector=contract_matrix(matrix, ints_y)

                ! Multiply with distance's y-component (r_y), if there are
                ! kappa vectors with Y_ index larger than lmax left
                if (kappa(Y_)<lmax) then
                    forall(k=1:ndim(Z_)) matrix(:,k) = matrix(:,k) * gridpoints_y
                end if 
            end if
            ! Integrate along z
            k=k+1
            cart_mp = dot_product(vector, ints_z)
            ! Multiply vector with distance's z-component (r_z)
            if(kappa(Z_)<lmax) vector = vector * gridpoints_z

            ! Multiply the cartesian contribution with the spherical harmonics
            ! and add them to the result array
            call harmo%next_lm(l, m, coeff, continue_lm_iteration, 1)
            do while(continue_lm_iteration)
                mp(lm_map(l,m)) = mp(lm_map(l,m)) + coeff * cart_mp
 
                ! get the next meaningful l, m combination and the corresponding coefficient
                ! for the monomial kappa
                call harmo%next_lm(l, m, coeff, continue_lm_iteration, 1)
            end do
            call iter%next(kappa, continue_iteration, dimflip)
        end do
        deallocate(cube)
        deallocate(matrix)
        deallocate(vector)
        deallocate(ints_x, ints_y, ints_z)
    contains
        function contract_cube(cube, vector) result(matrix)
            real(REAL64), intent(in) :: cube  (:,:,:)
            real(REAL64), intent(in) :: vector(:)
            real(REAL64)             :: matrix(size(cube,2), size(cube,3))
            integer                  :: j,k
! The CUDA code below works, but is impossibly slow for small arrays, Thus it is
! commented out
!#ifdef HAVE_CUDA
!            type(CUDACube)         :: cuda_cube
!             type(CUDAVector)       :: cuda_vector
!             type(CUDAMatrix)       :: cuda_matrix
! 
!             call CUBLAS_init()
!             cuda_cube = CUDACube(cube)
!             cuda_vector = CUDAVector(vector)
!             call cuda_cube%upload()
!             call cuda_vector%upload()
! 
!             cuda_matrix = CUDAMatrix(matrix)
! 
!             do k = 1, size(cube, Z_)
!                  call CUBLAS_dgemv_t(k,&
!                         cuda_cube%slice(Z_, k),&
!                         cuda_vector,&
!                         cuda_matrix%slice(Y_, k),&
!                         1.d0, 0.d0, 1, 1)
!             end do
!             
!  
!             call cuda_matrix%download()
! 
!             call cuda_matrix%destroy()
!             call cuda_vector%destroy()
!             call cuda_cube%destroy()
!             call CUBLAS_destroy()
! 
!            !print *, "matrix after", maxval(matrix)
! #else
            forall(j=1:size(cube,2),k=1:size(cube,3))
                matrix(j,k) = dot_product( cube(:,j,k), vector )
            end forall
            !print *, "correct matrix after", maxval(matrix)
! #endif
        end function

        function contract_matrix(matrix, vector) result(vector_out)
            real(REAL64), intent(in) :: matrix(:,:)
            real(REAL64), intent(in) :: vector(:)
            real(REAL64)             :: vector_out(size(matrix,2))

            integer                  :: k
!#ifdef HAVE_CUDA
!            type(CUDAVector)       :: cuda_vector, cuda_vector_out
!            type(CUDAMatrix)       :: cuda_matrix

!            call CUBLAS_init()

!            call CUDAmatrix_init(cuda_matrix, matrix)
!            call CUDAvector_init(cuda_vector, vector)
!            call CUDAmatrix_upload(cuda_matrix)
!            call CUDAvector_upload(cuda_vector)

!            call CUDAvector_init(cuda_vector_out, vector_out)

!            call CUBLAS_dgemv(k,&
!                    cuda_matrix,&
!                    cuda_vector,&
!                    cuda_vector_out, Y_, k),&
!                    1.d0, 0.d0, 1, 1)
            
            
  
!            call CUDAmatrix_download(cuda_vector)

!            call CUDAmatrix_destroy(cuda_matrix)
!            call CUDAvector_destroy(cuda_vector)
!            call CUDAvector_destroy(cuda_vector_out)
!            call CUBLAS_destroy()

            !print *, "matrix after", maxval(matrix)
!#else
            forall(k=1:size(matrix,2))
                vector_out(k) = dot_product( matrix(:,k), vector )
            end forall
!#endif
        end function
    end function

    !> Return a string containing formatted information about the grid, etc.

    !> Example output:
    !!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    !!========================================
    !!Label: [MO#1]
    !!----------------- Cube -----------------
    !!   Shape:      (  31 x   31 x   31 )
    !!   Total size:       29791
    !!   * X range:       -12.28  -  11.72
    !!   * Y range:       -12.57  -  11.43
    !!   * Z range:       -12.00  -  12.00
    !!--------------- Bubbles ----------------
    !!  Number of bubbles:    9
    !!               LMAX:    2
    !!========================================
    !!               Number of coefficients
    !!   Bubble       per radial function
    !!       1                3757
    !!       2                3757
    !!       3                4033
    !!       4                2401
    !!       5                2401
    !!       6                2401
    !!       7                2401
    !!       8                2401
    !!       9                2401
    !!  TOTAL:      233577
    !!---------------- TOTAL -----------------
    !! Number of coeffs:     263368
    !!========================================
    !!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    function Function3D_info(self) result(info)
        class(Function3D), intent(in) :: self
        character(:), allocatable :: info

        character(80)        :: line
        character, parameter :: NL=new_line(" ")

        real(REAL64)           :: ranges(2,3)
        integer                :: nbub, ibub, lmax
        integer(INT64)         :: cubshape(3),bubshape,cubtotal,total

        ranges  =self%grid%get_range()
        cubshape=self%grid%get_shape()
        nbub    =self%bubbles%get_nbub()
        cubtotal=product(cubshape)
        lmax    =self%bubbles%get_lmax()

        allocate(character(1000) :: info)

        write(info,'(&
            &"========================================",  "'//NL//'",&
            &"Label: [",          a                ,"]",  "'//NL//'",&
            &"----------------- Cube -----------------",  "'//NL//'",&
            &"   Shape:      (",i4," x ",i4," x ",i4," )","'//NL//'",&
            &"   Total size:  ",i10,                      "'//NL//'",&
            &"   * X range:       ",f6.2,"  - ",f6.2,     "'//NL//'",&
            &"   * Y range:       ",f6.2,"  - ",f6.2,     "'//NL//'",&
            &"   * Z range:       ",f6.2,"  - ",f6.2,     "'//NL//'",&
            &"--------------- Bubbles ----------------",  "'//NL//'",&
            &"  Number of bubbles: ",i4,                  "'//NL//'",&
            &"               LMAX: ",i4,                  "'//NL//'",&
            &"               Number of coefficients",     "'//NL//'",&
            &"   Bubble       per radial function"                   &
            &)') trim(self%label),cubshape,cubtotal,ranges,nbub,lmax
        total=0
        do ibub=1,nbub
            bubshape=self%bubbles%gr(ibub)%p%get_shape()
            total=total+bubshape
            write(line,'(&
            &      i8,                  i20)'),ibub,bubshape
            info=append(info, line)
        end do
        total=total*(lmax+1)**2
        write(line,'("  TOTAL:  ",i10)') total
        info=append(info,line)
        info=append(info,"---------------- TOTAL -----------------")
        total=total+cubtotal
        write(line,'(" Number of coeffs: ",i10)') total
        info=append(info,line)
        info=append(info,"========================================")
        info=trim(info)
        return
    contains
        function append(str1, str2) result(str)
            character(*), intent(in) :: str1
            character(*), intent(in) :: str2
            character(:), allocatable:: str

            str=trim(str1)//new_line(" ")//str2
        end function
    end function

    !> Interface for new Bubbles transformation functions

    !> @todo Change to function returning Bubbles

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                           Function3DEvaluator functions                 %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function Function3DEvaluator_init(high_memory_profile) result(new)
        !> If memory is allocated for gradients too (takes more memory, but makes the calculation faster)
        logical,          intent(in)                   :: high_memory_profile
        !> The initialized evaluator
        type(Function3DEvaluator)                      :: new

        new%cube_evaluator    = CubeEvaluator(high_memory_profile)
        new%bubbles_evaluator = BubblesEvaluator(high_memory_profile)
    end function



    !> Evaluate the gradients of an input Function3D object to three (3) Function3D objects
    !! (derivative_x, derivative_y, and derivative_z)
    subroutine Function3DEvaluator_evaluate_gradients_as_Function3Ds(self, input_function, &
                                                                     derivative_x, derivative_y, derivative_z, &
                                                                     ignore_bubbles, ignore_cube)
        !> evaluator object
        class(Function3DEvaluator), intent(inout) :: self
        !> input function3d object
        type(Function3D),          intent(in)     :: input_function    
        !> output derivatives as function3d objects
        type(Function3D),          intent(out)    :: derivative_x, derivative_y, derivative_z
        logical,         optional, intent(in)     :: ignore_bubbles, ignore_cube
        logical                                   :: ignore_bubbles_, ignore_cube_
        type(Bubbles)                             :: bubbles_derivative_x, bubbles_derivative_y, bubbles_derivative_z
        type(Function3D)                          :: temp
        real(REAL64),          allocatable        :: temp_cube(:, :, :)
        
        ignore_cube_ = .FALSE.
        ignore_bubbles_ = .FALSE.
        if (present(ignore_cube)) ignore_cube_ = ignore_cube
        if (present(ignore_bubbles)) ignore_bubbles_ = ignore_bubbles
        
        ! init the result functions and destroy the bubbles from them
        call derivative_x%init_copy(input_function, copy_content = .FALSE.)
        call derivative_x%bubbles%destroy()
        call derivative_y%init_copy(input_function, copy_content = .FALSE.)
        call derivative_y%bubbles%destroy()
        call derivative_z%init_copy(input_function, copy_content = .FALSE.)
        call derivative_z%bubbles%destroy()

        if (.not. ignore_bubbles_) then
            ! evaluate the gradients of bubbles as bubbles
            call self%bubbles_evaluator%evaluate_gradients_as_bubbles &
                    (input_function%bubbles, bubbles_derivative_x, bubbles_derivative_y, bubbles_derivative_z)
                    
            ! and set the result gradient bubbles to the result objects
            derivative_x%bubbles = Bubbles(bubbles_derivative_x, copy_content = .TRUE.)
            derivative_y%bubbles = Bubbles(bubbles_derivative_y, copy_content = .TRUE.)
            derivative_z%bubbles = Bubbles(bubbles_derivative_z, copy_content = .TRUE.)
            call bubbles_derivative_x%destroy()
            call bubbles_derivative_y%destroy()
            call bubbles_derivative_z%destroy()
        end if

        
        if (.not. ignore_cube_) then
            allocate(temp_cube(input_function%grid%axis(X_)%get_shape(), &
                               input_function%grid%axis(Y_)%get_shape(), &
                               input_function%grid%axis(Z_)%get_shape()))
            call self%cube_evaluator%evaluate_grid(input_function%cube, input_function%grid, temp_cube, &
                                                   derivative_x%cube, derivative_y%cube, derivative_z%cube)
            deallocate(temp_cube)
        end if  
        
        
    end subroutine

    subroutine Function3DEvaluator_evaluate_points(self, input_function, result_points, derivative_x, derivative_y, derivative_z, &
                                                   derivative_points_x, derivative_points_y, derivative_points_z, &
                                                   ignore_cube, ignore_bubbles, ibubs)
        !> evaluator object
        class(Function3DEvaluator),  intent(inout) :: self
        !> input function3d object
        class(Function3D),           intent(in)    :: input_function   
        !> output points object 
        type(Points),                intent(inout) :: result_points
        !> input function3d objects containing the derivatives of the input_function in x, y, and z directions
        class(Function3D), optional, intent(in)    :: derivative_x, derivative_y, derivative_z  
        !> output points objects for the derivatives of the input_function in x, y, and z directions
        type(Points),      optional, intent(inout) :: derivative_points_x, derivative_points_y, derivative_points_z 
        !> If the cube/bubbles are ignored, if not present, nothing is ignored
        logical,           optional, intent(in)    :: ignore_cube, ignore_bubbles       
        !> The global order numbers of evaluated bubbles, if not given all bubbles are evaluated
        integer,           optional, intent(in)    :: ibubs(:)
        logical                                    :: ignore_cube_, ignore_bubbles_
        !> output points objects for the derivatives of the input_function in x, y, and z directions
        type(Points)                               :: bubbles_derivative_points_x, &
                                                      bubbles_derivative_points_y, &
                                                      bubbles_derivative_points_z, &
                                                      bubbles_result_points
        
        ignore_cube_ = .FALSE.
        if (present(ignore_cube)) ignore_cube_ = ignore_cube
        ignore_bubbles_ = .FALSE.
        if (present(ignore_bubbles)) ignore_bubbles_ = ignore_bubbles
        
        if (present(derivative_x) .and. present(derivative_y) .and. present(derivative_z)) then
            ! evaluate the points from bubbles
            if (.not. ignore_bubbles_) call self%bubbles_evaluator%evaluate_points_from_bubbles( &
                                                input_function%bubbles, derivative_x%bubbles, &
                                                derivative_y%bubbles, derivative_z%bubbles, &
                                                result_points, derivative_points_x, &
                                                derivative_points_y, derivative_points_z, &
                                                ibubs = ibubs)
    
            ! store the results from bubbles to temporary variables, if there is need for this
            if (.not. ignore_bubbles_ .and. .not. ignore_cube_ .and. present(derivative_points_x)) &
                bubbles_derivative_points_x = Points(derivative_points_x, copy_content = .TRUE.)
            if (.not. ignore_bubbles_ .and. .not. ignore_cube_ .and. present(derivative_points_y)) &
                bubbles_derivative_points_y = Points(derivative_points_y, copy_content = .TRUE.)
            if (.not. ignore_bubbles_ .and. .not. ignore_cube_ .and. present(derivative_points_z)) &
                bubbles_derivative_points_z = Points(derivative_points_z, copy_content = .TRUE.)
            if (.not. ignore_bubbles_ .and. .not. ignore_cube_) &
                bubbles_result_points = Points(result_points, copy_content = .TRUE.)

            ! evaluate points from cubes
            if (.not. ignore_cube_) call self%cube_evaluator%evaluate_points_from_cubes( &
                                              input_function%cube, derivative_x%cube, & 
                                              derivative_y%cube, derivative_z%cube, &
                                              input_function%grid, result_points, &
                                              derivative_points_x, derivative_points_y, derivative_points_z)
        else
            
            if (.not. ignore_bubbles_) call self%bubbles_evaluator%evaluate_points(input_function%bubbles, &
                                                result_points, derivative_points_x, derivative_points_y, &
                                                derivative_points_z, ibubs = ibubs)
            
            ! store the results from bubbles to temporary variables, if there is need for this
            if (.not. ignore_bubbles_ .and. .not. ignore_cube_ .and. present(derivative_points_x)) &
                bubbles_derivative_points_x = Points(derivative_points_x, copy_content = .TRUE.)
            if (.not. ignore_bubbles_ .and. .not. ignore_cube_ .and. present(derivative_points_y)) &
                bubbles_derivative_points_y = Points(derivative_points_y, copy_content = .TRUE.)
            if (.not. ignore_bubbles_ .and. .not. ignore_cube_ .and. present(derivative_points_z)) &
                bubbles_derivative_points_z = Points(derivative_points_z, copy_content = .TRUE.)
            if (.not. ignore_bubbles_ .and. .not. ignore_cube_) &
                bubbles_result_points = Points(result_points, copy_content = .TRUE.)

            if (.not. ignore_cube_)    call self%cube_evaluator%evaluate_points(input_function%cube, input_function%grid, &
                                                result_points, derivative_points_x, derivative_points_y, derivative_points_z)
        end if
        

        ! add the results from bubbles, if there is need for this
        if (.not. ignore_bubbles_ .and. .not. ignore_cube_ .and. present(derivative_points_x)) then
            call derivative_points_x%add_in_place(bubbles_derivative_points_x)
            call bubbles_derivative_points_x%destroy()
        end if
        if (.not. ignore_bubbles_ .and. .not. ignore_cube_ .and. present(derivative_points_y)) then
            call derivative_points_y%add_in_place(bubbles_derivative_points_y)
            call bubbles_derivative_points_y%destroy()
        end if
        if (.not. ignore_bubbles_ .and. .not. ignore_cube_ .and. present(derivative_points_z)) then
            call derivative_points_z%add_in_place(bubbles_derivative_points_z)
            call bubbles_derivative_points_z%destroy()
        end if
        if (.not. ignore_bubbles_ .and. .not. ignore_cube_ ) then
            call result_points%add_in_place(bubbles_result_points)
            call bubbles_result_points%destroy()
         end if
    end subroutine

    subroutine Function3DEvaluator_evaluate_grid(self, input_function, result_cube, derivative_x, derivative_y, derivative_z, &
                                                   derivative_cube_x, derivative_cube_y, derivative_cube_z, &
                                                   ignore_cube, ignore_bubbles, ibubs)
        !> evaluator object
        class(Function3DEvaluator),  intent(inout) :: self
        !> input function3d object
        class(Function3D),           intent(in)    :: input_function   
        !> output points object 
        real(REAL64),                intent(inout) :: result_cube(:, :, :)
        !> input function3d objects containing the derivatives of the input_function in x, y, and z directions
        class(Function3D), optional, intent(in)    :: derivative_x, derivative_y, derivative_z  
        !> output points objects for the derivatives of the input_function in x, y, and z directions
        real(REAL64),      optional, intent(inout) :: derivative_cube_x(:, :, :), derivative_cube_y(:, :, :), &
                                                      derivative_cube_z (:, :, :)
        !> If the cube/bubbles are ignored, if not present, nothing is ignored
        logical,           optional, intent(in)    :: ignore_cube, ignore_bubbles       
        !> The global order numbers of evaluated bubbles, if not given all bubbles are evaluated
        integer,           optional, intent(in)    :: ibubs(:)
        logical                                    :: ignore_cube_, ignore_bubbles_
        !> output points objects for the derivatives of the input_function in x, y, and z directions
        real(REAL64),      allocatable             :: bubbles_derivative_cube_x(:, :, :), &
                                                      bubbles_derivative_cube_y(:, :, :), &
                                                      bubbles_derivative_cube_z(:, :, :), &
                                                      bubbles_result_cube(:, :, :)
        
        ignore_cube_ = .FALSE.
        if (present(ignore_cube)) ignore_cube_ = ignore_cube
        ignore_bubbles_ = .FALSE.
        if (present(ignore_bubbles)) ignore_bubbles_ = ignore_bubbles
        
        if (present(derivative_x) .and. present(derivative_y) .and. present(derivative_z)) then
            ! evaluate the points from bubbles
            if (.not. ignore_bubbles_) call self%bubbles_evaluator%evaluate_grid_from_bubbles( &
                                                input_function%bubbles, input_function%grid, &
                                                derivative_x%bubbles, &
                                                derivative_y%bubbles, &
                                                derivative_z%bubbles, &
                                                result_cube, derivative_cube_x, &
                                                derivative_cube_y, derivative_cube_z, &
                                                ibubs = ibubs)

            if (.not. ignore_bubbles_ .and. .not. ignore_cube_ .and. present(derivative_cube_x)) &
                bubbles_derivative_cube_x = derivative_cube_x
            if (.not. ignore_bubbles_ .and. .not. ignore_cube_ .and. present(derivative_cube_y)) &
                bubbles_derivative_cube_y = derivative_cube_y
            if (.not. ignore_bubbles_ .and. .not. ignore_cube_ .and. present(derivative_cube_z)) &
                bubbles_derivative_cube_z = derivative_cube_z
            if (.not. ignore_bubbles_ .and. .not. ignore_cube_) &
                bubbles_result_cube = result_cube
    
            if (.not. ignore_cube_) then
                result_cube(:, :, :)       = input_function%cube (:, :, :)
                derivative_cube_x(:, :, :) = derivative_x%cube(:, :, :)
                derivative_cube_y(:, :, :) = derivative_y%cube(:, :, :)
                derivative_cube_z(:, :, :) = derivative_z%cube(:, :, :)
            end if

        else
            if (.not. ignore_bubbles_) call self%bubbles_evaluator%evaluate_grid(input_function%bubbles, &
                                                input_function%grid, result_cube, derivative_cube_x, &
                                                derivative_cube_y, derivative_cube_z, ibubs = ibubs)
            if (.not. ignore_bubbles_ .and. .not. ignore_cube_ .and. present(derivative_cube_x)) &
                bubbles_derivative_cube_x = derivative_cube_x
            if (.not. ignore_bubbles_ .and. .not. ignore_cube_ .and. present(derivative_cube_y)) &
                bubbles_derivative_cube_y = derivative_cube_y
            if (.not. ignore_bubbles_ .and. .not. ignore_cube_ .and. present(derivative_cube_z)) &
                bubbles_derivative_cube_z = derivative_cube_z
            if (.not. ignore_bubbles_ .and. .not. ignore_cube_) &
                bubbles_result_cube = result_cube

            if (.not. ignore_cube_)    call self%cube_evaluator%evaluate_grid(input_function%cube, input_function%grid, &
                                                result_cube, derivative_cube_x, &
                                                derivative_cube_y, derivative_cube_z)

        end if

        ! add the results from bubbles, if there is need for this
        if (.not. ignore_bubbles_ .and. .not. ignore_cube_ .and. present(derivative_cube_x)) then
            derivative_cube_x(:, :, :) = derivative_cube_x(:, :, :) + bubbles_derivative_cube_x(:, :, :)
            deallocate(bubbles_derivative_cube_x)
        end if
        if (.not. ignore_bubbles_ .and. .not. ignore_cube_ .and. present(derivative_cube_y)) then
            derivative_cube_y(:, :, :) = derivative_cube_y(:, :, :) + bubbles_derivative_cube_y(:, :, :)
            deallocate(bubbles_derivative_cube_y)
        end if
        if (.not. ignore_bubbles_ .and. .not. ignore_cube_ .and. present(derivative_cube_z)) then
            derivative_cube_z(:, :, :) = derivative_cube_z(:, :, :) + bubbles_derivative_cube_z(:, :, :)
            deallocate(bubbles_derivative_cube_z)
        end if
        if (.not. ignore_bubbles_ .and. .not. ignore_cube_) then
            result_cube(:, :, :) = result_cube(:, :, :) + bubbles_result_cube(:, :, :)
            deallocate(bubbles_result_cube)
        end if
    end subroutine

    subroutine Function3DEvaluator_evaluate_divergence_points(self, input_function_x,  &
                                                             input_function_y, &
                                                             input_function_z, &
                                                             divergence_points)
        !> evaluator object
        class(Function3DEvaluator), intent(inout) :: self
        !> input function3d object
        class(Function3D),          intent(in)    :: input_function_x, input_function_y, input_function_z    
        !> Output Points object
        type(Points),               intent(inout) :: divergence_points
        type(Points)                              :: divergence_bubbles_points
        
        divergence_bubbles_points = Points(divergence_points)
        call self%bubbles_evaluator%evaluate_divergence_points(input_function_x%bubbles, &
            input_function_y%bubbles, input_function_z%bubbles, divergence_bubbles_points)
        call self%cube_evaluator%evaluate_divergence_points(input_function_x%cube, &
            input_function_y%cube, input_function_z%cube, input_function_x%grid, divergence_points)
        call divergence_points%add_in_place(divergence_bubbles_points)
        call divergence_bubbles_points%destroy()
    end subroutine

    subroutine Function3DEvaluator_evaluate_divergence_as_Function3D(self, &
                                                             input_function_x,  &
                                                             input_function_y, &
                                                             input_function_z, &
                                                             divergence, &
                                                             ignore_bubbles, ignore_cube)
        !> evaluator object
        class(Function3DEvaluator), intent(inout) :: self
        !> input function3d objects containing the three parts of the input vector
        class(Function3D),          intent(in)    :: input_function_x, input_function_y, input_function_z     
        !> output function3d object containing the divergence of the input vector
        type(Function3D),           intent(out)   :: divergence
        !> If the cube/bubbles are ignored, if not present, nothing is ignored
        logical,           optional, intent(in)   :: ignore_cube, ignore_bubbles  
        !> Temporary pointers to the 
        real(REAL64), pointer                     :: cube_divergence(:)
        type(Bubbles)                             :: temp_x, temp_y, temp_z
        logical                                   :: ignore_cube_, ignore_bubbles_

        ignore_cube_ = .FALSE.
        if (present(ignore_cube)) ignore_cube_ = ignore_cube
        ignore_bubbles_ = .FALSE.
        if (present(ignore_bubbles)) ignore_bubbles_ = ignore_bubbles
  
        
        call divergence%init_copy(input_function_x)
        call divergence%bubbles%destroy()

        if (.not. ignore_cube_) call self%cube_evaluator%evaluate_divergence_grid(input_function_x%cube, &
            input_function_y%cube, input_function_z%cube, input_function_x%grid, divergence%cube)

        if (.not. ignore_bubbles_) call self%bubbles_evaluator%evaluate_divergence_as_bubbles(input_function_x%bubbles, &
            input_function_y%bubbles, input_function_z%bubbles, divergence%bubbles)        


    end subroutine

     subroutine Function3DEvaluator_destroy_stored_objects(self)
        !> CubeEvaluator object
        class(Function3DEvaluator), intent(inout)  :: self

        call self%cube_evaluator%destroy_output_points()
        call self%bubbles_evaluator%destroy_output_points()
        call self%cube_evaluator%destroy_output_cubes()
        call self%bubbles_evaluator%destroy_output_cubes()
        call self%cube_evaluator%destroy_input_cube()
        call self%bubbles_evaluator%destroy_input_bubbles()
    end subroutine

    subroutine Function3DEvaluator_set_output_points(self, result_points, force_reallocate)
        !> CubeEvaluator object
        class(Function3DEvaluator), intent(inout)  :: self
        !> The output grid for the evaluator
        type(Points),               intent(in)     :: result_points
        !> If the reallocation of the poins is forced
        logical,  optional,         intent(in)     :: force_reallocate

        call self%cube_evaluator%set_output_points(result_points, force_reallocate)
        call self%bubbles_evaluator%set_output_points(result_points, force_reallocate)
    end subroutine

    function Function3DEvaluator_get_bubbles_evaluator(self) result(bubbles_evaluator)
        !> evaluator object
        class(Function3DEvaluator), intent(inout), target  :: self
        type(BubblesEvaluator),                    pointer :: bubbles_evaluator 

        bubbles_evaluator => self%bubbles_evaluator
    end function

    function Function3DEvaluator_get_cube_evaluator(self) result(cube_evaluator)
        !> evaluator object
        class(Function3DEvaluator), intent(inout), target  :: self
        type(CubeEvaluator),                       pointer :: cube_evaluator 

        cube_evaluator => self%cube_evaluator
    end function

    subroutine Function3DEvaluator_cuda_init(self)
        !> evaluator object
        class(Function3DEvaluator), intent(inout)  :: self

    end subroutine

    subroutine Function3DEvaluator_cuda_destroy(self)
        !> evaluator object
        class(Function3DEvaluator), intent(inout)  :: self

    end subroutine

    subroutine Function3DEvaluator_destroy(self)
        !> evaluator object
        class(Function3DEvaluator), intent(inout)  :: self
        call self%cube_evaluator%destroy()
        call self%bubbles_evaluator%destroy()
    end subroutine


!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                           CubeEvaluator functions                 %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function CubeEvaluator_init(high_memory_profile) result(new)
        !> If Gradients are evaluated also
        logical,          intent(in)         :: high_memory_profile
        !> The initialized evaluator
        type(CubeEvaluator)                  :: new
        
        new%high_memory_profile = high_memory_profile
#ifdef HAVE_CUDA
        call new%cuda_init()
#endif
    end function


    subroutine CubeEvaluator_evaluate_points_from_cubes(self, input_function_cube, &
                  derivative_x, derivative_y, derivative_z, input_grid, output_points, &
                  output_derivative_x_points, output_derivative_y_points, &
                  output_derivative_z_points)
        !> evaluator object
        class(CubeEvaluator),      intent(inout)    :: self
        !> input cube
        real(REAL64),  target,     intent(in)       :: input_function_cube(:, :, :)
        !> derivative cubes of input cube in x, y, and z directions
        real(REAL64),  target,     intent(in)       :: derivative_x(:, :, :), derivative_y(:, :, :), &
                                                       derivative_z(:, :, :)
        type(Grid3D),  target,     intent(in)       :: input_grid
        
        type(Points),              intent(inout)    :: output_points
        type(Points),              intent(inout)    :: output_derivative_x_points
        type(Points),              intent(inout)    :: output_derivative_y_points
        type(Points),              intent(inout)    :: output_derivative_z_points
#ifdef HAVE_CUDA
        real(REAL64), pointer                       :: cube_pointer(:, :, :)
#else
        real(REAL64), allocatable                   :: temp_array(:, :)      
        type(Grid1DPointer)                         :: grid1d_pointers(3)
#endif                      
        
#ifdef HAVE_CUDA

        call self%set_output_points(output_points)

        
        ! NOTE: downloading from cuda cube is asynchronous, thus the synchronization is required
        ! evaluate the input function cube at points
        call self%set_input_cube(input_function_cube, input_grid)
        call self%cuda_evaluate_points_without_gradients(output_points)
        call CUDASync_all()

        ! evaluate the derivative x cube at points
        call self%set_input_cube(derivative_x, input_grid)
        call self%cuda_evaluate_points_without_gradients(output_derivative_x_points)
        call CUDASync_all()

        ! evaluate the derivative y cube at points
        call self%set_input_cube(derivative_y, input_grid)
        call self%cuda_evaluate_points_without_gradients(output_derivative_y_points)
        call CUDASync_all()

        ! evaluate the derivative z cube at points
        call self%set_input_cube(derivative_z, input_grid)
        call self%cuda_evaluate_points_without_gradients(output_derivative_z_points)
        call CUDASync_all()
        

#else
        grid1d_pointers(X_)%p => input_grid%axis(X_)
        grid1d_pointers(Y_)%p => input_grid%axis(Y_)
        grid1d_pointers(Z_)%p => input_grid%axis(Z_)
        
        self%interpolator = Interpolator(grid1d_pointers)
        output_points%values(:) = reshape(self%interpolator%eval(input_function_cube, &
                                          output_points%point_coordinates%coordinates), &
                                          [size(output_points%point_coordinates%coordinates, 2)])
        output_derivative_x_points%values(:) = reshape(self%interpolator%eval(derivative_x, &
                                                       output_derivative_x_points%point_coordinates%coordinates), &
                                                       [size(output_points%point_coordinates%coordinates, 2)])
        output_derivative_y_points%values(:) = reshape(self%interpolator%eval(derivative_y, &
                                                       output_derivative_y_points%point_coordinates%coordinates), &
                                                       [size(output_points%point_coordinates%coordinates, 2)])
        output_derivative_z_points%values(:) = reshape(self%interpolator%eval(derivative_z, &
                                                       output_derivative_z_points%point_coordinates%coordinates), &
                                                       [size(output_points%point_coordinates%coordinates, 2)])
        call self%interpolator%destroy()
        
#endif
    end subroutine

    subroutine CubeEvaluator_destroy_input_cube(self)
        !> evaluator object
        class(CubeEvaluator),      intent(inout)    :: self
#ifdef HAVE_CUDA
        if (allocated(self%input_cuda_cube)) then
            call self%input_cuda_cube%destroy()
            deallocate(self%input_cuda_cube)
        end if
#endif
    end subroutine

    subroutine CubeEvaluator_set_input_cube(self, input_function_cube, input_grid)
        !> evaluator object
        class(CubeEvaluator),      intent(inout)    :: self
        !> input function3d object
        real(REAL64),              intent(in)       :: input_function_cube(:, :, :)
        !> The grid defining the input cube
        type(Grid3D),   target,    intent(in)       :: input_grid
        logical                                     :: reallocate, all_memory_at_all_devices
#ifdef HAVE_CUDA
        ! check if there is need to reallocate the input_cuda_cube
        reallocate = .not. allocated(self%input_cuda_cube) .or. .not. associated(self%input_grid)
        if (.not. reallocate) then
            if (.not. self%input_grid%is_equal(input_grid)) then
                reallocate = .TRUE.
            end if
        end if

        ! if the
        if (reallocate) then
            ! if there exists an input cuda cube, deallocate all memory related to it
            call self%destroy_input_cube()

            ! reset the grid-pointer
            self%input_grid => input_grid

            ! in order to be able to evaluate random points, we must have all memory at all devices
            all_memory_at_all_devices = .TRUE.
            self%input_cuda_cube = CudaCube(self%input_grid%get_shape(), all_memory_at_all_devices)
    
            call CubeEvaluator_set_input_cube_cuda(self%cuda_interface, self%input_cuda_cube%cuda_interface)
            call CubeEvaluator_set_input_grid_cuda(self%cuda_interface, self%input_grid%get_cuda_interface())
        end if


        ! finally set the host cube and upload it
        call self%input_cuda_cube%set_host(input_function_cube)
        call self%input_cuda_cube%upload()
#endif
    end subroutine

    

    subroutine CubeEvaluator_evaluate_points(self, input_function_cube, &
                                                   input_grid, &
                                                   output_points,  &
                                                   output_derivative_x_points, &
                                                   output_derivative_y_points, &
                                                   output_derivative_z_points)
        !> evaluator object
        class(CubeEvaluator),      intent(inout)    :: self
        !> input cube
        real(REAL64),              intent(in)       :: input_function_cube(:, :, :)
        type(Grid3D),  target,     intent(in)       :: input_grid
        type(Points),              intent(inout)    :: output_points
        type(Points),   optional,  intent(inout)    :: output_derivative_x_points
        type(Points),   optional,  intent(inout)    :: output_derivative_y_points
        type(Points),   optional,  intent(inout)    :: output_derivative_z_points
#ifndef HAVE_CUDA
        real(REAL64), allocatable                   :: temp_array(:, :)    
        type(Grid1DPointer)                         :: grid1d_pointers(3)
#endif                      
        
#ifdef HAVE_CUDA
        
        
        call self%set_input_cube(input_function_cube, input_grid)
        call self%set_output_points(output_points)
        call CUDASync_all()

        if (self%high_memory_profile .and. present(output_derivative_x_points) &
                                    .and. present(output_derivative_y_points) &
                                    .and. present(output_derivative_z_points)) then
            call output_points%set_cuda_interface(self%result_points%get_cuda_interface())
            call output_derivative_x_points%set_cuda_interface(self%output_derivative_x_points%get_cuda_interface())
            call output_derivative_y_points%set_cuda_interface(self%output_derivative_y_points%get_cuda_interface())
            call output_derivative_z_points%set_cuda_interface(self%output_derivative_z_points%get_cuda_interface())

            ! start the evaluation and evaluate gradients
            call Evaluator_evaluate_points_cuda(self%cuda_interface, output_points%get_cuda_interface(), &
                                                output_derivative_x_points%get_cuda_interface(), &
                                                output_derivative_y_points%get_cuda_interface(), &
                                                output_derivative_z_points%get_cuda_interface(), 3)
            ! start downloading the results.
            call output_points%cuda_download()
            call output_derivative_x_points%cuda_download()
            call output_derivative_y_points%cuda_download()
            call output_derivative_z_points%cuda_download()
            call output_points%dereference_cuda_interface()
            call output_derivative_x_points%dereference_cuda_interface()
            call output_derivative_y_points%dereference_cuda_interface()
            call output_derivative_z_points%dereference_cuda_interface()
            call CUDASync_all()
        else
            ! start the evaluation and evaluate gradients one by one
            
            call self%cuda_evaluate_points_without_gradients(output_points)
            call CUDASync_all()

            if (present(output_derivative_x_points)) then 
                call self%cuda_evaluate_points_derivative(X_, output_derivative_x_points)
                call CUDASync_all()
            end if

            if (present(output_derivative_y_points)) then
                call self%cuda_evaluate_points_derivative(Y_, output_derivative_y_points)
                call CUDASync_all()
            end if

            if (present(output_derivative_z_points)) then
                call self%cuda_evaluate_points_derivative(Z_, output_derivative_z_points)
                call CUDASync_all()
            end if     

        end if
        
        call self%input_cuda_cube%unset_host()
#else
        grid1d_pointers(X_)%p => input_grid%axis(X_)
        grid1d_pointers(Y_)%p => input_grid%axis(Y_)
        grid1d_pointers(Z_)%p => input_grid%axis(Z_)

        if (      present(output_derivative_x_points) &
            .and. present(output_derivative_y_points) &
            .and. present(output_derivative_z_points)) then
            self%interpolator = Interpolator(grid1d_pointers, 1)
        else
            self%interpolator = Interpolator(grid1d_pointers)
        end if
        temp_array(:, :) = self%interpolator%eval(input_function_cube, output_points%point_coordinates%coordinates)
        call self%interpolator%destroy()

        output_points%values(:) = temp_array(:, 1)
        if (present(output_derivative_x_points)) then
            output_derivative_x_points%values(:) = temp_array(:, 4)
        end if
        if (present(output_derivative_y_points)) then
            output_derivative_y_points%values(:) = temp_array(:, 3)
        end if
        if (present(output_derivative_z_points)) then
            output_derivative_z_points%values(:) = temp_array(:, 2)
        end if
        deallocate(temp_array)
        
#endif
    end subroutine

    subroutine CubeEvaluator_evaluate_grid(self, input_function_cube, input_grid, output_function_cube, &
                   output_derivative_x_cube, output_derivative_y_cube, output_derivative_z_cube)
        !> evaluator object
        class(CubeEvaluator),      intent(inout)    :: self
        !> input function3d object
        real(REAL64),              intent(in)       :: input_function_cube(:, :, :)
        type(Grid3D),  target,     intent(in)       :: input_grid
        real(REAL64),              intent(inout)    :: output_function_cube(:, :, :)
        real(REAL64), optional,    intent(inout)    :: output_derivative_x_cube(:, :, :)
        real(REAL64), optional,    intent(inout)    :: output_derivative_y_cube(:, :, :)
        real(REAL64), optional,    intent(inout)    :: output_derivative_z_cube(:, :, :)
#ifdef HAVE_CUDA
        real(REAL64), pointer                       :: cube_pointer(:, :, :)
#else
        real(REAL64), allocatable                   :: temp_array(:, :)    
        type(Grid1DPointer)                         :: grid1d_pointers(3)
#endif
        
        ! copy the input function cube to output function cube, as there is no need to 
        ! do any kind of evaluation for this
        output_function_cube(:, :, :) = input_function_cube(:, :, :)

#ifdef HAVE_CUDA
        

        call self%set_input_cube(input_function_cube, input_grid)
        call self%set_output_grid(input_grid)
        call CUDASync_all()

 
        if (self%high_memory_profile .and. present(output_derivative_x_cube) &
                                    .and. present(output_derivative_y_cube) &
                                    .and. present(output_derivative_z_cube)) then
            call self%gradient_cuda_cube_x%set_to_zero()
            call self%gradient_cuda_cube_y%set_to_zero()
            call self%gradient_cuda_cube_z%set_to_zero()
            call CUDASync_all()

            call self%gradient_cuda_cube_x%set_host(output_derivative_x_cube)
            call self%gradient_cuda_cube_y%set_host(output_derivative_y_cube)
            call self%gradient_cuda_cube_z%set_host(output_derivative_z_cube)

            ! start the evaluation and evaluate gradients
            call Evaluator_evaluate_grid_cuda(self%cuda_interface, self%grid%get_cuda_interface(), &
                                              self%result_cuda_cube%cuda_interface, &
                                              self%gradient_cuda_cube_x%cuda_interface, &
                                              self%gradient_cuda_cube_y%cuda_interface, &
                                              self%gradient_cuda_cube_z%cuda_interface, 3)
            ! start downloading the results. NOTE: this is asynchronous,
            ! the cpu-gpu sync will happen when 'get_results' or 'get_gradients'
            ! is called.
            call self%gradient_cuda_cube_x%download()
            call self%gradient_cuda_cube_y%download()
            call self%gradient_cuda_cube_z%download()
            call CUDASync_all()
        else
            if (present(output_derivative_x_cube)) then
                call self%cuda_evaluate_grid_derivative(X_, output_derivative_x_cube)
                call CUDASync_all()
            end if

            if (present(output_derivative_y_cube)) then
                call self%cuda_evaluate_grid_derivative(Y_, output_derivative_y_cube)
                call CUDASync_all()
            end if

            if  (present(output_derivative_z_cube)) then
                call self%cuda_evaluate_grid_derivative(Z_, output_derivative_z_cube)
                call CUDASync_all()    
            end if
        end if

#else
        

        if (     present(output_derivative_x_cube) &
            .or. present(output_derivative_y_cube) &
            .or. present(output_derivative_z_cube)) then
            grid1d_pointers(X_)%p => input_grid%axis(X_)
            grid1d_pointers(Y_)%p => input_grid%axis(Y_)
            grid1d_pointers(Z_)%p => input_grid%axis(Z_)
            self%interpolator = Interpolator(grid1d_pointers, 1)

            allocate(temp_array(product(input_grid%get_shape()), 4)) 
            temp_array(:, :) = self%interpolator%eval(input_function_cube, input_grid%get_all_grid_points())
            if (present(output_derivative_x_cube)) output_derivative_x_cube(:, :, :) = &
                                                       reshape(temp_array(:, 4), shape(output_derivative_x_cube))
            if (present(output_derivative_y_cube)) output_derivative_y_cube(:, :, :) = &
                                                       reshape(temp_array(:, 3), shape(output_derivative_y_cube))
            if (present(output_derivative_z_cube)) output_derivative_z_cube(:, :, :) = &
                                                       reshape(temp_array(:, 2), shape(output_derivative_z_cube))
            call self%interpolator%destroy()
            deallocate(temp_array)
        end if
        
        
#endif
    end subroutine

    subroutine CubeEvaluator_evaluate_divergence_grid(self, input_function_cube_x, &
                                  input_function_cube_y, input_function_cube_z, &
                                  input_grid, output_cube)
        !> evaluator object
        class(CubeEvaluator),      intent(inout)    :: self
        !> input cubes in x, y and z direction
        real(REAL64),              intent(in)       :: input_function_cube_x(:, :, :), & 
                                                       input_function_cube_y(:, :, :), &
                                                       input_function_cube_z(:, :, :)
        type(Grid3D),  target,     intent(in)       :: input_grid
        real(REAL64),              intent(inout)    :: output_cube(:, :, :)
#ifndef HAVE_CUDA
        real(REAL64), allocatable                   :: temp_array(:, :)    
        type(Grid1DPointer)                         :: grid1d_pointers(3)
#endif                      
        
#ifdef HAVE_CUDA
        call self%set_output_grid(input_grid)
        call self%result_cuda_cube%set_host(output_cube)
        call self%result_cuda_cube%set_to_zero()

        ! upload the data needed to evaluated gradients in x direction
        call self%set_input_cube(input_function_cube_x, input_grid)
        call CUDASync_all()

        ! start the evaluation of gradients in x-direction
        call Evaluator_evaluate_grid_x_gradients_cuda(self%cuda_interface, &
                 self%grid%get_cuda_interface(), self%result_cuda_cube%cuda_interface)
        call CUDASync_all()

        ! upload the data needed to evaluated gradients in y direction
        call self%set_input_cube(input_function_cube_y, input_grid)
        call CUDASync_all()

        ! start the evaluation of gradients in y-direction
        call Evaluator_evaluate_grid_y_gradients_cuda(self%cuda_interface, &
                 self%grid%get_cuda_interface(), self%result_cuda_cube%cuda_interface)
        call CUDASync_all()

        ! upload the data needed to evaluated gradients in z direction
        call self%set_input_cube(input_function_cube_z, input_grid)
        call CUDASync_all()
        
        ! start the evaluation of gradients in z-direction
        call Evaluator_evaluate_grid_z_gradients_cuda(self%cuda_interface, &
                 self%grid%get_cuda_interface(), self%result_cuda_cube%cuda_interface)
        call CUDASync_all()

        ! start downloading the results. NOTE: this is asynchronous,
        ! the cpu-gpu sync will happen when 'get_results' or 'get_gradients'
        ! is called.
        call self%result_cuda_cube%download()
        call self%input_cuda_cube%unset_host()
#else
        
        grid1d_pointers(X_)%p => input_grid%axis(X_)
        grid1d_pointers(Y_)%p => input_grid%axis(Y_)
        grid1d_pointers(Z_)%p => input_grid%axis(Z_)
        self%interpolator = Interpolator(grid1d_pointers, 1)

        allocate(temp_array(product(input_grid%get_shape()), 4)) 
        temp_array(:, :)     = self%interpolator%eval(input_function_cube_x, input_grid%get_all_grid_points())
        output_cube(:, :, :) = reshape(temp_array(:, 4), shape(input_function_cube_x))
        temp_array(:, :)     = self%interpolator%eval(input_function_cube_y, input_grid%get_all_grid_points())
        output_cube(:, :, :) = output_cube(:, :, :) + reshape(temp_array(:, 3), shape(input_function_cube_x))
        temp_array(:, :)     = self%interpolator%eval(input_function_cube_z, input_grid%get_all_grid_points())
        output_cube(:, :, :) = output_cube(:, :, :) + reshape(temp_array(:, 2), shape(input_function_cube_x))
        call self%interpolator%destroy()
        deallocate(temp_array)
        
#endif
    end subroutine

    subroutine CubeEvaluator_evaluate_divergence_points(self, input_function_cube_x, &
                                 input_function_cube_y, input_function_cube_z, input_grid, output_points)
        !> evaluator object
        class(CubeEvaluator),      intent(inout)    :: self
        !> input cubes in x, y and z direction
        real(REAL64),              intent(in)       :: input_function_cube_x(:, :, :), & 
                                                       input_function_cube_y(:, :, :), &
                                                       input_function_cube_z(:, :, :)
        type(Grid3D),  target,     intent(in)       :: input_grid
        type(Points),              intent(inout)    :: output_points
#ifdef HAVE_CUDA
        real(REAL64), pointer                       :: cube_pointer(:, :, :)
#else
        real(REAL64), allocatable                   :: temp_array(:, :)    
        type(Grid1DPointer)                         :: grid1d_pointers(3)
#endif                      
        
#ifdef HAVE_CUDA
        call self%set_output_points(output_points)
        call output_points%set_cuda_interface(self%result_points%get_cuda_interface())

        ! upload the data needed to evaluated gradients in x direction
        call self%set_input_cube(input_function_cube_x, input_grid)
        call CUDASync_all()

        ! start the evaluation of gradients in x-direction
        call Evaluator_evaluate_points_x_gradients_cuda(self%cuda_interface, output_points%get_cuda_interface())
        call CUDASync_all()

        ! upload the data needed to evaluated gradients in y direction
        call self%set_input_cube(input_function_cube_y, input_grid)
        call CUDASync_all()

        ! start the evaluation of gradients in y-direction
        call Evaluator_evaluate_points_y_gradients_cuda(self%cuda_interface, output_points%get_cuda_interface())
        call CUDASync_all()

        ! upload the data needed to evaluated gradients in z direction
        call self%set_input_cube(input_function_cube_z, input_grid)
        call CUDASync_all()
        
        ! start the evaluation of gradients in z-direction
        call Evaluator_evaluate_points_z_gradients_cuda(self%cuda_interface, output_points%get_cuda_interface())
        call CUDASync_all()

        ! start downloading the results. NOTE: this is asynchronous,
        ! the cpu-gpu sync will happen when 'get_results' or 'get_gradients'
        ! is called.
        call output_points%cuda_download()
        call output_points%dereference_cuda_interface()
        call CUDASync_all()
        call self%input_cuda_cube%unset_host()
#else
        
        grid1d_pointers(X_)%p => input_grid%axis(X_)
        grid1d_pointers(Y_)%p => input_grid%axis(Y_)
        grid1d_pointers(Z_)%p => input_grid%axis(Z_)
        self%interpolator = Interpolator(grid1d_pointers, 1)

        allocate(temp_array(product(input_grid%get_shape()), 4))
        temp_array(:, :) = self%interpolator%eval(input_function_cube_x, output_points%point_coordinates%coordinates)
        output_points%values(:) = temp_array(:, 4)
        temp_array(:, :) = self%interpolator%eval(input_function_cube_y, output_points%point_coordinates%coordinates)
        output_points%values(:) = output_points%values(:) + temp_array(:, 3)
        temp_array(:, :) = self%interpolator%eval(input_function_cube_z, output_points%point_coordinates%coordinates)
        output_points%values(:) = output_points%values(:) + temp_array(:, 2)
        call self%interpolator%destroy()
        deallocate(temp_array)
#endif
    end subroutine

    subroutine CubeEvaluator_destroy(self)
        !> evaluator object
        class(CubeEvaluator), intent(inout)  :: self
        
        call Evaluator_destroy(self)
#ifdef HAVE_CUDA
        call self%cuda_destroy()
#endif 
        nullify(self%grid)
    end subroutine





#ifdef HAVE_CUDA


    subroutine CubeEvaluator_cuda_init(self)
        !> evaluator object
        class(CubeEvaluator), intent(inout)  :: self
        integer                              :: high_memory_profile_int
        logical                              :: all_memory_at_all_devices
        type(C_PTR)                          :: gradient_cuda_cube_x, gradient_cuda_cube_y, &
                                                gradient_cuda_cube_z, c_pointer  
        

        high_memory_profile_int = 0
        if (self%high_memory_profile) high_memory_profile_int = 1

        ! call the cuda init for the non-arranged points 
        self%cuda_interface = CubeEvaluator_init_cuda( &
            stream_container)

    end subroutine

    subroutine CubeEvaluator_cuda_destroy(self)
        !> evaluator object
        class(CubeEvaluator), intent(inout)  :: self


        call self%destroy_input_cube()

    end subroutine
#endif

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                           Operator3d functions                          %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    ! interface to the method below to allow the operator to be used
    function Operator3D_operate_on(self, func) result(new)
        !> Input function
        class(Operator3D), intent(in) :: self
        !> Operator
        class(Function3D), intent(in) :: func
        
        class(Function3D), allocatable    :: new
        
        call self%operate_on(func, new)


        !select type(temp)
        !    type is (Function3D)
        !        new = temp
        !    class default
        !        new = temp
        !end select
        !allocate(new, source = temp)
        !call temp%destroy()
        !deallocate(temp)
     
        
    end function

    pure subroutine Operator3D_set_transformation_weights(self, weights) 
        !> Input operator
        class(Operator3D), intent(inout) :: self
        real(REAL64),      intent(in)    :: weights(:)
        self%w = weights
    end subroutine 

   !> Return a new function resulting from the application of operator op onto
    !! an input function. 
    !
    ! Note, modified to include partial application of operator3d.
    ! Because of this, operator .apply. points to the functions above.
    subroutine Operator3D_operate_on_Function3D(self, func, new, cell_limits, only_cube)
        !> Operator
        class(Operator3D)              :: self
        !> Input function
        class(Function3D), intent(in)  :: func
        !> Result function
        class(Function3D), allocatable, intent(inout), target :: new
        !> Limits in the area in which the operator is applied
        integer, intent(in), optional :: cell_limits(2, 3)
        !> if only cube is taken into account
        logical, intent(in), optional :: only_cube
        integer                       :: function_type
        logical                       :: operate_bubbles


        type(Bubbles)                 :: result_bubbles
        !call pinfo("Applying operator")
        
        ! allocate the default function as Function3D
        if (.not. allocated(new)) then
            allocate(Function3D :: new)
            ! init the way the new object is parallelized, use the input functions parallelization
            ! if nothing else has been specified at 'self%result_parallelization_info'
            
            call new%init_explicit(self%result_parallelization_info, &
                            label=trim(func%label)//".copy" )
            call new%set_type(self%get_result_type())
        else 
            new%cube = 0.0d0
            call new%bubbles%destroy()
            if (allocated(new%cube_contaminants)) deallocate(new%cube_contaminants)
            
   
            if (allocated(new%taylor_series_bubbles)) then
                call new%taylor_series_bubbles%destroy()
                deallocate(new%taylor_series_bubbles)
            end if
        end if
        
        if (present(only_cube)) then
            if (.not. only_cube) then 
                operate_bubbles = .TRUE.
            else
                operate_bubbles = .FALSE.
            end if
        else 
            operate_bubbles = .TRUE.
        end if 
    
        call self%operator_bubbles(func, new, operate_bubbles)

        if (present(cell_limits)) then
            call self%operator_cube(func, new, cell_limits)
        else
            call self%operator_cube(func, new)
        end if

        if (operate_bubbles) then
            call new%parallelization_info%communicate_bubbles(new%bubbles)
            call new%precalculate_taylor_series_bubbles()
        end if
        
        if(debug_g>0) then
            call new%bubbles%print(new%get_label())
        end if
    end subroutine

    subroutine Operator3D_operator_cube(self, func, new, cell_limits)
        !> Operator
        class(Operator3D)                 :: self
        !> Input function
        class(Function3D), intent(in)     :: func
        !> Result function
        class(Function3D), allocatable, intent(inout), target :: new
        !> Limits in the area in which the operator is applied
        integer, intent(in), optional     :: cell_limits(2, 3)
        ! Limits of the operated area in grid points
        integer                           :: cube_limits(2, 3)

        ! Cube pointer
        real(REAL64), pointer             :: all_cube(:, :, :)
        real(REAL64), allocatable         :: cube(:, :, :), temp_cube(:, :, :)
#ifdef HAVE_CUDA
        type(CUDACube)                    :: tmp1, tmp2
        integer                           :: dims(X_:Z_, IN_:OUT_) ! (3, 2) array
#endif

        ! get the cube which is operated, in case no limits are specified
        ! the entire cube is operated      
        if (present(cell_limits)) then
            ! each cell has nlip-1 grid points
            cube_limits(1, :) = (cell_limits(1, :) - [1, 1, 1]) * (self%gridout%get_nlip() - 1) + 1
            cube_limits(2, :) = (cell_limits(2, :)) * (self%gridout%get_nlip() - 1) + 1
            all_cube => func%get_cube()
            cube = all_cube(cube_limits(1, X_) : cube_limits(2, X_), &
                            cube_limits(1, Y_) : cube_limits(2, Y_), &
                            cube_limits(1, Z_) : cube_limits(2, Z_))
        else 
            cube = func%get_cube()
        end if
        ! get the (limited) cube of the function3d and transform it and assign 
        ! it as the result function's cube
        
        !call bigben%split("Operator cube")
#ifdef HAVE_CUDA
        if (self%cuda_inited .and. .not. self%suboperator) then
            new%cube = 0.0d0
            ! set the cuda cube host cubes
            call self%output_cuda_cube%set_host(new%cube)
            call self%input_cuda_cube%set_host(cube)

            ! upload the cuda cube data to gpu
            call self%output_cuda_cube%upload()
            call self%input_cuda_cube%upload()
            call CUDASync_all()

            ! dims: number of grid points per each axis 
            dims=self%get_dims()
            
            ! create temporary cubes residing only at GPU
            tmp1 = CUDACube([dims(X_,OUT_),dims(Y_,IN_),dims(Z_,IN_)])
            tmp2 = CUDACube([dims(X_,OUT_),dims(Y_,OUT_),dims(Z_,IN_)])

            ! transform the data
            call self%transform_cuda_cube(self%input_cuda_cube, self%output_cuda_cube, &
                                          self%cuda_blas, self%cuda_fx, self%cuda_fy, self%cuda_fz, &
                                          tmp1, tmp2)

            call tmp1%destroy()
            call tmp2%destroy()
            
            call CUDASync_all()
            ! and download the output data
            call self%output_cuda_cube%download()
            call CUDASync_all()

            ! unset the the input and output host cubes
            call self%output_cuda_cube%unset_host()
            call self%input_cuda_cube%unset_host()
            call CUDASync_all()
        else
            temp_cube = self%transform_cube(cube)
            new%cube = temp_cube
            deallocate(temp_cube)
        end if
#else
        temp_cube = self%transform_cube(cube)
        new%cube = temp_cube
        deallocate(temp_cube)
#endif
        deallocate(cube)
        nullify(all_cube)
        !call bigben%stop()
    end subroutine

    subroutine Operator3D_operator_bubbles(self, func, new, operate_bubbles)
        !> Operator
        class(Operator3D)              :: self
        !> Input function
        class(Function3D), intent(in)  :: func
        !> Result function
        class(Function3D), allocatable, intent(inout), target :: new
        !> if only cube is taken into account
        logical, intent(in)            :: operate_bubbles
        type(Bubbles)                  :: temp
        ! operate on bubbles


        if (operate_bubbles) then
            !call bigben%split("Operator bubbles")
            
            temp = self%transform_bubbles(func%bubbles)
            new%bubbles = Bubbles(temp, copy_content = .TRUE.)
            call temp%destroy()
            !call bigben%stop()
        end if
        
    end subroutine

    

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
    subroutine Operator3D_destroy(self)
        class(Operator3D), intent(inout) :: self

#ifdef HAVE_CUDA
        call self%cuda_destroy()
#endif
        if (allocated(self%f(X_)%p)) deallocate(self%f(X_)%p)
        if (allocated(self%f(Y_)%p)) deallocate(self%f(Y_)%p)
        if (allocated(self%f(Z_)%p)) deallocate(self%f(Z_)%p)

        if (allocated(self%w))  deallocate(self%w)
        nullify(self%result_parallelization_info)
        nullify(self%gridin)
        nullify(self%gridout)
    end subroutine

    


!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                           Projector3d functions                         %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    !> Constructor for projector operator
    function Projector3D_init(gridin, gridout, bubble_grids_in, bubble_grids_out) result(new)
        type(Projector3D)                           :: new
        !> Grid3D of input functions
        type(Grid3D), intent(in), target            :: gridin
        !> Grid3D on which the output functions are constructed. If not given,
        !! construct on the input grid
        type(Grid3D), intent(in), target, optional  :: gridout
        !> Array of Grid1D functions used in the interpolation of the bubbles.
        !! If not given a copy of bubbles will be made instead of projection.
        type(Grid1DPointer), intent(in), optional  :: bubble_grids_in(:)
        !> Array of Grid1D functions from which the bubbles of output functions are constructed.
        !! If not given a copy of bubbles will be made instead of projection.
        type(Grid1DPointer), intent(in), optional  :: bubble_grids_out(:)

        type(Grid3D), pointer                       :: gridout_p
        integer                                     :: i

        if(present(gridout)) then
            gridout_p=>gridout
        else
            gridout_p=>gridin
        end if

        if (present(bubble_grids_in)) then
            new%bubble_grids_in = bubble_grids_in
            ! allocate and init the interpolators for the bubbles
            allocate(new%interpolators(size(new%bubble_grids_in)))
            do i = 1, size(new%bubble_grids_in)
                new%interpolators(i) = SimpleInterpolator1D(new%bubble_grids_in(i)%p)
            end do
            if (present(bubble_grids_out)) then
                new%bubble_grids_out = bubble_grids_out
            else
                new%bubble_grids_out = bubble_grids_in
            end if
        end if

        new%gridin  => gridin
        new%gridout => gridout_p
        new%w       = [1.d0]
        new%f=alloc_transformation_matrices(new%gridin, new%gridout, 1)

        new%f(X_)%p(:,:,1) = interp_matrix(new%gridin, new%gridout, X_)
        new%f(Y_)%p(:,:,1) = interp_matrix(new%gridin, new%gridout, Y_, trsp='t')
        new%f(Z_)%p(:,:,1) = interp_matrix(new%gridin, new%gridout, Z_, trsp='t')
        

#ifdef HAVE_CUDA
        new%cuda_inited = .FALSE.
#endif
    end function

    !> Destructor
    subroutine Projector3D_destroy(self)
        class(Projector3D), intent(inout) :: self
        integer                           :: i

#ifdef HAVE_CUDA
        call self%cuda_destroy()
#endif
        if (allocated(self%f(X_)%p)) deallocate(self%f(X_)%p)
        if (allocated(self%f(Y_)%p)) deallocate(self%f(Y_)%p)
        if (allocated(self%f(Z_)%p)) deallocate(self%f(Z_)%p)

        if (allocated(self%w))  deallocate(self%w)
        nullify(self%result_parallelization_info)
            !if (allocated(self%gridin)) then
        nullify(self%gridin)
        nullify(self%gridout)


        if (allocated(self%bubble_grids_in)) then
            deallocate(self%bubble_grids_in)
        end if

        if (allocated(self%bubble_grids_out)) then
            deallocate(self%bubble_grids_out)
        end if

    end subroutine




    !> Projects the cube of the func to the output grid and stores it
    !! to new.
    subroutine Projector3D_operator_cube(self, func, new, cell_limits)
        !> Operator
        class(Projector3D)                 :: self
        !> Input function
        class(Function3D), intent(in)     :: func
        !> Result function
        class(Function3D), allocatable, intent(inout), target :: new
        !> Limits in the area in which the operator is applied
        integer, intent(in), optional     :: cell_limits(2, 3)

        ! check if the gridout equals or is similar with the input grid,
        ! if this is the case, just copy the content
        if (self%gridout%is_equal(self%gridin) .or. self%gridout%is_similar(self%gridin)) then
            new%cube(:, :, :) = func%cube(:, :, :) 
        else
            call Operator3D_operator_cube(self, func, new, cell_limits)
        end if
    end subroutine

    !> Bubbles transformation operation for `Projector3D`: return a copy
    !! of bubsin.
    function Projector3D_transform_bubbles(self, bubsin) result(new)
        class(Projector3D), intent(in) :: self
        type(Bubbles), intent(in)     :: bubsin
        type(Bubbles)                 :: new
        call self%transform_bubbles_sub(bubsin, new)
        return
    end function

    !> Bubbles transformation operation for `Projector3D`: return a copy
    !! of bubsin.
    subroutine Projector3D_transform_bubbles_sub(self, bubsin, bubsout)
        class(Projector3D), intent(in) :: self
        type(Bubbles), intent(in)      :: bubsin
        type(Bubbles), intent(inout)   :: bubsout
        logical                        :: do_copy
        integer                        :: ibub, i, order_number
        type(Grid1DPointer)            :: grids(bubsin%get_nbub())

        do i = 1, bubsin%get_nbub()
            grids(i) = self%bubble_grids_out(bubsin%get_ibub(i))
        end do
        

        ! do only copy of the bubbles, if all the grids are equal or similar
        do_copy = all([(bubsin%gr(ibub)%p%is_similar(self%bubble_grids_out(ibub)%p), &
                        ibub = 1, bubsin%get_nbub())]) .or. .not. allocated(self%interpolators)

        
        bubsout = Bubbles(bubsin%get_lmax(), bubsin%get_centers(), bubsin%global_centers, &
                          grids, &
                          self%bubble_grids_out, &
                          bubsin%get_z(), &
                          bubsin%get_global_z(), &
                          bubsin%get_k(), bubsin%get_ibubs(), &
                          bubsin%get_nbub_global())

        if (do_copy) then
            call bubsout%copy_content(bubsin)
        else
            do i = 1, bubsout%get_nbub()
                ! get the global 
                ibub = bubsout%get_ibub(i)

                ! get order number of ibub in bubsin
                if (bubsin%contains_ibub(ibub)) then
                    order_number = bubsin%get_order_number(ibub)
                    bubsout%bf(i)%p(:, :) = &
                                self%interpolators(ibub)%eval(bubsin%bf(order_number)%p, &
                                                        bubsout%gr(i)%p%get_coord())
                else
                    print *, "ERROR: Projector input bubbles does not contain the output bubbles. &
                             &(@Projector3D_transform_bubbles_sub)."
                    stop
                end if
            end do
        end if
        return
    end subroutine

    ! Wrapper(s) to operator calls

    !> Projects the cube of a function to the grid of another function.
    !!
    !! The `res` function is identical to source except for its cube. In
    !! case, `source` and `mould` share the same grid, the cube of `res`
    !! is a fresh copy of source.
    function cube_project(cubein, gridin, gridout) result(cubeout)
        !> Input cube
        real(REAL64), intent(in)   :: cubein(:,:,:)
        !> Grid of `cubein`
        class(Grid3D), intent(in)  :: gridin
        !> Grid onto which `cubein` is to be projected
        class(Grid3D), intent(in)  :: gridout

        real(REAL64), allocatable  :: cubeout(:,:,:)
        real(REAL64)               :: qmin(3), qmax(3)

        type(Projector3D)           :: projector
        integer                     :: cube_limits(2, 3)
        integer                     :: dims_out(3)

        call bigben%split("Cube project")
        !call pdebug("function_project()", 1)
        if(gridin%is_equal(gridout) .or. gridin%is_similar(gridout)) then
            ! Source and mould share the same grid
            ! Copy the cube of source
            cubeout=cubein
        else if (gridin%is_subgrid_of(gridout)) then 
            ! get the cube coordinates of the input cube at the output cube
            dims_out = gridout%get_shape()
            ! get start and end coordinates of gridin
            qmin = gridin%get_qmin()
            qmax = gridin%get_qmax()

            ! get the coordinates of the start and end coordinates of gridin in the 'gridout'
            cube_limits(1, :) = (gridout%get_icell(qmin) - [1, 1, 1]) * (gridout%get_nlip()-1) + [1, 1, 1]
            cube_limits(2, :) = cube_limits(1, :) - [1, 1, 1]  + gridin%get_shape() 
            
            allocate(cubeout(dims_out(X_), dims_out(Y_), dims_out(Z_)), source = 0.0d0)
            cubeout(cube_limits(1, X_) : cube_limits(2, X_),  &
                    cube_limits(1, Y_) : cube_limits(2, Y_),  &
                    cube_limits(1, Z_) : cube_limits(2, Z_))  &
                      = cubein                                                       
        else
            projector = Projector3D(gridin, gridout)
            cubeout= projector%transform_cube( cubein )
            call projector%destroy()
        end if
        call bigben%stop()
    end function

#ifdef HAVE_CUDA
    subroutine Operator3D_cuda_init(self)
        class(Operator3D), intent(inout) :: self
        integer                          :: dims(X_:Z_, IN_:OUT_) ! (3, 2) array
        integer                          :: stream_pointer_count, t_points_per_stream
        type(C_PTR)                      :: container

        if (.not. self%cuda_inited) then
            ! dims: number of grid points per each axis 
            dims=self%get_dims()
            
            if (allocated(self%stream_container)) then
                container = self%stream_container
            else
                container = stream_container
            end if 
            
            t_points_per_stream = size(self%f(X_)%p, 3) / &
                (StreamContainer_get_number_of_devices(container) * STREAMS_PER_DEVICE)
            if (t_points_per_stream == 0) t_points_per_stream = 1
            stream_pointer_count = (t_points_per_stream+1) * (dims(Z_,IN_) + dims(Z_,IN_) + dims(Y_,OUT_))
            self%cuda_blas = CUDABlas(stream_pointer_count, container)


            
            ! init the input & output cuda cubes
            if (.NOT. self%suboperator) then
                self%input_cuda_cube = CudaCube(self%gridin%get_shape(), container = container)
                self%output_cuda_cube = CudaCube(self%gridout%get_shape(), container = container)
            end if

            self%cuda_fx = CUDACube(self%f(X_)%p, container = container)
            self%cuda_fy = CUDACube(self%f(Y_)%p, container = container)
            self%cuda_fz = CUDACube(self%f(Z_)%p, container = container)

            call self%cuda_fx%upload()
            call self%cuda_fy%upload()
            call self%cuda_fz%upload()
            self%cuda_inited = .TRUE.
        end if
    end subroutine

    subroutine Operator3D_cuda_destroy(self)
        class(Operator3D), intent(inout) :: self
        
        if (self%cuda_inited) then
            call self%cuda_fx%destroy()
            call self%cuda_fy%destroy()
            call self%cuda_fz%destroy()
            call self%cuda_tmp1%destroy()
            call self%cuda_tmp2%destroy()
            call self%input_cuda_cube%destroy()
            call self%output_cuda_cube%destroy()
            call self%cuda_blas%destroy()
            self%cuda_inited = .FALSE.
        end if
    end subroutine
#endif

    !> Transform cube (matmul tandem) and bubbles
    recursive function Operator3D_transform_cube(self, cubein) result(cubeout)
        class(Operator3D), intent(in) :: self
        !> Input cube
        real(REAL64), intent(in)      :: cubein(:,:,:)

        real(REAL64), allocatable     :: cubeout(:,:,:)

        integer                       :: iz, iy, ip, stream_pointer_count, t_points_per_stream
        integer                       :: dims(X_:Z_, IN_:OUT_) ! (3, 2) array
        character(22)                 :: label

#ifdef HAVE_CUDA
        type(CUDACube)                :: CUcubein, CUcubeout,&
                                         CUfx, CUfy, CUfz, tmp1, tmp2
        type(CUDAMatrix)              :: slice1, slice2, slice3
        type(CUDABlas)                :: cuda_blas
#else
        real(REAL64), allocatable     :: tmp1(:,:)
        real(REAL64), allocatable     :: tmp_u(:,:,:)
        
#endif
        

        ! dims: number of grid points per each axis 
        dims=self%get_dims()
        ! allocate result matrix
        allocate(cubeout( dims(X_, OUT_), dims(Y_, OUT_), dims(Z_, OUT_) ), source = 0.0d0 )
        write(label, '("Operator cube (R=",i0,")")') size(self%w)
        call bigben%split(label)

        if (verbo_g>0) call progress_bar(0,size(self%w))

        call bigben%split("t-pointsf")
        ! go through all t-points
#ifdef HAVE_CUDA
        !call pdebug("Using CUDA", 1)
        call bigben%split("CuBLAS initialization")
        t_points_per_stream = size(self%f(X_)%p, 3) / &
            (StreamContainer_get_number_of_devices(stream_container) * STREAMS_PER_DEVICE)
        if (t_points_per_stream == 0) t_points_per_stream = 1
        stream_pointer_count = (t_points_per_stream+1) * (dims(Z_,IN_) + dims(Z_,IN_) + dims(Y_,OUT_))
        cuda_blas = CUDABlas(stream_pointer_count)
        call bigben%stop()

        call bigben%split("CUDA alloc. and upload")
        CUcubein = CUDACube(cubein)
        CUcubeout = CUDACube(cubeout)
        CUfx = CUDACube(self%f(X_)%p)
        CUfy = CUDACube(self%f(Y_)%p)
        CUfz = CUDACube(self%f(Z_)%p)

        ! upload data to GPU device
        call CUcubein%upload()
        call CUfx%upload()
        call CUfy%upload()
        call CUfz%upload()

        
        ! create temporary cubes residing only at GPU
        tmp1 = CUDACube([dims(X_,OUT_),dims(Y_,IN_),dims(Z_,IN_)])
        tmp2 = CUDACube([dims(X_,OUT_),dims(Y_,OUT_),dims(Z_,IN_)])

        call self%transform_cuda_cube(CUcubein, CUcubeout, cuda_blas, CUfx, CUfy, CUfz, tmp1, tmp2)
        ! create temporary cubes residing only at GPU
        call tmp1%destroy()
        call tmp2%destroy()
        call bigben%stop()
#else
        
        allocate(tmp1(    dims(X_, OUT_), dims(Y_, IN_) ) )
        allocate(tmp_u(   dims(X_, OUT_), dims(Y_, OUT_), dims(Z_, IN_) ) )

        iploop: do ip=1,size(self%w)
            ! go through all xy slices in the  cube      
            do iz=1,dims(Z_,IN_)
                ! multiply tranformation matrices 2d-slice with the corresponding input cube
                tmp1=xmatmul(self%f(X_)%p(:,:,ip),cubein(:,:,iz))
                tmp_u(:,:,iz)=xmatmul(tmp1,self%f(Y_)%p(:,:,ip))
            end do
            do iy=1,dims(Y_,OUT_)
                
                cubeout(:,iy,:) = cubeout(:,iy,:) + &
                        self%w(ip)*xmatmul(tmp_u(:,iy,:),self%f(Z_)%p(:,:,ip))
            end do
            if (verbo_g>0) call progress_bar(ip)
        end do iploop
#endif
            
        call bigben%stop()

#ifdef HAVE_CUDA
! Download cube
        call bigben%split("CUDA download")
        call CUcubeout%download()
        call bigben%stop()
#endif

        ! This could be done using CUDA. Worth it?
        if (self%coda /= 0.d0) then
            call bigben%split("Coda")
            cubeout=cubeout + &
                self%coda * cube_project(cubein, self%gridin, self%gridout)
            call bigben%stop()
        end if

#ifdef HAVE_CUDA
        ! Finalize CUDA objects
        call bigben%split("CUDA destroy")
        call CUcubein%destroy()
        call CUcubeout%destroy()
        call CUfx%destroy()
        call CUfy%destroy()
        call CUfz%destroy()
        call cuda_blas%destroy()
        call bigben%stop()
#else
        deallocate(tmp1)
        deallocate(tmp_u)
#endif
        call bigben%stop()
    end function


#ifdef HAVE_CUDA

    recursive subroutine Operator3D_transform_cuda_cube(self, cubein, cubeout, cuda_blas, &
                                                            cuda_fx, cuda_fy, cuda_fz, tmp1, tmp2)
        class(Operator3D),  intent(in), target  :: self
        type(CUDACube),   intent(in)      :: cubein
        type(CUDACube),   intent(inout)   :: cubeout
        type(CUDABlas),   intent(in)      :: cuda_blas
        type(CUDACube),   intent(in)      :: cuda_fx, cuda_fy, cuda_fz
        type(CUDACube),   intent(inout)   :: tmp1, tmp2
        type(CUDAMatrix)                  :: slice1, slice2, slice3
        integer                           :: iz, iy, ip
        integer                           :: dims(X_:Z_, IN_:OUT_) ! (3, 2) array
        integer                           :: number_of_devices, device, streams_per_device
        type(C_PTR), allocatable          :: waited_events(:)

        character(22)                     :: label

        dims = self%get_dims()
        ! create temporary cubes residing only at GPU
        ! get number of devices used in this blas
        number_of_devices = cuda_blas%get_number_of_devices()
        streams_per_device = cuda_blas%get_streams_per_device()
        
        ! if the output matrix is large enough, use the stream based approach
        if (dims(X_, OUT_) + dims(Y_, OUT_) + dims(Z_, OUT_) > 600) then
            do ip=1, size(self%w), number_of_devices 
                do device = 1, min(number_of_devices, size(self%w)-ip+1)
                    do iz=1,dims(Z_,IN_)
                        slice1 = cuda_fx%slice(Z_,ip+device-1)
                        slice2 = cubein%slice(Z_,iz)
                        slice3 = tmp1%slice(Z_,iz)
                        call cuda_blas%mm_multiplication(device, iz,&
                                slice1,&
                                slice2,&
                                slice3,&
                                1.d0, 0.d0)
                        call slice1%destroy()
                        call slice2%destroy()
                        call slice3%destroy()
                    end do
                    do iz=1,dims(Z_,IN_)
                        slice1 = tmp1%slice(Z_,iz)
                        slice2 = cuda_fy%slice(Z_,ip+device-1)
                        slice3 = tmp2%slice(Z_,iz)
                        call cuda_blas%mm_multiplication(device, iz,&
                                slice1,&
                                slice2,&
                                slice3,1.d0,0.d0)
                        call slice1%destroy()
                        call slice2%destroy()
                        call slice3%destroy()
                    end do
                end do
                if (streams_per_device > 1) call CUDASync_all()
                do device = 1, min(number_of_devices, size(self%w)-ip+1)
                    do iy=1,dims(Y_,OUT_)
                        slice1 = tmp2%slice(Y_,iy)
                        slice2 = cuda_fz%slice(Z_,ip+device-1)
                        slice3 = cubeout%slice(Y_,iy)
                        call cuda_blas%mm_multiplication(device, iy,&
                                slice1,&
                                slice2,&
                                slice3,&
                                self%w(ip+device-1), 1.d0)
                        call slice1%destroy()
                        call slice2%destroy()
                        call slice3%destroy()
                    end do
                    
                end do
                !call CUDAsync(ip)
            end do
        else ! otherwise, use the batched dgemm
            allocate(waited_events(number_of_devices))
            do ip=1, size(self%w), number_of_devices 
                ! to x direction
                do device = 1, min(number_of_devices, size(self%w)-ip+1)
                    ! we must wait the the z-direction is complete before we can start the next round
                    waited_events(device) = StreamContainer_record_device_event(stream_container, device)
                    slice1 = cuda_fx%slice(Z_,ip+device-1)
                    call cuda_blas%mm_multiplication_batched(ip+device-1, (ip-1)/number_of_devices, slice1, &
                             cubein, tmp1, 1.d0, 0.d0, Z_, waited_events(device))
                    call slice1%destroy()
                end do

                ! to y direction
                do device = 1, min(number_of_devices, size(self%w)-ip+1)
                    slice1 = cuda_fy%slice(Z_,ip+device-1)
                    call cuda_blas%mm_multiplication_batched(ip+device-1, (ip-1)/number_of_devices, tmp1, &
                             slice1, tmp2, 1.0d0, 0.0d0, Z_, waited_events(device))
                    call slice1%destroy()
                end do
                
                ! to z direction
                do device = 1, min(number_of_devices, size(self%w)-ip+1)
                    slice1 = cuda_fz%slice(Z_, ip+device-1)
                    ! we must wait the y-direction to be complete before we can call the z-direction
                    waited_events(device) = StreamContainer_record_device_event(stream_container, device)

                    call cuda_blas%mm_multiplication_batched(ip+device-1, (ip-1)/number_of_devices, tmp2, &
                             slice1, cubeout, self%w(ip+device-1), 1.0d0, Y_, waited_events(device))
                    call slice1%destroy()
                    
                end do
            end do
            deallocate(waited_events)
        end if
        ! This could be done using CUDA. Worth it?
        !if (self%coda /= 0.d0) then
        !    call bigben%split("Coda")
        !    cubeout=cubeout + &
        !        self%coda * cube_project(cubein, self%gridin, self%gridout)
        !    call bigben%stop()
        !end if
    end subroutine
#endif

    

end module
