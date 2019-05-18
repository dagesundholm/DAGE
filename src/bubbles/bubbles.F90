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
!> @file bubbles.F90
!! Representation of bubbles
!!  \f$ \sum_{Alm}{r^k_A f^A_{lm}(r_A)  Y^A_{lm} (\theta_A, \phi_A)} \f$

!> @todo bubbles_class documentation
module bubbles_class
    use globals_m
    use xmatrix_m
    use grid_class
    use LIPBasis_class
    use harmonic_class
    use timer_m
    use CartIter_class
    use evaluators_class
    use MultiPoleTools_m
    use RealSphericalHarmonics_class
    use CudaObject_class
    use Evaluators_class
! debugging
    use MemoryLeakChecker_m
    use Points_class
#ifdef HAVE_CUDA 
    use ISO_C_BINDING
    use CUDAInjector_class   
    use cuda_m
#endif
#ifdef HAVE_MPI
    use mpi
#endif

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!               bubbles.f90                                   !
!   Poisson solver for spherical potentials                   !
!   Algorithm by Charlotte Froese Fischer and Dage Sundholm   !
!   Originally coded by Dage Sundholm                         !
!   Translated to Fortran90 by Sergio Losilla (IV.2009)       !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    implicit none
    private

    public :: Bubbles, BubblesMultipliers, BubblesMultiplier, BubblesEvaluator
#ifdef HAVE_CUDA
    public :: Bubbles_destroy_cuda
#endif

    public :: operator(*)
    public :: operator(+)

    public :: assignment(=)

    public :: alpha_bubbles
    public :: beta_bubbles

    public :: symbol2z
    public :: bubbles_multipliers
    

    interface Bubbles
        module procedure :: Bubbles_init_explicit
        module procedure :: Bubbles_init_copy
    end interface

    !> Representation of the Bubbles component of three-dimensional scalar
    !! functions \ref function3d_class::function3d "Function3D"

    !> \f$\sum_{Alm}{r^k_A f^A_{lm}(r_A) Y_{Alm} (\theta_A,\phi_A)}\f$
    type, extends(CudaObject) :: Bubbles
        private
        !> Number of centers present
        integer(INT32) :: nbub=0
        !> Number of centers globally
        integer(INT32) :: nbub_global = 0
        !> Order numbers of centers present
        integer(INT32), public, allocatable :: ibubs(:)
        !> Maximum angular momentum
        integer(INT32), public :: lmax=0

        !> @todo Change `self%%numf` to a pure function `self%%numf()`\n
        !! It doesn't work yet due to a gfortran bug:\n
        !! http://gcc.gnu.org/bugzilla/show_bug.cgi?id=55960\n
        !! This should be fixed with gfortran-4.9.x (apparently fixed with
        !! trunk version from 2014.01.25)

        !> Total number of radial functions (lmax+1)**2
        integer(INT32), public :: numf=0
        !> Exponent of \f$r\f$
        integer(INT32) :: k=0
        !> Bubble centers
        real(REAL64), public, allocatable :: centers(:,:)
        !> Glabal bubble centers
        real(REAL64), public, allocatable :: global_centers(:,:)
        !> Nuclear charges
        real(REAL64),         allocatable :: z(:)
        !> Global Nuclear charges
        real(REAL64),         allocatable :: global_z(:)
        !> xwh, dimension of gr = number of atoms, i.e, = nbub
        type(Grid1DPointer),     public, allocatable :: gr(:)
        type(Grid1DPointer),     public, allocatable :: global_grids(:)
        type(REAL64_2D_Pointer), public, allocatable :: bf(:)
        type(REAL64_2D), private, allocatable        :: bf_data(:)
    contains
        procedure, private :: alloc_bf   => bubbles_alloc_bf

        procedure :: get_lmax => bubbles_get_lmax
        procedure :: get_nbub => bubbles_get_nbub
        procedure :: get_nbub_global => bubbles_get_nbub_global
        procedure :: get_k    => bubbles_get_k

        ! %%%%%%% Specific interfaces (for private use)
        procedure, private :: bubbles_get_grid_pick, bubbles_get_grid_all
        procedure, private :: bubbles_get_global_grid_pick, bubbles_get_global_grid_all
        procedure, private :: bubbles_get_z_pick, bubbles_get_z_all
        procedure, private :: bubbles_get_global_z_pick, bubbles_get_global_z_all
        procedure, private :: bubbles_get_centers_pick, bubbles_get_centers_all, &
                              Bubbles_get_centers_pick_many
        procedure, private :: bubbles_get_global_centers_pick, bubbles_get_global_centers_all, &
                              Bubbles_get_global_centers_pick_many
        procedure, private :: bubbles_get_f_pick, bubbles_get_f_all
        procedure, private :: bubbles_print_pick, bubbles_print_all
        procedure, private :: bubbles_integrate_pick, bubbles_integrate_all
        procedure, private :: bubbles_integrate_radial_pick, bubbles_integrate_radial_all
        procedure, private :: bubbles_eval_pick, bubbles_eval_all
        procedure, private :: Bubbles_simple_eval_pick
        procedure, private :: Bubbles_get_contaminants, Bubbles_get_foreign_bubbles_contaminants
        procedure, private :: Bubbles_subtract_bubbles, Bubbles_subtract_bubbles_ibubs
        procedure, private :: Bubbles_get_sub_bubbles, Bubbles_get_sub_bubbles_with_indices
        

        generic :: get_grid    => bubbles_get_grid_pick, bubbles_get_grid_all
        generic :: get_global_grid => bubbles_get_global_grid_pick, bubbles_get_global_grid_all
        generic :: get_z       => bubbles_get_z_pick, bubbles_get_z_all
        generic :: get_global_z=> bubbles_get_global_z_pick, bubbles_get_global_z_all
        generic :: get_centers => bubbles_get_centers_pick, bubbles_get_centers_all, &
                                  Bubbles_get_centers_pick_many
        generic :: get_global_centers => Bubbles_get_global_centers_pick, &
                                  Bubbles_get_global_centers_all, &
                                  Bubbles_get_global_centers_pick_many
        generic :: get_f       => bubbles_get_f_pick, bubbles_get_f_all

        generic, public :: subtract_bubbles => Bubbles_subtract_bubbles, Bubbles_subtract_bubbles_ibubs
        generic, public :: get_sub_bubbles  => Bubbles_get_sub_bubbles, Bubbles_get_sub_bubbles_with_indices

        procedure :: get_order_number => bubbles_get_order_number
        procedure :: get_multipole_moments => bubbles_get_multipole_moments
        procedure :: get_ibub   => bubbles_get_ibub
        procedure :: get_ibubs  => bubbles_get_ibubs
        procedure :: contains_ibub => Bubbles_contains_ibub
        procedure :: get_ibubs_within_range  => Bubbles_get_ibubs_within_range
        procedure :: set_k      => bubbles_set_k
        procedure :: decrease_k => bubbles_decrease_k
        procedure :: increase_k => bubbles_increase_k
        procedure :: get_nuclear_contaminants => Bubbles_get_nuclear_contaminants
        
        procedure :: merge_with => Bubbles_merge
        procedure :: is_merge_of => Bubbles_is_merge_of
        procedure :: get_merge_ibubs => Bubbles_get_merge_ibubs
        procedure :: get_intersection => Bubbles_get_intersection
        procedure :: get_intersection_ibubs => Bubbles_get_intersection_ibubs
        procedure :: is_intersection_of => Bubbles_is_intersection_of
        procedure :: copy_content => Bubbles_copy_content

        procedure :: print_out => Bubbles_print_out
        generic :: print      => bubbles_print_pick, bubbles_print_all
        generic :: integrate  => bubbles_integrate_pick, bubbles_integrate_all
        generic :: integrate_radial => bubbles_integrate_radial_pick, bubbles_integrate_radial_all
        generic :: eval       => bubbles_eval_pick, bubbles_eval_all, &
                                 Bubbles_simple_eval_pick
        procedure :: rpow     => bubbles_rpow

        procedure :: eval_radial => bubbles_eval_radial
#ifdef HAVE_CUDA
        procedure :: eval_3dgrid_cuda => Bubbles_eval_3dgrid_cuda
#endif
        procedure :: eval_3dgrid => bubbles_eval_3dgrid
        procedure :: destroy => bubbles_destroy
        generic   :: get_contaminants => Bubbles_get_contaminants, &
                                      Bubbles_get_foreign_bubbles_contaminants
        procedure :: make_taylor      => bubbles_make_taylor
        procedure :: project_onto     => bubbles_project_onto

        procedure, private :: bubbles_product
        generic, public    :: operator(*) => bubbles_product

        procedure, private :: get_radial_derivatives => Bubbles_get_radial_derivatives
        procedure          :: radial_derivative      => Bubbles_radial_derivative
        procedure          :: extrapolate_origo      => Bubbles_extrapolate_origo

        ! IO functions
        procedure, public  :: write         => bubbles_write
        procedure, public  :: read          => Bubbles_read
        procedure, public  :: read_and_init => Bubbles_read_and_init

        procedure, public :: init_copy      => Bubbles_init_copy_sub
  
#ifdef HAVE_CUDA
        procedure, public  :: cuda_init => Bubbles_cuda_init
        procedure, public  :: cuda_upload_all => Bubbles_cuda_upload_all
        procedure, public  :: cuda_upload     => Bubbles_cuda_upload
        procedure, public  :: cuda_download => Bubbles_cuda_download
        procedure, public  :: cuda_destroy => Bubbles_cuda_destroy
        procedure, public  :: set_cuda_interface   => Bubbles_set_cuda_interface
        procedure, public  :: cuda_get_sub_bubbles => Bubbles_cuda_get_sub_bubbles
#endif

        ! Not so straightforward to put here, because having the input
        ! objects of class(Bubbles) causes some headaches
!        procedure, private :: bubbles_add
!        generic, public    :: operator(+) => bubbles_add

        ! Uhm gfortran 4.8 does not like this with a ICE in some other part of
        ! the code...
!        procedure, private :: bubbles_assign_real
        !generic, public    :: assignment(=) => bubbles_assign_real
    end type Bubbles

    interface assignment(=)
        module procedure bubbles_assign_real
        !module procedure bubbles_assign
    end interface

    interface operator(+)
        module procedure bubbles_add
    end interface

    interface operator(*)
        module procedure :: bubbles_times_real64
        module procedure :: real64_times_bubbles
    end interface

#ifdef HAVE_CUDA
    interface
        subroutine cuda_bubbleset_eval_grid_new(gdims,xgr,ygr,zgr,cube) bind(C)
            use ISO_C_BINDING
            integer(C_INT) :: gdims(3)
            real(C_DOUBLE) :: xgr(*)
            real(C_DOUBLE) :: ygr(*)
            real(C_DOUBLE) :: zgr(*)
            real(C_DOUBLE) :: cube(*)
        end subroutine
    end interface

    interface
        subroutine bubbles_cuda_product(f1_bubbles, f2_bubbles,  &
                                    coefficients, number_of_terms, result_lm, &
                                    positions, result_lm_size, nbub, result_bubbles) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: f1_bubbles
            type(C_PTR), value    :: f2_bubbles
            real(C_DOUBLE)        :: coefficients(*)
            integer(C_INT)           :: number_of_terms(*)
            integer(C_INT)           :: result_lm(*)
            integer(C_INT)           :: positions(*)
            integer(C_INT), value :: result_lm_size
            integer(C_INT), value :: nbub
            type(C_PTR), value    :: result_bubbles
        end subroutine
    end interface
    
        
    interface
        subroutine bubble_upload_cuda(bubbles, ibub, lmax, bf) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: bubbles
            integer(C_INT), value :: ibub
            integer(C_INT), value :: lmax
            real(C_DOUBLE)        :: bf(*)
        end subroutine
    end interface

    interface
        subroutine bubble_upload_all_cuda(bubbles, ibub, lmax, k, bf) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: bubbles
            integer(C_INT), value :: ibub
            integer(C_INT), value :: lmax
            integer(C_INT), value :: k
            real(C_DOUBLE)        :: bf(*)
        end subroutine
    end interface

    interface
        subroutine bubbles_set_processor_configuration_cuda(bubbles, processor_order_number, &
                       number_of_processors) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: bubbles
            integer(C_INT), value :: processor_order_number
            integer(C_INT), value :: number_of_processors
        end subroutine
    end interface

    interface
        subroutine bubbles_upload_all_cuda(bubbles) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: bubbles
        end subroutine
    end interface

    interface
        real(C_DOUBLE) function bubbles_integrate_cuda(bubbles) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: bubbles
        end function
    end interface
    
    interface
        subroutine bubble_inject_new(grid, bubble, device_cube, lmin) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: grid
            type(C_PTR), value    :: bubble
            type(C_PTR), value    :: device_cube
            integer(C_INT), value :: lmin
        end subroutine
    end interface
    
    interface
        subroutine bubbles_destroy_cuda(bubbles) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: bubbles
        end subroutine
    end interface

    interface
        type(C_PTR) function bubbles_init_cuda(nbub) bind(C)
            use ISO_C_BINDING
            integer(C_INT), value    :: nbub
        end function
    end interface

    interface
        subroutine bubble_init_cuda(bubbles, grid, i, ibub, center, lmax, &
                                 k, charge, streamContainer) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: bubbles
            type(C_PTR),    value :: grid
            integer(C_INT), value :: i
            integer(C_INT), value :: ibub
            real(C_DOUBLE)        :: center(3)
            integer(C_INT), value :: lmax
            integer(C_INT), value :: k
            real(C_DOUBLE), value :: charge
            type(C_PTR),    value :: streamContainer
        end subroutine
    end interface

    interface
        subroutine Bubble_add_cuda(bubbles, bubbles1, ibub) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: bubbles
            type(C_PTR),    value :: bubbles1
            integer(C_INT), value :: ibub
        end subroutine
    end interface

    interface
        subroutine Bubbles_add_cuda(bubbles, bubbles1) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: bubbles
            type(C_PTR),    value :: bubbles1
        end subroutine
    end interface

    

    interface
        subroutine Bubbles_inject_cuda(bubbles, grid, lmin, cudacube) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: bubbles
            type(C_PTR), value    :: grid
            integer(C_INT), value :: lmin
            type(C_PTR), value    :: cudacube
        end subroutine
    end interface

    interface
        type(C_PTR) function Bubbles_init_page_locked_f_cuda(lmax, shape) bind(C)
            use ISO_C_BINDING
            integer(C_INT), value  :: lmax
            integer(C_INT), value  :: shape
        end function
    end interface

    interface
        pure subroutine Bubbles_destroy_page_locked_f_cuda(f) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: f
        end subroutine
    end interface

    interface
        type(C_PTR) function Bubbles_get_sub_bubbles_cuda(bubbles, ibubs, nbub) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: bubbles
            integer(C_INT)        :: ibubs(*)
            integer(C_INT), value :: nbub
        end function
    end interface
#endif

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                   BubblesMultiplier definition                          %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!% BubblesMultiplier stores the variables needed for multiplication of     %
!% two bubbles with specific lmax-values. After initialization the object  %
!% can be used to multiply any instances of bubbles with correct lmax      %
!% values, regardless of their k-value or content.                         %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    type, extends(CudaObject) :: BubblesMultiplier
        !> The Maximum l value for the input bubbles which have 
        !! the smaller lmax value
        integer              :: smaller_lmax
        !> The Maximum l value for the input bubbles which have 
        !! the larger lmax value
        integer              :: larger_lmax
        !> The object containing the variables related to multiplication of 
        !! spherical harmonics
        type(YProduct)       :: yproduct
        !> If this object is the owner of the bubbles objects, i.e., if this
        !! is the first to be inited
        logical, private         :: first
#ifdef HAVE_CUDA
        !> Pointer to C++/Cuda object with type Bubbles for the smaller 
        !! lmax value-bubbles
        type(C_PTR), allocatable :: bubbles1
        !> Pointer to C++/Cuda object with type Bubbles for the larger 
        !! lmax value-bubbles
        type(C_PTR), allocatable :: bubbles2
        !> Pointer to C++/Cuda object with type Bubbles for the Taylor series 
        !! bubbles for smaller lmax value-bubbles
        type(C_PTR), allocatable :: taylor_series_bubbles1
        !> Pointer to C++/Cuda object with type Bubbles for the Taylor series 
        !! bubbles for smaller lmax value-bubbles
        type(C_PTR), allocatable :: taylor_series_bubbles2
        !> Pointer to C++/Cuda object with type Bubbles for result bubbles
        type(C_PTR), allocatable :: result_bubbles
#endif
    contains
        procedure :: multiply           => BubblesMultiplier_multiply
        procedure :: communicate_result => BubblesMultiplier_communicate_result
        procedure :: destroy            => BubblesMultiplier_destroy
#ifdef HAVE_CUDA
        procedure, public :: cuda_init    => BubblesMultiplier_cuda_init
        procedure, public :: cuda_destroy => BubblesMultiplier_cuda_destroy
#endif
    end type

    interface BubblesMultiplier
        module procedure :: BubblesMultiplier_init
    end interface


!--------------------------- CUDA/C++ interfaces ---------------------------------!
!  See implementations at bubbles_multiplier.cu                                   !
!---------------------------------------------------------------------------------!

#ifdef HAVE_CUDA
    interface
        type(C_PTR) function BubblesMultiplier_init_cuda(bubbles1, bubbles2,  &
                                    result_bubbles, taylor_series_bubbles1, &
                                    taylor_series_bubbles2, lmax, coefficients, number_of_terms, &
                                    result_lm, positions, result_lm_size, processor_order_number,  &
                                    number_of_processors, streamContainer) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: bubbles1
            type(C_PTR), value    :: bubbles2
            type(C_PTR), value    :: result_bubbles
            type(C_PTR), value    :: taylor_series_bubbles1
            type(C_PTR), value    :: taylor_series_bubbles2
            integer(C_INT), value :: lmax
            real(C_DOUBLE)        :: coefficients(*)
            integer(C_INT)        :: number_of_terms(*)
            integer(C_INT)        :: result_lm(*)
            integer(C_INT)        :: positions(*)
            integer(C_INT), value :: result_lm_size
            integer(C_INT), value :: processor_order_number
            integer(C_INT), value :: number_of_processors
            type(C_PTR), value    :: streamContainer
        end function
    end interface
    
        
    interface
        subroutine BubblesMultiplier_download_result_cuda(multiplier, lmax, ibubs, nbub) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: multiplier
            integer(C_INT), value :: lmax
            integer(C_INT)        :: ibubs(*)
            integer(C_INT), value :: nbub
        end subroutine
    end interface

    interface
        subroutine BubblesMultiplier_destroy_cuda(multiplier) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: multiplier
        end subroutine
    end interface

    interface
        subroutine BubblesMultiplier_multiply_bubble_cuda(multiplier, ibub, &
                                   bubble1_bf, bubble2_bf, result_bubble_bf, &
                                   taylor_series_bubble1_bf, taylor_series_bubble2_bf, &
                                   lmax1, lmax2, tlmax1, tlmax2 &
                                   ) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: multiplier
            integer(C_INT), value :: ibub
            real(C_DOUBLE)        :: bubble1_bf(*)
            real(C_DOUBLE)        :: bubble2_bf(*)
            real(C_DOUBLE)        :: result_bubble_bf(*)
            real(C_DOUBLE)        :: taylor_series_bubble1_bf(*)
            real(C_DOUBLE)        :: taylor_series_bubble2_bf(*)
            integer(C_INT), value :: lmax1, lmax2, tlmax1, tlmax2
        end subroutine
    end interface

    interface
        subroutine BubblesMultiplier_set_ks(multiplier, bubble1_k, bubble2_k, result_bubble_k,  &
            taylor_series_bubble1_k, taylor_series_bubble2_k)  bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: multiplier
            integer(C_INT), value :: bubble1_k
            integer(C_INT), value :: bubble2_k
            integer(C_INT), value :: result_bubble_k
            integer(C_INT), value :: taylor_series_bubble1_k
            integer(C_INT), value :: taylor_series_bubble2_k
        end subroutine
    end interface
#endif


!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                   BubblesMultipliers definition                         %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%  BubblesMultipliers is a container that contains variable amount of     %
!%  BubblesMultiplier-objects. The containers purpose is to initialize and %
!%  select correct BubblesMultiplier for input Bubbles and call the        %
!%  multiplication method of the correct BubblesMultiplier.                %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    type :: BubblesMultipliers
        type(BubblesMultiplier), allocatable :: multipliers(:)
        type(BubblesMultiplier), pointer     :: first_multiplier => NULL()
        integer                              :: multiplier_count = 0
        integer                              :: lmax = 20
    contains 
        procedure :: init_from_mould => BubblesMultipliers_init_multipliers_from_mould
        procedure :: multiply        => BubblesMultipliers_multiply
        procedure :: get_multiplier  => BubblesMultipliers_get_multiplier
        procedure :: destroy         => BubblesMultipliers_destroy
    end type
    
    
    type(BubblesMultipliers)      :: bubbles_multipliers


!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                   BubblesEvaluator definition                           %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!% BubblesEvaluator allows the fast evaluation of a set of predefined      %
!% points. NOTE: this should not be used for evaluation of bubbles at      %
!% 3D grid. For that, there is a preexisting and less memory consuming     %
!% option: eval_3dgrid_cuda of Bubbles.                                    %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    type, extends(Evaluator) :: BubblesEvaluator
        !> The prototype bubbles used to initialize stuff
        type(Bubbles), allocatable  :: bubbles
#ifdef HAVE_CUDA
        type(C_PTR),   allocatable  :: bubbles_cuda
#endif 
        !> Gradient factors for the evaluation of spherical harmonics gradients
        !! in x, y and z directions. The dimensions are (lmax+1)**2, 1 or 2.
        !! 2 in the case of x and y, 1 in the case of z.
        real(REAL64), allocatable   :: gradient_factors_x(:, :), gradient_factors_y(:, :), gradient_factors_z(:) 

    contains
        procedure :: destroy             => BubblesEvaluator_destroy
        procedure :: destroy_input_bubbles        => BubblesEvaluator_destroy_input_bubbles
        procedure :: set_input_bubbles            => BubblesEvaluator_set_input_bubbles
        procedure :: evaluate_grid                => BubblesEvaluator_evaluate_grid
        procedure :: evaluate_points              => BubblesEvaluator_evaluate_points
        procedure :: evaluate_divergence_points   => BubblesEvaluator_evaluate_divergence_points
        procedure :: evaluate_points_from_bubbles => BubblesEvaluator_evaluate_points_from_bubbles
        procedure :: evaluate_grid_from_bubbles   => BubblesEvaluator_evaluate_grid_from_bubbles
        procedure :: evaluate_gradients_from_VSH => &
                         BubblesEvaluator_evaluate_gradients_from_VSH
        procedure :: evaluate_contracted_gradients_from_VSH => &
                         BubblesEvaluator_evaluate_contracted_gradients_from_VSH
        procedure :: evaluate_gradients_as_VSH => &
                         BubblesEvaluator_evaluate_gradients_as_VSH
        procedure, private :: init_gradient_factors => &
                         BubblesEvaluator_init_gradient_factors
        procedure :: convert_gradients_from_VSH_to_bubbles => &
                         BubblesEvaluator_convert_gradients_from_VSH_to_bubbles 
        procedure :: evaluate_gradients_as_bubbles => &
                         BubblesEvaluator_evaluate_gradients_as_bubbles 
        procedure :: evaluate_divergence_as_bubbles => &
                         BubblesEvaluator_evaluate_divergence_as_bubbles 
#ifdef HAVE_CUDA
        procedure :: cuda_init     => BubblesEvaluator_cuda_init
        procedure :: cuda_destroy  => BubblesEvaluator_cuda_destroy
#endif
    end type

    interface BubblesEvaluator
        module procedure :: BubblesEvaluator_init
    end interface

!--------------------------- CUDA/C++ interfaces ---------------------------------!
!  See implementations at bubbles_cuda.cu                                         !
!---------------------------------------------------------------------------------!

#ifdef HAVE_CUDA
    interface
        type(C_PTR) function BubblesEvaluator_init_cuda(streamContainer) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: streamContainer
        end function
    end interface

    
    interface
        subroutine BubblesEvaluator_set_bubbles_cuda(bubbles_evaluator, bubbles) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: bubbles_evaluator
            type(C_PTR), value    :: bubbles
        end subroutine
    end interface
#endif

contains

    function fuzz_factor(r) result(f)
        real(REAL64):: r(:),f(size(r))

!        f=0.5d0*erfc(4.d0*(r-1.d0))
        f=0.0d0
    end function

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                                Constructors                            %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    !> Explicitly construct Bubbles.

    !> Radial functions are allocated and set to 0.
    function Bubbles_init_explicit(lmax, centers, global_centers, grids, global_grids, &
                                   z, global_z, k, ibubs, nbub_global) result(new)
        type(Bubbles)                        :: new
        integer,        intent(in)           :: lmax
        type(Grid1DPointer),   intent(in)    :: grids(:)
        type(Grid1DPointer),   intent(in)    :: global_grids(:)
        real(REAL64),   intent(in)           :: centers(3,size(grids))
        real(REAL64),   intent(in)           :: global_centers(:, :)
        real(REAL64),   intent(in)           :: z(size(grids))
        real(REAL64),   intent(in)           :: global_z(size(global_grids))
        integer,        intent(in), optional :: k
        integer(INT32), intent(in), optional :: ibubs(size(grids))
        integer(INT32), intent(in), optional :: nbub_global

        integer :: ibub

        new%nbub     = size(grids)
        new%lmax     = lmax
        new%numf     = (lmax+1)**2
        new%centers  = centers
        new%global_centers = global_centers
        new%z        = z
        new%global_z = global_z
        new%gr       = grids
        new%global_grids = global_grids

        if(present(k)) new%k=k

        if (present(ibubs)) then
            new%ibubs = ibubs
            new%nbub_global = nbub_global
        else 
            new%ibubs = [(ibub, ibub = 1, new%nbub)]
            new%nbub_global = new%nbub
        end if 
        call new%alloc_bf() 
        return
    end function

    !> Read grids from binary stream
    !! NOTE: The global grids must be set before calling this
    subroutine Bubbles_read_and_init(self, fd, bubblestart)
        class(Bubbles), intent(inout) :: self
        integer(INT32)                :: fd
        integer(INT32)                :: i, ibub
        integer(INT32)                :: ndim
        real(REAl64), allocatable     :: z(:), centers(:, :), ibubs(:)
        integer(INT64),intent(in),optional :: bubblestart

        if (present(bubblestart)) then
            read(fd,pos=bubblestart) self%nbub, self%nbub_global, self%lmax, self%k
        else
            read(fd) self%nbub, self%nbub_global,self%lmax, self%k
        end if
        self%numf = (self%lmax+1)**2

        allocate(self%z(self%nbub))
        allocate(self%global_z(self%nbub_global))
        allocate(self%global_centers(3, self%nbub_global))
        allocate(self%centers(3, self%nbub))
        allocate(self%ibubs(self%nbub))
        do i=1,self%nbub
            read(fd) self%z(i), self%centers(:,i)
            read(fd) self%ibubs(i)
        end do

        do ibub=1,self%nbub_global
            read(fd) self%global_centers(:, ibub)
            read(fd) self%global_z(ibub)
        end do

        ! set the correct local grids
        allocate(self%gr(self%nbub))
        do i = 1, self%nbub
            self%gr(i) = self%global_grids(self%ibubs(i)) 
        end do

        ! allocate the space to store the data
        call self%alloc_bf()
        do ibub = 1, self%nbub
            ! Read spherical functions
            read(fd) self%bf(ibub)%p(:,:)
        end do
    end subroutine 

    !> See \ref function3d_class::function3d_write "Function3D%write()" for a
    !! detailed description of the binary format).
    subroutine Bubbles_read(self, fd, bubblestart)
        class(Bubbles), intent(inout) :: self
        integer(INT32) :: fd
        integer(INT32) :: nbub,nbub_global,lmax,k
        integer(INT32) :: ibub, i
        integer(INT32) :: ndim
        integer(INT64),intent(in),optional :: bubblestart

        if (present(bubblestart)) then
            read(fd,pos=bubblestart) nbub,nbub_global,lmax,k
        else
            read(fd) nbub,nbub_global,lmax,k
        end if

        self%nbub = nbub
        self%nbub_global = nbub_global
        self%lmax = lmax
        self%numf = (lmax+1)**2
        self%k    = k

        do i=1,nbub
            read(fd) self%z(i),self%centers(:,i)
            read(fd) self%ibubs(i)
        end do

        do ibub=1,self%nbub_global
            read(fd) self%global_centers(:, ibub)
            read(fd) self%global_z(ibub)
        end do

        ! set the correct local grids
        if (allocated(self%gr)) deallocate(self%gr)
        allocate(self%gr(self%nbub))
        do i = 1, self%nbub
            self%gr(i) = self%global_grids(self%ibubs(i)) 
        end do

        do ibub = 1, nbub
            ! Read spherical functions
            read(fd) self%bf(ibub)%p(:,:)
        end do
    end subroutine

    
    function Bubbles_init_copy(bubsin, copy_content, lmax, k) result(new)
        type(Bubbles), intent(in)            :: bubsin
        logical,       intent(in), optional  :: copy_content
        !> Angular momentum number of the new Function instance, if different
        !! from `bubsin%%lmax`
        integer(INT32), intent(in), optional :: lmax
        !> \f$k\f$ exponent of the new Function instance. If `copy_content`,
        !! this is ignored (`new%%k` is `bubsin%%k`)
        integer(INT32), intent(in), optional :: k
        type(Bubbles)                        :: new

        call new%init_copy(bubsin, copy_content = copy_content, lmax = lmax, k = k)
    end function

    !> Create a copy of another Bubbles instance.
    subroutine Bubbles_init_copy_sub(new, bubsin, copy_content, lmax, k)
        !> The result object
        class(Bubbles), intent(inout)        :: new

        type(Bubbles), intent(in)            :: bubsin
        logical,       intent(in), optional  :: copy_content
        !> Angular momentum number of the new Function instance, if different
        !! from `bubsin%%lmax`
        integer(INT32), intent(in), optional :: lmax
        !> \f$k\f$ exponent of the new Function instance. If `copy_content`,
        !! this is ignored (`new%%k` is `bubsin%%k`)
        integer(INT32), intent(in), optional :: k

        
        !> Copy content temporary variable
        logical                              :: cc
        integer(INT32)                       :: l, ibub
        integer                              :: numf_cp

        if(present(lmax)) then
            l=lmax
        else
            l=bubsin%lmax
        end if

        if (present(copy_content)) then
            cc = copy_content
        else
            cc = .FALSE.
        end if
        new%nbub     = size(bubsin%gr)
        new%lmax     = l
        new%numf     = (l+1)**2
        new%centers  = bubsin%centers
        new%global_centers = bubsin%global_centers
        new%z        = bubsin%z
        new%global_z = bubsin%global_z
        new%gr       = bubsin%gr
        new%global_grids = bubsin%global_grids
        new%ibubs    = bubsin%ibubs
        new%nbub_global = bubsin%nbub_global
        call new%alloc_bf()

        ! Do not copy content unless explicitly requested
        if(cc) then
            new%k=bubsin%k
        ! k will take the explicitly passed value
        else if(present(k)) then
           new%k=k
        end if

        if(cc) then
            numf_cp=min(bubsin%numf,new%numf)
            forall (ibub=1 : new%nbub)
                new%bf(ibub)%p(:,:numf_cp)=bubsin%bf(ibub)%p(:,:numf_cp)
            end forall
        end if
        return
    end subroutine

    function Bubbles_get_sub_bubbles_with_indices(self, ibubs, copy_content) result (new)
        class(Bubbles), intent(in)  :: self
        integer(INT32), intent(in)  :: ibubs(:)
        logical, optional, intent(in) :: copy_content
        real(REAL64)               :: centers(3, self%nbub)
        real(REAL64)               :: z(self%nbub)
        type(Grid1DPointer)        :: grids(self%nbub)
        

        
        type(Bubbles) :: new

        logical :: cc
        integer(INT32) :: i, j, ibub, nbub

        integer        :: numf_cp

        nbub = 0
        ! get only the bubbles that have centers within ranges
        do i = 1, self%nbub
            do j = 1, size(ibubs)
                if (self%ibubs(i) == ibubs(j)) then 
                    nbub = nbub + 1
                    centers(:, nbub) = self%centers(:, i)
                    grids(nbub)%p => self%gr(i)%p
                    z(nbub) = self%z(i)
                    exit
                end if
            end do 
        end do


        ! init empty bubbles
        new =  Bubbles( lmax      = self%lmax, &
                        centers   = centers(:, : nbub), &
                        global_centers = self%global_centers, &
                        grids     = grids(: nbub), &
                        global_grids = self%global_grids, &
                        z         = z(: nbub), &
                        global_z  = self%global_z, &
                        k         = self%k,    &
                        ibubs     = ibubs(: nbub), & 
                        nbub_global = self%nbub_global)

        ! copy the content from the self
        if (present(copy_content) .and. copy_content) then
            do i=1, size(new%ibubs)
                ibub = new%ibubs(i)
                numf_cp = min(self%numf,new%numf)
                new%bf(i)%p(:,:numf_cp)=self%bf(self%get_order_number(ibub))%p(:,:numf_cp)
            end do
        end if
    end function 

    !> Create a copy of another Bubbles instance but only take bubbles
    !! withing ranges into account
    function Bubbles_get_sub_bubbles(self, ranges, copy_content) result(new)
        class(Bubbles), intent(in)    :: self
        real(REAL64), intent(in)      :: ranges(2, 3)
        logical, optional, intent(in) :: copy_content
        integer(INT32), allocatable   :: ibubs(:)
        type(Bubbles)                 :: new


        ibubs = self%get_ibubs_within_range(ranges)
 
        ! init empty bubbles
        new =  self%get_sub_bubbles(ibubs, copy_content = copy_content)
        deallocate(ibubs)
        return
    end function

    !> Create a copy of another Bubbles instance but only take bubbles
    !! withing ranges into account
    function Bubbles_get_ibubs_within_range(self, ranges) result(result_ibubs)
        class(Bubbles), intent(in)  :: self
        real(REAL64), intent(in)    :: ranges(2, 3)
        integer(INT32)              :: ibubs(self%nbub)
        integer(INT32), allocatable :: result_ibubs(:)

        integer(INT32)              :: i, ibub, nbub


        nbub = 0
        ! get only the bubbles that have centers within ranges
        do i = 1, self%nbub
            if (all(self%centers(:, i) - ranges(1, :) >= 0.0) .and. &
                    all(ranges(2, :) - self%centers(:, i) > 0.0)) then
                ! increase the number of result bubbles 
                nbub = nbub + 1

                ! add the ibub to the result array
                ibubs(nbub) = self%ibubs(i)
            end if 
        end do
        result_ibubs = ibubs(:nbub)
        return
    end function

    !> Create a copy of another Bubbles instance but only take bubbles
    !! withing ranges into account
    function Bubbles_subtract_bubbles_ibubs(self, ibubs, copy_content) result(new)
        class(Bubbles), intent(in)  :: self
        integer(INT32), intent(in)  :: ibubs(:)
        logical, optional, intent(in) :: copy_content
        integer(INT32)              :: jbubs(self%nbub)

        
        type(Bubbles) :: new

        logical :: found
        integer(INT32) :: i, j, ibub, nbub

        integer        :: numf_cp

        nbub = 0
        ! get only the indices of bubbles that are in self but are not in 'bubbles'
        do i = 1, self%nbub
            found = .FALSE.
            do j = 1, size(ibubs)
                if (ibubs(j) == self%ibubs(i)) then
                    found = .TRUE.
                    exit
                end if 
            end do 
            
            ! ibub of i is not in bubbles
            if (.not. found) then
                nbub = nbub + 1
                jbubs(nbub) = self%ibubs(i)
            end if 
        end do

 
        ! init empty bubbles
        new =  self%get_sub_bubbles(jbubs(: nbub), copy_content = copy_content)
        return
    end function

    !> Create a copy of another Bubbles instance but only take bubbles
    !! within range into account
    function Bubbles_subtract_bubbles(self, bubbls, copy_content) result(new)
        class(Bubbles), intent(in)  :: self
        type(Bubbles), intent(in)   :: bubbls
        logical, optional, intent(in) :: copy_content
 
        type(Bubbles)               :: new
 
        ! init empty bubbles
        new =  self%subtract_bubbles(bubbls%ibubs, copy_content = copy_content)
        return
    end function

    !> Get the common ibubs between 'self' and 'bubsin'.
    pure function Bubbles_get_intersection_ibubs(self, bubsin) result(ibubs_result)
        class(Bubbles),           intent(in) :: self
        type(Bubbles),            intent(in) :: bubsin
        integer(INT32)                       :: nbub, ibub, i, ibubs(self%nbub)
        integer(INT32),         allocatable  :: ibubs_result(:)

        nbub = 0
        do i = 1, self%nbub
            ibub = self%ibubs(i)
            if (bubsin%contains_ibub(ibub)) then
                nbub = nbub + 1 
                ibubs(nbub) = ibub
            end if
        end do 

        ibubs_result = ibubs(:nbub)
    end function

    !> Checks if 'self' could be an intersection of 'bubbles1' and 'bubbles2'
    pure function Bubbles_is_intersection_of(self, bubbles1, bubbles2) result(is_intersection)
        class(Bubbles),           intent(in) :: self
        type(Bubbles),            intent(in) :: bubbles1
        type(Bubbles),            intent(in) :: bubbles2
        logical                              :: is_intersection
        integer(INT32),         allocatable  :: ibubs(:)
        integer(INT32)                       :: i

        if (.not. allocated(self%ibubs)) then
            is_intersection = .FALSE.
        else
            ibubs = bubbles1%get_intersection_ibubs(bubbles2)
            if (size(ibubs) /= size(self%ibubs)) then
                is_intersection = .FALSE.
            else
                is_intersection = all([(ibubs(i) == self%ibubs(i), i = 1, size(ibubs))])
            end if
            deallocate(ibubs)
        end if
        
    end function

    !> Create a new Bubbles instance by combining two sets of Bubbles so
    !! that only the ones present in both sets are 
    function Bubbles_get_intersection(self, bubsin, lmax) result(new)
        class(Bubbles),           intent(in) :: self
        type(Bubbles),            intent(in) :: bubsin
        integer(INT32), optional, intent(in) :: lmax
        real(REAL64)                         :: centers(3, self%nbub)
        real(REAL64)                         :: z(self%nbub)
        type(Grid1DPointer)                  :: grids(self%nbub)
        integer(INT32), allocatable          :: ibubs(:)
        
        type(Bubbles) :: new

        logical                              :: do_copy
        integer(INT32)                       :: i, j, lmax_, ibub, nbub, order_number

        integer        :: numf_cp
        allocate(ibubs(self%nbub))
        nbub = 0
        do i = 1, self%nbub
            ibub = self%ibubs(i)
            if (bubsin%contains_ibub(ibub)) then
                order_number = bubsin%get_order_number(ibub)
                nbub = nbub + 1 
                ibubs(nbub) = ibub
                centers(:, nbub) = self%centers(:, order_number)
                grids(nbub)%p => self%gr(order_number)%p
                z(nbub) = self%z(order_number)
            end if
        end do 

        if (present(lmax)) then
            lmax_ = lmax
        else
            lmax_ = max(bubsin%lmax, self%lmax)
        end if


 
        ! init empty bubbles
        new =  Bubbles( lmax      = lmax_, &
                        centers   = centers(:, : nbub), &
                        global_centers = self%global_centers, &
                        grids     = grids(:nbub), &
                        global_grids = self%global_grids, &
                        z         = z(:nbub), &
                        global_z  = self%global_z, &
                        k         = bubsin%k, &
                        ibubs     = ibubs( : nbub), &
                        nbub_global = self%nbub_global )

       
        deallocate(ibubs)
        return
    end function

    !> Get the all ibubs present in 'self' or in 'bubsin'. 
    pure function Bubbles_get_merge_ibubs(self, bubsin) result(ibubs_result)
        class(Bubbles),           intent(in) :: self
        type(Bubbles),            intent(in) :: bubsin
        integer(INT32)                       :: nbub, ibub, i, ibubs(self%nbub+bubsin%nbub)
        integer(INT32),         allocatable  :: ibubs_result(:)

        nbub = 0
        do ibub = 1, self%get_nbub_global()
            if (bubsin%contains_ibub(ibub) .or. self%contains_ibub(ibub)) then
                nbub = nbub + 1 
                ibubs(nbub) = ibub
            end if
        end do 

        ibubs_result = ibubs(:nbub)
    end function

    !> Checks if 'self' could be an mege of 'bubbles1' and 'bubbles2'
    pure function Bubbles_is_merge_of(self, bubbles1, bubbles2) result(is_merge)
        class(Bubbles),           intent(in) :: self
        type(Bubbles),            intent(in) :: bubbles1
        type(Bubbles),            intent(in) :: bubbles2
        logical                              :: is_merge
        integer(INT32),         allocatable  :: ibubs(:)
        integer(INT32)                       :: i

        if (.not. allocated(self%ibubs)) then
            is_merge = .FALSE.
        else
            ibubs = bubbles1%get_merge_ibubs(bubbles2)
            if (size(ibubs) /= size(self%ibubs)) then
                is_merge = .FALSE.
            else
                is_merge = all([(ibubs(i) == self%ibubs(i), i = 1, size(ibubs))])
            end if
            deallocate(ibubs)
        end if
        
    end function


    !> Create a new Bubbles instance by combining two sets of Bubbles
    function Bubbles_merge(self, bubsin, lmax) result(new)
        class(Bubbles), intent(in)           :: self
        type(Bubbles), intent(in)            :: bubsin
        integer(INT32), optional, intent(in) :: lmax
        real(REAL64)                         :: centers(3, bubsin%nbub + self%nbub)
        real(REAL64)                         :: z(bubsin%nbub + self%nbub)
        type(Grid1DPointer)                  :: grids(bubsin%nbub + self%nbub)
        integer(INT32)                       :: ibubs(bubsin%nbub + self%nbub)
        type(Bubbles)                        :: new
        integer(INT32)                       :: lmax_, ibub, nbub, order_number

        nbub = 0
        do ibub = 1, self%get_nbub_global()
            if (bubsin%contains_ibub(ibub) .or. self%contains_ibub(ibub)) then
                nbub = nbub + 1 
                ibubs(nbub) = ibub
                
                order_number = self%get_order_number(ibub)
                if (order_number /= 0) then
                    centers(:, nbub) = self%centers(:, order_number)
                    grids(nbub)%p => self%gr(order_number)%p
                    z(nbub) = self%z(order_number)
                else
                    order_number = bubsin%get_order_number(ibub)
                    centers(:, nbub) = bubsin%centers(:, order_number)
                    grids(nbub)%p => bubsin%gr(order_number)%p
                    z(nbub) = bubsin%z(order_number)
                end if
            end if
        end do 

        if (present(lmax)) then
            lmax_ = lmax
        else
            lmax_ = max(bubsin%lmax, self%lmax)
        end if


        ! init empty bubbles
        new =  Bubbles( lmax      = lmax_, &
                        centers   = centers(:, : nbub), &
                        global_centers = self%global_centers, &
                        grids     = grids(:nbub), &
                        global_grids = self%global_grids, &
                        z         = z(:nbub), &
                        global_z  = self%global_z, &
                        k         = bubsin%k, &
                        ibubs     = ibubs( : nbub), &
                        nbub_global = self%nbub_global )

        return
    end function

    !> Allocate radial functions once everything else is in place
    subroutine Bubbles_alloc_bf(self)
        class(Bubbles), intent(inout), target :: self
        integer                               :: i 
#ifdef HAVE_CUDA
        type(C_PTR)                           :: c_pointer

        allocate(self%bf(self%nbub))
        do i = 1, self%nbub
            c_pointer = Bubbles_init_page_locked_f_cuda(self%lmax, self%gr(i)%p%get_shape())
            call c_f_pointer(c_pointer, self%bf(i)%p, [self%gr(i)%p%get_shape(), self%numf])
        end do
        
#else
        allocate(self%bf_data(self%nbub))
        allocate(self%bf(self%nbub))
        do i=1, self%nbub
            allocate(self%bf_data(i)%p( self%gr(i)%p%get_shape(), self%numf ), source = 0.0d0)
            self%bf(i)%p => self%bf_data(i)%p
        end do
#endif
    end subroutine

    !> Write bubbles into binary stream

    !> See \ref function3d_class::function3d_write "Function3D%write()" for a
    !! detailed description of the binary format).
    subroutine Bubbles_write(self, fd)
        class(Bubbles) :: self
        integer        :: ibub
        integer(INT32), intent(in) :: fd
        ! real(REAL64), pointer :: cell_scales(:)
        write(fd) self%nbub
        write(fd) self%nbub_global
        write(fd) self%lmax
        write(fd) self%k

        do ibub=1,self%nbub
            write(fd) self%z(ibub)
            write(fd) self%centers(:,ibub)
            write(fd) self%ibubs(ibub)
        end do 

        do ibub=1, self%nbub_global
            write(fd) self%global_centers(:, ibub)
            write(fd) self%global_z(ibub)
        end do

        do ibub=1,self%nbub
            write(fd) self%bf(ibub)%p(:,:)
        end do
    end subroutine
    
#ifdef HAVE_CUDA


    subroutine Bubbles_cuda_init(self)
        class(Bubbles), intent(inout) :: self
        type(Grid1D)                  :: grid
        type(REAL64_2D), allocatable  :: coeffs(:)
        integer                       :: i
  
        
        if (.not. allocated(self%cuda_interface)) then
            ! initialize the Bubbles cuda object
            self%cuda_interface = bubbles_init_cuda(self%get_nbub())
            coeffs=self%gr(1)%p%lip%coeffs(0)
            do i = 1, self%get_nbub()
                ! initialize the Bubble objects residing inside Bubbles
                call bubble_init_cuda( &
                                self%cuda_interface,  self%gr(i)%p%get_cuda_interface(), i, &
                                self%ibubs(i), self%centers(:, i), &
                                self%lmax, self%k, self%z(i), stream_container)
            end do
        end if
    end subroutine

    subroutine Bubbles_set_cuda_interface(self, cuda_interface)
        class(Bubbles), intent(inout)    :: self
        type(C_PTR),    intent(in)       :: cuda_interface

        ! take correct sub bubbles of the input 'cuda_inteface' and set it as the cuda interface of 'self'
        self%cuda_interface = Bubbles_get_sub_bubbles_cuda(cuda_interface, self%ibubs, size(self%ibubs))

    end subroutine

    subroutine Bubbles_cuda_upload_all(self, cuda_interface)
        class(Bubbles),           intent(in)    :: self
        type(C_PTR),    optional, intent(in)    :: cuda_interface
        integer                                 :: i
    
        if (present(cuda_interface)) then
            ! because we are uploading all bubbles data to all devices of all nodes, we can safely
            ! split the possible integration to all processors
            call bubbles_set_processor_configuration_cuda(cuda_interface, iproc, nproc)
            do i = 1, self%get_nbub()
                call bubble_upload_all_cuda(cuda_interface, self%ibubs(i), self%lmax, self%k, self%bf(i)%p)
            end do
        else if (allocated(self%cuda_interface)) then
            ! because we are uploading all bubbles data to all devices of all nodes, we can safely
            ! split the possible integration to all processors
            call bubbles_set_processor_configuration_cuda(self%cuda_interface, iproc, nproc)
            do i = 1, self%get_nbub()
                call bubble_upload_all_cuda(self%cuda_interface, self%ibubs(i), self%lmax, self%k, self%bf(i)%p)
            end do
        else
            print *, "ERROR: Trying to upload bubbles (@Bubbles_cuda_upload_all) &
                      &without a valid cuda_interface given as input."
            stop
        end if
    end subroutine

    !> Get a sub bubbles of the Cuda/C++ object and return it
    function Bubbles_cuda_get_sub_bubbles(self, cuda_interface) result(new)
        class(Bubbles),           intent(in) :: self
        type(C_PTR),    optional, intent(in) :: cuda_interface
        type(C_PTR)                          :: new
        
        if (present(cuda_interface)) then
            new = Bubbles_get_sub_bubbles_cuda(cuda_interface, self%ibubs, size(self%ibubs))
        else if (allocated(self%cuda_interface)) then
            new = Bubbles_get_sub_bubbles_cuda(self%cuda_interface, self%ibubs, size(self%ibubs))
        else
            print *, "ERROR: Trying to get cuda sub bubbles (@Bubbles_cuda_get_sub_bubbles) &
                      &without a valid cuda_interface given as input."
            stop
        end if
    end function

    subroutine Bubbles_cuda_upload(self, cuda_interface)
        class(Bubbles),           intent(in) :: self
        type(C_PTR),    optional, intent(in) :: cuda_interface
        integer                              :: i
        
        ! for now, we are calculating all data of bubble multiplication on all nodes 
        ! (but the calculation is split between all devices (GPUs) of the node).
        ! Thus we are setting the node configuration to processor_order_number: 0 and 
        ! number of processors: 1
        call bubbles_set_processor_configuration_cuda(cuda_interface, 0, 1)
        if (present(cuda_interface)) then
            do i = 1, self%get_nbub()
                call bubble_upload_cuda(cuda_interface, self%ibubs(i), self%lmax, self%bf(i)%p)
            end do
        else if (allocated(self%cuda_interface)) then
            do i = 1, self%get_nbub()
                call bubble_upload_cuda(self%cuda_interface, self%ibubs(i), self%lmax, self%bf(i)%p)
            end do
        else
            print *, "ERROR: Trying to upload bubbles (@Bubbles_cuda_upload) &
                      &without a valid cuda_interface given as input."
            stop
        end if
    end subroutine

    subroutine Bubbles_cuda_upload_bubble(self, i, cuda_interface)
        class(Bubbles), intent(in)    :: self
        integer,        intent(in)    :: i
        type(C_PTR), optional         :: cuda_interface
        
        
        ! for now, we are calculating all data of bubble multiplication on all nodes 
        ! (but the calculation is split between all devices (GPUs) of the node).
        ! Thus we are setting the node configuration to processor_order_number: 0 and 
        ! number of processors: 1
        call bubbles_set_processor_configuration_cuda(cuda_interface, 0, 1)
        if (present(cuda_interface)) then
            call bubble_upload_cuda(cuda_interface, self%ibubs(i), self%lmax, self%bf(i)%p)
        else if (allocated(self%cuda_interface)) then
            call bubble_upload_cuda(self%cuda_interface, self%ibubs(i), self%lmax, self%bf(i)%p)
        else
            print *, "ERROR: Trying to upload a bubble (@Bubbles_cuda_upload_bubble) &
                      &without a valid cuda_interface given as input."
            stop
        end if
    end subroutine

    subroutine Bubbles_cuda_download(self)
        class(Bubbles), intent(inout) :: self
        integer                       :: i
  
        
        if (allocated(self%cuda_interface)) then
            do i = 1, self%get_nbub()
                !call bubble_download(self%cuda_interface, ibub, self%bf(ibub)%p)
            end do
        end if
    end subroutine
    
    subroutine Bubbles_cuda_destroy(self)
        class(Bubbles), intent(inout) :: self
        integer                       :: i
        
        if (allocated(self%cuda_interface)) then
            call bubbles_destroy_cuda(self%cuda_interface)
            deallocate(self%cuda_interface)
        end if
    end subroutine
#endif 

    !> Construct Bubbles representing Taylor expansions at the Bubbles centers.

    !> \f[
    !!  \tilde{f}_{lm}^A(r_A) =
    !!  \sum_{|\boldsymbol{\kappa}|\le K} C^{\boldsymbol{\kappa}}_{lm}
    !!  \frac {(\partial^{\boldsymbol{\kappa}} {f^{\ne}_A})(\vec{R}_A)}
    !!       {\boldsymbol{\kappa}!}r_A^{|\boldsymbol{\kappa}|}
    !! \f]
    !! \f[
    !! f^{\ne}_A(\vec{r})=
    !! f^\Delta(\vec{r})+\sum_{B\ne A}\sum_{lm}f_{Blm} Y_{lm}(\theta_B,\phi_B).
    !! \f]
    function Bubbles_make_taylor(self, bubble_contaminants, cube_contaminants, &
                                 tmax, non_overlapping) result(new)
        class(Bubbles), intent(in)    :: self
        !> The coefficients of the Taylor expansion 
        real(REAL64), intent(in)      :: bubble_contaminants(:,:)
        !> The coefficients of the Taylor expansion 
        real(REAL64), intent(in)      :: cube_contaminants(:,:)
        !> Truncation order of the Taylor expansion \f$K\f$
        integer, intent(in)           :: tmax
        !
        logical, intent(in)           :: non_overlapping

        type(Bubbles) :: new

        real(REAL64), allocatable :: f(:), weights(:), rpow(:, :)

        integer :: nbub, i, j, l,m,n,ibub

        type(CartIter) :: cart
        logical :: done_in, continue_iteration, continue_c2s_iteration
        integer(INT32) :: icart, kappa(3), kappa_abs
        real(REAL64) :: coeff

        real(REAL64), allocatable :: one_per_kappa_factorial(:)

        type(Cart2SphIter) :: c2s
        call bigben%split("Make taylor")
        ! Initialize 1/|kappa|!
        allocate(one_per_kappa_factorial(0:tmax))
        one_per_kappa_factorial(0)=1.d0
        do kappa_abs=1,tmax
            one_per_kappa_factorial(kappa_abs)=one_per_kappa_factorial(kappa_abs-1)/kappa_abs ! 1/j!
        end do
        ! create an empty copy of self
        call bigben%split("Make taylor - coefficients")
        new=Bubbles(lmax=max(self%lmax,tmax), &
                    centers = self%global_centers, &
                    global_centers = self%global_centers, &
                    grids = self%global_grids, &
                    global_grids = self%global_grids, &
                    z = self%global_z, &
                    global_z = self%global_z, &
                    k=self%k, &
                    ibubs = [(i, i = 1, self%nbub_global)], &
                    nbub_global = self%nbub_global)

        c2s=Cart2SphIter(new%lmax)
        call bigben%stop()
        do i=1,new%nbub
            ibub = new%ibubs(i)
            ! Transform contaminants into the spherical base
            allocate(f(new%gr(i)%p%get_shape()))
            allocate(weights(new%gr(i)%p%get_shape()))
            allocate(rpow(new%gr(i)%p%get_shape(), 0:tmax))
            if (non_overlapping) then
                weights = w( new%gr(i)%p%get_coord() )
            else
                weights = w2( new%gr(i)%p%get_coord() )
            end if

            do kappa_abs = 0, tmax
                rpow(:, kappa_abs) = new%rpow(i, kappa_abs - new%k)  &
                                     * weights                       &
                                      ! * 1 / (\kappa !)
                                     * one_per_kappa_factorial(kappa_abs ) 
            end do

            ! Iterate over derivative terms
            cart=CartIter(ndim=3,maximum_modulus=tmax)
            icart=0
            call cart%next(kappa, continue_iteration)
            do while(continue_iteration)
                icart=icart+1

                ! Conmpute c * r**k, c is the actual coefficient in the
                ! Taylor series
                kappa_abs = sum(kappa)
                f = (bubble_contaminants(icart,i) + cube_contaminants(icart,i))   &
                       ! * r^(|\kappa|) * weights * 1 / |\kappa|!
                       * rpow(:, kappa_abs)   

                ! Iterate over all spherical harmonics to which this
                ! Cartesian term contributes
                call c2s%init_loop_over_sph(kappa)
                call c2s%next_sph(coeff,l,m, continue_c2s_iteration)
                do while(continue_c2s_iteration)
                    new%bf(i)%p(:, idx(l, m)) =  new%bf(i)%p(:, idx(l, m)) + coeff * f
                    call c2s%next_sph(coeff,l,m, continue_c2s_iteration)
                end do
                call cart%next(kappa, continue_iteration)
            end do
            deallocate(rpow)
            deallocate(weights)
            deallocate(f)
        end do
        deallocate(one_per_kappa_factorial)
        call bigben%stop()
        call c2s%destroy()
    end function

    !> Damping factor
    !! \f$\omega(r)=\frac{1}{2}\mathrm{erfc}\left(r-\frac{2}{r}\right)\f$
    elemental function w(r)
        real(REAL64), intent(in) :: r
        real(REAL64) :: w, b

        b = r+0.d0
        if(r>epsilon(r)) then
            w=0.5d0*erfc(1d0*b-2.0d0/(b))
        else
            w=1.d0
        end if
    end function
    
    !> Damping factor
    !! \f$\omega(r)=\frac{1}{2}\mathrm{erfc}\left(r-\frac{2}{r}\right)\f$
    elemental function w2(r)
        real(REAL64), intent(in) :: r
        real(REAL64) :: w2, b

        b = r+0.d0
        if(r>epsilon(r)) then
            w2=0.5d0*erfc(1d0*b-2d0/(b))
        else
            w2=1.d0
        end if
    end function

    function alpha_bubbles(dens, contam_cube, lmax) result(new)
        ! density contaminants x erfc(r - 1/r)
        type(Bubbles) :: new,dens
        real(REAL64) :: contam_cube(:,:)

        integer(INT32) :: lmax,ibub,ilm
        real(REAL64), pointer :: r(:),w(:)
!        real(REAL64), parameter :: p=0.8, q=2.d0
        real(REAL64), parameter :: p=1.d0, q=1.d0

        new=dens%make_taylor(dens%get_contaminants(lmax), contam_cube, lmax, .FALSE.)
        do ibub=1,new%nbub
            r => new%gr(ibub)%p%get_coord()
            allocate(w(size(r)))
            w(1)  =  1.d0
            w(2:) = 0.5d0 * erfc( p*q**2*r(2:)-p/r(2:))
            do ilm=1,new%numf
                new%bf(ibub)%p(:,ilm)=new%bf(ibub)%p(:,ilm) * w
            end do
        end do

        return

    end function

    function beta_bubbles(dens,lmax) result(new)
        ! density bubbles x nuclear contaminants
        type(Bubbles) :: new, dens
        real(REAL64),allocatable :: contam(:,:), cube_contam(:, :)

        integer(INT32) :: lmax

        allocate(contam( (lmax+1)*(lmax+2)*(lmax+3)/6, dens%nbub))
        allocate(cube_contam( (lmax+1)*(lmax+2)*(lmax+3)/6, dens%nbub), source = 0.0d0)
        
        contam = dens%get_nuclear_contaminants(lmax)
        contam = contam + dens%get_contaminants(lmax)
        new    = dens * dens%make_taylor(contam, cube_contam, lmax, .FALSE.)

        deallocate(contam)
        deallocate(cube_contam)
        return

    end function

    !> Destructor
    pure subroutine Bubbles_destroy(self)
        class(Bubbles), intent(inout) :: self
        integer(INT32)                :: ibub

        !call pdebug('Destroying bubbles', 1)
        if (allocated(self%centers)) deallocate(self%centers)
        if (allocated(self%global_centers)) deallocate(self%global_centers)
        if (allocated(self%z))       deallocate(self%z)
        if (allocated(self%global_z))deallocate(self%global_z)
        if (allocated(self%gr)) then
            do ibub=1,self%nbub
                if (allocated(self%bf_data)) deallocate(self%bf_data(ibub)%p)
#ifdef HAVE_CUDA
                if (allocated(self%bf)) call Bubbles_destroy_page_locked_f_cuda(c_loc(self%bf(ibub)%p))
#else
                nullify(self%bf(ibub)%p)
#endif
            end do
            if (allocated(self%bf))      deallocate(self%bf)
            if (allocated(self%bf_data)) deallocate(self%bf_data)
            deallocate(self%gr)
        end if
        if (allocated(self%ibubs)) deallocate(self%ibubs)
#ifdef HAVE_CUDA
        if (allocated(self%cuda_interface)) deallocate(self%cuda_interface)
#endif
        if (allocated(self%global_grids)) deallocate(self%global_grids)
        self%nbub=0
        self%numf=0
        self%lmax=-1
    end subroutine

    pure function Bubbles_get_lmax(self) result(res)
        class(Bubbles), intent(in) :: self
        integer(INT32) :: res

        res=self%lmax
        return
    end function

    pure function Bubbles_get_nbub(self) result(res)
        class(Bubbles), intent(in) :: self
        integer(INT32) :: res

        res=self%nbub
        return
    end function

    pure function Bubbles_get_nbub_global(self) result(res)
        class(Bubbles), intent(in) :: self
        integer(INT32) :: res

        res=self%nbub_global
        return
    end function

    pure function Bubbles_get_k(self) result(res)
        class(Bubbles), intent(in) :: self
        integer(INT32) :: res

        res=self%k
        return
    end function

    pure function Bubbles_get_order_number(self, ibub) result(res)
        class(Bubbles), intent(in) :: self
        integer(INT32), intent(in) :: ibub 
        integer(INT32)             :: i, res

        res = 0
        do i = 1, self%nbub
            if (self%ibubs(i) == ibub) then
                res = i
                return
            end if
        end do
    end function

    ! Get the bubbles number for bubble with order number 'order_number'
    pure function Bubbles_get_ibub(self, order_number) result(ibub)
        class(Bubbles), intent(in) :: self
        integer(INT32), intent(in) :: order_number
        integer(INT32)             :: ibub

        ibub = self%ibubs(order_number)
    end function

    ! Get the bubbles number for bubble with order number 'order_number'
    function Bubbles_get_ibubs(self) result(ibubs)
        class(Bubbles), intent(in), target :: self
        integer(INT32), pointer            :: ibubs(:)

        ibubs => self%ibubs
    end function

    ! Get the bubbles number for bubble with order number 'order_number'
    pure function Bubbles_contains_ibub(self, ibub) result(contains_ibub)
        class(Bubbles), intent(in) :: self
        integer(INT32), intent(in) :: ibub
        integer(INT32)             :: i
        logical                    :: contains_ibub

        contains_ibub = .FALSE.
        do i = 1, self%get_nbub()
            if (self%ibubs(i) == ibub) then
                contains_ibub = .TRUE.
            end if
        end do
    end function

    pure function Bubbles_get_centers_pick(self,order_number) result(res)
        class(Bubbles), intent(in) :: self
        integer(INT32), intent(in) :: order_number
        real(REAL64) :: res(3)

        res=self%centers(:, order_number)
        return
    end function

    pure function Bubbles_get_centers_pick_many(self, ibubs) result(res)
        class(Bubbles), intent(in) :: self
        integer(INT32), intent(in) :: ibubs(:)
        real(REAL64)               :: res(3, size(ibubs))
        integer(INT32)             :: i

        forall (i = 1:size(ibubs))
            res(:, i) = self%global_centers(:, ibubs(i))
        end forall
        return
    end function

    function Bubbles_get_centers_all(self) result(res)
        class(Bubbles), intent(in),target :: self
        real(REAL64),pointer :: res(:,:)

        res=>self%centers(:,:)
        return
    end function

    pure function Bubbles_get_global_centers_pick(self,order_number) result(res)
        class(Bubbles), intent(in) :: self
        integer(INT32), intent(in) :: order_number
        real(REAL64) :: res(3)

        res=self%global_centers(:, order_number)
        return
    end function

    pure function Bubbles_get_global_centers_pick_many(self, ibubs) result(res)
        class(Bubbles), intent(in) :: self
        integer(INT32), intent(in) :: ibubs(:)
        real(REAL64)               :: res(3, size(ibubs))
        integer(INT32)             :: i

        forall (i = 1:size(ibubs))
            res(:, i) = self%global_centers(:, ibubs(i))
        end forall
        return
    end function

    function Bubbles_get_global_centers_all(self) result(res)
        class(Bubbles), intent(in),target :: self
        real(REAL64),pointer :: res(:,:)

        res=>self%global_centers(:,:)
        return
    end function

    pure function Bubbles_get_z_pick(self,ibub) result(res)
        class(Bubbles), intent(in) :: self
        integer(INT32), intent(in) :: ibub
        real(REAL64) :: res

        res=self%z(ibub)
        return
    end function

    function Bubbles_get_z_all(self) result(res)
        class(Bubbles), intent(in),target :: self
        real(REAL64),pointer :: res(:)

        res=>self%z(:)
        return
    end function

    pure function Bubbles_get_global_z_pick(self,ibub) result(res)
        class(Bubbles), intent(in) :: self
        integer(INT32), intent(in) :: ibub
        real(REAL64) :: res

        res=self%global_z(ibub)
        return
    end function

    function Bubbles_get_global_z_all(self) result(res)
        class(Bubbles), intent(in),target :: self
        real(REAL64),pointer :: res(:)

        res=>self%global_z(:)
        return
    end function

    function Bubbles_get_grid_pick(self,i) result(res)
        class(Bubbles),  target  :: self
        type(Grid1D),    pointer :: res
        integer                  :: i

        res=>self%gr(i)%p
        return
    end function

    function Bubbles_get_grid_all(self) result(res)
        class(Bubbles)        :: self
        type(Grid1DPointer)   :: res(size(self%gr))

        res = self%gr
        return
    end function

    function Bubbles_get_global_grid_pick(self,ibub) result(res)
        class(Bubbles),  target  :: self
        type(Grid1D),    pointer :: res
        integer                  :: ibub

        res=>self%global_grids(ibub)%p
        return
    end function

    function Bubbles_get_global_grid_all(self) result(res)
        class(Bubbles)        :: self
        type(Grid1DPointer)   :: res(size(self%global_grids))

        res = self%global_grids
        return
    end function

    function Bubbles_get_f_pick(self,ibub,l,m) result(f_p)
        class(Bubbles), target :: self
        integer :: ibub,l,m
        real(REAL64),pointer :: f_p(:)

        f_p=>self%bf(ibub)%p(:,idx(l,m))
        return
    end function

    function Bubbles_get_f_all(self,ibub) result(f_p)
        class(Bubbles), target :: self
        integer :: ibub
        real(REAL64),pointer :: f_p(:,:)

        f_p=>self%bf(ibub)%p
        return
    end function

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!  ASSIGNMENT                                      !!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !> Set all coefficients of the bubble to `a`
    pure subroutine Bubbles_assign_real(self,a)
        class(Bubbles), intent(inout) :: self
        !> Constant
        real(REAL64), intent(in) :: a

        integer :: ibub

        forall (ibub=1:self%nbub)
            self%bf(ibub)%p(:,:)=a
        end forall

        return

    end subroutine
    
    !> Copy everything else but the cuda interfaces
    !pure subroutine Bubbles_assign(self, b)
    !    type(Bubbles), intent(inout) :: self
    !    type(Bubbles), intent(in) :: b
    !    self%nbub = b%nbub
    !    self%nbub_global = b%nbub_global
    !    if (allocated(b%ibubs)) self%ibubs = b%ibubs
    !    self%lmax = b%lmax
    !    self%numf = b%numf
    !    self%k = b%k
    !    if (allocated(b%centers)) self%centers = b%centers
    !    if (allocated(b%z)) self%z = b%z
    !    if (allocated(b%gr)) self%gr = b%gr
    !    if (allocated(b%bf)) self%bf = b%bf
    !end subroutine

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!  EVALUATION                                      !!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    !> Evaluate bubble \c ibub at given points \f$\{\mathbf{r}_i\}\f$.
    function Bubbles_eval_pick(self, points, dv, ibub, lmin) result(res)
        class(Bubbles),intent(in)  :: self
        !> 3 x N \c REAL64 array.
        !! \f$\{\mathbf{r}_i\}\f$; \c points(:,i)= \f$\mathbf{r}_i\f$
        real(REAL64), intent(in)   :: points(:,:)
        !> Derivative indices
        integer(INT32), intent(in) :: dv(3)
        !> Number of the bubble to be evaluated
        integer(INT32), intent(in) :: ibub
        !> Minimum angular momentum value that is evaluated 
        integer,        intent(in), optional :: lmin
        real(REAL64)               :: res(size(points,2)), tic, toc

        real(REAL64) :: coordinates(3,size(points,2))
        real(REAL64) :: distances(1, size(points, 2))
        real(REAL64) :: rfs(size(points,2), 0:sum(dv)), ys(size(points,2), 0:sum(dv))

        real(REAL64), pointer :: r0(:)
        real(REAL64), pointer :: r(:,:)

        real(REAL64), allocatable :: rfun(:), rpow(:)
        real(REAL64), allocatable :: rf_tmp(:,:)

        integer :: i,l,m,d,j,minimum_l

        type(Y_t),pointer :: y
        type(YBundle) :: yb

        type(Interpolator1D) :: interpol2
        type(Interpolator) :: interpol1
        type(Interpolator) :: interpol0

        if (present(lmin)) then
            minimum_l = lmin
        else
            minimum_l = 0
        end if

        d=sum(dv)

        ! Calculate relative coordinates of the points
        forall(i=1:size(points,2)) 
            coordinates(:,i)=points(:,i)-self%centers(:,ibub)
            distances(1, i) = sqrt(sum(coordinates(:, i)*coordinates(:, i)))
        end forall


        yb=YBundle(self%lmax)
        res=0.d0
        ! Iterate over spherical harmonics
        if(self%k>=0) then
            interpol2=Interpolator1D(self%gr(ibub)%p, d)
            rpow = self%rpow(ibub,self%k)
            do l=minimum_l,self%lmax
                do m=-l,l
                    rfun=self%get_f(ibub,l,m)
                    rfs=interpol2%eval( rpow * rfun, distances(1, :))
                    y  =>yb%pick(l,m)
                    ys = y%eval_coeffs(coordinates, distances(1, :), dv)
                    res=res+sum(ys*rfs(:,:d),dim=2)
                end do
            end do
            call interpol2%destroy()
            deallocate(rpow)
        else
            r0=>self%gr(ibub)%p%get_coord()
            allocate(r(1,size(r0)))
            r(1,:)=r0
            
            interpol1=Interpolator(self%gr(ibub:ibub), 1)
            interpol0=Interpolator(self%gr(ibub:ibub), 0)
            do l=minimum_l,self%lmax
                do m=-l,l
                    
                    rfun=self%get_f(ibub,l,m)
                    ! For each derivative
                    rfs=0.d0
                    do j=0,d
                        ! Construct radial function
                        ! Interpolate at desired points
                        rfs(:,j:j)= interpol0%eval(rfun, distances)
                        where(distances(1, :)>0.d0) &
                                rfs(:,j)=rfs(:,j) * distances(1, :)**(self%k-j)
                        if(j==d) exit
                        ! Compute f'(r)
                        rf_tmp=interpol1%eval(rfun, r)
                        ! Sanity check (should be 0)
                        ! d/dr r^k f(r) =
                        ! r^{k-1} *       (     k   *   f(r)     +   r   *  f'(r)   )
                        rfun= (self%k-j)*rfun(:)   + r(1,:)*rf_tmp(:,2)
                        deallocate(rf_tmp)
                    end do
                    
                    !\sum_i { P_i(x,y,z,r) * f^{(i)}(r) }
                    ! sum over all centers and derivatives in one go
                    y  =>yb%pick(l,m)
                    res=res+sum(y%eval_coeffs(coordinates, distances(1, :), dv)*rfs(:,:d),dim=2)
                end do
            end do
            nullify(r0)
            
            call interpol1%destroy()
            call interpol0%destroy()
            deallocate(r)
        end if
        call yb%destroy()
        deallocate(rfun)
        return
    end function

    

    !> Evaluate all bubbles at given points \f$\{\mathbf{r}_i\}\f$.
    function Bubbles_eval_all(self,points, dv, ibubs, lmin) result(res)
        class(Bubbles) :: self
        real(REAL64), intent(in)      :: points(:,:)
        integer, intent(in), optional  :: dv(3)
        integer, intent(in), optional :: ibubs(:)
        !> Minimum angular momentum value that is evaluated 
        integer,        intent(in), optional :: lmin

        integer                       :: minimum_l

        real(REAL64)                  :: res(size(points,2))
        integer, allocatable          :: ibubbles(:)

        integer(INT32) :: ibub, i, order_number

        if (present(ibubs)) then
            ibubbles = ibubs
        else
            ibubbles = self%ibubs
        end if

        if (present(lmin)) then
            minimum_l = lmin
        else
            minimum_l = 0
        end if

        res=0.d0
        
        if (present(dv)) then
            ! Iterate over bubbles
            do i=1, size(ibubbles)
                
                ! get the number of the bubble
                ibub = ibubbles(i)
                ! get the order number of ibub at this bubbles
                order_number = self%get_order_number(ibub)
                    res = res+self%eval(points,dv,order_number, lmin = minimum_l)
                
            end do
        else
            ! Iterate over bubbles
            do i=1, size(ibubbles)
                ! get the number of the bubble
                ibub = ibubbles(i)
                ! get the order number of ibub at this bubbles
                order_number = self%get_order_number(ibub)

                res = res+self%eval(points,order_number, lmin = minimum_l)
            end do
        end if
        if (allocated(ibubbles)) then
            deallocate(ibubbles)
        end if
    end function

    !> Evaluate single Bubble with order number 'order_number' at 'points'. This method 
    !! can not evaluate derivatives. For that, use 'Bubbles_eval_pick'.
    function Bubbles_simple_eval_pick(self, points, order_number, lmin) result(res)
        class(Bubbles), intent(in)           :: self
        !> The points where the bubble is evaluated
        real(REAL64),   intent(in)           :: points(:,:)
        !> The order number of the evaluated bubble (in the arrays of this type)
        integer,        intent(in)           :: order_number
        !> Minimum angular momentum value that is evaluated 
        integer,        intent(in)           :: lmin

        ! Other parameters
        type(SimpleInterpolator1D)    :: interpolator1d
        real(REAL64)                  :: res(size(points,2))

        real(REAL64)                  :: coordinates(3,size(points,2))
        real(REAL64)                  :: distances(size(points, 2))
        real(REAL64)                  :: harmonic_coefficients(size(points,2),&
                                                       (lmin**2 + 1) : (self%lmax+1)**2 )


        real(REAL64)                  :: rfun(self%gr(order_number)%p%get_shape()), &
                                         rpow(self%gr(order_number)%p%get_shape())
        real(REAL64), allocatable     :: bubble_values(:, :)

        integer                       :: i,l,m

        type(RealSphericalHarmonics)  :: spherical_harmonics

        type(Interpolator) :: interpol1

        distances = 0.0d0
        ! Calculate relative coordinates of the points
        forall(i=1:size(points,2)) 
            coordinates(:,i)=points(:,i)-self%centers(:,order_number)
            distances(i) = sqrt(sum(coordinates(:, i)*coordinates(:, i)))
        end forall


        
        res=0.d0
        interpolator1d = SimpleInterpolator1D(self%gr(order_number)%p)
        spherical_harmonics = RealSphericalHarmonics(self%lmax, lmin = lmin)
        
        harmonic_coefficients = spherical_harmonics%eval(coordinates, distances)
        call spherical_harmonics%destroy()
        if (self%k == 0) then
            res = sum(interpolator1d%eval(                                                &
                      self%bf(order_number)%p(:, lmin * lmin + 1: ), distances) &
                     * harmonic_coefficients, dim = 2)
        else if (self%k > 0) then
            bubble_values = self%bf(order_number)%p(:, lmin * lmin + 1: )
            rpow = self%rpow(order_number,self%k)
            forall (i = 1: (self%lmax + 1) ** 2)
                bubble_values(:, i) = bubble_values(:, i) * rpow
            end forall
            res = sum(interpolator1d%eval(bubble_values, distances) &
                     * harmonic_coefficients, dim = 2)
            deallocate(bubble_values) 
        else ! self%k < 0
            bubble_values = self%bf(order_number)%p(:, lmin * lmin + 1: )
            rpow = self%rpow(order_number,self%k)
            forall (i = 1: (self%lmax + 1) ** 2)
                bubble_values(:, i) = bubble_values(:, i) * rpow
            end forall
            res = sum(interpolator1d%eval(bubble_values, distances) * harmonic_coefficients, dim = 2)
            deallocate(bubble_values)
        end if
        call interpolator1d%destroy()
    end function

 
    !> Evaluate all bubbles at points given by the Cartesian product of three
    !! 1-dimensional grids:
    !! \f$\{\mathbf{r}_i\} = \{x_i\} \times \{y_i\} \times \{z_i\}\f$.
    function Bubbles_eval_3dgrid(self, grid, ibubs, lmin) result(cubeout)
        class(Bubbles), intent(inout)        :: self
        !> The 3d grid on which bubbles are evaluated
        type(Grid3D),   intent(in)           :: grid
        !> The global bubbles ids that are evaluated
        integer,        intent(in), optional :: ibubs(:)
        !> Minimum angular momentum value that is evaluated on the 3d grid
        integer,        intent(in), optional :: lmin
        !> The result cube (shape like the grid)
        real(REAL64), allocatable            :: cubeout(:,:,:)

        integer(INT32)                       :: minimum_l
        integer, allocatable                 :: ibubbles(:)
        logical                              :: cuda_preinited
#ifdef HAVE_CUDA
        type(Grid3D)                         :: temp_grid
        type(CudaCube)                       :: cuda_injection
#endif

        if (present(ibubs)) then
            ibubbles = ibubs
        else
            ibubbles = self%ibubs
        end if

        allocate(cubeout(grid%axis(X_)%get_shape(),  &
                         grid%axis(Y_)%get_shape(),  &
                         grid%axis(Z_)%get_shape()), source = 0.0d0)
#ifdef HAVE_CUDA
        if (present(lmin)) then
            minimum_l = lmin
        else
            minimum_l = 0
        end if       

        ! init the CudaCube to hold the injected stuff and set it to zero
        cuda_injection = CudaCube(cubeout, all_memory_at_all_devices = .FALSE.)
        call cuda_injection%set_to_zero()

        ! create a temporary grid as grid is intent(in)
        temp_grid = grid

        ! do the cuda injection & download the result
        call self%eval_3dgrid_cuda(temp_grid, cuda_injection, lmin=minimum_l)
        call cuda_injection%download()

        ! synchronize cpus with the gpus and clean up
        call CUDASync_all()
        call temp_grid%destroy()
        call cuda_injection%destroy()
        
#else
        cubeout = reshape(self%eval(grid%get_all_grid_points(), ibubs = ibubbles, lmin = lmin), &
                          [grid%axis(X_)%get_shape(), grid%axis(Y_)%get_shape(), grid%axis(Z_)%get_shape()])
#endif
        if (allocated(ibubbles)) then
            deallocate(ibubbles)
        end if
        return
    end function


#ifdef HAVE_CUDA
    !> Evaluate all bubbles at points given by the Cartesian product of three
    !! 1-dimensional grids:
    !! \f$\{\mathbf{r}_i\} = \{x_i\} \times \{y_i\} \times \{z_i\}\f$.
    subroutine Bubbles_eval_3dgrid_cuda(self, grid, cuda_cube, lmin)
        class(Bubbles), intent(inout)        :: self
        !> The 3d grid on which bubbles are evaluated, the grid's cuda-interface must be inited and uploaded
        type(Grid3D),   intent(inout)       :: grid
        !> The global bubbles ids that are evaluated
        type(CudaCube), intent(inout)       :: cuda_cube
        !> Minimum angular momentum value that is evaluated on the 3d grid
        integer,        intent(in), optional :: lmin
        logical                              :: cuda_preinited, grid_preinited
        integer                              :: minimum_l

        if (present(lmin)) then
            minimum_l = lmin
        else
            minimum_l = 0
        end if        
        ! try to upload the bubbles to GPU, if they are 
        ! already uploaded, this does nothing
        cuda_preinited = allocated(self%cuda_interface)
        grid_preinited = allocated(grid%cuda_interface)

        call self%cuda_upload_all(self%get_cuda_interface())
        call Bubbles_inject_cuda(self%get_cuda_interface(), grid%get_cuda_interface(), &
                                             minimum_l, cuda_cube%cuda_interface)

        call CUDASync_all()
        if (.not. grid_preinited) call grid%cuda_destroy()
        if (.not. cuda_preinited) call self%cuda_destroy()

        return
    end subroutine
#endif

    !> Interpolates the radial part of the all the bubbles at the points at
    !! the given *relative* positions.

    !> Rationale: This function is used when different bubbles must be
    !! interpolated at the same points. We don't want to repeat things such as
    !! computing the relative coordinates and evaluating the spherical
    !! harmonics. However, this should be better implemented as a function
    !! that interpolates a bunch of functions in one go.
    function Bubbles_eval_radial(self,dists) result(res)
        class(Bubbles),intent(in) :: self
        real(REAL64), intent(in) :: dists(:)
        real(REAL64) :: res(self%numf,self%nbub)
        integer(INT32) :: ilm, ibub
        type(SimpleInterpolator1D) :: interpol

        do ibub=1,self%nbub
            interpol=SimpleInterpolator1D(self%gr(ibub)%p)
            do ilm=1, self%numf
                res(ibub,ilm:ilm)=interpol%eval(self%bf(ibub)%p(:,ilm),dists(ibub:ibub))
            end do
            call interpol%destroy()
        end do
        return
    end function

    function Bubbles_get_radial_derivatives(self, ibub, distances, maximum_derivative_order) result(res)
        class(Bubbles),intent(in) :: self
        integer,      intent(in)  :: ibub
        real(REAL64), intent(in)  :: distances(:)
        integer,      intent(in)  :: maximum_derivative_order
        type(Interpolator1D)      :: interpol1, interpol2
        type(SimpleInterpolator1D):: interpol0
        integer                   :: l, m, j, ilm
        real(REAL64)              :: res(size(distances), 0 : maximum_derivative_order, &
                                         (self%lmax+1)**2)
        real(REAL64), pointer     :: r(:)
        real(REAL64), allocatable :: rfun(:), rpow(:), rfun_all(:, :)
        real(REAL64), allocatable :: rf_tmp(:, :, :)

        res=0.d0
        
        if (self%k >= 0) then
            interpol2=Interpolator1D(self%gr(ibub)%p, maximum_derivative_order)
            rpow = self%rpow(ibub,self%k)
            ! get all radial functions
            allocate(rfun_all(size(self%bf(ibub)%p, 1), size(self%bf(ibub)%p, 2)))
            ! Iterate over spherical harmonics and multiply with r^k
            forall (ilm = 1 : (self%lmax+1)**2)
                ! get r^k * f_Alm(r)
                rfun_all(:, ilm) = self%bf(ibub)%p(:, ilm) * rpow
            end forall

            ! interpolate r^k * f_A(r), residing now in rfun_all, at 'distances'
            res = interpol2%eval( rfun_all, distances)
            

            deallocate(rpow)
            deallocate(rfun_all)
            call interpol2%destroy()
        else
            ! NOTE: if the derivatives would be calculated for k < 0 values
            ! with the above method, severe numerical errors would occur
            ! NOTE: there might be an error here

            r => self%gr(ibub)%p%get_coord()
            ! get all radial functions
            rfun_all = self%bf(ibub)%p 

            ! init the interpolators 1 simple interpolator for the 
            ! interpolation and another for getting the first derivatives
            interpol0=SimpleInterpolator1D(self%gr(ibub)%p)
            interpol1=Interpolator1D(self%gr(ibub)%p, 1)
            
            ! loop over all derivative orders, do the calculation using
            ! derivative of product formula
            do j = 0, maximum_derivative_order
                ! Construct radial function
                ! Interpolate at desired points
                res(:, j, :) = interpol0%eval(rfun_all, distances)

                forall (ilm = 1 : (self%lmax+1)**2)
                    where (distances(:)>0.d0) &
                        res(:, j, ilm) = res(:, j, ilm) * distances(:)**(self%k-j)
                end forall
                
                if (j == maximum_derivative_order) exit

                ! Compute f'(r) for all l,m-pairs at points specified in 'distances'
                ! NOTE: there might be an error here, there is slight variations 
                ! caused by this approach when compared with evaluating the 
                ! derivative for the entire rfun_all with
                !rf_tmp  = interpol1%eval(rfun_all, r)
                rf_tmp=interpol1%eval_point_cells_array(rfun_all, distances)
                       
                forall (ilm = 1 : (self%lmax+1)**2)
                    ! d/dr r^k f(r) =
                    ! r^{k-1} *       (     k   *   f(r)     +   r   *  f'(r)   )
                    rfun_all(:, ilm) = (self%k-j) * rfun_all(:, ilm) + r*rf_tmp(:, 2, ilm)
                    !res(:, j+1:j+1, ilm) = (self%k-j)*res(:, j:j, ilm) + distances(:) * rf_tmp
                end forall
                deallocate(rf_tmp)
            end do
            deallocate(rfun_all)
            call interpol1%destroy()
            call interpol0%destroy()
            nullify(r)
        end if
    end function

    function Bubbles_radial_derivative(self,ibub) result(res)
        class(Bubbles),intent(in)  :: self
        integer(INT32), intent(in) :: ibub
        real(REAL64), dimension(:), allocatable :: res
        
        real(REAL64), dimension(:), pointer :: f
        integer(INT32) :: nlip
        integer(INT32) :: icell
        integer(INT32) :: j
        real(REAL64), dimension(:,:), allocatable :: lip_dev
        
        allocate(res(self%gr(ibub)%p%get_shape()))
        nlip=self%gr(ibub)%p%get_nlip()
        allocate(lip_dev(nlip,nlip))

        f => self%get_f(ibub,0,0)

        ! loop over cell
        do icell = 1, self%gr(ibub)%p%get_ncell()
            lip_dev = self%gr(ibub)%p%lip_dev_m(icell)
            ! loop over each grid in a cell
            do j = 1, nlip
               ! the last grid is overwritten by the first 
               ! grid of the next cell
               res((icell-1)*(nlip-1)+j)=&
               sum(f((icell-1)*(nlip-1)+1:icell*(nlip-1)+1)*lip_dev(j,:))
            end do
        end do
    end function

    subroutine Bubbles_extrapolate_origo(self, order, lmin, lmax)
        class(Bubbles),   intent(inout) :: self
        integer,          intent(in)    :: order
        integer, optional, intent(in)   :: lmin, lmax
        integer                         :: i, n, nmin, nmax

        if (present(lmin)) then
            nmin = lmin*lmin + 1
        else
            nmin = 1
        end if

        if (present(lmax)) then
            nmax = (lmax+1)*(lmax+1)
        else
            nmax = (self%get_lmax()+1)**2
        end if

    
        do i = 1, self%get_nbub()
            do n = nmin, nmax
                       
                ! use linear extrapolation to get the point at zero
                !result_bubbles%bf(i)%p(1, n) = result_bubbles%bf(i)%p(2, n) &
                !      - (result_bubbles%bf(i)%p(3, n) - result_bubbles%bf(i)%p(2, n)) 

                ! lagrange interpolation:
                if (order == 2) then
                    self%bf(i)%p(1, n) = &
                        2.0d0 * self%bf(i)%p(2, n) &
                        -  1.0d0 * self%bf(i)%p(3, n) 
                        
                else if (order == 3) then
                    self%bf(i)%p(1, n) = &
                           3.0d0 * self%bf(i)%p(2, n) &
                        -  3.0d0 * self%bf(i)%p(3, n) &
                        +  1.0d0 * self%bf(i)%p(4, n)
                
                else if (order == 6) then
                    self%bf(i)%p(1, n) = &
                           6.0d0 * self%bf(i)%p(2, n) &
                        - 15.0d0 * self%bf(i)%p(3, n) &
                        + 20.0d0 * self%bf(i)%p(4, n) &
                        - 15.0d0 * self%bf(i)%p(5, n) &
                        +  6.0d0 * self%bf(i)%p(6, n) &
                        -  1.0d0 * self%bf(i)%p(7, n)
                end if
            end do
        end do
    end subroutine


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! BUBBLE PRODUCT                                                 !!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !> Compute the products of bubbles *on the same centers*.

    !> \f[f_{lm}^A(r_i)=
    !!    \sum_{l1m1l2m2} < Y_{lm} | Y_{l_1m_1} Y_{l_2m_2} >
    !!                    g_{l_1m_1}^A(r_i) h_{l_2m_2}^A(r_i)\f]
    !! The coefficients \f$< Y_{lm} | Y_{l_1m_1} Y_{l_2m_2} >\f$ are given by
    !! harmonic_class::YBundle::prodcoeffs
    function Bubbles_product(bubs1, bubs2) result(new)
        class(Bubbles), intent(in)     :: bubs1
        class(Bubbles), intent(in)     :: bubs2
        type(Bubbles)                  :: new
        
        call bigben%split("Bubbles product")
        ! initialize the result object
        
        call bigben%split("Bubbles intersection")
        new   = bubs1%get_intersection(bubs2, lmax = bubs1%lmax+bubs2%lmax)
        new%k = bubs1%k + bubs2%k
        new   = 0.d0
        call bigben%stop()

        ! do the multiplication
        !call bubbles_multipliers%init_from_mould(bubs1)
        call bubbles_multipliers%multiply(bubs1, bubs2, new)
        call bigben%stop()
        return
    end function

    !> Add two bubbles (same centers and grids assumed!).

    !> The `k` value is `min(b1%%k, b2%%k)`
    function Bubbles_add(b1,b2) result(bout)
        type(Bubbles), intent(in), target :: b1
        type(Bubbles), intent(in), target :: b2
        type(Bubbles)             :: bout

        type(Bubbles), pointer    :: b_k_small
        type(Bubbles), pointer    :: b_k_large

        integer                   :: i, j, l, m, n
        integer                   :: lmax, k_difference

        call bigben%split("Bubbles add")
        lmax=max(b1%lmax, b2%lmax)

        if(b1%k < b2%k) then
            b_k_small => b1
            b_k_large => b2
        else
            b_k_small => b2
            b_k_large => b1
        end if
        bout= b_k_large%merge_with(b_k_small, lmax = lmax)
        bout%k = b_k_small%k
        call bout%copy_content(b_k_small)
        
        k_difference = b_k_large%get_k() - b_k_small%get_k()
        do i=1, b_k_large%nbub
            j = bout%get_order_number(b_k_large%ibubs(i))
            do n = 1, (b_k_large%lmax+1)**2
                bout%bf(j)%p(:,n) = bout%bf(j)%p(:,n) +  &
                    b_k_large%bf(i)%p(:,n)*b_k_large%rpow(i, k_difference)
            end do
            
        end do
        call bigben%stop()
    end function

    !> The `k` value is `min(b1%%k, b2%%k)`
!     subroutine Bubbles_add_in_place(self, b2)
!         type(Bubbles), intent(in), target :: self
!         type(Bubbles), intent(in), target :: b2
!         type(Bubbles)             :: bout
! 
! 
!         integer                   :: i, j, n
!         integer                   :: lmax
! 
!         call bigben%split("Bubbles add in place")
! 
! 
!         do i=1, self%nbub
!             j = b2%get_order_number(self%ibubs(i))
!             forall (n = 1 : (b_k_small%lmax+1)**2)
!                 self%bf(i)%p(:,n)=self%bf(i)%p(:,n) + b2%bf(j)%p(:,n)
!             end forall
!             
!         end do
!         call bigben%stop()
!     end function

    pure subroutine Bubbles_copy_content(self, other_bubbles)
        class(Bubbles), intent(inout) :: self
        type(Bubbles), intent(in)     :: other_bubbles
        integer                       :: numf_cp, i, ibub

        self%k = other_bubbles%k
        numf_cp=min(self%numf, other_bubbles%numf)
        do i=1, self%nbub
            ibub = self%ibubs(i)
            if (other_bubbles%contains_ibub(ibub)) then
                ! init the values of new with values of self
                self%bf(i)%p(:, :numf_cp) = other_bubbles%bf( &
                    other_bubbles%get_order_number(ibub))%p(:, :numf_cp)
            else
                ! init set the value to zero
                self%bf(i)%p(:, :) = 0.0d0
            end if
        end do

        if (numf_cp < self%numf) then
            forall (i=1 : self%nbub)
                ! init the values of new with values of self
                self%bf(i)%p(:, numf_cp+1:) = 0.0d0
            end forall
        end if
        
    end subroutine

    !> Set new value for \ref bubbles::k "Bubbles%k" (wrapper for \ref
    !! bubbles_decrease_k "Bubbles%decrease_k" and \ref bubbles_increase_k
    !! "Bubbles%increase_k")
    subroutine Bubbles_set_k(self, k_new)
        class(Bubbles)             :: self
        integer, intent(in)        :: k_new

        !write(ppbuf, '("Setting Bubbles%k to ",g0)') k_new
        !call pdebug(ppbuf,1)

        if(k_new>self%k) then
            call self%increase_k(k_new-self%k)
        else if (k_new<self%k) then
            call self%decrease_k(k_new-self%k)
        end if
    end subroutine

    !> Decrease \ref bubbles::k "Bubbles%k" by `k_delta` (multiply radial
    !! functions by `k_delta`)
    subroutine Bubbles_decrease_k(self, k_delta)
        class(Bubbles)             :: self
        integer, intent(in)        :: k_delta

        integer                    :: ibub, l, m, j
        real(REAL64), allocatable  :: rkdelta(:)

        ! Argument check (k_delta<0)
        if (k_delta>0) then
            write(ppbuf, '("In Bubbles_decrease_k: k_delta (",g0,") should be smaller than 0! Ignoring.")') k_delta
            call perror(ppbuf)
            return
        else
            !write(ppbuf, '("Decreasing Bubbles%k by ",g0)') k_delta
            !call pdebug(ppbuf,1)
        end if

        
        do ibub=1,self%nbub
            rkdelta=self%rpow(ibub, -k_delta)
            forall(j=1:(self%lmax+1)**2)
                self%bf(ibub)%p(:,j) = self%bf(ibub)%p(:,j) * rkdelta
            end forall
            deallocate(rkdelta)
        end do
        self%k=self%k+k_delta
    end subroutine

    !> Increase \ref bubbles::k "Bubbles%k" by `k_delta` (divide radial
    !! functions by `k_delta`).
    !! NOTE: This procedure is more complicated than \ref bubbles_decrease_k
    !! "Bubbles%decrease_k" ! because it implies dividing by 0 at r=0!

    !> The radial functions are divided by `r**k_delta. The value at \f$r=0\f$
    !! is calculated by explicitly constructing the polynomial at the first
    !! cell, lowering the degree by `k_delta` (i.e. dividing by \f$r^2\f$) and
    !! picking the last coefficient. This assumes that all discarded
    !! coefficients are 0.
    subroutine Bubbles_increase_k(self, k_delta)
        implicit none
        class(Bubbles)              :: self
        integer, intent(in)         :: k_delta

        type(REAL64_2D), allocatable :: coeffs(:)

        real(REAL64), allocatable   :: p_loc(:,:)
        real(REAL64), allocatable   :: p_shf(:,:)

        integer  :: ibub
        integer  :: ilip, j, nlip, m
        real(REAL64), pointer      :: cell_scales(:)
        real(REAL64)               :: middle_point
        real(REAL64), allocatable  :: rkdelta(:)

        real(REAL64), parameter    :: THRESHOLD=1.d-8

        ! Argument check (k_delta>0)
        if (k_delta<0) then
            write(ppbuf, '("In Bubbles_increase_k: k_delta (",g0,") should be larger than 0! Ignoring.")') k_delta
            call perror(ppbuf)
            return
        else
            !write(ppbuf, '("Increasing Bubbles%k by ",g0)') k_delta
            !call pdebug(ppbuf,1)
        end if

        nlip=self%gr(1)%p%get_nlip()
        coeffs=self%gr(1)%p%lip%coeffs(0)

        do ibub=1,self%nbub
            cell_scales =>self%gr(ibub)%p%get_cell_scales()
            ! calculate the middle point of the first cell
            middle_point = cell_scales(1) * ( (nlip-1)/2 )

            ! Computing the value of the function at r=0 (can't divide by 0...)
            ! Compute polynomial in local (cell) coordinates
            p_loc=xmatmul( transpose(coeffs(1)%p),&
                             self%bf(ibub)%p(1:nlip,:) )
            !p_loc = eval_polys(self%gr(ibub)%p%x2cell(self%bf(ibub)%p(1:nlip, :))) 
            ! Shift polynomial to radial coordinates
            ! Compute b_j's such that
            ! \sum b_j x^j = \sum c_j ((x-mp)/h)^j
            allocate(p_shf(nlip, (self%lmax+1)**2))
            p_shf(1,:) =  p_loc(1,:) !eval_polys(coeffs(1)%p, self%gr(ibub)%p%x2cell(self%bf(ibub)%p(1, :))) & 
                           ! * self%bf(ibub)%p(1:nlip, :) !
            p_shf(2:,:)=0.d0
            do ilip=2,nlip
                do j=1, (self%lmax+1)**2
                    p_shf(ilip,j) = ( p_shf(ilip,j) - middle_point * p_shf(ilip-1,j) ) / cell_scales(1)
                    p_shf(ilip,j) = p_shf(ilip,j) + p_loc(ilip,j)
                end do
            end do

            nullify(cell_scales)
            deallocate(p_loc)

            ! Column to use
            m=nlip-k_delta

            ! Check we are not dividing too much
            if ( any( sum(abs(p_shf(m+1:,:)),dim=1)/abs(p_shf(m,:)) > THRESHOLD) ) then
                write(ppbuf,'("Bubbles%increase_k(): Trying to divide by r^",i2," when the radial")') k_delta
                call perror(ppbuf)
                write(ppbuf,'("function cannot be factorized as r^",i2," f(r).")') k_delta
                call perror(ppbuf)
            end if

            rkdelta=self%rpow(ibub, -k_delta)
            forall(j=1:(self%lmax+1)**2)
                self%bf(ibub)%p(:,j) = self%bf(ibub)%p(:,j) * rkdelta
            end forall
            self%bf(ibub)%p(1,:)=p_shf(m,:)
            deallocate(rkdelta)
            deallocate(p_shf)
        end do
        deallocate(coeffs(1)%p)
        deallocate(coeffs)
        self%k=self%k+k_delta
    end subroutine

    !> Compute the Taylor series coefficients of 'contaminant_bubbles' at the
    !! bubble centers of 'self'.

    !> The elements of the output matrix are given by
    !! \f[
    !! C_{\boldsymbol{\alpha}, A} =
    !! \sum_{A\neq B}\sum_{lm} \partial^\boldsymbol{\alpha}_\mathbf{r}
    !!                   {f^A_{lm}(r_A) Y^A_{lm} (\theta_A,\phi_A)}
    !!                                          \Big|_{\mathbf{R}_B} \f]
    function Bubbles_get_foreign_bubbles_contaminants(self, contaminant_bubbles, tmax) result(result_contams)
        class(Bubbles), intent(in) :: self
        !> Bubbles from which contaminants are extracted
        type(Bubbles), intent(in) :: contaminant_bubbles
        !> Order of the Taylor series
        integer, intent(in)        :: tmax

        !                              Derivative orders        Centers
        real(REAL64) :: contams( (tmax+1)*(tmax+2)*(tmax+3)/6, self%nbub)
        real(REAL64) :: result_contams( (tmax+1)*(tmax+2)*(tmax+3)/6, self%nbub_global)

        integer(INT32)         :: ibub, jbub,ider,idv
        real(REAL64)           :: vectors(3, self%nbub)
        real(REAL64)           :: distances(1, self%nbub)
        real(REAL64)           :: rfs(self%nbub, 0:tmax, (contaminant_bubbles%lmax+1)**2)

        integer(INT32) :: i, j, k, l, m, ilm
        type(Y_t), pointer :: y

        type(CartIter) :: cart
        logical        :: continue_iteration
        type(Interpolator) :: interpol
        integer(INT32) :: dv(3)

        type(YBundle) :: yb

        contams=0.d0

        yb=YBundle(contaminant_bubbles%lmax)

        ! go through all contaminant bubbles
        do i=1,contaminant_bubbles%nbub

            ! Calculate relative coordinates of all 'self' bubbles
            do j=1,self%nbub
                vectors(:, j)   = self%centers(:, j) - contaminant_bubbles%centers(:,i)
                distances(:,j) = sqrt( sum( vectors(:, j) * vectors(:, j) ) )
            end do

            ! Obtain radial derivatives of bubbles object 'contaminant_bubbles'
            ! at the locations of 'self' bubbles, derivates of order 0-tmax
            ! are evaluated
            interpol=Interpolator(contaminant_bubbles%gr(i:i),tmax)
            ilm=0
            do l=0, contaminant_bubbles%lmax
                do m=-l,l
                    ilm=ilm+1
                    do j = 1, size(distances)
                        rfs(j:j,:, ilm)=interpol%eval(contaminant_bubbles%get_f(i,l,m), distances(:,j:j))
                    end do
                end do
            end do
            call interpol%destroy()

            !Iterate over cartesian derivatives
            cart=CartIter(3, maximum_modulus = tmax)
            idv=0
            call cart%next(dv, continue_iteration)
            do while(continue_iteration)
                ! dv: array indicating the times the derivate X_, Y_, and Z_
                ! axis are derivated 
                idv=idv+1
                ! sum of derivate's order
                ider=sum(dv)

                ! Iterate over spherical harmonics
                ilm=0
                do l=0, contaminant_bubbles%lmax
                    do m=-l,l
                        ilm=ilm+1

                        ! pick spherical harmonics with l, m
                        y=>yb%pick(l,m)

                        ! go through all bubbles of self
                        do j=1,self%nbub
                            jbub = self%ibubs(j)
                            !\sum_{lm} { Y_{lm}(x,y,z,r) * f_{lm}(r) }
                            ! sum over all derivatives
                            contams(idv, j) = contams(idv, j)   &
                               + sum(y%eval_coeffs(vectors(:, j:j), distances(1, j:j),  dv)   &
                                   * rfs(j:j, :ider, ilm))
                            
                        end do
                    end do
                end do
                
                call cart%next(dv, continue_iteration)
            end do
            call cart%destroy()
        end do
        call yb%destroy()

        result_contams = 0.0d0
        do i = 1, self%nbub
            ibub = self%get_ibub(i)
            result_contams(:, ibub) = contams(:, i) 
        end do
    end function

    !> Compute the Taylor series coefficients of every other bubble at the
    !! bubble centers.

    !> The elements of the output matrix are given by
    !! \f[
    !! C_{\boldsymbol{\alpha}, A} =
    !! \sum_{A\neq B}\sum_{lm} \partial^\boldsymbol{\alpha}_\mathbf{r}
    !!                   {f^A_{lm}(r_A) Y^A_{lm} (\theta_A,\phi_A)}
    !!                                          \Big|_{\mathbf{R}_B} \f]
    function Bubbles_get_contaminants(self,tmax) result(contams)
        class(Bubbles), intent(in) :: self
        !> Order of the Taylor series
        integer, intent(in)        :: tmax
        !                              Derivative orders        Centers
        real(REAL64)               :: contams( (tmax+1)*(tmax+2)*(tmax+3)/6, self%nbub_global)
        integer(INT32)             :: ibub, jbub,k,ider,idv
        real(REAL64)               :: vectors(3, self%nbub_global - 1)
        real(REAL64)               :: distances(self%nbub_global - 1) 
        real(REAL64)               :: rfs(self%nbub_global-1, 0:tmax, (self%lmax+1)**2)

        integer(INT32) :: i,j,l,m,ilm
        type(Y_t), pointer :: y

        type(CartIter) :: cart
        integer(INT32) :: dv(3)
        logical        :: continue_iteration

        type(YBundle) :: yb

        contams=0.d0

        yb=YBundle(self%lmax)
 
         
        do i=1, self%nbub
            ibub = self%ibubs(i)
            k=1

            ! Calculate relative coordinates of all other bubbles
            do jbub=1,self%nbub_global
                if(ibub==jbub) cycle
                vectors(:,k)   = self%global_centers(:, jbub) - self%global_centers(:, ibub)
                distances(k) = sqrt( sum( vectors(:, k) * vectors(:,k) ) )
                k=k+1
            end do

            ! Obtain radial derivatives of this bubbles object 'self'
            ! at the locations of other bubbles, derivates of order 0-tmax
            ! are evaluated
            rfs = self%get_radial_derivatives(i, distances, tmax)
            !Iterate over derivatives
            cart=CartIter(3, tmax)
            idv=0
            call cart%next(dv, continue_iteration)
            do while(continue_iteration)
                ! dv: array indicating the times the derivate X_, Y_, and Z_
                ! axis are derivated 
                idv=idv+1
                ! sum of derivate's order
                ider=sum(dv)

                ! Iterate over spherical harmonics
                ilm=0
                do l=0,self%lmax
                    do m=-l,l
                        ilm=ilm+1

                        ! pick spherical harmonics with l, m
                        y=>yb%pick(l,m)
                        k=1

                        ! go through all bubbles except the one where number of bubble
                        ! is equal with jbub
                        do jbub=1, self%nbub_global
                            if(ibub==jbub) cycle
                            !\sum_{lm} { Y_{lm}(x,y,z,r) * f_{lm}(r) }
                            ! sum over all derivatives
                            contams(idv, jbub) = contams(idv, jbub)   &
                               + sum(y%eval_coeffs(vectors(:, k:k), distances(k:k), dv)   &
                                   * rfs(k:k, :ider, ilm))
                            k=k+1
                        end do
                    end do
                end do
                call cart%next(dv, continue_iteration)
            end do
            call cart%destroy()
        end do
        call yb%destroy()
    end function

    !> Contaminants for nuclear potential
    !! \f$ \sum_{A\neq B lm} \partial_{\alpha_i}
    !!                   {f^A_{lm}(r_A) Y^A_{lm} (\theta_A,\phi_A)}
    !!                                    \Big|_{\mathbf{R}_B} \f$
    function Bubbles_get_nuclear_contaminants(self,lmax) result(res)
        class(Bubbles)  :: self
        integer(INT32)  :: lmax
        !                      Derivative orders        Centers
        real(REAL64)    :: res( (lmax+1)*(lmax+2)*(lmax+3)/6, self%nbub)
        integer(INT32)  :: ibub, jbub
        real(REAL64)    :: point(3, 1), distance(1)

        type(rminus1_t) :: op

        call rminus1_init(op,lmax)

        res=0.d0
        do ibub=1,self%nbub
            do jbub=1,ibub-1
                point(:, 1) = self%centers(:,ibub)-self%centers(:,jbub)
                distance(1) = sqrt(sum(point(:, 1)*point(:, 1)))
                res(:,ibub:ibub)=res(:,ibub:ibub)&
                            -self%z(jbub)*rminus1_eval(op, point, distance)
            end do
            do jbub=ibub+1,self%nbub
                point(:, 1) = self%centers(:,ibub)-self%centers(:,jbub)
                distance(1) = sqrt(sum(point(:, 1)*point(:, 1)))
                res(:,ibub:ibub)=res(:,ibub:ibub)&
                           -self%z(jbub)*rminus1_eval(op,point, distance)
            end do
        end do
    end function

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! OUTPUT                                                          !!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    subroutine Bubbles_print_out(self, lmax_, lmin_, point)
        class(Bubbles), intent(in)     :: self
        integer, optional, intent(in) :: lmax_, lmin_
        real(REAL64), optional        :: point
        integer                       :: n, i, ibub, lmin, lmax, grid_point_min, m(1), grid_point_max
        real(REAL64), pointer          :: coord(:)
        character(512)                :: text, temp, text2
        real(REAL64)                  :: point_
        if (present(lmax_)) then
            if (lmax_ <= self%get_lmax()) then
                lmax = lmax_
            else
                lmax = self%get_lmax()
            end if
        else
            lmax = self%get_lmax()
        end if

        if (present(lmin_)) then
            lmin = lmin_
        else
            lmin = 0
        end if

        point_ = 0.0d0
        if (present(point)) point_ = point
        print *, "POINT IS", point_
        do ibub = 1, self%get_nbub()
            write(*, '("---------------------- BUBBLE ",i1," -------------------")') ibub
            print *, "  sum ", ibub, sum(self%bf(ibub)%p)
            coord => self%gr(ibub)%p%get_coord()
            m = minloc(abs(coord-point_)) 
            grid_point_min = max(m(1)-7, 1)
            grid_point_max = grid_point_min + 14
            do i = grid_point_min+1, grid_point_max
                write(*, '(i5, ": ", f7.4)', advance = 'no') i, coord(i) - point_
            end do 
            write(*, *) ""
            print *, grid_point_min, "coord at", coord(grid_point_min)
            do n = lmin ** 2 +1, (lmax + 1) ** 2
                print *, "  sum  n", n, sum(self%bf(ibub)%p(:, n))
                print '("  maxval", i6, ":", E12.5, " at ", i6)', n, maxval(self%bf(ibub)%p(:, n)), maxloc(self%bf(ibub)%p(:, n))
                print '("  minval", i6, ":", E12.5, " at ", i6)', n, minval(self%bf(ibub)%p(:, n)), minloc(self%bf(ibub)%p(:, n))
                print *, grid_point_min, "-", grid_point_max, "n:", n
                write(text, '(E12.5)') self%bf(ibub)%p(grid_point_min, n)
                do i = grid_point_min+1, grid_point_max
                    if (i == m(1)) then
                        write(temp, '(", p", E12.5)') self%bf(ibub)%p(i, n)
                    else if (mod(i, 6) == 1) then
                        write(temp, '(", c", E12.5)') self%bf(ibub)%p(i, n)
                    else
                        write(temp, '(",  ", E12.5)') self%bf(ibub)%p(i, n)
                    end if
                    text = trim(text) // trim(temp)
                end do
                print *, trim(text)
            end do

        end do
        
        do ibub = 1, self%get_nbub()
            coord => self%gr(ibub)%p%get_coord()
            do n = 1, (self%get_lmax() + 1) ** 2
                do i = 1, size(self%bf(ibub)%p, 1)-1
                    if (self%bf(ibub)%p(i, n) * self%bf(ibub)%p(i+1, n) < 0.0d0 .and. abs(self%bf(ibub)%p(i, n)) > 1d-4) then
                        print '("ZERO POINT at ",i1, i6, "(", F8.5, "), n:", i3 ":", E12.5, E12.5)', &
                            ibub, i, coord(i), n, self%bf(ibub)%p(i, n), self%bf(ibub)%p(i+1, n)
                    end if
                end do
            end do

        end do
        
    end subroutine

    subroutine Bubbles_print_all(self,fname)
        class(Bubbles),intent(in) :: self
        character(*) :: fname
        integer(INT32) :: ibub

        do ibub=1,self%nbub
            call self%print(trim(fname)//char(48+ibub)//'.bub',ibub)
        end do
    end subroutine

    !> Output selected bubble radial functions in human-readable form.

    !> Produce a file `fname` with the radial grid in the first column and the
    !! corresponding value of \f$ r^k_A f^A_{lm}(r_A) \f$ for every l,m in
    !! every other column.
    subroutine Bubbles_print_pick(self,fname,ibub)
        class(Bubbles),intent(in) :: self
        integer(INT32) :: ibub
        character(*) :: fname
        character(20):: ff
        integer(INT32) :: i,j
        real(REAL64),allocatable :: rk(:)
        real(REAL64),pointer     :: r(:)

        open(20,file=fname)
        write(ff,*) 1+self%numf
        write(ff,*) '('//trim(adjustl(ff))//'e20.12)'

        r  =>self%gr(ibub)%p%get_coord()
        rk = self%rpow(ibub,self%k)

        do i=1, self%gr(ibub)%p%get_shape()
            write(20,ff) r(i), (self%bf(ibub)%p(i,j) * rk(i), j=1,self%numf)
        end do
        deallocate(rk)
        nullify(r)
        close(20)
    end subroutine

!!    subroutine read_xyz(coordfile,nuclei,num_nuclei)
!!        ! Reads nuclear coordinates and charges from filename to a type(nucleus) 1-D array
!!        ! Sets num_nuclei to the number of nuclei as well
!!    character(len=*) :: coordfile
!!    type(Bubbles),allocatable  :: nuclei(:)
!!    integer(INT32) :: i,err,num_nuclei
!!
!!    open(36,file=coordfile,action='read')!,iostat=err)
!!    read(36,*) num_nuclei
!!    ! Skip comment line
!!    read(36,*)
!!
!!    allocate(nuclei(num_nuclei))
!!    do i=1,num_nuclei
!!        read(36,*,iostat=err), nuclei(i)%id,nuclei(i)%x,nuclei(i)%y,nuclei(i)%z,nuclei(i)%d0  !,nuclei(i)%alpha
!!        if(err/=0) then
!!            write(ppbuf,*),'Error reading '//trim(coordfile)//' at line ',2+i
!!            print*, nuclei(i)%id,nuclei(i)%x,nuclei(i)%y,nuclei(i)%z,nuclei(i)%d0  !,nuclei(i)%alpha
!!            call perror(ppbuf)
!!            stop
!!        end if
!!        nuclei(i)%charge=symbol2z(nuclei(i)%id(:index(nuclei(i)%id,"-")-1))
!!    end do
!!    close(36)
!!    nuclei%x=nuclei%x*A2AU
!!    nuclei%y=nuclei%y*A2AU
!!    nuclei%z=nuclei%z*A2AU
!!!    print*,minval(nuclei%x),maxval(nuclei%x)
!!!    print*,minval(nuclei%y),maxval(nuclei%y)
!!!    print*,minval(nuclei%z),maxval(nuclei%z)
!!    return
!!    end subroutine read_xyz

    !> Return the atomic number given a chemical symbol
    integer function symbol2z(symbol) result(Z)
        ! Transform element symbol into Z
        character(*) :: symbol
        select case(symbol)
            case('H'); Z=1
            case('He');Z=2
            case('Li');Z=3
            case('Be');Z=4
            case('B'); Z=5
            case('C'); Z=6
            case('N'); Z=7
            case('O'); Z=8
            case('F'); Z=9
            case('Ne');Z=10
            case('Na');Z=11
            case('Mg');Z=12
            case default
                call pwarn('Element "'//symbol//'"not implemented yet!')
                Z=888
        end select
        return
    end function symbol2z

    !> Compute the integral of a bubble center over all space.
    !!
    !! This method evaluates the integral
    !!
    !! \f[ I_{A} = \sum_{l,m}\int r^k f^{Alm}(r_A) Y_l^m(\theta_A, \varphi_A)d^3r\f]
    !!
    !! for a given bubble center A.  
    !!
    !! The angular intergral is effectively of the form \f$\langle
    !! lm|00\rangle\f$. Because spherical harmonics are orthogonal, the
    !! sum over spherical harmonics thus always reduces to a single term
    !! -- only the s-component does not vanish.
    function Bubbles_integrate_pick(self, ibub, k) result(res)
        class(Bubbles),intent(in)            :: self
        !> The index of the bubble center
        integer(INT32), intent(in)           :: ibub
        !> An optional exponent that can be used to override the
        !! exponent attribute of ibub
        integer(INT32), intent(in), optional :: k
        !> The value of the integral
        real(REAL64)                         :: res
        real(REAL64),  allocatable           :: rpow(:) 
        real(REAL64),  pointer               :: f(:)

        type(Integrator1D)                     :: integr
        integer                              :: ktmp

        !print *, allocated(self%bf)
        if (present(k)) then
            ktmp = k
        else
            ktmp = self%k
        endif
        !print *, ktmp
        integr=Integrator1D(self%gr(ibub)%p)
        ! Only the s component doesn't vanish
        rpow = self%rpow(ibub, 2+ktmp) 
        f => self%get_f(ibub,0,0)
        res=FOURPI * &
            integr%eval(f * rpow)
        deallocate(rpow)
        nullify(f)
        call integr%destroy()
        !print *, "bubbles integral", res
        return
    end function

    !> Compute the integral of all the bubble centers over all space.
    !!
    !! This method evaluates the integral
    !!
    !! \f[ I = \sum_A\sum_{l,m}\int r_A^k f^{Alm}(r_A) Y_l^m(\theta_A, \varphi_A)d^3r\f]
    !!
    !! for all centers A and contracts the value to a scalar.
    !!
    !! See @ref bubbles_integrate_pick for more info.
    function Bubbles_integrate_all(self, k) result(res)
        class(Bubbles),intent(in)            :: self

        !> An optional array of exponents that allows
        !! overriding the exponent attribute of a bubbles object
        !! on a center-by-center basis.
        integer(INT32), intent(in), optional :: k(:)
        real(REAL64)                         :: res

        real(REAL64)                         :: tempval
        integer(INT32)                       :: ibub
        integer(INT32), allocatable          :: ktmp(:)
        logical                              :: cuda_integrate 

        cuda_integrate = .FALSE.

#ifdef HAVE_CUDA        
        if (allocated(self%cuda_interface)) then
            res = Bubbles_integrate_cuda(self%cuda_interface)
            cuda_integrate = .TRUE.
        end if
#endif
        if (.not. cuda_integrate) then
            allocate(ktmp(self%nbub))
            !call pdebug("bubble integrals", 1)
            ktmp = self%k
            if (present(k)) then
                if (size(k) == self%nbub) ktmp = k 
            endif

            res=0.d0
            do ibub=1,self%nbub
                res=res+self%integrate(ibub, ktmp(ibub))
            end do
            deallocate(ktmp)
        end if
        return
    end function

    !> Computes the radial integral of a given bubble function.
    !!
    !! This method computes the integral
    !!
    !! \f[ I^{Alm}= \sum_{l,m}\int\limits_0^\infty r_A^2 f^{Alm}(r_A) d_A^3r\f]
    !!
    !! for a given bubble center *A*. The result is a scalar.
    !!
    !! See @ref bubbles_integrate_radial_all for a method to compute 
    !! the radial integrals for all indices l,m.
    function Bubbles_integrate_radial_pick(self, i, l, m) result(res)
        class(Bubbles)             :: self
        !> The bubble center index
        integer(INT32), intent(in) :: i
        integer(INT32), intent(in) :: l
        integer(INT32), intent(in) :: m
        real(REAL64)               :: res

        real(REAL64), pointer      :: r(:)

        type(Integrator1D)         :: integr

        integr=Integrator1D(self%gr(i)%p)
        r => self%gr(i)%p%get_coord()
        ! The r^2 factors comes from the volume element
        res= integr%eval( self%get_f(i,l,m) * r*r )
        nullify(r)
        call integr%destroy()
    end function

    !> Computes all the radial integral of a given bubble center.
    !!
    !! This method computes all the integrals
    !!
    !! \f[ I^{Alm} = \sum_{l,m}\int\limits_0^\infty r_A^2 f^{Alm}(r_A) d_A^3r\f]
    !! 
    !! in a given bubble center A and returns the values in a vector.
    !!
    !! The semantics of this method differ somewhat from @ref
    !! bubbles_integrate_all that returns a scalar. Use the idx(l,m)
    !! function to access the individual integrals \f$I^{Alm}\f$.
    function Bubbles_integrate_radial_all(self, i) result(res)
        class(Bubbles)             :: self
        integer(INT32), intent(in) :: i
        real(REAL64), allocatable  :: res(:)

        real(REAL64), pointer      :: r(:)
        type(Integrator1D)         :: integr
        integer                    :: m, l

        allocate(res((self%lmax+1)**2), source=0.0d0)
        integr=Integrator1D(self%gr(i)%p)
        r => self%gr(i)%p%get_coord()
        do l=0, self%lmax
            do m=-l,l
                res(idx(l,m)) = integr%eval(self%get_f(i, l, m)*r*r)
            enddo
        enddo
        nullify(r)
        call integr%destroy()
        return

    end function


    !> Multiply bubbles by a constant.
    function REAL64_times_Bubbles(factor, bubs) result(new)
        real(REAL64), intent(in)   :: factor
        type(Bubbles), intent(in)  :: bubs

        type(Bubbles)              :: new

        integer                    :: ibub

        new=Bubbles(bubs, copy_content = .TRUE.)

        forall(ibub=1:new%nbub)
            new%bf(ibub)%p = factor * new%bf(ibub)%p
        end forall
    end function

    !> Multiply bubbles by a constant.
    function Bubbles_times_REAL64(bubs, factor) result(new)
        type(Bubbles), intent(in)  :: bubs
        real(REAL64), intent(in)   :: factor

        type(Bubbles)              :: new

        new=factor * bubs
    end function

    !> Create a copy of the object with a new set of radial grids.
    function Bubbles_project_onto(self, target_grids) result(new)
        class(Bubbles),intent(in)        :: self
        type(Grid1DPointer), intent(in)  :: target_grids(:)

        type(Bubbles)                    :: new

        integer                          :: ibub, l, m, j
        real(REAL64), pointer            :: r(:)
        real(REAL64), pointer            :: r_w(:,:)

        type(Interpolator)               :: interpol

        new =  Bubbles( lmax        = self%lmax, &
                        centers     = self%centers, &
                        global_centers = self%global_centers, &
                        grids       = target_grids, &
                        global_grids = self%global_grids, &
                        z           = self%z, &
                        global_z  = self%global_z, &
                        k           = self%k, &
                        ibubs       = self%ibubs, &
                        nbub_global = self%nbub_global )

        do ibub=1,self%nbub
            interpol=Interpolator(self%gr(ibub:ibub))
            j=0
            r=>new%gr(ibub)%p%get_coord()
            allocate(r_w(1,size(r)))
            r_w(1,:)=r
            do l=0,min(self%lmax, new%lmax)
                do m=-l,l
                    j=j+1
                    new%bf(ibub)%p(:,j:j)=reshape(interpol%eval( self%bf(ibub)%p(:,j), r_w),[size(r_w),1])
                end do
            end do
            deallocate(r_w)
            nullify(r)
            call interpol%destroy()
        end do
    end function

    !> Compute \f$ r^n \f$. If n<0, r(1) is set to 0.d0 to avoid NaN.
    function Bubbles_rpow(self, order_number, n) result(rpow)
        class(Bubbles), intent(in) :: self
        integer, intent(in)        :: order_number
        integer, intent(in)        :: n

        real(REAL64)               :: rpow(self%gr(order_number)%p%get_shape())

        associate( r => self%gr(order_number)%p%get_coord() )
            if (n>=0) then
                rpow=r**n
            else
                rpow(1)=0.d0
                rpow(2:)=r(2:)**n
            end if
        end associate
    end function

    function Bubbles_get_multipole_moments(self, lmax, evaluation_point) result(qbub)
        class(Bubbles), intent(in)        :: self
        integer, optional, intent(in)     :: lmax
        real(REAL64), intent(in), optional:: evaluation_point(3)
        real(REAL64), allocatable         :: qbub(:, :)
        real(REAL64)                      :: bubble_center(3)
        real(REAL64), allocatable         :: bubble_poles(:)
        integer                           :: i
        integer                           :: mpol_lmax

        if (present(lmax)) then
            allocate(qbub((lmax+1)**2, self%nbub))
            mpol_lmax = lmax
        else
            allocate(qbub((self%lmax+1)**2, self%nbub))
            mpol_lmax = self%get_lmax()
        end if

        ! evaluate bubble multipole moments one by one
        do i=1, self%get_nbub()
            ! get the bubble center
            bubble_center = self%get_centers(i)
              
            bubble_poles = self%integrate_radial(i)
            ! bubble_multipoles is at multipole_tools.f90
            qbub(:, i) =  bubble_multipoles(bubble_poles, bubble_center, &
                                 lmax = mpol_lmax, bubbles_lmax = self%get_lmax(), to=evaluation_point)

            deallocate(bubble_poles)
        end do
    end function

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                   BubblesMultiplier methods                             %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function BubblesMultiplier_init(smaller_lmax_bubbles, larger_lmax_bubbles, first_multiplier, & 
                 smaller_lmax_taylor_bubbles, larger_lmax_taylor_bubbles) result(multiplier)
        !! pointer to the input Bubbles object which has smaller lmax value
        type(Bubbles), intent(in)               :: smaller_lmax_bubbles
        !! pointer to the input Bubbles object which has larger lmax value
        type(Bubbles), intent(in)               :: larger_lmax_bubbles
        type(BubblesMultiplier), pointer        :: first_multiplier
        type(Bubbles), intent(in), optional     :: smaller_lmax_taylor_bubbles
        type(Bubbles), intent(in), optional     :: larger_lmax_taylor_bubbles
        !> The result object
        type(BubblesMultiplier)                 :: multiplier

        !! copies for the input objects
        type(Bubbles), allocatable              :: smaller_lmax_bubbles_copy
        type(Bubbles), allocatable              :: larger_lmax_bubbles_copy
        type(Bubbles), allocatable              :: smaller_lmax_taylor_bubbles_copy
        type(Bubbles), allocatable              :: larger_lmax_taylor_bubbles_copy
        type(Bubbles)                           :: result_object
        
        ! the final lmax                
        integer                                 :: lmax
#ifdef HAVE_CUDA_PROFILING
        call start_nvtx_timing("Init Bubbles Multiplier. ")
#endif
        lmax =  smaller_lmax_bubbles%get_lmax()+larger_lmax_bubbles%get_lmax()
        ! Initialize the coefficients of spherical harmonics product
        multiplier%yproduct=YProduct(lmax, lmax)

        ! init a new multiplier and add it to array
        multiplier%smaller_lmax = smaller_lmax_bubbles%get_lmax()
        multiplier%larger_lmax = larger_lmax_bubbles%get_lmax()
#ifdef HAVE_CUDA
        if (associated(first_multiplier)) then
            multiplier%bubbles1 = first_multiplier%bubbles1
            multiplier%bubbles2 = first_multiplier%bubbles2
            multiplier%result_bubbles = first_multiplier%result_bubbles
            multiplier%taylor_series_bubbles1 = first_multiplier%taylor_series_bubbles1
            multiplier%taylor_series_bubbles2 = first_multiplier%taylor_series_bubbles2
            multiplier%first = .FALSE.
            
            ! finally initialize the cuda part of the multiplier
            call multiplier%cuda_init()
        else
        
            multiplier%first = .TRUE.
            result_object = Bubbles(smaller_lmax_bubbles, lmax = lmax)

            ! get the copies of the lmax bubbles, as the input functions are of intent(in)
            smaller_lmax_bubbles_copy = Bubbles(smaller_lmax_bubbles, copy_content = .TRUE.)
            larger_lmax_bubbles_copy = Bubbles(larger_lmax_bubbles, copy_content = .TRUE.)

            ! get the cuda interfaces to the C++/Cuda objects
            multiplier%bubbles1 = smaller_lmax_bubbles_copy%get_cuda_interface()
            multiplier%bubbles2 = larger_lmax_bubbles_copy%get_cuda_interface()
            multiplier%result_bubbles = &
                result_object%get_cuda_interface()
            ! get the copies of the taylor series bubbles and get the interfaces to C++/cuda objects
            if (present(larger_lmax_taylor_bubbles) .and. present(smaller_lmax_taylor_bubbles)) then
                ! get the copies
                smaller_lmax_taylor_bubbles_copy = Bubbles(smaller_lmax_taylor_bubbles, copy_content = .TRUE.)
                larger_lmax_taylor_bubbles_copy = Bubbles(larger_lmax_taylor_bubbles, copy_content = .TRUE.)
        
                ! get the interfaces
                multiplier%taylor_series_bubbles1 = &
                    smaller_lmax_taylor_bubbles_copy%get_cuda_interface()
                multiplier%taylor_series_bubbles2 = &
                    larger_lmax_taylor_bubbles_copy%get_cuda_interface()
            else
                multiplier%taylor_series_bubbles1 = C_NULL_PTR
                multiplier%taylor_series_bubbles2 = C_NULL_PTR
            end if

            ! finally initialize the cuda part of the multiplier
            call multiplier%cuda_init()

            ! cut the fortran connection between result bubbles and the multiplier
            call result_object%dereference_cuda_interface()

            ! destroy and deallocate the copies (but do not destroy the cuda parts, which live on in multiplier)
            call smaller_lmax_bubbles_copy%destroy()
            call larger_lmax_bubbles_copy%destroy()
            call result_object%destroy()
            deallocate(smaller_lmax_bubbles_copy)
            deallocate(larger_lmax_bubbles_copy)
            
            
            if (present(larger_lmax_taylor_bubbles) .and. present(smaller_lmax_taylor_bubbles)) then
                call smaller_lmax_taylor_bubbles_copy%destroy()
                call larger_lmax_taylor_bubbles_copy%destroy()
                deallocate(smaller_lmax_taylor_bubbles_copy)
                deallocate(larger_lmax_taylor_bubbles_copy)
            end if
        end if

#endif
#ifdef HAVE_CUDA_PROFILING
        call stop_nvtx_timing()
#endif
    end function

#ifdef HAVE_CUDA
    subroutine BubblesMultiplier_cuda_init(self)
        class(BubblesMultiplier), intent(inout)          :: self
        if (.not. allocated(self%cuda_interface)) then
            self%cuda_interface = BubblesMultiplier_init_cuda(self%bubbles1, self%bubbles2, &
                     self%result_bubbles, self%taylor_series_bubbles1, self%taylor_series_bubbles2, &
                     self%smaller_lmax+self%larger_lmax, &
                     self%yproduct%coefficients, self%yproduct%number_of_terms, &
                     self%yproduct%result_order_numbers, self%yproduct%pos, &
                     size(self%yproduct%coefficients), 0, 1, stream_container)
        end if
    end subroutine

    subroutine BubblesMultiplier_cuda_destroy(self)
        class(BubblesMultiplier), intent(inout) :: self

        if (allocated(self%cuda_interface)) then
            call BubblesMultiplier_destroy_cuda(self%cuda_interface)
            if (self%first) then
                if (allocated(self%bubbles1)) call Bubbles_destroy_cuda(self%bubbles1)
                if (allocated(self%bubbles2)) call Bubbles_destroy_cuda(self%bubbles2)
                if (allocated(self%taylor_series_bubbles1)) &
                    call Bubbles_destroy_cuda(self%taylor_series_bubbles1)
                if (allocated(self%taylor_series_bubbles2)) &
                    call Bubbles_destroy_cuda(self%taylor_series_bubbles2)
                if (allocated(self%result_bubbles)) &
                    call Bubbles_destroy_cuda(self%result_bubbles)
            end if
            if (allocated(self%cuda_interface))         deallocate(self%cuda_interface)
            if (allocated(self%bubbles1))               deallocate(self%bubbles1)
            if (allocated(self%bubbles2))               deallocate(self%bubbles2)
            if (allocated(self%taylor_series_bubbles1)) deallocate(self%taylor_series_bubbles1)
            if (allocated(self%taylor_series_bubbles2)) deallocate(self%taylor_series_bubbles2)
            if (allocated(self%result_bubbles))         deallocate(self%result_bubbles)
        end if
    end subroutine
#endif

    !> Multiply two bubbles and store the result to result_bubbles. This
    !! Function does the nodewise multiplication and communicates the 
    !! results to all nodes.
    !!
    !! There are two options:
    !!  (1) the entire multiplication, which includes adding the bubbles with their 
    !!      taylor series counterparts and finally doing the multiplication for these sums,
    !!      i.e., (bubbles1 + taylor_series_bubbles1) * (bubbles2 + taylor_series_bubbles2)
    !!  (2) simple multiplication
    !!      smaller_lmax_bubbles * larger_lmax_bubbles
    !!
    !! The option (1) is called when taylor_series_bubbles1 and taylor_series_bubbles2 are provided,
    !! otherwise, the option (2) is used.
    subroutine BubblesMultiplier_multiply(self, bubbles1, bubbles2, result_bubbles, &
                                          taylor_series_bubbles1, taylor_series_bubbles2)
        class(BubblesMultiplier), intent(inout), target :: self
        !> First input Bubbles object that is multiplied with bubbles2, this must have
        !! smaller or equal lmax than bubbles2 and must have lmax equal with smaller_lmax
        !! variable of 'self'
        type(Bubbles), target,    intent(in)    :: bubbles1
        !> Second input Bubbles object that is multiplied with bubbles1, this must have
        !! larger or equal lmax than bubbles1 and must have lmax equal with larger_lmax
        !! variable of 'self'
        type(Bubbles), target,    intent(in)    :: bubbles2
        !> Result Bubbles object that will contain the result of the multiplication 
        !! after the call. This must have lmax=bubbles1%lmax+bubbles2%lmax.
        type(Bubbles), target,    intent(inout) :: result_bubbles
        !> The off-diagonal taylor series  part of input 'bubbles1' object 
        type(Bubbles), optional, target,   intent(in)  :: taylor_series_bubbles1
        !> The off-diagonal taylor series  part of input 'bubbles2' object 
        type(Bubbles), optional, target,   intent(in)  :: taylor_series_bubbles2
        integer(INT32)                          :: l1, m1, l2, m2, ms, i, n1, n2, bubs1_i, bubs2_i, &
                                                   sz, ibub, order_number, order_number2, lmax
        integer(INT32), pointer                 :: order_numbers(:)
        real(REAL64),   pointer                 :: bf1(:, :), bf2(:, :), tbf1(:, :), tbf2(:, :), rbf(:, :) 
        real(REAL64),   pointer                 :: coefficients(:)
        real(REAL64),   allocatable, target     :: temp1(:, :), temp2(:, :)
        type(Bubbles), allocatable              :: bubbles1_copy
        type(Bubbles), allocatable              :: bubbles2_copy, taylor_series_bubbles2_copy

        if (present(taylor_series_bubbles1)) then
            lmax =   min(max(taylor_series_bubbles1%get_lmax(), bubbles1%get_lmax()) &
                   + max(taylor_series_bubbles2%get_lmax(), bubbles2%get_lmax()), &
                   self%smaller_lmax+self%larger_lmax)
        else
            lmax = min(bubbles1%get_lmax() + bubbles2%get_lmax(), self%smaller_lmax+self%larger_lmax)
        end if
        
        ! If the result, bubbles' lmax is not equal with the resulting lmax, let's
        ! reinit the bubbles
        if (      result_bubbles%get_lmax() /= lmax .or. &
            (      present(taylor_series_bubbles1) .and. &
             .not. result_bubbles%is_merge_of(taylor_series_bubbles1, taylor_series_bubbles2)) .or. &
            (.not. present(taylor_series_bubbles1) .and. &
             .not. result_bubbles%is_intersection_of(bubbles1, bubbles2)) ) then
            call result_bubbles%destroy()

            ! if there is the taylor series bubbles present, this means that the result bubbles will have 
            ! all the bubbles from the bubbles1 and bubbles2, if not, it will only contain the bubbles
            ! that are present in both.
            if (present(taylor_series_bubbles1)) then
                result_bubbles = taylor_series_bubbles1%merge_with(taylor_series_bubbles2,&
                                                                   lmax = lmax)
            else
                result_bubbles = bubbles1%get_intersection(bubbles2, lmax = lmax)
            end if
        end if


        ! set the correct k to the result bubbles
        result_bubbles%k = bubbles1%get_k() + bubbles2%get_k()

#ifdef HAVE_CUDA
           
  
        !bubbles2_copy = bubbles2
        !taylor_series_bubbles2_copy = taylor_series_bubbles2
        if (present(taylor_series_bubbles1) .and. present(taylor_series_bubbles2)) then
            ! option (1), see subroutine comments
            ! set the correct k-values first
            call BubblesMultiplier_set_ks(self%cuda_interface, bubbles1%get_k(), bubbles2%get_k(),  &
                         result_bubbles%get_k(), taylor_series_bubbles1%get_k(), taylor_series_bubbles2%get_k())
            do order_number = 1, result_bubbles%get_nbub()
                ibub = result_bubbles%get_ibub(order_number)
                if (bubbles1%contains_ibub(ibub)) then
                    order_number2 = bubbles1%get_order_number(ibub)
                    bf1 => bubbles1%bf(order_number2)%p
                else 
                    call C_F_POINTER(C_NULL_PTR, bf1, [0, 0]) 
                end if

                if (bubbles2%contains_ibub(ibub)) then
                    order_number2 = bubbles2%get_order_number(ibub)
                    bf2 => bubbles2%bf(order_number2)%p
                else
                    call C_F_POINTER(C_NULL_PTR, bf2, [0, 0])
                end if
                
                order_number2 = taylor_series_bubbles1%get_order_number(ibub)
                tbf1 => taylor_series_bubbles1%bf(order_number2)%p
                order_number2 = taylor_series_bubbles2%get_order_number(ibub)
                tbf2 => taylor_series_bubbles2%bf(order_number2)%p
                rbf  => result_bubbles%bf(order_number)%p
                call BubblesMultiplier_multiply_bubble_cuda( &
                            self%get_cuda_interface(), ibub, bf1, bf2, rbf, tbf1, tbf2, &
                            bubbles1%get_lmax(), bubbles2%get_lmax(), &
                            taylor_series_bubbles1%get_lmax(), &
                            taylor_series_bubbles2%get_lmax())
                nullify(tbf1, tbf2, bf1, bf2, rbf)

            end do
        else
            ! option (2), see subroutine comments
            call BubblesMultiplier_set_ks(self%cuda_interface, bubbles1%get_k(), bubbles2%get_k(),  &
                                          result_bubbles%get_k(), 0, 0)
            call C_F_POINTER(C_NULL_PTR, tbf1, [0, 0]) 
            call C_F_POINTER(C_NULL_PTR, tbf2, [0, 0]) 
            do order_number = 1, result_bubbles%get_nbub()
                ibub = result_bubbles%get_ibub(order_number)

                ! set the pointers
                order_number2 = bubbles1%get_order_number(ibub)
                bf1 => bubbles1%bf(order_number2)%p
                order_number2 = bubbles2%get_order_number(ibub)
                bf2 => bubbles2%bf(order_number2)%p
                rbf  => result_bubbles%bf(order_number)%p
                
                ! do the cuda-call
                call BubblesMultiplier_multiply_bubble_cuda( &
                            self%get_cuda_interface(), ibub, bf1, bf2, rbf, tbf1, tbf2, &
                            bubbles1%get_lmax(), bubbles2%get_lmax(), 0, 0)
            end do
        end if
 
        call CUDASync_all()

         ! download the result to cpu
        call BubblesMultiplier_download_result_cuda(self%cuda_interface, result_bubbles%get_lmax(), &
                                                    result_bubbles%get_ibubs(), result_bubbles%get_nbub())
        call CUDASync_all()

        !call bubbles2_copy%destroy()
        !call taylor_series_bubbles2_copy%destroy()
 
        ! communicate between nodes to get complete result bubbles at 
        ! every node, DISABLED at the moment because the communication takes
        ! so long. It is more economical to calculate the entire multiplication at
        ! each node
        !call self%communicate_result(result_bubbles)
#else      
        result_bubbles = 0.0d0
        ! set the correct k to the result bubbles
        if (present(taylor_series_bubbles1) .and. present(taylor_series_bubbles2)) then
            bubbles1_copy = bubbles1 + taylor_series_bubbles1
            bubbles2_copy = bubbles2 + taylor_series_bubbles2
        else
            bubbles1_copy = Bubbles(bubbles1, copy_content = .TRUE.)
            bubbles2_copy = Bubbles(bubbles2, copy_content = .TRUE.)
        end if  

        ! Init new as the intersection of bubbles, i.e., all bubbles 
        ! that are in both 'bubs1' and 'bubs2' are in 'new'
        ! Iterate over centers
        do ibub=1,result_bubbles%nbub
            ! get order number of bubble ibub in bubs1 and bubs2
            bubs1_i = bubbles1%get_order_number(result_bubbles%ibubs(ibub))
            bubs2_i = bubbles2%get_order_number(result_bubbles%ibubs(ibub))
            ! Iterate over l1
            do l1=0,bubbles1%get_lmax()
                ! Iterate over m1
                do m1=-l1,l1
                    n1=idx(l1,m1)
                    ! Iterate over l2>=l1
                    do l2=0,bubbles2%get_lmax()
                        do m2=-l2,l2
                            n2=idx(l2,m2)
!> @todo Change prodcoeffs into an iterator that returns coefficients and
!! then is disposed, e.g.\
!! ~~~~~~~~~~~~~~~~~~~~~~~~~~~{.F90}
!! prodcoeffs=YProdCoeffs(max([l1,l2,l]))
!! ...
!! while prodcoeffs%next(l1,m1,l2,m2,l,m,cf)
!!   f%p(:,idx(l,m)) += cf * f1%p(:,idx(l1,m1)) * f2%p(:,idx(l2,m2))
!!           -- OR --
!! while prodcoeffs%next(lm1,lm2,lm,cf)
!!   f%p(:,lm) += cf * f1%p(:,lm1) * f2%p(:,lm2)
!! ~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            ! Get which spherical harmonics result from the
                            ! product Y1*Y2 (nout) and which are their
                            ! coefficients
                            call self%yproduct%get_coeffs(l1,m1,l2,m2,sz, order_numbers, coefficients)
                            !print *, l1, m1, l2, m2, "order nos", order_numbers,coefficients, sz
                            ! Iterate over the spherical harmonics coming out
                            ! from t the product of Y1 and Y2
                            do i=1,sz
                                ! Calculate product of radial functions.
                                result_bubbles%bf(ibub)%p(:, order_numbers(i))= &
                                     result_bubbles%bf(ibub)%p(:, order_numbers(i)) + &
                                        coefficients(i) * bubbles1_copy%bf(bubs1_i)%p(:,n1) &
                                                         * bubbles2_copy%bf(bubs2_i)%p(:,n2)
                            end do
                            nullify(order_numbers)
                            nullify(coefficients)
                        end do
                    end do
                end do
            end do
        end do

        call bubbles1_copy%destroy()
        call bubbles2_copy%destroy()
#endif
    end subroutine

    !> Exchange information between nodes to get complete result bubbles
    !! as at least the cuda/c++ implementation calculates only part of the
    !! result. The part calculated by each processor is by number of points,
    !! meaning that each lmax of one point is calculated by one node. The
    !! points are split between nodes.
    subroutine BubblesMultiplier_communicate_result(self, result_bubbles)
        class(BubblesMultiplier), intent(in)    :: self
        type(Bubbles),            intent(inout) :: result_bubbles
        integer                                 :: number_of_points, remainder, communication_type, ierr
        integer                                 :: processor_number, bf_shape(2), &
                                                   communication_shape(2), start_indices(2), &
                                                   ibub
        logical                                 :: main

        if (nproc > 1) then
#ifdef HAVE_MPI
#ifdef HAVE_OMP
        !$OMP BARRIER
        !$OMP MASTER 
        
#endif
            call mpi_is_thread_main(main, ierr)
            if (main) then
                do ibub = 1, result_bubbles%get_nbub()
                    bf_shape = shape(result_bubbles%bf(ibub)%p)
                    ! because all the l,m pairs are communicated for each point, the
                    ! first lm -index will always be 0 (C-indexing used by MPI), and the communicated subarray 
                    ! will have the other axis length which is equal with number of l,m -pairs total.
                    start_indices(2) = 0
                    start_indices(1) = 0
                    communication_shape(2) = bf_shape(2)
                    do processor_number = 0, nproc-1
                        ! get the number of points handled by this node
                        if (mod(bf_shape(1), nproc) > processor_number) then
                            remainder = 1
                        else
                            remainder = 0
                        end if
                        communication_shape(1) = bf_shape(1) / nproc + remainder
                        ! create the subarray structure
                        call MPI_TYPE_CREATE_SUBARRAY(2, bf_shape, communication_shape, start_indices, &
                                            MPI_ORDER_FORTRAN, MPI_DOUBLE_PRECISION, communication_type, &
                                            ierr)
                        call MPI_TYPE_COMMIT(communication_type, ierr)
                        
                        ! broadcast the subarray contents from processor 'processor_number' to the other nodes    
                        call MPI_BCAST(result_bubbles%bf(ibub)%p(1, 1), 1, communication_type, &
                                    processor_number, MPI_COMM_WORLD, ierr)

                        call MPI_TYPE_FREE(communication_type, ierr)

                        ! get the start index of the next communication by increasing the current
                        ! start index by the shape of the communication
                        start_indices(1) = start_indices(1) + communication_shape(1)
                    end do
                end do
                
                call MPI_BARRIER(MPI_COMM_WORLD, ierr)
            end if
#ifdef HAVE_OMP
        !$OMP END MASTER
        !$OMP BARRIER
#endif
#endif
        end if
    end subroutine

    subroutine BubblesMultiplier_destroy(self)
        class(BubblesMultiplier), intent(inout) :: self
        call self%yproduct%destroy()
#ifdef HAVE_CUDA
        if (allocated(self%cuda_interface)) deallocate(self%cuda_interface)
#endif
    end subroutine

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                   BubblesMultipliers methods                            %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    !> Multiply the bubbles1 with bubbles2 and store the result to result_bubbles.
    !! Selects/initializes and uses a BubblesMultiplier to do that 
    subroutine BubblesMultipliers_multiply(self, bubbles1, bubbles2, result_bubbles, &
                   taylor_series_bubbles1, taylor_series_bubbles2)
        class(BubblesMultipliers), target, intent(inout) :: self
        !> First input Bubbles object that is multiplied with bubbles2
        type(Bubbles), target,             intent(in)    :: bubbles1
        !> Second input Bubbles object that is multiplied with bubbles1
        type(Bubbles), target,             intent(in)    :: bubbles2
        !> The result Bubbles object, which should be initialized before calling this function
        type(Bubbles),                     intent(inout) :: result_bubbles
        type(Bubbles), target, optional,   intent(in)    :: taylor_series_bubbles1
        type(Bubbles), target, optional,   intent(in)    :: taylor_series_bubbles2
        type(Bubbles)                                    :: temp
        !! pointer to the input Bubbles object which has smaller lmax value
        type(Bubbles), pointer                           :: smaller_lmax_bubbles
        !! pointer to the input Bubbles object which has larger lmax value
        type(Bubbles), pointer                           :: larger_lmax_bubbles
        type(Bubbles), pointer                           :: larger_lmax_taylor_bubbles
        type(Bubbles), pointer                           :: smaller_lmax_taylor_bubbles
        type(BubblesMultiplier),   pointer               :: multiplier

        ! determine which of the input bubbles has larger and smaller lmax 
        ! and set the corresponding pointers
        if (bubbles1%get_lmax() <= bubbles2%get_lmax()) then
            smaller_lmax_bubbles => bubbles1
            larger_lmax_bubbles  => bubbles2
            if (present(taylor_series_bubbles1) .and. present(taylor_series_bubbles2)) then
                larger_lmax_taylor_bubbles => taylor_series_bubbles2
                smaller_lmax_taylor_bubbles => taylor_series_bubbles1
            end if
        else
            smaller_lmax_bubbles => bubbles2
            larger_lmax_bubbles  => bubbles1
            if (present(taylor_series_bubbles1) .and. present(taylor_series_bubbles2)) then
                larger_lmax_taylor_bubbles => taylor_series_bubbles1
                smaller_lmax_taylor_bubbles => taylor_series_bubbles2
            end if
        end if
        
        ! get a suitable multiplier, if none is stored the method creates a suitable one
        if (present(taylor_series_bubbles1) .and. present(taylor_series_bubbles2)) then
            multiplier => self%get_multiplier(bubbles1, bubbles2, &
                              taylor_series_bubbles1, taylor_series_bubbles2)
        else
            multiplier => self%get_multiplier(bubbles1, bubbles2)
        end if
        
        call bigben%split("Multiplication")
        if (present(taylor_series_bubbles1) .and. present(taylor_series_bubbles2)) then
            ! we now have the correct multiplier, let's use it to do the multiplication
            call multiplier%multiply(smaller_lmax_bubbles, larger_lmax_bubbles, result_bubbles, &
                                    smaller_lmax_taylor_bubbles, larger_lmax_taylor_bubbles)
        else 
            call multiplier%multiply(smaller_lmax_bubbles, larger_lmax_bubbles, result_bubbles)
            
        end if
        
        nullify(multiplier)
        call bigben%stop()
    end subroutine

    !> Returns a suitable multiplier for the input bubbles, if none or no suitable multiplier
    !! is stored this function creates a suitable multiplier
    function BubblesMultipliers_get_multiplier(self, bubbles1, bubbles2, &
                 taylor_series_bubbles1, taylor_series_bubbles2, complex_version) &
                 result(multiplier)
        class(BubblesMultipliers), target, intent(inout) :: self
        !> First input Bubbles object that is multiplied with bubbles2
        type(Bubbles), target,             intent(in)    :: bubbles1
        !> Second input Bubbles object that is multiplied with bubbles1
        type(Bubbles), target,             intent(in)    :: bubbles2
        type(Bubbles), target, optional,   intent(in)    :: taylor_series_bubbles1
        type(Bubbles), target, optional,   intent(in)    :: taylor_series_bubbles2
        logical,               optional                  :: complex_version
        !! pointer to the input Bubbles object which has smaller lmax value
        type(Bubbles), pointer                           :: smaller_lmax_bubbles
        !! pointer to the input Bubbles object which has larger lmax value
        type(Bubbles), pointer                           :: larger_lmax_bubbles
        type(Bubbles), pointer                           :: larger_lmax_taylor_bubbles
        type(Bubbles), pointer                           :: smaller_lmax_taylor_bubbles
        type(BubblesMultiplier),   pointer               :: multiplier
        logical                                          :: multiplier_found
        integer                                          :: i

        ! determine which of the input bubbles has larger and smaller lmax 
        ! and set the corresponding pointers
        if (bubbles1%get_lmax() <= bubbles2%get_lmax()) then
            smaller_lmax_bubbles => bubbles1
            larger_lmax_bubbles  => bubbles2
            if (present(taylor_series_bubbles1) .and. present(taylor_series_bubbles2)) then
                larger_lmax_taylor_bubbles => taylor_series_bubbles2
                smaller_lmax_taylor_bubbles => taylor_series_bubbles1
            end if
        else
            smaller_lmax_bubbles => bubbles2
            larger_lmax_bubbles  => bubbles1
            if (present(taylor_series_bubbles1) .and. present(taylor_series_bubbles2)) then
                larger_lmax_taylor_bubbles => taylor_series_bubbles1
                smaller_lmax_taylor_bubbles => taylor_series_bubbles2
            end if
        end if

        ! determine if suitable multiplier is already stored in the object
        if (.not. allocated(self%multipliers)) then
            multiplier_found = .FALSE.
        else
            multiplier_found = .FALSE.
            do i = 1, self%multiplier_count
                if (smaller_lmax_bubbles%get_lmax() == self%multipliers(i)%smaller_lmax .AND. & 
                    larger_lmax_bubbles%get_lmax() == self%multipliers(i)%larger_lmax ) then
                    multiplier_found = .TRUE.
                    multiplier => self%multipliers(i)
                    exit
                end if
            end do
        end if
    
        if (.not. allocated(self%multipliers)) then
            allocate(self%multipliers(15))
            !first_multiplier => NULL()
        end if 

        ! if we did not find a multiplier, let's create a new one
        if (.not. multiplier_found) then
            !print *, "--------- MULTIPLIER NOT FOUND.", bubbles1%get_lmax(), bubbles2%get_lmax()
            call bigben%split("Init multiplier")
            self%multiplier_count = self%multiplier_count + 1

            if (present(taylor_series_bubbles1) .and. present(taylor_series_bubbles2)) then
                self%multipliers(self%multiplier_count) = BubblesMultiplier(smaller_lmax_bubbles, larger_lmax_bubbles, &
                                                   self%first_multiplier, &
                                                   smaller_lmax_taylor_bubbles, larger_lmax_taylor_bubbles)
            else
                if (present(complex_version)) then
                    if (complex_version) then
                        self%multipliers(self%multiplier_count) = BubblesMultiplier(smaller_lmax_bubbles, larger_lmax_bubbles, &
                                                    self%first_multiplier, &
                                                    smaller_lmax_bubbles, larger_lmax_bubbles)
                    else 
                        self%multipliers(self%multiplier_count) = BubblesMultiplier(smaller_lmax_bubbles, larger_lmax_bubbles, &
                        self%first_multiplier)
                    end if
                else
                   self%multipliers(self%multiplier_count) = BubblesMultiplier(smaller_lmax_bubbles, &
                        larger_lmax_bubbles, self%first_multiplier)
                end if
            end if
       
            ! finally, let's get the pointer to the recently initialized multiplier
            multiplier => self%multipliers(self%multiplier_count)
            if (self%multiplier_count == 1) self%first_multiplier => self%multipliers(self%multiplier_count)
            call bigben%stop()
        end if
    end function

    !> Init multipliers from a Bubbles-mould 
    subroutine BubblesMultipliers_init_multipliers_from_mould(self, mould)
        class(BubblesMultipliers), intent(inout) :: self
        type(Bubbles),             intent(in)    :: mould
        type(Bubbles)                            :: bubbles2, bubbles3, bubbles4, bubbles5
        type(BubblesMultiplier), pointer         :: multiplier

        ! get the lmax, 2 multiplier
        bubbles2 = Bubbles(mould, lmax=15)
        bubbles3 = Bubbles(mould, lmax=15)
        bubbles4 = Bubbles(mould, lmax=15)
        bubbles5 = Bubbles(mould, lmax=15)
        nullify(self%first_multiplier)
        multiplier => self%get_multiplier(bubbles2, bubbles3, bubbles4, bubbles5)
        call bubbles2%destroy()
        call bubbles3%destroy()
        call bubbles4%destroy()
        call bubbles5%destroy()

    end subroutine

    !> Destroys the fortran and cuda objects contained in this object
    subroutine BubblesMultipliers_destroy(self)
        class(BubblesMultipliers), intent(inout) :: self
        integer                                 :: i
        if (allocated(self%multipliers)) then
            do i = 1, self%multiplier_count
#ifdef HAVE_CUDA
                call self%multipliers(i)%cuda_destroy()
#endif
                call self%multipliers(i)%destroy()
            end do
            deallocate(self%multipliers)
        end if
        self%multiplier_count = 0
        nullify(self%first_multiplier)
    end subroutine

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                   BubblesEvaluator methods                              %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    

    function BubblesEvaluator_init(high_memory_profile) result(new)
        logical,      intent(in)             :: high_memory_profile
        type(BubblesEvaluator)               :: new

        new%high_memory_profile = high_memory_profile
        
        ! initialize the cabability to evaluate the gradients
        call new%init_gradient_factors(20)

#ifdef HAVE_CUDA
        call new%cuda_init()
#endif 
    end function


    !> Inits the coefficients needed for evaluation of gradients of Real Regular Solid Harmonics
    !! and consequently real spherical harmonics and Bubbles.
    subroutine BubblesEvaluator_init_gradient_factors(self, lmax)
        class(BubblesEvaluator), intent(inout)       :: self
        integer,                 intent(in)          :: lmax
        real(REAL64),            allocatable         :: n_map(:), o_map(:)
        integer                                      :: l, m

        allocate(n_map((lmax+1)**2))
        allocate(o_map((lmax+1)**2))
        allocate(self%gradient_factors_x((lmax+1)**2, 2))
        allocate(self%gradient_factors_y((lmax+1)**2, 2))
        allocate(self%gradient_factors_z((lmax+1)**2))
        do l = 0, lmax
            do m = -l, l
                ! evaluate the values for the assistance funtions o and n
                o_map(idx(l, m)) = o(l, m, o_map)
                n_map(idx(l, m)) = n(l, m, n_map)
            end do

            ! NOTE: the o and n values of order l must be pre-evaluated, thus
            ! this and the previous loop cannot be merged
            do m = -l, l

                ! evaluate the gradient factor to z-direction
                self%gradient_factors_z(idx(l, m)) = c(l, m, self%gradient_factors_z)

                ! evaluate the gradient factors to x-direction
                self%gradient_factors_x(idx(l, m), 1) = e(l, m, n_map, o_map)
                self%gradient_factors_x(idx(l, m), 2) = f(l, m, n_map, o_map)

                ! evaluate the gradient factors to y-direction
                self%gradient_factors_y(idx(l, m), 1) = f(l, m, n_map, o_map)
                self%gradient_factors_y(idx(l, m), 2) = -e(l, m, n_map, o_map)
                
            end do
        end do

        ! do a couple of fixes to the recursion results of f's
        ! (m=-1 for the x derivatives is 0 and m = +1 for the y derivatives)
        do l = 1, lmax
            self%gradient_factors_x(idx(l, -1), 2) = 0.0d0
            self%gradient_factors_y(idx(l,  1), 2) = 0.0d0
        end do
        self%gradient_factors_x(idx(0, 0), 2) = 0.0d0
        self%gradient_factors_y(idx(0, 0), 2) = 0.0d0
        deallocate(n_map, o_map)
        !call print_lm_map(self%gradient_factors_x(:, 1), lmax)
        !call print_lm_map(self%gradient_factors_x(:, 2), lmax)
        !call print_lm_map(self%gradient_factors_y(:, 1), 4)
        !call print_lm_map(self%gradient_factors_y(:, 2), 4)
        !call print_lm_map(self%gradient_factors_z, lmax)
    end subroutine

    subroutine print_lm_map(lm_map, lmax)
        real(REAL64), intent(in)  :: lm_map(:)
        integer,      intent(in)  :: lmax
        character(256)            :: line
        integer                   :: l, m
        do m = lmax, -lmax, -1
            
            write(*,'(i2, ":", a)', advance='no') m, repeat(" ",7*abs(m))
            do l = abs(m), lmax 
                write(*, '(f7.3) ', advance="no") lm_map(idx(l, m))
            end do
            write(*, *) " "
        end do
    end subroutine

    pure function a(l, m)
        integer,     intent(in)  :: l, m
        !> The result value
        real(REAL64)             :: a

        if (abs(m) /= l) then
            a = (2.0d0*l-1.0d0) / sqrt(dble((l+m)*(l-m)))
        else if (l == 1 .or. l == 0) then
            a = 1.0d0
        else if (abs(m) == l) then
            a = sqrt(dble(2*l-1)/dble(2*l))
        else  
            a = 1.0d0
        end if
    end function

    pure function b(l, m)
        integer,     intent(in)  :: l, m
        !> The result value
        real(REAL64)             :: b
        if (l-abs(m) >= 2) then
            b = sqrt(dble((l+m-1)*(l-m-1)) / dble((l+m)*(l-m)))
        else
            b = 0.0d0
        end if
    end function

    pure function c(l, m, c_map)
        integer,      intent(in) :: l, m
        !> The result value
        real(REAL64)             :: c
        !> A Map containing the c-values for previous l, m values.
        !! Should contain at least values with l = l-1
        real(REAL64), intent(in) :: c_map(:)

        if (l-abs(m) >= 1) then
            c = a(l, m) + (a(l, m) * c_map(idx(l-1, m)) - 2.0d0 * b(l, m)) / a(l-1, m)
        else if (l == 1 .and. m == 0) then
            c =  1.0d0
        else
            c = 0.0d0
        end if
    end function

    pure function n(l, m, n_map)
        integer,      intent(in) :: l, m
        !> The result value
        real(REAL64)             :: n
        !> A Map containing the n-values for previous l, m values. 
        !! Should contain at least values with l in {l, l-1}
        real(REAL64), intent(in) :: n_map(:)

        if  (abs(m) > l) then
            n = 1.0d0
        else if (l == 0 .and. m == 0) then
            n = 1.0d0
        else if (l == m) then
            n = a(l, m) * n_map(idx(l-1, m-1))
        else if (l == -m) then
            n = a(l, m) * n_map(idx(l-1, m+1))
        else
            if (abs(m) <= l -1 ) then
                n = dble(l+abs(m)) / sqrt(dble(l-m)*dble(l+m)) * n_map(idx(l-1, m))
            else
                n = 1.0d0
            end if
        end if
    end function
    
    pure function o(l, m, o_map)
        integer,      intent(in) :: l, m
        !> The result value
        real(REAL64)             :: o
        !> A Map containing the o-values for previous l, m values. 
        !! Should contain at least values with l in {l, l-1}
        real(REAL64), intent(in) :: o_map(:)

        if (l-abs(m) <= 0) then
            if (l <= 0) then
                o = 1.0d0
            else if (m == -2 .and. l == 2) then
                o = a(2, 2)
            else
                o = a(l, l) * o_map(idx(l-1, abs(m)-1))
            end if
        else if (l-abs(m) == 1) then
            o = a(l, m) * o_map(idx(l-1, m))
        else if (mod(l-abs(m), 2) == 1) then
            o = a(l, m) * o_map(idx(l-1, m)) - b(l, m) * o_map(idx(l-2, m))
        
        else if (mod(l-abs(m), 2) == 0) then
            o = -b(l, m) * o_map(idx(l-2, m))
        end if
    end function
  

    function e(l, m, n_map, o_map)
        integer,      intent(in) :: l, m
        !> The result value
        real(REAL64)             :: e
        !> A Map containing the o-values for previous l, m values. 
        !! Should contain at least values with l in {l, l-1}
        real(REAL64), intent(in) :: o_map(:)
        !> A Map containing the n-values for previous l, m values. 
        !! Should contain at least values with l in {l, l-1}
        real(REAL64), intent(in) :: n_map(:)
        if (abs(m) > l) then
            e = 1.0
        else if (m >= 0) then
            e = m * n_map(idx(l, m))
            if (l /= 0 .and. abs(m-1) <= l-1) then
                e = e / n_map(idx(l-1, m-1))
            end if
        else
            e =   (max_x(l, -m) + max_r2x2(l, -m)) * o_map(idx(l, m))
            if (m+1 <= l-1 .and. l -1 >= 0) then
                  e = e - abs(m) * n_map(idx(l, m)) / n_map(idx(l-1, m+1)) * o_map(idx(l-1, m+1))
            end if
            if (m-1 >= -l) then
                e = e / o_map(idx(l-1, m-1))
            end if
        end if
    end function
    


    function f(l, m, n_map, o_map)
        integer,      intent(in) :: l, m
        !> The result value
        real(REAL64)             :: f
        !> A Map containing the n-values for previous l, m values. 
        !! Should contain at least values with l = [input] l-1
        real(REAL64), intent(in) :: n_map(:)
        !> A Map containing the o-values for previous l, m values. 
        !! Should contain at least values with l = [input] l-1
        real(REAL64), intent(in) :: o_map(:)
        if (abs(m) > l) then
            f = 0.0
        else if (l == 0 .and. m == 0) then
            f = 1.0
        else if (m >= 0) then
            f =  (max_x(l, m) + max_r2x2(l, m)) * o_map(idx(l, m)) 
            if (abs(m-1) <= l-1 .and. l-1 >= 0) then
                f = f - m * n_map(idx(l, m)) /  n_map(idx(l-1, m-1)) * o_map(idx(l-1, m-1))
            end if
            if (m+1 <= l-1) then
                f = f / o_map(idx(l-1, m+1))
            end if
        else
            f = abs(m) * n_map(idx(l, m)) / n_map(idx(l-1, m+1))
        end if
    end function

    pure function max_x(l, m)
        integer,      intent(in) :: l, m
        !> The result value
        real(REAL64)             :: max_x
        if (m >= 0) then
            max_x = m
        else
            max_x = abs(m)-1
        end if
    end function


    pure function max_r2x2(l, m)
        integer,      intent(in) :: l, m
        !> The result value
        real(REAL64)             :: max_r2x2
        max_r2x2 = (l-abs(m)) / 2 * 2
    end function
    

#ifdef HAVE_CUDA
    subroutine BubblesEvaluator_cuda_destroy(self)
        class(BubblesEvaluator), intent(inout)       :: self

    end subroutine

    subroutine BubblesEvaluator_cuda_init(self) 
        class(BubblesEvaluator), intent(inout)       :: self

        self%cuda_interface = BubblesEvaluator_init_cuda(stream_container)
     end subroutine
#endif

    !> Evaluate the Bubbles. 
    !! NOTE: if cuda is enabled this subroutine continues to evaluate in the background
    !! and 'get_results' will synchronize the cuda and cpus
    subroutine BubblesEvaluator_evaluate_grid_from_bubbles(self, bubbls, grid, derivative_x, &
                                                           derivative_y, derivative_z, &
                                                           output_function_cube, &
                                                           output_derivative_x_cube, &
                                                           output_derivative_y_cube, &
                                                           output_derivative_z_cube, ibubs)
        class(BubblesEvaluator), intent(inout) :: self
        type(Bubbles),           intent(in)    :: bubbls
        type(Grid3D),            intent(in)    :: grid
        type(Bubbles),           intent(in)    :: derivative_x, derivative_y, derivative_z
        real(REAL64),            intent(inout) :: output_function_cube(:, :, :)
        real(REAL64), optional,  intent(inout) :: output_derivative_x_cube(:, :, :)
        real(REAL64), optional,  intent(inout) :: output_derivative_y_cube(:, :, :)
        real(REAL64), optional,  intent(inout) :: output_derivative_z_cube(:, :, :)
        integer,   optional,     intent(in)    :: ibubs(:)
        real(REAL64), allocatable              :: temp_cube(:, :, :)
        integer                                :: output_shape(3)
#ifdef HAVE_CUDA
        call self%set_output_grid(grid)
        
        call self%set_input_bubbles(bubbls, ibubs)
        call self%cuda_evaluate_grid_without_gradients(output_function_cube)
        call CUDASync_all()

        call self%set_input_bubbles(derivative_x, ibubs)
        call self%cuda_evaluate_grid_without_gradients(output_derivative_x_cube)
        call CUDASync_all()

        call self%set_input_bubbles(derivative_y, ibubs)
        call self%cuda_evaluate_grid_without_gradients(output_derivative_y_cube)
        call CUDASync_all()

        call self%set_input_bubbles(derivative_z, ibubs)
        call self%cuda_evaluate_grid_without_gradients(output_derivative_z_cube)
        call CUDASync_all()
#else 
        output_shape = shape(output_function_cube)
        output_function_cube(:, :, :) = &
            reshape(bubbls%eval(grid%get_all_grid_points(), ibubs = ibubs), output_shape)

        output_derivative_x_cube(:, :, :) = &
            reshape(derivative_x%eval(grid%get_all_grid_points(), ibubs = ibubs), output_shape)

        output_derivative_y_cube(:, :, :) = &
            reshape(derivative_y%eval(grid%get_all_grid_points(), ibubs = ibubs), output_shape)

        output_derivative_z_cube(:, :, :) = &
            reshape(derivative_z%eval(grid%get_all_grid_points(), ibubs = ibubs), output_shape)
#endif
    end subroutine

     !> Evaluate the Bubbles. 
    !! NOTE: if cuda is enabled this subroutine continues to evaluate in the background
    !! and 'get_results' will synchronize the cuda and cpus
    subroutine BubblesEvaluator_evaluate_points_from_bubbles(self, bubbls, derivative_x, derivative_y, derivative_z, &
                                                             result_points, output_derivative_x_points, &
                                                             output_derivative_y_points, &
                                                             output_derivative_z_points, ibubs)
        class(BubblesEvaluator), intent(inout) :: self
        type(Bubbles),           intent(in)    :: bubbls
        type(Bubbles),           intent(in)    :: derivative_x, derivative_y, derivative_z
        type(Points),            intent(inout) :: result_points, output_derivative_x_points, &
                                                  output_derivative_y_points, &
                                                  output_derivative_z_points
        integer,   optional,     intent(in)    :: ibubs(:)
#ifdef HAVE_CUDA

        call self%set_output_points(result_points)
        
        call self%set_input_bubbles(bubbls, ibubs)
        call self%cuda_evaluate_points_without_gradients(result_points)
        call CUDASync_all()

        call self%set_input_bubbles(derivative_x, ibubs)
        call self%cuda_evaluate_points_without_gradients(output_derivative_x_points)
        call CUDASync_all()

        call self%set_input_bubbles(derivative_y, ibubs)
        call self%cuda_evaluate_points_without_gradients(output_derivative_y_points)
        call CUDASync_all()

        call self%set_input_bubbles(derivative_z, ibubs)
        call self%cuda_evaluate_points_without_gradients(output_derivative_z_points)
        call CUDASync_all()
#else 
        
        output_derivative_x_points%values(:) = &
            derivative_x%eval(output_derivative_x_points%point_coordinates%coordinates, ibubs = ibubs)
        output_derivative_y_points%values(:) = &
            derivative_y%eval(output_derivative_y_points%point_coordinates%coordinates, ibubs = ibubs)
        output_derivative_z_points%values(:) = &
            derivative_z%eval(output_derivative_z_points%point_coordinates%coordinates, ibubs = ibubs)
        result_points%values(:) = &
            bubbls%eval(result_points%point_coordinates%coordinates)
#endif
    end subroutine

    subroutine BubblesEvaluator_destroy_input_bubbles(self)
        class(BubblesEvaluator),    intent(inout) :: self
        
        if (allocated(self%bubbles)) then
#ifdef HAVE_CUDA
            call self%bubbles%cuda_destroy()
#endif
            call self%bubbles%destroy()
            deallocate(self%bubbles)
        end if
    end subroutine

    subroutine BubblesEvaluator_set_input_bubbles(self, input_bubbles, ibubs)
        class(BubblesEvaluator),    intent(inout) :: self
        type(Bubbles),              intent(in)    :: input_bubbles
        integer,         optional,  intent(in)    :: ibubs(:)
        logical                                   :: reallocate
        type(C_PTR)                               :: current_bubbles

        reallocate = .not. allocated(self%bubbles)
        if (.not. reallocate) then
            if (     self%bubbles%get_nbub_global() /= input_bubbles%get_nbub_global() &
                .or. self%bubbles%get_lmax() < input_bubbles%get_lmax()) then
                reallocate = .TRUE.
            end if
        end if
    
        if (reallocate) then
            call self%destroy_input_bubbles()

            self%bubbles = Bubbles(input_bubbles, copy_content = .TRUE.)
#ifdef HAVE_CUDA
            call self%bubbles%cuda_init()
#endif
        end if

#ifdef HAVE_CUDA
        if (allocated(self%bubbles_cuda)) then
            call Bubbles_destroy_cuda(self%bubbles_cuda)
            deallocate(self%bubbles_cuda)
        end if

        if (present(ibubs)) then
            self%bubbles_cuda = Bubbles_get_sub_bubbles_cuda(self%bubbles%get_cuda_interface(), ibubs, size(ibubs))
            current_bubbles = self%bubbles_cuda
        else
            current_bubbles = self%bubbles%get_cuda_interface()
        end if
       
        call BubblesEvaluator_set_bubbles_cuda(self%get_cuda_interface(), current_bubbles)


        ! Note: the upload method has synchonization at its end, whereas the
        ! evaluation does not
        call input_bubbles%cuda_upload_all(current_bubbles)
#endif
    end subroutine

    !> Evaluate the Bubbles. 
    !! NOTE: if cuda is enabled this subroutine continues to evaluate in the background
    !! and 'get_results' will synchronize the cuda and cpus
    subroutine BubblesEvaluator_evaluate_points(self, bubbls, output_points, &
            output_derivative_x_points, output_derivative_y_points, output_derivative_z_points, ibubs)
        class(BubblesEvaluator), intent(inout) :: self
        type(Bubbles),           intent(in)    :: bubbls
        type(Points),            intent(inout) :: output_points
        integer,      optional,  intent(in)    :: ibubs(:)
        type(Points), optional,  intent(inout) :: output_derivative_x_points, output_derivative_y_points, &
                                                  output_derivative_z_points
#ifdef HAVE_CUDA

        call self%set_input_bubbles(bubbls, ibubs)
        call self%set_output_points(output_points)
        

        if (      self%high_memory_profile &
            .and. present(output_derivative_x_points) &
            .and. present(output_derivative_y_points) &
            .and. present(output_derivative_z_points)) then
            call output_points%set_cuda_interface(self%result_points%get_cuda_interface())
            call output_derivative_x_points%set_cuda_interface(self%output_derivative_x_points%get_cuda_interface())
            call output_derivative_y_points%set_cuda_interface(self%output_derivative_y_points%get_cuda_interface())
            call output_derivative_z_points%set_cuda_interface(self%output_derivative_z_points%get_cuda_interface())
            call output_points%cuda_set_to_zero()
            call output_derivative_x_points%cuda_set_to_zero()
            call output_derivative_y_points%cuda_set_to_zero()
            call output_derivative_z_points%cuda_set_to_zero()

            call Evaluator_evaluate_points_cuda(self%cuda_interface, output_points%get_cuda_interface(), &
                                                output_derivative_x_points%get_cuda_interface(), &
                                                output_derivative_y_points%get_cuda_interface(), &
                                                output_derivative_z_points%get_cuda_interface(), 3)

            ! start downloading the results.
            call output_points%cuda_download()
            call output_derivative_x_points%cuda_download()
            call output_derivative_y_points%cuda_download()
            call output_derivative_z_points%cuda_download()
            call CUDASync_all()
        else
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


#else 
        ! NOTE: the following procedure produces invalid values near the cusp
        
        if (      present(output_derivative_x_points) &
            .and. present(output_derivative_y_points) &
            .and. present(output_derivative_z_points)) then
            output_derivative_x_points%values(:) = &
                bubbls%eval(output_derivative_x_points%point_coordinates%coordinates,[1,0,0], ibubs = ibubs)
            output_derivative_y_points%values(:) = &
                bubbls%eval(output_derivative_y_points%point_coordinates%coordinates,[0,1,0], ibubs = ibubs)
            output_derivative_z_points%values(:) =  &
                bubbls%eval(output_derivative_y_points%point_coordinates%coordinates,[0,0,1], ibubs = ibubs)
                
        end if
        output_points%values(:) = bubbls%eval(output_points%point_coordinates%coordinates, ibubs = ibubs)
#endif
    end subroutine

    !> Evaluate the Bubbles. 
    !! NOTE: if cuda is enabled this subroutine continues to evaluate in the background
    !! and 'get_results' will synchronize the cuda and cpus
    subroutine BubblesEvaluator_evaluate_grid(self,  bubbls, grid, output_function_cube, &
                                              output_derivative_x_cube, output_derivative_y_cube, &
                                              output_derivative_z_cube, ibubs)
        class(BubblesEvaluator), intent(inout)      :: self
        type(Bubbles),             intent(in)       :: bubbls
        type(Grid3D),              intent(in)       :: grid
        real(REAL64),              intent(inout)    :: output_function_cube(:, :, :)
        real(REAL64), optional,    intent(inout)    :: output_derivative_x_cube(:, :, :)
        real(REAL64), optional,    intent(inout)    :: output_derivative_y_cube(:, :, :)
        real(REAL64), optional,    intent(inout)    :: output_derivative_z_cube(:, :, :)
        integer,      optional,    intent(in)       :: ibubs(:)
        integer                                     :: output_shape(3)
#ifdef HAVE_CUDA
        type(C_PTR)                                 :: original_bubbles, current_bubbles
        
        call self%set_input_bubbles(bubbls, ibubs)
        call self%set_output_grid(grid)

        if (self%high_memory_profile .and. present(output_derivative_x_cube) &
                                     .and. present(output_derivative_y_cube) &
                                     .and. present(output_derivative_z_cube)  ) then
            call self%result_cuda_cube%set_to_zero()
            call self%gradient_cuda_cube_x%set_to_zero()
            call self%gradient_cuda_cube_y%set_to_zero()
            call self%gradient_cuda_cube_z%set_to_zero()
            call CUDASync_all()

            call self%result_cuda_cube%set_host(output_function_cube)
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
            call self%result_cuda_cube%download()
            call self%gradient_cuda_cube_x%download()
            call self%gradient_cuda_cube_y%download()
            call self%gradient_cuda_cube_z%download()
            call CUDASync_all()
        else
            call self%cuda_evaluate_grid_without_gradients(output_function_cube)
            call CUDASync_all()
            
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
        output_shape = shape(output_function_cube)
        if (present(output_derivative_x_cube)) &
            output_derivative_x_cube(:, :, :) = &
                reshape(bubbls%eval(grid%get_all_grid_points(),[1,0,0], ibubs = ibubs), output_shape)
        if (present(output_derivative_y_cube)) &
            output_derivative_y_cube(:, :, :) = &
                reshape(bubbls%eval(grid%get_all_grid_points(),[0,1,0], ibubs = ibubs), output_shape)
        if (present(output_derivative_z_cube)) &
            output_derivative_z_cube(:, :, :) =  &
                reshape(bubbls%eval(grid%get_all_grid_points(),[0,0,1], ibubs = ibubs), output_shape)
                
        output_function_cube(:, :, :) = &
            reshape(bubbls%eval(grid%get_all_grid_points(), ibubs = ibubs), output_shape)
#endif
    end subroutine

    !> Evaluate the gradients of Bubbles-object to vector spherical harmonics
    subroutine BubblesEvaluator_evaluate_gradients_as_VSH(self, bubbls, first_term, second_term)
        class(BubblesEvaluator), intent(in)    :: self
        type(Bubbles),           intent(in)    :: bubbls
        type(Bubbles),           intent(out)   :: first_term
        type(Bubbles),           intent(out)   :: second_term
        type(Interpolator1D)                   :: interpolator_
        integer                                :: i
        real(REAL64), pointer                  :: coord(:) 
        real(REAL64), pointer                  :: input_bubbles_values(:, :)
        !> evaluation arrays for derivation
        real(REAL64), allocatable               :: values(:, :, :)

        
        ! evaluate the first term, which is the derivative of the radial part
        first_term = Bubbles(bubbls, copy_content = .FALSE., k = bubbls%k)
        do i = 1, bubbls%get_nbub()
            coord                => bubbls%gr(i)%p%get_coord()
            input_bubbles_values => bubbls%get_f(i)

            ! get the interpolator that is used in evaluation of the derivatives
            interpolator_ = Interpolator1D(bubbls%gr(i)%p, 1, ignore_first = .TRUE.)
        
            ! derivate the radial part
            values = interpolator_%eval(input_bubbles_values, coord)

            ! and set it to the result object
            first_term%bf(i)%p(:, :) = values(:, 2, :) 
            deallocate(values)

            ! destroy the interpolator
            call interpolator_%destroy()
        end do

        ! second term is the copy of original input bubbles divided by r
        second_term = Bubbles(bubbls, copy_content = .TRUE.)
        second_term%k = bubbls%k -1

    end subroutine

    subroutine BubblesEvaluator_evaluate_gradients_as_bubbles(self, bubbls, derivative_x, derivative_y, derivative_z)
        class(BubblesEvaluator), intent(in)    :: self
        !> The input bubbles for which we are evaluating the gradients
        type(Bubbles),           intent(in)    :: bubbls
        !> The derivative of the input bubbles to the x direction
        type(Bubbles),           intent(out)    :: derivative_x
        !> The derivative of the input bubbles to the y direction
        type(Bubbles),           intent(out)    :: derivative_y   
        !> The derivative of the input bubbles to the z direction
        type(Bubbles),           intent(out)    :: derivative_z   
        !> The first term of VSH containing the derivative of the radial part of the input bubbles,
        !! ie., d/dr f_lm
        type(Bubbles)                          :: first_term
        !> The second term of VSH containing the radial part of the input bubbles
        type(Bubbles)                          :: second_term  

        call self%evaluate_gradients_as_VSH(bubbls, first_term, second_term)
        call self%convert_gradients_from_VSH_to_bubbles(&
                 first_term, second_term, derivative_x, derivative_y, derivative_z)

        call first_term%destroy()
        call second_term%destroy()

    end subroutine

    subroutine BubblesEvaluator_evaluate_divergence_as_bubbles(self, vector_x, vector_y, vector_z, divergence)
        class(BubblesEvaluator), intent(in)    :: self
        !> The y component of the input bubbles vector
        type(Bubbles),           intent(in)     :: vector_x
        !> The y component of the input bubbles vector
        type(Bubbles),           intent(in)     :: vector_y   
        !> The z component of the input bubbles vector
        type(Bubbles),           intent(in)     :: vector_z   
        !> The z component of the input bubbles vector
        type(Bubbles),           intent(out)    :: divergence  

        type(Interpolator1D)                   :: interpolator_
        integer                                :: i, l, m, n, output_k
        real(REAL64), pointer                  :: coord(:) 
        real(REAL64), pointer                  :: input_bubbles_values(:, :)
        !> evaluation arrays for derivation
        real(REAL64), allocatable               :: values(:, :, :)
        type(Bubbles)                           :: x_unitary, y_unitary, z_unitary, temp_x, temp_y, temp_z, temp, temp2
        real(REAL64), allocatable               :: rpow(:), rpow2(:)

            
        
        output_k = 0
        
        ! evaluate the first term, which is the derivative of the radial part
        temp_x = Bubbles(vector_x, k = output_k, copy_content = .FALSE.)
        temp_y = Bubbles(vector_y, k = output_k, copy_content = .FALSE.)
        temp_z = Bubbles(vector_z, k = output_k, copy_content = .FALSE.)
        temp_x = 0.0d0
        temp_y = 0.0d0
        temp_z = 0.0d0


        do i = 1, vector_x%get_nbub()
            coord                => vector_x%gr(i)%p%get_coord()

            ! get the interpolator that is used in evaluation of the derivatives
            interpolator_ = Interpolator1D(vector_x%gr(i)%p, 1, ignore_first = .TRUE.)
        
            ! derivate the radial part and set it to the temp object for x
            values = interpolator_%eval(vector_x%get_f(i), coord)
            temp_x%bf(i)%p(:, :) = values(:, 2, :) 
            deallocate(values)

            ! derivate the radial part and set it to the temp object for x
            values = interpolator_%eval(vector_y%get_f(i), coord)
            temp_y%bf(i)%p(:, :) = values(:, 2, :) 
            deallocate(values)

            ! derivate the radial part and set it to the temp object for x
            values = interpolator_%eval(vector_z%get_f(i), coord)
            temp_z%bf(i)%p(:, :) = values(:, 2, :) 
            deallocate(values)

            ! destroy the interpolator
            call interpolator_%destroy()
        end do

        ! init the unitary coordinate bubbles x/r, y/r, and z/r
        x_unitary    = Bubbles(vector_x, k = 0, lmax = 1, copy_content = .FALSE.)
        y_unitary    = Bubbles(vector_x, k = 0, lmax = 1, copy_content = .FALSE.)
        z_unitary    = Bubbles(vector_x, k = 0, lmax = 1, copy_content = .FALSE.)
        do i = 1, x_unitary%get_nbub()
            x_unitary%bf(i)%p(:, :) = 0.0d0
            x_unitary%bf(i)%p(:, 4) = 1.0d0
            y_unitary%bf(i)%p(:, :) = 0.0d0
            y_unitary%bf(i)%p(:, 2) = 1.0d0
            z_unitary%bf(i)%p(:, :) = 0.0d0
            z_unitary%bf(i)%p(:, 3) = 1.0d0
        end do
        

        ! evaluate two of three terms in the gradients: 
        !      Y_lm x/r * d/dr f_lm + S_lm  d/dx 1/r^l
        !    = x/r 1/r ( r d/dr f_lm - l f_lm ) Y_lm
        ! (NOTE: in the loop the x/r is not included and the general result
        !        is stored to temp)
        ! (NOTE2: the 1/r is included as the k = -1 factor of the bubbles)
        do i = 1, vector_x%get_nbub()
            rpow  = vector_x%rpow(i, -1-output_k)
            rpow2 = vector_x%rpow(i, -output_k)
            forall (l = 0 : vector_x%get_lmax())
                forall (n = 1 : 2*l+1)
                    temp_x%bf(i)%p(:, l*l+n) = &
                        +(-l+vector_x%k) * vector_x%bf(i)%p(:, l*l+n) * rpow(:) &
                        + temp_x%bf(i)%p(:, l*l+n)  * rpow2(:)
                    temp_y%bf(i)%p(:, l*l+n) = &
                        +(-l+vector_y%k) * vector_y%bf(i)%p(:, l*l+n) * rpow(:) &
                        + temp_y%bf(i)%p(:, l*l+n)  * rpow2(:)
                    temp_z%bf(i)%p(:, l*l+n) = &
                        +(-l+vector_z%k) * vector_z%bf(i)%p(:, l*l+n) * rpow(:) &
                        + temp_z%bf(i)%p(:, l*l+n)  * rpow2(:) 
                end forall
            end forall
            deallocate(rpow, rpow2)
        end do
        ! multiply x_unitary = x/r with temp_x = 1/r ( r d/dr f_lm - l f_lm ) Y_lm and 
        ! add to the final divergence
        temp  = x_unitary * temp_x
        divergence = Bubbles(temp, copy_content = .TRUE.)
        call temp_x%destroy()
        call temp%destroy()
 
        temp = y_unitary * temp_y
        temp2 = divergence + temp
        call temp%destroy()
        call divergence%destroy()
        divergence = Bubbles(temp2, copy_content = .TRUE.)
        call temp_y%destroy()
        call temp2%destroy()

        temp = z_unitary * temp_z
        temp2 = divergence + temp
        call temp%destroy()
        call temp_z%destroy()
        call divergence%destroy()
        divergence = Bubbles(temp2, copy_content = .TRUE.)
        call temp2%destroy()
    
        ! destroy the temporary bubbles
        call x_unitary%destroy()
        call y_unitary%destroy()
        call z_unitary%destroy()

        ! evaluate the final term of the gradients
        !     f_lm 1/r^l d/dx S_lm = f_lm 1/r (g_xlm * Y_l-1,m-1 + h_xlm * Y_l-1,m+1)
        !     f_lm 1/r^l d/dy S_lm = f_lm 1/r (g_ylm * Y_l-1,-m-1 + h_ylm * Y_l-1,-m+1)
        !     f_lm 1/r^l d/dz S_lm = f_lm 1/r (c_zlm * Y_l-1,m)
        ! This is done using the pre-evaluated factors (g_xlm, h_xlm, g_ylm, h_ylm, c_zlm) stored in 
        ! self%gradient_factors_x, self%gradient_factors_y, and self%gradient_factors_z.
        ! The evaluation is done by 'BubblesEvaluator_init_gradient_factors'
        ! (NOTE: also this part contains extra 1/r, which is already included in the result bubbles)
        do i = 1, vector_x%get_nbub()
            rpow  = vector_x%rpow(i, -1-output_k)
            ! NOTE: we can start from 1 because the gradient of S_00 = 1 is 0 
            do l = 1, vector_x%get_lmax()
                do m = -l, l
                    if (abs(m-1) <= l-1) then
                        divergence%bf(i)%p(:, idx(l-1, m-1)) = divergence%bf(i)%p(:, idx(l-1, m-1)) &
                            + self%gradient_factors_x(idx(l, m), 1) * vector_x%bf(i)%p(:, idx(l, m)) * rpow(:)

                        divergence%bf(i)%p(:, idx(l-1, -m+1)) = divergence%bf(i)%p(:, idx(l-1, -m+1)) &
                            + self%gradient_factors_y(idx(l, m), 2) * vector_y%bf(i)%p(:, idx(l, m)) * rpow(:)
                    end if
                    if (abs(m+1) <= l-1) then
                        divergence%bf(i)%p(:, idx(l-1, m+1)) = divergence%bf(i)%p(:, idx(l-1, m+1)) &
                            + self%gradient_factors_x(idx(l, m), 2) * vector_x%bf(i)%p(:, idx(l, m)) * rpow(:)
                        
                        divergence%bf(i)%p(:, idx(l-1, -m-1)) = divergence%bf(i)%p(:, idx(l-1, -m-1)) &
                            + self%gradient_factors_y(idx(l, m), 1) * vector_y%bf(i)%p(:, idx(l, m)) * rpow(:)
                    end if
                    if (abs(m) /= l) then
                        divergence%bf(i)%p(:, idx(l-1, m)) = divergence%bf(i)%p(:, idx(l-1, m)) &
                            + self%gradient_factors_z(idx(l, m)) * vector_z%bf(i)%p(:, idx(l, m)) * rpow(:)
                    end if
                end do
            end do
            divergence%bf(i)%p(1, :) = 0.0d0
            deallocate(rpow)
        end do

        !call divergence%extrapolate_origo(6, lmax = 0)
        
    end subroutine

    !> Converts the gradients stored in Vector Spherical Harmonics (input Bubbles first_term and second_term)
    !! to the result Bubbles (derivative_x, derivative_y, and derivative_z). 
    !!
    !! Using this subroutine allows user to evaluate the gradients of bubbles to another set of bubbles.
    subroutine BubblesEvaluator_convert_gradients_from_VSH_to_bubbles(self, first_term, second_term, &
                                                            derivative_x, derivative_y, derivative_z)
        class(BubblesEvaluator), intent(in)    :: self
        !> The first term of VSH containing the derivative of the radial part of the input bubbles,
        !! ie., d/dr f_lm
        type(Bubbles),           intent(in)    :: first_term
        !> The second term of VSH containing the radial part of the input bubbles
        type(Bubbles),           intent(in)    :: second_term  
        !> The derivative of the input bubbles to the x direction
        type(Bubbles),           intent(out)    :: derivative_x
        !> The derivative of the input bubbles to the y direction
        type(Bubbles),           intent(out)    :: derivative_y   
        !> The derivative of the input bubbles to the z direction
        type(Bubbles),           intent(out)    :: derivative_z   

        type(Bubbles)                           :: x_unitary, y_unitary, z_unitary, temp
        real(REAL64), allocatable               :: rpow(:)
        integer                                 :: i, l, m, n


        ! init the unitary coordinate bubbles x/r, y/r, and z/r
        x_unitary    = Bubbles(first_term, k = 0, lmax = 1, copy_content = .FALSE.)
        y_unitary    = Bubbles(first_term, k = 0, lmax = 1, copy_content = .FALSE.)
        z_unitary    = Bubbles(first_term, k = 0, lmax = 1, copy_content = .FALSE.)
        do i = 1, x_unitary%get_nbub()
            x_unitary%bf(i)%p(:, :) = 0.0d0
            x_unitary%bf(i)%p(:, 4) = 1.0d0
            y_unitary%bf(i)%p(:, :) = 0.0d0
            y_unitary%bf(i)%p(:, 2) = 1.0d0
            z_unitary%bf(i)%p(:, :) = 0.0d0
            z_unitary%bf(i)%p(:, 3) = 1.0d0
        end do
        
        
        temp = Bubbles(first_term, copy_content = .FALSE., k = first_term%k)

        ! evaluate two of three terms in the gradients: 
        !      Y_lm x/r * d/dr f_lm + S_lm  d/dx 1/r^l
        !    = x/r 1/r ( r d/dr f_lm - l f_lm ) Y_lm
        ! (NOTE: in the loop the x/r is not included and the general result
        !        is stored to temp)
        ! (NOTE2: the 1/r is included as the k = -1 factor of the bubbles)
        do i = 1, first_term%get_nbub()
            rpow = first_term%rpow(i, -1)
            forall (l = 0 : first_term%get_lmax())
                forall (n = 1 : 2*l+1)
                    temp%bf(i)%p(:, l*l+n) = &
                        +(-l+first_term%k) * second_term%bf(i)%p(:, l*l+n)  * rpow(:) &
                        + first_term%bf(i)%p(:, l*l+n) 
                end forall
            end forall
            deallocate(rpow)
        end do
        ! multiply x_unitary = x/r with temp = 1/r ( r d/dr f_lm - l f_lm ) Y_lm
        derivative_x = x_unitary * temp
        derivative_y = y_unitary * temp
        derivative_z = z_unitary * temp
    
        ! destroy the temporary bubbles
        call temp%destroy()
        call x_unitary%destroy()
        call y_unitary%destroy()
        call z_unitary%destroy()

        ! evaluate the final term of the gradients
        !     f_lm 1/r^l d/dx S_lm = f_lm 1/r (g_xlm * Y_l-1,m-1 + h_xlm * Y_l-1,m+1)
        !     f_lm 1/r^l d/dy S_lm = f_lm 1/r (g_ylm * Y_l-1,-m-1 + h_ylm * Y_l-1,-m+1)
        !     f_lm 1/r^l d/dz S_lm = f_lm 1/r (c_zlm * Y_l-1,m)
        ! This is done using the pre-evaluated factors (g_xlm, h_xlm, g_ylm, h_ylm, c_zlm) stored in 
        ! self%gradient_factors_x, self%gradient_factors_y, and self%gradient_factors_z.
        ! The evaluation is done by 'BubblesEvaluator_init_gradient_factors'
        ! (NOTE: also this part contains extra 1/r, which is already included in the result bubbles)
        do i = 1, first_term%get_nbub()
            rpow = first_term%rpow(i, -1)
            ! NOTE: we can start from 1 because the gradient of S_00 = 1 is 0 
            do l = 1, first_term%get_lmax()
                do m = -l, l
                    if (abs(m-1) <= l-1) then
                        derivative_x%bf(i)%p(:, idx(l-1, m-1)) = derivative_x%bf(i)%p(:, idx(l-1, m-1)) &
                            + self%gradient_factors_x(idx(l, m), 1) * second_term%bf(i)%p(:, idx(l, m)) * rpow(:)

                        derivative_y%bf(i)%p(:, idx(l-1, -m+1)) = derivative_y%bf(i)%p(:, idx(l-1, -m+1)) &
                            + self%gradient_factors_y(idx(l, m), 2) * second_term%bf(i)%p(:, idx(l, m))  * rpow(:)
                    end if
                    if (abs(m+1) <= l-1) then
                        derivative_x%bf(i)%p(:, idx(l-1, m+1)) = derivative_x%bf(i)%p(:, idx(l-1, m+1)) &
                            + self%gradient_factors_x(idx(l, m), 2) * second_term%bf(i)%p(:, idx(l, m))  * rpow(:)
                        
                        derivative_y%bf(i)%p(:, idx(l-1, -m-1)) = derivative_y%bf(i)%p(:, idx(l-1, -m-1)) &
                            + self%gradient_factors_y(idx(l, m), 1) * second_term%bf(i)%p(:, idx(l, m))  * rpow(:)
                    end if
                    if (abs(m) /= l) then
                        derivative_z%bf(i)%p(:, idx(l-1, m)) = derivative_z%bf(i)%p(:, idx(l-1, m)) &
                            + self%gradient_factors_z(idx(l, m)) * second_term%bf(i)%p(:, idx(l, m))  * rpow(:)
                    end if
                end do
            end do
            derivative_x%bf(i)%p(1, 1:) = 0.0d0
            derivative_y%bf(i)%p(1, 1:) = 0.0d0
            derivative_z%bf(i)%p(1, 1:) = 0.0d0
            
            deallocate(rpow)
        end do
        !call derivative_x%extrapolate_origo(6, lmax = 0)
        !call derivative_y%extrapolate_origo(6, lmax = 0)
        !call derivative_z%extrapolate_origo(6, lmax = 0)
    end subroutine

    !> Evaluate the vector spherical harmonics (VSH) at unitary coordinates (|a| = 1.0) relative to the bubble 
    !! centers.
    function BubblesEvaluator_evaluate_gradients_from_VSH(self, first_term, second_term, unitary_coordinates) result(results)
        class(BubblesEvaluator), intent(inout) :: self
        !> The first term of VSH containing the derivative of the radial part of the input bubbles
        type(Bubbles),           intent(in)    :: first_term
        !> The second term of VSH containing the radial part of the input bubbles divided by r
        type(Bubbles),           intent(in)    :: second_term   
        !> The unitary coordinates relative to the bubble centers where the gradients are evaluated 
        real(REAL64),            intent(in)    :: unitary_coordinates(:, :)
        type(Interpolator1D)                   :: interpolator_
        integer                                :: i, j, n, length, offset, lmax
        real(REAL64), pointer                  :: coord(:) 
        real(REAL64), pointer                  :: input_bubbles_values(:, :)
        !> result arrays for derivatives
        type(REAL64_2D), allocatable           :: results(:)
        real(REAL64),    allocatable           :: rpow(:), spherical_harmonics_points(:, :), &
                                                  spherical_harmonics_gradients(:, :, :)
        type(RealSphericalHarmonics)           :: harmonics

        ! allocate the result array (for all the bubbles)
        allocate(results(first_term%get_nbub()))
        lmax = first_term%get_lmax()
        
        !> init the SphericalHarmonics with derivative evaluation on.
        harmonics = RealSphericalHarmonics(lmax, normalization = 1, init_derivatives = .TRUE.)
        allocate(spherical_harmonics_points(size(unitary_coordinates, 2), (lmax+1)**2))
        allocate(spherical_harmonics_gradients(size(unitary_coordinates, 2), (lmax+1)**2, 3))

        ! evaluate spherical harmonics and their gradients for all the points defined previously
        spherical_harmonics_points(:, :) = &
            harmonics%eval(unitary_coordinates)
        spherical_harmonics_gradients(:, :, :) = &
            harmonics%evaluate_gradients(unitary_coordinates)
        call harmonics%destroy()
        

        do i = 1, first_term%get_nbub()
            rpow = first_term%rpow(i, -1)
            allocate(results(i)%p(first_term%gr(i)%p%get_shape() * size(unitary_coordinates, 2), 3), &
                     source = 0.0d0)
            offset = 1
            do j = 1, size(unitary_coordinates, 2)
                do n = 1, (lmax + 1)**2
                    results(i)%p(offset:offset+first_term%gr(i)%p%get_shape()-1, X_) = &
                        results(i)%p(offset:offset+first_term%gr(i)%p%get_shape()-1, X_) &
                        +   first_term%bf(i)%p(:, n) &
                          * unitary_coordinates(X_, j) &
                          * spherical_harmonics_points(j, n) &
                        +   second_term%bf(i)%p(:, n) * rpow(:) & 
                          * spherical_harmonics_gradients(j, n, X_)
                    results(i)%p(offset:offset+first_term%gr(i)%p%get_shape()-1, Y_) = &
                        results(i)%p(offset:offset+first_term%gr(i)%p%get_shape()-1, Y_) &
                        +   first_term%bf(i)%p(:, n) &
                          * unitary_coordinates(Y_, j) &
                          * spherical_harmonics_points(j, n) &
                        +   second_term%bf(i)%p(:, n) * rpow(:) &
                          * spherical_harmonics_gradients(j, n, Y_)
                    results(i)%p(offset:offset+first_term%gr(i)%p%get_shape()-1, Z_) = &
                        results(i)%p(offset:offset+first_term%gr(i)%p%get_shape()-1, Z_) &
                        +   first_term%bf(i)%p(:, n) &
                          * unitary_coordinates(Z_, j) &
                          * spherical_harmonics_points(j, n) &
                        +   second_term%bf(i)%p(:, n) * rpow(:) & 
                          * spherical_harmonics_gradients(j, n, Z_)
                end do
                offset = offset + first_term%gr(i)%p%get_shape()
            end do
            print *, "final offset", offset
            deallocate(rpow)
        end do
        deallocate(spherical_harmonics_points, spherical_harmonics_gradients)

    end function
    
    !> Evaluate the contracted gradients of bubbles from vector spherical harmonics (VSH)
    !! at unitary coordinates (|a| = 1.0) relative to the bubble 
    !! centers.
    function BubblesEvaluator_evaluate_contracted_gradients_from_VSH( &
                 self, first_term, second_term, unitary_coordinates) result(results)
        class(BubblesEvaluator), intent(inout) :: self
        !> The first term of VSH containing the derivative of the radial part of the input bubbles
        type(Bubbles),           intent(in)    :: first_term
        !> The second term of VSH containing the radial part of the input bubbles divided by r
        type(Bubbles),           intent(in)    :: second_term   
        !> The unitary coordinates relative to the bubble centers where the gradients are evaluated 
        real(REAL64),            intent(in)    :: unitary_coordinates(:, :)
        !> result arrays for derivatives
        type(REAL64_1D), allocatable           :: results(:)
        !> result arrays for derivatives
        type(REAL64_2D), allocatable           :: gradients(:)
        integer                                :: i
        
        gradients = self%evaluate_gradients_from_VSH(first_term, second_term, unitary_coordinates)
        allocate(results(first_term%get_nbub()))
        do i = 1, first_term%get_nbub()
            allocate(results(i)%p(first_term%gr(i)%p%get_shape() * size(unitary_coordinates, 2)))
            results(i)%p(:) =   gradients(i)%p(:, X_) * gradients(i)%p(:, X_) &
                              + gradients(i)%p(:, Y_) * gradients(i)%p(:, Y_) &
                              + gradients(i)%p(:, Z_) * gradients(i)%p(:, Z_)
            deallocate(gradients(i)%p)
        end do
        deallocate(gradients)
    end function


    !> Evaluate divergence of the Bubbles at points or at grid. 
    !! NOTE: if cuda is enabled this subroutine continues to evaluate in the background
    !! and 'get_results' will synchronize the cuda and cpus
    subroutine BubblesEvaluator_evaluate_divergence_points(self, bubbls_x, bubbls_y, bubbls_z, output_points)
        class(BubblesEvaluator), intent(inout) :: self
        type(Bubbles),           intent(in)    :: bubbls_x, bubbls_y, bubbls_z
        type(Points),            intent(inout) :: output_points
        integer                                :: ibub
#ifdef HAVE_CUDA
        call self%set_output_points(output_points)
        
        call self%set_input_bubbles(bubbls_x)
        call self%cuda_evaluate_points_derivative(X_, output_points, set_to_zero = .TRUE., download = .FALSE.)
        call CUDASync_all()
        call self%set_input_bubbles(bubbls_y)
        call self%cuda_evaluate_points_derivative(Y_, output_points, set_to_zero = .FALSE., download = .FALSE.)
        call CUDASync_all()
        call self%set_input_bubbles(bubbls_z)
        call self%cuda_evaluate_points_derivative(Z_, output_points, set_to_zero = .TRUE., download = .TRUE.)
        call CUDASync_all()
#else 
        ! NOTE: the following procedure produces invalid values near the cusp
        
        output_points%values(:) = bubbls_x%eval(output_points%point_coordinates%coordinates,[1,0,0])
        output_points%values(:) =   output_points%values(:)  &
                                  + bubbls_y%eval(output_points%point_coordinates%coordinates,[0,1,0])
        output_points%values(:) =   output_points%values(:)  &
                                  + bubbls_z%eval(output_points%point_coordinates%coordinates,[0,0,1])
#endif
    end subroutine

    

    !> Destroy the BubblesEvaluator. 
    subroutine BubblesEvaluator_destroy(self)
        class(BubblesEvaluator), intent(inout) :: self
        
        
#ifdef HAVE_CUDA
        call self%cuda_destroy()
#endif
        if (allocated(self%gradient_factors_x)) deallocate(self%gradient_factors_x)
        if (allocated(self%gradient_factors_y)) deallocate(self%gradient_factors_y)
        if (allocated(self%gradient_factors_z)) deallocate(self%gradient_factors_z)
        
        nullify(self%grid)
        call self%destroy_input_bubbles()
        call Evaluator_destroy(self)
    end subroutine

end module
