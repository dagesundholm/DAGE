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
module Helmholtz3D_class

    use Globals_m
    use Function3D_class
    use Grid_class
    use Bubbles_class
    use potential_class

    use GaussQuad_class
    use Coulomb3D_class
    use Laplacian3D_class

    use Harmonic_class
    use Evaluators_class
    use ParallelInfo_class

! debugging
    use MemoryLeakChecker_m

#ifdef HAVE_CUDA
    use cuda_m
#endif
#ifdef HAVE_OMP
    use omp_lib
#endif

    implicit none

    private

    public    :: Helmholtz3D, Helmholtz3DArray, FirstModSphBessel, FirstModSphBesselCollection, &
                 SecondModSphBessel, SecondModSphBesselCollection, ModSphBesselCollection, &
                 factorial_real, FirstModSphBesselCubeCollection
    public    :: assignment(=)

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                   Helmholtz3D definition                                %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    !> Does the helmholtz thingie
    type, extends(Operator3D) :: Helmholtz3D
        !> Quadrature used for integration. NOTE: this must be the
        !! same as used in initialization of 'coulomb_operator'
        type(GaussQuad)                  :: quadrature
        !> Modified coulomb operator to operate the cube
        class(Coulomb3D), pointer        :: coulomb_operator
        !> Current energy of the optimized system
        real(REAL64)                     :: energy
        !> The maximum angular momentum of the result bubbles.
        !! The resulting extra bubbles are injected to the cube
        integer                          :: lmax = 4
    contains
        procedure :: operate_on          => Helmholtz3D_operate
        procedure :: transform_cube      => Helmholtz3D_cube
        procedure :: transform_bubbles   => Helmholtz3D_bubbles
        procedure :: destroy             => Helmholtz3D_destroy
        procedure :: set_energy          => Helmholtz3D_set_energy
        procedure :: transform_bubbles_sub => Helmholtz3D_transform_bubbles_sub
#ifdef HAVE_CUDA
        procedure           :: cuda_init_child_operator => Helmholtz3D_cuda_init_child_operator
        procedure, public   :: cuda_prepare   => Helmholtz3D_cuda_prepare
        procedure, public   :: cuda_unprepare => Helmholtz3D_cuda_unprepare
#endif
 
    end type

    !> Constructor for Helmholtz3D -operator
    interface Helmholtz3D
        module procedure :: Helmholtz3D_init
    end interface



!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                   Helmholtz3D array definition                          %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!   This object is a dummy in order to be able to define arrays of         %
!   Helmholtz3D objects                                                    %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    type Helmholtz3DArray
        class(Helmholtz3D), allocatable :: op
    end type


    interface assignment(=)
        module procedure :: Helmholtz3D_assign
        module procedure :: Helmholtz3DArray_assign
    end interface

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                   FirstModSphBessel definition                          %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%   Type used to evaluate First Modified Spherical Bessel functions       %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    type           ::    FirstModSphBessel
        integer(kind=8),allocatable    ::    sinh_coeffs(:)
        integer(kind=8),allocatable    ::    cosh_coeffs(:)
        ! The value at x = 0. This is needed, if the x is smaller than the
        ! threshold value at which the number is evaluated. For l = 0 the value
        ! is 1, for others, it is 0
        real(REAL64)           ::    zero_value
    contains
        procedure  :: eval        => FirstModSphBessel_evaluate
        procedure  :: destroy     => FirstModSphBessel_destroy
    end type

    !> Constructor for SecondModSphBessel
    interface FirstModSphBessel
        module procedure :: FirstModSphBessel_init
    end interface


!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                   ModSphBesselCollection definition                     %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!% Interface for Generic Modified Spherical Bessel Function Colleection    %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    type, abstract           ::    ModSphBesselCollection
        integer                   :: lmax
        integer                   :: lmin
        ! The value at x = 0. This is needed, if the x is smaller than the
        ! threshold value at which the number is evaluated. For l = 0 the value
        ! is 1, for others, it is 0
        real(REAL64), allocatable :: zero_values(:)
        real(REAL64), allocatable :: small_thresholds(:)            
        real(REAL64), public      :: scaling_factor = 1
        
    contains
        procedure(eval),    public, deferred  :: eval  
        procedure(destroy), public, deferred  :: destroy
    end type

    abstract interface
        function eval(self, parameter_values) result(res)
            import REAL64, ModSphBesselCollection
            class(ModSphBesselCollection), intent(in)      :: self
            real(REAL64),                  intent(in)      :: parameter_values(:)
            real(REAL64)                                   :: res(size(parameter_values), 0:self%lmax)
        end function

        pure subroutine destroy(self)
            import ModSphBesselCollection
            class(ModSphBesselCollection), intent(inout)  :: self
        end subroutine

    end interface

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                   FirstModSphBesselCollection definition                %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%   Type used to evaluate multiple First Modified Spherical Bessel        %
!%   functions simultaneously at multiple points. This method used         % 
!%   recursion unlike the FirstModSphBessel function. This choice is       %
!%   made to gain more numerical stability.                                %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    type, extends(ModSphBesselCollection)      ::    FirstModSphBesselCollection
    contains
        procedure          :: eval           => FirstModSphBesselCollection_evaluate
        procedure, private :: evaluate_small => FirstModSphBesselCollection_evaluate_small
        procedure          :: destroy        => FirstModSphBesselCollection_destroy
    end type

    !> Constructor for FirstModSphBesselCollection
    interface FirstModSphBesselCollection
        module procedure :: FirstModSphBesselCollection_init
    end interface

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                   FirstModSphBesselCubeCollection definition            %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%   Type used to evaluate multiple First Modified Spherical Bessel        %
!%   functions simultaneously at cube.                                     %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    type, extends(FirstModSphBesselCollection)      ::    FirstModSphBesselCubeCollection
#ifdef HAVE_CUDA
        type(C_PTR)        :: cuda_interface
#endif
    contains
        procedure          :: eval_grid      => FirstModSphBesselCubeCollection_evaluate_grid
        procedure, private :: eval_grid_cpu  => FirstModSphBesselCubeCollection_evaluate_grid_cpu
        procedure          :: destroy        => FirstModSphBesselCubeCollection_destroy
    end type

    !> Constructor for SecondModSphBessel
    interface FirstModSphBesselCubeCollection
        module procedure :: FirstModSphBesselCubeCollection_init
    end interface

#ifdef HAVE_CUDA

    interface 
         type(C_PTR) function FirstModifiedSphericalCubeBessels_init_cuda( &
                 lmin, lmax, shape, streamContainer) bind(C)
             use ISO_C_BINDING
             integer(C_INT), value :: lmin
             integer(C_INT), value :: lmax
             integer(C_INT)        :: shape(3)
             type(C_PTR),    value :: streamContainer
         end function
     end interface

    interface 
         pure subroutine FirstModifiedSphericalCubeBessels_destroy_cuda( &
                 bessels) bind(C)
             use ISO_C_BINDING
             type(C_PTR), value    :: bessels
         end subroutine
     end interface



    interface 
         subroutine FirstModifiedSphericalCubeBessels_set_kappa_cuda( &
                 bessels, kappa) bind(C)
             use ISO_C_BINDING
             type(C_PTR), value    :: bessels
             real(C_DOUBLE), value :: kappa
         end subroutine
     end interface


     interface 
         subroutine FirstModifiedSphericalCubeBessels_evaluate_cuda( &
                 bessels, grid, center) bind(C)
             use ISO_C_BINDING
             type(C_PTR), value    :: bessels
             type(C_PTR), value    :: grid
             real(C_DOUBLE)        :: center(3)
         end subroutine
     end interface

    interface 
         subroutine FirstModifiedSphericalCubeBessels_download_cuda( &
                 bessels, host_results, host_results_shape) bind(C)
             use ISO_C_BINDING
             type(C_PTR), value    :: bessels
             real(C_DOUBLE)        :: host_results(*)
             integer(C_INT)        :: host_results_shape(4)
         end subroutine
     end interface

    interface 
         subroutine firstmodifiedsphericalcubebessels_register_result_array_cuda( &
                 bessels, host_results, host_results_shape) bind(C)
             use ISO_C_BINDING
             type(C_PTR), value    :: bessels
             real(C_DOUBLE)        :: host_results(*)
             integer(C_INT)        :: host_results_shape(4)
         end subroutine
     end interface

    interface 
         subroutine firstmodifiedsphericalcubebessels_unregister_result_array_cuda( &
                 bessels, host_results) bind(C)
             use ISO_C_BINDING
             type(C_PTR), value    :: bessels
             real(C_DOUBLE)        :: host_results(*)
         end subroutine
     end interface

#endif

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                   SecondModSphBesselCollection definition               %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%   Type used to evaluate multiple Second Modified Spherical Bessel       %
!%   functions simultaneously at multiple points. This method used         % 
!%   recursion unlike the SecondModSphBessel function. This choice is      %
!%   made to gain more numerical stability.                                %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    type, extends(ModSphBesselCollection)     ::    SecondModSphBesselCollection
    contains
        procedure          :: eval           => SecondModSphBesselCollection_evaluate
        procedure, private :: evaluate_small => SecondModSphBesselCollection_evaluate_small
        procedure          :: destroy        => SecondModSphBesselCollection_destroy
        procedure          :: eval_grid      => SecondModSphBesselCollection_evaluate_grid
    end type

    !> Constructor for SecondModSphBessel
    interface SecondModSphBesselCollection
        module procedure :: SecondModSphBesselCollection_init
    end interface


! #ifdef HAVE_CUDA
!     interface 
!         subroutine SecondModSphBesselCollection_cuda_evaluate_grid( &
!                 grid, lmax, scaling_factor, kappa, center, cubes) bind(C)
!             use ISO_C_BINDING
!             type(C_PTR), value    :: grid
!             integer(C_INT), value :: lmax
!             real(C_DOUBLE), value :: scaling_factor
!             real(C_DOUBLE), value :: kappa
!             real(C_DOUBLE)        :: center(3)
!             real(C_DOUBLE)        :: cubes(*)
!         end subroutine
!     end interface
! #endif

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                   SecondModSphBessel definition                         %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%  Type used to evaluate Second Modified Spherical Bessel functions       %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    type           ::    SecondModSphBessel
        integer, allocatable   ::    coeffs(:)
    
        contains
        procedure  :: eval        => SecondModSphBessel_evaluate
        procedure  :: destroy     => SecondModSphBessel_destroy
    end type

    !> Constructor for SecondModSphBessel
    interface SecondModSphBessel
        module procedure :: SecondModSphBessel_init
    end interface

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%                   BesselCollection definition                           %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%  Collection of modified spherical bessel functions                      %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    type           ::   BesselCollection
        type(FirstModSphBessel),  allocatable  :: first(:)
        type(SecondModSphBessel), allocatable  :: second(:)

    contains  
        procedure :: destroy     => BesselCollection_destroy
    end type

    !> Constructor for BesselCollection
    interface BesselCollection
        module procedure :: BesselCollection_init
    end interface



contains

    !> Constructor for Helmholtz3D object
    function Helmholtz3D_init(coulomb_operator, energy, quadrature, lmax) result(new)
        type(Helmholtz3D)                     :: new
        class(Coulomb3D), intent(in), target  :: coulomb_operator
        real(REAL64),     intent(in)          :: energy
        !> Gaussian quadrature
        !! If no quadrature is given, then the quadrature of the
        !! coulomb operator is used.
        type(GaussQuad), optional     :: quadrature
        !> The maximum angular momentum of the result bubbles.
        !! The resulting extra bubbles are injected to the cube
        integer, optional, intent(in) :: lmax

        
        
        if(present(quadrature)) then
            new%quadrature = quadrature
        else
            new%quadrature =  coulomb_operator%gaussian_quadrature
        endif

        if (present(lmax)) then
            new%lmax = lmax
        end if
        
        new%coulomb_operator => coulomb_operator
        
        call new%set_energy(energy)
        new%result_parallelization_info => coulomb_operator%result_parallelization_info
        new%suboperator = .FALSE.
#ifdef HAVE_CUDA
        new%cuda_inited = .FALSE.
#endif
    end function

#ifdef HAVE_CUDA
    subroutine Helmholtz3D_cuda_prepare(self)
        class(Helmholtz3D), intent(inout)          :: self  
    end subroutine

    subroutine Helmholtz3D_cuda_unprepare(self)
        class(Helmholtz3D), intent(inout)          :: self  
    end subroutine
#endif

    subroutine Helmholtz3D_set_energy(self, energy)
        class(Helmholtz3D), intent(inout)          :: self  
        real(REAL64),       intent(in)             :: energy

        ! Gaussian quadrature weights and tpoints
        real(REAL64), allocatable                  :: weights(:), tpoints(:)
        integer                                    :: i
        
        self%energy = energy

        ! get quadrature weights, and tpoints
        weights = self%quadrature%get_weights()
        tpoints = self%quadrature%get_tpoints()   

        ! multiply quadrature weights with e^[(2 E) / (4 t_i^2)]
        ! this makes the Coulomb3D operator to be a Helmholtz operator
        ! if the energy is below limit -0.05, the use a bare coulomb operator to
        ! approximate the helmholtz operator
        if (energy < -0.005d0) then
            forall (i=1 : size(tpoints))
                weights(i)=weights(i)*exp(2d0*energy/(4d0*tpoints(i)**2))
            end forall
            
            ! put the weights to the operator
            call self%coulomb_operator%set_transformation_weights(TWOOVERSQRTPI * weights)
        else
            call self%coulomb_operator%set_transformation_weights(TWOOVERSQRTPI * weights)
        end if



        deallocate(weights)
        deallocate(tpoints)
    end subroutine  

    subroutine Helmholtz3D_operate(self, func, new, cell_limits, only_cube)
        class(Helmholtz3D)              :: self 
        !> The input function. Note this should be a multiple of function and
        !! corresponding potential times two
        class(Function3D),  intent(in)  :: func  
        !> Limits in the area in which the operator is applied. NOTE: not used in this operator
        integer, intent(in), optional   :: cell_limits(2, 3)
        !> if only cube is taken into account.
        logical, intent(in), optional   :: only_cube

        ! Result function, is of type Function3D
        class(Function3D), allocatable, intent(inout), target  :: new
        real(REAL64), allocatable       :: centers(:, :)
        logical                         :: operate_bubbles

        if (present (only_cube)) then
            operate_bubbles = .not. only_cube
        else
            operate_bubbles = .TRUE.
        end if

        ! Apply the modified coulomb operator
        call self%coulomb_operator%operate_on(func, new, only_cube = .TRUE.)

        ! Apply the bubbles
        if (operate_bubbles) then
            
            new%bubbles = Bubbles(func%bubbles)

            ! then operate
            call self%transform_bubbles_sub(func%bubbles, new%bubbles)
            !call new%inject_extra_bubbles(self%lmax)
        end if
        call self%set_energy(0.0d0)

    end subroutine


    !> This function is empty and is here only to fulfill 
    !! the requirements of a Operator3D interface
    function Helmholtz3D_cube(self, cubein) result(cubeout)
       class(Helmholtz3D), intent(in) :: self
       real(REAL64), intent(in)      :: cubein(:,:,:)
       real(REAL64),allocatable      :: cubeout(:,:,:)
     
       cubeout=cubein
       return
    end function

#ifdef HAVE_CUDA
    subroutine Helmholtz3D_cuda_init_child_operator(self)
        class(Helmholtz3D),      intent(inout)  :: self

        self%cuda_inited = self%coulomb_operator%cuda_inited
    end subroutine
#endif


    !> Convolves the helmholtz kernel with the bubbles part
    function Helmholtz3D_bubbles(self, bubsin) result(new)
        class(Helmholtz3D), intent(in) :: self
        !> Input bubbles. In order for this to work, this must be the bubbles
        !! of a potential 
        type(Bubbles), intent(in)      :: bubsin
        ! Result bubbles
        type(Bubbles)                  :: new

        new   = Bubbles(bubsin, copy_content = .FALSE., k = 0)
        call self%transform_bubbles_sub(bubsin, new)
    end function
    

    !> Convolves the helmholtz kernel with the bubbles part
    subroutine Helmholtz3D_transform_bubbles_sub(self, bubsin, new)
        class(Helmholtz3D), intent(in) :: self
        !> Input bubbles. In order for this to work, this must be the bubbles
        !! of a potential 
        type(Bubbles), intent(in)      :: bubsin
        ! Result bubbles
        type(Bubbles), intent(inout)   :: new

        !  kappa -parameter of Helmholtz kernel: \sqrt(-2.0 * E)
        real(REAL64)                   :: kappa

        !> Collection of Modified Spherical Bessel functions (First and Second kind)
        type(BesselCollection)            :: bessels
        type(FirstModSphBesselCollection) :: first_bessels
        type(SecondModSphBesselCollection) :: second_bessels
 

        integer                       :: i, j, k, lmax
        ! for loop clarity
        integer                       :: l,m, ibub, id, first_l, domain(2), &
                                         last_l, l_m(2), first_bubble, last_bubble, &
                                         first_idx, last_idx
        ! r and r'
        real(REAL64), pointer         :: r(:)
        
        type(Grid1D), pointer         :: grid
        real(REAL64), pointer         :: vals(:,:), vals0(:,:)
        ! integrator stuff
        real(REAL64), allocatable     :: rpow(:), temp1(:), temp2(:)
        real(REAL64), allocatable     :: first_bessel_values(:, :), second_bessel_values(:, :)

        type(InOutIntegrator1D)       :: integrator2


        new   = 0.0d0
        lmax  = bubsin%get_lmax()
        kappa = sqrt(-2.0d0*self%energy)

        ! initialize the bessel function coefficients
        !bessels = BesselCollection(lmax+1) 
        first_bessels = FirstModSphBesselCollection(lmax)
        second_bessels = SecondModSphBesselCollection(lmax)

        domain = self%result_parallelization_info%get_bubble_domain_indices &
                     (bubsin%get_nbub(), bubsin%get_lmax())
        first_bubble = (domain(1)-1) / (bubsin%get_lmax() + 1)**2 + 1
        last_bubble = (domain(2)-1) / (bubsin%get_lmax() + 1)**2 + 1 
 
        ! Go through all bubbles
        do ibub=1, bubsin%get_nbub() !first_bubble, last_bubble
            vals  => new%get_f(ibub)
            vals0 => bubsin%get_f(ibub)
            grid  => new%get_grid(ibub)
            integrator2=InOutIntegrator1D(grid)

            

            ! get coordinates of the bubble grid
            r => grid%get_coord()
        
            allocate(first_bessel_values (size(r), 0 : lmax))
            allocate(second_bessel_values(size(r), 0 : lmax))
            allocate(temp1(size(r)))
            allocate(temp2(size(r)))
            first_bessel_values = first_bessels%eval(kappa * r)
            second_bessel_values = second_bessels%eval(kappa * r)

            ! evaluate r^2 (and add the k value to that)
            rpow = bubsin%rpow(ibub, 2+bubsin%get_k())

            ! get the first idx, i.e., the last l, m pair calculated by this processor
            first_idx = 1
            !if (ibub == first_bubble) then
            !    first_idx = domain(1) - (ibub-1) * (bubsin%get_lmax() + 1)**2
            !end if
            l_m = lm(first_idx)
            first_l = l_m(1)
            
            ! get the last idx, i.e., the last l, m pair calculated by this processor
            last_idx = (bubsin%get_lmax() + 1)**2
            !if (ibub == last_bubble) then
            !    last_idx = domain(2) - (ibub-1) * (bubsin%get_lmax() + 1)**2
            !end if
            l_m = lm(last_idx)
            last_l = l_m(1)

#ifdef HAVE_OMP
            !$OMP PARALLEL  DO &
            !$OMP& PRIVATE(l, m, id, temp1, temp2) 
#endif  
            do l=first_l, last_l
                id = l * l + 1
                ! values of modified spherical bessel functions
                do m=-l,l
                    if (id > last_idx) then
                       exit
                    end if
                    temp1(:) = rpow * first_bessel_values(:, l) * vals0(:, id)
                    !call extrapolate_first_nlip7(6, temp1)
                    temp2(:) = rpow * second_bessel_values(:, l) * vals0(:, id)
                    if (l == 0) call extrapolate_first_nlip7(3, bubsin%gr(ibub)%p%get_grid_type(), temp2)
                    if (l /= 0) temp2(1) = 0.0d0
                    vals(:, id) =   &
                              (second_bessel_values(:, l)  *  &
                               integrator2%outwards( temp1 ) + &
                               first_bessel_values(:, l)   *  &
                               integrator2%inwards( temp2 ))
                    

                    id = id +1
                end do
                !if (id == last_idx) then
                !   exit
                !end if
            end do
#ifdef HAVE_OMP
            !$OMP END PARALLEL DO
#endif
            vals(:, :) = vals(:, :) * 8.0d0 * kappa
            !do l=first_l, last_l
            !    id = l * l + 1
            !    do m=-l,l
            !        if (abs(sum(vals(:, id))) > 1.0d2 ) print *, "sum", l, m, ":", sum(vals(:, id))
            !        id = id +1
            !    end do
            !end do
            deallocate(first_bessel_values)
            deallocate(second_bessel_values)
            deallocate(rpow, temp1, temp2)
            call integrator2%destroy()
            nullify(vals)
            nullify(vals0)
            nullify(r)
            nullify(grid)
        end do
        call first_bessels%destroy()
        call second_bessels%destroy()  
        return
    end subroutine

    subroutine Helmholtz3D_assign(operator1, operator2)
        class(Helmholtz3D), intent(inout), allocatable :: operator1
        class(Helmholtz3D), intent(in)                 :: operator2

        allocate(operator1, source = operator2)
    end subroutine

    subroutine Helmholtz3DArray_assign(operator1, operator2)
        type(Helmholtz3DArray), intent(inout)          :: operator1
        class(Helmholtz3D), intent(in)                 :: operator2

        allocate(operator1%op, source = operator2)
    end subroutine

    !> destructor for Helmholtz3D object
    subroutine Helmholtz3D_destroy(self)
        class(Helmholtz3D), intent(inout)  :: self
        call self%quadrature%destroy()
        nullify(self%coulomb_operator)
    end subroutine 


!--------------------------------------------------------------------!
!        Modified Spherical Bessel Functions (1st and 2nd)           !
!--------------------------------------------------------------------!

    !> this function returns the coefficients of a 
    !! modified spherical Bessel function of the first kind.
    !! Recursion relation is
    !! \f[
    !!     g_n = g_{n-2} - \frac{2n-1}{r} g_{n-1}
    !! \f]
    !! Modified function is
    !! \f[
    !!     \hat{I}_{n+½}(r) = g_n(r) sinh(r) + g_{-(n+1)}(r) cosh(r) 
    !! \f]
    pure recursive function first_kind_coeff(l,lmax) result(res)
        integer, intent(in)     :: l
        integer, intent(in)     :: lmax
        integer                 :: res(lmax +1)
        integer                 :: i
        integer                 :: tmp(lmax +1)

        res=0

 
        if (l>=0) then
            if (l==0) then
                res(1)=1
            else if (l==1) then
                res(2)=-1
            
            else if (l>=2) then
                res=first_kind_coeff(l-2,lmax)
                tmp=first_kind_coeff(l-1,lmax)
                do i=1,lmax
                    res(i+1)=res(i+1)-(2*l-1)*tmp(i)
                end do
            end if
        else
            res=first_kind_coeff(l+2,lmax)
            tmp=first_kind_coeff(l+1,lmax)
            do i=1,lmax
                res(i+1)=res(i+1)+(2*l+3)*tmp(i)
            end do
        end if

    end function

    !> this function returns the coefficients of a 
    !! modified spherical Bessel function of the second kind.
    !! Recursion relation is:
    !! \f[
    !!     g_n= g_{n-2} + \frac{2n-1}{r} g_{n-1}
    !! \f]
    !! Modified function is
    !! \f[
    !!     \hat{K}_{n+½}(r) = g_n(r) \frac{2}{\pi} e^{-r}
    !! \f]

    pure recursive function second_kind_coeff(l,lmax) result(res)
        integer, intent(in)     :: l
        integer, intent(in)     :: lmax
        integer                 :: res(lmax+1)
        integer                 :: i,j
        integer                 :: tmp(lmax+1)

        if(l>=0) then
            j=l
        else
            j=-l
        endif

        res=0
        if (j==0) then
            res(1)=1
        else if (j==1) then
            res(1)=1
            res(2)=1
        else if (j>=2) then
            res=second_kind_coeff(l-2,lmax)
            tmp=second_kind_coeff(l-1,lmax)
            do i=1,lmax
                res(i+1)=res(i+1)+(2*l-1)*tmp(i)
            end do

        endif

    end function

    pure function SecondModSphBessel_evaluate(self, values) result(res)
        class(SecondModSphBessel), intent(in) :: self
        real(REAL64),              intent(in) :: values(:)
        real(REAL64)                          :: res(size(values))
        integer                               :: i,j

        res=0e0
        
        do i=1,size(values)
            if(values(i)>1e-10) then
                do j=1, size(self%coeffs)
                    res(i)=res(i)+real(self%coeffs(j))*values(i)**(-j)
                end do
                res(i)=res(i)*exp(-values(i)) * PI/(2.0d0)
            end if
        end do

    end function 

    pure function SecondModSphBessel_init(l) result(new)
        integer, intent(in)               :: l
        type(SecondModSphBessel)          :: new
        allocate(new%coeffs(l+2))
        new%coeffs = second_kind_coeff(l,l+1)
    end function

    pure function FirstModSphBessel_init(l) result(new)
        integer, intent(in)     :: l
        type(FirstModSphBessel) :: new
        
        allocate(new%sinh_coeffs(l+2))
        allocate(new%cosh_coeffs(l+2))
        new%sinh_coeffs = first_kind_coeff(l,l+1)
        new%cosh_coeffs = first_kind_coeff(-l-1,l+1)

        if (l == 0) then
            new%zero_value = 1.0d0
        else
            new%zero_value = 0.0d0
        end if
    end function

    pure function FirstModSphBessel_evaluate(self, values) result(res)
        class(FirstModSphBessel), intent(in) :: self
        real(REAL64),             intent(in) :: values(:)
        real(REAL64)                         :: res(size(values)), cosh_value, sinh_value
        integer                              :: i,j, THRESHOLD

        res= 0.0d0
        do i=1,size(values)
            if(values(i)>1e-10) then
                cosh_value = 0.0d0
                sinh_value = 0.0d0
                ! notice that size(self%sinh_coeffs) == size(self%cosh_coeffs)
                do j=1,size(self%sinh_coeffs)
                    sinh_value = sinh_value + dble(self%sinh_coeffs(j))*values(i)**(-j)
                    cosh_value = cosh_value + dble(self%cosh_coeffs(j))*values(i)**(-j) 
                end do
                res(i) = cosh_value * cosh(values(i)) + sinh_value * sinh(values(i))
            else
                res(i) = self%zero_value
            end if
                      
        end do
    end function

    pure subroutine FirstModSphBessel_destroy(self)
        class(FirstModSphBessel), intent(inout)  :: self
        deallocate(self%sinh_coeffs)
        deallocate(self%cosh_coeffs)
    end subroutine

    pure subroutine SecondModSphBessel_destroy(self)
        class(SecondModSphBessel), intent(inout)  :: self
        deallocate(self%coeffs)
    end subroutine

!--------------------------------------------------------------------!
!  First Modified Spherical Bessel Function Collection               !
!   (FirstModSphBesselCollection) - functions                        !
!--------------------------------------------------------------------!

    pure function FirstModSphBesselCollection_init(lmax, scaling_factor) result(new)
        integer, intent(in)                :: lmax
        real(REAL64), intent(in), optional :: scaling_factor
        type(FirstModSphBesselCollection)  :: new
        
        new%lmax = lmax
        if (present(scaling_factor)) then
            new%scaling_factor = scaling_factor
        else
            new%scaling_factor = 1.0d0
        end if
        allocate(new%zero_values(0 : lmax), source = 0.0d0)
        allocate(new%small_thresholds(0 : lmax), source = 0.0d0)
        !new%small_thresholds = 10.0d0
        !if (lmax >= 0) new%small_thresholds(0) = 0.001d0
        !if (lmax >= 1) new%small_thresholds(1) = 0.003d0
        !if (lmax >= 2) new%small_thresholds(2) = 0.005d0
        !if (lmax >= 3) new%small_thresholds(3) = 0.015d0
        !if (lmax >= 4) new%small_thresholds(4) = 0.050d0
        !if (lmax >= 5) new%small_thresholds(5) = 0.110d0
        !if (lmax >= 6) new%small_thresholds(6) = 0.280d0
        !if (lmax >= 7) new%small_thresholds(7) = 0.420d0
        !if (lmax >= 8) new%small_thresholds(8) = 1.000d0
        !if (lmax >= 9) new%small_thresholds(9) = 3.000d0
        new%small_thresholds = 0.0d0
        new%zero_values(0) = 1.0d0
    end function

    function FirstModSphBesselCollection_evaluate(self, parameter_values) result(res)
        class(FirstModSphBesselCollection), intent(in) :: self
        real(REAL64),             intent(in)           :: parameter_values(:)
        real(REAL64)                                   :: res(size(parameter_values), 0:self%lmax)
        real(REAL128)                                  :: reslt(size(parameter_values), 0:self%lmax), &
                                                          coeffs(-self%lmax-2:self%lmax+2)
        real(REAL128)                                  :: x, temp
        real(REAL128), parameter                       :: ZERO = epsilon(0.0d0)
        integer      , parameter                       :: RECURSION_LIMIT = 17
        integer                                        :: i,l, recursion_max, npoints

        npoints = size(parameter_values)
        recursion_max = min(self%lmax, RECURSION_LIMIT)
        res= 0.0d0
#ifdef HAVE_OMP
        !$OMP PARALLEL DO IF (npoints > 1000)&
        !$OMP& PRIVATE(x, l) 
#endif 
        do i=1, npoints
            x = parameter_values(i)
            if(abs(x) > self%small_thresholds(self%lmax)) then
                
                !reslt(i, 0) = sinh(x) / x 
                !reslt(i, 1) = (x * cosh(x) - sinh(x)) / x**2 / self%scaling_factor 
                ! upward recursion
                !do l = 2, self%lmax
                !    reslt(i, l) = -(2*l-1) * reslt(i, l-1) / x / self%scaling_factor &
                !                  + reslt(i, l-2) / (self%scaling_factor**2)
                !    print *, "l", l, reslt(i, l)
                !end do
                reslt(i, self%lmax) = self%evaluate_small(x, self%lmax, 300)
                reslt(i, self%lmax-1) = self%evaluate_small(x, self%lmax-1, 300)
                ! downward recursion
                do l = self%lmax - 2, 0, -1
                    reslt(i, l) = (2*l+3) * reslt(i, l+1) / x * self%scaling_factor  &
                                  + reslt(i, l+2) * self%scaling_factor ** 2
                end do
            else
                if (abs(x) < ZERO) then
                    reslt(i, :) = self%zero_values
                else
                    do l = 0, self%lmax
                        reslt(i, l) = self%evaluate_small(x, l, 300)
                    end do
                end if
            end if
                      
        end do
#ifdef HAVE_OMP
        !$OMP END PARALLEL DO
#endif
        res = reslt
    end function



    pure function FirstModSphBesselCollection_evaluate_small(self, z, n, max_number_of_terms) result(reslt)
        class(FirstModSphBesselCollection), intent(in)    :: self
        real(REAL128),                       intent(in)   :: z
        integer,                            intent(in)    :: n
        integer,                            intent(in)    :: max_number_of_terms
        real(REAL128)                                     :: reslt, prefactor, divider, addition, &
                                                             factorial_i, a, b
        integer                                           :: i

        !print *, "e small", z, n
        prefactor = z / (self%scaling_factor)
        prefactor = prefactor**n

        ! the divider is a floating point number to allow larger evaluated n values,
        ! if integer (64 bit) would be used the max n would be 16. By allowing
        ! the divider to have some inaccuracy we can get decent numbers
        ! for larger n values.
        divider = 1.0d0  
        do i = 1, 2*n+1, 2
            divider = divider * i 
        end do

        prefactor = prefactor / divider

        addition =  1.0d0
        reslt = 1.0d0
        divider = 1
        factorial_i = 1.0d0
        a = (0.5d0*z**2)
        b = 1.0d0
        i = 1
        do while (addition > 1d-15 * reslt .and. i < max_number_of_terms)
            ! multiply the factorial
            b = b * a
            factorial_i = factorial_i * i
            divider = divider * (2*n + 2*i + 1)
            addition = b / (factorial_i * divider )
            reslt = reslt + addition
            i = i + 1
        end do
        reslt = prefactor * reslt
    end function

    pure subroutine FirstModSphBesselCollection_destroy(self)
        class(FirstModSphBesselCollection), intent(inout)  :: self
        if (allocated(self%zero_values)) deallocate(self%zero_values)
        if (allocated(self%small_thresholds)) deallocate(self%small_thresholds)
    end subroutine


!--------------------------------------------------------------------!
!  Second Modified Spherical Bessel Function Cube Collection         !
!   (FirstModSphBesselCubeCollection) - functions                       !
!--------------------------------------------------------------------!

    function FirstModSphBesselCubeCollection_init(lmin, lmax, shape) result(new)
        integer, intent(in)                    :: lmin
        integer, intent(in)                    :: lmax
        integer, intent(in)                    :: shape(3)
        type(FirstModSphBesselCubeCollection)  :: new

        new%lmin = lmin
        new%lmax = lmax
#ifdef HAVE_CUDA
        new%cuda_interface = FirstModifiedSphericalCubeBessels_init_cuda(lmin, lmax, shape, stream_container)
#endif
        allocate(new%zero_values(0 : lmax), source = 0.0d0)
        allocate(new%small_thresholds(0 : lmax), source = 0.0d0)
        new%small_thresholds = 0.0d0
        new%zero_values(0) = 1.0d0

    end function

    function FirstModSphBesselCubeCollection_evaluate_grid(self, grid, center, kappa, no_cuda) result(result_cubes)
        class(FirstModSphBesselCubeCollection), intent(in) :: self
        type(Grid3D),                           intent(in) :: grid
        real(REAL64),                           intent(in) :: center(3)
        real(REAL64),    optional,              intent(in) :: kappa
        logical,         optional,              intent(in) :: no_cuda
        real(REAL64)                                       :: result_cubes(grid%axis(X_)%get_shape(), &
                                                                       grid%axis(Y_)%get_shape(), &
                                                                       grid%axis(Z_)%get_shape(), &
                                                                       0 : self%lmax)   
        real(REAL64)                                       :: kappa_value
        logical                                            :: evaluate_cuda

        if (present(kappa)) then
            kappa_value = kappa
        else
            kappa_value = 1.0d0;
        end if

#ifdef HAVE_CUDA
        if (present(no_cuda)) then
            if (no_cuda) then
                evaluate_cuda = .FALSE.
            else
                evaluate_cuda = .TRUE.
            end if 
        else
            evaluate_cuda = .FALSE.
        end if 

        if (evaluate_cuda) then
            ! initialize the cuda grid
            call FirstModifiedSphericalCubeBessels_register_result_array_cuda(self%cuda_interface, &
                        result_cubes, shape(result_cubes))
            call FirstModifiedSphericalCubeBessels_set_kappa_cuda(self%cuda_interface, kappa_value)
            call FirstModifiedSphericalCubeBessels_evaluate_cuda(&
                    self%cuda_interface, grid%cuda_interface, center)
            call FirstModifiedSphericalCubeBessels_download_cuda(self%cuda_interface, result_cubes, shape(result_cubes))
            call FirstModifiedSphericalCubeBessels_unregister_result_array_cuda(self%cuda_interface, result_cubes)
        else
            result_cubes = self%eval_grid_cpu(grid, center, kappa_value)
        end if
#else
        result_cubes = self%eval_grid_cpu(grid, center, kappa_value)
#endif
    end function 

    function FirstModSphBesselCubeCollection_evaluate_grid_cpu(self, grid, center, kappa) result(result_cubes)
        class(FirstModSphBesselCubeCollection), intent(in) :: self
        type(Grid3D),                           intent(in) :: grid
        real(REAL64),                           intent(in) :: center(3)
        real(REAL64),    optional,              intent(in) :: kappa
        real(REAL64)                                       :: result_cubes(grid%axis(X_)%get_shape(), &
                                                                       grid%axis(Y_)%get_shape(), &
                                                                       grid%axis(Z_)%get_shape(), &
                                                                       0 : self%lmax)  

        type(Grid3D)                                       :: temp_grid
        real(REAL64)                                       :: kappa_value
        real(REAL64), allocatable                          :: kappa_r(:, :, :)
        real(REAL64)                                       :: gridpoints_x(grid%axis(X_)%get_shape()), &
                                                              gridpoints_y(grid%axis(Y_)%get_shape()), &
                                                              gridpoints_z(grid%axis(Z_)%get_shape())
        integer                                            :: o, p, q
 
        gridpoints_x = grid%axis(X_)%get_coord() - center(X_)
        gridpoints_y = grid%axis(Y_)%get_coord() - center(Y_)
        gridpoints_z = grid%axis(Z_)%get_coord() - center(Z_)
       
        allocate(kappa_r (     grid%axis(X_)%get_shape(), &
                               grid%axis(Y_)%get_shape(), &
                               grid%axis(Z_)%get_shape()) )

        ! evaluate kappa_r and positions relative to center of the farfield box for each grid point
        forall(o = 1 : grid%axis(X_)%get_shape(), p = 1 : grid%axis(Y_)%get_shape(), q = 1 : grid%axis(Z_)%get_shape())
            kappa_r(o, p, q) = kappa * sqrt(gridpoints_x(o) ** 2 + gridpoints_y(p) ** 2 + gridpoints_z(q) ** 2 )
        end forall

        ! evaluate modified spherical bessel function values at each cube point and reshape the result to be of a cube's shape
        result_cubes = reshape( &
                            self%eval(reshape(kappa_r,  [grid%axis(X_)%get_shape() * &
                                grid%axis(Y_)%get_shape() * grid%axis(Z_)%get_shape()])), &
                            [grid%axis(X_)%get_shape(), grid%axis(Y_)%get_shape(), grid%axis(Z_)%get_shape(), self%lmax+1])
        deallocate(kappa_r)
    end function 

    pure subroutine FirstModSphBesselCubeCollection_destroy(self) 
        class(FirstModSphBesselCubeCollection), intent(inout) :: self
        if (allocated(self%zero_values)) deallocate(self%zero_values)
        if (allocated(self%small_thresholds)) deallocate(self%small_thresholds)
#ifdef HAVE_CUDA
        call FirstModifiedSphericalCubeBessels_destroy_cuda(self%cuda_interface)
#endif
    end subroutine
!--------------------------------------------------------------------!
!  Second Modified Spherical Bessel Function Collection              !
!   (SecondModSphBesselCollection) - functions                       !
!--------------------------------------------------------------------!

    pure function SecondModSphBesselCollection_init(lmax, scaling_factor) result(new)
        integer, intent(in)                :: lmax
        real(REAL64), intent(in), optional :: scaling_factor
        type(SecondModSphBesselCollection) :: new

        if (present(scaling_factor)) then
            new%scaling_factor = scaling_factor
        else
            new%scaling_factor = 1.0d0
        end if
        new%lmax = lmax
        allocate(new%zero_values(0 : lmax), source = 0.0d0)
        allocate(new%small_thresholds(0 : lmax), source = 0.0d0)
       
        !if (lmax >= 0) new%small_thresholds(0) = 0.001d0
        !if (lmax >= 1) new%small_thresholds(1) = 0.003d0
        !if (lmax >= 2) new%small_thresholds(2) = 0.005d0
        !if (lmax >= 3) new%small_thresholds(3) = 0.015d0
        !if (lmax >= 4) new%small_thresholds(4) = 0.050d0
        !if (lmax >= 5) new%small_thresholds(5) = 0.110d0
        !if (lmax >= 6) new%small_thresholds(6) = 0.280d0
        !if (lmax >= 7) new%small_thresholds(7) = 0.420d0
        !if (lmax >= 8) new%small_thresholds(8) = 1.000d0
        !if (lmax >= 9) new%small_thresholds(9) = 3.000d0
    end function

    function SecondModSphBesselCollection_evaluate(self, parameter_values) result(res)
        class(SecondModSphBesselCollection), intent(in) :: self
        real(REAL64),                        intent(in) :: parameter_values(:)
        real(REAL64)                                    :: res(size(parameter_values), 0:self%lmax)
        real(REAL128)                                   :: reslt(size(parameter_values), 0:self%lmax), &
                                                           x, ax
        integer                                         :: i,l, recursion_max, npoints
        real(REAL128), parameter                        :: ZERO = epsilon(0.0d0)
        res= 0.0d0
        npoints = size(parameter_values)
#ifdef HAVE_OMP
        !$OMP PARALLEL DO IF (npoints > 1000)&
        !$OMP& PRIVATE(x, ax, l) 
#endif 
        do i=1, npoints
            x = parameter_values(i)
            ax = abs(x)
            if(ax > self%small_thresholds(self%lmax)) then
                reslt(i, 0) = exp(-x) / x * pi/2 
                reslt(i, 1) = exp(-x) * (x + 1) * self%scaling_factor / (x**2) * pi/2 
                do l = 2, self%lmax
                    reslt(i, l) = (2*l-1) * reslt(i, l-1) * self%scaling_factor / x &
                                  + reslt(i, l-2) * self%scaling_factor ** 2 
                                   
                end do
            else
                if (ax < ZERO) then
                    reslt(i, :) = self%zero_values
                else
                    forall (l = 0 : self%lmax)
                        reslt(i, l) = self%evaluate_small(x, l)
                    end forall
                end if
            end if
                      
        end do
#ifdef HAVE_OMP
        !$OMP END PARALLEL DO
#endif
        
        res(:, :) = reslt(:, :) 
    end function

    pure function SecondModSphBesselCollection_evaluate_small(self, z, n) result(reslt)
        class(SecondModSphBesselCollection), intent(in)    :: self
        real(REAL128),                       intent(in)   :: z
        integer,                            intent(in)    :: n
        real(REAL128)                                     :: reslt
        real(REAL64)                                      :: prefactor1, prefactor2, divider, divider2
        integer                                           :: i
        integer, parameter                                :: TERM_COUNT = 120

        
        prefactor1 = self%scaling_factor**n * z ** n  
        prefactor2 = self%scaling_factor**n / ((-1)**(n) * z**(n+1))
        divider = 1  
        do i = 1, 2*n-1, 2
            divider = divider * i 
        end do

        prefactor1 = prefactor1 / (divider * (2*n+1))
        prefactor2 = prefactor2 * divider

        reslt = prefactor1 - prefactor2
        divider = 1
        divider2 = 1
        do i = 1, TERM_COUNT
            divider = divider * (2*n + 2*i + 1)
            divider2 = divider2 * (-2*n + 2*i - 1)
            reslt = reslt + (0.5d0*z**2) ** i / factorial_real(i) * &
                                   ( prefactor1 / divider - prefactor2 / divider2) 
        end do
        reslt = pi/2 * (-1)**(n+1) * reslt

    end function

    function SecondModSphBesselCollection_evaluate_grid(self, grid, center, kappa) result(result_cubes)
        class(SecondModSphBesselCollection), intent(in):: self
        type(Grid3D),                       intent(in) :: grid
        real(REAL64),                       intent(in) :: center(3)
        real(REAL64),    optional,          intent(in) :: kappa
        real(REAL64)                                   :: result_cubes(grid%axis(X_)%get_shape(), &
                                                                       grid%axis(Y_)%get_shape(), &
                                                                       grid%axis(Z_)%get_shape(), &
                                                                       0 : self%lmax)   
        type(Grid3D)                                   :: temp_grid
        real(REAL64)                                   :: kappa_value
        real(REAL64), allocatable                      :: kappa_r(:, :, :)
        real(REAL64)                                   :: gridpoints_x(grid%axis(X_)%get_shape()), &
                                                          gridpoints_y(grid%axis(Y_)%get_shape()), &
                                                          gridpoints_z(grid%axis(Z_)%get_shape())
        integer                                        :: o, p, q

        if (present(kappa)) then
            kappa_value = kappa
        else
            kappa_value = 1.0d0;
        end if

#ifdef HAVE_CUDA

        temp_grid = grid
        ! initialize the cuda grid
        !call SecondModSphBesselCollection_cuda_evaluate_grid(&
        !        temp_grid%get_cuda_interface(), self%lmax, self%scaling_factor, kappa_value, center, result_cubes)
        call temp_grid%cuda_destroy()
        call temp_grid%destroy() 
#else
        gridpoints_x = grid%axis(X_)%get_coord() - center(X_)
        gridpoints_y = grid%axis(Y_)%get_coord() - center(Y_)
        gridpoints_z = grid%axis(Z_)%get_coord() - center(Z_)
       
        allocate(kappa_r (     grid%axis(X_)%get_shape(), &
                               grid%axis(Y_)%get_shape(), &
                               grid%axis(Z_)%get_shape()) )

        ! evaluate kappa_r and positions relative to center of the farfield box for each grid point
        forall(o = 1 : grid%axis(X_)%get_shape(), p = 1 : grid%axis(Y_)%get_shape(), q = 1 : grid%axis(Z_)%get_shape())
            kappa_r(o, p, q) = kappa_value * sqrt(gridpoints_x(o) ** 2 + gridpoints_y(p) ** 2 + gridpoints_z(q) ** 2 )
        end forall  
        

        ! evaluate modified spherical bessel function values at each cube point and reshape the result to be of a cube's shape
        result_cubes = reshape( &
                            self%eval(reshape(kappa_r,  [grid%axis(X_)%get_shape() * &
                                grid%axis(Y_)%get_shape() * grid%axis(Z_)%get_shape()])), &
                            [grid%axis(X_)%get_shape(), grid%axis(Y_)%get_shape(), grid%axis(Z_)%get_shape(), self%lmax+1])
        deallocate(kappa_r)
#endif
    end function 

    pure subroutine SecondModSphBesselCollection_destroy(self)
        class(SecondModSphBesselCollection), intent(inout)  :: self
        if (allocated(self%zero_values)) deallocate(self%zero_values)
        if (allocated(self%small_thresholds)) deallocate(self%small_thresholds)
    end subroutine


    pure subroutine BesselCollection_destroy(self)
        class(BesselCollection), intent(inout) :: self
        integer                                :: i

        do i = 0, size(self%first) -1 
            call self%first(i)%destroy()
        end do
        do i = 0,  size(self%second) -1
            call self%second(i)%destroy()
        end do
        deallocate(self%first)
        deallocate(self%second)
    end subroutine


    pure function BesselCollection_init(l)  result(new)
        integer, intent(in)               :: l
        type(BesselCollection)            :: new
        integer                           :: i
        type(FirstModSphBessel)           :: first(0 : l)
        type(SecondModSphBessel)          :: second(0 : l)

        i = 0
        do i = 0, l
            first(i)  = FirstModSphBessel(i)
            second(i) = SecondModSphBessel(i)
        end do
        new%first =first
        new%second=second
    end function

    !pure elemental function factorial_128(x) result(reslt)
    !     integer, intent(in)         :: x
    !     integer(kind=16)            :: i, reslt
         

    !     reslt = 1
    !     do i = 1, x
    !         reslt = reslt * i 
    !     end do
    !end function






end module
