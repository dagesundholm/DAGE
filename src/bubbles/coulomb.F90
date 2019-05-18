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
module Coulomb3D_class
    use Function3D_class ! contains the Projector3D & Operator3D classes
    use Function3D_types_m
    use Grid_class 
    use globals_m
    use GaussQuad_class
    use Bubbles_class
    use timer_m
    use xmatrix_m
    use LIPBasis_class
    use mpi_m
    use Evaluators_class
    use expo_m
    use ParallelInfo_class
    use RealSphericalHarmonics_class
    use harmonic_class
#ifdef HAVE_OMP
    use omp_lib
#endif
 
    public :: Coulomb3D
    public    :: assignment(=)

     !> Computes the electrostatic potential of Function3D in the output grid.
    type, extends(Operator3D)        :: Coulomb3D
        logical, public              :: cut_end_borders(3)
        type(Function3D)             :: nuclear_potential
        type(GaussQuad)              :: gaussian_quadrature
        class(ParallelInfo), pointer :: in_parallel_info
    contains
        procedure, public :: get_potential_bubbles => Coulomb3D_get_potential_bubbles
        procedure, public :: get_nuclear_potential => Coulomb3D_get_nuclear_potential
        procedure, private :: get_off_diagonal_nuclear_bubbles => &
                                  Coulomb3D_get_off_diagonal_nuclear_bubbles
        procedure :: transform_bubbles  => Coulomb3D_genpot_bubbles
        procedure :: evaluate_bubbles_potential  => Coulomb3D_evaluate_bubbles_potential
        procedure  :: get_suboperator   => Coulomb3D_get_suboperator
        procedure          :: destroy   => Coulomb3D_destroy
        procedure, public   :: calculate_coda => Coulomb3D_calculate_coda
        procedure, public   :: get_dims       => Coulomb3D_get_dims
        procedure, public   :: init_explicit_sub => Coulomb3D_init_explicit_sub
        procedure, public   :: init_copy_sub     => Coulomb3D_init_copy_sub
#ifdef HAVE_CUDA
        procedure, public   :: cuda_prepare   => Coulomb3D_cuda_prepare
        procedure, public   :: cuda_unprepare => Coulomb3D_cuda_unprepare
#endif
    end type
    
    interface Coulomb3D
        module procedure :: Coulomb3D_init
    end interface

    interface assignment(=)
        module procedure :: Coulomb3D_assign
    end interface

contains

#ifdef HAVE_CUDA
    subroutine Coulomb3D_cuda_prepare(self)
        class(Coulomb3D), intent(inout)  :: self
    end subroutine

    subroutine Coulomb3D_cuda_unprepare(self)
        class(Coulomb3D), intent(inout)  :: self
    end subroutine
#endif

    subroutine Coulomb3D_assign(operator1, operator2)
        class(Coulomb3D), intent(inout), allocatable :: operator1
        class(Coulomb3D), intent(in)                 :: operator2

        allocate(operator1, source = operator2)
    end subroutine
    
    subroutine Coulomb3D_init_copy_sub(self, original_operator)
        class(Coulomb3D),   intent(inout)          :: self
        type(Coulomb3D),    intent(in)             :: original_operator
        
        call self%init_explicit_sub(original_operator%in_parallel_info, original_operator%result_parallelization_info, &
                                    original_operator%nuclear_potential%bubbles, original_operator%gaussian_quadrature)
    end subroutine
    
    !> Constructor for Coulomb operator (\f$ \hat{O}(\rho)\rightarrow V\f$)
    !! \sa bubbles_genpot
    subroutine Coulomb3D_init_explicit_sub(self, input_parallel_info, output_parallel_info, bubbls, gauss)
        class(Coulomb3D),    intent(inout)          :: self
        !> Type of the input parallelization (contains info of the input grid)
        class(ParallelInfo), intent(in),   target   :: input_parallel_info
        !> Type of the input parallelization (contains info of the input grid)
        class(ParallelInfo), intent(in),   target   :: output_parallel_info
        !> Bubbles object containing all the correct charges and centers of atoms
        type(Bubbles),       intent(in)             :: bubbls
        !> Parameters for the Gaussian quadrature. Default initialization if not
        !! given (see gaussquad)
        type(GaussQuad),     intent(in), optional   :: gauss
        


        real(REAL64),pointer                       :: tp(:), tw(:)
        real(REAL64)                               :: tic, toc
        

        integer(INT32) :: sz, ip, i, ibub, j, k

        integer        :: jmax(3)

        call bigben%split("Coulomb operator initialization")
        if(verbo_g>0) call pinfo("Coulomb operator initialization")
        
        call bigben%split("Gaussian Quadrature initialization")
        if(present(gauss)) then
            self%gaussian_quadrature=gauss
        else
            self%gaussian_quadrature=GaussQuad()
        end if
        call bigben%stop()
        
        ! Quadrature parameters
        tp=>self%gaussian_quadrature%get_tpoints()
        tw=>self%gaussian_quadrature%get_weights()

        sz=size(tp)

        self%gridin  => input_parallel_info%get_grid()
        self%gridout => output_parallel_info%get_grid()
        self%w       = TWOOVERSQRTPI * tw
        self%coda    = PI/self%gaussian_quadrature%get_tlog()**2
        self%result_type = F3D_TYPE_CUSP

        ! set the output result function parallel info
        self%result_parallelization_info => output_parallel_info
        self%in_parallel_info         => input_parallel_info
 
        call bigben%split("Make Coulomb3d matrices")

        ! the make_Coulomb3d_matrices function is in file 'exponentials.F90'
        self%f       = make_Coulomb3D_matrices(self%gridin, self%gridout, tp, jmax)
        
        self%nuclear_potential = Function3D(self%result_parallelization_info, type=F3D_TYPE_NUCP)
        self%nuclear_potential%bubbles =  self%get_potential_bubbles(bubbls)
        self%nuclear_potential%cube = 0.0d0
        call self%nuclear_potential%precalculate_taylor_series_bubbles()
        self%cut_end_borders = .FALSE.
        self%suboperator = .FALSE.

        call bigben%stop()

        call bigben%stop()

    end subroutine

    
    !> Constructor for Coulomb operator (\f$ \hat{O}(\rho)\rightarrow V\f$)
    !! \sa bubbles_genpot
    function Coulomb3D_init(input_parallel_info, output_parallel_info, bubbls, gauss) result(new)
        !> Type of the input parallelization (contains info of the input grid)
        class(ParallelInfo), intent(in)             :: input_parallel_info
        !> Type of the input parallelization (contains info of the input grid)
        class(ParallelInfo), intent(in)             :: output_parallel_info
        !> Bubbles object containing all the correct charges and centers of atoms
        type(Bubbles),       intent(in)             :: bubbls
        !> Parameters for the Gaussian quadrature. Default initialization if not
        !! given (see gaussquad)
        type(GaussQuad),     intent(in), optional   :: gauss
        type(Coulomb3D)                             :: new
         
        

        call new%init_explicit_sub(input_parallel_info, output_parallel_info, bubbls, gauss)
    end function


    function Coulomb3D_genpot_bubbles(self, bubsin) result(new)
        class(Coulomb3D), intent(in)    :: self
        !> Input bubbles
        type(Bubbles),    intent(in)    :: bubsin
        type(Bubbles)                   :: new
        
        ! init bubbles like 'bubsin', do not copy the content
        new=Bubbles(bubsin, k = 0)

        call self%evaluate_bubbles_potential(bubsin, new)
    end function

    !> Bubbles transformation operation for `Projector3D`: compute
    !! compute the electrostatic potential caused by bubsin

    !> Given a charge distribution
    !! \f$\rho(r_A,\theta_A,\phi_A)= \rho_{Alm}(r_A) Y_{lm}(\theta_A,\phi_A)\f$
    !! The radial part of the potential is given by
    !! \f[
    !! V_{Alm}(r_A) = \frac{4\pi}{2l+1} \left[
    !!    r_A^{-l+1}\int_0^{r_A}      \rho_{Alm}(s_A) s_A^{l+2} \mathrm{d}s_A +
    !!    r_A^l     \int_{r_A}^\infty \rho_{Alm}(s_A) s_A^{1-l} \mathrm{d}s_A
    !!                                  \right]
    !! \f]
    subroutine Coulomb3D_evaluate_bubbles_potential(self, bubsin, new)
        class(Coulomb3D), intent(in)    :: self
        !> Input bubbles
        type(Bubbles),    intent(in)    :: bubsin
        !> Output coulomb-potential bubbles created from input
        type(Bubbles),    intent(inout) :: new
        !> Iterator variable ibub and number of bubbles 'nbub'
        integer(INT32)                  :: ibub, nbub

        real(REAL64),pointer            :: densf(:, :)
        real(REAL64),pointer            :: potf(:, :)
        real(REAL64), allocatable       :: rpow1(:), rpow2(:), rpow3(:), rpow4(:)
        integer                         :: l,m, domain(2), first_bubble, last_bubble, id, first_idx, last_idx, &
                                         first_l, first_m, last_l, last_m, l_m(2)
        type(InOutIntegrator1D)         :: integrator

        !call pdebug("genpot_bubbles()", 1)
        if(.not. (bubsin%get_nbub()>0)) return

        !if (bubsin%get_k()/=0) then
        !    write(ppbuf,'("bubbles_genpot(): Not implemented for k/=0 (got k=",g0,")")'), bubsin%get_k()
        !    call perror(ppbuf)
        !end if
        
        new = 0.0d0
        nbub=bubsin%get_nbub()
        if (.not.nbub>0) return
        if (verbo_g>0) call progress_bar(0,nbub)
        
        l = 0
        m = 0
        domain = self%result_parallelization_info%get_bubble_domain_indices &
                     (bubsin%get_nbub(), bubsin%get_lmax())
        first_bubble = (domain(1)-1) / (bubsin%get_lmax() + 1)**2 + 1
        last_bubble = (domain(2)-1) / (bubsin%get_lmax() + 1)**2 + 1     
 
        ! iterate over all bubbles
        do ibub=first_bubble, last_bubble
            integrator=InOutIntegrator1D(bubsin%get_grid(ibub))
            
            densf=>bubsin%get_f(ibub)
            potf=> new% get_f(ibub)
 
            ! get the first idx, i.e., the last l, m pair calculated by this processor
            first_idx = 1
            if (ibub == first_bubble) then
                first_idx = domain(1) - (ibub-1) * (bubsin%get_lmax() + 1)**2
            end if
            l_m = lm(first_idx)
            first_l = l_m(1)
            
            ! get the last idx, i.e., the last l, m pair calculated by this processor
            last_idx = (bubsin%get_lmax() + 1)**2
            if (ibub == last_bubble) then
                last_idx = domain(2) - (ibub-1) * (bubsin%get_lmax() + 1)**2
            end if
            l_m = lm(last_idx)
            last_l = l_m(1)
#ifdef HAVE_OMP
            !$OMP PARALLEL &
            !$OMP& PRIVATE(rpow1, rpow2, rpow3, rpow4, m, l, id) 
#endif
            allocate(rpow1(size(densf)))
            allocate(rpow2(size(densf)))
            allocate(rpow3(size(densf)))
            allocate(rpow4(size(densf)))
            ! Iterate over l
#ifdef HAVE_OMP
            !$OMP DO
#endif
            do l=first_l,last_l
                id = l*l +1
                rpow1 = bubsin%rpow(ibub, -(l+1))
                rpow2 = bubsin%rpow(ibub,    l   + 2 + bubsin%get_k())
                rpow3 = bubsin%rpow(ibub,    l  )
                rpow4 = bubsin%rpow(ibub, -(l+1) + 2 + bubsin%get_k())
                ! Iterate over m
                do m= -l, l
                    if (id > last_idx) then
                       exit
                    end if
                    ! V(r) = 4 \pi /(2l+1) \times \left[
                    potf(:, id)= FOURPI/(2*l+1) *&
                    ! r^{-(l+1)}
                         ( rpow1 * &
                    ! \int_0^r \rho(s) s^{l+2} ds +
                         integrator%outwards( densf(:, id) * rpow2) + &
                    ! r^l
                           rpow3 * &
                    ! \int_r^\infty \rho(s) s^{-(l+1)+2} ds \right]
                         integrator%inwards ( densf(:, id) * rpow4) )

                   
                    id = id + 1
                end do
                !if (id == last_idx) then
                !   exit 
                !end if
                
            end do
#ifdef HAVE_OMP
            !$OMP END DO
#endif
            deallocate(rpow1, rpow2, rpow3, rpow4)
#ifdef HAVE_OMP
            !$OMP END PARALLEL
#endif
            !if (verbo_g>0) call progress_bar(ibub)
            call integrator%destroy()
            nullify(potf)
            nullify(densf)
        end do

        return
    end subroutine

    subroutine Coulomb3D_calculate_coda(self, cubein, cubeout)
        class(Coulomb3D), intent(in)    :: self
        real(REAL64),     intent(in)    :: cubein(:, :, :)
        real(REAL64),     intent(inout) :: cubeout(:, :, :)
        
        if (self%coda /= 0.d0) then
            call bigben%split("Coda")
            cubeout=cubeout + &
                self%coda * cube_project(cubein, self%gridin, self%gridout)
            call bigben%stop()
        end if
    end subroutine

    function Coulomb3D_get_nuclear_potential(self) result(new)
        !> Input function
        class(Coulomb3D), intent(inout), target :: self
        type(Function3D), pointer               :: new

        new => self%nuclear_potential
        ! TODO: coda term is missing from the cube part, investigate if this is
        ! needed
        

    end function

    function Coulomb3D_get_potential_bubbles(self, input_bubbles) result(potential_bubbles)
        !> Input function
        class(Coulomb3D), intent(in)        :: self
        type(Bubbles),    intent(in)        :: input_bubbles
        type(Bubbles)                       :: potential_bubbles, diagonal_bubbles
        integer                             :: ibub
        real(REAL64)                        :: distance, integrate
        real(REAL64), pointer               :: bubble_values(:, :)
        lmax = 3

        ! init new bubbles from the input_function bubbles with all bubbles present, 
        ! and set k of 'new' to -1 (values will be multiplied with r^-1 at evaluation)
        diagonal_bubbles = Bubbles(lmax = lmax, centers = input_bubbles%get_global_centers(),&
                                   global_centers = input_bubbles%get_global_centers(), & 
                                   grids = input_bubbles%get_global_grid(), &
                                   global_grids = input_bubbles%get_global_grid(), &
                                   z = input_bubbles%get_global_z(), global_z = input_bubbles%get_global_z(), &
                                   k = -1, ibubs = [(i, i = 1, input_bubbles%get_nbub_global())], &
                                   nbub_global = input_bubbles%get_nbub_global()) 
        diagonal_bubbles = 0.0d0

        ! set value of the l = 0 bubble to -charge
        do ibub = 1, diagonal_bubbles%get_nbub()
            bubble_values => diagonal_bubbles%get_f(ibub)
            bubble_values(:, 1) = - diagonal_bubbles%get_z(ibub)
            nullify(bubble_values)
        end do
        
        potential_bubbles = Bubbles(diagonal_bubbles, copy_content = .TRUE.) ! + off_diagonal_bubbles
        call diagonal_bubbles%destroy()
    end function

    function Coulomb3D_get_off_diagonal_nuclear_bubbles(self, input_bubbles) result(off_diagonal_bubbles)
        !> Input function
        class(Coulomb3D), intent(in)        :: self
        type(Bubbles),    intent(in)        :: input_bubbles
        type(Bubbles)                       :: off_diagonal_bubbles
        type(RealSphericalHarmonics)        :: harmonics
        integer                             :: lmax, i, j, ibub, jbub, l, id
        real(REAL64), allocatable           :: translations(:, :), charges(:), spherical_harmonics(:, :)
        real(REAL64), allocatable           :: centers(:, :), all_charges(:)
        real(REAL64)                        :: distance, integrate
        real(REAL64), pointer               :: r(:)
        lmax = 2

        
        centers     = input_bubbles%get_centers()
        all_charges = input_bubbles%get_z()

        harmonics = RealSphericalHarmonics(lmax)
        off_diagonal_bubbles = Bubbles(input_bubbles, lmax = lmax, k = -1)
        off_diagonal_bubbles = 0.0d0

        ! evaluate the off diagonal terms in the potential. The evaluation
        ! is done by calculating the multipole moments caused by individual
        ! point charges at atom positions other than the bubble evaluated 
        ! at time.
        allocate(translations(3, input_bubbles%get_nbub()-1))
        allocate(charges(input_bubbles%get_nbub()-1))
        do ibub = 1, input_bubbles%get_nbub()
            r => input_bubbles%gr(ibub)%p%get_coord()
            do jbub = 1, ibub - 1
                translations(:, jbub) = centers(:, jbub) - centers(:, ibub)
                charges(jbub) = all_charges(jbub)
            end do 
            do jbub = ibub+1, input_bubbles%get_nbub()
                translations(:, jbub-1) = centers(:, jbub) - centers(:, ibub)
                charges(jbub-1) = all_charges(jbub)
            end do 
            
            spherical_harmonics = harmonics%eval(translations)
            print *, "spherical harmonics", spherical_harmonics
            do i = 1, size(translations, 2)
                distance = sqrt(sum(translations(:, i)**2))
                j = 1

                ! get the first index 'j' where distance > r(j)
                do while(r(j) < distance)
                    j = j + 1
                end do
                print *, "r(j)", r(j), "distance", distance, "charge", charges(i), &
                         "translation", translations(:, i)
                ! evaluate the laplace expansion
                do l = 0, lmax
                    forall (id = l**2+1 : (l+1)**2)
                        ! evaluate the potential for points r where distance > r
                        off_diagonal_bubbles%bf(ibub)%p(:j-1, id) = &
                            off_diagonal_bubbles%bf(ibub)%p(:j-1, id) &
                            - (-1.0d0)**l * (r(:j-1) / distance) ** (l+1) &
                            * spherical_harmonics(i, id)  &
                            * charges(i) 

                        ! evaluate the potential for points r where distance < r
                        off_diagonal_bubbles%bf(ibub)%p(j:, id) = &
                            off_diagonal_bubbles%bf(ibub)%p(j:, id) &
                            - (-1.0d0)**l * (distance / r(j:)) **(l)   &
                            * spherical_harmonics(i, id) &
                            * charges(i)
                    end forall
                end do
            end do
            nullify(r)
            deallocate(spherical_harmonics)
        end do 
        deallocate(translations)
        deallocate(centers, all_charges, charges)
    end function

    ! get a part of the operator that does the operation only with part
    ! of the original output grid specified in 'gridout' 
    function Coulomb3D_get_suboperator(self, in_cell_limits, out_cell_limits, &
                                       input_parallel_info, output_parallel_info, &
                                       gauss, cut_end_borders) result(new)
        !> Input function
        class(Coulomb3D), intent(in) :: self
        !> The cell limits of the output area of the suboperator
        integer,             intent(in)           :: out_cell_limits(2, 3)
        !> The cell limits of the input area of the suboperator
        integer,             intent(in)           :: in_cell_limits(2, 3)
        !> Type of the input parallelization (contains info of the input grid)
        class(ParallelInfo), intent(in), target   :: input_parallel_info
        !> Type of the input parallelization (contains info of the input grid)
        class(ParallelInfo), intent(in), target   :: output_parallel_info
        type(GaussQuad), optional                 :: gauss
        logical,             intent(in), optional :: cut_end_borders(3)
        integer                                   :: out_cube_limits(2, 3), in_cube_limits(2, 3), &
                                                     grid_in_shape(3), grid_out_shape(3)
        type(GaussQuad)                           :: gau
        integer                                   :: jmax(3)
        type(Coulomb3D)                           :: new
        type(REAL64_3D)                           :: borders(3)

        if(present(gauss)) then
            gau=gauss
        else
            gau=GaussQuad()
        end if
 
        ! calculate the limits in the in cube (or in the f matrix of self)
        in_cube_limits(1, :) = (in_cell_limits(1, :) - (/1, 1, 1/)) * (self%gridin%get_nlip() - 1) + 1
        in_cube_limits(2, :) = (in_cell_limits(2, :)) * (self%gridin%get_nlip() - 1) + 1        

        ! calculate the limits in the out cube (or in the f matrix of self)
        out_cube_limits(1, :) = (out_cell_limits(1, :) - (/1, 1, 1/)) * (self%gridout%get_nlip() - 1) + 1
        out_cube_limits(2, :) = (out_cell_limits(2, :)) * (self%gridout%get_nlip() - 1) + 1
         
        ! set input and output grids         
        new%gridout => output_parallel_info%get_grid()
        new%gridin  => input_parallel_info%get_grid()
            
        ! get the shapes of the grids in gridpoints
        grid_in_shape = new%gridin%get_shape()  
        grid_out_shape = new%gridout%get_shape()  

        ! if the parallelization info was given in the initialization of the main operator use it
        ! in other cases the parallelization info of the input function3d is used
        new%result_parallelization_info => output_parallel_info
 
        ! get the f matrix, start by getting the borders, which are different in the suboperator when 
        ! comparing with the main operator
        borders = make_Coulomb3D_matrices(new%gridin, new%gridout, gau%get_tpoints(), jmax, only_border=.TRUE.)

        if (present(cut_end_borders)) then
            allocate(new%f(3))
            new%cut_end_borders = cut_end_borders
            new%f = borders
            !if (cut_end_borders(X_)) then
            !    allocate(new%f(X_)%p(grid_out_shape(X_)-1, grid_in_shape(X_), size(borders(X_)%p, 3)))
            !    new%f(X_)%p = borders(X_)%p(1:grid_out_shape(X_)-1, :, :)
            !else
            !    allocate(new%f(X_)%p(grid_out_shape(X_), grid_in_shape(X_), size(borders(X_)%p, 3)))
            !    new%f(X_)%p = borders(X_)%p
            !end if
            !if (cut_end_borders(Y_)) then
            !    allocate(new%f(Y_)%p(grid_in_shape(Y_), grid_out_shape(Y_)-1, size(borders(Y_)%p, 3)))
            !    new%f(Y_)%p = borders(Y_)%p(:, 1:grid_out_shape(Y_)-1, :)
            !else
            !    allocate(new%f(Y_)%p(grid_in_shape(Y_), grid_out_shape(Y_), size(borders(Y_)%p, 3)))
            !    new%f(Y_)%p = borders(Y_)%p
            !end if
            !if (cut_end_borders(Z_)) then
            !    allocate(new%f(Z_)%p(grid_in_shape(Z_), grid_out_shape(Z_)-1, size(borders(Z_)%p, 3)))
            !    new%f(Z_)%p = borders(Z_)%p(:, 1:grid_out_shape(Z_)-1, :)
            !else
            !    allocate(new%f(Z_)%p(grid_in_shape(Z_), grid_out_shape(Z_), size(borders(Z_)%p, 3)))
            !    new%f(Z_)%p = borders(Z_)%p
            !end if
        else
            new%cut_end_borders = .FALSE.
            new%f = borders
        end if
        

        if (allocated(borders(X_)%p)) deallocate(borders(X_)%p)
        if (allocated(borders(Y_)%p)) deallocate(borders(Y_)%p)
        if (allocated(borders(Z_)%p)) deallocate(borders(Z_)%p)

        ! the middle is the same, and thus we can copy it from the main operator
        new%f(X_)%p(2 : grid_out_shape(X_) - 1, 2 : grid_in_shape(X_) -1, :) =            & 
                     self%f(X_)%p(out_cube_limits(1, X_) +1 : out_cube_limits(2, X_) -1,  &
                     in_cube_limits(1, X_) +1 : in_cube_limits(2, X_) -1, :) 
        new%f(Y_)%p(2 : grid_in_shape(Y_) - 1, 2 : grid_out_shape(Y_) -1, :) =            &
                     self%f(Y_)%p(in_cube_limits(1, Y_) +1  : in_cube_limits(2, Y_) -1,   &
                     out_cube_limits(1, Y_) +1 : out_cube_limits(2, Y_) -1, :) 
        new%f(Z_)%p(2 : grid_in_shape(Z_) - 1, 2 : grid_out_shape(Z_) -1, :) =            &
                     self%f(Z_)%p(in_cube_limits(1, Z_) +1  : in_cube_limits(2, Z_) -1,   &
                     out_cube_limits(1, Z_) +1 : out_cube_limits(2, Z_) -1, :) 

        !new%f(X_)%p(1 : grid_out_shape(X_), 1 : grid_in_shape(X_) , :) =            & 
        !             self%f(X_)%p(out_cube_limits(1, X_) : out_cube_limits(2, X_),  &
        !             in_cube_limits(1, X_) : in_cube_limits(2, X_), :) 
        !new%f(Y_)%p(1 : grid_in_shape(Y_), 1 : grid_out_shape(Y_), :) =            &
        !             self%f(Y_)%p(in_cube_limits(1, Y_)  : in_cube_limits(2, Y_),   &
        !             out_cube_limits(1, Y_) : out_cube_limits(2, Y_), :) 
        !new%f(Z_)%p(1 : grid_in_shape(Z_), 1 : grid_out_shape(Z_), :) =            &
        !             self%f(Z_)%p(in_cube_limits(1, Z_)  : in_cube_limits(2, Z_),   &
        !             out_cube_limits(1, Z_) : out_cube_limits(2, Z_), :) 

#ifdef HAVE_CUDA
        new%cuda_inited = .FALSE.
#endif

        ! other parameters remain the same
        !new%coda    = self%coda
        new%result_type = self%result_type
        new%w = self%w
        new%suboperator = .TRUE.
        
    end function

    subroutine Coulomb3D_destroy(self)
        class(Coulomb3D), intent(inout) :: self
        
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
        call self%nuclear_potential%destroy()
    end subroutine

    pure function Coulomb3D_get_dims(self) result(dims)
        class(Coulomb3D), intent(in)  :: self

        integer                       :: dims(X_:Z_, 2) !(3,2)

        dims(:,1) =self%gridin%get_shape()
        dims(:,2)=self%gridout%get_shape()
    
        !if (self%cut_end_borders(X_)) dims(X_, 2) = dims(X_, 2)-1
        !if (self%cut_end_borders(Y_)) dims(X_, 2) = dims(Y_, 2)-1
        !if (self%cut_end_borders(Z_)) dims(X_, 2) = dims(Z_, 2)-1
    end function
end module
