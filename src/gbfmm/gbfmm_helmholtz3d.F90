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
module GBFMMHelmholtz3D_class
#ifdef HAVE_MPI
    use MPI
#endif
    use ISO_FORTRAN_ENV
    use Function3D_class
    use Function3D_types_m
    use Multipole_class
    use Coulomb3D_class
    use GaussQuad_class
    use Helmholtz3D_class
    use GBFMMFunction3D_class
    use GBFMMCoulomb3D_class
    use ParallelInfo_class
    use GBFMMParallelInfo_class
    use RealSphericalHarmonics_class
    use Harmonic_class
    use HelmholtzMultipoleTools_class
    use MemoryLeakChecker_m
#ifdef HAVE_CUDA
    use cuda_m
    use ISO_C_BINDING
#endif
#ifdef HAVE_OMP
    use omp_lib
#endif
    use Grid_class
    use XMatrix_m
    use Globals_m
    implicit none

    public      :: GBFMMHelmholtz3D

    private

    

    type, extends(Helmholtz3D) :: GBFMMHelmholtz3D
        !> The normalization of Spherical used in the evaluation of the Helmholtz potential 2: is conventional
        !! and 1 (or any other is Racah normalization). The choice won't affect the result, but there might be slight
        !! changes in execution speed.
        integer                              :: normalization = 2
        !> the maximum quantum number 'l' used in farfield potential evaluation at the final step. 
        integer                              :: farfield_potential_lmax = 22
        !> The maximum quantum number 'l' used in the farfield potential evaluation at the multipole
        !! evaluation step. This number must be larger than 'farfield_potential_lmax'. 
        !! The larger the difference between this and 'farfield_potential_lmax' gives more accuracy
        !! to the potential evaluation. However, increased number will slow the process down. 
        integer                              :: farfield_potential_input_lmax = 22
        logical                              :: multipole_evaluation = .FALSE.
        real(REAL64),            allocatable :: multipole_moments(:, :)
        real(REAL64),            allocatable :: local_expansions(:, :)
        logical                              :: naive = .TRUE.
        type(GBFMMParallelInfo), pointer     :: input_parallel_info
        type(GBFMMParallelInfo), pointer     :: output_parallel_info
#ifdef HAVE_CUDA
        type(C_PTR)                          :: multipole_evaluator
#endif
    contains
        
        procedure               :: operate_on                   => GBFMMHelmholtz3D_operate
        procedure               :: operate_farfield             => GBFMMHelmholtz3D_operate_farfield
        procedure               :: simplify                     => GBFMMHelmholtz3D_simplify
        procedure               :: operate_nearfield            => GBFMMHelmholtz3D_operate_nearfield
        procedure, private      :: calculate_cube_multipoles_cpu=> GBFMMHelmholtz3D_calculate_cube_multipoles_cpu
        procedure               :: evaluate_potential_me        => GBFMMHelmholtz3D_evaluate_potential_me
        procedure, private      :: evaluate_potential_me_cpu    => GBFMMHelmholtz3D_evaluate_potential_me_cpu
        procedure               :: evaluate_potentials_le        => GBFMMHelmholtz3D_evaluate_potentials_le
        procedure, private      :: evaluate_potentials_le_cpu    => GBFMMHelmholtz3D_evaluate_potentials_le_cpu
#ifdef HAVE_CUDA
        procedure               :: cuda_init                    => GBFMMHelmholtz3D_cuda_init
        procedure               :: cuda_init_child_operator     => GBFMMHelmholtz3D_cuda_init_child_operator
        procedure               :: cuda_prepare                 => GBFMMHelmholtz3D_cuda_init_harmonics
        procedure               :: cuda_unprepare               => GBFMMHelmholtz3D_cuda_destroy_harmonics
        procedure, private      :: calculate_cube_multipoles_cuda => GBFMMHelmholtz3D_calculate_cube_multipoles_cuda
        procedure, private      :: evaluate_potentials_le_cuda   => GBFMMHelmholtz3D_evaluate_potentials_le_cuda
#endif
        procedure               :: set_energy                   => GBFMMHelmholtz3D_set_energy
        procedure               :: get_cube_local_expansion     => GBFMMHelmholtz3D_get_cube_local_expansion
        procedure               :: get_cube_comparison_potential_cpu => GBFMMHelmholtz3D_get_cube_comparison_potential_cpu
        procedure               :: get_cube_comparison_potential   => GBFMMHelmholtz3D_get_cube_comparison_potential
        procedure               :: get_complex_cube_multipoles  => GBFMMHelmholtz3D_get_complex_cube_multipoles
        procedure               :: destroy                      => GBFMMHelmholtz3D_destroy
        procedure               :: translate_multipoles         => GBFMMHelmholtz3D_translate_multipoles
        procedure               :: translate_complex_multipoles => GBFMMHelmholtz3D_translate_complex_multipoles
    end type


    interface GBFMMHelmholtz3D
        module procedure GBFMMHelmholtz3D_init
    end interface


!------------ CUDA Interfaces ----------------- !

#ifdef HAVE_CUDA
    interface
        type(C_PTR) function GBFMMHelmholtz3D_init_cuda( &
               grid_in, grid_out, lmax, domain, &
               input_start_indices_x, input_end_indices_x, &
               input_start_indices_y, input_end_indices_y, &
               input_start_indices_z, input_end_indices_z, &
               output_start_indices_x, output_end_indices_x, &
               output_start_indices_y, output_end_indices_y, &
               output_start_indices_z, output_end_indices_z, &
               streamContainer)  bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: grid_in
            type(C_PTR), value    :: grid_out
            integer(C_INT), value :: lmax
            integer(C_INT)        :: domain(2)
            integer(C_INT)        :: input_start_indices_x(*) 
            integer(C_INT)        :: input_end_indices_x(*) 
            integer(C_INT)        :: input_start_indices_y(*) 
            integer(C_INT)        :: input_end_indices_y(*) 
            integer(C_INT)        :: input_start_indices_z(*) 
            integer(C_INT)        :: input_end_indices_z(*) 
            integer(C_INT)        :: output_start_indices_x(*) 
            integer(C_INT)        :: output_end_indices_x(*) 
            integer(C_INT)        :: output_start_indices_y(*) 
            integer(C_INT)        :: output_end_indices_y(*) 
            integer(C_INT)        :: output_start_indices_z(*) 
            integer(C_INT)        :: output_end_indices_z(*) 
            type(C_PTR), value    :: streamContainer

        end function
    end interface

    interface
        type(C_PTR) function GBFMMHelmholtz3D_init_child_operator_cuda( &
               gbfmm_coulomb3d, lmax)  bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: gbfmm_coulomb3d
            integer(C_INT), value :: lmax

        end function
    end interface

    interface
        subroutine GBFMMHelmholtz3D_init_harmonics_cuda(gbfmm_helmholtz3d)  bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: gbfmm_helmholtz3d

        end subroutine
    end interface

    interface
        subroutine GBFMMHelmholtz3D_destroy_harmonics_cuda(gbfmm_helmholtz3d)  bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: gbfmm_helmholtz3d

        end subroutine
    end interface

    interface
        subroutine GBFMMHelmholtz3D_destroy_cuda(gbfmm_helmholtz3d)  bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: gbfmm_helmholtz3d

        end subroutine
    end interface

    interface
        subroutine GBFMMHelmholtz3D_evaluate_potential_le_cuda(gbfmm_helmholtz3d, local_expansion, output_cube)  bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: gbfmm_helmholtz3d
            real(C_DOUBLE)        :: local_expansion(*)
            type(C_PTR), value    :: output_cube

        end subroutine
    end interface

    interface
        subroutine GBFMMHelmholtz3D_calculate_multipole_moments_cuda(gbfmm_helmholtz3d, input_cube)  bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: gbfmm_helmholtz3d
            type(C_PTR), value    :: input_cube

        end subroutine
    end interface

    interface 
        type(C_PTR) function GBFMMHelmholtz3D_get_box_stream_container_cuda(gbfmm_helmholtz3d, ibox) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value    :: gbfmm_helmholtz3d
            integer(C_INT), value    :: ibox
        end function
    end interface

    interface
        subroutine GBFMMHelmholtz3D_download_multipole_moments_cuda(gbfmm_helmholtz3d, host_multipole_moments)  bind(C)
            use ISO_C_BINDING
            type(C_PTR), value       :: gbfmm_helmholtz3d
            real(C_DOUBLE)           :: host_multipole_moments(*)

        end subroutine
    end interface

    interface
        subroutine GBFMMHelmholtz3D_upload_domain_boxes_cuda(gbfmm_helmholtz3d, input_cube)  bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: gbfmm_helmholtz3d
            type(C_PTR), value    :: input_cube

        end subroutine
    end interface

    interface
        subroutine GBFMMHelmholtz3D_set_energy_cuda(gbfmm_helmholtz3d, energy)  bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: gbfmm_helmholtz3d
            real(C_DOUBLE), value :: energy

        end subroutine
    end interface
#endif

contains
    function GBFMMHelmholtz3D_init(coulomb_operator, energy, quadrature, lmax, &
                farfield_potential_lmax, farfield_potential_input_lmax, adaptive_mode, &
                normalization) result(new)
        !> Input coulomb operator. Note: Should be of type GBFMMCoulomb3D
        class(Coulomb3D),     intent(in),  target  :: coulomb_operator
        !> Energy used in the construction of the operator. Used in the parameter 
        !! 'kappa' of the kernel defined \sqrt(-2*energy) 
        real(REAL64),         intent(in)           :: energy
        !> Gaussian quadrature
        !! If no quadrature is given, then default values used are
        !! nlin=10 and nlog=10. These are the same default values 
        !! used in Coulomb3D
        type(GaussQuad), optional, intent(in)     :: quadrature
        !> The maximum angular momentum of the result bubbles.
        !! The resulting extra bubbles are injected to the cube
        integer, optional, intent(in)             :: lmax
        !> The maximum angular momentum of multipole expansion in the farfield
        !! potential calculation
        integer, optional, intent(in)             :: farfield_potential_lmax
        !> The maximum angular momentum of multipole expansion in the farfield
        !! potential calculation
        integer, optional, intent(in)             :: farfield_potential_input_lmax
        !> Normalization used in spherical harmonics 2=conventional, any other is 
        !! Racah's normalization
        integer, optional, intent(in)             :: normalization
        !> If we are using the adpative mode in the local farfield evaluation. If not 
        !! specified, the adaptive mode will be used
        logical, optional, intent(in)             :: adaptive_mode

        integer                                   :: box_count
        
        type(GBFMMHelmholtz3D)                    :: new
        
        select type(coulomb_operator)
            type is (Coulomb3D)
                print *, "ERROR trying to initialize GBFMMHelmholtz3D (at GBFMMHelmholtz3D_cuda_init):&
                         & Invalid coulomb operator type: Coulomb3D, should be GBFMMCoulomb3D."
        end select
        
        if (present (lmax)) then
            new%lmax = lmax
        end if

        if (present (farfield_potential_lmax)) then
            new%farfield_potential_lmax = farfield_potential_lmax
        end if

        if (present (farfield_potential_input_lmax)) then
            new%farfield_potential_input_lmax = farfield_potential_input_lmax
        end if

        if(present(quadrature)) then
            new%quadrature = quadrature
        else
            new%quadrature =  GaussQuad()
        endif

        if (present(normalization)) then
            new%normalization = normalization
        end if
        
        new%coulomb_operator => coulomb_operator

        
       

        ! init a gbfmm helmholtz parallel info using the parallel_info of the 'coulomb_operator'
        ! that should of type GBFMMCoulomb3D
        select type(coul_operator => new%coulomb_operator)
            type is (GBFMMCoulomb3D)
                new%input_parallel_info  =>  coul_operator%input_parallel_info
                new%output_parallel_info =>  coul_operator%output_parallel_info
                new%result_parallelization_info => coul_operator%result_parallelization_info
        end select

         ! get the total number of boxes in all levels
        box_count = new%input_parallel_info%get_total_box_count()

        ! allocate memory for the multipole moments and local expansion and the result cube
        allocate(new%multipole_moments((new%farfield_potential_input_lmax + 1) ** 2, box_count), source = 0.0d0)

        call new%set_energy(energy)

    end function

#ifdef HAVE_CUDA
    subroutine GBFMMHelmholtz3D_cuda_init_child_operator(self)
        class(GBFMMHelmholtz3D), intent(inout)  :: self
        integer                                 :: box_number, ibox, domain(2)

        integer                                 :: box_cell_index_limits(2, 3)
        type(C_PTR)                             :: box_stream_container

        ! get starting and ending indices of this processors domain
        domain = self%input_parallel_info%get_domain(self%input_parallel_info%maxlevel)
        
        if (.not. self%coulomb_operator%is_cuda_inited()) then
            print *, "ERROR trying to initialize Cuda of GBFMMHelmholtz3D (at GBFMMHelmholtz3D_cuda_init_child_operator):&
                     & Input Coulomb operator has not been cuda-initialized!"
            stop
        end if
        ! init the CUDA object responsible for multipole evaluations, etc.
        self%cuda_interface = GBFMMHelmholtz3D_init_child_operator_cuda( &
                                   self%coulomb_operator%cuda_interface, &
                                   self%farfield_potential_input_lmax)

        ! init the input & output cuda cubes
        self%input_cuda_cube  = self%coulomb_operator%input_cuda_cube
        self%output_cuda_cube = self%coulomb_operator%output_cuda_cube
        self%cuda_tmp1        = self%coulomb_operator%cuda_tmp1
        self%cuda_tmp2        = self%coulomb_operator%cuda_tmp2


        ! mark coulomb operator and self to be inited
        self%cuda_inited = .TRUE.
    end subroutine

    subroutine GBFMMHelmholtz3D_cuda_init(self)
        class(GBFMMHelmholtz3D), intent(inout)  :: self
        integer                                 :: box_number, ibox, domain(2)
        integer, allocatable                    :: input_start_indices_x(:), input_start_indices_y(:), &
                                                   input_start_indices_z(:), input_end_indices_x(:), &
                                                   input_end_indices_y(:),   input_end_indices_z(:)
        integer, allocatable                    :: output_start_indices_x(:), output_start_indices_y(:), &
                                                   output_start_indices_z(:), output_end_indices_x(:), &
                                                   output_end_indices_y(:),   output_end_indices_z(:)
        integer                                 :: input_box_cell_index_limits(2, 3), output_box_cell_index_limits(2, 3)
        type(C_PTR)                             :: box_stream_container

        ! get starting and ending indices of this processors domain
        domain = self%input_parallel_info%get_domain(self%input_parallel_info%maxlevel)

        allocate(input_start_indices_x(domain(2)-domain(1)+1))
        allocate(input_start_indices_y(domain(2)-domain(1)+1))
        allocate(input_start_indices_z(domain(2)-domain(1)+1))
        allocate(input_end_indices_x(domain(2)-domain(1)+1))
        allocate(input_end_indices_y(domain(2)-domain(1)+1))
        allocate(input_end_indices_z(domain(2)-domain(1)+1))
        allocate(output_start_indices_x(domain(2)-domain(1)+1))
        allocate(output_start_indices_y(domain(2)-domain(1)+1))
        allocate(output_start_indices_z(domain(2)-domain(1)+1))
        allocate(output_end_indices_x(domain(2)-domain(1)+1))
        allocate(output_end_indices_y(domain(2)-domain(1)+1))
        allocate(output_end_indices_z(domain(2)-domain(1)+1))

        do ibox = domain(1), domain(2)
            ! the order number of box in domain
            box_number = ibox - domain(1) + 1

            input_box_cell_index_limits = self%input_parallel_info%get_box_cell_index_limits(ibox, &
                                           level = self%input_parallel_info%maxlevel)
            input_start_indices_x(box_number) = input_box_cell_index_limits(1, X_) -1
            input_start_indices_y(box_number) = input_box_cell_index_limits(1, Y_) -1
            input_start_indices_z(box_number) = input_box_cell_index_limits(1, Z_) -1
            input_end_indices_x(box_number) = input_box_cell_index_limits(2, X_)
            input_end_indices_y(box_number) = input_box_cell_index_limits(2, Y_)
            input_end_indices_z(box_number) = input_box_cell_index_limits(2, Z_)

            output_box_cell_index_limits = self%output_parallel_info%get_box_cell_index_limits(ibox, &
                                           level = self%output_parallel_info%maxlevel)
            output_start_indices_x(box_number) = output_box_cell_index_limits(1, X_) -1
            output_start_indices_y(box_number) = output_box_cell_index_limits(1, Y_) -1
            output_start_indices_z(box_number) = output_box_cell_index_limits(1, Z_) -1
            output_end_indices_x(box_number) = output_box_cell_index_limits(2, X_)
            output_end_indices_y(box_number) = output_box_cell_index_limits(2, Y_)
            output_end_indices_z(box_number) = output_box_cell_index_limits(2, Z_)
        end do

        ! init the input & output cuda cubes
        self%input_cuda_cube = CudaCube(self%coulomb_operator%gridin%get_shape())
        self%output_cuda_cube = CudaCube(self%coulomb_operator%gridout%get_shape())
        self%cuda_tmp1 = CudaCube([self%gridout%axis(X_)%get_shape(), &
                                   self%gridin%axis(Y_)%get_shape(), &
                                   self%gridin%axis(Z_)%get_shape()])
        self%cuda_tmp2 = CudaCube([self%gridout%axis(X_)%get_shape(), &
                                   self%gridout%axis(Y_)%get_shape(), &
                                   self%gridin%axis(Z_)%get_shape()])
 
        ! init the CUDA object responsible for multipole evaluations, etc.
        self%cuda_interface = GBFMMHelmholtz3D_init_cuda(self%coulomb_operator%gridin%get_cuda_interface(), &
                                   self%coulomb_operator%gridout%get_cuda_interface(), &
                                   self%farfield_potential_input_lmax, domain, &
                                   input_start_indices_x, input_end_indices_x, &
                                   input_start_indices_y, input_end_indices_y, &
                                   input_start_indices_z, input_end_indices_z, &
                                   output_start_indices_x, output_end_indices_x, &
                                   output_start_indices_y, output_end_indices_y, &
                                   output_start_indices_z, output_end_indices_z, &
                                   stream_container)
        deallocate(input_start_indices_x, input_start_indices_y, input_start_indices_z, &
                   input_end_indices_x, input_end_indices_y, input_end_indices_z)
        deallocate(output_start_indices_x, output_start_indices_y, output_start_indices_z, &
                   output_end_indices_x, output_end_indices_y, output_end_indices_z)

        select type(coulomb_operator => self%coulomb_operator)
            type is (GBFMMCoulomb3D)
                do ibox = domain(1), domain(2)
                    ! get streamcontainer handling the box
                    !box_stream_container = GBFMMCoulomb3D_get_box_stream_container_cuda(self%cuda_interface, ibox)

                    ! the order number of box in domain
                    box_number = ibox - domain(1) + 1

                    ! set the stream container for the nearfield operator
                    !call self%nearfield_coulomb_operator(box_number)%set_stream_container(box_stream_container)

                    
                    ! init the cuda of the nearfield operators
                    call coulomb_operator%nearfield_coulomb_operator(box_number)%cuda_init()
                end do
        end select

        ! copy the input & output cuda pointers to the coulomb operator
        self%coulomb_operator%input_cuda_cube  = self%input_cuda_cube
        self%coulomb_operator%output_cuda_cube = self%output_cuda_cube
        self%coulomb_operator%cuda_tmp1        = self%cuda_tmp1
        self%coulomb_operator%cuda_tmp2        = self%cuda_tmp2


        ! mark coulomb operator and self to be inited
        self%coulomb_operator%cuda_inited      = .TRUE.
        self%cuda_inited = .TRUE.
    end subroutine

    subroutine GBFMMHelmholtz3D_cuda_destroy_harmonics(self)
        class(GBFMMHelmholtz3D), intent(inout)  :: self

        call GBFMMHelmholtz3D_destroy_harmonics_cuda(self%get_cuda_interface())
    end subroutine

    subroutine GBFMMHelmholtz3D_cuda_init_harmonics(self)
        class(GBFMMHelmholtz3D), intent(inout)  :: self

        call GBFMMHelmholtz3D_init_harmonics_cuda(self%get_cuda_interface())
    end subroutine
#endif

    subroutine GBFMMHelmholtz3D_set_energy(self, energy)
        class(GBFMMHelmholtz3D), intent(inout)          :: self  
        real(REAL64),       intent(in)                  :: energy

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
#ifdef HAVE_CUDA
        if (self%cuda_inited) then
            call GBFMMHelmholtz3D_set_energy_cuda(self%cuda_interface, energy)
        end if
#endif


        deallocate(weights)
        deallocate(tpoints)
    end subroutine  

    subroutine GBFMMHelmholtz3D_operate(self, func, new, cell_limits, only_cube)
        class(GBFMMHelmholtz3D)                      :: self 
        !> The input function. Note this should be a multiple of function and
        !! corresponding potential times two
        class(Function3D),  intent(in)               :: func  
        ! Result function, is of type Function3D
        class(Function3D), allocatable, intent(inout), target:: new
        !> Limits in the area in which the operator is applied. NOTE: not used in this operator
        integer, intent(in), optional                :: cell_limits(2, 3)
        !> if only cube is taken into account.
        logical, intent(in), optional                :: only_cube
        type(Bubbles)                                :: result_bubbles
        integer                                      :: domain(2), cube_limits(2, 3), box_limits(2, 3), ibox, &
                                                        lmax, nearfield_limits(2, 3), thread_id, level
        logical                                      :: operate_bubbles
        real(REAL64)                                 :: temp, nearfield_time, farfield_time, bubbles_time
        type(Function3D), target                     :: result_function

        real(REAL64), allocatable                    :: injection(:, :, :)

#ifdef HAVE_CUDA
        type(CudaCube)                :: cuda_injection
#endif
        
        ! get the boxes in this processors domain at the maximum level of recursion
        domain = self%input_parallel_info%get_domain()
        


        
        ! allocate space for local expansions
        allocate(self%local_expansions((self%farfield_potential_lmax + 1) ** 2, &
                                        self%input_parallel_info%get_total_box_count()), source = 0.0d0)
        
        call bigben%split("Calculate GBFMM Helmholtz Potential")
         
        operate_bubbles = .TRUE.
        if (present(only_cube)) then
            operate_bubbles = .not. only_cube
        end if

        if (.not. allocated(new)) then
            ! Initialize the result object. This operator's result object always has the same 
            ! parallelization as the operated object 'func'.
            allocate(Function3D :: new) 
            call new%init_explicit( self%output_parallel_info, &
                                    label=trim(func%label)//".h",   &
                                    type = F3D_TYPE_CUSP )
            new%bubbles = Bubbles(func%bubbles)
        else
            new%cube = 0.0d0
            new%grid => self%output_parallel_info%get_grid()
            new%bubbles = 0.0d0
        end if

#ifdef HAVE_CUDA    
        call bigben%split("Upload to CUDA")
        
        ! set the input and output cubes, upload the input cube and set the output cube to zero 
        call self%input_cuda_cube%set_host(func%cube)
        call self%output_cuda_cube%set_host(new%cube)
        call self%input_cuda_cube%upload()
        
        call self%output_cuda_cube%set_to_zero()

        ! make sure that the input cube is successfully uploaded before starting multipole evaluation
        call CUDASync_all()
        call bigben%stop()
        call bigben%split("Multipole moments")
#endif

#ifdef HAVE_CUDA 
        call self%calculate_cube_multipoles_cuda()
#else
        call self%calculate_cube_multipoles_cpu(func, no_cuda = .FALSE.)
#endif
           
        ! communicate the multipoles to all the processors that need them
        call self%input_parallel_info%communicate_matrix(self%multipole_moments, self%input_parallel_info%maxlevel)

        level = 1
        ! calculate the needed parent level multipoles by translating the maxlevel multipoles
        ! to parent level and summing them together
        do level = self%input_parallel_info%maxlevel - 1, self%input_parallel_info%start_level, -1
            call self%translate_multipoles(self%multipole_moments, level)
            call self%input_parallel_info%communicate_matrix(self%multipole_moments, level)
        end do

        
        if (operate_bubbles) then
            result_bubbles = Bubbles(func%bubbles)

            allocate(injection(self%coulomb_operator%gridout%axis(X_)%get_shape(),  &
                               self%coulomb_operator%gridout%axis(Y_)%get_shape(), &
                               self%coulomb_operator%gridout%axis(Z_)%get_shape()), source = 0.0d0)
        end if
        
        call bigben%stop()
#ifdef HAVE_OMP
        call bigben%split("Others")
        !$OMP PARALLEL PRIVATE(thread_id) NUM_THREADS(3)
            thread_id = omp_get_thread_num()
            if (thread_id == 1) then
                call self%operate_nearfield(func, new)
            ! NOTE: as bubbles is doing communication, we must set the thread that operates 
            !       the bubbles to be 0
            else if (thread_id == 2) then
                !call omp_set_num_threads(OMP_GLOBAL_THREAD_COUNT-1 / 2)
                call self%operate_farfield(func, new)
            else
                ! Apply the bubbles
                if (operate_bubbles) then
                    !call omp_set_num_threads(OMP_GLOBAL_THREAD_COUNT-1 / 2)
                    call self%transform_bubbles_sub(func%bubbles, result_bubbles)
                end if
            end if
        !$OMP END PARALLEL
        ! Communicate the bubbles
        if (operate_bubbles) then
            call new%parallelization_info%communicate_bubbles(result_bubbles)
            call new%bubbles%copy_content(result_bubbles)
        end if
        call bigben%stop()
#else
        call self%operate_nearfield(func, new)
        call self%operate_farfield(func, new)
        if (operate_bubbles) then
            call self%transform_bubbles_sub(func%bubbles, result_bubbles)
            call new%parallelization_info%communicate_bubbles(result_bubbles)
            call new%bubbles%copy_content(result_bubbles)
        end if
#endif
        
#ifdef HAVE_CUDA  
        call CUDASync_all()
#endif
        call bigben%split("Inject extra and simplify")
        call self%simplify(new)

        
#ifdef HAVE_CUDA  
        call CUDASync_all()
        call self%output_cuda_cube%download() 

        ! unset the host cubes from the input_cube structures
        call self%input_cuda_cube%unset_host()
        call self%output_cuda_cube%unset_host()
        call CUDASync_all()
#endif
        call bigben%stop()
        
        deallocate(self%local_expansions)

        if (operate_bubbles) call new%inject_extra_bubbles(self%lmax, injected_bubbles = result_bubbles)

        call result_bubbles%destroy()

        
        ! communicate the needed data (nearfield cube borders) with other processors 
        ! (when there is only one processor, there is no communication)
        call new%communicate_cube_borders(reversed_order = .TRUE.)
        call bigben%stop()
        call self%set_energy(0.0d0)
      
    end subroutine

    subroutine GBFMMHelmholtz3D_operate_nearfield(self, func, potential)
        class(GBFMMHelmholtz3D), intent(inout)     :: self
        !> The input function. Note this should be a multiple of function and
        !! corresponding potential times two
        class(Function3D),  intent(in)                 :: func
        class(Function3D),  allocatable, intent(inout) :: potential 
        type(GBFMMFunction3D)                          :: gbfmm_potential
        integer                                        :: cube_limits(2, 3), box_limits(2, 3), domain(2), &
                                                          ibox

        !call bigben%split("Operate nearfield")
        ! Apply the modified coulomb operator at the nearfield for the cube
        select type(coulomb_operator => self%coulomb_operator)
            type is (GBFMMCoulomb3D)
                gbfmm_potential = GBFMMFunction3D(coulomb_operator%output_parallel_info, coulomb_operator%lmax, &
                         coulomb_operator%get_result_type(), downward_pass = coulomb_operator%downward_pass, &
                         bubble_count = potential%bubbles%get_nbub_global())

                ! calculate the nearfield cube transformation
                call coulomb_operator%calculate_nearfield_potential(gbfmm_potential, &
                        potential)
        end select

        
        
#ifndef HAVE_CUDA
        ! add the nearfield cube part to the cube of the result Function3D-object (new)
        domain = self%output_parallel_info%get_domain(self%output_parallel_info%maxlevel)
        do ibox = domain(1), domain(2)
            box_limits = self%output_parallel_info%get_box_cell_index_limits(ibox,  &
                             self%output_parallel_info%maxlevel, global = .FALSE.)
            cube_limits = self%output_parallel_info%get_cube_ranges(box_limits)

            ! set the cube values of 'new' at the area of box 'ibox' with
            ! the calculated nearfield_potential values
            potential%cube(cube_limits(1, X_) : cube_limits(2, X_), &
                           cube_limits(1, Y_) : cube_limits(2, Y_), &
                           cube_limits(1, Z_) : cube_limits(2, Z_)) = &
                               gbfmm_potential%nearfield_potential(ibox)%cube
        end do
#endif

        ! deallocate memory of 'potential'
        call gbfmm_potential%destroy()
        !call bigben%stop()
    end subroutine

    subroutine GBFMMHelmholtz3D_operate_farfield(self, func, potential)
        class(GBFMMHelmholtz3D), intent(inout)     :: self
        !> The input function. Note this should be a multiple of function and
        !! corresponding potential times two
        class(Function3D),  intent(in)             :: func
        class(Function3D),  allocatable, intent(inout) :: potential 
        integer                               :: ibox, jbox, domain(2), box_limits(2, 3), &
                                                 j, box_count, l, m, n, o, p, q, level, farfield_box_limits(2, 3)
        integer,      allocatable             :: farfield_indices(:), parent_farfield_indices(:)
        real(REAL64), allocatable             :: local_expansion(:, :),  &
                                                 farfield_box_centers(:, :), farfield_multipole_moments(:, :), &
                                                 summed_local_expansion(:)
        real(REAL64)                          :: box_center(3), box_center_farfield(3), kappa, startt, endt, endt2
        real(REAL64), allocatable             :: integrated_cube(:, :, :)
        real(REAL64), allocatable, target     :: result_cube(:, :, :)
        real(REAL64), pointer                 :: result_cube_pointer(:, :, :)
        ! relocated positions in x, y, and z directions
        real(REAL64), allocatable             :: gridpoints_x(:)
        real(REAL64), allocatable             :: gridpoints_y(:)
        real(REAL64), allocatable             :: gridpoints_z(:)

        type(SecondModSphBesselCollection)    :: second_bessels

        ! Limits of the operated area in grid points
        integer                               :: cube_limits(2, 3)
        type(RealSphericalHarmonics)          :: harmo
        type(Grid3D)                          :: grid
        real(REAL64)                          :: normalization_factor

        type(HelmholtzMultipoleConverter)     :: converter

        ! evaluate the kappa parameter of the Helmholtz kernel
        kappa = sqrt(-2.0d0*self%energy)
        

        ! get the boxes in this processors domain at the maximum level of recursion
        domain = self%input_parallel_info%get_domain()
        
#ifdef HAVE_CUDA 
        converter = HelmholtzMultipoleConverter(self%farfield_potential_lmax, kappa, &
                       self%farfield_potential_input_lmax, normalization = self%normalization)
#else

        
        ! initialize bessels objects and the converter taking into account the need
        if (self%multipole_evaluation) then
            second_bessels = SecondModSphBesselCollection(self%farfield_potential_lmax)
        else 
            converter = HelmholtzMultipoleConverter(self%farfield_potential_lmax, kappa, &
                       self%farfield_potential_input_lmax, normalization = self%normalization)
        end if

        allocate(result_cube(size(potential%cube, 1), size(potential%cube, 2), & 
                             size(potential%cube, 3)), source = 0.0d0)

#endif
        box_center = 0.0

        harmo=RealSphericalHarmonics(self%farfield_potential_lmax, normalization = self%normalization)
        do level = 2, self%input_parallel_info%maxlevel
            domain = self%input_parallel_info%get_domain(level)
            do ibox = domain(1), domain(2)
                
                ! evaluate box center and limits in coordinates (box_limits) and in cube indices
                box_center = self%input_parallel_info%get_box_center(ibox, level)
                box_limits = self%input_parallel_info%get_box_cell_index_limits(ibox,  &
                                level)
                cube_limits = self%input_parallel_info%get_cube_ranges(box_limits)
                grid = func%grid%get_subgrid(box_limits)


                !  in naive implementation the farfield indicies should 
                ! contain every other box but the current box and the ones in the nearfield)
                ! evaluation_type = 3 is the naive implementation
                if (self%naive) then
                    farfield_indices = self%input_parallel_info%get_local_farfield_indices(ibox,  &
                                        self%input_parallel_info%maxlevel, evaluation_type = 3)

                    allocate(farfield_box_centers(3, size(farfield_indices)))
                    allocate(farfield_multipole_moments((self%farfield_potential_input_lmax + 1) ** 2, &
                                          size(farfield_indices)))
                ! retrieve helmholtz farfield indices. In adaptive mode the first set of indices
                ! will contain boxes in local farfield which parents contain nearest neighbors 
                else if (self%input_parallel_info%start_level == 1) then
                    farfield_indices = self%input_parallel_info%get_local_farfield_indices(ibox, level, 1)

                    
                    ! get the parent level boxes that are neighbors of the parent of the 'ibox' but do not contain
                    ! nearest neighbors of 'ibox'
                    parent_farfield_indices = self%input_parallel_info%get_local_farfield_indices(ibox, level, 2)
 
                    allocate(farfield_box_centers(3, size(farfield_indices) + size(parent_farfield_indices)))
                    allocate(farfield_multipole_moments((self%farfield_potential_input_lmax + 1) ** 2, &
                                          size(farfield_indices) + size(parent_farfield_indices)))
                
                ! otherwise there will be all boxes in the local farfield regardless of if their
                ! parents contain nearest neighbors
                else
                    farfield_indices = self%input_parallel_info%get_local_farfield_indices(ibox, level)
                    allocate(farfield_box_centers(3, size(farfield_indices)))
                    allocate(farfield_multipole_moments((self%farfield_potential_input_lmax + 1) ** 2, &
                                          size(farfield_indices)))
                end if
#ifndef HAVE_CUDA
                if (self%multipole_evaluation) then
                    result_cube(cube_limits(1, X_) : cube_limits(2, X_), &
                                cube_limits(1, Y_) : cube_limits(2, Y_), &
                                cube_limits(1, Z_) : cube_limits(2, Z_)) = 0.0d0
                end if
#endif
                ! evaluate the interaction of the box 'ibox' with each farfield box.
                ! The evaluation is performed in such manner that the origo is set to 
                ! the center of the farfield box. Thus, for each farfield box, the spherical harmonics
                ! and K(kappa r) have to be evaluated separately.
                do j = 1, size(farfield_indices)
                    jbox = farfield_indices(j)
                    ! get the center of the farfield box
                    box_center_farfield = self%input_parallel_info%get_box_center(jbox, level)

                    
#ifndef HAVE_CUDA
                    ! if the self%multipole_evaluation is .TRUE. (not recommended for anything but testing 
                    ! purposes) we will calculate the potential box by box
                    if (self%multipole_evaluation) then
                        integrated_cube = self%evaluate_potential_me(harmo, second_bessels, grid, box_center_farfield, &
                                        self%multipole_moments(:, jbox + self%input_parallel_info%get_level_offset(level)), kappa)
                        result_cube(cube_limits(1, X_) : cube_limits(2, X_), &
                                    cube_limits(1, Y_) : cube_limits(2, Y_), &
                                    cube_limits(1, Z_) : cube_limits(2, Z_)) = &
                                    result_cube(cube_limits(1, X_) : cube_limits(2, X_), &
                                                cube_limits(1, Y_) : cube_limits(2, Y_), &
                                                cube_limits(1, Z_) : cube_limits(2, Z_)) + &
                                    integrated_cube
                        !print *, "cube multipole result", 8*kappa*integrated_cube(1, 1, 1)
                        farfield_box_limits = self%input_parallel_info%get_box_cell_index_limits(jbox, level)
                        gridpoints_x = grid%axis(X_)%get_coord()
                        gridpoints_y = grid%axis(Y_)%get_coord()
                        gridpoints_z = grid%axis(Z_)%get_coord()
                        !print *, "starting evaluation for point: ", [gridpoints_x(1), gridpoints_y(1), gridpoints_z(1)] 
                        !deallocate(cube)
                        deallocate(integrated_cube, gridpoints_x, gridpoints_y, gridpoints_z)
                    ! otherwise we will store the farfield centers and multipole moments to specific arrays
                    ! for convertion
                    else
#endif
                        farfield_box_centers(:, j) = box_center_farfield
                        farfield_multipole_moments(:, j) = self%multipole_moments(:,  &
                            jbox + self%input_parallel_info%get_level_offset(level))
#ifndef HAVE_CUDA
                    end if
#endif
                    
                end do 
                !print *, "comparison", comparison_potential, "multipole value", &
                !        8*kappa*result_cube(cube_limits(1, X_), &
                !                    cube_limits(1, Y_), &
                !                    cube_limits(1, Z_))

                ! if we are in the adaptive mode, there are additional 
                ! boxes to evaluate from the parent level
                if (.NOT. self%naive .and. self%input_parallel_info%start_level == 1) then
               

                    ! evaluate the interaction of the box 'ibox' with each farfield box.
                    ! The evaluation is performed in such manner that the origo is set to 
                    ! the center of the farfield box. Thus, for each farfield box, the spherical harmonics
                    ! and K(kappa r) have to be evaluated separately.
                    do j = 1, size(parent_farfield_indices)
                        jbox = parent_farfield_indices(j)
                        ! get the center of the farfield box
                        box_center_farfield = self%input_parallel_info%get_box_center(jbox, level-1)
                        
                        
#ifndef HAVE_CUDA
                        ! if the self%multipole_evaluation is .TRUE. (not recommended for anything but testing 
                        ! purposes) we will calculate the potential box by box
                        if (self%multipole_evaluation) then
                            result_cube(cube_limits(1, X_) : cube_limits(2, X_), &
                                        cube_limits(1, Y_) : cube_limits(2, Y_), &
                                        cube_limits(1, Z_) : cube_limits(2, Z_)) = &
                                        result_cube(cube_limits(1, X_) : cube_limits(2, X_), &
                                                    cube_limits(1, Y_) : cube_limits(2, Y_), &
                                                    cube_limits(1, Z_) : cube_limits(2, Z_)) + &
                                        self%evaluate_potential_me(harmo, second_bessels, grid, box_center_farfield, &
                                            self%multipole_moments &
                                                (:, jbox + self%input_parallel_info%get_level_offset(level-1)), kappa)
                        ! otherwise we will store the farfield centers and multipole moments to specific arrays
                        ! for convertion
                        else 
#endif
                            farfield_box_centers(:, j+size(farfield_indices)) = box_center_farfield
                            farfield_multipole_moments(:, j+size(farfield_indices)) = self%multipole_moments(:,  &
                                jbox + self%input_parallel_info%get_level_offset(level-1))
#ifndef HAVE_CUDA
                        end if
#endif

                        
                    end do 
                    deallocate(parent_farfield_indices)
                end if

                ! if the multipole_evaluation is not on (default) the farfield potential evaluation is done
                ! using the local expansion, to which the farfield multipole moments are converted to
                ! this procedure saves a huge amount of time, and it should be used always, except 
                ! when the code is tested
                if (.not. self%multipole_evaluation) then
                    local_expansion = converter%translate(farfield_multipole_moments, farfield_box_centers, box_center)
                    ! sum the converted local expansions together
                    do j = 1, size(farfield_multipole_moments, 2)
                        self%local_expansions(:, ibox) = self%local_expansions(:, ibox) + local_expansion(:, j)
                    end do
                    deallocate(local_expansion)
                end if
                
                  
                call grid%destroy()
                deallocate(farfield_multipole_moments, farfield_box_centers, farfield_indices)
                
                !print *, "ibox", ibox, "farfield time:", endt-startt, "convert time", endt2 - startt
            end do
            
        end do
        
#ifndef HAVE_CUDA
        if (self%multipole_evaluation) then
            result_cube_pointer => result_cube
            call self%output_parallel_info%communicate_cube_borders(result_cube_pointer, reversed_order = .TRUE.)
            potential%cube = potential%cube + result_cube
            call second_bessels%destroy()
        end if
#endif
        call harmo%destroy()
        call converter%destroy()
    end subroutine

    subroutine GBFMMHelmholtz3D_simplify(self, potential)
        class(GBFMMHelmholtz3D), intent(inout)     :: self
        class(Function3D),  allocatable, intent(inout) :: potential 
        
        real(REAL64), allocatable, target     :: result_cube(:, :, :)
        real(REAL64), pointer                 :: result_cube_pointer(:, :, :)

#ifdef HAVE_CUDA
        call self%evaluate_potentials_le_cuda(self%local_expansions)
#else
        ! evaluate the results if we are not running with multipole evaluation flag on
        if (.not. self%multipole_evaluation) then
            ! allocate space for a result cube
            allocate(result_cube(size(potential%cube, 1), size(potential%cube, 2), & 
                                size(potential%cube, 3)), source = 0.0d0)

            call self%evaluate_potentials_le(potential%grid, self%local_expansions, sqrt(-2.0d0*self%energy), &
                     result_cube, no_cuda = .FALSE.)

            result_cube_pointer => result_cube
            call self%output_parallel_info%communicate_cube_borders(result_cube_pointer, reversed_order = .TRUE.)
            potential%cube = potential%cube + result_cube
            deallocate(result_cube)
            nullify(result_cube_pointer)
        end if
#endif
    end subroutine

    !> Evaluate Helmholtz potential from 'local_expansion' centered at 'reference_point'
    !! at every point of the 'grid'. This function makes the choice between CUDA and CPU 
    !! options.
    subroutine GBFMMHelmholtz3D_evaluate_potentials_le(self, grid, &
                 local_expansion, kappa, result_cube, no_cuda)
        class(GBFMMHelmholtz3D),           intent(in)        :: self    
        !> Grid object describing the evaluation area.
        type(Grid3D),                      intent(in)        :: grid
        !> The origo of the evaluation area, used in spherical harmonics and bessel value
        !> The evaluated local expansion
        real(REAL64),                      intent(in)        :: local_expansion(:, :)
        !> The kappa parameter of the Helmholtz-kernel
        real(REAL64),                      intent(in)        :: kappa
        !> The result cube
        real(REAL64),                      intent(inout)     :: result_cube(grid%axis(X_)%get_shape(), &
                                                                            grid%axis(Y_)%get_shape(), &
                                                                            grid%axis(Z_)%get_shape())
        !> Force disable cuda usage (only valid if cuda is enabled by the device)
        logical, optional,                 intent(in)        :: no_cuda

        call self%evaluate_potentials_le_cpu(grid,  local_expansion, kappa, result_cube)
    end subroutine

    !> Evaluate Helmholtz potential from 'local_expansion' centered at 'reference_point'
    !! at every point of the 'grid' using CPU.
    subroutine GBFMMHelmholtz3D_evaluate_potentials_le_cpu(self, grid,  &
                                                      local_expansion, kappa, result_cube, no_cuda)
        class(GBFMMHelmholtz3D),           intent(in)        :: self   
        !> Grid object describing the evaluation area.
        type(Grid3D),                      intent(in)        :: grid
        !> The origo of the evaluation area, used in spherical harmonics and bessel value
        !> The evaluated local expansion
        real(REAL64),                      intent(in)        :: local_expansion(:, :)
        !> The kappa parameter of the Helmholtz-kernel
        real(REAL64),                      intent(in)        :: kappa
        !> The result cube
        real(REAL64),                      intent(inout)     :: result_cube(grid%axis(X_)%get_shape(), &
                                                                            grid%axis(Y_)%get_shape(), &
                                                                            grid%axis(Z_)%get_shape())
        !> Force disable cuda usage (only valid if cuda is enabled by the device)
        logical, optional,                 intent(in)        :: no_cuda
        !> The addition cube
        real(REAL64)                                         :: addition_cube(grid%axis(X_)%get_shape(), &
                                                                            grid%axis(Y_)%get_shape(), &
                                                                            grid%axis(Z_)%get_shape())
        
        !> FirstModSphBesselcubeCollection object initialized to lmin: 0, lmax: self%farfield_potential_lmax
        type(FirstModSphBesselCubeCollection)                :: first_bessels 
        !> RealSphericalCubeHarmonics object that will be initialized to lmin: 0, lmax: self%farfield_potential_lmax
        type(RealSphericalCubeHarmonics)                     :: harmo    
        real(REAL64), allocatable                            :: bessel_values(:, :, :, :) 
        real(REAL64), allocatable                            :: spherical_harmonics_cube(:, :, :, :)
        real(REAL64)                                         :: normalization_factor
        integer                                              :: ibox, l, m, domain(2), box_limits(2, 3), cube_limits(2, 3)
        !> Grid object describing the evaluation area.
        type(Grid3D)                                         :: subgrid
        !! determination
        real(REAL64)                                         :: box_center(3)

        addition_cube = 0.0d0
        domain = self%output_parallel_info%get_domain(self%output_parallel_info%maxlevel)

        do ibox = domain(1), domain(2)
            ! evaluate box center and limits in coordinates (box_limits) and in cube indices
            box_center = self%output_parallel_info%get_box_center(ibox, self%output_parallel_info%maxlevel)
            box_limits = self%output_parallel_info%get_box_cell_index_limits(ibox,  &
                            self%output_parallel_info%maxlevel)
            cube_limits = self%output_parallel_info%get_cube_ranges(box_limits)
            subgrid = grid%get_subgrid(box_limits)

            if (ibox == domain(1)) then
                ! Initialize First Kind of Modified Spherical Bessel Function (I_l+½) and Spherical Harmonics
                first_bessels = FirstModSphBesselCubeCollection(0, self%farfield_potential_lmax, subgrid%get_shape())
                harmo=RealSphericalCubeHarmonics(0, self%farfield_potential_lmax,  &
                                                 self%normalization, subgrid%get_shape())
                
                ! Evaluate bessel & spherical harmonics values on the subgrid
                bessel_values = first_bessels%eval_grid(subgrid, box_center, kappa, no_cuda = no_cuda)
                spherical_harmonics_cube = harmo%eval_grid(subgrid, box_center, no_cuda = no_cuda)

            end if

            normalization_factor = 1.0d0

            addition_cube(cube_limits(1, X_) : cube_limits(2, X_), &
                          cube_limits(1, Y_) : cube_limits(2, Y_), &
                          cube_limits(1, Z_) : cube_limits(2, Z_)) = 0.0d0

            ! go through all quantum number pairs (l, m) and multiply the spherical harmonics with first
            ! modified bessel function values and the local expansion values
            do l = 0, self%farfield_potential_lmax
                if (self%normalization /= 2) then
                    normalization_factor = (2*l+1)/dble(4*pi)
                end if
                do m = -l, l
                    addition_cube(cube_limits(1, X_) : cube_limits(2, X_), &
                            cube_limits(1, Y_) : cube_limits(2, Y_), &
                            cube_limits(1, Z_) : cube_limits(2, Z_)) = &
                        addition_cube(cube_limits(1, X_) : cube_limits(2, X_), &
                            cube_limits(1, Y_) : cube_limits(2, Y_), &
                            cube_limits(1, Z_) : cube_limits(2, Z_)) + &
                        bessel_values(:, :, :, l+1) * spherical_harmonics_cube(:, :, :, lm_map(l, m)) * &
                        local_expansion(lm_map(l, m), ibox) * normalization_factor
                end do
            end do  
            call subgrid%destroy()
        end do
        deallocate(bessel_values, spherical_harmonics_cube)
        call first_bessels%destroy()
        call harmo%destroy()
        result_cube = result_cube + 8.0d0*kappa*addition_cube
    end subroutine


#ifdef HAVE_CUDA
    !> Evaluate Helmholtz potential from 'local_expansion' centered at 'reference_point'
    !! at every point of the 'grid' using CUDA.
    subroutine GBFMMHelmholtz3D_evaluate_potentials_le_cuda(self, local_expansion)
        class(GBFMMHelmholtz3D),           intent(in)    :: self
        !> The local expansion being evaluated
        real(REAL64),                      intent(in)    :: local_expansion(:, :)

        call GBFMMHelmholtz3D_evaluate_potential_le_cuda(self%cuda_interface, &
                 local_expansion, self%output_cuda_cube%cuda_interface)
        
    end subroutine
#endif

    !> Evaluate Helmholtz potential from 'multipole_expansion' centered at 'reference_point'
    !! at every point of the 'grid'. This function makes the choice between CUDA and CPU 
    !! options.
    function GBFMMHelmholtz3D_evaluate_potential_me(self, harmo, second_bessels, grid, reference_point,  &
                 multipole_expansion, kappa, no_cuda) result(result_cube) 
        class(GBFMMHelmholtz3D),           intent(in)    :: self
        !> RealSphericalHarmonics object initialized to lmin: 0, lmax: self%farfield_potential_lmax
        type(RealSphericalHarmonics),      intent(in)    :: harmo
        !> FirstModSphBesselCollection object initialized to lmin: 0, lmax: self%farfield_potential_lmax
        type(SecondModSphBesselCollection), intent(in)    :: second_bessels
        !> Grid object describing the evaluation area.
        type(Grid3D),                      intent(inout) :: grid
        !> The origo of the evaluation area, used in spherical harmonics and bessel value
        !! determination
        real(REAL64),                      intent(in)    :: reference_point(3)
        !> The local expansion being evaluated
        real(REAL64),                      intent(in)    :: multipole_expansion((self%farfield_potential_lmax+1)**2)
        !> The kappa parameter of the Helmholtz-kernel
        real(REAL64),                      intent(in)    :: kappa
        !> The result cube
        real(REAL64)                                     :: result_cube(grid%axis(X_)%get_shape(), &
                                                                        grid%axis(Y_)%get_shape(), &
                                                                        grid%axis(Z_)%get_shape())
        !> Force disable cuda usage (only valid if cuda is enabled by the device)
        logical, optional,                 intent(in)    :: no_cuda

!#ifdef HAVE_CUDA
!        if (present(no_cuda)) then
!            if (no_cuda) then
!                call self%evaluate_potential_le_cpu(harmo, first_bessels, grid, reference_point, result_cube)
!            else 
!                call self%evaluate_potential_le_cuda(harmo, grid, reference_point, result_cube)   
!            end if
!        else
!            call self%evaluate_potential_le_cuda(harmo, grid, reference_point, result_cube)   
!        end if
!#else
         call self%evaluate_potential_me_cpu(harmo, second_bessels, grid, reference_point, &
                                             multipole_expansion, kappa, result_cube)
!#endif
    end function

    !> Evaluate Helmholtz potential from 'local_expansion' centered at 'reference_point'
    !! at every point of the 'grid' using CPU.
    subroutine GBFMMHelmholtz3D_evaluate_potential_me_cpu(self, harmo, second_bessels, grid, reference_point, &
                                                          multipole_expansion, kappa, result_cube)
        class(GBFMMHelmholtz3D),           intent(in)    :: self
        !> RealSphericalHarmonics object initialized to lmin: 0, lmax: self%farfield_potential_lmax
        type(RealSphericalHarmonics),      intent(in)    :: harmo
        !> FirstModSphBesselCollection object initialized to lmin: 0, lmax: self%farfield_potential_lmax
        type(SecondModSphBesselCollection), intent(in)   :: second_bessels
        !> Grid object describing the evaluation area.
        type(Grid3D),                      intent(in)    :: grid
        !> The center of the multipole moment, used in spherical harmonics and bessel value
        !! determination
        real(REAL64),                      intent(in)    :: reference_point(3)
        !> The evaluated local expansion
        real(REAL64),                      intent(in)    :: multipole_expansion((self%farfield_potential_lmax+1)**2)
        !> The kappa parameter of the Helmholtz-kernel
        real(REAL64),                      intent(in)    :: kappa
        !> The result cube
        real(REAL64),                      intent(inout) :: result_cube(grid%axis(X_)%get_shape(), &
                                                                        grid%axis(Y_)%get_shape(), &
                                                                        grid%axis(Z_)%get_shape())

        real(REAL64)                                     :: bessel_values(grid%axis(X_)%get_shape(), grid%axis(Y_)%get_shape(), &
                                                                        grid%axis(Z_)%get_shape(), 0:self%farfield_potential_lmax)
        real(REAL64)                                     :: spherical_harmonics_cube(grid%axis(X_)%get_shape(), &
                                                                                      grid%axis(Y_)%get_shape(), &
                                                                                      grid%axis(Z_)%get_shape(), &
                                                                                      (self%farfield_potential_lmax+1)**2)
        real(REAL64)                                     :: normalization_factor
        integer                                          :: l, m

        
        result_cube = 0.0d0

        ! evaluate first modified spherical bessel function values at all cube positions 
        bessel_values = second_bessels%eval_grid(grid, reference_point, kappa)


        ! evaluate spherical harmonics at cube positions and reshape the result as cube shaped arrays
        ! for each (angular momentum) l, m pair
        spherical_harmonics_cube = harmo%eval_grid(grid, reference_point)
        normalization_factor = 1.0d0
        ! go through all quantum number pairs (l, m) and multiply the spherical harmonics with first
        ! modified bessel function values and the local expansion values
        do l = 0, self%farfield_potential_lmax
            if (self%normalization /= 2) then
                normalization_factor = (2*l+1)/dble(4*pi)
            end if
            do m = -l, l
                result_cube = result_cube + &
                    bessel_values(:, :, :, l) * spherical_harmonics_cube(:, :, :, lm_map(l, m)) * &
                    multipole_expansion(lm_map(l, m)) * normalization_factor
            end do
        end do  
    end subroutine

    

    !> This function translates previously calculated multipoles to the parent level 
    subroutine GBFMMHelmholtz3D_translate_multipoles(self, multipole_moments, parent_level)
        class(GBFMMHelmholtz3D), intent(in)    :: self
        !> multipole moments in a matrix that contains has (self%lmax + 1)**2 rows (first index)
        !! and the number of box (including level offset) in the columns
        real(REAL64),            intent(inout) :: multipole_moments(:, :)   
        !> number of parent level to which its children are translated to  
        integer,                 intent(in)    :: parent_level

        type(HelmholtzMultipoleTranslator)     :: translator
        ! Temporary variable to store child multipole moments
        real(REAL64), allocatable              :: child_multipole_moments(:, :), &
                                                  translated_multipole_moments(:, :)

        integer                                :: ibox, i
        integer                                :: child_box, parent_box
        integer, allocatable                   :: child_indices(:)
        integer                                :: domain(2)

        real(REAL64), allocatable              :: child_positions(:, :)
      
        real(REAL64)                           :: parent_position(3)

        translator = HelmholtzMultipoleTranslator(self%farfield_potential_input_lmax, sqrt(-2.0d0*self%energy), &
                                                  self%farfield_potential_input_lmax, normalization = self%normalization)

        ! starting and ending indices of this processors domain
        domain = self%input_parallel_info%get_domain(parent_level)
        ibox = domain(1)

        do ibox = domain(1), domain(2)
            !print *, "Translating multipoles for ibox", ibox, "on level", parent_level
            ! the parent box index with the offset caused by other levels
            parent_box = self%input_parallel_info%get_level_offset(parent_level) + ibox
            parent_position = self%input_parallel_info%get_box_center(ibox, parent_level) 

            child_indices = self%input_parallel_info%get_child_indices(ibox, parent_level)
            allocate(child_multipole_moments((self%farfield_potential_input_lmax + 1)**2, size(child_indices)))
            allocate(child_positions(3, size(child_indices)))
            do i = 1, size(child_indices)
                child_box = child_indices(i)
                child_multipole_moments(:, i) = &
                    multipole_moments(:, self%input_parallel_info%get_level_offset(parent_level + 1) + child_box)
                child_positions(:, i) = self%input_parallel_info%get_box_center(child_box, parent_level+1)
            end do
            allocate(translated_multipole_moments((self%farfield_potential_input_lmax + 1)**2, size(child_indices)))
            translated_multipole_moments = translator%translate(child_multipole_moments, &
                child_positions, parent_position)


            multipole_moments(:, parent_box) = 0.0d0
            do i = 1, size(translated_multipole_moments, 2)
                multipole_moments(:, parent_box) =     &
                    multipole_moments(:, parent_box) + translated_multipole_moments(:, i)
            end do
            deallocate(translated_multipole_moments)
            deallocate(child_multipole_moments)           
            deallocate(child_indices)
            deallocate(child_positions)
        end do
        call translator%destroy()
        return
    end subroutine

    !> This function translates previously calculated multipoles to the parent level 
    subroutine GBFMMHelmholtz3D_translate_complex_multipoles(self, multipole_moments, parent_level)
        class(GBFMMHelmholtz3D), intent(in)    :: self
        !> multipole moments in a matrix that contains has (self%lmax + 1)**2 rows (first index)
        !! and the number of box (including level offset) in the columns
        complex*16,              intent(inout) :: multipole_moments(:, :)   
        !> number of parent level to which its children are translated to  
        integer,                 intent(in)    :: parent_level

        type(HelmholtzMultipoleTranslator)     :: translator
        ! Temporary variable to store child multipole moments
        complex*16, allocatable                :: child_multipole_moments(:, :), &
                                                  translated_multipole_moments(:, :)

        integer                                :: ibox, i
        integer                                :: child_box, parent_box
        integer, allocatable                   :: child_indices(:)
        integer                                :: domain(2)

        real(REAL64), allocatable              :: child_positions(:, :)
      
        real(REAL64)                           :: parent_position(3)

        translator = HelmholtzMultipoleTranslator(self%farfield_potential_lmax, sqrt(-2.0d0*self%energy), &
                                                  self%farfield_potential_lmax, normalization = self%normalization)

        ! starting and ending indices of this processors domain
        domain = self%input_parallel_info%get_domain(parent_level)
         
        do ibox = domain(1), domain(2)
            !print *, "Translating multipoles for ibox", ibox, "on level", parent_level
            ! the parent box index with the offset caused by other levels
            parent_box = self%input_parallel_info%get_level_offset(parent_level) + ibox
            parent_position = self%input_parallel_info%get_box_center(ibox, parent_level) 

            child_indices = self%input_parallel_info%get_child_indices(ibox, parent_level)
            allocate(child_multipole_moments((self%farfield_potential_lmax + 1)**2, size(child_indices)))
            allocate(child_positions(3, size(child_indices)))
            do i = 1, size(child_indices)
                child_box = child_indices(i)
                child_multipole_moments(:, i) = &
                    multipole_moments(:, self%input_parallel_info%get_level_offset(parent_level + 1) + child_box)
                child_positions(:, i) = self%input_parallel_info%get_box_center(child_box, parent_level+1)
            end do
            allocate(translated_multipole_moments((self%farfield_potential_lmax + 1)**2, size(child_indices)))
            translated_multipole_moments = translator%translate(child_multipole_moments, &
                child_positions, parent_position)


            multipole_moments(:, parent_box) = 0.0d0
            do i = 1, size(translated_multipole_moments, 2)
                multipole_moments(:, parent_box) =     &
                    multipole_moments(:, parent_box) + translated_multipole_moments(:, i)
            end do
            deallocate(translated_multipole_moments)
            deallocate(child_multipole_moments)           
            deallocate(child_indices)
            deallocate(child_positions)
        end do
        call translator%destroy()
        return
    end subroutine

    !> Compute the Helmholtz Multipole moments up to order `self%lmax`.
    function GBFMMHelmholtz3D_get_complex_cube_multipoles(self, func, reference_point, limits) &
                 result(complex_multipole_moments)
        class(GBFMMHelmholtz3D), intent(in) :: self
        !> The input function. Note this should be a multiple of function and
        !! corresponding potential times two
        class(Function3D),  intent(in)       :: func  
        !> Reference point, where the multipoles are evaluated
        real(REAL64),      intent(in)       :: reference_point(3)
        !> Limits of the multipole evaluation in the grid
        integer, optional, intent(in)       :: limits(2, 3)

        ! result object containing cube 'helmholtz multipole moments' at the end of this function
        complex*16                          :: complex_multipole_moments((self%farfield_potential_lmax+1)**2)

        real(REAL64), allocatable           :: cube  (:,:,:), kappa_r(:, :, :), r(:, :, :), &
                                               cube_pos(:, :, :, :), &
                                               bessel_values(:, :, :, :)
        complex*16,   allocatable           :: complex_spherical_harmonics_cube(:, :, :, :)

        ! integrals x, y, and z directions
        real(REAL64), allocatable           :: ints_x(:)
        real(REAL64), allocatable           :: ints_y(:)
        real(REAL64), allocatable           :: ints_z(:)

        ! relocated positions in x, y, and z directions
        real(REAL64), allocatable       :: gridpoints_x(:)
        real(REAL64), allocatable       :: gridpoints_y(:)
        real(REAL64), allocatable       :: gridpoints_z(:)

        integer                         :: ndim(3)
        integer                         :: i,j,k,l,m,n,nlip, id

        type(ComplexSphericalHarmonics) :: harmo
        real(REAL64)                    :: coeff
        real(REAL64)                    :: kappa
        type(FirstModSphBesselCollection) :: bessels

        complex_multipole_moments = 0.d0
        harmo=ComplexSphericalHarmonics(self%farfield_potential_lmax, normalization = self%normalization)
        
        ! number of lagrange interpolation polynomials per cell
        nlip=func%grid%get_nlip()
        kappa = sqrt(-2.0d0*self%energy)

       
        if (present(limits)) then
            ndim=(limits(2, :) - limits(1, :) + (/1, 1, 1/)) * (nlip-1) + 1
            ! Note: X_ = 1, Y_ = 2, Z_ = 3
            ints_x=func%grid%axis(X_)%get_ints(limits(:, X_))
            ints_y=func%grid%axis(Y_)%get_ints(limits(:, Y_))
            ints_z=func%grid%axis(Z_)%get_ints(limits(:, Z_))

            ! get gridpoints relative to the reference_point
            gridpoints_x=func%grid%axis(X_)%get_coord(limits(:, X_)) - reference_point(X_)
            gridpoints_y=func%grid%axis(Y_)%get_coord(limits(:, Y_)) - reference_point(Y_)
            gridpoints_z=func%grid%axis(Z_)%get_coord(limits(:, Z_)) - reference_point(Z_)

            ! handle only limited part of the cube
            cube=func%cube((limits(1, X_)-1)*(nlip-1)+1: (limits(2, X_))*(nlip-1)+1,&
                           (limits(1, Y_)-1)*(nlip-1)+1: (limits(2, Y_))*(nlip-1)+1,&
                           (limits(1, Z_)-1)*(nlip-1)+1: (limits(2, Z_))*(nlip-1)+1)
        else
            ndim=func%grid%get_shape()

            ! Note: X_ = 1, Y_ = 2, Z_ = 3
            ints_x=func%grid%axis(X_)%get_ints()
            ints_y=func%grid%axis(Y_)%get_ints()
            ints_z=func%grid%axis(Z_)%get_ints()

            ! get gridpoints relative to the reference_point
            gridpoints_x=func%grid%axis(X_)%get_coord() - reference_point(X_)
            gridpoints_y=func%grid%axis(Y_)%get_coord() - reference_point(Y_)
            gridpoints_z=func%grid%axis(Z_)%get_coord() - reference_point(Z_)


            ! handle all of the cube
            cube=func%cube
 
        end if

        allocate(r(size(cube, 1), size(cube, 2), size(cube, 3)), source = 0.0d0)
        allocate(cube_pos(3, size(cube, 1), size(cube, 2), size(cube, 3)), source = 0.0d0)

        ! evaluate kappa r and positions for each grid point
        forall(k = 1 : size(cube, 3), j = 1 : size(cube, 2), i = 1 : size(cube, 1))
            r(i, j, k) = sqrt(gridpoints_x(i) ** 2 + gridpoints_y(j) ** 2 + gridpoints_z(k) ** 2 )
            cube_pos(X_, i, j, k) = gridpoints_x(i)
            cube_pos(Y_, i, j, k) = gridpoints_y(j)
            cube_pos(Z_, i, j, k) = gridpoints_z(k)
        end forall

        kappa_r = kappa * r
        deallocate(gridpoints_x, gridpoints_y, gridpoints_z)

        ! evaluate spherical harmonics at cube positions and reshape the result as cube shaped arrays
        ! for each (angular momentum) l, m pair
        complex_spherical_harmonics_cube = reshape( &
            harmo%eval( reshape( cube_pos, [3, size(cube, 1) * size(cube, 2) * size(cube, 3)] ), &
                        reshape( r, [size(cube, 1) * size(cube, 2) * size(cube, 3)])          ), &
            [size(cube, 1), size(cube, 2), size(cube, 3), (self%farfield_potential_lmax+1)**2]) 
 
        deallocate(cube_pos)
        deallocate(r)
        call harmo%destroy()

        ! Initialize First Kind of Modified Spherical Bessel Function (I_l+½) 
        bessels = FirstModSphBesselCollection(self%farfield_potential_lmax)
        allocate(bessel_values(size(cube, 1), size(cube, 2), size(cube, 3), 0:self%farfield_potential_lmax))
        bessel_values = reshape( &
                            bessels%eval(reshape(kappa_r, [size(cube, 1) * size(cube, 2) * size(cube, 3)])), &
                            [size(cube, 1), size(cube, 2), size(cube, 3), self%farfield_potential_lmax+1])
        call bessels%destroy()
        deallocate(kappa_r)

        ! Go through all spherical harmonics, calculate corresponding x, y and z axis values 
        ! for first modified spherical bessel functions and then c
        do l = 1, self%farfield_potential_lmax
            forall (m = -l : l)
                ! multiply spherical harmonics with corresponding bessel values and cube
                complex_spherical_harmonics_cube(:, :, :, lm_map(l, m)) = &
                    complex_spherical_harmonics_cube(:, :, :, lm_map(l, m)) * &
                    bessel_values(:, :, :, l) * cube
            end forall
        
            ! Finally calculate the following:
            !    \int I(k r) Y_lm f^{\Delta} dx dy dz
            do m = -l, l
                ! Integrate over x, y, and z for the previously calculated product
                complex_multipole_moments(lm_map(l, m)) = contract_complex_cube( &
                    complex_spherical_harmonics_cube(:, :, :, lm_map(l, m)), ints_x, ints_y, ints_z)
            end do
        end do
        deallocate(bessel_values)
        deallocate(cube)
        deallocate(ints_x, ints_y, ints_z)
        deallocate(complex_spherical_harmonics_cube)
    end function

    

    subroutine GBFMMHelmholtz3D_calculate_cube_multipoles_cpu(self, func, no_cuda)
        class(GBFMMHelmholtz3D), intent(inout)   :: self
        !> Input function for which the multipoles are calculated
        type(Function3D)                         :: func
        !> Force disable cuda usage (only valid if cuda is enabled by the device)
        logical, optional, intent(in)            :: no_cuda
        ! objeect for spherical harmonics evaluation
        type(RealSphericalCubeHarmonics)         :: harmo

        ! helmholtz kernel parameter    
        real(REAL64)                             :: kappa
        real(REAL64), allocatable                :: spherical_harmonics_cube(:, :, :, :), &
                                                    bessel_values(:, :, :, :)

        ! integrals x, y, and z directions
        real(REAL64), allocatable                :: ints_x(:), ints_y(:), ints_z(:)

        integer                                  :: ibox, l, m
        
        integer                                  :: domain(2), cube_limits(2, 3), box_limits(2, 3)
        real(REAL64)                             :: box_center(3)
                                                        
        type(Grid3D)                             :: subgrid
        type(FirstModSphBesselCubeCollection)    :: bessels

        ! initialize the Helmholtz-kernel parameter kappa        
        kappa = sqrt(-2.0d0*self%energy)

        ! get domain for this computational node
        domain = self%input_parallel_info%get_domain(self%input_parallel_info%maxlevel)

        do ibox = domain(1), domain(2)
            ! evaluate box center and limits in coordinates (box_limits) and in cube indices
            box_center = self%input_parallel_info%get_box_center(ibox, self%input_parallel_info%maxlevel)
            box_limits = self%input_parallel_info%get_box_cell_index_limits(ibox,  &
                            self%input_parallel_info%maxlevel)
            cube_limits = self%input_parallel_info%get_cube_ranges(box_limits)
            subgrid = func%grid%get_subgrid(box_limits)

            if (ibox == domain(1)) then
                ! Initialize First Kind of Modified Spherical Bessel Function (I_l+½) and Spherical Harmonics
                bessels = FirstModSphBesselCubeCollection(0, self%farfield_potential_input_lmax, subgrid%get_shape())
                harmo=RealSphericalCubeHarmonics(0, self%farfield_potential_input_lmax,  &
                                                 self%normalization, subgrid%get_shape())
                
                ! Evaluate bessel & spherical harmonics values on the subgrid
                bessel_values = bessels%eval_grid(subgrid, box_center, kappa, no_cuda = no_cuda)
                spherical_harmonics_cube = harmo%eval_grid(subgrid, box_center, no_cuda = no_cuda)
            end if

            ! Get integrals for each axis, note: X_ = 1, Y_ = 2, Z_ = 3
            ints_x = subgrid%axis(X_)%get_ints()
            ints_y = subgrid%axis(Y_)%get_ints()
            ints_z = subgrid%axis(Z_)%get_ints()

            ! Go through all spherical harmonics, calculate corresponding x, y and z axis values 
            ! for first modified spherical bessel functions and then c
            do l = 0, self%farfield_potential_input_lmax
    
                forall (m = -l : l)
                    ! multiply spherical harmonics with corresponding bessel values and cube
                    spherical_harmonics_cube(:, :, :, lm_map(l, m)) = &
                        spherical_harmonics_cube(:, :, :, lm_map(l, m)) * &
                        bessel_values(:, :, :, l+1) * &
                        func%cube(cube_limits(1, X_) : cube_limits(2, X_), &
                                  cube_limits(1, Y_) : cube_limits(2, Y_), &
                                  cube_limits(1, Z_) : cube_limits(2, Z_))
                end forall
            
                ! Finally calculate the following:
                !    \int I(k r) Y_lm f^{\Delta} dx dy dz
                do m = -l, l
                    ! Integrate over x, y, and z for the previously calculated product
                    self%multipole_moments(lm_map(l, m), ibox) = contract_cube(spherical_harmonics_cube(:, :, :, lm_map(l, m)), & 
                                                                    ints_x, ints_y, ints_z)
                end do
            end do
            call subgrid%destroy()
            deallocate(ints_x)
            deallocate(ints_y)
            deallocate(ints_z)
        end do
        deallocate(bessel_values)
        deallocate(spherical_harmonics_cube)
        call harmo%destroy()
        call bessels%destroy()
    end subroutine

#ifdef HAVE_CUDA

    subroutine GBFMMHelmholtz3D_calculate_cube_multipoles_cuda(self)
        class(GBFMMHelmholtz3D), intent(inout)  :: self
      
        
        !call bigben%split("Calculate Helmholtz - multipole moments")
        call GBFMMHelmholtz3D_set_energy_cuda(self%cuda_interface, self%energy)
        call CUDASync_all()
        call GBFMMHelmholtz3D_calculate_multipole_moments_cuda(self%cuda_interface, &
                 self%input_cuda_cube%cuda_interface)
        call GBFMMHelmholtz3D_download_multipole_moments_cuda(self%cuda_interface, self%multipole_moments)
        !call bigben%stop()
    end subroutine 
#endif

    !> Compute the Real Helmholtz Local expansion moments up to order `self%farfield_potential_input_lmax`, for center 'reference_point'.
    !! The multipoles are defined M_lm = \int dx dy dz Y_lm \rho, regardless of the normalization the
    !! spherical harmonics Y_lm use. 
    function GBFMMHelmholtz3D_get_cube_local_expansion(self, func, harmo, reference_point, limits) result(multipole_moments)
        class(GBFMMHelmholtz3D), intent(in) :: self
        !> The input function. Note this should be a multiple of function and
        !! corresponding potential times two
        class(Function3D),  intent(in), target :: func  
        ! objeect for spherical harmonics evaluation
        type(RealSphericalHarmonics), intent(in) :: harmo
        !> Reference point, where the multipoles are evaluated
        real(REAL64),      intent(in)       :: reference_point(3)
        !> Limits of the multipole evaluation in the grid
        integer, optional, intent(in)       :: limits(2, 3)

        ! result object containing cube 'helmholtz multipole moments' at the end of this function
        real(REAL64)                        :: multipole_moments((self%farfield_potential_lmax+1)**2)

        real(REAL64), allocatable           :: cube  (:,:,:), &            
                                               spherical_harmonics_cube(:, :, :, :), &
                                               bessel_values(:, :, :, :)

        ! integrals x, y, and z directions
        real(REAL64), allocatable           :: ints_x(:)
        real(REAL64), allocatable           :: ints_y(:)
        real(REAL64), allocatable           :: ints_z(:)

        integer                             :: i,j,k,l,m,n,nlip, id

        real(REAL64)                      :: kappa, startt, startt2, endt, endt2
        type(SecondModSphBesselCollection):: bessels
        type(Grid3D), target              :: subgrid
        type(Grid3D), pointer             :: grid

        multipole_moments = 0.d0
        
        ! initialize the Helmholtz-kernel parameter kappa        
        kappa = sqrt(-2.0d0*self%energy)

        if (present(limits)) then
            subgrid = func%grid%get_subgrid(limits)
            grid    => subgrid

            ! number of lagrange interpolation polynomials per cell
            nlip=func%grid%get_nlip()

            ! handle only limited part of the cube
            cube=func%cube((limits(1, X_)-1)*(nlip-1)+1: (limits(2, X_))*(nlip-1)+1,&
                           (limits(1, Y_)-1)*(nlip-1)+1: (limits(2, Y_))*(nlip-1)+1,&
                           (limits(1, Z_)-1)*(nlip-1)+1: (limits(2, Z_))*(nlip-1)+1)
        else


            ! handle all of the cube
            cube=func%cube

            grid => func%grid
 
        end if

        
        ! Note: X_ = 1, Y_ = 2, Z_ = 3
        ints_x=grid%axis(X_)%get_ints()
        ints_y=grid%axis(Y_)%get_ints()
        ints_z=grid%axis(Z_)%get_ints()
        
        ! calculate spherical harmonics values for all l: [0 <= l <= lmax] at all points 
        spherical_harmonics_cube = harmo%eval_grid(grid, reference_point)

        ! Initialize First Kind of Modified Spherical Bessel Function (I_l+½) 
        bessels = SecondModSphBesselCollection(self%farfield_potential_lmax)
        allocate(bessel_values(size(cube, 1), size(cube, 2), size(cube, 3), 0:self%farfield_potential_lmax))
        bessel_values = bessels%eval_grid(grid, reference_point, kappa)
        call bessels%destroy()
        ! Go through all spherical harmonics, calculate corresponding x, y and z axis values 
        ! for first modified spherical bessel functions and then c
        do l = 0, self%farfield_potential_lmax
 
            forall (m = -l : l)
                ! multiply spherical harmonics with corresponding bessel values and cube
                spherical_harmonics_cube(:, :, :, lm_map(l, m)) = &
                    spherical_harmonics_cube(:, :, :, lm_map(l, m)) * &
                    bessel_values(:, :, :, l) * cube
            end forall
        
            ! Finally calculate the following:
            !    \int I(k r) Y_lm f^{\Delta} dx dy dz
            do m = -l, l
                !print *, "l, m", l, m, "map", lm_map(l, m), maxval(spherical_harmonics_cube(:, :, :, lm_map(l, m))), &
                !         minval(spherical_harmonics_cube(:, :, :, lm_map(l, m)))
                ! Integrate over x, y, and z for the previously calculated product
                multipole_moments(lm_map(l, m)) = contract_cube(spherical_harmonics_cube(:, :, :, lm_map(l, m)), & 
                                                                ints_x, ints_y, ints_z)
            end do
        end do
        nullify(grid)
        call subgrid%destroy()
        deallocate(bessel_values)
        deallocate(cube)
        deallocate(ints_x, ints_y, ints_z)
        deallocate(spherical_harmonics_cube)
    end function

    !> destructor for Helmholtz3D object
    subroutine GBFMMHelmholtz3D_destroy(self)
        class(GBFMMHelmholtz3D), intent(inout)  :: self
#ifdef HAVE_CUDA
        if (allocated(self%cuda_interface)) then
            call GBFMMHelmholtz3D_destroy_cuda(self%cuda_interface)
        end if
#endif
        nullify(self%input_parallel_info)
        nullify(self%output_parallel_info)
        nullify(self%result_parallelization_info)
        nullify(self%gridin)
        nullify(self%gridout)
        call self%quadrature%destroy()
        nullify(self%coulomb_operator)
        if (allocated(self%multipole_moments)) then
            deallocate(self%multipole_moments)
        end if
    end subroutine 

    function GBFMMHelmholtz3D_get_cube_comparison_potential(self, func, limits, reference_point) &
           result(integration_value)
        class(GBFMMHelmholtz3D), intent(in)      :: self
        !> The input function. Note this should be a multiple of function and
        !! corresponding potential times two
        class(Function3D),  intent(in), target :: func  
        !> Limits of the multipole evaluation in the grid
        integer, optional, intent(in)       :: limits(2, 3)
        !> Reference point, where the multipoles are evaluated
        real(REAL64),      intent(in)       :: reference_point(3)
        real(REAL64), allocatable           :: cube  (:,:,:)
        type(Grid3D), target                :: subgrid
        type(Grid3D), pointer               :: grid
        integer                             :: nlip
        real(REAL64)                             :: integration_value

        if (present(limits)) then
            subgrid = func%grid%get_subgrid(limits)
            grid    => subgrid

            ! number of lagrange interpolation polynomials per cell
            nlip=func%grid%get_nlip()

            ! handle only limited part of the cube
            cube=func%cube((limits(1, X_)-1)*(nlip-1)+1: (limits(2, X_))*(nlip-1)+1,&
                           (limits(1, Y_)-1)*(nlip-1)+1: (limits(2, Y_))*(nlip-1)+1,&
                           (limits(1, Z_)-1)*(nlip-1)+1: (limits(2, Z_))*(nlip-1)+1)
        else


            ! handle all of the cube
            cube=func%cube

            grid => func%grid
 
        end if

        integration_value = self%get_cube_comparison_potential_cpu(grid, cube, reference_point)
        nullify(grid)
        call subgrid%destroy()
        deallocate(cube)
    end function

    !> Get e^-kappa*r / r calculated the easy, but VERY slow way
    function GBFMMHelmholtz3D_get_cube_comparison_potential_cpu(self, grid, cube, reference_point) &
             result(integration_value)
        class(GBFMMHelmholtz3D), intent(in)      :: self
        !> Evaluation grid
        type(Grid3D), intent(in)                 :: grid
        real(REAL64), intent(in)                 :: cube(grid%axis(X_)%get_shape(), &
                                                         grid%axis(Y_)%get_shape(), &
                                                         grid%axis(Z_)%get_shape()) 
        !> The cube for which the evaluation is made, must be of shape of parameter 'grid'
        real(REAL64)                             :: result_cube(grid%axis(X_)%get_shape(), &
                                                                grid%axis(Y_)%get_shape(), &
                                                                grid%axis(Z_)%get_shape()) 
        !> Reference point, where the multipoles are evaluated
        real(REAL64),      intent(in)            :: reference_point(3)
        ! helmholtz kernel parameter    
        real(REAL64)                             :: kappa

        ! gridpoints x, y, and z directions
        real(REAL64)                             :: gridpoints_x(grid%axis(X_)%get_shape()), &
                                                    gridpoints_y(grid%axis(Y_)%get_shape()), &
                                                    gridpoints_z(grid%axis(Z_)%get_shape())
        real(REAL64)                             :: integration_value
        integer                                  :: o, p, q
        
        type(Integrator)                         :: integr

        
        real(REAL64)                      :: startt, startt2, endt, endt2
        type(FirstModSphBesselCollection) :: bessels

        ! initialize the Helmholtz-kernel parameter kappa        
        kappa = sqrt(-2.0d0*self%energy)

 
        ! initialize the result object to r
        gridpoints_x = grid%axis(X_)%get_coord() - reference_point(X_)
        gridpoints_y = grid%axis(Y_)%get_coord() - reference_point(Y_)
        gridpoints_z = grid%axis(Z_)%get_coord() - reference_point(Z_)

        ! evaluate r to result cube
        forall(o = 1 : grid%axis(X_)%get_shape(), p = 1 : grid%axis(Y_)%get_shape(), q = 1 : grid%axis(Z_)%get_shape())
            result_cube(o, p, q) = sqrt(gridpoints_x(o) ** 2 + gridpoints_y(p) ** 2 + gridpoints_z(q) ** 2 )
        end forall  

        ! get e^-kr/r
        result_cube = cube * exp(-kappa * result_cube) / result_cube

       
        integr=Integrator(grid%axis)
        integration_value = integr%eval(result_cube)
        call integr%destroy()

        
    end function

    
end module
