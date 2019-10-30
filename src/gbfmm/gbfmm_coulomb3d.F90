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
!> @file gbfmm_coulomb3d.f90 This file implements operator used to 
!! evaluate Coulomb potential via 'Grid-Based Fast Multipole Method'
!
! Published in: <name of the article>
!
! A finite-element approach for solving the Coulomb problem of SCF theory.
!
! This is an assistance class for the Grid-based Fast Multipole Method. Contains
! the GBFMMCoulomb3D object derived from Coulomb3D to allow Coulomb3D-like 
! use of the GBFMM method. Now the GBFMM Coulomb potential can be evaluated as 
!
!     allocate(V_operator, source = GBFMMCoulomb3D(grid=grid,         &
!                                         maxlevel=maxlevel, &
!                                         lmax=lmax))
!     V = V_operator .apply. rho 
!
! where rho is the charge density, of type Function3D, and V_operator is of type
! class(Coulomb3D) and V is of type 'GBFMMFunction3D', which extends 'Function3D'.
! 'grid' is a Grid3D object, maxlevel is the number of recursive box splits and lmax
! is the number of TODO taken into account in the multipole calculation.
!
! 
! Elias Toivanen (2014), Pauli Parkkinen (2015-2016)
!


module GBFMMCoulomb3D_class
#ifdef HAVE_MPI
    use MPI
#endif
    use ISO_FORTRAN_ENV
    use Function3D_class
    use Function3D_types_m
    use Multipole_class
    use Coulomb3D_class
    use GaussQuad_class
    use GBFMMFunction3D_class
    use GBFMMParallelInfo_class
    use Generator_class
    use MemoryLeakChecker_m
    use ParallelInfo_class
    use RealSphericalHarmonics_class
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

    public      :: GBFMMCoulomb3D
    public      :: function3d_boxed_density

    private

    type, extends(Coulomb3D) :: GBFMMCoulomb3D
        integer                         :: start_level
        integer                         :: maxlevel
        integer                         :: lmax
        logical                         :: adaptive_mode = .FALSE.
        !> If the downward pass is done. Note: The downward pass is needed
        !! when accurate potentials are needed
        logical                         :: downward_pass = .TRUE.
        real(REAL64)                    :: step
        real(REAL64)                    :: boxmin(3)
        real(REAL64)                    :: rootlen(3)
        integer                         :: FMM_NODE = 0
        real(REAL64),            allocatable  :: multipole_moments(:,:)
        real(REAL64),            allocatable :: bubble_multipole_moments(:,:)
        integer,                 allocatable :: nearfield_limits(:, :, :)
        type(Coulomb3D),         allocatable :: nearfield_coulomb_operator(:), &
                                                nearfield_coulomb_operators(:, :)
        type(Grid3D),            allocatable :: input_grids(:)
        type(Grid3D),            allocatable :: output_grids(:)
        type(SerialInfo),        allocatable :: input_parallel_infos(:)
        type(SerialInfo),        allocatable :: output_parallel_infos(:)
        type(GBFMMParallelInfo), pointer     :: input_parallel_info
        type(GBFMMParallelInfo), pointer     :: output_parallel_info
        type(SerialInfo),        allocatable :: global_serial_info
        type(REAL64_3D),         allocatable :: interaction_matrices(:)
        type(integer_1D),        allocatable :: farfield_indices(:)
    contains
        procedure           :: destroy                      => &
                    GBFMMCoulomb3D_destroy
        procedure, private  :: calculate_potential          => &
                    GBFMMCoulomb3D_calculate_potential
        procedure, private  :: operate_bubbles              => &
                    GBFMMCoulomb3D_operate_bubbles
        procedure, private  :: operate_farfield             => &
                    GBFMMCoulomb3D_operate_farfield
        procedure, public   :: calculate_nearfield_potential=> &
                    GBFMMCoulomb3D_calculate_nearfield_potential
        procedure, private  :: nearfield_operators          => &
                    GBFMMCoulomb3D_nearfield_operators
        procedure, private  :: calculate_local_farfield_potential => &
                    GBFMMCoulomb3D_calculate_local_farfield_potential
        procedure, private  :: calculate_farfield_bubbles_potential      => &
                    GBFMMCoulomb3D_calculate_farfield_bubbles_potential
        procedure, private  :: translate_potential          => &
                    GBFMMCoulomb3D_translate_potential
        procedure, private  :: translate_multipoles         => &
                    GBFMMCoulomb3D_translate_multipoles
        procedure           :: operate_on                   => &
                    GBFMMCoulomb3D_calculate_potential
        procedure, private  :: communicate_multipole_moments=> &
                    GBFMMCoulomb3D_communicate_multipole_moments
        procedure, private  :: init_nearfield      => &
                    GBFMMCoulomb3D_init_nearfield
        procedure, private  :: calculate_farfield_contaminants => &
                    GBFMMCoulomb3D_calculate_farfield_contaminants
        procedure, public   :: set_transformation_weights =>   &
                    GBFMMCoulomb3D_set_transformation_weights       
        procedure               :: evaluate_potential_le_box      => &
                    GBFMMCoulomb3D_evaluate_potential_le_box
        procedure, private      :: evaluate_potential_le_box_cpu  => &
                    GBFMMCoulomb3D_evaluate_potential_le_box_cpu
#ifdef HAVE_CUDA
        procedure               :: cuda_init => &
                    GBFMMCoulomb3D_cuda_init
        procedure               :: cuda_destroy => &
                    GBFMMCoulomb3D_cuda_destroy
        procedure               :: cuda_prepare => &
                    GBFMMCoulomb3D_cuda_init_harmonics
        procedure               :: cuda_unprepare => &
                    GBFMMCoulomb3D_cuda_destroy_harmonics
#endif
        procedure, public       :: calculate_multipole_moments => &
                    GBFMMCoulomb3D_calculate_multipole_moments
        procedure, public       :: simplify => &
                    GBFMMCoulomb3D_simplify
        procedure, private      :: init_interaction_matrix => &
                    GBFMMCoulomb3D_init_interaction_matrix
        procedure, private      :: init_interaction_matrices => &
                    GBFMMCoulomb3D_init_interaction_matrices
                    
    end type

#ifdef HAVE_CUDA

    

    interface
        type(C_PTR) function GBFMMCoulomb3D_init_cuda( &
               grid_in, grid_out, lmax, domain,  &
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
        subroutine GBFMMCoulomb3D_init_harmonics_cuda(gbfmm_coulomb3d)  bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: gbfmm_coulomb3d

        end subroutine
    end interface

    interface
        subroutine GBFMMCoulomb3D_destroy_harmonics_cuda(gbfmm_coulomb3d)  bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: gbfmm_coulomb3d

        end subroutine
    end interface

    interface
        subroutine GBFMMCoulomb3D_destroy_cuda(gbfmm_coulomb3d)  bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: gbfmm_coulomb3d

        end subroutine
    end interface

    interface
        subroutine GBFMMCoulomb3D_evaluate_potential_le_cuda(gbfmm_coulomb3d, local_expansion, output_cube)  bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: gbfmm_coulomb3d
            real(C_DOUBLE)        :: local_expansion(*)
            type(C_PTR), value    :: output_cube

        end subroutine
    end interface

    interface
        subroutine GBFMMCoulomb3D_calculate_multipole_moments_cuda(gbfmm_coulomb3d, input_cube)  bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: gbfmm_coulomb3d
            type(C_PTR), value    :: input_cube

        end subroutine
    end interface

    interface 
        type(C_PTR) function GBFMMCoulomb3D_get_box_stream_container_cuda(gbfmm_coulomb3d, ibox) bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value    :: gbfmm_coulomb3d
            integer(C_INT), value    :: ibox
        end function
    end interface

    interface
        subroutine GBFMMCoulomb3D_download_multipole_moments_cuda(gbfmm_coulomb3d, host_multipole_moments)  bind(C)
            use ISO_C_BINDING
            type(C_PTR), value       :: gbfmm_coulomb3d
            real(C_DOUBLE)           :: host_multipole_moments(*)

        end subroutine
    end interface

    interface
        subroutine GBFMMCoulomb3D_upload_domain_boxes_cuda(gbfmm_coulomb3d, input_cube)  bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: gbfmm_coulomb3d
            type(C_PTR), value    :: input_cube

        end subroutine
    end interface


 
#endif

    interface GBFMMCoulomb3D
        module procedure :: GBFMMCoulomb3D_init
        module procedure :: GBFMMCoulomb3D_init_operator_parallel_info
    end interface
    

contains

    function GBFMMCoulomb3D_init_operator_parallel_info(input_parallel_info, output_parallel_info, &
                                                        bubbls, lmax, gaussian_quadrature) result(new)
        !> parallelization info of the input function
        class(ParallelInfo), intent(in)             :: input_parallel_info
        !> parallelization info of the output function
        class(ParallelInfo), intent(in)             :: output_parallel_info
        !> prototype bubbles object
        type(Bubbles), intent(in)                   :: bubbls
        !> Truncation of the multipole expansion (number of multipole shells)
        integer, intent(in), optional               :: lmax 
        !> If we are using the adpative mode in the local farfield evaluation
        type(GBFMMCoulomb3D)                        :: new
        !> Input gaussian quadrature
        type(GaussQuad), intent(in), optional       :: gaussian_quadrature

        select type(input_parallel_info)
            type is (GBFMMParallelInfo)
                select type(output_parallel_info)
                    type is (GBFMMParallelInfo)
                        new = GBFMMCoulomb3D(input_parallel_info%grid, bubbls, &
                            input_parallel_info, output_parallel_info, &
                            lmax = lmax, &
                            gaussian_quadrature = gaussian_quadrature)
                    type is (SerialInfo)
                        print *, "ERROR initializing GBFMMCoulomb3D: input parallel &
                                & info type is not gbfmmparallelinfo but a serialinfo"
                        stop
                    class default
                        print *, "ERROR: input parallel info type is not gbfmmparallelinfo"
                        stop
                end select
            type is (SerialInfo)
                print *, "ERROR initializing GBFMMCoulomb3D: input parallel &
                          & info type is not gbfmmparallelinfo but a serialinfo"
                stop
            class default
                print *, "ERROR: input parallel info type is not gbfmmparallelinfo"
                stop
        end select
        

    end function 
    
    !> Electric potential of `rho` computed with a finite-element FMM algorithm
    !!
    !! TODO: The input function is now assumed to be distributed among
    !! 8**maxlevel boxes that are stored in the input vector `rho' in
    !! Morton order (Z-order curve, see the `unmorton/morton` routines.
    !! This should be generalized to accommodate bubbles.
    function GBFMMCoulomb3D_init(global_grid, bubbls, &
                                 input_parallel_info, output_parallel_info, &
                                 lmax, adaptive_mode, maxlevel,  &
                                 gaussian_quadrature) result(new)
        !> Grid of the operated density 
        type(Grid3D), intent(in)                              :: global_grid
        type(Bubbles), intent(in)                             :: bubbls
        !> parallelization info of the input function
        type(GBFMMParallelInfo), intent(in), target           :: input_parallel_info
        !> parallelization info of the output function
        type(GBFMMParallelInfo), intent(in), target           :: output_parallel_info
        !> Number of divisions, depth of the octree
        integer, intent(in), optional                         :: maxlevel
        !> Truncation of the multipole expansion (number of multipole shells)
        integer, intent(in), optional                         :: lmax 
        !> If we are using the adpative mode in the local farfield evaluation
        logical, intent(in), optional                         :: adaptive_mode
        !> Input gaussian quadrature
        type(GaussQuad), intent(in), optional                 :: gaussian_quadrature
 
        
        real(REAL64)                                          :: ranges(2, 3)

        ! The result operator
        type(GBFMMCoulomb3D)                                  :: new
        ! Temporary variable to store the total number of boxes in all levels
        integer                                               :: boxes_tot

        call bigben%split("Init GBFMMCoulomb3D -operator")
        if (present(maxlevel)) then
            new%maxlevel = maxlevel
        else if (global_grid%gbfmm_maxlevel > 0) then
            new%maxlevel = global_grid%gbfmm_maxlevel
        else
            new%maxlevel = 2
        end if
        
        if (present(lmax)) then
            new%lmax = lmax
        else
            new%lmax = 12
        end if 

        if (present(adaptive_mode)) then
            new%adaptive_mode = adaptive_mode
        else 
            new%adaptive_mode = (input_parallel_info%start_level == 1)   
        end if

        if (new%adaptive_mode) then
            new%start_level = 1
        else
            new%start_level = 2
        end if 

        new%input_parallel_info => input_parallel_info
        new%output_parallel_info => output_parallel_info
        new%result_parallelization_info => output_parallel_info
        new%boxmin = global_grid%get_qmin()
        new%result_type = F3D_TYPE_CUSP
        new%nuclear_potential = Function3D(new%result_parallelization_info, type=F3D_TYPE_NUCP)
        new%nuclear_potential%bubbles =  new%get_potential_bubbles(bubbls)
        new%nuclear_potential%cube = 0.0d0
        call new%nuclear_potential%precalculate_taylor_series_bubbles()
        ! boxes_tot is total number of boxes in all levels
        boxes_tot = new%input_parallel_info%get_total_box_count()
        ! Allocate space for multipole moments
        allocate(new%multipole_moments((new%lmax + 1)**2, boxes_tot))
        ! parallel info must be initialized before these
        call new%init_nearfield(global_grid, &
                 bubbls, gaussian_quadrature = gaussian_quadrature)
        call new%init_interaction_matrices()

        ! set output and input grids to be the grid of the input function.
        ! This setting is only used in 'coda evaluation'
        new%gridin => new%input_parallel_info%get_grid()
        new%gridout => new%output_parallel_info%get_grid()
        call bigben%stop()

    end function


    !> Initializes the nearfield operators used to evaluate the nearfield potential 
    !! in the GBFMM procedure. The result operators are stored in 'self%nearfield_coulomb_operator'.
    !! Additionally the nearfield cell limits and all the related grids and parallel infos 
    !! are evaluated and stored in self.
    subroutine GBFMMCoulomb3D_init_nearfield(self, grid, bubbls, gaussian_quadrature)
        class(GBFMMCoulomb3D)                 :: self
        !> The global grid handled by this operator on all resources
        type(Grid3D), intent(in)              :: grid 
        type(Bubbles), intent(in)             :: bubbls
        type(GaussQuad), intent(in), optional :: gaussian_quadrature
        ! box limits for box ibox and jbox and nearfield as integers
        integer                               :: limits_ibox(2, 3), limits_jbox(2, 3), &
                                                 limits_nearfield(2,3), box_number, j, &
                                                 ibox, jbox 

        ! the coulomb operator for entire area
        type(Coulomb3D)                       :: main_coul
        type(GaussQuad)                       :: quadrature
      
        ! the numbers of the first and last box handled by this processor
        integer                               :: domain(2)

        logical                               :: cut_end_borders(3)
        
        ! uncomment the row below if the grid based init is used
        type(Grid3D)                 :: grid_ibox, grid_nearfield!, grid_jbox, grid_nearfield
        
        if (present(gaussian_quadrature)) then
            quadrature = gaussian_quadrature
        else
            quadrature = GaussQuad(25, 25)
        end if
        
        ! initialize main operator, comment this if the grid based init is in use
        self%global_serial_info = SerialInfo(grid)
        main_coul = Coulomb3D(self%global_serial_info, self%global_serial_info, bubbls, quadrature)

        ! get starting and ending indices of this processors domain
        domain = self%input_parallel_info%get_domain(self%maxlevel)
        ! allocate domain(2) - domain(1) + 1 nearfield operators and nearfield limits,
        ! start from 1 to save space
        allocate(self%nearfield_coulomb_operator(domain(2) - domain(1) + 1))
        allocate(self%nearfield_limits(2, 3, domain(2) - domain(1) + 1))
        allocate(self%input_grids(domain(2) - domain(1) + 1))
        allocate(self%input_parallel_infos(domain(2) - domain(1) + 1))
        allocate(self%output_grids(domain(2) - domain(1) + 1))
        allocate(self%output_parallel_infos(domain(2) - domain(1) + 1))
 
        do ibox = domain(1), domain(2)
            ! the order number of box in domain
            box_number = ibox - domain(1) + 1
            ! evaluate box's cell limits & get a grid corresponding to these limits
            limits_ibox = self%input_parallel_info%get_box_cell_index_limits(ibox, self%maxlevel, global = .TRUE.)
            self%output_grids(box_number) = grid%get_subgrid(limits_ibox)
            self%output_parallel_infos(box_number) = SerialInfo(self%output_grids(box_number))

            ! determine if the last index of cube is evaluated in each direction
            !cut_end_borders = .FALSE.
            cut_end_borders(X_) = limits_ibox(2, X_) /= grid%axis(X_)%get_ncell()
            cut_end_borders(Y_) = limits_ibox(2, Y_) /= grid%axis(Y_)%get_ncell()
            cut_end_borders(Z_) = limits_ibox(2, Z_) /= grid%axis(Z_)%get_ncell()
            

            ! get the nearfield area for ibox, start from the area of ibox and broaden it to include
            ! all the nearest neighbors (this takes )
            limits_nearfield = self%input_parallel_info%get_nearfield_cell_index_limits(ibox, global = .TRUE.)
            self%input_grids(box_number) = grid%get_subgrid(limits_nearfield)
            self%input_parallel_infos(box_number) = SerialInfo(self%input_grids(box_number))
        
            ! Uncomment the rows below for the traditional grid based coulomb operator initialization
            ! ------------------------------------------------------------------------------
            !grid_ibox = grid%get_subgrid(limits_ibox)
            !grid_nearfield = grid%get_subgrid(limits_nearfield)
            !self%nearfield_coulomb_operator(box_number) = Coulomb3D(grid_nearfield, &
            !                    grid_ibox,&
            !                    quadrature, &
            !                    result_parallelization_info = SerialInfo(grid_ibox))
            !call grid_ibox%destroy()
            !call grid_nearfield%destroy()
            ! ------------------------------------------------------------------------------
            ! store the limits of the nearfield and the corresponding coulomb operator
            
            self%nearfield_coulomb_operator(box_number) = &
                   main_coul%get_suboperator(limits_nearfield, limits_ibox, &
                                             self%input_parallel_infos(box_number), &
                                             self%output_parallel_infos(box_number), &
                                             quadrature, cut_end_borders)

            self%nearfield_limits(:, :, box_number) =  &
                self%input_parallel_info%get_nearfield_cell_index_limits(ibox, global = .FALSE.)
           
        enddo
        
        call main_coul%destroy()
        call quadrature%destroy()
    end subroutine

#ifdef HAVE_CUDA
    subroutine GBFMMCoulomb3D_cuda_init(self)
        class(GBFMMCoulomb3D), intent(inout)  :: self
        integer                               :: box_number, ibox, domain(2)
        integer, allocatable                  :: input_start_indices_x(:), input_start_indices_y(:), &
                                                 input_start_indices_z(:), input_end_indices_x(:), &
                                                 input_end_indices_y(:),   input_end_indices_z(:)
        integer, allocatable                  :: output_start_indices_x(:), output_start_indices_y(:), &
                                                 output_start_indices_z(:), output_end_indices_x(:), &
                                                 output_end_indices_y(:),   output_end_indices_z(:)
        integer                               :: input_box_cell_index_limits(2, 3), output_box_cell_index_limits(2, 3)
        type(C_PTR)                           :: box_stream_container

        ! get starting and ending indices of this processors domain
        domain = self%input_parallel_info%get_domain(self%maxlevel)

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

            input_box_cell_index_limits = self%input_parallel_info%get_box_cell_index_limits(ibox, level = self%maxlevel)
            input_start_indices_x(box_number) = input_box_cell_index_limits(1, X_) -1
            input_start_indices_y(box_number) = input_box_cell_index_limits(1, Y_) -1
            input_start_indices_z(box_number) = input_box_cell_index_limits(1, Z_) -1
            input_end_indices_x(box_number) = input_box_cell_index_limits(2, X_)
            input_end_indices_y(box_number) = input_box_cell_index_limits(2, Y_)
            input_end_indices_z(box_number) = input_box_cell_index_limits(2, Z_)

            output_box_cell_index_limits = self%output_parallel_info%get_box_cell_index_limits(ibox, level = self%maxlevel)
            output_start_indices_x(box_number) = output_box_cell_index_limits(1, X_) -1
            output_start_indices_y(box_number) = output_box_cell_index_limits(1, Y_) -1
            output_start_indices_z(box_number) = output_box_cell_index_limits(1, Z_) -1
            output_end_indices_x(box_number) = output_box_cell_index_limits(2, X_)
            output_end_indices_y(box_number) = output_box_cell_index_limits(2, Y_)
            output_end_indices_z(box_number) = output_box_cell_index_limits(2, Z_)
        end do
        ! init the input & output cuda cubes
        self%input_cuda_cube = CudaCube(self%gridin%get_shape())
        self%output_cuda_cube = CudaCube(self%gridout%get_shape())
        self%cuda_tmp1 = CudaCube([self%gridout%axis(X_)%get_shape(), &
                                   self%gridin%axis(Y_)%get_shape(), &
                                   self%gridin%axis(Z_)%get_shape()])
        self%cuda_tmp2 = CudaCube([self%gridout%axis(X_)%get_shape(), &
                                   self%gridout%axis(Y_)%get_shape(), &
                                   self%gridin%axis(Z_)%get_shape()])

        
        ! init the CUDA object responsible for multipole evaluations, etc.
        self%cuda_interface = GBFMMCoulomb3D_init_cuda(self%gridin%get_cuda_interface(), &
                                   self%gridout%get_cuda_interface(), &
                                   self%lmax, domain, &
                                   input_start_indices_x, input_end_indices_x, &
                                   input_start_indices_y, input_end_indices_y, &
                                   input_start_indices_z, input_end_indices_z, &
                                   output_start_indices_x, output_end_indices_x, &
                                   output_start_indices_y, output_end_indices_y, &
                                   output_start_indices_z, output_end_indices_z, &
                                   stream_container)
        self%cuda_inited = .TRUE.
        deallocate(input_start_indices_x, input_start_indices_y, input_start_indices_z, &
                   input_end_indices_x, input_end_indices_y, input_end_indices_z)
        deallocate(output_start_indices_x, output_start_indices_y, output_start_indices_z, &
                   output_end_indices_x, output_end_indices_y, output_end_indices_z)

        do ibox = domain(1), domain(2)
            ! get streamcontainer handling the box
            !box_stream_container = GBFMMCoulomb3D_get_box_stream_container_cuda(self%cuda_interface, ibox)

            ! the order number of box in domain
            box_number = ibox - domain(1) + 1

            ! set the stream container for the nearfield operator
            !call self%nearfield_coulomb_operator(box_number)%set_stream_container(box_stream_container)

            
            ! init the cuda of the nearfield operators
            call self%nearfield_coulomb_operator(box_number)%cuda_init()
        end do
    end subroutine

    
    subroutine GBFMMCoulomb3D_cuda_init_harmonics(self)
        class(GBFMMCoulomb3D), intent(inout)  :: self

        call GBFMMCoulomb3D_init_harmonics_cuda(self%get_cuda_interface())
    end subroutine

    subroutine GBFMMCoulomb3D_cuda_destroy_harmonics(self)
        class(GBFMMCoulomb3D), intent(inout)  :: self

        call GBFMMCoulomb3D_destroy_harmonics_cuda(self%get_cuda_interface())
    end subroutine

    subroutine GBFMMCoulomb3D_cuda_destroy(self)
        class(GBFMMCoulomb3D), intent(inout)  :: self
        integer                               :: box_number, ibox, domain(2)
        

        ! get starting and ending indices of this processors domain
        domain = self%input_parallel_info%get_domain(self%maxlevel)
        if (allocated(self%nearfield_coulomb_operator)) then
            do ibox = domain(1), domain(2)
                ! the order number of box in domain
                box_number = ibox - domain(1) + 1
                call self%nearfield_coulomb_operator(box_number)%cuda_destroy()
            end do
        end if
        if (allocated(self%cuda_interface)) call GBFMMCoulomb3D_destroy_cuda(self%cuda_interface)

        ! destroy the input, output, and temp arrays
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
    
    !> Implementation of the FMM algorithm as an operator. This method is called with
    !! .apply. operator
    subroutine GBFMMCoulomb3D_calculate_potential(self, func, new, cell_limits, only_cube)
        class(GBFMMCoulomb3D)                :: self
        class(Function3D), intent(in)        :: func
        !> The result function containing everything needed for energy calculation 
        !! using GBFMM
        class(Function3D), intent(inout), allocatable, target  :: new 

        type(GBFMMFunction3D),    pointer    :: potential
        !> Limits in the area in which the operator is applied. There is no use for this
        !! argument in this class, but is only introduced because of the function interface
        !! in Function3D.
        integer, intent(in), optional        :: cell_limits(2, 3)
        !> if only cube is taken into account. 
        logical, intent(in), optional        :: only_cube

        ! loop indices
        integer                              :: ibox
        integer                              :: ilevel
 
        ! box center
        real(REAL64)                         :: box_center(3), tic, toc, toc2
        !real(REAL64)                        :: cell_size(3)
        integer                              :: boxes_tot
        ! box limits as integers
        integer                              :: box_limits(2, 3)
        integer                              :: domain(2)

        ! MPI variables
        integer                              :: first_box, last_box, thread_id
        integer                              :: ierr
        integer                              :: nproc = 1
        integer                              :: nmpiproc = 1
        integer                              :: iproc = 0
        logical                              :: calculate_bubbles
write(*,*) 'begin GBFMMCoulomb3D_calculate_potential'

        call bigben%split("Calculate GBFMM Coulomb Potential")
        calculate_bubbles = .TRUE.
        if (present(only_cube)) then
            calculate_bubbles = .not. only_cube
        end if
        if (.not. allocated(new)) then
            ! allocate the polymorphic result variable 
            allocate(GBFMMFunction3D :: new)
            ! set the non-polymorphic pointer to point to new
            select type(new)
                type is (GBFMMFunction3D)
                    potential => new
            end select 
            ! initialize the result object
            call potential%init(self%output_parallel_info, self%lmax, &
                        self%get_result_type(), downward_pass = self%downward_pass, &
                        bubble_count = func%bubbles%get_nbub())
            potential%bubbles = Bubbles(func%bubbles)
            potential%bubbles = 0.0d0

        else
            ! set the non-polymorphic pointer to point to new
            select type(new)
                type is (GBFMMFunction3D)
                    potential => new
            end select 

            potential%cube = 0.0d0
            potential%bubbles = 0.0d0
        end if
        potential%farfield_potential = 0.0d0
        self%multipole_moments = 0.0d0
#ifdef HAVE_CUDA    
        ! TODO: upload piece-wise. (For the next line to be usable, there has to be another function that
        ! uploads the nearfield of the boxes also).
        !call GBFMMCoulomb3D_upload_domain_boxes(self%cuda_inteface, self%input_cuda_cube)
        ! instead, we are doing a global upload above
        call CUDASync_all() 
        call bigben%split("Upload to CUDA")

        ! register the input and output cubes, upload the input cube and set the output cube to zero 
        call self%input_cuda_cube%set_host(func%cube)
        call self%output_cuda_cube%set_host(new%cube)
        call self%input_cuda_cube%upload()
        call self%output_cuda_cube%set_to_zero()

        ! make sure that the input cube is successfully uploaded before starting multipole evaluation
        call CUDASync_all()
        call bigben%stop()
#endif

#ifdef HAVE_CUDA       
        call bigben%split("Calculate multipole moments CUDA")
        call GBFMMCoulomb3D_calculate_multipole_moments_cuda(self%cuda_interface, self%input_cuda_cube%cuda_interface)
        call GBFMMCoulomb3D_download_multipole_moments_cuda(self%cuda_interface, self%multipole_moments)
        call CUDASync_all()
        call bigben%stop()
#else
        call self%calculate_multipole_moments(func)
#endif

        ! Initialize upward pass with multipole evaluation
        ! calculate the maximum level multipole moments
        call bigben%split("Communicate multipole moments")
        call self%input_parallel_info%communicate_matrix(self%multipole_moments, self%maxlevel)
        call bigben%stop()

        ! Upward pass: translate multipoles to coarser level
        do ilevel = self%maxlevel-1, self%start_level, -1
            call bigben%split("Translate cube multipole moments")
            call potential%translate_multipoles(self%multipole_moments, ilevel)
            call bigben%stop()

            ! Make sure that each processor has the correct multipole moments
            ! for the next steps
            call bigben%split("Communicate multipole moments")
            call self%input_parallel_info%communicate_matrix(self%multipole_moments, ilevel)
            call bigben%stop()
        enddo

        ! Calculate the nearfield part of the potential, note: this must be done after the
        ! farfield is calculated
#ifdef HAVE_OMP
        !$OMP BARRIER
        !$OMP PARALLEL PRIVATE(thread_id) NUM_THREADS(3)
            thread_id = omp_get_thread_num()
            

            if (thread_id == 1) then
                call self%calculate_nearfield_potential(potential, func)
            else if (thread_id == 2) then
                call self%operate_farfield(potential)
            else
                !call omp_set_num_threads(OMP_GLOBAL_THREAD_COUNT-2)
                call self%operate_bubbles(potential, func, calculate_bubbles)
                !$OMP MASTER
                    if (calculate_bubbles) call potential%parallelization_info%communicate_bubbles(potential%bubbles)                
                !$OMP END MASTER
            end if
        !$OMP END PARALLEL
#else
        call self%operate_farfield(potential)
        call self%calculate_nearfield_potential(potential, func)
        call self%operate_bubbles(potential, func, calculate_bubbles)
        if (calculate_bubbles) call potential%parallelization_info%communicate_bubbles(potential%bubbles)
#endif
        
#ifdef HAVE_CUDA
        call CUDASync_all()
#endif
        call bigben%split("Simplify")
        call self%simplify(potential)
        call bigben%stop()

#ifdef HAVE_CUDA    
        call CUDASync_all()
        call bigben%split("Download from CUDA")
        call self%output_cuda_cube%download() 
        call CUDASync_all()

        ! unregister the host cubes from the input_cube structures
        call self%input_cuda_cube%unset_host()
        call self%output_cuda_cube%unset_host()
        call bigben%stop()
#endif

        ! do MPI-communication between nodes to communicate the borders between node-domains
        call potential%communicate_cube_borders(reversed_order = .TRUE.)

        ! precalculate the taylor series bubbles to save some time in future multiplications
        call potential%precalculate_taylor_series_bubbles()
        call bigben%stop()

        !print *, "Coulomb potential nearfield time", toc-tic
write(*,*) 'begin GBFMMCoulomb3D_calculate_potential'
    end subroutine



    subroutine GBFMMCoulomb3D_calculate_multipole_moments(self, func)
        class(GBFMMCoulomb3D), intent(inout) :: self
        type(Function3D),      intent(in)    :: func
        integer                              :: domain(2), ibox, nlip, grid_shape(3)
        integer,      allocatable            :: box_limits(:, :, :)
        real(REAL64), allocatable            :: box_centers(:, :)

        call bigben%split("Calculate cube multipole moments")
        domain = self%input_parallel_info%get_domain(self%maxlevel)

        allocate(box_limits(2, 3, domain(1):domain(2)))
        box_limits(:, :, domain(1)) = self%input_parallel_info%get_box_cell_index_limits(domain(1), &
            self%maxlevel, global = .FALSE.)
        
        allocate(box_centers(3, domain(1):domain(2)))
        box_centers(:, domain(1)) = self%input_parallel_info%get_box_center(domain(1), self%maxlevel)

        do ibox = domain(1)+1, domain(2)

            ! evaluate box center and cell limits
            box_centers(:, ibox) = self%input_parallel_info%get_box_center(ibox, self%maxlevel)
            box_limits(:, :, ibox) = self%input_parallel_info%get_box_cell_index_limits(ibox, &
                self%maxlevel, global = .FALSE.)


        end do
        call bigben%split("Evaluation")
        do ibox = domain(1), domain(2)
            self%multipole_moments(:, ibox) = func%cube_multipoles(self%lmax, box_centers(:, ibox),&
                                                box_limits(:, :, ibox))
        end do
        
        call bigben%stop()
        deallocate(box_centers, box_limits)

        call bigben%stop()

    end subroutine
    
    
    !> Before translating multipoles to coarser level, the processord doing the 
    !! translation must have all the child-level multipoles and in order to 
    !! calculate the local farfield contribution, all the children of 
    !! neighbors of parent must be present 
    subroutine GBFMMCoulomb3D_communicate_multipole_moments(self, child_level)
        class(GBFMMCoulomb3D)          :: self
        integer, intent(in)            :: child_level
    end subroutine

    subroutine GBFMMCoulomb3D_operate_farfield(self, potential)
        class(GBFMMCoulomb3D)                         :: self
        type(GBFMMFunction3D), intent(inout)          :: potential
        integer                                       :: ilevel


        !call bigben%split("Calculate bubble multipole moments")
        !self%bubble_multipole_moments = func%bubbles%get_multipole_moments(lmax = self%lmax)
        !call bigben%stop()
        ! Initialize potential generation at level 2
        !call bigben%split("Calculate all farfield potential")

        !call bigben%split("Calculate farfield potential")
        ilevel = 2
        call self%calculate_local_farfield_potential(potential,  ilevel)
        !call bigben%stop()

        do ilevel = 3, self%maxlevel
            ! translate potential from level ilevel-1 to parent (ilevel) level, 
            ! if the downward pass flag is on
            if (self%downward_pass) then
                !call bigben%split("Translate potential")
                call self%translate_potential(potential, ilevel)
                !call bigben%stop()
            end if 
            ! calculate local farfield potential
            !call bigben%split("Calculate farfield potential")
            call self%calculate_local_farfield_potential(potential, ilevel)
            !call bigben%stop()
        enddo
        
        !print *, "Coulomb potential farfield time", toc-tic, "multipoles time", toc2-tic
        !call bigben%split("Calculate farfield bubbles potential")
        !call self%calculate_farfield_bubbles_potential(func%bubbles, potential)
        !call bigben%stop()

    end subroutine


    subroutine GBFMMCoulomb3D_operate_bubbles(self, potential, rho, calculate_bubbles)
        class(GBFMMCoulomb3D), target :: self
        !> result object to which the result farfield potential is stored
        type(GBFMMFunction3D), intent(inout) &
                                      :: potential
        ! charge density rho
        type(Function3D),  intent(in) :: rho
        logical,           intent(in) :: calculate_bubbles
        ! Do the bubbles according to the standard Coulomb3D procedures
        if (calculate_bubbles) then
            
            if (rho%bubbles%get_nbub() > 0) then
                call self%evaluate_bubbles_potential(rho%bubbles, potential%bubbles)
            end if

        end if
        
    end subroutine


    
    !> Construct the numerical potentials caused by nearfield for boxes 
    subroutine GBFMMCoulomb3D_calculate_nearfield_potential(self, potential, rho)
        class(GBFMMCoulomb3D), target :: self
        !> result object to which the result farfield potential is stored
        type(GBFMMFunction3D), intent(inout) &
                                      :: potential
        ! charge density rho
        type(Function3D), intent(in)  :: rho

        integer                       :: ibox, box_number
        integer                       :: domain(2)
        integer                       :: ibub, i


        ! taylor expansion derivatives of the cube potential at the bubble centers         
        real(REAL64), allocatable    :: cube_contaminants(:, :), box_cube_contaminants(:, :)

        ! the result potential for each box is stored temporarily in the
        ! nearfield_potential
        class(Function3D), allocatable :: nearfield_potential

        ! pointer to a operator used for each box
        type(Coulomb3D), pointer     :: coul

        logical                      ::  full_cuda_evaluation
   
        
        ! box limits for box ibox and its nearfield as cell number integers
        integer                             :: nearfield_cell_index_limits(2,3), &
                                               nearfield_cube_limits(2, 3), &
                                               box_cell_index_limits(2, 3), &
                                               box_cube_limits(2, 3), &
                                               tmp1_limits(2, 3), &
                                               tmp2_limits(2, 3)
        real(REAL64)                        :: limits_nearfield(2, 3)

#ifdef HAVE_CUDA
        type(CUDACube)                      :: box_cube, nearfield_cube, tmp1, tmp2
        type(C_PTR)                         :: box_stream_container
        integer                             :: dims(3, 2)

        
        if (self%cuda_inited) then
            full_cuda_evaluation = .TRUE.
        else
            full_cuda_evaluation = .FALSE.
        end if

#endif
        ! get starting and ending indices of this processors domain
        domain = self%input_parallel_info%get_domain(self%maxlevel)

#ifdef HAVE_CUDA

        if (full_cuda_evaluation) then

            ! ibox is the number of the box in the entire area
            do ibox = domain(1), domain(2) 
                ! get streamcontainer handling the box
                !box_stream_container = GBFMMCoulomb3D_get_box_stream_container_cuda(self%cuda_interface, ibox)

                ! box number is the order number of the box handled by this processor
                box_number = ibox - domain(1) + 1

                ! get the coulomb operator corresponding to current box
                coul => self%nearfield_coulomb_operator(box_number)
            
                ! get the limits of the nearfield for ibox
                nearfield_cell_index_limits = self%nearfield_limits(:, :, box_number)
                nearfield_cube_limits = self%input_parallel_info%get_cube_ranges( &
                                            nearfield_cell_index_limits)

                ! get the limits of the box area for ibox
                box_cell_index_limits = self%output_parallel_info%get_box_cell_index_limits(ibox)
                box_cube_limits = self%output_parallel_info%get_cube_ranges( &
                                            box_cell_index_limits)

                
                dims = coul%get_dims()
                tmp1_limits(1, :) = [1, 1, 1]
                tmp1_limits(2, :) = [dims(1, 2), dims(2, 1), dims(3, 1)]
                tmp2_limits(1, :) = [1, 1, 1]
                tmp2_limits(2, :) = [dims(1, 2), dims(2, 2), dims(3, 1)]
                

                if (coul%cut_end_borders(X_)) box_cube_limits(2, X_) = box_cube_limits(2, X_) -1
                if (coul%cut_end_borders(Y_)) box_cube_limits(2, Y_) = box_cube_limits(2, Y_) -1
                if (coul%cut_end_borders(Z_)) box_cube_limits(2, Z_) = box_cube_limits(2, Z_) -1

                ! get the subcubes corresponding to nearfield and box area
                nearfield_cube = self%input_cuda_cube%get_subcube(nearfield_cube_limits)
                box_cube = self%output_cuda_cube%get_subcube(box_cube_limits)
                tmp1 = self%cuda_tmp1%get_subcube(tmp1_limits)
                tmp2 = self%cuda_tmp2%get_subcube(tmp2_limits)

                ! and finally do the operation
                call coul%transform_cuda_cube(nearfield_cube, box_cube, coul%cuda_blas, &
                                              coul%cuda_fx, coul%cuda_fy, coul%cuda_fz, tmp1, tmp2)
                call CUDASync_all()

                ! do the clean up
                call tmp1%destroy()
                call tmp2%destroy()
                call box_cube%destroy()
                call nearfield_cube%destroy()
            end do
            call CUDASync_all()
            !call self%calculate_coda(rho%cube, potential%cube)
        end if
        
#else 
        full_cuda_evaluation = .FALSE.
#endif
        if (.not. full_cuda_evaluation) then
            ! ibox is the number of the box in the entire area
            do ibox = domain(1), domain(2) 
                ! box number is the order number of the box handled by this processor
                box_number = ibox - domain(1) + 1

                ! get the coulomb operator corresponding to current box
                coul => self%nearfield_coulomb_operator(box_number)

                ! get the limits of the nearfield for ibox
                nearfield_cell_index_limits = self%nearfield_limits(:, :, box_number)
                limits_nearfield = self%input_parallel_info%get_nearfield_limits(ibox)

                ! nearfield self-potential potential evaluation: the result is stored to 'nearfield_potential'
                call coul%operate_on(rho, nearfield_potential, nearfield_cell_index_limits, only_cube = .TRUE.) 

                ! to be able to assign the pointer type to the array, let's handle it as
                ! a Function3D type that it should be
                select type(nearfield_potential)
                    type is (Function3D)
                        potential%nearfield_potential(ibox) = nearfield_potential
                        !call bigben%split("Farfield Inject")
                        !potential%nearfield_potential(ibox)%cube = potential%nearfield_potential(ibox)%cube + &
                        !    self%evaluate_potential_le(solid_harmonics, potential%nearfield_potential(ibox)%grid, &
                        !                               self%input_parallel_info%get_box_center(ibox, self%maxlevel), &
                        !                               potential%farfield_potential(:, ibox))
                            !potential%evaluate_farfield_potential_grid(solid_harmonics, &
                            !                                    potential%nearfield_potential(ibox)%grid, &
                            !                                    ibox)
                        !call bigben%stop()
                        
                        ! add ibox's contribution to the total cube contaminants
                        !if (calculate_bubbles .and. potential%bubbles%get_nbub() > 0) then 
                            !call bigben%split("Contaminants")
                        !    potential%nearfield_potential(ibox)%bubbles = potential%bubbles 
                        !    box_cube_contaminants = potential%nearfield_potential(ibox)%get_cube_contaminants( &
                        !                                potential%taylor_order, self%input_parallel_info%get_box_limits(ibox))
                        !    cube_contaminants = cube_contaminants + box_cube_contaminants
                        !    deallocate(box_cube_contaminants)
                            !call bigben%stop()
                        !end if
                        
                end select

                ! deallocate memory of the temporary variable 'nearfield_potential'
                call nearfield_potential%destroy()
                deallocate(nearfield_potential) 
                nullify(coul)
            end do
        end if

        

    end subroutine

    subroutine GBFMMCoulomb3D_simplify(self, potential)
        class(GBFMMCoulomb3D), target   :: self
        !> result object to which the result farfield potential is stored
        class(GBFMMFunction3D), intent(inout) &
                                        :: potential
        integer                         :: ibox, cube_limits(2, 3), box_cell_limits(2, 3), domain(2), box_number
        type(RealRegularSolidHarmonics) :: solid_harmonics
        logical                         :: full_cuda_evaluation

#ifdef HAVE_CUDA
        if (self%cuda_inited) then
            full_cuda_evaluation = .TRUE.
        else
            full_cuda_evaluation = .FALSE.
        end if
#else
        full_cuda_evaluation = .FALSE.
#endif
        
        ! ibox is the number of the box in the entire area
        if (full_cuda_evaluation) then
#ifdef HAVE_CUDA
            call GBFMMCoulomb3D_evaluate_potential_le_cuda(self%cuda_interface, &
                     potential%farfield_potential, self%output_cuda_cube%cuda_interface)
            call CUDASync_all()
#endif
        else
            ! get starting and ending indices of this processors domain
            domain = self%output_parallel_info%get_domain(self%maxlevel)
    
            solid_harmonics = RealRegularSolidHarmonics(self%lmax)
            do ibox = domain(1), domain(2) 
                ! box number is the order number of the box handled by this processor
                box_number = ibox - domain(1) + 1

                ! get the starting and ending cells indexes of the box
                box_cell_limits = self%output_parallel_info%get_box_cell_index_limits(ibox, self%maxlevel, global = .FALSE.)

                ! get the cube limits for the box
                cube_limits = self%output_parallel_info%get_cube_ranges(box_cell_limits)
                
                           

                potential%cube(cube_limits(1, X_) : cube_limits (2, X_),   &
                            cube_limits(1, Y_) : cube_limits (2, Y_),   &
                            cube_limits(1, Z_) : cube_limits (2, Z_)) = &
                    potential%cube(cube_limits(1, X_) : cube_limits (2, X_),   &
                            cube_limits(1, Y_) : cube_limits (2, Y_),   &
                            cube_limits(1, Z_) : cube_limits (2, Z_)) + &
                    self%evaluate_potential_le_box(solid_harmonics, self%nearfield_coulomb_operator(box_number)%gridout, &
                                                    self%output_parallel_info%get_box_center(ibox, self%maxlevel), &
                                                    potential%farfield_potential(:, ibox))

            end do
            call solid_harmonics%destroy()
        end if

            
    end subroutine

    !> Construct the numerical potentials for boxes ista:iend
    subroutine GBFMMCoulomb3D_nearfield_operators(self, potential, rho)
        class(GBFMMCoulomb3D), target:: self
        !> result object to which the result farfield potential is stored
        class(GBFMMFunction3D), intent(inout) &
                                     :: potential
        ! charge density rho
        type(Function3D), intent(in) :: rho

        integer                      :: boxveci(3)
        integer                      :: boxvecj(3)
        integer                      :: ibox, jbox, box_number, j
        integer, allocatable         :: nearest_neighbors(:)
        integer                      :: domain(2)

        ! the result potential for each box is stored temporarily in the
        ! nearfield_potential
        class(Function3D), allocatable   :: nearfield_potential

        ! pointer to a operator used for each box
        type(Coulomb3D), pointer     :: coul

   
        
        ! box limits for box ibox and its nearfield as cell number integers
        integer                             :: limits_nearfield(2,3), limits_jbox(2,3)
    
        ! get starting and ending indices of this processors domain
        domain = self%input_parallel_info%get_domain(self%maxlevel)
   
        ! ibox is the number of the box in the entire area
        do ibox = domain(1), domain(2)
            ! box number is the order number of the box handled by this processor
            box_number = ibox - domain(1) + 1

            nearest_neighbors = self%input_parallel_info%get_nearest_neighbors_indices(ibox, self%maxlevel)
            ! get the limits of the nearfield for ibox
            do j = 1, size(nearest_neighbors)
                jbox = nearest_neighbors(j)
                
                ! get the limits of the nearest neighbor box
                limits_jbox = self%input_parallel_info%get_box_cell_index_limits(jbox, self%maxlevel)
 
                ! get the coulomb operator corresponding to current box
                coul => self%nearfield_coulomb_operators(j, box_number)                

                ! nearfield self-potential potential evaluation
                call coul%operate_on(rho, nearfield_potential, limits_jbox)
                ! to be able to assign the pointer type to the array, let's handle it as
                ! a Function3D type that it should be
                select type(nearfield_potential)
                    type is (Function3D)
                        potential%nearfield_potential(ibox) = potential%nearfield_potential(ibox) &
                                                              + nearfield_potential
                end select
                call nearfield_potential%destroy()
                deallocate(nearfield_potential)

            end do
            deallocate(nearest_neighbors)
            
            ! finalize timing of this loop
        enddo


    end subroutine

    

    subroutine GBFMMCoulomb3D_calculate_farfield_bubbles_potential(self, all_bubbles, potential)
        class(GBFMMCoulomb3D)                :: self
        type(Bubbles), intent(in)            :: all_bubbles
        !> result object to which the result farfield potential is stored
        type(GBFMMFunction3D), intent(inout) :: potential
        !> 
        real(REAL64)                         :: nearfield_limits(2, 3), box_limits(2, 3)
        type(Bubbles)                        :: farfield_bubbles, box_bubbles
        integer                              :: ibub, jbub, domain(2), index_bubu
        integer, allocatable                 :: ibubs(:), jbubs(:)

        type(InteractionMatrix)              :: interaction_matrix
        ! matrix in which the interaction matrix is evaluated
        real(REAL64), allocatable            :: Tmx(:,:,:), Tmx_bubu(:,:,:)
        real(REAL64)                         :: distances(3, all_bubbles%get_nbub()), center_ibox(3)
        real(REAL64)                         :: bubu_distances(3,  all_bubbles%get_nbub() * &
                                                                 all_bubbles%get_nbub())

        integer                              :: ibox, i, j

        interaction_matrix = InteractionMatrix(self%lmax)

        ! get the starting and ending box indices for the current processor
        domain = self%input_parallel_info%get_domain(self%input_parallel_info%maxlevel)
        do ibox = domain(1), domain(2)

            ! calculate the position of the currently handled box
            ! the position is calculated by the center of the box 
            center_ibox = self%input_parallel_info%get_box_center(ibox, self%input_parallel_info%maxlevel)
            nearfield_limits = self%input_parallel_info%get_nearfield_limits(ibox)
            box_limits       = self%input_parallel_info%get_box_limits(ibox)

            ! get global indices bubbles residing in local farfield (no copying of parameters is made)
            farfield_bubbles = all_bubbles%subtract_bubbles(all_bubbles%get_sub_bubbles(nearfield_limits))
            box_bubbles = all_bubbles%get_sub_bubbles(box_limits)
            ibubs = farfield_bubbles%get_ibubs()
            jbubs = box_bubbles%get_ibubs()

            do i = 1, farfield_bubbles%get_nbub()
                distances(:, i) = farfield_bubbles%get_centers(i) - center_ibox 

                ! get distances between farfield bubbles and box bubbles
                do j = 1, box_bubbles%get_nbub()
                    index_bubu = box_bubbles%get_nbub()*(j - 1) + i
                    bubu_distances(:, index_bubu) = farfield_bubbles%get_centers(i) - box_bubbles%get_centers(j)
                end do
            end do


            ! evaluate interaction matrix for the distances between the box center and the farfield bubbles
            Tmx = interaction_matrix%eval(distances(:, : farfield_bubbles%get_nbub()))
            
            Tmx_bubu = interaction_matrix%eval( &
                         bubu_distances(:, : farfield_bubbles%get_nbub() * box_bubbles%get_nbub()))

            
            do i = 1, farfield_bubbles%get_nbub()
                ibub = ibubs(i)
                potential%farfield_potential(:, ibox) = potential%farfield_potential(:, ibox) & 
                                                      + xmatmul(Tmx(:,:,i),                   &
                                                        self%bubble_multipole_moments(:, ibub))
                ! calculate farfield bubble-bubble potential for each bubble residing in this box
                do j = 1, box_bubbles%get_nbub()
                    index_bubu = box_bubbles%get_nbub()*(j - 1) + i
                    jbub = jbubs(i)
                    potential%bubbles_farfield_potential(:, jbub) = potential%bubbles_farfield_potential(:, jbub) & 
                                                      + xmatmul(Tmx_bubu(:,:, index_bubu),                   &
                                                        self%bubble_multipole_moments(:, ibub))
                end do
            enddo

            deallocate(Tmx)
            deallocate(Tmx_bubu)
            deallocate(jbubs)
            deallocate(ibubs)
            call box_bubbles%destroy()
            call farfield_bubbles%destroy()
        end do
        call interaction_matrix%destroy()
        


    end subroutine 
    
    subroutine GBFMMCoulomb3D_init_interaction_matrices(self)
        class(GBFMMCoulomb3D),   intent(inout) :: self
        ! number of boxes in child levels for box at level numlevel and
        ! box at level numlevel - 1
        integer                                :: box_i, level, domain(2), boxes_tot
        type(InteractionMatrix)                :: interaction_matrix
     

        interaction_matrix = InteractionMatrix(self%lmax)
        ! boxes_tot is total number of boxes in all levels
        boxes_tot = self%input_parallel_info%get_total_box_count()
        allocate(self%interaction_matrices(boxes_tot))
        allocate(self%farfield_indices(boxes_tot))
        ! get the starting and ending box indices for the current processor
        do level = self%start_level, self%input_parallel_info%maxlevel
            domain = self%input_parallel_info%get_domain(level)
            do box_i = domain(1), domain(2)
                call self%init_interaction_matrix(interaction_matrix, box_i, level)
            end do
        end do
        call interaction_matrix%destroy()
    end subroutine 

    subroutine GBFMMCoulomb3D_init_interaction_matrix(self, interaction_matrix, box_i, numlevel)
        class(GBFMMCoulomb3D),   intent(inout) :: self
        type(InteractionMatrix), intent(in)    :: interaction_matrix
        !> Handled box number without offset caused by child levels
        integer,                 intent(in)    :: box_i
        !> number of recursion at which the potential is calculated
        integer,                 intent(in)    :: numlevel
        ! maximum number of local farfield boxes is 189
        integer, parameter              :: MAX_LFF = 189
        
        ! center of the box i
        real(REAL64)                    :: center_ibox(3), center_jbox(3)
        ! distances from box i to the local farfield box
        real(REAL64)                    :: distances(3, MAX_LFF)
        
        ! temporary variable to store the indices of the local farfield
        ! boxes of box i
        integer, allocatable            :: local_farfield_indices(:)
        
        ! array to store the box numbers (including offset) of local farfield boxes 
        integer                         :: interactions(MAX_LFF)
        ! number of boxes in child levels for box at level numlevel and
        ! box at level numlevel - 1
        integer                         :: offset, parent_offset
        
        ! counters for local farfield boxes at the current and parent level
        integer                         :: n_lff, n_lff_parent, ibox, box_j, jbox
        
        interactions = 0

        ! calculate the position of the currently handled box
        ! the position is calculated by the center of the box 
        center_ibox = self%input_parallel_info%get_box_center(box_i, numlevel)
        
        
            
        ! Determine what boxes at numlevel belong to the local farfield of ibox
        ! n_lff : counter of the l(ocal) f(ar) f(ield) boxes 
        n_lff = 0 
        n_lff_parent = 0
        
        ! get the offsets in the result and multipole matrix caused by child level
        ! boxes at level numlevel and numlevel - 1 (parent)
        offset = self%input_parallel_info%get_level_offset(numlevel)
        parent_offset = self%input_parallel_info%get_level_offset(numlevel - 1)
        
        
        ! Get the number of box i with the offset
        ibox = box_i + offset

        ! if we are in the 'adaptive_mode' then only get the local farfield boxes which parents
        ! contain nearest neighbors 
        if (self%adaptive_mode) then
            local_farfield_indices = self%input_parallel_info%get_local_farfield_indices(box_i, numlevel, 1)

        ! otherwise get the children of the nearest neighbors of the parent of the ibox that are not
        ! nearest neighbors with ibox
        else
            local_farfield_indices = self%input_parallel_info%get_local_farfield_indices(box_i, numlevel)
        end if


        do box_j = 1, size(local_farfield_indices)
            ! get the box_j number in the entire domain
            jbox = local_farfield_indices(box_j)
                
            ! counter of local farfield neighbors
            n_lff = n_lff + 1
            center_jbox = self%input_parallel_info%get_box_center(jbox, numlevel)
            ! calculate distance between ibox and jbox
            distances(:, n_lff) = center_jbox - center_ibox
                
            ! store the index of the current box for later use
            interactions(n_lff) = jbox + offset             

        enddo
        deallocate(local_farfield_indices)

        if (self%adaptive_mode) then
            ! get the parent level boxes that are neighbors of the parent of the 'ibox' but do not contain
            ! nearest neighbors of 'ibox'
            local_farfield_indices = self%input_parallel_info%get_local_farfield_indices(box_i, numlevel, 2)
            do box_j = 1, size(local_farfield_indices)
                ! get the box_j number in the entire domain
                jbox = local_farfield_indices(box_j)
            
                ! counter of local farfield neighbors
                n_lff_parent = n_lff_parent + 1
                ! calculate the distance between ibox's parent and jbox and add to distance table
                center_jbox = self%input_parallel_info%get_box_center(jbox, numlevel - 1)
                distances(:, n_lff + n_lff_parent) = center_jbox - center_ibox
                    
                ! store the index of the current box for later use
                interactions(n_lff + n_lff_parent) = jbox + parent_offset             

            enddo
            deallocate(local_farfield_indices)
        end if 
           
        ! Evaluate interaction matrix for the  distances between box i and local farfield boxes
        self%interaction_matrices(ibox)%p = interaction_matrix%eval(distances(:, : n_lff + n_lff_parent))
        self%farfield_indices(ibox)%p = interactions(: n_lff + n_lff_parent)
    end subroutine
    
    !> Subroutine to calculate the local farfield contribution to the coulomb potential
    !! at level 'numlevel', the result potential is stored to 'potential' object
    subroutine GBFMMCoulomb3D_calculate_local_farfield_potential(self, potential, numlevel)
        class(GBFMMCoulomb3D)           :: self
        !> result object to which the result farfield potential is stored
        type(GBFMMFunction3D), intent(inout) :: potential
        !> number of recursion at which the potential is calculated
        integer, intent(in)             :: numlevel
        ! number of boxes in child levels for box at level numlevel and
        ! box at level numlevel - 1
        integer                         :: offset, parent_offset

        ! numbers of boxes i and j without offset caused by the child levels
        integer                         :: box_i, box_j ! box index

        ! numbers of boxes i and j with offset caused by child levels
        integer                         :: ibox, j
        

        ! variable to store starting and ending box indices for the current processor
        integer                         :: domain(2)

        !real(REAL64)                    :: distances_bubo(3, MAX_LFF)
        !type(InteractionMatrix)         :: interaction_matrix
     

        !interaction_matrix = InteractionMatrix(self%lmax)
        ! get the offsets in the result and multipole matrix caused by child level
        ! boxes at level numlevel and numlevel - 1 (parent)
        offset = self%input_parallel_info%get_level_offset(numlevel)
        
        ! get the starting and ending box indices for the current processor
        domain = self%input_parallel_info%get_domain(numlevel)
        do box_i = domain(1), domain(2)
            ! Get the number of box i with the offset
            ibox = box_i + offset
            
            ! Calculate the local farfield potential V = \sum_o V_i = \sum_i T_j \cdot Q 
            do j = 1, size(self%farfield_indices(ibox)%p)
                potential%farfield_potential(:, ibox) = potential%farfield_potential(:, ibox) & 
                                                      + xmatmul(self%interaction_matrices(ibox)%p(:,:,j),          &
                                                        self%multipole_moments(:, self%farfield_indices(ibox)%p(j)))
            enddo
            
        enddo
        !call interaction_matrix%destroy()
        return
    end subroutine
    
    

    ! If a multipole moment is known with respect to some center, the
    ! corresponding moment in some other center can be known without recomputing the multipole
    ! integral.
    ! 
    ! That is, one can translate multipole moments between expansion centers. 
    ! This transformation is most compactly expressed as a matrix-vector product
    !
    !  q(N) = W(O-N) q(O)
    !
    !> This function translates previously calculated multipoles to the parent level
    !! Deprecated, use translate_potential of GBFMMFunction3D 
    subroutine GBFMMCoulomb3D_translate_multipoles(self, offset, numlevel)
        class(GBFMMCoulomb3D)  :: self
        integer, intent(in)         :: offset(:)
        integer, intent(in)         :: numlevel

        type(TranslationMatrix)     :: translation_matrix
        real(REAL64)                :: child_multipole_moments((self%lmax + 1)**2)

        integer                     :: ibox
        integer                     :: child_box
        integer                     :: parent_box
        integer                     :: child_box_vector(3)
        integer                     :: domain(2)

        real(REAL64)                :: child_position(3), child_size(3)
        real(REAL64)                :: parent_position(3), parent_size(3)

        translation_matrix = TranslationMatrix(self%lmax)

        ! size of a child box
        child_size = self%rootlen/(2**(numlevel+1))
        ! size of a parent box
        parent_size = self%rootlen/(2**numlevel)
        ! starting and ending indices of this processors domain
        domain = self%input_parallel_info%get_domain(numlevel + 1)
         
        do ibox = domain(1), domain(2)
            ! get the integer vector of ibox
            child_box_vector = self%input_parallel_info%get_box_vector(ibox, numlevel + 1)
            ! the position of the center of the child box
            child_position = self%boxmin + (child_box_vector + 0.5d0) * child_size
            ! the position of the center of the parent box
            parent_position = self%boxmin                                                                    &
                         + (self%input_parallel_info%get_parent_box_vector(child_box_vector, numlevel +1) + 0.5d0) &
                         * parent_size

            ! child box index with offset
            child_box = offset(numlevel + 1) + ibox
            ! parent box index with offset
            parent_box = offset(numlevel) + self%input_parallel_info%get_parent_index(ibox, numlevel + 1)

            ! copy existing child box multipole moment to the variable
            child_multipole_moments = self%multipole_moments(:, child_box)
    
            ! translate the potential of the child box to the position
            ! of the parent box
            call translation_matrix%apply(child_multipole_moments, from=child_position,  &
                                          to=parent_position)
            ! add the value of the child box multipole moment to the parent box
            ! multipole moment   
            self%multipole_moments(:, parent_box) = self%multipole_moments(:,parent_box) &
                                                    + child_multipole_moments

        enddo
        call translation_matrix%destroy()
        return
    end subroutine
    
    !> Translates local farfield potential from parent's position to child's and adds
    !! it to child potential
    subroutine GBFMMCoulomb3D_translate_potential(self, potential, numlevel)
        class(GBFMMCoulomb3D)      :: self
        !> result object in which the child and parent level potentials are
        type(GBFMMFunction3D), intent(inout) :: potential
        !> number of child level to which parent level potential is translated  
        integer, intent(in)               :: numlevel
        ! variable to store the number of boxes stored on child levels
        ! for 'numlevel' and 'numlevel - 1'
        integer                           :: offset, parent_offset

        type(TranslationMatrix)           :: translation_matrix

        ! temporary variable to store potential of the currently handled box's parent
        real(REAL64)                      :: parent_potential((self%lmax+1)**2)

        ! number of box in numlevel and numlevel + 1
        integer                           :: ibox, parent_index
        ! variables to store the ibox + offset and parent_index + parent_offset
        integer                           :: child_box, parent_box

        ! variable to store the first and last index that belong to
        ! current processor at level 'numlevel'
        integer                           :: domain(2)

        ! variables to store the cartesian coordinates of 
        ! centers of the box and its parent
        real(REAL64)                      :: parent_position(3), child_position(3)

        translation_matrix = TranslationMatrix(self%lmax)
        call translation_matrix%transpose()
        offset = self%input_parallel_info%get_level_offset(numlevel)
        parent_offset = self%input_parallel_info%get_level_offset(numlevel - 1)
        ! get the domain of current processor
        domain = self%input_parallel_info%get_domain(numlevel)
        ! From parent to children
        do ibox = domain(1), domain(2)
            ! the position of the center of the child box
            child_position = self%input_parallel_info%get_box_center(ibox, numlevel)
            ! the position of the center of the parent box
            parent_index = self%input_parallel_info%get_parent_index(ibox, numlevel)
            parent_position = self%input_parallel_info%get_box_center(parent_index, numlevel-1)

            ! child box index with offset
            child_box = offset + ibox 
            ! parent box index with offset
            parent_box = parent_offset + parent_index

            ! copy the potential of the parent box
            parent_potential = potential%farfield_potential(:, parent_box)

            ! Translate potential of the parent box to the position of the child box
            call translation_matrix%apply(parent_potential, from=child_position, to=parent_position)

            ! Add translated child level potential of parent to child_box's potential 
            potential%farfield_potential(:, child_box) = potential%farfield_potential(:, child_box) &
                                                         + parent_potential
        enddo
        call translation_matrix%destroy()

    end subroutine
    

    pure subroutine GBFMMCoulomb3D_set_transformation_weights(self, weights) 
        !> Input function
        class(GBFMMCoulomb3D), intent(inout) :: self
        real(REAL64),          intent(in)    :: weights(:)
        integer                              :: ibox, domain(2), box_number

        domain = self%input_parallel_info%get_domain(self%maxlevel)
        do ibox = domain(1), domain(2)
            box_number = ibox - domain(1) + 1
            call self%nearfield_coulomb_operator(box_number)%set_transformation_weights(weights)
        end do
        self%w = weights
    end subroutine 

    subroutine GBFMMCoulomb3D_calculate_farfield_contaminants(self, potential) 
        class(GBFMMCoulomb3D), intent(in)          :: self
        type(GBFMMFunction3D), intent(inout)       :: potential
        integer                                    :: domain(2)
        integer                                    :: ibox, m, l, nlip
        type(Bubbles)                              :: bubs
        type(Bubbles)                              :: box_bubbles
        integer                                    :: length  
        
        real(REAL64)                 :: farfield_contaminants(         &
                                           (potential%taylor_order+1)* &
                                           (potential%taylor_order+2)*(potential%taylor_order+3)/6,  &
                                                        potential%bubbles%get_nbub_global())
        farfield_contaminants = 0.0d0
        domain = self%input_parallel_info%get_domain(self%maxlevel)
        do ibox = domain(1), domain(2)
            box_bubbles = potential%bubbles%get_sub_bubbles&
                (self%input_parallel_info%get_box_limits(ibox))

            ! Check if there are bubbles within the box
            if (box_bubbles%get_nbub() > 0) then
                ! transform the farfield potential to bubbles
                !bubs = potential%get_farfield_bubbles(ibox)

                ! get the contaminants of 'bubs' to 'box_bubbles'
                farfield_contaminants = farfield_contaminants + box_bubbles &
                     %get_contaminants(bubs, potential%taylor_order)
                call bubs%destroy()
            end if
            call box_bubbles%destroy()
        end do
        call self%input_parallel_info%sum_matrix(farfield_contaminants)
        potential%cube_contaminants = potential%cube_contaminants + farfield_contaminants
 
    end subroutine

    !> Evaluate Coulomb potential from 'local_expansion' centered at 'reference_point'
    !! at every point of the 'grid'. This function makes the choice between CUDA and CPU 
    !! options.
    function GBFMMCoulomb3D_evaluate_potential_le_box(self, harmo, grid, reference_point,  &
                 local_expansion) result(result_cube) 
        class(GBFMMCoulomb3D),           intent(in)    :: self
        !> RealSphericalHarmonics object initialized to lmin: 0, lmax: self%lmax
        type(RealRegularSolidHarmonics),      intent(in) :: harmo
        !> Grid object describing the evaluation area.
        type(Grid3D),                      intent(inout) :: grid
        !> The origo of the evaluation area, used in solid harmonics
        !! determination
        real(REAL64),                      intent(in)    :: reference_point(3)
        !> The local expansion being evaluated
        real(REAL64),                      intent(in)    :: local_expansion((self%lmax+1)**2)
        !> The result cube
        real(REAL64)                                     :: result_cube(grid%axis(X_)%get_shape(), &
                                                                        grid%axis(Y_)%get_shape(), &
                                                                        grid%axis(Z_)%get_shape())

        call self%evaluate_potential_le_box_cpu(harmo, grid, reference_point, &
                                                    local_expansion, result_cube)
    end function


    !> Evaluate Coulomb potential from 'local_expansion' centered at 'reference_point'
    !! at every point of the 'grid' using CPU.
    subroutine GBFMMCoulomb3D_evaluate_potential_le_box_cpu(self, harmo, grid, reference_point, &
                                                          local_expansion, result_cube)
        class(GBFMMCoulomb3D),             intent(in)    :: self
        !> RealSphericalHarmonics object initialized to lmin: 0, lmax: self%lmax
        type(RealRegularSolidHarmonics),   intent(in)    :: harmo
        !> Grid object describing the evaluation area.
        type(Grid3D),                      intent(in)    :: grid
        !> The origo of the evaluation area, used in spherical harmonics and bessel value
        !! determination
        real(REAL64),                      intent(in)    :: reference_point(3)
        !> The evaluated local expansion
        real(REAL64),                      intent(in)    :: local_expansion((self%lmax+1)**2)
        !> The result cube
        real(REAL64),                      intent(inout) :: result_cube(grid%axis(X_)%get_shape(), &
                                                                        grid%axis(Y_)%get_shape(), &
                                                                        grid%axis(Z_)%get_shape())

        real(REAL64)                                     :: solid_harmonics_cube(grid%axis(X_)%get_shape(), &
                                                                                 grid%axis(Y_)%get_shape(), &
                                                                                 grid%axis(Z_)%get_shape(), &
                                                                                 (self%lmax+1)**2)
        real(REAL64)                                     :: normalization_factor
        integer                                          :: l, m

        
        result_cube = 0.0d0

        ! evaluate spherical harmonics at cube positions and reshape the result as cube shaped arrays
        ! for each (angular momentum) l, m pair
        solid_harmonics_cube = harmo%eval_grid(grid, reference_point)
        !normalization_factor = 1.0d0
        ! go through all quantum number pairs (l, m) and multiply the spherical harmonics with first
        ! modified bessel function values and the local expansion values
        do l = 0, self%lmax
            !if (self%normalization /= 2) then
            !    normalization_factor = (2*l+1)/dble(4*pi)
            !end if
            do m = -l, l
                result_cube = result_cube + &
                    solid_harmonics_cube(:, :, :, lm_map(l, m)) * &
                    local_expansion(lm_map(l, m))! * normalization_factor
            end do
        end do  
    end subroutine


    subroutine GBFMMCoulomb3D_destroy(self)
        class(GBFMMCoulomb3D), intent(inout) :: self
        integer                              :: i, j
   
#ifdef HAVE_CUDA
        call self%cuda_destroy()
        call check_cuda_errors_from_fortran(1)
#endif
        if(allocated(self%global_serial_info)) then
            call self%global_serial_info%destroy()
            deallocate(self%global_serial_info)
        end if
        if(allocated(self%multipole_moments)) deallocate(self%multipole_moments)
        if(allocated(self%bubble_multipole_moments)) deallocate(self%bubble_multipole_moments)
        if(allocated(self%nearfield_limits))  deallocate(self%nearfield_limits)
        if(allocated(self%nearfield_coulomb_operator)) then

            do i = 1, size(self%nearfield_coulomb_operator)
                call self%nearfield_coulomb_operator(i)%destroy()
            end do
            deallocate(self%nearfield_coulomb_operator)
        end if 
        if(allocated(self%nearfield_coulomb_operators)) then
            do i = 1, size(self%nearfield_coulomb_operators, 1)
                do j = 1, size(self%nearfield_coulomb_operators, 2)
                    call self%nearfield_coulomb_operators(i, j)%destroy()
                end do
            end do
            deallocate(self%nearfield_coulomb_operators)
        end if 
        if(allocated(self%input_parallel_infos)) then
            do i = 1, size(self%input_parallel_infos)
                call self%input_parallel_infos(i)%destroy()
            end do
            deallocate(self%input_parallel_infos)
        end if 
#ifdef HAVE_CUDA
        call check_cuda_errors_from_fortran(1)
#endif
        if(allocated(self%output_parallel_infos)) then
            do i = 1, size(self%output_parallel_infos)
                call self%output_parallel_infos(i)%destroy()
            end do
            deallocate(self%output_parallel_infos)
        end if 
        
        if(allocated(self%input_grids)) then
            do i = 1, size(self%input_grids)
                call self%input_grids(i)%destroy()
            end do
            deallocate(self%input_grids)
        end if 
        
        if(allocated(self%output_grids)) then
            do i = 1, size(self%output_grids)
#ifdef HAVE_CUDA
                call self%output_grids(i)%cuda_destroy()
#endif
                call self%output_grids(i)%destroy()
            end do
            deallocate(self%output_grids)
        end if 
#ifdef HAVE_CUDA
        call check_cuda_errors_from_fortran(1)
#endif
        
        if(allocated(self%interaction_matrices)) then
            do i = 1, size(self%interaction_matrices)
                if (allocated(self%interaction_matrices(i)%p)) &
                    deallocate(self%interaction_matrices(i)%p)
                if (allocated(self%farfield_indices(i)%p)) &
                    deallocate(self%farfield_indices(i)%p)
            end do
            deallocate(self%interaction_matrices)
            deallocate(self%farfield_indices)
        end if 
        nullify(self%input_parallel_info)
        nullify(self%output_parallel_info)
        nullify(self%result_parallelization_info)
        nullify(self%gridin)
        nullify(self%gridout)
        if (allocated(self%w))  deallocate(self%w)
#ifdef HAVE_CUDA
        call check_cuda_errors_from_fortran(1)
#endif
        call self%nuclear_potential%destroy()
#ifdef HAVE_CUDA
        call check_cuda_errors_from_fortran(1)
#endif
    end subroutine
    
    
    function function3d_boxed_density(gen, maxlevel, ranges, stepmax, nlip, grid_type) result(rho)
        class(Generator), intent(in)        :: gen
        !> Charge density generator
        integer, intent(in), optional       :: maxlevel
        !> Maximum level of box division
        real(REAL64), intent(in)            :: ranges(2, 3)
        !> Starting and ending points of the cube
        real(REAL64), intent(in)            :: stepmax
        !> Maximum stepsize
        integer, intent(in)                 :: nlip
        integer(int32), intent(in)          :: grid_type
        !> Number of Lagrange interpolation polynomials per cell

        ! returned 3d function
        class(Function3D), allocatable      :: rho
        ! final grid
        type(Grid3D)                        :: grid

        integer                             :: iaxis
        integer                             :: ncell(3)
        integer                             :: cellmodulus
  
        ! grid stepsizes
        real(REAL64)                        :: stepsizes(3)
        real(REAL64), allocatable           :: stepsize_x(:)
        real(REAL64), allocatable           :: stepsize_y(:)
        real(REAL64), allocatable           :: stepsize_z(:)
        
        ! initialize all step sizes to the 'stepmax'
        stepsizes = stepmax
        ! get the axis step sizes in a way that each axis has cells divisible by 2**maxlevel
        do iaxis = 1, 3
            ncell(iaxis) = ceiling((ranges(2, iaxis) - ranges(1, iaxis)) /  (((nlip-1) * stepsizes(iaxis))))
            ! check how many cells away we are from fulfilling the  condition mentioned above
            cellmodulus = mod(ncell(iaxis), 2**maxlevel)
            ! adjust the stepsize so that the condition is fulfilled
            ncell(iaxis) = (ncell(iaxis) - cellmodulus)
            stepsizes(iaxis) = (ranges(2, iaxis) - ranges(1, iaxis)) / (ncell(iaxis) * (nlip-1))
            ncell(iaxis) = floor((ranges(2, iaxis) - ranges(1, iaxis)) / ((nlip-1)*stepsizes(iaxis))) 
        enddo

        ! allocate the axis stepsizes and give them values
        allocate(stepsize_x(ncell(X_)))
        allocate(stepsize_y(ncell(Y_)))
        allocate(stepsize_z(ncell(Z_)))
        stepsize_x = stepsizes(X_)
        stepsize_y = stepsizes(Y_)
        stepsize_z = stepsizes(Z_)

        ! init grid via Grid3D_init_step constructor
        grid = Grid3D(ranges(1, :), ncell, nlip, stepsize_x, stepsize_y, stepsize_z, grid_type)

        deallocate(stepsize_x, stepsize_y, stepsize_z)
        ! init density in the grid
        allocate(rho, source = gen%gen(grid))
        call grid%destroy()
    end function

end module
