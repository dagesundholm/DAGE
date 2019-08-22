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
module Core_class
    use Settings_class
    use LCAO_m
    use Coulomb3D_class
    use Laplacian3D_class
    use Helmholtz3D_class
    use GaussQuad_class
    use SCFCycle_class
    use SCF_class
    use DIIS_class
    use Grid_class
    use GBFMMLaplacian3D_class
    use GBFMMCoulomb3D_class
    use GBFMMHelmholtz3D_class
    use Potential_class, only: nuclear_repulsion
    use initial_guess_m
    use ConfigurationInteraction_class
    use MemoryLeakChecker_m
    use CoreEvaluator_class
    use Action_class
    use Bubbles_class
    use SCFEnergetics_class
#ifdef HAVE_MPI
    use mpi
#endif
#ifdef HAVE_OMP
    use omp_lib
#endif
#ifdef HAVE_CUDA
    use cuda_m
#endif
    implicit none

    public :: Core


    type :: Core
        type(ProgramSettings), allocatable  :: settings(:)
        type(Structure),       allocatable  :: structures(:)
        type(Action),          allocatable  :: actions(:)
        type(Basis),           allocatable  :: basis_sets(:)
        type(SCFEnergetics),   allocatable  :: scf_energetics(:)
    contains 
        procedure, private :: destroy_orbitals_and_electron_density &
                                                    => Core_destroy_orbitals_and_electron_density 
        procedure, private :: destroy_grids         => Core_destroy_grids
        procedure, private :: init_operators        => Core_init_operators
        procedure, private :: destroy_operators     => Core_destroy_operators
        procedure, private :: get_cube_grid         => Core_get_cube_grid
        procedure, private :: get_bubble_grids      => Core_get_bubble_grids
        procedure, private :: get_orbitals          => Core_get_orbitals
        procedure, private :: validate_settings_id  => Core_validate_settings_id
        procedure, private :: validate_basis_set_id => Core_validate_basis_set_id
        procedure, private :: validate_structure_id => Core_validate_structure_id
        procedure, public  :: run => Core_run
        procedure, private :: input_output_test           => Core_input_output_test
        procedure, private :: optimize_electron_structure => Core_optimize_electron_structure
    end type

contains

!--------------------------------------------------------------------------------------------!
!                     MAIN FUNCTION                                                          !
!--------------------------------------------------------------------------------------------!
!   The following function is the main functionality of the entire program package. Calling  !
!   the function executes all the desired actions specified in 'self%actions'.               !
!--------------------------------------------------------------------------------------------!

    !> Run all the actions of specified in the input 'Core' object.
    function Core_run(self) result(success)
        class(Core),          target      :: self
        type(Structure),      pointer     :: struct
        type(SCFEnergetics)      :: scf_energetics
        integer                  :: iaction, ierr, provided
        logical                  :: action_success, success, initialized
#ifdef BUBBLES_REVISION_NUMBER
        print *, "Bubbles library revision number: ", BUBBLES_REVISION_NUMBER
#endif
        
#ifdef HAVE_MPI
#ifdef HAVE_OMP
        
        !$OMP BARRIER
        !$OMP MASTER
        
        ! check if mpi is initialized, if not initialize it
        call mpi_initialized(initialized, ierr)
        if (.not. initialized) &
            call mpi_init_thread(MPI_THREAD_FUNNELED, provided, ierr)
        
        ! get number of processors and store it to global parameter nproc
        call mpi_comm_size(MPI_COMM_WORLD, nproc, ierr)
        ! get rank of the processor (index in range 0-nproc) and
        ! store it to global parameter iproc
        call mpi_comm_rank(MPI_COMM_WORLD, iproc, ierr)
        !$OMP END MASTER
        !$OMP BARRIER
#else
        ! check if mpi is initialized, if not initialize it
        call mpi_initialized(initialized, ierr)
        if (.not. initialized) &
            call mpi_init(ierr)
        
        ! get number of processors and store it to global parameter nproc
        call mpi_comm_size(MPI_COMM_WORLD, nproc, ierr)
        ! get rank of the processor (index in range 0-nproc) and
        ! store it to global parameter iproc
        call mpi_comm_rank(MPI_COMM_WORLD, iproc, ierr)
#endif 
#else
        nproc = 1
        iproc = 0 
#endif

#ifdef HAVE_CUDA
        streams_per_device = 32 / nproc / CUDA_get_number_of_devices() 
        if (.not. initialized) then
            call streamcontainer_init(stream_container, STREAMS_PER_DEVICE);
            call streamcontainer_enable_peer_to_peer(stream_container)
#ifdef HAVE_CUDA_PROFILING
            call start_cuda_profiling()
#endif
        end if
#endif
        success = .TRUE.
        do iaction = 1, size(self%actions)
            !call check_memory_status_cuda()

            ! check that the structure in in action is valid
            action_success = self%validate_structure_id(self%actions(iaction)%structure_id)
            if (action_success) then
                ! get the structure pointer
                struct => self%structures(self%actions(iaction)%structure_id)

                ! validate that the settings id for action is valid
                action_success = self%validate_settings_id(self%actions(iaction)%settings_id)
                
                ! validate that the basis set id for action is valid
                action_success = self%validate_basis_set_id(struct%basis_set_id)
            end if


            ! if the scf_energetics of action exists copy it to a local variable
            if (action_success .and. self%actions(iaction)%scf_energetics_id /= 0) then
                scf_energetics = self%scf_energetics(self%actions(iaction)%scf_energetics_id)
            end if

            ! perform the desired action
            if (action_success .and. self%actions(iaction)%name == ACTION_OPTIMIZE_ELECTRON_STRUCTURE) then
                action_success = self%optimize_electron_structure( &
                    self%actions(iaction), &
                    struct, &
                    self%settings(self%actions(iaction)%settings_id), &
                    self%basis_sets(struct%basis_set_id), &
                    scf_energetics)
            else if (action_success .and. self%actions(iaction)%name == ACTION_INPUT_OUTPUT_TEST) then
                action_success = self%input_output_test( &
                    self%actions(iaction), &
                    struct, &
                    self%settings(self%actions(iaction)%settings_id), &
                    self%basis_sets(struct%basis_set_id), &
                    scf_energetics)
            end if
            if (.not. action_success) success = .FALSE.
            call scf_energetics%destroy()
            nullify(struct)
            !call check_memory_status_cuda()
        end do
    

        
        ! TODO: move the finalize to somewhere else, as it cannot be called 
        !       multiple times per process
#ifdef HAVE_CUDA
        !call stop_cuda_profiling()
        !call streamcontainer_destroy(stream_container)
#endif
#ifdef HAVE_MPI
#ifdef HAVE_OMP
        
        !$OMP BARRIER
        !$OMP MASTER
        !call mpi_finalize(ierr)
        !$OMP END MASTER
        !$OMP BARRIER
#else
        !call mpi_finalize(ierr)
#endif
#endif
    end function

!--------------------------------------------------------------------------------------------!
!                     ACTION FUNCTIONS                                                       !
!--------------------------------------------------------------------------------------------!
!   The following subroutines and functions are the main functionalities of the program      !
!   called Actions. Each corresponds to a type of an 'Action' specified in the input         !
!   definition.                                                                              !
!--------------------------------------------------------------------------------------------!

    !> Optimizes electron structure for a system with structure 'struct' using
    !! program settings 'settings'.
    function Core_optimize_electron_structure(self, action_, struct, settings, basis_set, scf_energetics) &
             result(success)
        class(Core),           intent(inout)          :: self
        !> The action object defining the action ('optimize_electron_structure')
        !! and the input and output files and their locations
        type(Action)                                  :: action_
        !> Input structure object containing the atom positions and the 
        !! possible molecular orbital coefficients needed to create the the molecule 
        type(Structure),       intent(in)             :: struct
        !> Object containing all the program specific settings from the function3D
        !! settings to operator settings
        type(ProgramSettings), intent(in)             :: settings
        !> Basis set object containing the definitions of the basis functions 
        !! needed to generate the molecular orbitals using the molecular orbital
        !! coefficients stored in struct
        type(Basis),           intent(in), optional   :: basis_set
        !> The input SCFEnergetics type containing the energetic information of 
        !! the previous scf iterations.
        type(SCFEnergetics),   intent(inout)          :: scf_energetics
        class(Coulomb3D),   allocatable               :: coulomb_operator
        class(Laplacian3D), allocatable               :: laplacian_operator
        class(Helmholtz3D), allocatable               :: helmholtz_operator
        
        class(SCFCycle), allocatable                  :: scf_cycle
        class(SCFOptimizer), allocatable              :: scf_optimizer
        type(GaussQuad)                               :: quadrature
        type(Function3D),   allocatable               :: orbitals_a(:), orbitals_b(:), electron_density
        class(ParallelInfo), allocatable              :: orbital_parallel_info, electron_density_parallel_info
        integer                                       :: iorbital, nocc(2), i
        class(ConfigurationInteraction), allocatable  :: ci
        type(Grid3D)                                  :: cubegrid
        type(Grid1D), allocatable                     :: bubble_grids(:)
        type(CoreEvaluator)                           :: core_evaluator
        real(REAL64)                                  :: nuclear_repulsion_energy
        logical                                       :: success

        ! --- INITIALIZE NEEDED OBJECTS --- !
        ! init the global parameters
        print '("----------------------------------------------------------")'
        print '("       Optimizing electron structure of ", a )', trim(struct%name)
        print '("----------------------------------------------------------")'
        bigben = Chrono("Optimize Electron Structure", iproc)
        
        ! get the cube grid and the bubble grids
        call self%get_cube_grid(action_, struct, settings, cubegrid, orbital_parallel_info, electron_density_parallel_info)
        call self%get_bubble_grids(action_, struct, settings, bubble_grids)
        
                                       
        ! create the orbitals
        call self%get_orbitals(action_, struct, settings, orbital_parallel_info, &
                               electron_density_parallel_info, bubble_grids, &
                               basis_set, orbitals_a, orbitals_b, electron_density)
                               
        
        ! init the core evaluator
        core_evaluator = CoreEvaluator(orbitals_a(1))

        ! init the coulomb, laplacian, and helmholtz operators along with the gaussian quadrature
        call self%init_operators(settings, cubegrid, orbital_parallel_info, &
                                 electron_density_parallel_info, &
                                 orbitals_a(1), quadrature, &
                                 coulomb_operator, laplacian_operator, helmholtz_operator)

        ! init the correct SCF cycle
        call SCFCycle_init(settings, struct, basis_set, orbitals_a, orbitals_b, electron_density, &
                           laplacian_operator, coulomb_operator, helmholtz_operator, &
                           core_evaluator, scf_cycle)

        
        ! calculate the nuclear repulsion enrgy
        if (abs(scf_energetics%nuclear_repulsion_energy) < 1d-5) then
            scf_energetics%nuclear_repulsion_energy= &
                                        nuclear_repulsion(orbitals_a(1)%bubbles%get_global_z(), &
                                        orbitals_a(1)%bubbles%get_global_centers())
        end if

        ! init the SCFOptimizer to 'scf_optimizer'
        call SCFOptimizer_init(settings, action_, scf_cycle, scf_energetics, scf_optimizer)
        
        ! --- PERFORM CALCULATION --- !
        ! do the optimization
        call start_memory_follower()
        call scf_optimizer%optimize()
        call stop_memory_follower()
    
        ! do possible post-scf-optimization actions
        if (settings%ci_settings%evaluate_ci) then
            call ConfigurationInteraction_init(ci, scf_cycle, &
                                               settings%ci_settings%singles, &
                                               settings%ci_settings%doubles, &
                                               settings%ci_settings%triples, &
                                               settings%ci_settings%quadruples)
            call ci%optimize()
            call ci%destroy()
        end if

        ! TODO: take the b-spin orbitals into account
        if (settings%scf_settings%type /= 1) then
            call struct%write_structure(action_%output_folder, "structure", scf_cycle%coefficients_a)
        end if
        
        ! --- DESTROY AND DEALLOCATE MEMORY --- !
        ! destroy and deallocate the orbitals and the electron density
        call self%destroy_orbitals_and_electron_density(orbitals_a, orbitals_b, electron_density)

        ! destroy scf-cycle
        if (allocated(scf_cycle)) then
            call scf_cycle%destroy()
            deallocate(scf_cycle)
        end if
        
        ! destroy operators and grids
        call self%destroy_operators(quadrature, coulomb_operator, &
                                    laplacian_operator, helmholtz_operator)
        call self%destroy_grids(cubegrid, bubble_grids, orbital_parallel_info, &
                                electron_density_parallel_info)
        
        ! destroy the scf_optimizer
        if (allocated(scf_optimizer)) then
            call scf_optimizer%destroy()
            deallocate(scf_optimizer)
        end if
        
        call bigben%destroy()
        call core_evaluator%destroy()
        success = .TRUE.
    end function
    
    !> Perform the fortran part of the input/output test 
    function Core_input_output_test(self, action_, struct, settings, basis_set, scf_energetics) &
            result(success)
        class(Core)                                   :: self
        !> The action object defining the action ('optimize_electron_structure')
        !! and the input and output files and their locations
        type(Action)                                  :: action_
        !> Input structure object containing the atom positions and the 
        !! possible molecular orbital coefficients needed to create the the molecule 
        type(Structure),       intent(in)             :: struct
        !> Object containing all the program specific settings from the function3D
        !! settings to operator settings
        type(ProgramSettings), intent(in)             :: settings
        !> Basis set object containing the definitions of the basis functions 
        !! needed to generate the molecular orbitals using the molecular orbital
        !! coefficients stored in struct
        type(Basis),           intent(in), optional   :: basis_set
        !> The input SCFEnergetics type containing the energetic information of 
        !! the previous scf iterations.
        type(SCFEnergetics),   intent(inout)          :: scf_energetics
        
    
        class(ParallelInfo), allocatable              :: orbital_parallel_info, electron_density_parallel_info
        integer                                       :: number_of_orbitals
        type(SCFEnergetics)                           :: inited_scf_energetics               
        class(ConfigurationInteraction), allocatable  :: ci
        type(Grid3D)                                  :: cubegrid
        type(Grid1D), allocatable                     :: bubble_grids(:)
        type(CoreEvaluator)                           :: core_evaluator
        real(REAL64)                                  :: nuclear_repulsion_energy
        logical                                       :: success
        real(REAL64), allocatable                     :: mocoeffs_a(:, :), mocoeffs_b(:, :)
        
        print '("----------------------------------------------------------")'
        print '("       Performing input-output test for ", a )', trim(struct%name)
        print '("----------------------------------------------------------")'
        
        mocoeffs_a = struct%get_orbital_coefficients(basis_set, 0)
        if (.not. settings%scf_settings%restricted) then
            mocoeffs_b = struct%get_orbital_coefficients(basis_set, 1)
            ! write the input structure to output structure, the python program will test the correctness
            ! of the output
            call struct%write_structure(action_%output_folder, "structure", mocoeffs_a, mocoeffs_b)
            number_of_orbitals = size(mocoeffs_a, 2) + size(mocoeffs_b, 2)
            deallocate(mocoeffs_b)
        else
            ! write the input structure to output structure, the python program will test the correctness
            ! of the output
            call struct%write_structure(action_%output_folder, "structure", mocoeffs_a)
            number_of_orbitals = size(mocoeffs_a, 2)
        end if
        deallocate(mocoeffs_a)
        success = .TRUE.
        
        
        ! write the input energetics to output energetics, the python program will test the correctness
        ! of the output
        inited_scf_energetics = SCFEnergetics(scf_energetics, number_of_orbitals, 0)
        call inited_scf_energetics%write_energetics(action_%output_folder, "scf_energetics")
        
    end function

!--------------------------------------------------------------------------------------------!
!                     VALIDATOR FUNCTIONS                                                    !
!--------------------------------------------------------------------------------------------!

    !> Validate that the input structure id does exist within the given Core object.
    !! Returns logical value indicating the correctness of the input.
    function Core_validate_structure_id(self, structure_id) result(valid)
        class(Core), intent(in)      :: self
        integer,     intent(in)      :: structure_id
        logical                      :: valid

        valid = .TRUE.

        if (structure_id <= 0 .or. structure_id > size(self%structures)) then
            print '("INPUT ERROR: Invalid structure_id: ", i0 ,".  Allowed [1, ", i0,"].")', &
                   structure_id, size(self%structures)
            valid = .FALSE.
        end if
    end function

    !> Validate that the input settings id does exist within the given Core object.
    !! Returns logical value indicating the correctness of the input.
    function Core_validate_settings_id(self, settings_id) result(valid)
        class(Core), intent(in)      :: self
        integer,     intent(in)      :: settings_id
        logical                      :: valid

        valid = .TRUE.

        if (settings_id <= 0 .or. settings_id > size(self%settings)) then
            print '("INPUT ERROR: Invalid settings_id: ", i0 ,".  Allowed [1, ", i0,"].")', &
                   settings_id, size(self%settings)
            valid = .FALSE.
        end if
    end function

    !> Validate that the input basis set id does exist within the given Core object.
    !! Returns logical value indicating the correctness of the input.
    function Core_validate_basis_set_id(self, basis_set_id) result(valid)
        class(Core), intent(in)      :: self
        integer,     intent(in)      :: basis_set_id
        logical                      :: valid

        valid = .TRUE.

        if (basis_set_id <= 0 .or. basis_set_id > size(self%settings)) then
            print '("INPUT ERROR: Invalid basis_set_id: ", i0 ,".  Allowed [1, ", i0,"].")', &
                   basis_set_id, size(self%basis_sets)
            valid = .FALSE.
        end if
    end function

!--------------------------------------------------------------------------------------------!
!                     INITIATION FUNCTIONS                                                   !
!--------------------------------------------------------------------------------------------!
    
    !> Get the Grid3D used for the cube of the structure by initing from parameters
    !! or by reading existing grid from a file. The output grid is stored to 
    !! 'cubegrid'.
    subroutine Core_get_cube_grid(self, action_, struct, settings, cubegrid, &
                                  orbital_parallel_info, electron_density_parallel_info)
        !> The core object 
        class(Core),               intent(in)                 :: self
        !> The action object, used to get the output folder, if one is present
        type(Action),              intent(in)                 :: action_
        !> The structure object, containing the definitions needed to init
        !! new grid
        type(Structure),           intent(in)                 :: struct
        !> The settings object containing extra definitions of the initialized grid
        type(ProgramSettings),     intent(in)                 :: settings
        !> The output cube grid initialized by this subroutine
        type(Grid3D),              intent(inout)              :: cubegrid
        !> The output parallelinfo object used in orbital initialization
        class(ParallelInfo),       intent(inout), allocatable :: orbital_parallel_info
        !> The output parallelinfo object used in electron density initialization
        class(ParallelInfo),       intent(inout), allocatable :: electron_density_parallel_info
        
        ! get the grid3d of the cube from file or by initing from parameters
        if (action_%resume .and. file_exists(action_%output_folder, "cubegrid.g3d")) then
            cubegrid = Grid3D(action_%output_folder, "cubegrid.g3d")
        else
            cubegrid = struct%make_cubegrid(step      = settings%function3d_settings%cube_grid_spacing, &
                                            radius    = settings%function3d_settings%cube_cutoff_radius, &
                                            nlip      = settings%function3d_settings%cube_nlip, &
                                            grid_type = settings%function3d_settings%cube_grid_type)
            ! store the 3d grid if the orbital store mode is not 'No':0 or 'Only Bubbles: 2'
            if (      action_%store_result_functions /= DO_NOT_STORE_RESULT_FUNCTIONS &
                .and. action_%store_result_functions /= STORE_ONLY_BUBBLES) &
                call cubegrid%dump(action_%output_folder, "cubegrid")
        end if
        
        ! if we are using cuda, init stuff relevant to that
#ifdef HAVE_CUDA
        call cubegrid%cuda_init()
#endif

    
        ! create the parallelization infos
        call make_parallelization_info(orbital_parallel_info, &
                                       cubegrid = cubegrid,&
                                       gbfmm = settings%coulomb3d_settings%gbfmm, &
                                       is_potential_input = .FALSE.)
        call make_parallelization_info(electron_density_parallel_info, &
                                       cubegrid = cubegrid, &
                                       gbfmm = settings%coulomb3d_settings%gbfmm, &
                                       is_potential_input = .TRUE.)
    end subroutine
    
    
    
    

        
    
    !> Get the Grid1D objects used for the bubbles of the structure  by initing from parameters
    !! or by reading existing grids from a file. The output grids are stored to 
    !! 'bubble_grids'.
    subroutine Core_get_bubble_grids(self, action_, struct, settings, bubble_grids)
        !> The core object 
        class(Core),               intent(in)         :: self
        !> The action object, used to get the output folder, if one is present
        type(Action),              intent(in)         :: action_
        !> The structure object, containing the definitions needed to init
        !! new grids
        type(Structure),           intent(in)         :: struct
        !> The settings object containing extra definitions of the initialized grids
        type(ProgramSettings),     intent(in)         :: settings
        !> The output bubble grids initialized by this subroutine
        type(Grid1D), allocatable, intent(inout)      :: bubble_grids(:)
        ! loop variables
        integer                                       :: i
        ! get the bubble grids from file or by initing from parameters
        if (action_%resume .and. file_exists(action_%output_folder, "bubblegrids.g1d")) then
            call resume_Grid1Ds(bubble_grids, action_%output_folder, "bubblegrids.g1d")
        else
            call struct%make_bubblegrids(bubble_grids, &
                                         n0        = settings%function3d_settings%bubble_cell_count, &
                                         cutoff    = settings%function3d_settings%bubble_cutoff_radius, &
                                         nlip      = settings%function3d_settings%bubbles_nlip, &
                                         grid_type = settings%function3d_settings%bubbles_grid_type )
            ! store the 1d grids if the orbital store mode is not 'No':0 or 'Only Cube: 3'
            if (      action_%store_result_functions /= DO_NOT_STORE_RESULT_FUNCTIONS &
                .and. action_%store_result_functions /= STORE_ONLY_CUBES) &
                call store_Grid1Ds(bubble_grids, action_%output_folder, "bubblegrids.g1d")
        end if

#ifdef HAVE_CUDA
        if (allocated(bubble_grids)) then
            do i = 1, size(bubble_grids)
                call bubble_grids(i)%cuda_init()
            end do
        end if
#endif
    end subroutine
    
    

    !> Get the orbitals either using the parameters in 'structure_object' and 'settings'
    !! or by resuming previously calculated orbitals from binary files.
    subroutine Core_get_orbitals(self, action_, structure_object, settings, &
                                 orbital_parallel_info, electron_density_parallel_info, &
                                 bubble_grids, &
                                 basis_object, mos_a, mos_b, electron_density)
        class(Core), intent(in)                      :: self
        type(Action),    intent(in)                  :: action_
        type(Structure), intent(in)                  :: structure_object
        type(ProgramSettings), intent(in)            :: settings
        class(ParallelInfo), intent(in)              :: orbital_parallel_info, electron_density_parallel_info
        type(Grid1D), target,  intent(in)            :: bubble_grids(:)
        type(Basis), intent(in), optional            :: basis_object
        type(Function3D), allocatable, intent(inout) :: mos_a(:)
        type(Function3D), allocatable, intent(inout) :: mos_b(:)
        type(Function3D), allocatable, intent(inout) :: electron_density
        type(Function3D), allocatable                :: temp
        ! Molecule orbital coefficients
        real(REAL64), allocatable                    :: mocoeffs(:,:)
        ! nocc: Number of occupied orbitals, nvir: Number of virtual orbitals
        integer                                      :: nocc(2), i, nvir, number_of_resumable_orbitals(2), &
                                                        number_of_input_orbitals(2)
        type(Function3D), allocatable                :: mould
        real(REAL64)                                 :: norm, one_per_norm
        type(Bubbles)                                :: temp_bubbles


        
        ! Construct MOs
        mould = structure_object%make_f3d_mould(step=settings%function3d_settings%cube_grid_spacing, &
                                               lmax=settings%function3d_settings%bubble_lmax,  &
                                               bubble_grids = bubble_grids, &
                                               parallelization_info = orbital_parallel_info, &
                                               taylor_series_order = settings%function3d_settings%taylor_order, &
                                               bubbles_center_offset = settings%function3d_settings%bubbles_center_offset)
        electron_density = structure_object%make_f3d_mould(step=settings%function3d_settings%cube_grid_spacing, &
                                               lmax=settings%function3d_settings%bubble_lmax * 2,  &
                                               bubble_grids = bubble_grids, &
                                               parallelization_info = electron_density_parallel_info, &
                                               taylor_series_order = settings%function3d_settings%taylor_order, &
                                               bubbles_center_offset = settings%function3d_settings%bubbles_center_offset)
       
        call bubbles_multipliers%init_from_mould(mould%bubbles)
        nocc = structure_object%get_number_of_occupied_orbitals()
        nvir = structure_object%number_of_virtual_orbitals

        ! if we are using helmholtz-based update, allocate space for orbitals
        if (settings%scf_settings%type == SCF_TYPE_HELMHOLTZ) then
            allocate(mos_a(nocc(1) + nvir))
            
            ! allocate space for b-spin mos, if we do not have a restricted calculation
            if (.not. settings%scf_settings%restricted) then
                allocate(mos_b(nocc(2) + nvir))
            end if
        end if

        ! resume the orbitals that can be resumed from the output folder, otherwise set the number of resumable orbitals to 0
        if (action_%resume .and. settings%scf_settings%type == SCF_TYPE_HELMHOLTZ) then
            ! NOTE: 'get_number_of_resumable_orbitals' and 'resume_orbitals' are defined in scf_cycle.F90 
            number_of_resumable_orbitals = get_number_of_resumable_orbitals(action_%output_folder, &
                                                                            settings%scf_settings%restricted)
            call resume_orbitals(action_%output_folder, mould, settings%scf_settings%restricted, mos_a, mos_b, &
                                 [[1, min(number_of_resumable_orbitals(1), nocc(1) + nvir)], &
                                  [1, min(number_of_resumable_orbitals(2), nocc(2) + nvir)]]) 
            if (number_of_resumable_orbitals(1) > 0) then
                ! if the lmax of the settings is different than the one of the resumed orbitals,
                ! change the lmax to be the correct one
                if (settings%function3d_settings%bubble_lmax > mos_a(1)%bubbles%get_lmax()) then
                    do i = 1, size(mos_a)
                        temp_bubbles = Bubbles(mos_a(i)%bubbles, &
                                            lmax = settings%function3d_settings%bubble_lmax, &
                                            copy_content = .TRUE.)
                        call mos_a(i)%bubbles%destroy()
                        mos_a(i)%bubbles = Bubbles(temp_bubbles, copy_content = .TRUE.)
                        call temp_bubbles%destroy()
                    end do
                else if (settings%function3d_settings%bubble_lmax < mos_a(1)%bubbles%get_lmax()) then
                    do i = 1, size(mos_a)
                        call mos_a(i)%inject_extra_bubbles(settings%function3d_settings%bubble_lmax)
                    end do
                end if
            end if
            
            if (.not. settings%scf_settings%restricted .and. number_of_resumable_orbitals(2) > 0) then
                ! if the lmax of the settings is different than the one of the resumed orbitals,
                ! change the lmax to be the correct one
                if (settings%function3d_settings%bubble_lmax > mos_b(1)%bubbles%get_lmax()) then
                    do i = 1, size(mos_b)
                        temp_bubbles = Bubbles(mos_b(i)%bubbles, &
                                            lmax = settings%function3d_settings%bubble_lmax, &
                                            copy_content = .TRUE.)
                        call mos_b(i)%bubbles%destroy()
                        mos_b(i)%bubbles = Bubbles(temp_bubbles, copy_content = .TRUE.)
                        call temp_bubbles%destroy()
                    end do
                else if (settings%function3d_settings%bubble_lmax < mos_b(1)%bubbles%get_lmax()) then
                    do i = 1, size(mos_b)
                        call mos_b(i)%inject_extra_bubbles(settings%function3d_settings%bubble_lmax)
                    end do
                end if
            end if
        else
            number_of_resumable_orbitals = 0
        end if
        
        ! resume the orbitals, whic were not resumed from the output folder, from the input folder, if
        ! there is input orbitals
        if (number_of_resumable_orbitals(1) < nocc(1) + nvir .or. number_of_resumable_orbitals(2) < nocc(1) + nvir) then
            ! NOTE: 'get_number_of_resumable_orbitals' and 'resume_orbitals' are defined in scf_cycle.F90 
            number_of_input_orbitals = get_number_of_resumable_orbitals(action_%input_folder, &
                                                                        settings%scf_settings%restricted)
            call resume_orbitals(action_%input_folder, mould, settings%scf_settings%restricted, mos_a, mos_b, &
                                 [[number_of_resumable_orbitals(1) +1, min(number_of_input_orbitals(1), nocc(1) + nvir)], &
                                  [number_of_resumable_orbitals(2) +1, min(number_of_input_orbitals(2), nocc(2) + nvir)]]) 
        else
            number_of_input_orbitals = 0
        end if

        if (present(basis_object)) then
            ! create the orbitals that were not resumed from the files using the basis
            ! set, if there is one present
            
            ! if the type of the scf-loop is Helmholtz, create the linear combinations
            ! of the atomic orbitals
            if (settings%scf_settings%type == SCF_TYPE_HELMHOLTZ) then
                call basis_object%make_lcao_mos(structure_object, &
                        mould, mos_a, mos_b, [max(number_of_resumable_orbitals(1), number_of_input_orbitals(1))+1, &
                                            max(number_of_resumable_orbitals(2), number_of_input_orbitals(2))+1], &
                        settings%scf_settings%restricted)
            ! if the  read in basis functions to 'mos_a'
            else
                call basis_object%make_basis_functions(structure_object, mould, mos_a)
            end if
   
        else
            !mos = lobe_mos_from_structure(structure_object,mould, nocc, nocc, basis_object)
            !mos = SolidGaussian_mos_from_structure(structure_object, mould, nocc(1), nocc(1), basis_object)
        end if
        
        
        
        ! normalize the a MOs
        do i=1,size(mos_a)
            norm = mos_a(i) .dot. mos_a(i)
            one_per_norm = 1.0d0 / sqrt(norm)
            one_per_norm = truncate_number(one_per_norm, 2) ! used to be 4, lnw
            call mos_a(i)%product_in_place_REAL64(one_per_norm)
            call mos_a(i)%precalculate_taylor_series_bubbles()
        end do
        
        ! normalize the b MOs
        if (.not. settings%scf_settings%restricted) then
            do i=1, size(mos_b)
                norm = mos_b(i) .dot. mos_b(i)
                one_per_norm = 1.0d0 / sqrt(norm)
                one_per_norm = truncate_number(one_per_norm, 2) ! used to be 4, lnw
                
                call mos_b(i)%product_in_place_REAL64(one_per_norm)
                call mos_b(i)%precalculate_taylor_series_bubbles()
            end do
        end if
        

        write(*,*)
        write(*,'(&
            &"========================================","'//new_line(" ")//'",&
            &"===== Function3D grid information =o====","'//new_line(" ")//'",&
            &a)') mould%info()
        call mould%destroy()
        deallocate(mould)
    end subroutine
    
    !> Initializes orbitals and the quadrature from settings. The results operators
    !! and quadrature are stored to 'coulomb_operator', 'laplacian_operator',
    !! 'helmholtz_operator' and 'quadrature'. This subroutine can initialize only part of the operators,
    !! if the output variables are not present when calling the routine. However,
    !! initializing Helmholtz-operator requires the coulomb operator to be present and 
    !! gaussian quadrature is needed for coulomb and helmholtz operators.
    subroutine Core_init_operators(self, settings, cubegrid, orbital_parallel_info, &
                                   electron_density_parallel_info, &
                                   orbital, quadrature, &
                                   coulomb_operator, laplacian_operator, helmholtz_operator)
        !> The core object 
        class(Core),                               intent(in)    :: self
        !> The input settings object
        type(ProgramSettings),                     intent(in)    :: settings
        !> The grid of the input and output cubes for all operators
        type(Grid3D),                              intent(in)    :: cubegrid
        !> The parallelinfo object used in orbital initialization
        class(ParallelInfo),                       intent(in)    :: orbital_parallel_info
        !> The parallelinfo object used in electron density initialization
        class(ParallelInfo),                       intent(in)    :: electron_density_parallel_info
        !> An orbital, used as an example to initialize some things in operators
        type(Function3D),                          intent(in)    :: orbital
        !> Initialized gaussian quadrature object
        type(GaussQuad),                 optional, intent(inout) :: quadrature
        !> Initialized coulomb operator
        class(Coulomb3D),   allocatable, optional, intent(inout) :: coulomb_operator
        !> Initialized laplacian operator
        class(Laplacian3D), allocatable, optional, intent(inout) :: laplacian_operator
        !> Initialized helmholtz operator
        class(Helmholtz3D), allocatable, optional, intent(inout) :: helmholtz_operator
        
        ! init the quadrature, if slot for one is present
        if (present(quadrature)) then
            quadrature = GaussQuad(nlin = settings%quadrature_settings%nlin, &
                                nlog = settings%quadrature_settings%nlog, &
                                tstart= settings%quadrature_settings%tstart, &
                                tlin = settings%quadrature_settings%tlin, &
                                tlog = settings%quadrature_settings%tlog)
        end if

        
        
        
        ! init coulomb operator, if slot for one is present
        if (present(coulomb_operator)) then
            if (settings%coulomb3d_settings%gbfmm) then
                ! init gbfmm coulomb operator
                allocate(GBFMMCoulomb3D :: coulomb_operator)
                select type(coulomb_operator)
                    type is (GBFMMCoulomb3D)
                        coulomb_operator = GBFMMCoulomb3D( &
                                electron_density_parallel_info, &
                                orbital_parallel_info,&
                                orbital%bubbles, &
                                gaussian_quadrature = quadrature, &
                                lmax = settings%coulomb3d_settings%farfield_potential_input_lmax)
                end select
            else
                ! init serial coulomb operator
                allocate(Coulomb3D :: coulomb_operator)
                select type(coulomb_operator)
                    type is (Coulomb3D)
                        coulomb_operator = Coulomb3D(orbital_parallel_info, orbital_parallel_info, &
                                                     orbital%bubbles, gauss=quadrature)
                end select
            end if
#ifdef HAVE_CUDA
            call coulomb_operator%cuda_init()
#endif
                
        end if
        ! init laplacian operator, if slot for one is present
        if (present(laplacian_operator)) then
            if (settings%coulomb3d_settings%gbfmm) then
                ! init gbfmm laplacian operator
                allocate(GBFMMLaplacian3D :: laplacian_operator)
                select type(laplacian_operator)
                    type is (GBFMMLaplacian3D)
                        laplacian_operator = GBFMMLaplacian3D(orbital_parallel_info)
                end select
            else
                ! init serial laplacian operator
                allocate(Laplacian3D :: laplacian_operator)
                select type(laplacian_operator)
                    type is (Laplacian3D)
                        laplacian_operator = Laplacian3D(orbital_parallel_info, orbital_parallel_info) 
                end select
            end if
#ifdef HAVE_CUDA
            call laplacian_operator%cuda_init()
#endif
        end if
        
        ! init helmholtz operator, if slot for one is present
        if (present(helmholtz_operator)) then
            if (settings%helmholtz3d_settings%gbfmm) then
                ! init gbfmm helmholtz operator
                allocate(GBFMMHelmholtz3D :: helmholtz_operator)
                select type(helmholtz_operator)
                    type is (GBFMMHelmholtz3D)
                        helmholtz_operator = GBFMMHelmholtz3D( coulomb_operator, &
                            0.0d0, quadrature = quadrature, &
                            lmax = settings%function3d_settings%bubble_lmax, &
                            farfield_potential_lmax = settings%helmholtz3d_settings%farfield_potential_lmax, &
                            farfield_potential_input_lmax = &
                                settings%helmholtz3d_settings%farfield_potential_input_lmax)
                end select
            else
                ! init serial helmholtz operator
                allocate(Helmholtz3D :: helmholtz_operator)
                select type(helmholtz_operator)
                    type is (Helmholtz3D)
                        helmholtz_operator = Helmholtz3D(coulomb_operator, &
                                0.0d0, quadrature = quadrature,  &
                                lmax = settings%function3d_settings%bubble_lmax)
                end select
            end if
            
#ifdef HAVE_CUDA
            call helmholtz_operator%cuda_init_child_operator()
#endif
        end if
    end subroutine

    !> Init the correct SCFOptimizer with correct parameters using the
    !! settings, action, scf_cycle, and scf_energetics as input. The result object is 
    !! stored to 'scf_optimizer'.
    subroutine SCFOptimizer_init(settings, action_, scf_cycle, &
                                 scf_energetics, scf_optimizer)
        !> The input settings object
        type(ProgramSettings),            intent(in)    :: settings
        !> The action object, used to get the output folder, if one is present
        type(Action),                     intent(in)    :: action_
        ! the result object
        class(SCFCycle),                  intent(inout) :: scf_cycle
        ! the result object
        type(SCFEnergetics),              intent(in)    :: scf_energetics
        !> The initialized SCFOptimizer
        class(SCFOptimizer), allocatable, intent(inout) :: scf_optimizer
        
        ! check the type 1: powermethod, 2: kain
        if  (settings%scfoptimizer_settings%optimizer_type == SCF_OPTIMIZER_TYPE_POWER_METHOD) then
            allocate(scf_optimizer, source = PowerMethod(scf_cycle, & 
                settings%scfoptimizer_settings%maximum_iterations, &
                settings%scfoptimizer_settings%total_energy_convergence_threshold, &
                settings%scfoptimizer_settings%eigenvalue_convergence_threshold, &
                .FALSE., &
                settings%scf_settings%type == SCF_TYPE_HELMHOLTZ, &
                settings%scf_settings%update_weight, &
                action_, scf_energetics)) 
        else if (settings%scfoptimizer_settings%optimizer_type == SCF_OPTIMIZER_TYPE_POWER_KAIN) then
            select type(scf_cycle)
                class is (HelmholtzSCFCycle)
                    allocate(scf_optimizer, source = HelmholtzDIIS(scf_cycle, &
                        needed_guess_count = settings%scfoptimizer_settings%initialization_iteration_count, &
                        used_guess_count = settings%scfoptimizer_settings%stored_iteration_count, &
                        total_energy_convergence_threshold = settings%scfoptimizer_settings%total_energy_convergence_threshold, &
                        eigenvalue_convergence_threshold = settings%scfoptimizer_settings%eigenvalue_convergence_threshold, &
                        initialization_threshold = settings%scfoptimizer_settings%initialization_threshold, &
                        maximum_iterations = settings%scfoptimizer_settings%maximum_iterations))
                class default
                    print *, 'ERROR: Invalid SCF-type! KAIN requires the SCF-type to be a &
                           &bound-state Helmholtz-operator optimized one.'
                    stop
            end select 
        end if
    end subroutine
    
    !> Init SCF Cycle using the settings, input orbitals, and operators. 
    !! Store the result to scf_cycle.
    subroutine SCFCycle_init(settings, struct, basis_set, orbitals_a, orbitals_b, electron_density, &
                             laplacian_operator, coulomb_operator, helmholtz_operator, &
                             core_evaluator, scf_cycle)
        !> The input settings object
        type(ProgramSettings), intent(in)                 :: settings
        !> The input structure object
        type(Structure),       intent(in)                 :: struct
        !> The basis set object
        type(Basis),           intent(in)                 :: basis_set
        !> Input orbitals with spin a
        type(Function3D),      intent(in)                 :: orbitals_a(:)
        !> Input orbitals with spin b
        type(Function3D),      intent(in)                 :: orbitals_b(:)
        !> Input electron density holder
        type(Function3D),      intent(in)                 :: electron_density
        !> Used laplacian operator, could be either of the used types
        class(Laplacian3D),    intent(inout), target      :: laplacian_operator
        !> Coulomb operator, could be either of the used types
        class(Coulomb3D),      intent(inout),   target    :: coulomb_operator
        !> Helmholtz operator, could be either of the used types
        class(Helmholtz3D),    intent(inout), target      :: helmholtz_operator
        !> Evaluator to evaluate the cubes more accurately near core
        type(CoreEvaluator),   intent(inout), target      :: core_evaluator
        ! the result object
        class(SCFCycle),       intent(inout), allocatable :: scf_cycle
        
        integer                                           :: nocc(2)
        real(REAL64), allocatable                         :: mocoeffs_a(:, :), mocoeffs_b(:, :)
        
        ! get the number of occupied orbitals
        nocc = struct%get_number_of_occupied_orbitals()
        ! check the method type, 1: hartree fock, 2: dft
        if (settings%scf_settings%method == SCF_METHOD_HARTREE_FOCK) then
            ! check if we are doing a restricted calculation
            if (settings%scf_settings%restricted) then
                if(any(struct%external_electric_field /= 0.0d0)) then
                    call RHFCycle_with_EEF_init(orbitals_a, electron_density, &
                        nocc(1), nocc(2), struct%number_of_virtual_orbitals,&
                        laplacian_operator, coulomb_operator, helmholtz_operator, &
                        core_evaluator, struct%external_electric_field, scf_cycle)
                else
                    if (settings%scf_settings%type == SCF_TYPE_HELMHOLTZ) then
                        if (struct%multiplicity > 1) then
                            call ROHFCycle_init(orbitals_a, electron_density, nocc(1), nocc(2), struct%number_of_virtual_orbitals,&
                                laplacian_operator, coulomb_operator, helmholtz_operator, &
                                core_evaluator, scf_cycle, settings%hf_settings%rohf_a, &
                                settings%hf_settings%rohf_b, settings%hf_settings%rohf_f)
                        else
                            call RHFCycle_init(orbitals_a, electron_density, nocc(1), nocc(2), struct%number_of_virtual_orbitals,&
                                laplacian_operator, coulomb_operator, helmholtz_operator, &
                                core_evaluator, scf_cycle)
                        end if
                    else
                        ! get molecular orbital coefficients with spin 0: alpha
                        mocoeffs_a = struct%get_orbital_coefficients(basis_set, 0)
                        
                        call LCMORHFCycle_init(orbitals_a, electron_density, nocc(1), nocc(2), struct%number_of_virtual_orbitals, &
                            laplacian_operator, coulomb_operator, scf_cycle, mocoeffs_a)
                        deallocate(mocoeffs_a)
                    end if
                end if
            else
                call URHFCycle_init(orbitals_a, orbitals_b, electron_density, electron_density, &
                    nocc(1), nocc(2), struct%number_of_virtual_orbitals, struct%number_of_virtual_orbitals,  &
                    laplacian_operator, coulomb_operator, helmholtz_operator, scf_cycle)
                ! TODO: set the unrestricted hartree fock stuff here
            end if
        else if (settings%scf_settings%method == 2) then

#ifdef HAVE_DFT
            ! check if we are doing a restricted calculation
            if (settings%scf_settings%restricted) then
                if (settings%scf_settings%type == SCF_TYPE_HELMHOLTZ) then
                    call RDFTCycle_init(orbitals_a, electron_density, &
                        nocc(1), nocc(2), struct%number_of_virtual_orbitals,&
                        settings%dft_settings%exchange_type, settings%dft_settings%correlation_type, &
                        settings%dft_settings%xc_update_method, &
                        settings%dft_settings%xc_lmax, laplacian_operator, coulomb_operator, &
                        helmholtz_operator, core_evaluator, scf_cycle, &
                        settings%dft_settings%orbitals_density_evaluation, &
                        settings%dft_settings%fin_diff_order)
                else
                    ! get molecular orbital coefficients with spin 0: alpha
                    mocoeffs_a = struct%get_orbital_coefficients(basis_set, 0)
                    
                    ! init the Restricted DFT Cycle object which is updated with LCMO
                    call LCMORDFTCycle_init(orbitals_a, electron_density, nocc(1), nocc(2), struct%number_of_virtual_orbitals, &
                        laplacian_operator, coulomb_operator, core_evaluator, &
                        settings%dft_settings%exchange_type, &
                        settings%dft_settings%correlation_type, &
                        settings%dft_settings%xc_update_method, &
                        settings%dft_settings%xc_lmax, scf_cycle, mocoeffs_a, &
                        settings%dft_settings%orbitals_density_evaluation, &
                        settings%dft_settings%fin_diff_order)
                    deallocate(mocoeffs_a)
                end if
            end if
#else
            print *, "ERROR: DFT is not compiled in this version. Please recompile with option 'ENABLE_DFT' on to use DFT."
            stop
#endif


        end if
    end subroutine

!--------------------------------------------------------------------------------------------!
!                     DESTROY FUNCTIONS                                                      !
!--------------------------------------------------------------------------------------------!
    
    !> Destroys the operators and the quadrature. Effectively this subroutine 
    !! destroys and deallocated the objects created by 'Core_init_operators'.
    subroutine Core_destroy_operators(self, quadrature, coulomb_operator, laplacian_operator, helmholtz_operator)
        !> The core object 
        class(Core),                               intent(in)    :: self
        !> Destroyed gaussian quadrature object
        type(GaussQuad),                 optional, intent(inout) :: quadrature
        !> Destroyed Coulomb operator object
        class(Coulomb3D),   allocatable, optional, intent(inout) :: coulomb_operator
        !> Destroyed Laplacian operator object
        class(Laplacian3D), allocatable, optional, intent(inout) :: laplacian_operator
        !> Destroyed Helmholtz operator object
        class(Helmholtz3D), allocatable, optional, intent(inout) :: helmholtz_operator
        

        ! destroy coulomb operator, if one is present
        if (present(coulomb_operator)) then
            if (allocated(coulomb_operator)) then
                call coulomb_operator%destroy()
                deallocate(coulomb_operator)
            end if 
        end if
        
        ! destroy laplacian operator, if one is present
        if (present(laplacian_operator)) then
            if (allocated(laplacian_operator)) then
                call laplacian_operator%destroy()
                deallocate(laplacian_operator)
            end if 
        end if
        
        ! destroy helmholtz operator, if one is present
        if (present(helmholtz_operator)) then
            if (allocated(helmholtz_operator)) then
                call helmholtz_operator%destroy()
                deallocate(helmholtz_operator)
            end if 
        end if
        
        ! destroy quadrature, if one is present
        if (present(quadrature)) then
            call quadrature%destroy()
        end if
    end subroutine
    
    
    !> Destroys and deallocates orbitals and electron density. Effectively
    !! this subroutine undoes the memory allocations of the Core_get_orbitals - function.
    subroutine Core_destroy_orbitals_and_electron_density( &
                   self, orbitals_a, orbitals_b, electron_density)
        !> The core object 
        class(Core),                      intent(in)         :: self
        !> The orbitals with spin a, destroyed by this subroutine
        type(Function3D),  allocatable,   intent(inout)      :: orbitals_a(:)
        !> The orbitals with spin b, destroyed by this subroutine
        type(Function3D),  allocatable,   intent(inout)      :: orbitals_b(:)
        !> The orbitals with spin b, destroyed by this subroutine
        type(Function3D),  allocatable,   intent(inout)      :: electron_density
        ! loop indeces
        integer                                              :: iorbital
        
        ! let's destroy the orbitals and the electron density, as the cycles make copies of them
        if (allocated(orbitals_a)) then
            do iorbital = 1, size(orbitals_a)
                call orbitals_a(iorbital)%destroy()
            end do
            deallocate(orbitals_a)
        end if
        
        if (allocated(orbitals_b)) then
            do iorbital = 1, size(orbitals_b)
                call orbitals_b(iorbital)%destroy()
            end do
            deallocate(orbitals_b)
        end if
        
        if (allocated(electron_density)) then
            call electron_density%destroy()
            deallocate(electron_density)
        end if
    end subroutine 
    
    !> Destroys grids and the objects directly related to those, such 
    !! as parallelization infos.
    subroutine Core_destroy_grids(self, cubegrid, bubble_grids, &
                                  orbital_parallel_info, &
                                  electron_density_parallel_info)
        !> The core object 
        class(Core),                      intent(in)         :: self
        !> The cube grid destroyed by this subroutine
        type(Grid3D),                     intent(inout)      :: cubegrid
        !> The bubble grids destroyed by this subroutine
        type(Grid1D),        allocatable, intent(inout)      :: bubble_grids(:)
        !> The prallelization infos destroyed and deallocated by this subroutine
        class(ParallelInfo), allocatable, intent(inout)      :: orbital_parallel_info, &
                                                                electron_density_parallel_info
        ! loop indeces
        integer                                              :: i
        
        ! destroy the orbital parallel info, if it is allocated
        if (allocated(orbital_parallel_info)) then
#ifdef HAVE_CUDA
            call orbital_parallel_info%destroy_cuda()
#endif
            call orbital_parallel_info%destroy()
            deallocate(orbital_parallel_info)
        end if
        
        ! destroy the electron density parallel info, if it is allocated
        if (allocated(electron_density_parallel_info)) then
#ifdef HAVE_CUDA
            call electron_density_parallel_info%destroy_cuda()
#endif
            call electron_density_parallel_info%destroy()
            deallocate(electron_density_parallel_info)
        end if
        
        ! destroy the cube grid
#ifdef HAVE_CUDA
        call cubegrid%cuda_destroy()
#endif
        call cubegrid%destroy()
        
        ! go through all bubble grids and destroy the cuda and non-cuda 
        ! parts of them
        if (allocated(bubble_grids)) then
            do i=1, size(bubble_grids)
#ifdef HAVE_CUDA
                call bubble_grids(i)%cuda_destroy()
#endif
                call bubble_grids(i)%destroy()
            end do
            call bubbles_multipliers%destroy()
            deallocate(bubble_grids)
        end if
        
    end subroutine
    
end module
