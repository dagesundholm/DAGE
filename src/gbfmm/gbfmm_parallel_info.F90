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
! This module handles the communication and gridding of the gbfmm scheme.
module GBFMMParallelInfo_class
    use ISO_FORTRAN_ENV
    use Globals_m
    use ParallelInfo_class
    use Bubbles_class
    use Grid_class
    use Evaluators_class
#ifdef HAVE_CUDA
    use cuda_m
    use ISO_C_BINDING
#endif
#ifdef HAVE_MPI
    use MPI
#endif
#ifdef HAVE_OMP
    use omp_lib
#endif
    implicit none
    public      :: GBFMMParallelInfo

    private

    type                            :: GBFMMCommunication
        integer                     :: processor
        integer                     :: domain_in(2)
        integer                     :: domain_out(2)
    contains
        procedure                   :: communicate_matrix => & 
                                           GBFMMCommunication_communicate_matrix
        procedure                   :: communicate_cube   => & 
                                           GBFMMCommunication_communicate_cube
        procedure                   :: send_cube          => &
                                           GBFMMCommunication_send_cube
        procedure                   :: receive_cube       => &
                                           GBFMMCommunication_receive_cube
        procedure                   :: print              => & 
                                           GBFMMCommunication_print
    end type   

    type                            :: GBFMMProcessorCommunications
        !> The order number of current processor (the counting starts from 0)
        integer                                   :: iproc
        type(GBFMMCommunication), allocatable     :: communications(:)
    contains
        procedure                   :: communicate_matrix => & 
                                           GBFMMProcessorCommunications_communicate_matrix
        procedure                   :: communicate_cube => & 
                                           GBFMMProcessorCommunications_communicate_cube
        procedure                   :: communicate_cube_borders => & 
                                           GBFMMProcessorCommunications_communicate_cube_borders
        procedure                   :: order => &
                                           GBFMMProcessorCommunications_order
        procedure                   :: destroy => & 
                                           GBFMMProcessorCommunications_destroy
    end type

!--------------------------------------------------------------------!
!        GBFMMParallelInfo definition                                !
!--------------------------------------------------------------------!
! Generic Parallelization info for Grid Based Fast Multipole         !
! Method. Includes all operations that are common for the implementa-!
! tions of the Coulomb3D and Helmholtz3D operators and the operations!
! required by the ParallelInfo interface.                            !
!--------------------------------------------------------------------!

    type, extends(ParallelInfo) :: GBFMMParallelInfo
        integer                     :: start_level
        !> Maximum number of domain splits
        integer                     :: maxlevel
        !> Level at which integration is performed
        integer                     :: integration_level
        integer, allocatable        :: grid_shapes(:, :)
        integer, allocatable        :: grid_limits(:, :)
        integer, allocatable        :: grid_sizes (:)
        !> If the input function containing this is a potential input
        logical                     :: is_potential_input

        !> Starting and ending indices of the cell indices present for the
        !! current processor
        integer                     :: memory_cell_index_limits(2, 3)     
        !> The subgrid of the global grid containing the memory cell index limits
        type(Grid3D)                :: memory_grid
        !> Box size with cell counts in x, y and z direction on levels (shape: [3, 0:maxlevel])
        integer, allocatable        :: box_sizes(:, :)
        !> How many boxes are there stored before in the potential and multipole matrices
        !! before level 'level' boxes start (shape: [maxlevel])
        integer, allocatable        :: level_offsets(:)
        !> domain for each processor on each level indices: (nproc, level) 
        integer, allocatable        :: domain(:, :)
        !> table to store communications between processors
        type(GBFMMProcessorCommunications), allocatable  :: farfield_communications(:) 
        type(GBFMMProcessorCommunications)               :: nearfield_communications
        real(REAL64)                :: ranges(2, 3) 
        integer                     :: limits_multiplication(2, 3)

#ifdef HAVE_CUDA         
        !> The subgrid of the global grid containing the memory cell index limits
        type(Grid3D), allocatable   :: cuda_integrator_grids(:)
        type(C_PTR), allocatable    :: cuda_integrators(:)
#endif 
    contains
        procedure                   :: get_local_farfield_indices   => &
                                           GBFMMParallelInfo_get_local_farfield_indices 
        procedure, private          :: calculate_grid_shapes        => &
                                           GBFMMParallelInfo_calculate_grid_shapes
        procedure                   :: get_box_index                => &
                                           GBFMMParallelInfo_get_box_index
        procedure                   :: get_box_index_from_coordinates=> &
                                           GBFMMParallelInfo_get_box_index_from_coordinates
        procedure                   :: get_box_vector               => &
                                           GBFMMParallelInfo_get_box_vector 
        procedure                   :: get_child_indices            => &
                                           GBFMMParallelInfo_get_child_indices
        procedure                   :: get_grid_shape               => &
                                           GBFMMParallelInfo_get_grid_shape
        procedure                   :: get_grid_limit               => &
                                           GBFMMParallelInfo_get_grid_limit
        procedure                   :: get_nearest_neighbors_indices => &
                                           GBFMMParallelInfo_get_nearest_neighbors_indices
        procedure                   :: get_parent_box_vector         => &
                                           GBFMMParallelInfo_get_parent_box_vector
        procedure                   :: get_parent_index              => &
                                           GBFMMParallelInfo_get_parent_index 
        procedure                   :: get_total_box_count           => &
                                           GBFMMParallelInfo_get_total_box_count 
        procedure                   :: get_box_count                 => &
                                           GBFMMParallelInfo_get_box_count 
        procedure                   :: get_domain                     => &
                                           GBFMMParallelInfo_get_domain
        procedure                   :: split_calculation_domain      => &
                                           GBFMMParallelInfo_split_calculation_domain 
        procedure                   :: get_box_center                => &
                                           GBFMMParallelInfo_get_box_center
        procedure                   :: get_box_cell_index_limits     => &
                                           GBFMMParallelInfo_get_box_cell_index_limits
        procedure                   :: get_box_limits                => &
                                           GBFMMParallelInfo_get_box_limits
        procedure                   :: get_box_domain_cube_limits    => &
                                           GBFMMParallelInfo_get_box_domain_cube_limits
        procedure                   :: get_nearfield_cell_index_limits     => &
                                           GBFMMParallelInfo_get_nearfield_cell_index_limits
        procedure                   :: get_nearfield_limits     => &
                                           GBFMMParallelInfo_get_nearfield_limits
        procedure, private          :: get_box_processor             => &
                                           GBFMMParallelInfo_get_box_processor
        procedure, private          :: calculate_all_level_communications => &
                                           GBFMMParallelInfo_calculate_all_level_communications
        procedure, private          :: calculate_level_communications=> &
                                           GBFMMParallelInfo_calculate_level_communications
        procedure, private          :: calculate_required_farfield_domains    => &
                                           GBFMMParallelInfo_calculate_required_farfield_domains
        procedure, private          :: calculate_required_nearfield_domains    => &
                                           GBFMMParallelInfo_calculate_required_nearfield_domains
        procedure, private          :: split_domain_to_processors    => &
                                           GBFMMParallelInfo_split_domain_to_processors
        procedure, private          :: calculate_level_offsets       => &
                                           GBFMMParallelInfo_calculate_level_offsets
        procedure                   :: get_level_offset              => &
                                           GBFMMParallelInfo_get_level_offset
        procedure                   :: communicate_matrix                   => &
                                           GBFMMParallelInfo_communicate_matrix
        procedure                   :: communicate_cube                   => &
                                           GBFMMParallelInfo_communicate_cube
        procedure                   :: communicate_cube_borders           => &
                                           GBFMMParallelInfo_communicate_cube_borders
        procedure, public           :: get_grid                      => &
                                           GBFMMParallelInfo_get_grid
        procedure, public           :: get_global_grid                      => &
                                           GBFMMParallelInfo_get_global_grid
        procedure, public           :: get_cell_index_ranges               => &
                                           GBFMMParallelInfo_get_cell_index_ranges
        procedure, private          :: get_memory_cell_index_limits               => &
                                           GBFMMParallelInfo_get_memory_cell_index_limits
        procedure, public           :: get_cube_ranges               => &
                                           GBFMMParallelInfo_get_cube_ranges
        procedure, public           :: get_ranges               => &
                                           GBFMMParallelInfo_get_ranges
        procedure, public           :: sum_matrix                       => &
                                           GBFMMParallelInfo_sum_matrix
        procedure, public           :: sum_real                       => &
                                           GBFMMParallelInfo_sum_real
        procedure                   :: destroy                       => &
                                           GBFMMParallelInfo_destroy
        procedure                   :: integrate_cube               => &
                                           GBFMMParallelInfo_integrate_cube
        procedure                   :: get_domain_bubbles               => &
                                           GBFMMParallelInfo_get_domain_bubbles
        procedure                   :: get_domain_bubble_centers               => &
                                           GBFMMParallelInfo_get_domain_bubble_centers
        procedure                   :: get_domain_bubble_ibubs               => &
                                           GBFMMParallelInfo_get_domain_bubble_ibubs
        procedure, public           :: get_processor_number => &
                                           GBFMMParallelInfo_get_processor_number
        procedure, public           :: get_multiplication_cell_limits => &
                                           GBFMMParallelInfo_get_multiplication_cell_limits
        procedure, private          :: calculate_multiplication_cell_limits => &
                                           GBFMMParallelInfo_calculate_multiplication_cell_limits
        procedure, public           :: communicate_bubbles => &
                                           GBFMMParallelInfo_communicate_bubbles    
        procedure, public           :: communicate_bubble_content => &
                                           GBFMMParallelInfo_communicate_bubble_content       
#ifdef HAVE_CUDA
        procedure                   :: destroy_cuda => &
                                           GBFMMParallelInfo_destroy_cuda
        procedure                   :: init_cuda => &
                                           GBFMMParallelInfo_init_cuda
#endif
    end type 

    
    interface GBFMMParallelInfo
        module procedure :: GBFMMParallelInfo_init
    end interface  


contains

!--------------------------------------------------------------------!
!        GBFMMParallelInfo constructor                               !
!--------------------------------------------------------------------!
    function GBFMMParallelInfo_init(grid, grid_shape, maxlevel, adaptive_mode, is_potential_input) result(new)
        !> number of levels in gridding, i.e., how many times a
        !! the entire domain is split to smaller boxes
        type(Grid3D), intent(in), target     :: grid
        !> the way the box split is done in the highest level
        !! default is two boxes per axis (total 8 boxes)
        integer, intent(in), optional        :: grid_shape(3)
        !> the maximum level of recursion in domain split
        integer, intent(in), optional        :: maxlevel
        !> the starting level of energy and multipole calculation
        !! default is 2
        logical, intent(in), optional        :: adaptive_mode
        !> if the function containing this is a potential input
        logical, intent(in), optional        :: is_potential_input
        
        type(GBFMMParallelInfo)              :: new
        integer                              :: ilevel, ierr, box_count
        real(REAL64)                         :: ranges(2, 3)

        
        new%grid           =>  grid

        if (present(maxlevel)) then
            new%maxlevel = maxlevel
        else if (grid%gbfmm_maxlevel > 0) then
            new%maxlevel = grid%gbfmm_maxlevel
        else
            new%maxlevel = 2
        end if

        if (present(is_potential_input)) then
            new%is_potential_input = is_potential_input
        else
            new%is_potential_input = .FALSE.
        end if

        allocate(new%grid_shapes(3, 0 : new%maxlevel), source = 0)
        allocate(new%grid_limits(3, 0 : new%maxlevel), source = 0)
        allocate(new%box_sizes  (3, 0 : new%maxlevel), source = 0)
        allocate(new%grid_sizes (0:new%maxlevel), source = 0)
        allocate(new%level_offsets(new%maxlevel), source = 0)
        allocate(new%farfield_communications(new%maxlevel))
        new%grid_shapes(:, 0) = [0, 0, 0]
        if (present(grid_shape)) then
            new%grid_shapes(:, 1) = grid_shape
        else
            new%grid_shapes(:, 1) = [2, 2, 2]
        end if

        ! set adaptive mode on by default
        if (present(adaptive_mode)) then
            if (.not. adaptive_mode) then
                new%start_level = 2
            else
                new%start_level = 1
            end if
        else
            new%start_level = 2
        end if 
        
        new%ranges = grid%get_range()
        
        ! get values to new%grid_shapes and new%grid_sizes
        call new%calculate_grid_shapes()
  
        call new%calculate_level_offsets()
     
#ifdef HAVE_MPI
        ! get number of processors and store it to nproc
        call mpi_comm_size(MPI_COMM_WORLD, new%nproc, ierr)
        ! get rank of the processor (index in range 0-nproc) and
        ! store it to iproc
        call mpi_comm_rank(MPI_COMM_WORLD, new%iproc, ierr)
#else
        new%nproc = 1
        new%iproc = 0
#endif
        allocate(new%domain(new%maxlevel, new%nproc))
        call new%split_calculation_domain()
        call new%calculate_all_level_communications()
        new%memory_cell_index_limits = new%get_memory_cell_index_limits()
        new%limits_multiplication = new%calculate_multiplication_cell_limits(global = .FALSE.)

        new%memory_grid = new%grid%get_subgrid(new%memory_cell_index_limits)
#ifdef HAVE_CUDA
        call new%memory_grid%cuda_init()
#endif

        ! detemine the level that is used in the integration of the space
        do ilevel = 1, new%maxlevel
            box_count = product(new%get_grid_limit(ilevel))
            if (box_count >= new%nproc) then
                new%integration_level = ilevel
                exit
            else if (ilevel == new%maxlevel) then
                new%integration_level = maxlevel
            end if 
        end do 
#ifdef HAVE_CUDA
        call new%init_cuda()
#endif
       
    end function

!--------------------------------------------------------------------!
!        GBFMMParallelInfo methods                                   !
!--------------------------------------------------------------------!

    !> This subroutine splits the boxes at each level evenly for all available
    !! processors. The result is stored to self%domain - matrix, where the
    !! first index is the level and the second is the number of the processor.
    !! The result matrix contains the number of the first box that belongs to 
    !! the processors domain at the wanted level. 
    subroutine GBFMMParallelInfo_split_calculation_domain(self)
        class(GBFMMParallelInfo), intent(inout) :: self
        integer                                 :: iproc, jproc, nproc, ilevel, ibox, &
                                                   processor_modulus, processor_division, &
                                                   nbox

        nproc = self%nproc
        iproc = self%iproc
        
        ! go through all levels of division, and decide, which processors
        ! handle which tasks
        do ilevel = 1, self%maxlevel
            ! if the number of processors is larger than boxes in the 
            ! grid level, then calculate the same things on multiple
            ! processors (to avoid futile communication) and to save
            ! time 
            nbox = product(self%get_grid_limit(ilevel))
            if (nbox < nproc) then
                ! in this case, the processor_division is how many processors
                ! handle one box
                
                processor_division = nproc / nbox
                ! how many processors are 'left over'
                processor_modulus = mod(nproc, nbox)
                jproc = 1
                do ibox = 1, self%grid_sizes(ilevel)
                    if (ibox < processor_modulus) then
                        self%domain(ilevel, jproc : jproc + processor_division + 1) = ibox 
                        jproc = jproc + processor_division + 2
                    else
                        self%domain(ilevel, jproc : jproc + processor_division) = ibox 
                        jproc = jproc + processor_division + 1
                    end if
                end do   
            else 
                ! in this case, the processor_division is how many boxes
                ! one processor handles
                processor_division = nbox / nproc
                ! how many boxes are 'left over'
                processor_modulus = mod(nbox, nproc)
                ibox = 0
                do jproc = 0, nproc-1
                    if (jproc < processor_modulus) then
                        self%domain(ilevel, jproc+1) = ibox + processor_division + 1 
                        ibox = ibox + processor_division + 1
                    else
                        self%domain(ilevel, jproc+1) = ibox + processor_division 
                        ibox = ibox + processor_division 
                    end if
                end do 
            end if
        end do 
    end subroutine

    subroutine GBFMMParallelInfo_calculate_all_level_communications(self)
        class(GBFMMParallelInfo)                        :: self
        integer                                         :: ilevel
        do ilevel = self%start_level, self%maxlevel
            self%farfield_communications(ilevel) = self%calculate_level_communications(ilevel)
        end do
        self%nearfield_communications = self%calculate_level_communications(self%maxlevel, &
                                                    nearfield_communications = .TRUE.)
        call self%nearfield_communications%order()
    end subroutine

 

    function GBFMMParallelInfo_calculate_level_communications(self, level, nearfield_communications) &
                                                                  result(communications)
        class(GBFMMParallelInfo)                        :: self
        integer, intent(in)                             :: level
        logical, intent(in), optional                   :: nearfield_communications

        type(GBFMMProcessorCommunications)              :: communications
        type(GBFMMCommunication), allocatable           :: all_communications(:)
        integer, allocatable                            :: required_domains(:, :), &
                                                           receiver_domains(:, :), &
                                                           processor_domains(:, :),&
                                                           receiver_processor_domains(:, :)
        
        integer                                         :: communication_vector(3) 
        integer                                         :: idomain, jdomain, & 
                                                           kdomain, ldomain, &
                                                           icommunication,   &
                                                            jproc, kproc, ncommunications
        logical                                         :: opposite_found, domain_valid, domain_used, &
                                                           calculate_nearfield_communications

        communications%iproc = self%iproc
        if (present(nearfield_communications)) then
            calculate_nearfield_communications = nearfield_communications
        else
            calculate_nearfield_communications = .FALSE.
        end if
        if (calculate_nearfield_communications) then
            ! get the requirements needed for calculation of multipole moments and farfield potentials
            required_domains = self%calculate_required_nearfield_domains(self%iproc)
        else 
            ! get the requirements needed for calculation of multipole moments and farfield potentials
            required_domains = self%calculate_required_farfield_domains(level, self%iproc)
        end if 
        
        ! initialize dummy table of GBFMMCommunication objects
        allocate(all_communications(size(required_domains) * 10))
        ncommunications = 0        

        ! go through all requirement domains and solve the parts that have to be
        ! sent back
        do idomain = 1, size(required_domains, 1)
            processor_domains = self%split_domain_to_processors(required_domains(idomain, :), level)

            ! go through all the processor split domains
            do jdomain = 1, size(processor_domains, 1)
                ! the sending processor is in the first field of each processor_domain 
                ! row 
                jproc = processor_domains(jdomain, 1)
                opposite_found = .FALSE.             
                 
                ! solve the opposite direction data movement,
                ! only if the sender is not this processor
                if (jproc /= self%iproc) then
                    ! get the domains required by the sender of this data
                    if (calculate_nearfield_communications) then
                        receiver_domains = self%calculate_required_nearfield_domains(jproc)
                    else 
                        receiver_domains = self%calculate_required_farfield_domains(level, jproc)
                    end if 
                        
                    do kdomain = 1, size(receiver_domains, 1)
                        receiver_processor_domains = self%split_domain_to_processors & 
                                                          (receiver_domains(kdomain, :), level)
                        ! go through all the receiving requirements of the sending processor
                        ! in order to find the opposite of the original requirement 
                        do ldomain = 1, size(receiver_processor_domains, 1) 
                            ! the sending processor is in the first field
                            ! of each receiver_processor_domain row 
                            kproc = receiver_processor_domains(ldomain, 1)
                            
                            if (self%iproc == kproc) then
                                
                                ! check if domain marked with 'ldomain' is used in a previous communication
                                domain_valid = .TRUE.
                                do icommunication = 1, ncommunications
                                    domain_used =  all(                                       &
                                    [all_communications(icommunication)%processor == jproc,&
                                        all_communications(icommunication)%domain_out(1) ==   &
                                        receiver_processor_domains(ldomain, 2),               &
                                        all_communications(icommunication)%domain_out(2) ==   &
                                        receiver_processor_domains(ldomain, 3)])
                                    if (domain_used) then
                                        domain_valid = .FALSE.
                                        exit
                                    end if
                                end do
                                 
                                ! if not used in previous comunications, add the pair as new 
                                ! gbfmmcommunication
                                if (domain_valid) then
                                    ncommunications = ncommunications + 1
                                    all_communications(ncommunications) =                  &
                                        GBFMMCommunication(processor = jproc,              &
                                            domain_in = processor_domains(jdomain, 2 : 3), &
                                            domain_out= receiver_processor_domains(ldomain, 2 : 3))
                                    opposite_found = .TRUE.
                                    exit
                                end if 
                            end if
                        end do 

                        deallocate(receiver_processor_domains)

                        ! get out of the loop if the GBFMMCommunication object has already been
                        ! initialized
                        if (opposite_found) then
                            exit
                        end if 
                        
                    end do
                    deallocate(receiver_domains)
                end if 
                
            end do
            deallocate(processor_domains)
        end do
        ! add the real communications to the processorcommunications object
        communications%communications = all_communications( : ncommunications)
        deallocate(required_domains)
        deallocate(all_communications)
        
    end function    

    !> Calculates index ranges, required for calculation of current processors
    !! domain in multipole moment and potential calculations
    function GBFMMParallelInfo_calculate_required_farfield_domains(self, level, iproc) result(required_domains)
        class(GBFMMParallelInfo)       :: self
        integer, intent(in)            :: level, iproc
        integer                        :: parent_domain(2), &
                                          i_parent, idomain, jdomain, ndomain, &
                                          j_parent, i, j, domain_count, remove_index, &
                                          start_index, end_index, parent_index_count
        integer, allocatable           :: child_indices(:)
        integer, allocatable           :: all_required_domains(:, :), required_domains(:, :)
        integer, allocatable           :: parent_indices(:), parent_nearest_neighbors(:)
        logical                        :: parent_index_stored, continued
        
        ! get the domain for parent level
        parent_domain = self%get_domain(level - 1, iproc)

        allocate(all_required_domains(27 * (parent_domain(2) - parent_domain(1) + 1), 2))
        allocate(parent_indices  (27 * (parent_domain(2) - parent_domain(1) + 1)))
        ndomain = 0
        parent_index_count = 0

        ! go through all parent level boxes
        do i_parent = parent_domain(1), parent_domain(2)
            ! check if parent_index 'i_parent' has been stored to requests, if it is not, store its
            ! children range to required_domains
            parent_index_stored = any([(i_parent == parent_indices(i), i = 1, parent_index_count)])
            if (.not. parent_index_stored) then
                ndomain = ndomain + 1
                child_indices = self%get_child_indices(i_parent, level - 1)
                ! by definition, child indices are in sequental (e.g., 1, 2, 3, 4 ...), 
                ! thus we can store only the first and the last index
                all_required_domains(ndomain, 1) = child_indices(1)
                all_required_domains(ndomain, 2) = child_indices(size(child_indices))

                ! store the used parent level indices to parent_indices
                parent_index_count = parent_index_count + 1
                parent_indices(ndomain) = i_parent
                deallocate(child_indices)
            end if 
     
            ! get nearest neighbors of i_parent
            parent_nearest_neighbors = self%get_nearest_neighbors_indices(i_parent, level - 1)

            ! go through the nearest neighbors of i_parent
            do j = 1, size(parent_nearest_neighbors)
                j_parent = parent_nearest_neighbors(j)

                ! check if parent_index 'j_parent' has been stored to requests, if it is not, store its
                ! children range to required_domains
                parent_index_stored = any([(j_parent == parent_indices(i), i = 1, parent_index_count)])
                if (.not. parent_index_stored) then
                    
                    child_indices = self%get_child_indices(j_parent, level - 1)
                    continued = .FALSE.
                    remove_index = -1

                    ! start and end indeces of the newly found domain
                    start_index = child_indices(1)
                    end_index = child_indices(size(child_indices))

                    ! go through all found domains and check if the newly found domain continues
                    ! some of the existing ones
                    do idomain = 1, ndomain
                         ! check if domain  the newly found domain continues the
                         ! domain 'idomain'
                         if ( start_index == all_required_domains(idomain, 2) + 1) then
                             if (continued) then
                                 ! as this index will be attached to the domain 'jdomain'
                                 ! this domain will be removed from the results
                                 all_required_domains(jdomain, 1) = all_required_domains(idomain, 1)
                                 remove_index = idomain
                                 
                             else
                                 jdomain = idomain
                                 all_required_domains(jdomain, 2) = end_index
                                 continued = .TRUE.
                             end if 
                             
                         end if
                         ! check if domain 'idomain' belongs to the same domain as the newly
                         ! found domain from the end
                         if ( end_index + 1 == all_required_domains(idomain, 1)) then
                             if (continued) then
                                 ! as this index will be attached to the domain 'jdomain'
                                 ! this domain will be removed from the results
                                 all_required_domains(jdomain, 2) = all_required_domains(idomain, 2)
                                 remove_index = idomain
                             else
                                 jdomain = idomain
                                 all_required_domains(jdomain, 1) = start_index
                                 continued = .TRUE.
                             end if
                             
                         end if
                    end do

                    ! remove the extra removed domain, if such was found
                    if (remove_index /= -1) then
                        do idomain = remove_index + 1, ndomain
                              ! reduce the index of domains with index larger than remove_index
                             all_required_domains(idomain - 1, :) = all_required_domains(idomain, :)
                        end do
                        ndomain = ndomain - 1
                        
                    end if  
                    ! if the new domain does not continue an existing one, store it as a new 
                    ! domain to result array 'all_required_domains'
                    if (.not. continued) then
                        ndomain = ndomain + 1
                        ! by definition, child indices are in sequental (e.g., 1, 2, 3, 4 ...), 
                        ! thus we can store only the first and the last index
                        all_required_domains(ndomain, 1) = child_indices(1)
                        all_required_domains(ndomain, 2) = child_indices(size(child_indices))
                        
                    end if
                    ! store the used parent level indices to parent_indices
                    parent_index_count = parent_index_count + 1
                    parent_indices(parent_index_count) = j_parent
                    
                    deallocate(child_indices)
                end if 
            end do
            deallocate(parent_nearest_neighbors)
        end do
        required_domains = all_required_domains( : ndomain, :)
        deallocate(parent_indices)
        deallocate(all_required_domains)
        
    end function  

    !> Calculates index ranges, required for calculation of current processors
    !! domain in nearfield calculations
    function GBFMMParallelInfo_calculate_required_nearfield_domains(self, iproc) result(required_domains)
        class(GBFMMParallelInfo)       :: self
        integer, intent(in)            :: iproc
        integer                        :: domain(2), ibox, jbox, &
                                          i_parent, idomain, jdomain, ndomain, &
                                          j_parent, i, j, domain_count, remove_index 
        integer, allocatable           :: all_required_domains(:, :), required_domains(:, :)
        integer, allocatable           :: nearest_neighbors(:)
        logical                        :: index_stored, continued
        
        ! get the domain for the maxlevel for processor iproc
        domain = self%get_domain(self%maxlevel, iproc)
        allocate(all_required_domains((domain(2)- domain(1)) * 27, 2))
        ndomain = 0

        ! go through all parent level boxes
        do ibox = domain(1), domain(2)
            nearest_neighbors = self%get_nearest_neighbors_indices(ibox, self%maxlevel)

            ! go through the nearest neighbors of ibox
            do j = 0, size(nearest_neighbors)
                if (j == 0) then
                    jbox = ibox
                else
                    jbox = nearest_neighbors(j)
                end if

                ! check if index 'jbox' is already stored to a domain
                index_stored = any( &
                    [(jbox >= all_required_domains(i, 1) .and. jbox <= all_required_domains(i, 2), &
                      i = 1, ndomain)])

                if (.not. index_stored) then
                    continued = .FALSE.
                    remove_index = -1

                    ! go through all found domains and check if the newly found domain continues
                    ! some of the existing ones
                    do idomain = 1, ndomain
                        ! check if 'jbox' continues the domain 'idomain' from the end
                        if ( jbox == all_required_domains(idomain, 2) + 1) then
                            ! if 'continued' is .true. 'jbox' joins two previously found domains
                            if (continued) then
                                ! as 'idomain' will be attached to the domain 'jdomain'
                                ! it will be removed from the results
                                all_required_domains(jdomain, 1) = all_required_domains(idomain, 1)
                                remove_index = idomain
                                
                            else
                                jdomain = idomain
                                all_required_domains(jdomain, 2) = jbox
                                continued = .TRUE.
                            end if 
                             
                        end if
                        ! check if 'jbox' continues 'idomain'  from the start
                        if ( jbox == all_required_domains(idomain, 1) -1) then
                            ! if 'continued' is .true. 'jbox' joins two previously found domains
                            if (continued) then
                                ! as 'idomain' will be attached to the domain 'jdomain'
                                ! it will be removed from the results
                                all_required_domains(jdomain, 2) = all_required_domains(idomain, 2)
                                remove_index = idomain
                            else
                                jdomain = idomain
                                all_required_domains(jdomain, 1) = jbox
                                continued = .TRUE.
                            end if
                        end if
                        
                    end do

                    ! remove the extra removed domain, if such was found
                    if (remove_index /= -1) then
                        do idomain = remove_index + 1, ndomain
                              ! reduce the index of domains with index larger than remove_index
                             all_required_domains(idomain - 1, :) = all_required_domains(idomain, :)
                        end do
                        ndomain = ndomain - 1
                        
                    end if  

                    ! if the new domain does not continue an existing one, store it as a new 
                    ! domain to result array 'all_required_domains'
                    if (.not. continued) then
                        ndomain = ndomain + 1
                        all_required_domains(ndomain, 1) = jbox
                        all_required_domains(ndomain, 2) = jbox
                        
                    end if
                end if 
            end do
            deallocate(nearest_neighbors)
        end do
        required_domains = all_required_domains( : ndomain, :)
        deallocate(all_required_domains)
        
    end function 

    !> splits a box domain list of domains with corresponding processor numbers to processor specific contributions
    function GBFMMParallelInfo_split_domain_to_processors(self, domain, level) result(domains)
        class(GBFMMParallelInfo) :: self
        integer, intent(in)      :: domain(2)
        integer, intent(in)      :: level

        integer, allocatable     :: domains(:, :)
        integer                  :: processor_domain(2)
        integer                  :: first_processor, last_processor, iproc, ibox, ndomain, idomain

        ! get processor that handles the first box and the processor that handles the last
        ! box of the domain
        first_processor = self%get_box_processor(domain(1), level)
        last_processor = self%get_box_processor(domain(2), level)

        ! get the number of processor domains in the input domain
        ndomain = last_processor-first_processor + 1

        ! allocate space for the domains, the first index in each domain is 
        ! the number of the processor that contains it
        allocate(domains(ndomain, 3))
        idomain = 1
        do iproc = first_processor, last_processor
            domains(idomain, 1) = iproc
            processor_domain    = self%get_domain(level, iproc)

            ! use the domain start boundary as the first index of the processor
            ! domain, if the processor is the first one
            if (idomain == 1) then
                domains(idomain, 2) = domain(1)
            else
                domains(idomain, 2) = processor_domain(1) 
            end if 

            ! use the domain end boundary as the last index of the processor
            ! domain, if the processor is the last one
            if (idomain == ndomain) then
                domains(idomain, 3) = domain(2)
            else
                domains(idomain, 3) = processor_domain(2)
            end if
            
            idomain = idomain + 1
        end do 
        
    end function
 
    function GBFMMParallelInfo_get_local_farfield_indices(self, ibox, level, evaluation_type) result(local_farfield)
        class(GBFMMParallelInfo)             :: self
        integer, intent(in)                  :: ibox, level
        integer, intent(in), optional        :: evaluation_type
        integer, allocatable                 :: parent_nearest_neighbors(:), local_farfield(:), &
                                                nearest_neighbors(:)
        integer, allocatable                 :: child_indices(:), all_local_farfield(:)
       

        integer                              :: parent_index, i_parent_box, n_local_farfield, &
                                                i_child_box, i_nearest_neighbor, n_added_children, i
        integer                              :: grid_shape(3)
        logical                              :: is_nearest_neighbor, parent_contains_nearest_neighbor, &
                                                evaluate_whole, evaluate_whole_parents, evaluate_partial, &
                                                evaluate_all_at_child_level
                                                

        ! Determine the type how the local farfield is defined
        if (present (evaluation_type)) then
            select case(evaluation_type)
                ! default, all children of neigbors of the parent of a 'box i' which are not 
                ! neighbors of the 'box i' belong to the local farfield  
                case (0)
                    evaluate_whole_parents = .FALSE.
                    evaluate_partial = .TRUE.
                    evaluate_whole = .TRUE.
                    evaluate_all_at_child_level = .FALSE.
                ! get only the local farfield boxes having parents that contain one or more
                ! nearest neighbors children
                case (1) 
                    evaluate_whole_parents = .FALSE.
                    evaluate_partial = .TRUE.
                    evaluate_whole = .FALSE.
                    evaluate_all_at_child_level = .FALSE.
                ! get the parent indices of boxes that do not contain 
                ! nearest neighbors
                case (2) 
                    evaluate_whole_parents = .TRUE.
                    evaluate_partial = .FALSE.
                    evaluate_whole = .FALSE.
                    evaluate_all_at_child_level = .FALSE.
                    
                ! evaluate every box that does not belong to the local farfield
                ! at level 'level'
                case (3)
                    evaluate_all_at_child_level = .TRUE.
                    evaluate_partial = .FALSE.
                    evaluate_whole = .FALSE.
                    evaluate_whole_parents = .FALSE.
            end select
           
        else
            ! default, all children of neigbors of the parent of a 'box i' which are not 
            ! neighbors of the 'box i' belong to the local farfield
            evaluate_whole_parents = .FALSE.
            evaluate_partial = .TRUE.
            evaluate_whole = .TRUE.
            evaluate_all_at_child_level = .FALSE.
        end if
        
        ! Check if we are using the naive farfield evaluation (every box that does not belong
        ! to the nearest neighbors is evaluated at level 'level')
        if (evaluate_all_at_child_level) then
            nearest_neighbors = self%get_nearest_neighbors_indices(ibox, level)
            allocate(local_farfield(self%get_box_count(level) - size(nearest_neighbors) - 1))
            n_local_farfield = 0

            ! go through all boxes and do not add the nearest neighbors indices to 
            ! the local_farfield array
            do i = 1, self%get_box_count(level)
                is_nearest_neighbor = .FALSE.
                do i_nearest_neighbor = 1, size(nearest_neighbors)
                    if (i == nearest_neighbors(i_nearest_neighbor) .or. i == ibox) then
                        is_nearest_neighbor = .TRUE.
                        exit
                    end if
                end do
                
                ! if i was not a nearest neighbor of ibox, then add it to list
                if (.NOT. is_nearest_neighbor) then
                    n_local_farfield = n_local_farfield + 1
                    local_farfield(n_local_farfield) = i
                end if
            end do
        else
            ! get the shape of level 'level' grid
            grid_shape = self%get_grid_shape(level)
            ! allocate the maximum shape of the local farfield, which is the number of possible 
            ! nearest neighbors (26) times the number of children for a parent level box
            allocate(all_local_farfield(product(grid_shape)*26))

            ! get the parent's index and its nearest neighbors indices
            parent_index = self%get_parent_index(ibox, level)
            parent_nearest_neighbors = self%get_nearest_neighbors_indices(parent_index, level-1)
            nearest_neighbors = self%get_nearest_neighbors_indices(ibox, level)
    
            ! counter of local farfield indices
            n_local_farfield = 0
            
            ! go though all the nearest neighbors of the parent of box 'ibox'
            do i_parent_box = 1, size(parent_nearest_neighbors)
                ! get the children of the nearest neighbor of the parent of current box
                child_indices = self%get_child_indices(parent_nearest_neighbors(i_parent_box), level - 1)
                parent_contains_nearest_neighbor = .FALSE.
                n_added_children = 0
                do i_child_box = 1, size(child_indices)
                    is_nearest_neighbor = .FALSE.
                    ! check if child index is a nearest neighbor
                    do i_nearest_neighbor = 1, size(nearest_neighbors)
                        if (child_indices(i_child_box) == nearest_neighbors(i_nearest_neighbor)) then
                            is_nearest_neighbor = .TRUE.
                            parent_contains_nearest_neighbor = .TRUE.
                            exit
                        end if
                    end do
                    ! if it is not, add it to local farfield
                    if (.not. is_nearest_neighbor) then
                        n_local_farfield = n_local_farfield + 1
                        n_added_children = n_added_children + 1
                        all_local_farfield(n_local_farfield) = child_indices(i_child_box)
                    end if
                end do

                ! check if we only want the whole parent indeces of the local farfield 
                if (evaluate_whole_parents) then
                    ! remove the added children 
                    n_local_farfield = n_local_farfield - n_added_children
                    ! and if the parent does not contain children that are nearest neighbors of ibox
                    ! add the parent as a nearest neighbor
                    if (.not. parent_contains_nearest_neighbor) then
                        n_local_farfield = n_local_farfield + 1
                        all_local_farfield(n_local_farfield) = parent_nearest_neighbors(i_parent_box)
                    end if
                ! if parent does not contain a nearest neighbor or more, and such are not evaluated
                ! (.not. evaluate_whole) remove the previously added children from the list
                else if (.not. evaluate_whole .and. .not. parent_contains_nearest_neighbor) then
                    n_local_farfield = n_local_farfield - n_added_children

                ! if parent contains a nearest neighbor or more, and such are not evaluated
                ! (.not. evaluate_partial) remove the previously added children from the list
                else if (.not. evaluate_partial .and. parent_contains_nearest_neighbor) then
                    n_local_farfield = n_local_farfield - n_added_children
                end if
                
                
                ! deallocate_memory
                deallocate(child_indices)
                
            end do 
            ! get only the meaningful slice of result array
            local_farfield = all_local_farfield( : n_local_farfield)
            deallocate(all_local_farfield)
            deallocate(parent_nearest_neighbors)
            deallocate(nearest_neighbors)
        end if

    end function

    function GBFMMParallelInfo_get_nearest_neighbors_indices(self, ibox, level) result(neighbors)
        class(GBFMMParallelInfo)              :: self
        integer, intent(in)                  :: ibox
        integer, intent(in)                  :: level
        integer                              :: box_vector(3), neighbor_vector(3)
        integer                              :: all_neighbors(26)
        integer, allocatable                 :: neighbors(:)
        integer                              :: ix, iy, iz, nneighbor
        integer                              :: grid_limit(3)
        grid_limit = self%get_grid_limit(level)
        box_vector = self%get_box_vector(ibox, level)
        ! number of nearest neighbors
        nneighbor = 0
        do iz = -1, 1
            do iy = -1, 1
                do ix = -1, 1
                    if (ix /= 0 .or. iy /= 0 .or. iz /= 0) then
                        neighbor_vector = box_vector + [ix, iy, iz]
                        if (all(neighbor_vector >= 0)) then
                            if ( all([neighbor_vector(X_) < grid_limit(X_), &
                                      neighbor_vector(Y_) < grid_limit(Y_), &
                                      neighbor_vector(Z_) < grid_limit(Z_)])) then
                                nneighbor = nneighbor + 1
                                all_neighbors(nneighbor) = self%get_box_index(neighbor_vector, level)
                            end if 
                        end if
                    end if
                end do
            end do
        end do
        neighbors = all_neighbors(:nneighbor)
        
    end function

    function GBFMMParallelInfo_get_child_indices(self, ibox, level) result(child_indices)
        class(GBFMMParallelInfo)             :: self
        integer, intent(in)                  :: ibox
        integer, intent(in)                  :: level

        integer                              :: first_child, number_of_children, i
        integer                              :: child_indices(self%grid_sizes(level+1))
          
        ! how many children each box of level 'level' has        
        number_of_children = self%grid_sizes(level+1)
        first_child = (ibox - 1) * number_of_children
        
        child_indices = [(first_child+i, i=1, number_of_children)]
    end function
        

    function GBFMMParallelInfo_get_parent_index(self, ibox, level) result(parent)
        class(GBFMMParallelInfo)             :: self
        integer, intent(in)                  :: ibox
        integer, intent(in)                  :: level
        integer                              :: parent

        parent = (ibox -1) / self%grid_sizes(level) +1
    end function

    function GBFMMParallelInfo_get_parent_box_vector(self, box_vector, level) result(parent)
        class(GBFMMParallelInfo)             :: self
        integer, intent(in)                  :: box_vector(3)
        integer, intent(in)                  :: level
        integer                              :: parent(3)

        parent =  box_vector / self%get_grid_shape(level)
    end function


    !> Calculates shapes of the box grid on all levels and the associated
    !! 'grid_limits' (how many boxes there are to x, y, and z directions on 'level') ,
    !! 'grid_sizes' how many children are there for one parent level box at 'level', and
    !! 'box_sizes'  how many cells are included in a box at level 'level'
    subroutine GBFMMParallelInfo_calculate_grid_shapes(self)
        class(GBFMMParallelInfo), intent(inout):: self
        integer                                :: ilevel, jlevel
        !> shape of the box grid, how many boxes are there in the
        !! level of the first separation: default (octree version) is
        !! (2, 2, 2) 
        self%grid_sizes (0)    = 1
        self%grid_sizes (1)    = product(self%grid_shapes(:, 1))
        self%grid_limits(:, 0)    = self%grid_shapes(:, 0)
        self%grid_limits(:, 1)    = self%grid_shapes(:, 1)

        ! calculate box size in  number of cells on each axis
        self%box_sizes(X_, 0) = self%grid%get_ncell(X_)
        self%box_sizes(Y_, 0) = self%grid%get_ncell(Y_) 
        self%box_sizes(Z_, 0) = self%grid%get_ncell(Z_)

        self%box_sizes(X_, 1) = self%grid%get_ncell(X_) / self%grid_limits(X_, 1)
        self%box_sizes(Y_, 1) = self%grid%get_ncell(Y_) / self%grid_limits(Y_, 1)
        self%box_sizes(Z_, 1) = self%grid%get_ncell(Z_) / self%grid_limits(Z_, 1)
        do ilevel = 2, self%maxlevel
             ! if level is higher than 1, each level of the parent level is split into two
            self%grid_shapes(:, ilevel) = [2, 2, 2]
            
            self%grid_limits(:, ilevel) = self%grid_limits(:, ilevel-1) * self%grid_shapes(:, ilevel)
            self%grid_sizes (ilevel)    = product(self%grid_shapes(:, ilevel))


            
            ! calculate box size in  number of cells on each axis
            self%box_sizes(X_, ilevel) = self%grid%get_ncell(X_) / self%grid_limits(X_, ilevel)
            self%box_sizes(Y_, ilevel) = self%grid%get_ncell(Y_) / self%grid_limits(Y_, ilevel)
            self%box_sizes(Z_, ilevel) = self%grid%get_ncell(Z_) / self%grid_limits(Z_, ilevel)
        end do
        
    end subroutine

    !> Calculates shape of the box grid on level 'level'
    function GBFMMParallelInfo_get_grid_shape(self, level) result(grid_shape)
        class(GBFMMParallelInfo), target     :: self
        integer, intent(in)                  :: level
        !> shape of the box grid, how many boxes are there in the
        !! level of the first separation: default (octree version) is
        !! (2, 2, 2) 
        integer                              :: grid_shape(3)

        if (level == 0) then
            grid_shape = [1, 1, 1]
        else
            grid_shape = self%grid_shapes(:, level)
        end if
    end

    !> Calculates shape of the box grid on level 'level'
    pure function GBFMMParallelInfo_get_grid_limit(self, level) result(grid_limit)
        class(GBFMMParallelInfo), intent(in) :: self
        integer, intent(in)                  :: level
        !> shape of the box grid, how many boxes are there in the
        !! level of the first separation: default (octree version) is
        !! (2, 2, 2) 
        integer                              :: grid_limit(3)
    
        if (level == 0) then
            grid_limit = [1, 1, 1]
        else
            grid_limit = self%grid_limits(:, level)
        end if    
    end
        

    ! Returns the vector of the box 'ibox' at level 'level'
    recursive function GBFMMParallelInfo_get_box_vector(self, ibox, level) result(box_vector) 
        class(GBFMMParallelInfo)             :: self
        integer, intent(in)                  :: ibox, level
        integer                              :: box_vector(3), in_box_vector(3)
        integer                              :: grid_shape(3)
        integer                              :: ilevel, parent_index, level_remainder 

        grid_shape = self%get_grid_shape(level)
        ! if level calculated is larger than 
        if (level > 1) then
            parent_index = self%get_parent_index(ibox, level)
            
            ! get the vector of the parent and change its indexing to 'level'
            box_vector = self%get_box_vector(parent_index, level - 1) &
                         * self%get_grid_shape(level)
            ! calculate the index of the box inside the parent box and
            ! reduce 1 to get to indexing beginning from 0
            level_remainder = mod(ibox -1, self%grid_sizes(level))
            
        else
            box_vector = [0, 0, 0]
            ! just reduce 1 to get to indexing beginning from 0
            level_remainder = ibox -1
        end if

        ! calculate the vector inside the 'level' box
        in_box_vector(3) =  level_remainder / (grid_shape(1) * grid_shape(2))
        level_remainder  =  level_remainder - in_box_vector(3) * grid_shape(1) * grid_shape(2)
        in_box_vector(2) =  level_remainder / grid_shape(1)
        level_remainder  =  level_remainder - in_box_vector(2) * grid_shape(1)
        in_box_vector(1) =  level_remainder

        
        ! add in vector to box position
        box_vector = box_vector + in_box_vector
        

        
       
        
    end function

    ! Generate 1D index from integer indices at any level
    recursive function GBFMMParallelInfo_get_box_index(self, box_vector, level) result(box_index)
        class(GBFMMParallelInfo)            :: self
        integer, intent(in)                 :: box_vector(3)
        integer, intent(in)                 :: level
        integer                             :: box_index, parent_box_index
        integer                             :: level_remainder(3)
        integer                             :: parent_box_vector(3)
        integer                             :: grid_shape(3)
 
        grid_shape = self%get_grid_shape(level)
        if (level > 1) then
            parent_box_vector = self%get_parent_box_vector(box_vector, level) 
            parent_box_index = self%get_box_index(parent_box_vector, level-1)
            box_index = (parent_box_index - 1) * self%grid_sizes(level) + 1
            level_remainder = box_vector - parent_box_vector * self%get_grid_shape(level)
        else 
            level_remainder = box_vector
            box_index = 1
        end if 
        box_index = box_index                                            &
                    + level_remainder(1)                                 &
                    + level_remainder(2) * grid_shape(1)                 &
                    + level_remainder(3) * grid_shape(1) * grid_shape(2)  

    end function

    !> get number of boxes in all levels
    function GBFMMParallelInfo_get_total_box_count(self) result(box_count)
        class(GBFMMParallelInfo)            :: self
        integer                             :: box_count, ilevel, start_level
    
   
        box_count = 0
        do ilevel = self%start_level, self%maxlevel
            box_count = box_count + product(self%get_grid_limit(ilevel)) 
        end do

    end function

    !> get number of boxes in level 'level'
    function GBFMMParallelInfo_get_box_count(self, level) result(box_count)
        class(GBFMMParallelInfo)            :: self
        integer, intent(in)                 :: level
        integer                             :: box_count, ilevel, start_level
    
        box_count = product(self%get_grid_limit(level)) 
    end function

    !> get domain of the processor for level 'level'
    pure function GBFMMParallelInfo_get_domain(self, level, iproc) result(domain)
        class(GBFMMParallelInfo), intent(in) :: self
        integer, intent(in), optional        :: level
        integer, intent(in), optional        :: iproc
        integer                              :: domain(2), jproc, result_level

        if (present (level)) then
            result_level = level
        else
            result_level = self%maxlevel
        end if

        if (result_level == 0) then
            domain(1) = 1 
            domain(2) = 1
            return
        end if
 
        if (present(iproc)) then
            jproc = iproc
        else
            jproc = self%iproc
        end if

        ! starting index is the ending index of the previous processor +1
        ! Note: self%iproc starts from 0, but indices start from 1, thus
        if (jproc == 0) then
            domain(1) = 1
        else
            domain(1) = self%domain(result_level, jproc) + 1
        end if
        domain(2) = self%domain(result_level, jproc + 1) 
    end function

    !> get the number of the processor handling 'ibox' at level 'level'
    function GBFMMParallelInfo_get_box_processor(self, ibox, level) result(iproc)
        class(GBFMMParallelInfo)            :: self
        integer, intent(in)                 :: ibox, level
        integer                             :: processor_division, processor_modulus, &
                                               nbox, iproc, nproc

        nproc = self%nproc
        nbox = product(self%get_grid_limit(level))
        if (nbox < nproc) then
            ! in this case, the processor_division is how many processors
            ! handle one box
                
            processor_division = nproc / nbox
            ! how many processors are 'left over'
            processor_modulus = mod(nproc, nbox)
            
            iproc = processor_division / ibox
            
            if (mod(processor_division, ibox) == 0 .and. processor_modulus > 0) then
                iproc = iproc - 1
            end if 
        else 
            ! in this case, the processor_division is how many boxes
            ! one processor handles
            processor_division = nbox / nproc
            ! how many boxes are 'left over'
            processor_modulus = mod(nbox, nproc)
            iproc = (ibox - 1) / processor_division
            if (mod(ibox, processor_division) == 0 .and. processor_modulus > 0) then
                iproc = iproc - 1
            end if
        end if
    end function

    !> get domain of the processor for level 'level'
    function GBFMMParallelInfo_get_box_center(self, ibox, level) result(box_center)
        class(GBFMMParallelInfo), intent(in) :: self
        integer, intent(in)                  :: ibox, level
        integer                              :: box_vector(3)
        real(REAL64)                         :: box_center(3), box_size(3)
 
        box_size = (self%ranges(2, :) - self%ranges(1, :)) / (self%get_grid_limit(level) + 0.0d0)
        box_vector = self%get_box_vector(ibox, level)
        box_center = self%ranges(1, :) + (box_vector + 0.5d0) * box_size 
    end function

    !> get the index of the processor containing 'coordinates' for level 'level'
    function GBFMMParallelInfo_get_box_index_from_coordinates(self, coordinates, level) result(ibox)
        class(GBFMMParallelInfo)            :: self
        integer, intent(in)                 :: level
        integer                             :: ibox
        integer                             :: box_vector(3)
        real(REAL64), intent(in)            :: coordinates(3)
        integer                             :: cell_coordinates(3)
        ! size of a box in cell counts
        integer                             :: box_size(3)

        box_size = self%box_sizes(:, level)
 
        cell_coordinates = self%grid%cartesian_coordinates_to_cell_coordinates(coordinates)
        box_vector(X_) = floor(dble(cell_coordinates(X_) -1) / box_size(X_))
        box_vector(Y_) = floor(dble(cell_coordinates(Y_) -1) / box_size(Y_))
        box_vector(Z_) = floor(dble(cell_coordinates(Z_) -1) / box_size(Z_))
        ibox = self%get_box_index(box_vector, level) 
    end function

    !> get x,y,z starting and ending cell integer coordinates of box with number 'ibox' at maxlevel,
    !! in the local grid, i.e., coordinates in the local grid when cell size is (1, 1, 1)
    function GBFMMParallelInfo_get_box_cell_index_limits(self, ibox, level, global) result(limits)
        class(GBFMMParallelInfo)            :: self
        !> order number of box
        integer,             intent(in)     :: ibox
        !> level of the box for which the cell indices are evaluated, if not given
        !! maxlevel is used
        integer, optional,   intent(in)     :: level
        !> if the nearfield cell indices are given in the entire computational domain
        !! instead of the domain present for this processor
        logical, optional,     intent(in)   :: global
 
        ! temporary variable to contain the level of the box
        integer                             :: box_level       
 
        ! box position as integers
        integer                             :: box_vector(3)
        ! result matrix
        integer                             :: limits(2, 3)

        ! size of a box in cell counts
        integer                             :: box_size(3)

        ! the ranges of the cells that are in memory
        integer                             :: cell_ranges(2, 3)

        if (present(level)) then
            box_level = level
        else
            box_level = self%maxlevel
        end if
        ! get the coordinates of the box at level 'box_level' and the
        ! size of a box in number of cell at the same level
        box_vector = self%get_box_vector(ibox, box_level)
        box_size = self%box_sizes(:, box_level)


        limits(1, :) = (box_vector  * box_size)
        limits(2, :) = limits(1, :) + box_size
        limits(1, :) = limits(1, :) + [1, 1, 1]
        ! reduce the starting coordinates of the memoryview of the current processor 
        ! from the 
        if (.not. present(global) .or. .not. global ) then
            cell_ranges = self%get_cell_index_ranges()
            limits(2, :) = limits(2, :) - cell_ranges(1, :) + [1, 1, 1]
            limits(1, :) = limits(1, :) - cell_ranges(1, :) + [1, 1, 1]
        end if
    end function

    !> get x,y,z starting and ending cell integer coordinates of box with number 'ibox' at maxlevel,
    !! in the local grid, i.e., coordinates in the local grid when cell size is (1, 1, 1)
    function GBFMMParallelInfo_get_box_limits(self, ibox, level, global) result(limits)
        class(GBFMMParallelInfo)            :: self
        !> order number of box
        integer,             intent(in)     :: ibox
        !> level of the box for which the cell indices are evaluated, if not given
        !! maxlevel is used
        integer, optional,   intent(in)     :: level
        !> if the nearfield cell indices are given in the entire computational domain
        !! instead of the domain present for this processor
        logical, optional,     intent(in)   :: global
        ! result matrix
        real(REAL64)                        :: limits(2, 3)
        type(Grid3D), pointer               :: grid
        ! result matrix
        integer                             :: cell_limits(2, 3)
        real(REAL64), pointer               :: cell_starts(:), cell_lengths(:)
        
        cell_limits = self%get_box_cell_index_limits(ibox, level, global)
        
        if (present(global) .and. global) then
            grid => self%get_global_grid()
        else 
            grid => self%get_grid()
        end if
        cell_starts  => grid%axis(X_)%get_cell_starts()
        cell_lengths => grid%axis(X_)%get_cell_deltas()
        limits(1, X_) = cell_starts(cell_limits(1, X_))
        limits(2, X_) = cell_starts(cell_limits(2, X_)) + &
                        cell_lengths(cell_limits(2, X_))

        cell_starts  => grid%axis(Y_)%get_cell_starts()
        cell_lengths => grid%axis(Y_)%get_cell_deltas()
        limits(1, Y_) = cell_starts(cell_limits(1, Y_))
        limits(2, Y_) = cell_starts(cell_limits(2, Y_))+ &
                        cell_lengths(cell_limits(2, Y_))

        cell_starts  => grid%axis(Z_)%get_cell_starts()
        cell_lengths => grid%axis(Z_)%get_cell_deltas()
        limits(1, Z_) = cell_starts(cell_limits(1, Z_))
        limits(2, Z_) = cell_starts(cell_limits(2, Z_))+ &
                        cell_lengths(cell_limits(2, Z_))
        nullify(grid)
        nullify(cell_starts)
        nullify(cell_lengths)
        
    end function


    function GBFMMParallelInfo_get_nearfield_cell_index_limits(self, ibox, global) result(limits_nearfield)
        class(GBFMMParallelInfo)             :: self
        !> number of the box for which the nearfield cell index limits are calculated
        integer,                intent(in)   :: ibox
        !> if the nearfield cell indices are given in the entire computational domain
        !! instead of the domain present for this processor
        logical, optional,      intent(in)   :: global

        ! box limits for box ibox and jbox and nearfield as integers
        integer                              ::  limits_jbox(2, 3), &
                                                 limits_nearfield(2,3), box_number, j, &
                                                 jbox 

        ! nearest neighbor indices of an individual box
        integer, allocatable                 :: nearest_neighbors(:)

        ! get the limits of the box ibox and initialize the result array with it
        limits_nearfield = self%get_box_cell_index_limits(ibox, self%maxlevel, global)
        ! get the nearfield area for ibox, start from the area of ibox and broaden it to include
        ! all the nearest neighbors (this takes )

        ! get the nearest neighbor indices for box ibox at level self%maxlevel
        nearest_neighbors = self%get_nearest_neighbors_indices(ibox, self%maxlevel)
   
        do j = 1, size(nearest_neighbors)
            jbox = nearest_neighbors(j)
                
            ! get the limits of the nearest neighbor box
            limits_jbox = self%get_box_cell_index_limits(jbox, self%maxlevel, global)

            ! check if starting indices are smaller than in limits_nearfield and replace them
            ! if so
            if (limits_jbox(1, X_) < limits_nearfield(1, X_)) &
                limits_nearfield(1, X_) = limits_jbox(1, X_)
            if (limits_jbox(1, Y_) < limits_nearfield(1, Y_)) &
                limits_nearfield(1, Y_) = limits_jbox(1, Y_)
            if (limits_jbox(1, Z_) < limits_nearfield(1, Z_)) &
                limits_nearfield(1, Z_) = limits_jbox(1, Z_)
                   
            ! check if ending indices are larger than in limits_nearfield and replace them
            ! if so  
            if (limits_jbox(2, X_) > limits_nearfield(2, X_)) &
                limits_nearfield(2, X_) = limits_jbox(2, X_)
            if (limits_jbox(2, Y_) > limits_nearfield(2, Y_)) &
                limits_nearfield(2, Y_) = limits_jbox(2, Y_)
            if (limits_jbox(2, Z_) > limits_nearfield(2, Z_)) &
                limits_nearfield(2, Z_) = limits_jbox(2, Z_)
        end do
        deallocate(nearest_neighbors)
    end function

    function GBFMMParallelInfo_get_nearfield_limits(self, ibox, global) result(limits)
        class(GBFMMParallelInfo)            :: self
        !> order number of box
        integer,             intent(in)     :: ibox
        !> if the nearfield cell indices are given in the entire computational domain
        !! instead of the domain present for this processor
        logical, optional,     intent(in)   :: global
        ! result matrix
        real(REAL64)                         :: limits(2, 3)
        type(Grid3D), pointer                :: grid
        ! result matrix
        integer                             :: cell_limits(2, 3)
        real(REAL64), pointer               :: cell_starts(:), cell_lengths(:)
        
        cell_limits = self%get_nearfield_cell_index_limits(ibox, global)

        if (present(global) .and. global) then
            grid => self%get_global_grid()
        else 
            grid => self%get_grid()
        end if
        
        cell_starts  => grid%axis(X_)%get_cell_starts()
        cell_lengths => grid%axis(X_)%get_cell_deltas()
        limits(1, X_) = cell_starts(cell_limits(1, X_))
        limits(2, X_) = cell_starts(cell_limits(2, X_)) + &
                        cell_lengths(cell_limits(2, X_))

        cell_starts  => grid%axis(Y_)%get_cell_starts()
        cell_lengths => grid%axis(Y_)%get_cell_deltas()
        limits(1, Y_) = cell_starts(cell_limits(1, Y_))
        limits(2, Y_) = cell_starts(cell_limits(2, Y_))+ &
                        cell_lengths(cell_limits(2, Y_))

        cell_starts  => grid%axis(Z_)%get_cell_starts()
        cell_lengths => grid%axis(Z_)%get_cell_deltas()
        limits(1, Z_) = cell_starts(cell_limits(1, Z_))
        limits(2, Z_) = cell_starts(cell_limits(2, Z_))+ &
                        cell_lengths(cell_limits(2, Z_))
        
        nullify(grid)
        nullify(cell_starts)
        nullify(cell_lengths)
        
    end function

    function GBFMMParallelInfo_get_cell_index_ranges(self) result(cell_ranges)
        class(GBFMMParallelInfo), intent(in)           :: self
        integer                                        :: cell_ranges(2, 3)

        cell_ranges = self%memory_cell_index_limits
    end function

    function GBFMMParallelInfo_get_memory_cell_index_limits(self) result(limits_memory)
        class(GBFMMParallelInfo), intent(in)           :: self

        ! box limits for box ibox and jbox and nearfield as integers
        integer                      :: limits(2,3), limits_memory(2, 3), &
                                        ibox, domain(2)

        
        if (self%is_potential_input) then
            ! get the domain of this processor (1: first box belonging to this procesor, 2: last
            ! box belonging to this processor)
            domain = self%get_domain(self%maxlevel)
            ! get the nearfield area for domain(1), start from the area of ibox and broaden it to include
            ! all the nearest neighbors of the entire domain
            limits_memory = self%get_nearfield_cell_index_limits(domain(1), global = .TRUE.)

    
            do ibox = domain(1) + 1, domain(2)
                
                limits = self%get_nearfield_cell_index_limits(ibox, global = .TRUE.)
        

                ! check if starting indices are smaller than in limits and replace them
                ! if so
                if (limits(1, X_) < limits_memory(1, X_)) &
                    limits_memory(1, X_) = limits(1, X_)
                if (limits(1, Y_) < limits_memory(1, Y_)) &
                    limits_memory(1, Y_) = limits(1, Y_)
                if (limits(1, Z_) < limits_memory(1, Z_)) &
                    limits_memory(1, Z_) = limits(1, Z_)
                    
                ! check if ending indices are larger than in limits and replace them
                ! if so  
                if (limits(2, X_) > limits_memory(2, X_)) &
                    limits_memory(2, X_) = limits(2, X_)
                if (limits(2, Y_) > limits_memory(2, Y_)) &
                    limits_memory(2, Y_) = limits(2, Y_)
                if (limits(2, Z_) > limits_memory(2, Z_)) &
                    limits_memory(2, Z_) = limits(2, Z_)
            end do
        else
            limits_memory = self%calculate_multiplication_cell_limits(global = .TRUE.)
        end if
    end function

    !> Returns the offset caused by levels in the multipole_moment and 
    !! potential arrays
    subroutine GBFMMParallelInfo_calculate_level_offsets(self)
        class(GBFMMParallelInfo)             :: self
        integer                              :: ilevel  
        integer                              :: grid_limit(3)

        do ilevel = self%maxlevel-1, self%start_level, -1
            grid_limit = self%get_grid_limit(ilevel + 1)
            self%level_offsets(ilevel) = self%level_offsets(ilevel + 1) &
                             + product(self%get_grid_limit(ilevel + 1))
        end do
    end subroutine

    !> Returns the offset caused by levels in the multipole_moment and 
    !! potential arrays
    function GBFMMParallelInfo_get_level_offset(self, level) result(level_offset)
        class(GBFMMParallelInfo)             :: self
        integer, intent(in)                  :: level
        integer                              :: level_offset
        
        level_offset = self%level_offsets(level)
    end function

    !> Gets the bubbles residing within the domain of this processor
    function GBFMMParallelInfo_get_domain_bubbles(self, bubbls) result(result_bubbles)
        class(GBFMMParallelInfo)   :: self
        type(Bubbles), intent(in)  :: bubbls

        ! other parameters
        integer                    :: domain(2)
        integer                    :: ibox
        real(REAL64)               :: box_limits(2, 3)
        type(Bubbles)              :: new_bubbles, result_bubbles
        
        ! go through all boxes belonging to the domain of this processor
        ! and calculate individual contributions of each box
        domain = self%get_domain(level = self%integration_level)
        do ibox=domain(1), domain(2)
            ! get the limits of the box as cartesian coordinates
            box_limits = self%get_box_limits &
                (ibox, level = self%integration_level)

            ! get the bubbles with centers within this box and merge them with the bubbles
            ! from the earlier handled boxes
            new_bubbles = bubbls%get_sub_bubbles(box_limits, copy_content = .TRUE.)
            if (new_bubbles%get_nbub() > 0) then   
                if (result_bubbles%get_nbub() > 0) then
                    result_bubbles = result_bubbles%merge_with(new_bubbles)
                else 
                    result_bubbles  = new_bubbles
                end if
            end if   
            call new_bubbles%destroy()
        end do
        
    end function 

    !> Gets the bubbles residing within the domain of this processor
    function GBFMMParallelInfo_get_domain_bubble_ibubs(self, bubbls) result(ibubs)
        class(GBFMMParallelInfo)   :: self
        type(Bubbles), intent(in)  :: bubbls

        ! other parameters
        integer                    :: domain(2)
        integer                    :: ibox
        real(REAL64)               :: box_limits(2, 3)
        integer, allocatable       :: temp_ibubs1(:), temp_ibubs2(:), ibubs(:)
        
        allocate(ibubs(0))
        ! go through all boxes belonging to the domain of this processor
        ! and calculate individual contributions of each box
        domain = self%get_domain(level = self%integration_level)
        do ibox=domain(1), domain(2)
            ! get the limits of the box as cartesian coordinates
            box_limits = self%get_box_limits &
                (ibox, level = self%integration_level)

            ! get the bubbles indices with centers within this box and merge the indices with the 
            ! previous list
            temp_ibubs1 = bubbls%get_ibubs_within_range(box_limits)
            temp_ibubs2 = merge_index_lists(ibubs, temp_ibubs1)
            deallocate(ibubs)
            ibubs = temp_ibubs2
            deallocate(temp_ibubs1, temp_ibubs2)
        end do

    end function 

    !> Gets the bubbles residing within the domain of this processor
    function GBFMMParallelInfo_get_domain_bubble_centers(self, bubbls) result(result_bubble_centers)
        class(GBFMMParallelInfo)   :: self
        type(Bubbles), intent(in)  :: bubbls

        ! other parameters
        real(REAL64), allocatable  :: result_bubble_centers(:, :)
        integer, allocatable       :: ibubs(:)
        
        ! get the ibubs of bubbles within the domain
        ibubs = self%get_domain_bubble_ibubs(bubbls)

        ! finally get the centers
        result_bubble_centers = bubbls%get_centers(ibubs)
        deallocate(ibubs)
    end function 

#ifdef HAVE_CUDA

    subroutine GBFMMParallelInfo_init_cuda(self)
        class(GBFMMParallelInfo), intent(inout)  :: self
        integer                                  :: domain(2), ibox, &
                                                    box_cell_index_limits(2, 3), cube_ranges(2, 3)
        type(Grid3D)                             :: grid

        
        ! let's not init the cuda stuff for the 
        ! potential input, which is not used to integration nor multiplication
        if (.not. self%is_potential_input .and. .not. allocated(self%cuda_integrators)) then
            ! go through all boxes belonging to the domain of this processor
            ! and init individual integrators for each box
            domain = self%get_domain(self%integration_level)
            allocate(self%cuda_integrator_grids(domain(1):domain(2)))
            allocate(self%cuda_integrators(domain(1):domain(2)))
            !print *, "Integrating at level", self%integration_level, "domain", domain
            do ibox=domain(1), domain(2)
                ! get the limits of the box as cell indices and cartesian coordinates
                box_cell_index_limits = self%get_box_cell_index_limits &
                    (ibox, level = self%integration_level)

                ! get indices of this box in the cube
                cube_ranges = self%get_cube_ranges(box_cell_index_limits)
        
                self%cuda_integrator_grids(ibox) = self%grid%get_subgrid(box_cell_index_limits)
                self%cuda_integrators(ibox) = integrator3d_init(stream_container, &
                                                                self%cuda_integrator_grids(ibox)%get_cuda_interface())
            end do

            call self%init_Function3D_multiplier()
        end if
    end subroutine

    !> Function that destroys the cuda integrators.
    !! NOTE: this function only has to be called once at the end of execution for a parallelinfo type
    !! for each grid size
    subroutine GBFMMParallelInfo_destroy_cuda(self)
        class(GBFMMParallelInfo), intent(inout)  :: self
        integer                                  :: domain(2), ibox

        
        ! go through all boxes belonging to the domain of this processor
        ! and destroy individual integrator of each box
        domain = self%get_domain(self%integration_level)
        if (allocated(self%cuda_integrators)) then
            do ibox=domain(1), domain(2)
                call integrator3d_destroy(self%cuda_integrators(ibox))
                call self%cuda_integrator_grids(ibox)%cuda_destroy()
                call self%cuda_integrator_grids(ibox)%destroy()
            end do
            deallocate(self%cuda_integrator_grids)
            deallocate(self%cuda_integrators)
        end if
        if (allocated(self%function3d_multiplier)) then
            call Function3DMultiplier_destroy_cuda(self%function3d_multiplier)
            call self%function3d_multiplier_grid%cuda_destroy()
            call self%function3d_multiplier_grid%destroy()
            deallocate(self%function3d_multiplier)
        end if
    end subroutine
#endif

    function GBFMMParallelInfo_get_multiplication_cell_limits(self)  result(limits_multiplication)
        class(GBFMMParallelInfo), intent(in)  :: self
        ! box limits for ibox and multiplication domain as integers
        integer                      :: limits_multiplication(2,3)

        limits_multiplication = self%limits_multiplication
    end function

    function GBFMMParallelInfo_calculate_multiplication_cell_limits(self, global) result(limits_multiplication)
        class(GBFMMParallelInfo)             :: self
        logical, optional, intent(in)        :: global
        ! box limits for ibox and multiplication domain as integers
        integer                              ::  limits_ibox(2, 3), &
                                                 limits_multiplication(2,3), &
                                                 ibox, domain(2)

        logical                              :: global_indexing

        if (present(global)) then
            global_indexing = global
        else
            global_indexing = .FALSE.
        end if

        ! get the box-range that belong to the domain of this processor
        domain = self%get_domain()

        limits_multiplication = self%get_box_cell_index_limits(domain(1), self%maxlevel, global_indexing)
        do ibox = domain(1)+1, domain(2)
               
            ! get the limits of the nearest neighbor box
            limits_ibox = self%get_box_cell_index_limits(ibox, self%maxlevel, global_indexing)

            ! check if starting indices are smaller than in limits_nearfield and replace them
            ! if so
            if (limits_ibox(1, X_) < limits_multiplication(1, X_)) &
                limits_multiplication(1, X_) = limits_ibox(1, X_)
            if (limits_ibox(1, Y_) < limits_multiplication(1, Y_)) &
                limits_multiplication(1, Y_) = limits_ibox(1, Y_)
            if (limits_ibox(1, Z_) < limits_multiplication(1, Z_)) &
                limits_multiplication(1, Z_) = limits_ibox(1, Z_)
                   
            ! check if ending indices are larger than in limits_nearfield and replace them
            ! if so  
            if (limits_ibox(2, X_) > limits_multiplication(2, X_)) &
                limits_multiplication(2, X_) = limits_ibox(2, X_)
            if (limits_ibox(2, Y_) > limits_multiplication(2, Y_)) &
                limits_multiplication(2, Y_) = limits_ibox(2, Y_)
            if (limits_ibox(2, Z_) > limits_multiplication(2, Z_)) &
                limits_multiplication(2, Z_) = limits_ibox(2, Z_)
        end do
    
    end function

    function GBFMMParallelInfo_get_processor_number(self) result(reslt)
        class(GBFMMParallelInfo)              :: self
        integer                               :: reslt

        reslt = self%iproc
    end function
  
    !> Frees the memory allocated for the object 
    subroutine GBFMMParallelInfo_destroy(self)
        class(GBFMMParallelInfo), intent(inout)  :: self
        integer                                  :: i
        print *, "--------------------------------"
        print *, "DESTROY GBFMMPARALLEL INFO"
        print *, "--------------------------------"
        if (allocated (self%grid_shapes)) deallocate(self%grid_shapes)
        if (allocated (self%grid_limits)) deallocate(self%grid_limits)
        if (allocated (self%grid_sizes))  deallocate(self%grid_sizes)
        if (allocated (self%box_sizes))   deallocate(self%box_sizes)
        if (allocated (self%level_offsets)) deallocate(self%level_offsets)
        if (allocated (self%domain)) deallocate(self%domain)
        !if (allocated (self%neighbor_processors)) deallocate(self%neighbor_processors)
        if (allocated (self%farfield_communications)) then
            do i = 1, size(self%farfield_communications)
                call self%farfield_communications(i)%destroy()
            end do
            deallocate(self%farfield_communications)
        end if 
        call self%nearfield_communications%destroy()
#ifdef HAVE_CUDA
        if (allocated (self%cuda_integrators)) deallocate(self%cuda_integrators)
#endif
#ifdef HAVE_CUDA
        call self%memory_grid%cuda_destroy()
#endif
        call self%memory_grid%destroy()
        nullify(self%grid)
    end subroutine

    !> Get the grid representing the area present for current
    !! processor (boxes in the domain and their nearfields)
    function GBFMMParallelInfo_get_grid(self) result(grid)
        class(GBFMMParallelInfo), intent(in), target     :: self
        type(Grid3D), pointer                            :: grid
        grid => self%memory_grid
    end function

    function GBFMMParallelInfo_get_global_grid(self) result(grid)
        class(GBFMMParallelInfo), intent(in), target     :: self
        type(Grid3D), pointer                            :: grid

        grid => self%grid
    end function

    function GBFMMParallelInfo_get_cube_ranges(self, input_cell_limits) result(ranges)
        class(GBFMMParallelInfo), intent(in)      :: self
        ! start and end grid indices of the cell
        integer, intent(in), optional             :: input_cell_limits(2, 3)
        ! start and end grid indices of the cell and the cube
        integer                                   :: cell_limits(2, 3), ranges(2, 3)

        if (present(input_cell_limits)) then
            cell_limits = input_cell_limits
        else
            cell_limits = self%get_cell_index_ranges()
        end if
 
        ! each cell has nlip grid points and the first and two sequental cells share a grid point
        ranges(1, :) = (cell_limits(1, :) - [1, 1, 1]) * (self%grid%get_nlip() - 1) + 1
        ranges(2, :) = (cell_limits(2, :)) * (self%grid%get_nlip() - 1) + 1
            
    end function

    function GBFMMParallelInfo_get_ranges(self) result(ranges)
        class(GBFMMParallelInfo), intent(in)      :: self
        ! start and end grid indices of the cell and the cube
        real(REAL64)                              :: ranges(2, 3)
        type(Grid3D), pointer                     :: grid

        grid   => self%get_grid()
        ranges = grid%get_range()

        nullify(grid)
            
    end function

    function GBFMMParallelInfo_get_box_domain_cube_limits(self, domain) result(cube_indices)
        class(GBFMMParallelInfo),  intent(in) :: self
        integer,                   intent(in) :: domain(2)

        integer                               :: cell_indices(2, 3), box_cell_indices(2, 3), &
                                                 cube_indices(2, 3), ibox

        ! get cell indices of the sent box and the indices of the received box
        cell_indices = self%get_box_cell_index_limits(domain(1), &
                                                            level = self%maxlevel)
        do ibox = domain(1)+1, domain(2)
            ! get cell indices of the sent box and the indices of the received box
            box_cell_indices = self%get_box_cell_index_limits(ibox, &
                                                            level = self%maxlevel)
            cell_indices = merge_indices(box_cell_indices, cell_indices)
        end do
    
        cube_indices = self%get_cube_ranges(cell_indices)
    end function

    pure function merge_index_lists(list1, list2) result(result_list)
        integer, intent(in)  :: list1(:), list2(:)
        integer              :: temp_list(size(list1) + size(list2))
        integer              :: i, j, n
        integer, allocatable :: result_list(:)
        logical              :: found

        n = size(list1)
        temp_list(1:size(list1)) = list1
        do i = 1, size(list2)
            found = .FALSE.
            do j = 1, size(list1)
                if (temp_list(j) == list2(i)) then
                    found = .TRUE.
                    exit
                end if 
            end do
            
            if (.not. found) then
                n = n + 1
                temp_list(n) = list2(i)
            end if
        end do
        result_list = temp_list(1:n)
    end function 

    pure function merge_indices(indices1, indices2) result(result_indices)
        integer, intent(in)  :: indices1(2, 3), indices2(2, 3)
        integer              :: result_indices(2, 3)

        result_indices(1, X_) = min(indices1(1, X_), indices2(1, X_))
        result_indices(1, Y_) = min(indices1(1, Y_), indices2(1, Y_))
        result_indices(1, Z_) = min(indices1(1, Z_), indices2(1, Z_))

        result_indices(2, X_) = max(indices1(2, X_), indices2(2, X_))
        result_indices(2, Y_) = max(indices1(2, Y_), indices2(2, Y_))
        result_indices(2, Z_) = max(indices1(2, Z_), indices2(2, Z_))
    end function 

    pure function intersect_indices(indices1, indices2) result(result_indices)
        integer, intent(in)  :: indices1(2, 3), indices2(2, 3)
        integer              :: result_indices(2, 3)

        result_indices(1, X_) = max(indices1(1, X_), indices2(1, X_))
        result_indices(1, Y_) = max(indices1(1, Y_), indices2(1, Y_))
        result_indices(1, Z_) = max(indices1(1, Z_), indices2(1, Z_))

        result_indices(2, X_) = min(indices1(2, X_), indices2(2, X_))
        result_indices(2, Y_) = min(indices1(2, Y_), indices2(2, Y_))
        result_indices(2, Z_) = min(indices1(2, Z_), indices2(2, Z_))
    end function

    function GBFMMParallelInfo_integrate_cube(self, cube) result(reslt)
        class(GBFMMParallelInfo), intent(in)   :: self
        real(REAL64), intent(in)               :: cube(:, :, :)
        real(REAL64)                           :: reslt, temp, reslt2
        integer                                :: domain(2), cube_ranges(2, 3), &
                                                  box_cell_index_limits(2, 3), ibox, offset
        type(Integrator)                       :: integr
        type(Grid3D)                           :: grid

        

#ifdef HAVE_CUDA
        !call CUDASync_all()
        !call register_host_cube_cuda(cube, shape(cube))
        !call CUDASync_all()
#endif
        reslt = 0.0d0
        ! go through all boxes belonging to the domain of this processor
        ! and calculate individual contributions of each box
        domain = self%get_domain(self%integration_level)
        !print *, "Integrating at level", self%integration_level, "domain", domain
        do ibox=domain(1), domain(2)
            ! get the limits of the box as cell indices and cartesian coordinates
            box_cell_index_limits = self%get_box_cell_index_limits &
                (ibox, level = self%integration_level)

            ! get indices of this box in the cube
            cube_ranges = self%get_cube_ranges(box_cell_index_limits)

#ifdef HAVE_CUDA
            offset =   (cube_ranges(1, Z_)-1) * size(cube, X_) * size(cube, Y_) &
                     + (cube_ranges(1, Y_)-1) * size(cube, X_) &
                     + (cube_ranges(1, X_)-1)
            temp = 0.0d0
            call integrator3d_integrate( &
                     self%cuda_integrators(ibox), & 
                     cube, offset, &
                     shape(cube), temp )
            call CUDASync_all()
            reslt = reslt + temp
#else
            
    
            grid = self%grid%get_subgrid(box_cell_index_limits)
            integr=Integrator(grid%axis)
            temp = integr%eval(cube(cube_ranges(1, X_) : cube_ranges(2, X_), &
                                    cube_ranges(1, Y_) : cube_ranges(2, Y_), &
                                    cube_ranges(1, Z_) : cube_ranges(2, Z_) ) )
      
            reslt = reslt + temp
         
            call integr%destroy()
            call grid%destroy()
#endif
        end do
        !print *, "cube integrate", reslt 
#ifdef HAVE_CUDA
        !call unregister_host_cube_cuda(cube)
#endif
    end function


    subroutine GBFMMParallelInfo_communicate_matrix(self, matrix, level)
        class(GBFMMParallelInfo), intent(in):: self
        real(REAL64), intent(inout), target :: matrix(:, :)
        integer, intent(in)                 :: level
        real(REAL64), pointer               :: level_matrix(:, :) 
        integer                             :: ierr
        logical                             :: main
         
        level_matrix => matrix(:, 1 + self%get_level_offset(level) : )
#ifdef HAVE_OMP
        !$OMP BARRIER
        !$OMP MASTER 
        
#endif

#ifdef HAVE_MPI
        call mpi_is_thread_main(main, ierr)
        if (main) call self%farfield_communications(level)%communicate_matrix(level_matrix, self%nproc)


        call MPI_BARRIER(MPI_COMM_WORLD, ierr)
#endif
#ifdef HAVE_OMP
        !$OMP END MASTER
        !$OMP BARRIER
#endif
        nullify(level_matrix)
    end subroutine


    subroutine GBFMMParallelInfo_communicate_cube(self, cube, sum_borders, reversed_order)
        class(GBFMMParallelInfo), intent(in):: self
        real(REAL64), pointer, intent(inout):: cube(:, :, :)
        logical, intent(in), optional       :: sum_borders
        logical, intent(in), optional       :: reversed_order
        integer                             :: ierr
        logical                             :: main
         
#ifdef HAVE_OMP
        !$OMP BARRIER
        !$OMP MASTER 
#endif
#ifdef HAVE_MPI
        call mpi_is_thread_main(main, ierr)
        if (main) call self%nearfield_communications%communicate_cube(cube, self, self%nproc, &
                 sum_borders = sum_borders, reversed_order = reversed_order)

        call MPI_BARRIER(MPI_COMM_WORLD, ierr)
#endif
#ifdef HAVE_OMP
        !$OMP END MASTER
        !$OMP BARRIER
#endif
    end subroutine

    subroutine GBFMMParallelInfo_communicate_cube_borders(self, cube, reversed_order, sum_result)
        class(GBFMMParallelInfo), intent(in):: self
        real(REAL64), pointer, intent(inout):: cube(:, :, :)
        logical, intent(in), optional       :: reversed_order
        logical, intent(in), optional       :: sum_result
        integer                             :: ierr
        logical                             :: main
         
#ifdef HAVE_OMP
        !$OMP BARRIER
        !$OMP MASTER 
#endif
#ifdef HAVE_MPI
        call mpi_is_thread_main(main, ierr)
        if (main) call self%nearfield_communications%communicate_cube_borders(cube, self, self%nproc, &
                  reversed_order = reversed_order, sum_result = sum_result)

        call MPI_BARRIER(MPI_COMM_WORLD, ierr)
#endif
#ifdef HAVE_OMP
        !$OMP END MASTER
        !$OMP BARRIER
#endif
    end subroutine

    subroutine GBFMMProcessorCommunications_order(self)
         class(GBFMMProcessorCommunications)     :: self
         integer                                 :: icommunication, jcommunication, order_number
         type(GBFMMCommunication), allocatable   :: new_communications(:)

         allocate(new_communications(size(self%communications)))
         ! order the communications in manner that the smaller processors communications
         ! are in ascending order. i.e., if the current processor 'iproc' communicates to 
         ! self%processor and 'iproc' < 'self%processor' order the communications so that 
         ! the out communications (domain_out) are in ascending order, otherwise the order will be 
         ! done according to in communications (domain_in).
         do icommunication = 1, size(self%communications)
             order_number = 1
             do jcommunication = 1, size(self%communications)
                 if (icommunication /= jcommunication) then
                     if (self%communications(icommunication)%processor > &
                         self%communications(jcommunication)%processor ) then
                         order_number = order_number + 1
                     else if (self%communications(icommunication)%processor ==                 &
                              self%communications(jcommunication)%processor .AND.              &
                              self%iproc < self%communications(icommunication)%processor .AND. &
                              self%communications(icommunication)%domain_out(1) >              &
                              self%communications(jcommunication)%domain_out(1)) then
                         order_number = order_number + 1
                     else if (self%communications(icommunication)%processor ==                 &
                              self%communications(jcommunication)%processor .AND.              &
                              self%iproc > self%communications(icommunication)%processor .AND. &
                              self%communications(icommunication)%domain_in(1) >               &
                              self%communications(jcommunication)%domain_in(1)) then
                         order_number = order_number + 1
                     end if 
                 end if
             end do
             new_communications(order_number) = self%communications(icommunication)
         end do
         self%communications = new_communications
         deallocate(new_communications)
    end subroutine 

    ! this function takes care that communication happens in optimal order
    subroutine GBFMMProcessorCommunications_communicate_matrix(self, matrix, nproc)
        class(GBFMMProcessorCommunications)     :: self
        real(REAL64), intent(inout)             :: matrix(:, :)
        integer,      intent(in)                :: nproc
        integer                                 :: i, processor, counter, order_number

        order_number = 1
        processor = self%iproc
         
        ! do communication starting from the smallest
        do 
            call get_next_communication_partner(self%iproc, order_number, processor)
            if (processor >= nproc) exit
            counter = 0
            do i = 1, size(self%communications)
                if (processor == self%communications(i)%processor) then
                    call self%communications(i)%communicate_matrix(self%iproc, matrix)
                    counter = counter + 1
                end if
            end do
        end do
         
    end subroutine 

    subroutine get_next_communication_partner(iproc, order_number, processor_number) 
        integer, intent(in)    :: iproc
        integer, intent(inout) :: order_number, processor_number
        integer                :: level_shift, remainder, level, maxlevel
        logical                :: do_shift
        
        maxlevel = 0
        do 
            if (order_number < 2**(maxlevel+1)) exit

            maxlevel = maxlevel + 1
        end do

        processor_number = iproc
        remainder = order_number
        do level = maxlevel, 0, -1
            level_shift = 2**level
            do_shift = remainder >= level_shift
            if (do_shift) then
                if (mod(iproc, level_shift*2) < level_shift) then
                    processor_number = processor_number + level_shift
                else                                         
                    processor_number = processor_number - level_shift
                end if
                remainder = remainder - level_shift
            end if
        end do

        ! prepare for the next call
        order_number = order_number + 1
    end subroutine
   
    ! note: GBFMMProcessorCommunications_order has to be called before this.
    subroutine GBFMMProcessorCommunications_communicate_cube(self, cube, parallel_info, & 
                   nproc, sum_borders, reversed_order)
        class(GBFMMProcessorCommunications)     :: self
        real(REAL64), pointer,    intent(inout) :: cube(:, :, :)
        class(GBFMMParallelInfo), intent(in)    :: parallel_info
        integer,                  intent(in)    :: nproc
        logical,        optional, intent(in)    :: sum_borders
        logical,        optional, intent(in)    :: reversed_order
#ifdef HAVE_MPI        
        real(REAL64), allocatable, target       :: received_cube(:, :, :)
        real(REAL64), pointer                   :: received_cube_pointer(:, :, :)
        integer                                 :: i, processor, counter, current_processor
        integer                                 :: sent_cube_limits(2, 3), received_cube_limits(2, 3), &
                                                   indices(2, 3), order_number
        logical                                 :: do_sum, reverse_order, first
         

        current_processor = -1

        if (present(reversed_order)) then
            reverse_order = reversed_order
        else
            reverse_order = .FALSE.
        end if
         
        if (present(sum_borders)) then
            do_sum = sum_borders 
        else 
            do_sum = .FALSE.
        end if
         
        if (do_sum) then
            allocate(received_cube(size(cube, 1), size(cube, 2), size(cube, 3)), source = 0.0d0)
            received_cube_pointer => received_cube
        else
            received_cube_pointer => cube
        end if
        
        
        current_processor = iproc
        order_number = 1
        !print *, iproc, "starting communication", reverse_order
        ! do communication starting from the smallest
        if (reverse_order) then
            do 
                call get_next_communication_partner(self%iproc, order_number, current_processor)
                if (current_processor >= nproc) exit
                first = .TRUE.
                ! determine the largest cube containing all send and received communications between the two
                ! nodes (processors)
                do i = 1, size(self%communications)
                    if (self%communications(i)%processor == current_processor) then
                        if (first) then
                            sent_cube_limits = parallel_info%get_box_domain_cube_limits(self%communications(i)%domain_out)
                            received_cube_limits = parallel_info%get_box_domain_cube_limits(self%communications(i)%domain_in)
                            first = .FALSE.
                        else
                            sent_cube_limits = merge_indices(sent_cube_limits,  &
                                parallel_info%get_box_domain_cube_limits(self%communications(i)%domain_out))
                            received_cube_limits = merge_indices(received_cube_limits,  &
                                parallel_info%get_box_domain_cube_limits(self%communications(i)%domain_in))
                        end if
                    end if
                end do

                

                ! check if there is communication between these two processors
                ! if is, perform it
                if (.not. first) then
                    
                    
                    if (self%iproc < current_processor) then
                        call receive_cube(current_processor, received_cube_pointer, received_cube_limits)
                        call send_cube(current_processor, cube, sent_cube_limits)
                    else
                        call send_cube(current_processor, cube, sent_cube_limits)
                        call receive_cube(current_processor, received_cube_pointer, received_cube_limits)
                    end if

                    if (do_sum) then
                        cube = cube + received_cube_pointer
                        received_cube_pointer = 0.0d0
                    end if
                end if
            end do
        else
            do 
                call get_next_communication_partner(self%iproc, order_number, current_processor)
                if (current_processor >= nproc) exit
                first = .TRUE.
                ! determine the largest cube containing all send and received communications between the two
                ! nodes (processors)
                do i = 1, size(self%communications)
                    if (self%communications(i)%processor == current_processor) then
                        if (first) then
                            sent_cube_limits = parallel_info%get_box_domain_cube_limits(self%communications(i)%domain_out)
                            received_cube_limits = parallel_info%get_box_domain_cube_limits(self%communications(i)%domain_in)
                            first = .FALSE.
                        else
                            sent_cube_limits = merge_indices(sent_cube_limits,  &
                                parallel_info%get_box_domain_cube_limits(self%communications(i)%domain_out))
                            received_cube_limits = merge_indices(received_cube_limits,  &
                                parallel_info%get_box_domain_cube_limits(self%communications(i)%domain_in))
                        end if
                    end if
                end do
                ! check if there is communication between these two processors
                ! if is, perform it
                if (.not. first) then
                    if (self%iproc > current_processor) then
                        call receive_cube(current_processor, received_cube_pointer, received_cube_limits)
                        
                        call send_cube(current_processor, cube, sent_cube_limits)
                    else
                        call send_cube(current_processor, cube, sent_cube_limits)
                        call receive_cube(current_processor, received_cube_pointer, received_cube_limits)
                    end if

                    if (do_sum) then
                        cube = cube + received_cube_pointer
                        received_cube_pointer = 0.0d0
                    end if
                end if
             end do  
        end if
        !print *, iproc, "ending communication", reverse_order
         
        if (do_sum) then
            deallocate(received_cube)
        end if
        nullify(received_cube_pointer)
         
#endif
    end subroutine 

    ! note: GBFMMProcessorCommunications_order has to be called before this.
    subroutine GBFMMProcessorCommunications_communicate_cube_borders(self, cube, parallel_info, & 
                   nproc, reversed_order, sum_result)
        class(GBFMMProcessorCommunications)     :: self
        real(REAL64), pointer,    intent(inout) :: cube(:, :, :)
        class(GBFMMParallelInfo), intent(in)    :: parallel_info
        integer,                  intent(in)    :: nproc
        logical,        optional, intent(in)    :: reversed_order
        logical,        optional, intent(in)    :: sum_result
#ifdef HAVE_MPI
        integer                                 :: i, processor, counter, current_processor
        integer                                 :: sent_cube_limits(2, 3), received_cube_limits(2, 3), &
                                                   indices(2, 3), order_number
        logical                                 :: do_sum, reverse_order, first
        real(REAL64), allocatable, target       :: received_cube(:, :, :)
        real(REAL64), pointer                   :: received_cube_pointer(:, :, :)
         

        current_processor = -1

        if (present(reversed_order)) then
            reverse_order = reversed_order
        else
            reverse_order = .FALSE.
        end if

        if (present(sum_result)) then
            if (sum_result) then
                allocate(received_cube(size(cube, 1), size(cube, 2), size(cube, 3)), source = 0.0d0)
                received_cube_pointer => received_cube
            else
                received_cube_pointer => cube
            end if
        else
            received_cube_pointer => cube
        end if
        
        order_number = 1
        current_processor = iproc
        ! do communication starting from the smallest
        if (reverse_order) then
            do 
                call get_next_communication_partner(self%iproc, order_number, current_processor)
                if (current_processor >= nproc) exit
                first = .TRUE.
                ! determine the largest cube containing all send and received communications between the two
                ! nodes (processors)
                do i = 1, size(self%communications)
                    if (self%communications(i)%processor == current_processor) then
                        if (first) then
                            sent_cube_limits = parallel_info%get_box_domain_cube_limits(self%communications(i)%domain_out)
                            received_cube_limits = parallel_info%get_box_domain_cube_limits(self%communications(i)%domain_in)
                            first = .FALSE.
                        else
                            sent_cube_limits = merge_indices(sent_cube_limits,  &
                                parallel_info%get_box_domain_cube_limits(self%communications(i)%domain_out))
                            received_cube_limits = merge_indices(received_cube_limits,  &
                                parallel_info%get_box_domain_cube_limits(self%communications(i)%domain_in))
                        end if
                    end if
                end do

                ! check if there is communication between these two processors
                ! if is, perform it
                if (.not. first) then
                    indices = intersect_indices(sent_cube_limits, received_cube_limits)
                
                    if (self%iproc < current_processor) then
                        call receive_cube(current_processor, received_cube_pointer, indices)
                    else
                        call send_cube(current_processor, cube, indices)
                    end if
                end if
            end do
        else
            do 
                call get_next_communication_partner(self%iproc, order_number, current_processor)
                if (current_processor >= nproc) exit
                first = .TRUE.
                ! determine the largest cube containing all send and received communications between the two
                ! nodes (processors)
                do i = 1, size(self%communications)
                    if (self%communications(i)%processor == current_processor) then
                        if (first) then
                            sent_cube_limits = parallel_info%get_box_domain_cube_limits(self%communications(i)%domain_out)
                            received_cube_limits = parallel_info%get_box_domain_cube_limits(self%communications(i)%domain_in)
                            first = .FALSE.
                        else
                            sent_cube_limits = merge_indices(sent_cube_limits,  &
                                parallel_info%get_box_domain_cube_limits(self%communications(i)%domain_out))
                            received_cube_limits = merge_indices(received_cube_limits,  &
                                parallel_info%get_box_domain_cube_limits(self%communications(i)%domain_in))
                        end if
                    end if
                end do

                ! check if there is communication between these two processors
                ! if is, perform it
                if (.not. first) then
                    indices = intersect_indices(sent_cube_limits, received_cube_limits)
                    if (self%iproc > current_processor) then
                        call receive_cube(current_processor, received_cube_pointer, indices)
                    else
                        call send_cube(current_processor, cube, indices)
                    end if
                end if
             end do  
        end if

        if (present(sum_result)) then
            if (sum_result) then
                cube = cube + received_cube_pointer
                deallocate(received_cube)
            end if
        end if
        nullify(received_cube_pointer)
#endif
    end subroutine 
  
    pure subroutine GBFMMProcessorCommunications_destroy(self)
        class(GBFMMProcessorCommunications), intent(inout)  :: self
        if (allocated(self%communications)) deallocate(self%communications)
    end subroutine 

    subroutine GBFMMCommunication_print(self)
        class(GBFMMCommunication)           :: self
        print *, "to", self%processor
        print *, "Domain sent", self%domain_out
        print *, "Domain received", self%domain_in
    end subroutine

    subroutine GBFMMCommunication_communicate_matrix(self, iproc, matrix)
         class(GBFMMCommunication), intent(in) :: self
         !> number of this processor
         integer,      intent(in)              :: iproc
         real(REAL64), intent(inout), target   :: matrix(:, :)
#ifdef HAVE_MPI
         real(REAL64), pointer                 :: in_matrix(:, :), out_matrix(:, :)
         integer                               :: nsent, nreceived, ierr, status(MPI_STATUS_SIZE), &
                                                  send_type, receive_type, received_count, request
        


         status(MPI_ERROR) = MPI_SUCCESS
         in_matrix  => matrix(:, self%domain_in (1) : self%domain_in (2))
         out_matrix => matrix(:, self%domain_out(1) : self%domain_out(2))
         !in_shape = shape(in_matrix)
         ! create the type of the sent sub array
         !call MPI_Type_create_subarray(2, shape(matrix), shape(out_matrix), [0, self%domain_out(1) - 1], &
         !                         MPI_ORDER_FORTRAN, MPI_DOUBLE_PRECISION,  &
         !                         send_type, ierr)
         !call MPI_Type_commit(send_type, ierr)
         
         !print *, size(matrix), shape(out_matrix), size(out_matrix), size(in_matrix)
         if (iproc < self%processor) then
             ! send first, receive then     
             !call MPI_SEND (matrix, 1, send_type, &
             !              self%processor, tag, MPI_COMM_WORLD, ierr) 
             call MPI_SEND (matrix(1, self%domain_out(1)), size(out_matrix), MPI_DOUBLE_PRECISION, &
                               self%processor, self%processor, MPI_COMM_WORLD, ierr)
             !call handle_mpi_error(ierr)
             !call MPI_WAIT(request, status, ierr)
             !call handle_mpi_error(ierr)
             call MPI_RECV (matrix(1, self%domain_in(1)), size(in_matrix), MPI_DOUBLE_PRECISION, &
                           self%processor, MPI_ANY_TAG, MPI_COMM_WORLD, status, ierr) !request, ierr) 

             
             !call MPI_WAIT(request, status, ierr)
             !call handle_mpi_error(ierr)
             !call MPI_GET_COUNT(status, MPI_DOUBLE_PRECISION, received_count, ierr)
             !print *, "received", received_count, "reals with tag", status(MPI_TAG), "from", status(MPI_SOURCE), &
             !         "error", status(MPI_ERROR)
             
             !call handle_mpi_error(status(MPI_ERROR))
             !call handle_mpi_error(ierr)
         else
             if (iproc /= self%processor) then
                 !print *, iproc, "matrix shape", shape(matrix)
                 ! receive first, send afterwards
                 call MPI_RECV (matrix(1, self%domain_in(1)), size(in_matrix), MPI_DOUBLE_PRECISION, &
                               self%processor, MPI_ANY_TAG, MPI_COMM_WORLD, status, ierr) !request, ierr) 
    
                 !call MPI_GET_COUNT(status, MPI_DOUBLE_PRECISION, received_count, ierr)
                 !print *, iproc, "received", received_count, "reals with tag", status(MPI_TAG), "from", status(MPI_SOURCE), &
                 !     "error", status(MPI_ERROR)
                 !call handle_mpi_error(status(MPI_ERROR))
                 
                 !call MPI_WAIT(request, status, ierr)
                 !call handle_mpi_error(ierr)
                 !call MPI_SEND (matrix, 1, send_type, &
                 !          self%processor, tag, MPI_COMM_WORLD, ierr) 
                 call MPI_SEND (matrix(1, self%domain_out(1)), size(out_matrix), MPI_DOUBLE_PRECISION, &
                               self%processor, self%processor, MPI_COMM_WORLD, ierr) !request, ierr) 
                 !call handle_mpi_error(ierr)
                 
                 !call MPI_WAIT(request, status, ierr)
                 !call handle_mpi_error(ierr)
                
             end if
             ! otherwise do nothing, as the data should be there already
         end if
         nullify(in_matrix)
         nullify(out_matrix)
#endif
         
    end subroutine 

    subroutine GBFMMParallelInfo_communicate_bubbles(self, bubbls)
        class(GBFMMParallelInfo), intent(in) :: self
        type(Bubbles),         intent(inout) :: bubbls
        integer                              :: ibub
        logical                              :: main
        real(REAL64), allocatable            :: bubble_content_copy(:, :)

        do ibub = 1, bubbls%get_nbub()
            bubble_content_copy = bubbls%bf(ibub)%p
            call self%communicate_bubble_content(ibub, bubbls%get_nbub(), bubbls%get_lmax(), bubble_content_copy)
            bubbls%bf(ibub)%p(:, :) = bubble_content_copy
            deallocate(bubble_content_copy)
        end do

    end subroutine

    subroutine GBFMMParallelInfo_communicate_bubble_content(self, ibub, nbub, lmax, bubble_content)
        class(GBFMMParallelInfo), intent(in)   :: self
        integer,                  intent(in)   :: ibub, nbub, lmax
        real(REAL64),             intent(inout):: bubble_content(:, :)
        integer                                :: ierr, first_bubble, last_bubble, domain(2), &
                                                  first_idx,  last_idx, processor
        logical                                :: main


#ifdef HAVE_MPI
        !print *, "---------------ibub", ibub, self%iproc
        call mpi_is_thread_main(main, ierr)
        if (main) then
            ! Get the contributions from all processors, every other processor
            ! except the one that calculates the potential, adds zero
            do processor = 0, nproc-1
                domain = self%get_bubble_domain_indices(nbub, lmax, processor)
                
                !print *, self%iproc, ", proc ", processor, "domain: ", domain
                ! get the first and last bubble handled by this processor
                first_bubble = (domain(1)-1) / (lmax + 1)**2 + 1
                last_bubble = (domain(2)-1) / (lmax + 1)**2 + 1  
                if (ibub >= first_bubble .and. ibub <= last_bubble) then
                    ! get the first idx, i.e., the last l, m pair calculated by this processor
                    first_idx = 1
                    if (ibub == first_bubble) then
                        first_idx = domain(1) - (ibub-1) * (lmax + 1)**2
                    end if
                    
                    ! get the last idx, i.e., the last l, m pair calculated by this processor
                    last_idx = (lmax + 1)**2
                    if (ibub == last_bubble) then
                        last_idx = domain(2) - (ibub-1) * (lmax + 1)**2
                    end if
                    ! broadcast the results to all other processors
                    !print *, "here", self%iproc
                    call mpi_bcast(bubble_content(1, first_idx), &
                        size(bubble_content, 1) * (last_idx - first_idx +1), &
                        MPI_DOUBLE_PRECISION, processor, MPI_COMM_WORLD, ierr)  
                    !print *, self%iproc, "ibub", ibub, "sum", sum(bubble_content(:, first_idx:last_idx)), &
                    !        "ids", first_idx, last_idx 
                end if
                !call MPI_BARRIER(MPI_COMM_WORLD, ierr)  
            end do
        end if
        !print *, "done communication"
#endif
    end subroutine

    subroutine GBFMMParallelInfo_sum_matrix(self, matrix)
        class(GBFMMParallelInfo), intent(in) :: self
        real(REAL64), intent(inout)           :: matrix(:, :)
        integer                               :: ierr
        logical                               :: main 

#ifdef HAVE_OMP
        !$OMP MASTER 
#endif
#ifdef HAVE_MPI
        ! Get the contributions from all processors, every other processor
        ! except the one that calculates the potential, adds zero
        call mpi_is_thread_main(main, ierr)
        call handle_mpi_error(ierr)
        if (main) then
            call mpi_allreduce(MPI_IN_PLACE, matrix,&
                        size(matrix), MPI_DOUBLE_PRECISION, MPI_SUM, &
                        MPI_COMM_WORLD, ierr)
        end if
        
#endif
#ifdef HAVE_OMP
        !$OMP END MASTER
        !$OMP BARRIER
#endif
    end subroutine

    subroutine GBFMMParallelInfo_sum_real(self, number)
        class(GBFMMParallelInfo), intent(in)   :: self
        real(REAL64), intent(inout)            :: number
        integer                                :: ierr
        logical                             :: main

#ifdef HAVE_OMP
        !$OMP MASTER 
#endif
#ifdef HAVE_MPI
        call mpi_is_thread_main(main, ierr)
        call handle_mpi_error(ierr)
        if (main) then
            call mpi_allreduce(MPI_IN_PLACE, number, &
                        1, MPI_DOUBLE_PRECISION, MPI_SUM, &
                        MPI_COMM_WORLD, ierr)
        end if
        call handle_mpi_error(ierr)
#endif
#ifdef HAVE_OMP
        !$OMP END MASTER
        !$OMP BARRIER
#endif
    end subroutine



    subroutine GBFMMCommunication_communicate_cube(self, iproc, sent_cube, received_cube, parallel_info, &
                                                 reversed_order) 
        class(GBFMMCommunication), intent(in) :: self
        !> number of this processor
        integer,      intent(in)              :: iproc
        real(REAL64), intent(in),    pointer  :: sent_cube(:, :, :)
        real(REAL64), intent(inout), pointer  :: received_cube(:, :, :)
        class(GBFMMParallelInfo), intent(in)  :: parallel_info
        logical, intent(in)                   :: reversed_order
        integer                               :: i, ibox

        


#ifdef HAVE_MPI
        


        if ((reversed_order .and. iproc < self%processor) .or. &
                (.not. reversed_order .and. iproc > self%processor)) then
            ! receive first, send afterwards
            do i = 1, self%domain_out(2) - self%domain_out(1) + 1
                ibox = self%domain_in(1) + i - 1
                call self%receive_cube(iproc, received_cube, parallel_info, ibox)

                ibox = self%domain_out(1) + i - 1
                call self%send_cube(iproc, sent_cube, parallel_info, ibox)
            end do 
            
            
        else
            ! send first, receive then 
            do i = 1, self%domain_out(2) - self%domain_out(1) + 1
                ibox = self%domain_out(1) + i - 1
                call self%send_cube(iproc, sent_cube, parallel_info, ibox)

                ibox = self%domain_in(1) + i - 1
                call self%receive_cube(iproc, received_cube, parallel_info, ibox)
            end do      
        end if
#endif
         
    end subroutine 


    subroutine GBFMMCommunication_send_cube(self, iproc, cube, parallel_info, ibox)
        class(GBFMMCommunication), intent(in) :: self
        !> number of this processor
        integer,      intent(in)              :: iproc, ibox
        real(REAL64), intent(in),  pointer    :: cube(:, :, :)
        class(GBFMMParallelInfo), intent(in)  :: parallel_info
#ifdef HAVE_MPI
        integer                               :: nreceived, ierr, status(MPI_STATUS_SIZE), &
                                                  send_type, send_cell_indices(2, 3), &
                                                  send_cube_indices(2, 3), send_start_indices(3), &
                                                  send_cube_shape(3)

        ! get cell indices of the sent box
        send_cell_indices = parallel_info%get_box_cell_index_limits(ibox, &
                                                        level = parallel_info%maxlevel)

        ! get the indices of this box in the cube
        send_cube_indices = parallel_info%get_cube_ranges(send_cell_indices)

            
        call send_cube(self%processor, cube, send_cube_indices)
            
        !call MPI_TYPE_FREE(send_type, ierr)
        
#endif
    end subroutine 

    subroutine GBFMMCommunication_receive_cube(self, iproc, cube, parallel_info, ibox)
        class(GBFMMCommunication), intent(in) :: self
        !> number of this processor
        integer,      intent(in)              :: iproc, ibox
        real(REAL64), intent(inout), pointer  :: cube(:, :, :)
        class(GBFMMParallelInfo), intent(in)  :: parallel_info
#ifdef HAVE_MPI
        integer                               :: nreceived, ierr, status(MPI_STATUS_SIZE), &
                                                  receive_type, receive_cell_indices(2, 3), &
                                                  receive_cube_indices(2, 3), receive_start_indices(3), &
                                                  receive_cube_shape(3)



        receive_cell_indices = parallel_info%get_box_cell_index_limits(ibox, &
                                                       level = parallel_info%maxlevel)

        ! get indices of this box in the cube
        receive_cube_indices = parallel_info%get_cube_ranges(receive_cell_indices)

        ! call the receive
        call receive_cube(self%processor, cube, receive_cube_indices)
#endif
    end subroutine 

    
#ifdef HAVE_MPI
    subroutine send_cube(processor, cube, send_cube_limits)
        !> number of this processor
        integer,      intent(in)              :: processor, send_cube_limits(2, 3)
        real(REAL64), intent(in),  pointer    :: cube(:, :, :)

        integer                               :: ierr, status(MPI_STATUS_SIZE), &
                                                  send_type, &
                                                  send_start_indices(3), &
                                                  send_cube_shape(3), request
        send_type = 0
            
        ! according to mpi manual, the send indices must start from 0, thus we are
        ! reducing 1 from each axis
        send_start_indices = send_cube_limits(1, :) - [1, 1, 1]

        ! solve the shape of the sent cube, NOTE: using send_start_indices, because it has
        ! the [1, 1, 1] conviniently removed
        send_cube_shape = send_cube_limits(2, :) - send_start_indices

        ! create the type of the sent sub array
        call MPI_TYPE_CREATE_SUBARRAY(3, shape(cube), send_cube_shape, send_start_indices, &
                             MPI_ORDER_FORTRAN, MPI_DOUBLE_PRECISION, send_type, &
                             ierr)
        call handle_mpi_error(ierr)
        call MPI_TYPE_COMMIT(send_type, ierr)
        call handle_mpi_error(ierr)
        
        call MPI_SEND(cube, 1, send_type, processor, processor, MPI_COMM_WORLD, ierr) !request, ierr)
        call handle_mpi_error(ierr)
        !call MPI_WAIT(request, status, ierr)
        call handle_mpi_error(ierr)
        call MPI_TYPE_FREE(send_type, ierr)
        call handle_mpi_error(ierr)

    end subroutine 

    subroutine receive_cube(processor, cube, receive_cube_limits)
        !> number of this processor
        integer,      intent(in)              :: processor, receive_cube_limits(2, 3)
        real(REAL64), intent(inout), pointer  :: cube(:, :, :)
        integer, volatile                     :: ierr, status(MPI_STATUS_SIZE), receive_type
        integer                               ::  receive_start_indices(3), &
                                                  receive_cube_shape(3), received_count, request
        
        status(MPI_ERROR) = MPI_SUCCESS
        ! according to mpi manual, the send indices must start from 0, thus we are
        ! reducing 1 from each axis
        receive_start_indices = receive_cube_limits(1, :) - [1, 1, 1]

        ! solve the shape of the sent cube, NOTE: using send_start_indices, because it has
        ! the [1, 1, 1] conviniently removed
        receive_cube_shape = receive_cube_limits(2, :) - receive_start_indices
        
        ! create the type of the received sub cube
        ! MPI_TYPE_CREATE_SUBARRAY(NDIMS, ARRAY_OF_SIZES, ARRAY_OF_SUBSIZES,
        !     ARRAY_OF_STARTS, ORDER, OLDTYPE, NEWTYPE, IERROR)
        call MPI_TYPE_CREATE_SUBARRAY(3, shape(cube), receive_cube_shape, receive_start_indices, &
                            MPI_ORDER_FORTRAN, MPI_DOUBLE_PRECISION, receive_type, &
                            ierr)
        call handle_mpi_error(ierr)
        call MPI_TYPE_COMMIT(receive_type, ierr)
        
        call handle_mpi_error(ierr)
        call MPI_RECV(cube, 1, receive_type, processor, iproc, MPI_COMM_WORLD, status, ierr) !request, ierr)
        !call MPI_WAIT(request, status, ierr)
        call handle_mpi_error(ierr)
        call MPI_GET_COUNT(status, receive_type, received_count, ierr)
        !print *, iproc, "received", received_count, "subarrays with tag", status(MPI_TAG), "from", status(MPI_SOURCE), &
        !              "error", status(MPI_ERROR), "request", request
        call handle_mpi_error(status(MPI_ERROR))
        call handle_mpi_error(ierr)
        call MPI_TYPE_FREE(receive_type, ierr)
        call handle_mpi_error(ierr)
    end subroutine 

    subroutine handle_mpi_error(ierr)
        integer, intent(in)   :: ierr

        
        if (ierr == MPI_ERR_BUFFER) then
            print *, "buffer error"
        else if (ierr == MPI_ERR_COUNT) then
            print *, "mpi count error"
        else if (ierr == MPI_ERR_TYPE) then
            print *, "mpi type error"
        else if (ierr == MPI_ERR_COMM) then
            print *, "mpi communicator error"
        else if (ierr == MPI_ERR_OP) then
            print *, "mpi operator error"
        else if (ierr == MPI_ERR_RANK) then   
            print *, "invalid rank error"
        else if (ierr == MPI_ERR_ROOT) then   
            print *, "invalid root error"
        else if (ierr == MPI_ERR_GROUP) then   
            print *, "Null group passed to function"
        else if (ierr == MPI_ERR_TOPOLOGY) then   
            print *, "Invalid topology"
        else if (ierr == MPI_ERR_DIMS) then   
            print *, "Illegal dimension argument"
        else if (ierr == MPI_ERR_ARG) then   
            print *, "Invalid argument"
        else if (ierr == MPI_ERR_TRUNCATE) then   
            print *, "message truncated on receive "
        else if (ierr == MPI_ERR_UNKNOWN) then   
            print *, "Unknown error"
        else if (ierr == MPI_ERR_IN_STATUS) then   
            print *, "Look in status for error value"
        else if (ierr == MPI_ERR_PENDING) then   
            print *, "Pending request"
        else if (ierr == MPI_ERR_REQUEST) then   
            print *, "illegal mpi_request handle"
        else if (ierr == MPI_ERR_LASTCODE) then   
            print *, "Last error code -- always at end "
        else if (ierr /= MPI_SUCCESS) then   
            print *, "other error", ierr
        end if 
        
        if (ierr /= MPI_SUCCESS) then
            stop
        end if
    end subroutine
#endif
end module
