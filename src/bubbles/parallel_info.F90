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
! This module handles the communication 
module ParallelInfo_class
    use ISO_FORTRAN_ENV
    use Globals_m
    use Evaluators_class
    use Grid_class
    use Bubbles_class
#ifdef HAVE_CUDA
    use cuda_m
    use ISO_C_BINDING
#endif
    implicit none

    public :: ParallelInfo, SerialInfo
#ifdef HAVE_CUDA
    public :: Function3DMultiplier_destroy_cuda, Function3DMultiplier_download_result_cuda
    public :: Function3DMultiplier_multiply_cuda, Function3DMultiplier_init_cuda
    public :: Function3DMultiplier_upload_cubes_cuda, Function3DMultiplier_set_host_result_cube_cuda
#endif
    public :: assignment(=)

    private 

    type, abstract              :: ParallelInfo
        !> Pointer to the grid of the total area of the cube
        type(Grid3D), pointer                              :: grid
        integer                                            :: iproc = 0, nproc = 1
#ifdef HAVE_CUDA
        type(C_PTR), allocatable                           :: function3d_multiplier
        type(Grid3D)                                       :: function3d_multiplier_grid
#endif
    contains
        procedure(get_cube_ranges), public, deferred       :: get_cube_ranges
        procedure(get_cell_index_ranges), public, deferred :: get_cell_index_ranges
        procedure(get_ranges), public, deferred            :: get_ranges
        procedure(get_grid), public, deferred              :: get_grid
        procedure(get_global_grid), public, deferred       :: get_global_grid
        procedure(communicate_cube), public, deferred      :: communicate_cube
        procedure(communicate_cube_borders), public, deferred  :: communicate_cube_borders
        procedure(sum_matrix), public, deferred            :: sum_matrix
        procedure(sum_real), public, deferred              :: sum_real
        procedure(integrate_cube), public, deferred        :: integrate_cube
        procedure(destroy), public, deferred               :: destroy
#ifdef HAVE_CUDA
        procedure(destroy_cuda), public, deferred          :: destroy_cuda
#endif
        procedure(get_domain_bubbles), public, deferred    :: get_domain_bubbles
        procedure(get_domain_bubble_centers), public, deferred  :: get_domain_bubble_centers
        procedure(get_domain_bubble_ibubs), public, deferred  :: get_domain_bubble_ibubs
        procedure(get_processor_number), public, deferred  :: get_processor_number
        procedure, public                                  :: get_bubble_domain_indices => &
                                                              ParallelInfo_get_bubble_domain_indices
        procedure, public                                  :: communicate_bubbles => &
                                                              ParallelInfo_communicate_bubbles
        
        procedure, public                                  :: get_multiplication_cell_limits => &
                                                                  ParallelInfo_get_multiplication_cell_limits
#ifdef HAVE_CUDA  
        procedure, public                                  :: init_Function3D_multiplier => &
                                                                  ParallelInfo_init_Function3D_multiplier
#endif
    end type

    abstract interface
        function get_cube_ranges(self, input_cell_limits) result(ranges)
            import ParallelInfo
            class(ParallelInfo), intent(in)    :: self
            ! start and end grid indices of the cell
            integer, intent(in), optional      :: input_cell_limits(2, 3)
            ! start and end indeces of the cube grid
            integer                            :: ranges(2, 3)
        end function

        function get_cell_index_ranges(self) result(ranges)
            import ParallelInfo
            class(ParallelInfo), intent(in)    :: self
            ! start and end cell indeces of the grid
            integer                            :: ranges(2, 3)
        end function

        function get_ranges(self) result(ranges)
            import ParallelInfo, REAL64
            class(ParallelInfo), intent(in)    :: self
            ! start and end cell cartesian coordinates of the grid
            real(REAL64)                       :: ranges(2, 3)
        end function

        function get_grid(self) result(grid)
            import ParallelInfo, Grid3D
            class(ParallelInfo), intent(in), target    :: self
            type(Grid3D), pointer                      :: grid
        end function

        function get_global_grid(self) result(grid)
            import ParallelInfo, Grid3D
            class(ParallelInfo), intent(in), target    :: self
            type(Grid3D), pointer                      :: grid
        end function

        subroutine communicate_cube(self, cube, sum_borders, reversed_order)
            import ParallelInfo, REAL64
            class(ParallelInfo), intent(in)          :: self
            real(REAL64), intent(inout), pointer     :: cube(:, :, :)
            logical, intent(in), optional            :: sum_borders
            logical, intent(in), optional            :: reversed_order
        end subroutine

        subroutine communicate_cube_borders(self, cube, reversed_order, sum_result)
            import ParallelInfo, REAL64
            class(ParallelInfo), intent(in)          :: self
            real(REAL64), intent(inout), pointer     :: cube(:, :, :)
            logical, intent(in), optional            :: reversed_order
            logical, intent(in), optional            :: sum_result
        end subroutine

        subroutine sum_matrix(self, matrix)
            import ParallelInfo, REAL64
            class(ParallelInfo), intent(in) :: self
            real(REAL64), intent(inout)     :: matrix(:, :)
        end subroutine

        subroutine sum_real(self, number)
            import ParallelInfo, REAL64
            class(ParallelInfo), intent(in)   :: self
            real(REAL64), intent(inout)     :: number
        end subroutine

        function integrate_cube(self, cube) result(reslt)
            import ParallelInfo, REAL64
            class(ParallelInfo), intent(in) :: self
            real(REAL64),        intent(in) :: cube(:, :, :)
            real(REAL64)                    :: reslt
        end function

#ifdef HAVE_CUDA
        subroutine destroy_cuda(self)
            import ParallelInfo
            class(ParallelInfo), intent(inout) :: self
        end subroutine
#endif
 
        subroutine destroy(self)
            import ParallelInfo
            class(ParallelInfo), intent(inout) :: self
        end subroutine

        function get_domain_bubbles(self, bubbls) result(result_bubbles)
            import ParallelInfo, Bubbles
            class(ParallelInfo)              :: self
            type(Bubbles),        intent(in) :: bubbls
            type(Bubbles)                    :: result_bubbles
        end function

        function get_domain_bubble_centers(self, bubbls) result(domain_bubble_centers)
            import ParallelInfo, Bubbles, REAL64
            class(ParallelInfo)              :: self
            type(Bubbles),        intent(in) :: bubbls
            real(REAL64), allocatable        :: domain_bubble_centers(:, :)
        end function

        function get_domain_bubble_ibubs(self, bubbls) result(ibubs)
            import ParallelInfo, Bubbles, INT32
            class(ParallelInfo)              :: self
            type(Bubbles),        intent(in) :: bubbls
            integer(INT32), allocatable      :: ibubs(:)
        end function

        function get_processor_number(self) result(reslt)
            import ParallelInfo
            class(ParallelInfo)              :: self
            integer                          :: reslt
        end function

    end interface

    interface assignment(=)
        module procedure :: ParallelInfo_assign
    end interface
    
    type, public, extends(ParallelInfo) :: SerialInfo
#ifdef HAVE_CUDA
        type(C_PTR), allocatable      :: cuda_integrator
#endif
    contains
        procedure, public               :: get_cube_ranges => SerialInfo_get_cube_ranges
        procedure, public               :: get_ranges => SerialInfo_get_ranges
        procedure, public               :: get_cell_index_ranges => SerialInfo_get_cell_index_ranges
        procedure, public               :: get_grid => SerialInfo_get_grid
        procedure, public               :: get_global_grid => SerialInfo_get_global_grid
        procedure, public               :: communicate_cube => SerialInfo_communicate_cube
        procedure, public               :: communicate_cube_borders => SerialInfo_communicate_cube_borders
        procedure, public               :: sum_matrix => SerialInfo_sum_matrix
        procedure, public               :: sum_real => SerialInfo_sum_real
        procedure, public               :: destroy => SerialInfo_destroy
        procedure, public               :: integrate_cube => SerialInfo_integrate_cube
        procedure, public               :: get_domain_bubbles => SerialInfo_get_domain_bubbles
        procedure, public               :: get_domain_bubble_centers => SerialInfo_get_domain_bubble_centers
        procedure, public               :: get_domain_bubble_ibubs => SerialInfo_get_domain_bubble_ibubs
        procedure, public               :: get_processor_number => SerialInfo_get_processor_number
#ifdef HAVE_CUDA
        procedure, public               :: destroy_cuda => SerialInfo_destroy_cuda
#endif
    end type

    interface SerialInfo
         module procedure :: SerialInfo_init
    end interface
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%               Function3DMultiplier interfaces                          %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#ifdef HAVE_CUDA

    
    interface
        type(C_PTR) function Function3DMultiplier_init_cuda(grid, streamContainer) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value :: grid
            type(C_PTR), value :: streamContainer
        end function
    end interface

    interface
        subroutine Function3DMultiplier_destroy_cuda(multiplier) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value :: multiplier
        end subroutine
    end interface

    interface
        subroutine Function3DMultiplier_download_result_cuda(multiplier) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value :: multiplier
        end subroutine
    end interface
    
        
    interface
        subroutine Function3DMultiplier_upload_cubes_cuda(multiplier, &
                       cube1, cube1_offset, cube1_shape, cube1_lmax, &
                       cube2, cube2_offset, cube2_shape, cube2_lmax) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: multiplier
            real(C_DOUBLE), intent(in) :: cube1(*)
            integer(C_INT), value :: cube1_offset
            integer(C_INT)        :: cube1_shape(3)
            integer(C_INT), value :: cube1_lmax
            real(C_DOUBLE), intent(in) :: cube2(*)
            integer(C_INT), value :: cube2_offset
            integer(C_INT)        :: cube2_shape(3)
            integer(C_INT), value :: cube2_lmax
        end subroutine
    end interface

    interface
        subroutine Function3DMultiplier_set_host_result_cube_cuda(multiplier, &
                       cube, cube_offset, cube_shape) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: multiplier
            real(C_DOUBLE)        :: cube(*)
            integer(C_INT), value :: cube_offset
            integer(C_INT)        :: cube_shape(3)
        end subroutine
    end interface

    interface
        subroutine Function3DMultiplier_multiply_cuda(multiplier, &
                       bubbles1, bubbles2, result_bubbles) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value :: multiplier
            type(C_PTR), value :: bubbles1
            type(C_PTR), value :: bubbles2
            type(C_PTR), value :: result_bubbles
        end subroutine
    end interface
#endif

contains

    subroutine ParallelInfo_assign(self, parallel_info)
        class(ParallelInfo), intent(inout), allocatable      :: self
        class(ParallelInfo), intent(in)                      :: parallel_info
        allocate(self, source = parallel_info)
    end subroutine

#ifdef HAVE_CUDA
    subroutine ParallelInfo_init_Function3D_multiplier(self)
        class(ParallelInfo), intent(inout)        :: self
        type(Grid3D), pointer                     :: grid
        type(Grid3D)                              :: subgrid
        integer                                   :: multiplication_cell_limits(2, 3)
        
        if (.not. allocated(self%function3d_multiplier)) then
        
            ! select only the part of the grid that needs to be multiplied by this node
            multiplication_cell_limits = self%get_multiplication_cell_limits()

            ! get the grid 
            grid => self%get_grid()

            ! and get the part corresponding needed in multiplication
            self%function3d_multiplier_grid = grid%get_subgrid(multiplication_cell_limits)

            ! and init the multiplier
            self%function3d_multiplier = Function3DMultiplier_init_cuda( &
                                            self%function3d_multiplier_grid%get_cuda_interface(), &
                                            stream_container)

        end if
    end subroutine
#endif

    
    function ParallelInfo_get_bubble_domain_indices(self, nbub, lmax, iproc) result(domain)
        class(ParallelInfo), intent(in)  :: self
        integer,             intent(in)  :: nbub, lmax
        integer, optional,   intent(in)  :: iproc
        integer                          :: domain(2), total_bubble_count, remainder, &
                                            bubbles_per_processor, processor_number

        if (present(iproc)) then
            processor_number = iproc
        else
            processor_number = self%iproc
        end if

        total_bubble_count = nbub * (lmax+1)**2
        if (self%nproc >= total_bubble_count) then
            if ((processor_number +1) <= total_bubble_count) then
                domain(1) = processor_number +1
                domain(2) = processor_number +1
            else 
                domain(1) = 0
                domain(2) = -1
            end if
        else 
            remainder = mod(total_bubble_count, self%nproc)
            bubbles_per_processor = total_bubble_count / self%nproc
            if (processor_number >= remainder) then
                domain(1) = remainder + processor_number * bubbles_per_processor + 1
                domain(2) = domain(1) + bubbles_per_processor -1
            else
                domain(1) = processor_number + processor_number * bubbles_per_processor + 1
                domain(2) = domain(1) + bubbles_per_processor
            end if
        end if
    end function

    subroutine ParallelInfo_communicate_bubbles(self, bubbls)
        class(ParallelInfo), intent(in)    :: self
        type(Bubbles),       intent(inout) :: bubbls
        integer                            :: ibub
        do ibub = 1, bubbls%get_nbub()
            call self%sum_matrix(bubbls%bf(ibub)%p)
        end do
    end subroutine

    function ParallelInfo_get_multiplication_cell_limits(self) result(limits_multiplication)
        class(ParallelInfo), intent(in)    :: self
        ! box limits for ibox and multiplication domain as integers
        integer                            :: limits_multiplication(2,3)

        limits_multiplication(1, :) = 1
        limits_multiplication(2, :) = [self%grid%axis(X_)%get_ncell(), &
                                       self%grid%axis(Y_)%get_ncell(), &
                                       self%grid%axis(Z_)%get_ncell()]
    end function 

#ifdef HAVE_CUDA
    !> Function that destroys the cuda integrator.
    !! NOTE: this function only has to be called once at the end of execution for a parallelinfo type
    !! for each grid size
    subroutine SerialInfo_destroy_cuda(self)
        class(SerialInfo), intent(inout)  :: self
        if (allocated(self%cuda_integrator)) then
            call Integrator3D_destroy(self%cuda_integrator)
            deallocate(self%cuda_integrator)
        end if
        
        if (allocated(self%function3d_multiplier)) then
            call Function3DMultiplier_destroy_cuda(self%function3d_multiplier)
            call self%function3d_multiplier_grid%cuda_destroy()
            call self%function3d_multiplier_grid%destroy()
            deallocate(self%function3d_multiplier)
        end if
    end subroutine
#endif

    function SerialInfo_init(grid) result(new)
        type(Grid3D), intent(in), target  :: grid
        type(SerialInfo)                  :: new

        new%grid => grid
        new%nproc = 1
        new%iproc = 0
#ifdef HAVE_CUDA
        ! NOTE: the following function resides in integrator.cu -file and is a 
        ! wrapper for a class constructor, the stream_container is a global variable
        ! inited at core.F90
        new%cuda_integrator = integrator3d_init(stream_container, new%grid%get_cuda_interface())
        call new%init_Function3D_multiplier()
#endif
    end function

    function SerialInfo_get_cube_ranges(self, input_cell_limits) result(ranges)
        class(SerialInfo), intent(in)      :: self
        ! start and end grid indices of the cell
        integer, intent(in), optional             :: input_cell_limits(2, 3)
        ! start and end grid indices of the cube
        integer                            :: ranges(2, 3), cell_limits(2, 3)

        if (present(input_cell_limits)) then
            cell_limits = input_cell_limits
        else
            cell_limits(1, :) = 1
            cell_limits(2, X_) = self%grid%get_ncell(X_)
            cell_limits(2, Y_) = self%grid%get_ncell(Y_)
            cell_limits(2, Z_) = self%grid%get_ncell(Z_)
        end if
 
        ! each cell has nlip grid points and the first and two sequental cells share a grid point
        ranges(1, :) = (cell_limits(1, :) - [1, 1, 1]) * (self%grid%get_nlip() - 1) + 1
        ranges(2, :) = (cell_limits(2, :)) * (self%grid%get_nlip() - 1) + 1
    end function

    function SerialInfo_get_ranges(self) result(ranges)
        class(SerialInfo), intent(in)      :: self
        ! start and end grid indices of the cube
        real(REAL64)                       :: ranges(2, 3)

        ranges = self%grid%get_range()
    end function

    function SerialInfo_get_cell_index_ranges(self) result(ranges)
        class(SerialInfo), intent(in)      :: self
        ! start and end cell indices of the cube
        integer                            :: ranges(2, 3)

        ranges(1, :) = 1
        ranges(2, X_) = self%grid%get_ncell(X_)
        ranges(2, Y_) = self%grid%get_ncell(Y_)
        ranges(2, Z_) = self%grid%get_ncell(Z_)
    end function

    function SerialInfo_get_grid(self) result(grid)
        class(SerialInfo), intent(in), target    :: self
        type(Grid3D), pointer                    :: grid
        grid => self%grid
    end function

    function SerialInfo_get_global_grid(self) result(grid)
        class(SerialInfo), intent(in), target    :: self
        type(Grid3D), pointer                    :: grid
        grid => self%grid
    end function

    subroutine SerialInfo_communicate_cube(self, cube, sum_borders, reversed_order)
        class(SerialInfo), intent(in)            :: self
        real(REAL64), intent(inout), pointer     :: cube(:, :, :)
        logical, intent(in), optional            :: sum_borders
        logical, intent(in), optional            :: reversed_order
    end subroutine

    subroutine SerialInfo_communicate_cube_borders(self, cube, reversed_order, sum_result)
        class(SerialInfo), intent(in)            :: self
        real(REAL64), intent(inout), pointer     :: cube(:, :, :)
        logical, intent(in), optional            :: reversed_order
        logical, intent(in), optional            :: sum_result
    end subroutine

    subroutine SerialInfo_sum_matrix(self, matrix)
        class(SerialInfo), intent(in)   :: self
        real(REAL64), intent(inout)     :: matrix(:, :)
    end subroutine

    subroutine SerialInfo_sum_real(self, number)
        class(SerialInfo), intent(in)   :: self
        real(REAL64), intent(inout)     :: number
    end subroutine

    subroutine SerialInfo_destroy(self)
        class(SerialInfo), intent(inout) :: self
 
        nullify(self%grid)
#ifdef HAVE_CUDA
        call self%destroy_cuda()
#endif
    end subroutine

    function SerialInfo_integrate_cube(self, cube) result(reslt)
        class(SerialInfo),   intent(in) :: self
        real(REAL64),        intent(in) :: cube(:, :, :)
        real(REAL64)                    :: reslt
        type(Integrator)                :: integr

#ifdef HAVE_CUDA
        !call register_host_cube_cuda(cube, shape(cube))
        ! NOTE: the following function resides in integrator.cu -file and is a 
        ! wrapper for a integrator3d C++/cuda class function
        call integrator3d_integrate(self%cuda_integrator, cube, 0, shape(cube), reslt)
        !call unregister_host_cube_cuda(cube)
#else    
        integr=Integrator(self%grid%axis)
        reslt=integr%eval(cube)
        
        call integr%destroy()
#endif
    end function

    function SerialInfo_get_domain_bubbles(self, bubbls) result(result_bubbles)
        class(SerialInfo)                :: self
        type(Bubbles),        intent(in) :: bubbls
        type(Bubbles)                    :: result_bubbles

        result_bubbles = bubbls
    end function

    function SerialInfo_get_domain_bubble_centers(self, bubbls) result(domain_bubble_centers)
        class(SerialInfo)                :: self
        type(Bubbles),        intent(in) :: bubbls
        real(REAL64), allocatable        :: domain_bubble_centers(:, :)

        domain_bubble_centers = bubbls%get_centers()
    end function

    function SerialInfo_get_domain_bubble_ibubs(self, bubbls) result(ibubs)
        class(SerialInfo)                :: self
        type(Bubbles),        intent(in) :: bubbls
        integer(INT32), allocatable      :: ibubs(:)

        ibubs = bubbls%get_ibubs()
    end function

    function SerialInfo_get_processor_number(self) result(reslt)
        class(SerialInfo)              :: self
        integer                         :: reslt

        reslt = 1
    end function
end module