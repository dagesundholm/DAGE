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
module CoreEvaluator_class
    use ISO_FORTRAN_ENV
    use globals_m
    use lipbasis_class
    use function3d_class
    use Function3D_types_m
    use Grid_class 
    use bubbles_class
    use Evaluators_class
    use RealSphericalHarmonics_class
    use Laplacian3D_class
    use Points_class
#ifdef HAVE_OMP
    use omp_lib
#endif
    
    public :: CoreEvaluator, destroy_core_functions
    public :: assignment(=)
    
    type :: CoreEvaluator
        type(Function3DEvaluator), allocatable    :: core_grid_evaluators(:)
        type(Laplacian3D),         allocatable    :: core_laplacians(:)
        type(Grid3D),              allocatable    :: core_grids(:)
        type(Grid3D),              allocatable    :: core_sparse_grids(:)
        type(SerialInfo),          allocatable    :: core_parallel_infos(:)
        type(PointCoordinates),    allocatable    :: core_point_coordinates(:)
        real(REAL64)                              :: core_magnification =  1.96d0
        integer                                   :: core_grid_point_count = 25
    contains
        procedure, private :: init_core_grids         => CoreEvaluator_init_core_grids
        procedure, private :: init_core_point_coordinates &
                                                      => CoreEvaluator_init_core_point_coordinates
        procedure, private :: collapse_core_cube      => CoreEvaluator_collapse_core_cube
        procedure, public  :: collapse_core_functions => CoreEvaluator_collapse_core_functions
        procedure, public  :: init_core_functions     => CoreEvaluator_init_core_functions
        procedure, public  :: evaluate_core_electron_density => CoreEvaluator_evaluate_core_electron_density
        procedure, public  :: evaluate_core_gradient  => CoreEvaluator_evaluate_core_gradient
        procedure, public  :: evaluate_core_divergence=> CoreEvaluator_evaluate_core_divergence
        procedure, public  :: evaluate_core_gradient_and_collapse  => CoreEvaluator_evaluate_core_gradient_and_collapse
        procedure, public  :: evaluate_core_divergence_and_collapse=> CoreEvaluator_evaluate_core_divergence_and_collapse
        procedure, public  :: evaluate_core_laplacian => CoreEvaluator_evaluate_core_laplacian
        procedure, public  :: evaluate_contaminants   => CoreEvaluator_evaluate_contaminants
        procedure, public  :: evaluate_points_and_collapse &
                                                      => CoreEvaluator_evaluate_points_and_collapse
        procedure, public  :: destroy                 => CoreEvaluator_destroy
    end type

    
    interface CoreEvaluator
        module procedure :: CoreEvaluator_init
    end interface
contains

    function CoreEvaluator_init(prototype_function, core_magnification, core_grid_point_count) result(new)
        type(Function3D),  intent(in) :: prototype_function
        integer, optional, intent(in) :: core_grid_point_count
        integer, optional, intent(in) :: core_magnification
        type(CoreEvaluator), target   :: new
        integer                       :: i, j, ibubs(prototype_function%bubbles%get_nbub() - 1)
        type(Grid3D), pointer         :: grid
        type(Function3D)              :: temp

        if (present(core_grid_point_count)) new%core_grid_point_count = core_grid_point_count
        if (present(core_magnification))    new%core_magnification    = core_magnification
        call temp%init_copy(prototype_function, lmax = 14)
        call new%init_core_grids(prototype_function)
        allocate(new%core_grid_evaluators(size(new%core_grids)))
        do i = 1, size(new%core_grids)
            grid => new%core_grids(i)
            new%core_grid_evaluators(i) = Function3DEvaluator(.FALSE.)
        end do
        call temp%destroy()
    end function
    
    subroutine CoreEvaluator_destroy(self)
        class(CoreEvaluator),        intent(inout) :: self
        integer                                    :: i
        
        if (allocated(self%core_grids)) then
            do i = 1, size(self%core_grids)
#ifdef HAVE_CUDA
                call self%core_grids(i)%cuda_destroy()
#endif
                call self%core_grids(i)%destroy()
            end do
        end if
        
        if (allocated(self%core_sparse_grids)) then
            do i = 1, size(self%core_sparse_grids)
#ifdef HAVE_CUDA
                call self%core_sparse_grids(i)%cuda_destroy()
#endif
                call self%core_sparse_grids(i)%destroy()
            end do
        end if
        
        if (allocated(self%core_parallel_infos)) then
            do i = 1, size(self%core_parallel_infos)
#ifdef HAVE_CUDA
                call self%core_parallel_infos(i)%destroy_cuda()
#endif
                call self%core_parallel_infos(i)%destroy()
            end do
        end if
        
        if (allocated(self%core_grid_evaluators)) then
            do i = 1, size(self%core_grid_evaluators)
                call self%core_grid_evaluators(i)%destroy()
            end do
        end if
        
        if (allocated(self%core_laplacians)) then
            do i = 1, size(self%core_laplacians)
#ifdef HAVE_CUDA
                call self%core_laplacians(i)%cuda_destroy()
#endif
                call self%core_laplacians(i)%destroy()
            end do
        end if
        
        if (allocated(self%core_point_coordinates)) then
            do i = 1, size(self%core_point_coordinates)
                call self%core_point_coordinates(i)%destroy()
            end do
            deallocate(self%core_point_coordinates)
        end if
    end subroutine

    subroutine CoreEvaluator_init_core_point_coordinates(self, global_point_coordinates)
        class(CoreEvaluator),        intent(inout) :: self
        type(PointCoordinates),      intent(in)    :: global_point_coordinates
        integer                                    :: i
        if (allocated(self%core_point_coordinates)) then
            do i = 1, size(self%core_point_coordinates)
                call self%core_point_coordinates(i)%destroy()
            end do
            deallocate(self%core_point_coordinates)
        end if
        allocate(self%core_point_coordinates(size(self%core_grids)))
        do i = 1, size(self%core_grids)
            self%core_point_coordinates(i) = &
                global_point_coordinates%get_point_coordinates_within_grid(self%core_grids(i))
        end do

    end subroutine


    subroutine CoreEvaluator_init_core_grids(self, prototype_function)
        implicit none
        class(CoreEvaluator),        intent(inout) :: self
        type(Function3D),            intent(in)    :: prototype_function
        integer                                    :: i, x, y, z, nlip, ix, iy, iz, ngridpoints, offset, &
                                                      number_of_cells, start_x, start_y, start_z, &
                                                      end_x, end_y, end_z, grid_type
        real(REAL64)                               :: qmin(3), center(3), c2(3)
        ! steps / h / scales of all cells (and all axes).  By design this routine only does equisized cells
        real(REAL64), allocatable                  :: stepx(:), stepy(:), stepz(:)
        real(REAL64), pointer                      :: coordinates_x(:), coordinates_y(:), coordinates_z(:)

        
        nlip = prototype_function%grid%get_nlip()
        grid_type = prototype_function%grid%get_grid_type()
        ngridpoints = nint(self%core_magnification * self%core_grid_point_count)
        coordinates_x => prototype_function%grid%axis(X_)%get_coord()
        coordinates_y => prototype_function%grid%axis(Y_)%get_coord()
        coordinates_z => prototype_function%grid%axis(Z_)%get_coord()
        allocate(self%core_grids(prototype_function%bubbles%get_nbub()))
        allocate(self%core_sparse_grids(prototype_function%bubbles%get_nbub()))
        allocate(self%core_parallel_infos(prototype_function%bubbles%get_nbub()))
        allocate(self%core_laplacians(prototype_function%bubbles%get_nbub()))
        
        allocate(stepx((ngridpoints-1) / (nlip-1)))
        allocate(stepy((ngridpoints-1) / (nlip-1)))
        allocate(stepz((ngridpoints-1) / (nlip-1)))
        do i = 1, prototype_function%bubbles%get_nbub()
            center = prototype_function%bubbles%get_centers(i)
            c2(X_) = prototype_function%grid%axis(X_)%coordinate_to_grid_point_coordinate(center(X_))
            c2(Y_) = prototype_function%grid%axis(Y_)%coordinate_to_grid_point_coordinate(center(Y_))
            c2(Z_) = prototype_function%grid%axis(Z_)%coordinate_to_grid_point_coordinate(center(Z_))
            
            if (mod(self%core_grid_point_count, 2) == 0) then
                start_x = floor(c2(X_)) - self%core_grid_point_count / 2 + 1
                start_y = floor(c2(Y_)) - self%core_grid_point_count / 2 + 1
                start_z = floor(c2(Z_)) - self%core_grid_point_count / 2 + 1
                end_x   = ceiling(c2(X_)) + self%core_grid_point_count / 2 - 1
                end_y   = ceiling(c2(Y_)) + self%core_grid_point_count / 2 - 1
                end_z   = ceiling(c2(Z_)) + self%core_grid_point_count / 2 - 1
            else
                start_x = nint(c2(X_)) - self%core_grid_point_count / 2
                start_y = nint(c2(Y_)) - self%core_grid_point_count / 2
                start_z = nint(c2(Z_)) - self%core_grid_point_count / 2 
                end_x   = nint(c2(X_)) + self%core_grid_point_count / 2
                end_y   = nint(c2(Y_)) + self%core_grid_point_count / 2 
                end_z   = nint(c2(Z_)) + self%core_grid_point_count / 2
            end if

            qmin(X_) = coordinates_x(start_x)
            qmin(Y_) = coordinates_y(start_y)
            qmin(Z_) = coordinates_z(start_z)
        
            stepx(:) = (coordinates_x(end_x) - coordinates_x(start_x)) / (ngridpoints-1)
            stepy(:) = (coordinates_y(end_y) - coordinates_y(start_y)) / (ngridpoints-1)
            stepz(:) = (coordinates_z(end_z) - coordinates_z(start_z)) / (ngridpoints-1)
            self%core_grids(i) = Grid3D(qmin, &
                                        [(ngridpoints-1) / (nlip-1), &
                                         (ngridpoints-1) / (nlip-1), &
                                         (ngridpoints-1) / (nlip-1)], &
                                        nlip, stepx, stepy, stepz, grid_type)
#ifdef HAVE_CUDA
            call self%core_grids(i)%cuda_init()
#endif
            self%core_parallel_infos(i) = SerialInfo(self%core_grids(i))
            self%core_laplacians(i) = Laplacian3D(self%core_parallel_infos(i), &
                                                  self%core_parallel_infos(i))
#ifdef HAVE_CUDA
            call self%core_laplacians(i)%cuda_init()
#endif
            stepx(:) = (coordinates_x(end_x) - coordinates_x(start_x)) / (end_x - start_x)
            stepy(:) = (coordinates_y(end_y) - coordinates_y(start_y)) / (end_y - start_y)
            stepz(:) = (coordinates_z(end_z) - coordinates_z(start_z)) / (end_z - start_y)
            self%core_sparse_grids(i) = Grid3D(qmin, &
                                               [(end_x - start_x) / (nlip-1), &
                                                (end_y - start_y) / (nlip-1), &
                                                (end_z - start_z) / (nlip-1)], &
                                               nlip, stepx, stepy, stepz, grid_type)
        end do
        deallocate(stepx, stepy, stepz)
        nullify(coordinates_x, coordinates_y, coordinates_z)
    end subroutine

    subroutine CoreEvaluator_evaluate_contaminants(self, global_function, taylor_order, ignore_bubbles)
        class(CoreEvaluator),           intent(inout) :: self
        type(Function3D),               intent(inout) :: global_function
        integer,           optional,    intent(in)    :: taylor_order
        logical,           optional,    intent(in)    :: ignore_bubbles
        type(Function3D),  allocatable                :: core_function(:), temp_function(:)
        real(REAL64),      allocatable                :: cube_contaminants(:, :), bubbles_contaminants(:, :)
        integer                                       :: cube_shape(3), taylor_order_
        integer                                       :: i
        integer                                       :: ibubs(global_function%bubbles%get_nbub() - 1)
        logical                                       :: ignore_bubbles_
        
        taylor_order_ = global_function%taylor_order
        if (present(taylor_order)) taylor_order_ = taylor_order
        ignore_bubbles_ = .FALSE.
        if (present(ignore_bubbles)) ignore_bubbles_ = ignore_bubbles
        
        call self%init_core_functions(core_function, global_function, copy_bubbles = .TRUE.)
        call self%init_core_functions(temp_function, global_function, copy_bubbles = .FALSE.)

        cube_contaminants = global_function%get_cube_contaminants(taylor_order_)
        bubbles_contaminants = global_function%bubbles%get_contaminants(taylor_order_)
        deallocate(cube_contaminants)
        deallocate(bubbles_contaminants)

        if (allocated(global_function%cube_contaminants))   deallocate(global_function%cube_contaminants)
        if (.not. ignore_bubbles_ .and. allocated(global_function%bubbles_contaminants)) &
            deallocate(global_function%bubbles_contaminants)
        allocate(global_function%cube_contaminants((taylor_order_+1)*(taylor_order_+2)*&
                                  (taylor_order_+3)/6, global_function%bubbles%get_nbub_global()), &
                                   source = 0.0d0)
        if (.not. ignore_bubbles_) allocate(global_function%bubbles_contaminants((taylor_order_+1)*(taylor_order_+2)*&
                                  (taylor_order_+3)/6, global_function%bubbles%get_nbub_global()), &
                                   source = 0.0d0)
        global_function%taylor_order = taylor_order_
        
        do i = 1, global_function%bubbles%get_nbub()
            do j = 1, i-1
                ibubs(j) = j
            end do
            do j = i+1, global_function%bubbles%get_nbub()
                ibubs(j-1) = j
            end do
            ! NOTE: the evaluation with the core grid evaluators only takes into account 
            ! bubbles with ibub != i
            call self%core_grid_evaluators(i)%evaluate_grid(core_function(i), temp_function(i)%cube(:, :, :), &
                                                            ibubs = ibubs, ignore_bubbles = ignore_bubbles_)
            
            cube_contaminants =  &
                temp_function(i)%get_cube_contaminants(taylor_order_, &
                                                       bubbls = global_function%bubbles)
            
            global_function%cube_contaminants(:, :) =   global_function%cube_contaminants(:, :) &
                                                      + cube_contaminants(:, :)

            deallocate(cube_contaminants)
            
        end do
        call destroy_core_functions(core_function)
        call destroy_core_functions(temp_function)
    end subroutine

    subroutine CoreEvaluator_evaluate_core_laplacian(self, global_function, global_laplacian)
        class(CoreEvaluator),           intent(in)    :: self
        type(Function3D),               intent(in)    :: global_function
        class(Function3D),              intent(inout) :: global_laplacian
        type(Function3D),  allocatable                :: core_function(:), core_laplacian(:)
        class(Function3D), allocatable                :: kin   
        integer                                       :: i
        
        call self%init_core_functions(core_function, global_function)
        allocate(core_laplacian(global_function%bubbles%get_nbub()))
        do i = 1, global_function%bubbles%get_nbub()
            call self%core_laplacians(i)%operate_on(core_function(i), kin, only_cube = .TRUE.)
            call core_laplacian(i)%init_copy(kin, copy_content = .TRUE.)
            call kin%destroy()
            deallocate(kin)
        end do
        call self%collapse_core_functions(global_laplacian, core_laplacian)
        call destroy_core_functions(core_laplacian)
    end subroutine

    subroutine CoreEvaluator_evaluate_core_divergence(self, global_input_x, global_input_y, global_input_z, core_divergence)
        class(CoreEvaluator),           intent(inout) :: self
        type(Function3D),               intent(in)    :: global_input_x, global_input_y, global_input_z
        type(Function3D),  allocatable, intent(out)   :: core_divergence(:)
        type(Function3D),  allocatable                :: core_input_x(:), core_input_y(:), core_input_z(:)
        integer                                       :: i
        call self%init_core_functions(core_input_x, global_input_x)
        call self%init_core_functions(core_input_y, global_input_y)
        call self%init_core_functions(core_input_z, global_input_z)
        allocate(core_divergence(global_input_x%bubbles%get_nbub()))
        do i = 1, global_input_x%bubbles%get_nbub()
            call self%core_grid_evaluators(i)%evaluate_divergence_as_Function3D(core_input_x(i), core_input_y(i), &
                                                                     core_input_z(i), core_divergence(i), &
                                                                     ignore_bubbles = .TRUE.)
        end do
        call destroy_core_functions(core_input_x)
        call destroy_core_functions(core_input_y)
        call destroy_core_functions(core_input_z)
    end subroutine

    subroutine CoreEvaluator_evaluate_core_divergence_and_collapse(self, global_input_x, global_input_y, &
                                                                   global_input_z, global_divergence)
        class(CoreEvaluator),           intent(inout) :: self
        type(Function3D),               intent(in)    :: global_input_x, global_input_y, global_input_z
        type(Function3D),               intent(inout) :: global_divergence
        type(Function3D),  allocatable                :: core_input_x(:), core_input_y(:), core_input_z(:), core_divergence(:)
        integer                                       :: i
        
        call self%evaluate_core_divergence(global_input_x, global_input_y, global_input_z, core_divergence)
        call self%collapse_core_functions(global_divergence, core_divergence)
        call destroy_core_functions(core_divergence)
    end subroutine
    
    subroutine CoreEvaluator_evaluate_core_electron_density(self, &
                                                    orbitals, &
                                                    core_electron_density)
        class(CoreEvaluator),           intent(inout) :: self
        type(Function3D),               intent(in)    :: orbitals(:)
        type(Function3D),  allocatable, intent(out)   :: core_electron_density(:)
        type(Function3D),  allocatable                :: temp
        type(Function3D),  allocatable                :: core_orbital(:)
        type(Bubbles)                                 :: result_bubbles
        integer                                       :: i, j
        allocate(core_electron_density(orbitals(1)%bubbles%get_nbub()))
        do i = 1, size(orbitals)
            call orbitals(i)%multiply_bubbles(orbitals(i), result_bubbles)
            call self%init_core_functions(core_orbital, orbitals(i), copy_bubbles = .TRUE.)
            do j = 1, size(core_electron_density)
                call core_orbital(j)%multiply_sub(core_orbital(j), temp, &
                                                  result_bubbles = result_bubbles)
                call temp%product_in_place_REAL64(2.0d0)
                if (i == 1) then
                    call core_electron_density(j)%init_copy(temp, copy_content = .TRUE.)
                else
                    call core_electron_density(j)%add_in_place(temp)
                end if
                call temp%destroy()
                deallocate(temp)
            end do
            
            call destroy_core_functions(core_orbital)
            call result_bubbles%destroy()
        end do
    end subroutine

    subroutine CoreEvaluator_evaluate_core_gradient(self, core_function, &
                                                    core_gradients_x, core_gradients_y, core_gradients_z)
        class(CoreEvaluator),           intent(inout) :: self
        type(Function3D),               intent(in)    :: core_function(:)
        type(Function3D),  allocatable, intent(out)   :: core_gradients_x(:), core_gradients_y(:), core_gradients_z(:)
        integer                                       :: i
        allocate(core_gradients_x(size(core_function)))
        allocate(core_gradients_y(size(core_function)))
        allocate(core_gradients_z(size(core_function)))
        do i = 1, size(core_function)
            call self%core_grid_evaluators(i)%evaluate_gradients_as_Function3Ds &
                 (core_function(i), core_gradients_x(i), &
                  core_gradients_y(i), core_gradients_z(i), ignore_bubbles = .TRUE.)
        end do
    end subroutine

    subroutine CoreEvaluator_evaluate_points_and_collapse(self, core_function, global_points, &
                                                          bubbls, reinit_core_point_coordinates)
        class(CoreEvaluator),           intent(inout) :: self
        type(Function3D),               intent(inout) :: core_function(:)
        type(Points),                   intent(inout) :: global_points
        type(Bubbles),                  intent(in)    :: bubbls 
        logical,                        intent(in)    :: reinit_core_point_coordinates
        integer                                       :: i
        type(Points), allocatable                     :: temp_points

        ! init the point coordinates object of the points that are within 
        ! the area of the core-grid
        if (reinit_core_point_coordinates) &
            call self%init_core_point_coordinates(global_points%point_coordinates)

        do i = 1, bubbls%get_nbub()
            temp_points = Points(self%core_point_coordinates(i))
            call core_function(i)%bubbles%destroy()
            core_function(i)%bubbles = Bubbles(bubbls, copy_content = .TRUE.)
            
            if (reinit_core_point_coordinates) &
                call self%core_grid_evaluators(i)%set_output_points(temp_points, force_reallocate = .TRUE.)
            
            ! do the evaluation of the points within the cube and 
            ! overwrite those points in the global points
            call self%core_grid_evaluators(i)%evaluate_points &
                 (core_function(i), temp_points)
            call global_points%overwrite(temp_points)

            ! clean up
            call temp_points%destroy()
            deallocate(temp_points)
            call core_function(i)%bubbles%destroy()
        end do
    end subroutine

    subroutine CoreEvaluator_evaluate_core_gradient_and_collapse(self, core_function, global_derivative_x, &
                                                    global_derivative_y, global_derivative_z)
        class(CoreEvaluator),           intent(inout) :: self
        type(Function3D),               intent(in)    :: core_function(:)
        type(Function3D),               intent(inout) :: global_derivative_x, global_derivative_y, global_derivative_z
        type(Function3D),  allocatable                :: core_gradients_x(:), &
                                                         core_gradients_y(:), core_gradients_z(:)
        integer                                       :: i
        call self%evaluate_core_gradient(core_function, &
                                         core_gradients_x, core_gradients_y, core_gradients_z)
        call self%collapse_core_functions(global_derivative_x, core_gradients_x)
        call self%collapse_core_functions(global_derivative_y, core_gradients_y)
        call self%collapse_core_functions(global_derivative_z, core_gradients_z)
        call destroy_core_functions(core_gradients_x)
        call destroy_core_functions(core_gradients_y)
        call destroy_core_functions(core_gradients_z)
    end subroutine

    function CoreEvaluator_collapse_core_cube(self, core_grid, core_sparse_grid, core_cube) result(res)
        class(CoreEvaluator),        intent(in)  :: self
        type(Grid3D), target,        intent(in)  :: core_grid
        type(Grid3D),               intent(in)   :: core_sparse_grid
        real(REAL64),               intent(in)   :: core_cube(:, :, :)
        real(REAL64)                             :: res(self%core_grid_point_count, &
                                                        self%core_grid_point_count, &
                                                        self%core_grid_point_count)
        integer                                  :: i, j, k, point_spacing, output_shape(3), input_shape(3)
        
        type(Interpolator)                       :: interpolator_
        type(Grid1DPointer)                      :: grid1d_pointers(3)
        real(REAL64), allocatable                :: gridpoints(:, :), results(:, :)
        
        
        
        grid1d_pointers(X_)%p => core_grid%axis(X_)
        grid1d_pointers(Y_)%p => core_grid%axis(Y_)
        grid1d_pointers(Z_)%p => core_grid%axis(Z_)
        interpolator_ = Interpolator(grid1d_pointers, 0)
        
        gridpoints = core_sparse_grid%get_all_grid_points()
        results = interpolator_%eval(core_cube, gridpoints)
        output_shape = core_sparse_grid%get_shape()
        res(:, :, :) = reshape(results(:, 1), output_shape)
        deallocate(results, gridpoints)
        
    end function

    subroutine CoreEvaluator_collapse_core_functions(self, global_function, core_functions, taylor_order)
        class(CoreEvaluator),        intent(in)     :: self
        class(Function3D),           intent(inout)  :: global_function
        type(Function3D),            intent(in)     :: core_functions(:)
        integer,           optional, intent(in)     :: taylor_order
        integer                                     :: i, x, y, z, nlip, start_x, start_y, start_z, &
                                                       end_x, end_y, end_z, taylor_order_
        real(REAL64)                                :: center(3), c2(3), &
                                                       collapsed_core(&
                                                       self%core_grid_point_count, &
                                                       self%core_grid_point_count, &
                                                       self%core_grid_point_count)
        real(REAL64), allocatable                   :: cube_contaminants(:, :)

        taylor_order_ = global_function%taylor_order
        if (present(taylor_order)) taylor_order_ = taylor_order
        
        nlip = global_function%grid%get_nlip()
        !cube_contaminants = global_function%get_cube_contaminants(taylor_order_)
        !print *, "cube contaminants before", cube_contaminants
        !if (allocated(global_function%cube_contaminants)) deallocate(global_function%cube_contaminants)
        !allocate(global_function%cube_contaminants((taylor_order_+1)*(taylor_order_+2)*&
        !                          (taylor_order_+3)/6, global_function%bubbles%get_nbub_global()), &
        !                           source = 0.0d0)
        global_function%taylor_order = taylor_order_

        do i = 1, global_function%bubbles%get_nbub()
            center = global_function%bubbles%get_centers(i)
            c2(X_) = global_function%grid%axis(X_)%coordinate_to_grid_point_coordinate(center(X_))
            c2(Y_) = global_function%grid%axis(Y_)%coordinate_to_grid_point_coordinate(center(Y_))
            c2(Z_) = global_function%grid%axis(Z_)%coordinate_to_grid_point_coordinate(center(Z_))
            
            if (mod(self%core_grid_point_count, 2) == 0) then
                start_x = floor(c2(X_)) - self%core_grid_point_count / 2 + 1
                start_y = floor(c2(Y_)) - self%core_grid_point_count / 2 + 1
                start_z = floor(c2(Z_)) - self%core_grid_point_count / 2 + 1
                end_x   = ceiling(c2(X_)) + self%core_grid_point_count / 2 - 1
                end_y   = ceiling(c2(Y_)) + self%core_grid_point_count / 2 - 1
                end_z   = ceiling(c2(Z_)) + self%core_grid_point_count / 2 - 1
            else
                start_x = nint(c2(X_)) - self%core_grid_point_count / 2
                start_y = nint(c2(Y_)) - self%core_grid_point_count / 2
                start_z = nint(c2(Z_)) - self%core_grid_point_count / 2 
                end_x   = nint(c2(X_)) + self%core_grid_point_count / 2
                end_y   = nint(c2(Y_)) + self%core_grid_point_count / 2 
                end_z   = nint(c2(Z_)) + self%core_grid_point_count / 2
            end if

            collapsed_core = self%collapse_core_cube(self%core_grids(i), self%core_sparse_grids(i), &
                                                     core_functions(i)%cube)   
            global_function%cube(start_x:end_x, start_y:end_y, start_z:end_z) = &
                 collapsed_core
            !cube_contaminants =  &
            !    core_functions(i)%get_cube_contaminants(taylor_order_, &
            !                                            bubbls = global_function%bubbles)
            !global_function%cube_contaminants(:, :) =   global_function%cube_contaminants(:, :) &
            !                                          + cube_contaminants(:, :)
            !deallocate(cube_contaminants)
        end do
        !print *, "cube contaminants after", global_function%cube_contaminants
        !cube_contaminants = global_function%bubbles%get_contaminants(taylor_order_)
        !print *, "bubbles contaminants after", cube_contaminants
        !deallocate(cube_contaminants)
    end subroutine
    
    subroutine CoreEvaluator_init_core_functions(self, core_functions, global_function, copy_bubbles)
        class(CoreEvaluator),        target,     intent(in)    :: self
        type(Function3D),            allocatable,intent(inout) :: core_functions(:)
        type(Function3D),            optional,   intent(in)    :: global_function
        logical,                     optional,   intent(in)    :: copy_bubbles
        integer                                                :: i, o, p, q, output_shape(3)
        real(REAL64), pointer                                  :: gridpoints_x(:), gridpoints_y(:), &
                                                                  gridpoints_z(:)
        real(REAL64), allocatable                              :: gridpoints(:, :), results(:, :)
        type(Interpolator)                                     :: interpolator_
        type(Grid1DPointer)                                    :: grid1d_pointers(3)
        logical                                                :: copy_bubbles_

        copy_bubbles_ = .FALSE.
        if (present(copy_bubbles)) copy_bubbles_ = copy_bubbles
        
        if(allocated(core_functions)) call destroy_core_functions(core_functions)
        allocate(core_functions(size(self%core_grids)))
        
        do i = 1, size(self%core_grids)
            call core_functions(i)%init_explicit(self%core_parallel_infos(i), type=F3D_TYPE_CUSP)
            
            if (copy_bubbles_) core_functions(i)%bubbles = Bubbles(global_function%bubbles, copy_content = .TRUE.)
        end do
        
        ! if global function is given, interpolate it at the points of the core
        ! function to get the initial values
        if (present(global_function)) then
            grid1d_pointers(X_)%p => global_function%grid%axis(X_)
            grid1d_pointers(Y_)%p => global_function%grid%axis(Y_)
            grid1d_pointers(Z_)%p => global_function%grid%axis(Z_)
            interpolator_ = Interpolator(grid1d_pointers, 0)
            do i = 1, size(self%core_grids)
                gridpoints = self%core_grids(i)%get_all_grid_points()
                results = interpolator_%eval(global_function%cube, gridpoints)
                output_shape = self%core_grids(i)%get_shape()
                core_functions(i)%cube(:, :, :) = reshape(results(:, 1), output_shape)
                deallocate(results)
                deallocate(gridpoints)
            end do
            call interpolator_%destroy()
        end if
    end subroutine 
    
    subroutine destroy_core_functions(core_functions)
        type(Function3D), allocatable,intent(inout) :: core_functions(:)
        integer                                     :: i
        
        do i = 1, size(core_functions)
            call core_functions(i)%destroy()
        end do
        deallocate(core_functions)
    end subroutine
end module
