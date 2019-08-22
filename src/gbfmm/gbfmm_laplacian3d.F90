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
module GBFMMLaplacian3D_class
    use ISO_FORTRAN_ENV
    use Laplacian3D_class
    use GBFMMParallelInfo_class
    use Grid_class
    use XMatrix_m
    use Globals_m
    use ParallelInfo_class
#ifdef HAVE_OMP
    use omp_lib
#endif
#ifdef HAVE_CUDA
    use cuda_m 
#endif
    implicit none

    public      :: GBFMMLaplacian3D

    private

    

    type, extends(Laplacian3D) :: GBFMMLaplacian3D
        type(Laplacian3D),       allocatable :: laplacian_operators(:)
        type(GBFMMParallelInfo), pointer     :: parallel_info
        type(SerialInfo),        allocatable :: parallel_infos(:)
        type(Grid3D),            allocatable :: grids(:)
        integer                              :: level
    contains
#ifdef HAVE_CUDA
        procedure           :: cuda_init                    => &
                    GBFMMLaplacian3D_cuda_init
        procedure           :: cuda_destroy                 => &
                    GBFMMLaplacian3D_cuda_destroy
#endif
        procedure           :: destroy                      => &
                    GBFMMLaplacian3D_destroy
        procedure           :: operate_on                   => &
                    GBFMMLaplacian3D_apply
        procedure, private  :: operate_cube                 => &
                    GBFMMLaplacian3D_operate_cube
                    
    end type


    interface GBFMMLaplacian3D
        module procedure :: GBFMMLaplacian3D_init
    end interface 
    

contains

    function GBFMMLaplacian3D_init(parallel_info) result(new)
        class(ParallelInfo),   intent(in), target  :: parallel_info
        type(GBFMMLaplacian3D)                     :: new
        integer                                    :: domain(2)
        integer                                    :: ibox, ilevel, box_count
        
        select type(parallel_info)
            type is (GBFMMParallelInfo)
                new%parallel_info => parallel_info
                
                domain = parallel_info%get_domain(parallel_info%integration_level)
                allocate(new%parallel_infos(domain(1):domain(2)))
                allocate(new%grids(domain(1):domain(2)))
                allocate(new%laplacian_operators(domain(1):domain(2)))
                new%result_type = F3D_TYPE_CUSP
                new%gridin => parallel_info%get_grid()
                new%gridout => parallel_info%get_grid()
                ibox = domain(1)

                ! initialize new laplacian operator for each box
                do ibox=domain(1), domain(2)
                    new%grids(ibox) = new%gridin%get_subgrid(parallel_info%get_box_cell_index_limits(ibox, &
                        level = parallel_info%integration_level))
                    new%parallel_infos(ibox) = SerialInfo(new%grids(ibox))
                    new%laplacian_operators(ibox) = Laplacian3D(new%parallel_infos(ibox), &
                                                                new%parallel_infos(ibox), &
                                                                suboperator = .TRUE.)
                end do
                new%result_parallelization_info => parallel_info
            class default
                print *, "ERROR: input parallel info type is not GBFMMParallelInfo,&
                         &like it should be"
                stop
        end select
        
    end function

#ifdef HAVE_CUDA
    subroutine GBFMMLaplacian3D_cuda_init(self)
        class(GBFMMLaplacian3D), intent(inout)  :: self
        integer                                 :: ibox, domain(2)
        ! get starting and ending indices of this processors domain
        domain = self%parallel_info%get_domain(self%parallel_info%integration_level)
        if (allocated(self%laplacian_operators)) then
            do ibox = domain(1), domain(2)
                call self%laplacian_operators(ibox)%cuda_init()
            end do
        end if

        ! init the input & output cuda cubes
        self%input_cuda_cube = CudaCube(self%gridin%get_shape())
        self%output_cuda_cube = CudaCube(self%gridout%get_shape())

        self%cuda_inited = .TRUE.
    end subroutine
    
    subroutine GBFMMLaplacian3D_cuda_destroy(self)
        class(GBFMMLaplacian3D), intent(inout)  :: self
        integer                                 :: ibox, domain(2)
        ! get starting and ending indices of this processors domain
        domain = self%parallel_info%get_domain(self%parallel_info%integration_level)
        if (allocated(self%laplacian_operators)) then
            do ibox = domain(1), domain(2)
                call self%laplacian_operators(ibox)%cuda_destroy()
            end do
        end if

        ! destroy the input & output cuda cubes
        if (self%cuda_inited) then
            call self%input_cuda_cube%destroy()
            call self%output_cuda_cube%destroy()
        
            self%cuda_inited = .FALSE.
        end if
    end subroutine
#endif

    subroutine GBFMMLaplacian3D_apply(self, func, new, cell_limits, only_cube)
        class(GBFMMLaplacian3D)                 :: self
        class(Function3D), intent(in)           :: func 
        !> The result function containing everything needed for energy calculation.
        !! The actual class type will be 
        class(Function3D), allocatable, intent(inout), target :: new 
        integer, intent(in), optional           :: cell_limits(2, 3)
        !> if only cube is taken into account. There is no use for this
        !! argument in this class, but is only introduced because of the function interface
        !! in Function3D.
        logical, intent(in), optional           :: only_cube
        integer                                 :: thread_id
        type(Bubbles)                           :: result_bubbles
        
        
        if (.not. allocated(new)) then
            allocate(Function3D :: new)
            call new%init_explicit(func%parallelization_info, type = self%get_result_type(), &
                        taylor_series_order = func%taylor_order)
            new%bubbles = Bubbles(func%bubbles, k=-2)
        else
            new%cube = 0.0d0
            new%bubbles = 0.0d0
        end if
#ifdef HAVE_OMP
        call omp_set_nested(.TRUE.);  
        !$OMP PARALLEL PRIVATE(thread_id) NUM_THREADS(2)
            thread_id = omp_get_thread_num()
            if (thread_id == 1) then
                call self%operate_cube(func, new)
            else 
                call omp_set_num_threads(OMP_GLOBAL_THREAD_COUNT-1)
                call self%laplacian_bubbles_sub(func%bubbles, new%bubbles)
                if (thread_id == 0) then
                    call new%parallelization_info%communicate_bubbles(new%bubbles)
                end if
            end if
        !$OMP END PARALLEL
        call omp_set_num_threads(OMP_GLOBAL_THREAD_COUNT)
#else
        call self%operate_cube(func, new)
        call self%laplacian_bubbles_sub(func%bubbles, new%bubbles)
        call new%parallelization_info%communicate_bubbles(new%bubbles)
#endif
        call new%communicate_cube_borders(sum_result = .TRUE.) 
    end subroutine

    subroutine GBFMMLaplacian3D_operate_cube(self, func, new)
        class(GBFMMLaplacian3D)                 :: self
        class(Function3D), intent(in)           :: func 
        !> The result function containing everything needed for energy calculation.
        !! The actual class type will be 
        class(Function3D), allocatable, intent(inout), target :: new 

        integer                                 :: domain(2)
        integer                                 :: ibox, box_cell_index_limits(2, 3), cube_ranges(2, 3)
        class(Function3D), allocatable          :: box_result
#ifdef HAVE_CUDA
        type(CUDACube)                          :: input_box_cube, output_box_cube, tmp1, tmp2
        integer                                 :: dims(3, 2)
        ! register the input and output cubes, upload the input cube and set the output cube to zero 
        call self%input_cuda_cube%set_host(func%cube)
        call self%output_cuda_cube%set_host(new%cube)
        call self%input_cuda_cube%upload()
        call self%output_cuda_cube%set_to_zero()
        call CUDASync_all()
#endif

        ! go through all boxes belonging to the domain of this processor
        ! and calculate individual contributions of each box
        domain = self%parallel_info%get_domain(self%parallel_info%integration_level)

        do ibox=domain(1), domain(2)
            ! get the limits of the box as cell indices and cartesian coordinates
            box_cell_index_limits = self%parallel_info%get_box_cell_index_limits &
                (ibox, level = self%parallel_info%integration_level)
            cube_ranges = self%parallel_info%get_cube_ranges(box_cell_index_limits)

#ifdef HAVE_CUDA

            if (ibox == domain(1)) then
                dims = self%laplacian_operators(ibox)%get_dims()
                tmp1 = CUDACube([dims(1,2),dims(2,1), dims(3,1)])
                tmp2 = CUDACube([dims(1,2),dims(2,2), dims(3,1)])
            end if

            ! get the subcubes corresponding to nearfield and box area
            input_box_cube = self%input_cuda_cube%get_subcube(cube_ranges)
            output_box_cube = self%output_cuda_cube%get_subcube(cube_ranges)
            call output_box_cube%set_to_zero()

            ! and finally do the operation
            call self%laplacian_operators(ibox)%transform_cuda_cube( &
                                          input_box_cube, output_box_cube, &
                                          self%laplacian_operators(ibox)%cuda_blas, &
                                          self%laplacian_operators(ibox)%cuda_fx, &
                                          self%laplacian_operators(ibox)%cuda_fy, &
                                          self%laplacian_operators(ibox)%cuda_fz, &
                                          tmp1, tmp2)

            ! do the clean up
            call CUDASync_all()
            call input_box_cube%destroy()
            call output_box_cube%destroy()
#else

            ! operate the partial laplacian operator on the corresponding box
            ! in the input function
            call self%laplacian_operators(ibox)%operate_on(func, box_result,  &
                    cell_limits = box_cell_index_limits, only_cube = .TRUE.)
            new%cube(cube_ranges(1, X_) : cube_ranges(2, X_),        &
                                 cube_ranges(1, Y_) : cube_ranges(2, Y_),        &
                                 cube_ranges(1, Z_) : cube_ranges(2, Z_) ) =     &
                new%cube(cube_ranges(1, X_) : cube_ranges(2, X_),    &
                                     cube_ranges(1, Y_) : cube_ranges(2, Y_),    &
                                     cube_ranges(1, Z_) : cube_ranges(2, Z_) )   & 
                + box_result%cube  
            call box_result%destroy()
            deallocate(box_result)     
#endif       
        end do

#ifdef HAVE_CUDA
        call tmp1%destroy()
        call tmp2%destroy()
        call self%output_cuda_cube%download()
        call CUDASync_all()
#endif
         
    end subroutine


    subroutine GBFMMLaplacian3D_destroy(self)
        class(GBFMMLaplacian3D), intent(inout) :: self
        integer                                :: domain(2)
        integer                                :: ibox
        
#ifdef HAVE_CUDA
        call self%cuda_destroy()
#endif
        domain = self%parallel_info%get_domain(self%parallel_info%integration_level)
        if (allocated(self%laplacian_operators)) then
            do ibox = domain(1), domain(2)
#ifdef HAVE_CUDA
                call self%grids(ibox)%cuda_destroy()
                call self%parallel_infos(ibox)%destroy()
#endif
                call self%laplacian_operators(ibox)%destroy() 
                call self%grids(ibox)%destroy()
                call self%parallel_infos(ibox)%destroy()
            end do
            deallocate(self%laplacian_operators)
            deallocate(self%grids)
            deallocate(self%parallel_infos)
        end if 
        nullify(self%parallel_info)
        
        
    end subroutine
end module
