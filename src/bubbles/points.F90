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

module Points_class
    use globals_m
    use ISO_FORTRAN_ENV
    use Grid_class
    use CudaObject_class

    implicit none

    public :: PointCoordinates, Points

!--------------------------------------------------------------------!
!        PointCoordinates Definition                                 !
!--------------------------------------------------------------------!

    type, extends(CudaObject) :: PointCoordinates
        !> The x, y, z -coordinates of the points
        real(REAL64), allocatable                :: coordinates(:, :)
        !> The parent points coordinates
        type(PointCoordinates), private, pointer :: parent_point_coordinates
        !> The ranges of the points in the parent point coordinates
        integer, private,   allocatable          :: parent_ranges(:, :)
    contains
        procedure, public :: is_subset_of => PointCoordinates_is_subset_of
        procedure, public :: are_equal    => PointCoordinates_are_equal
        procedure, public :: get_point_coordinates_within_grid &
                                          => PointCoordinates_get_point_coordinates_within_grid
        procedure, public :: destroy      => PointCoordinates_destroy
#ifdef HAVE_CUDA
        procedure, public :: cuda_init    => PointCoordinates_cuda_init
        procedure, public :: cuda_destroy => PointCoordinates_cuda_destroy
#endif
    end type

    interface PointCoordinates
       module procedure PointCoordinates_init
    end interface

#ifdef HAVE_CUDA
    interface
        type(C_PTR) function PointCoordinates_init_cuda(coordinates, number_of_points, stream_container) bind(C)
            use ISO_C_BINDING
            real(C_DOUBLE)        :: coordinates(*)
            integer(C_INT), value :: number_of_points       
            type(C_PTR), value    :: stream_container
        end function
    end interface

    interface
        type(C_PTR) function PointCoordinates_init_host_points_cuda(point_coordinates) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: point_coordinates
        end function
    end interface

    interface
        subroutine PointCoordinates_destroy_host_points_cuda(host_points) bind(C)
            use ISO_C_BINDING
            real(C_DOUBLE)        :: host_points(*)
        end subroutine
    end interface

    interface
        subroutine PointCoordinates_destroy_cuda(point_coordinates) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: point_coordinates
        end subroutine
    end interface
#endif

!--------------------------------------------------------------------!
!        Points Definition                                           !
!--------------------------------------------------------------------!
! The Points-class represents a set of values at the coordinates     !
! specified in point_coordinates                                     !
!--------------------------------------------------------------------! 

    type, extends(CudaObject) :: Points
        type(PointCoordinates), pointer          :: point_coordinates
        real(REAL64), pointer                    :: values(:)
#ifndef HAVE_CUDA
        !> The values of the points
        real(REAL64), private, allocatable       :: values_data(:)
#endif
    contains
        procedure, public :: add_in_place      => Points_add_in_place
        procedure, public :: product_in_place_REAL64 => Points_product_in_place_REAL64
        procedure, public :: overwrite         => Points_overwrite
        procedure, public :: destroy           => Points_destroy
#ifdef HAVE_CUDA
        procedure, public :: cuda_init         => Points_cuda_init
        procedure, public :: cuda_destroy      => Points_cuda_destroy
        procedure, public :: cuda_download     => Points_cuda_download
        procedure, public :: cuda_set_to_zero  => Points_cuda_set_to_zero
#endif
        ! Binary operators
        procedure, private :: Points_add
        generic, public :: operator(+) => Points_add

        procedure, private :: Points_product
        generic, public :: operator(*) => Points_product
    end type

    interface Points
       module procedure Points_init
       module procedure Points_init_copy
    end interface

#ifdef HAVE_CUDA
    interface
        type(C_PTR) function Points_init_cuda(point_coordinates, stream_container) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: point_coordinates
            type(C_PTR), value    :: stream_container
        end function
    end interface

    interface
        subroutine Points_destroy_cuda(points) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: points
        end subroutine
    end interface

    interface
        subroutine Points_set_to_zero_cuda(points) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: points
        end subroutine
    end interface

    interface
        subroutine Points_download_cuda(points, host_values) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: points
            real(C_DOUBLE)        :: host_values(*)
        end subroutine
    end interface
#endif
contains
!--------------------------------------------------------------------!
!        PointCoordinates Implementations                            !
!--------------------------------------------------------------------!

    function PointCoordinates_init(coordinates) result(new)
        real(REAL64), intent(in) :: coordinates(:, :)
        type(PointCoordinates)   :: new
        
        new%coordinates = coordinates
#ifdef HAVE_CUDA
        call new%cuda_init()
#endif
        nullify(new%parent_point_coordinates)
    end function

    subroutine PointCoordinates_destroy(self)
        class(PointCoordinates), intent(inout)   :: self

        if (allocated(self%coordinates))   deallocate(self%coordinates)
        if (allocated(self%parent_ranges)) deallocate(self%parent_ranges)
        if (associated(self%parent_point_coordinates)) nullify(self%parent_point_coordinates)
#ifdef HAVE_CUDA
        call self%cuda_destroy()
#endif
    end subroutine

#ifdef HAVE_CUDA
    subroutine PointCoordinates_cuda_init(self)
        class(PointCoordinates), intent(inout)   :: self

        if (.not. allocated(self%cuda_interface)) then
            self%cuda_interface = pointcoordinates_init_cuda &
                (self%coordinates, size(self%coordinates, 2), stream_container)
        end if

    end subroutine

    subroutine PointCoordinates_cuda_destroy(self)
        class(PointCoordinates), intent(inout)   :: self

        if (allocated(self%cuda_interface)) call pointcoordinates_destroy_cuda(self%cuda_interface)
    end subroutine
#endif

    function PointCoordinates_get_point_coordinates_within_grid(self, grid) result(new)
        class(PointCoordinates), target,  intent(in)   :: self
        type(Grid3D),                     intent(in)   :: grid
        type(PointCoordinates)                         :: new
        integer                                        :: i, start_index, range_counter, point_counter
        real(REAL64)                                   :: new_coordinates(3, size(self%coordinates, 2))
        integer                                        :: parent_ranges(2, size(self%coordinates, 2))

        range_counter = 0
        point_counter = 0
        start_index = -1
        do i = 1, size(self%coordinates, 2)
            if (grid%is_point_within_range(self%coordinates(:, i))) then
                if (start_index == -1) then
                    start_index = i
                end if
                point_counter = point_counter + 1
                new_coordinates(:, point_counter) = self%coordinates(:, i)
            else
                if (start_index /= -1) then
                    range_counter = range_counter + 1 
                    parent_ranges(1, range_counter) = start_index
                    parent_ranges(2, range_counter) = i-1
                    start_index = -1
                end if
            end if
        end do

        ! if the last point is within range, it closes the last range
        i = size(self%coordinates, 2)
        if (grid%is_point_within_range(self%coordinates(:, i))) then
            range_counter = range_counter + 1 
            parent_ranges(1, range_counter) = start_index
            parent_ranges(2, range_counter) = i
        end if 

        new = PointCoordinates(new_coordinates(:, :point_counter))
        new%parent_point_coordinates => self
        new%parent_ranges = parent_ranges(:, :range_counter)
        
    end function

    function PointCoordinates_is_subset_of(self, point_coordinates) result(res)
        class(PointCoordinates),  intent(in)   :: self
        class(PointCoordinates),  intent(in)   :: point_coordinates
        logical                                :: res

        res = associated(self%parent_point_coordinates)
        if (res) &
            res = self%parent_point_coordinates%are_equal(point_coordinates)
    end function

    function PointCoordinates_are_equal(self, point_coordinates) result(res)
        class(PointCoordinates),  intent(in)   :: self
        class(PointCoordinates),  intent(in)   :: point_coordinates
        logical                                :: res

        if (      associated(self%parent_point_coordinates) &
            .and. associated(point_coordinates%parent_point_coordinates) &
            .and. allocated(self%parent_ranges)  &
            .and. allocated(point_coordinates%parent_ranges)) then
            res =       associated(self%parent_point_coordinates, point_coordinates%parent_point_coordinates)
            if (res) res = all(shape(self%parent_ranges) == shape(point_coordinates%parent_ranges))
            if (res) res = all(self%parent_ranges == point_coordinates%parent_ranges)
            if (res) res = all(self%coordinates == point_coordinates%coordinates)
        else
            res = allocated(self%coordinates) .and. allocated(point_coordinates%coordinates)
            if (res) res = all(shape(self%parent_ranges) == shape(point_coordinates%parent_ranges))
            if (res) res = all(self%coordinates == point_coordinates%coordinates)
        end if
    end function

!--------------------------------------------------------------------!
!        Points Implementations                                      !
!--------------------------------------------------------------------!

    function Points_init(point_coordinates) result(new)
        type(PointCoordinates), intent(in), target :: point_coordinates
        type(Points),                       target :: new
#ifdef HAVE_CUDA
        type(C_PTR)                                :: c_pointer

        c_pointer = PointCoordinates_init_host_points_cuda(point_coordinates%cuda_interface)

        call c_f_pointer(c_pointer, new%values, [size(point_coordinates%coordinates, 2)])
#else
        allocate(new%values_data(size(point_coordinates%coordinates, 2)))
        new%values => new%values_data
#endif
        new%point_coordinates => point_coordinates
        
    end function

    function Points_init_copy(points2, copy_content) result(new)
        type(Points), target,   intent(in)         :: points2
        type(Points)                               :: new
        logical,      optional, intent(in)         :: copy_content
        logical                                    :: copy_content_

        copy_content_ = .FALSE.
        if (present(copy_content)) copy_content_ = copy_content
        new = Points(points2%point_coordinates)
        if (copy_content_) then
            new%values(:) = points2%values(:)
        end if
    end function

    subroutine Points_destroy(self) 
        class(Points), intent(inout)   :: self
#ifdef HAVE_CUDA
        call PointCoordinates_destroy_host_points_cuda(self%values)
#else
        if (allocated(self%values_data)) deallocate(self%values_data)
#endif
        nullify(self%point_coordinates)
        nullify(self%values)
    end subroutine

    function Points_add(self, points2) result(new)
        class(Points), intent(in)       :: self
        class(Points), intent(in)       :: points2
        type(Points)                    :: new
        logical                         :: coordinates_are_equal 
        integer                         :: i, first, last

        coordinates_are_equal = associated(self%point_coordinates, points2%point_coordinates)
        if (.not. coordinates_are_equal) &
            coordinates_are_equal = self%point_coordinates%are_equal(points2%point_coordinates)

        ! check if the points have the same coordinates, if they do, we can safely add them together
        if (coordinates_are_equal) then
            new = Points(self)
            new%values(:) = self%values(:) + points2%values(:)
        else if (self%point_coordinates%is_subset_of(points2%point_coordinates)) then
            new = Points(points2, copy_content = .TRUE.)
            first = 1
            do i = 1, size(self%point_coordinates%parent_ranges, 2)
                last =   self%point_coordinates%parent_ranges(2, i) &
                       - self%point_coordinates%parent_ranges(1, i) + first
                new%values(self%point_coordinates%parent_ranges(1, i) : &
                           self%point_coordinates%parent_ranges(2, i)) &
                     =    new%values(self%point_coordinates%parent_ranges(1, i) : &
                                     self%point_coordinates%parent_ranges(2, i)) &
                       +  self%values(first:last)
                first = last + 1
            end do
        else if (points2%point_coordinates%is_subset_of(self%point_coordinates)) then
            new = Points(self, copy_content = .TRUE.)
            first = 1
            do i = 1, size(points2%point_coordinates%parent_ranges, 2)
                last =   points2%point_coordinates%parent_ranges(2, i) &
                       - points2%point_coordinates%parent_ranges(1, i) + first
                new%values(points2%point_coordinates%parent_ranges(1, i) : &
                           points2%point_coordinates%parent_ranges(2, i)) &
                     =    new%values(points2%point_coordinates%parent_ranges(1, i) : &
                                     points2%point_coordinates%parent_ranges(2, i)) &
                       +  points2%values(first:last)
                first = last + 1
            end do
        else
            print *, "ERROR: The given 'Points' cannot be added together (@Points_add)"
            stop
        end if
    end function

    function Points_product(self, points2) result(new)
        class(Points), intent(in)       :: self
        class(Points), intent(in)       :: points2
        type(Points)                    :: new
        logical                         :: coordinates_are_equal 
        integer                         :: i, first, last

        coordinates_are_equal = associated(self%point_coordinates, points2%point_coordinates)
        if (.not. coordinates_are_equal) &
            coordinates_are_equal = self%point_coordinates%are_equal(points2%point_coordinates)

        ! check if the points have the same coordinates, if they do, we can safely add them together
        if (coordinates_are_equal) then
            new = Points(self)
            new%values(:) = self%values(:) * points2%values(:)
        else if (self%point_coordinates%is_subset_of(points2%point_coordinates)) then
            new = Points(points2, copy_content = .TRUE.)
            first = 1
            do i = 1, size(self%point_coordinates%parent_ranges, 2)
                last =   self%point_coordinates%parent_ranges(2, i) &
                       - self%point_coordinates%parent_ranges(1, i) + first
                new%values(self%point_coordinates%parent_ranges(1, i) : &
                           self%point_coordinates%parent_ranges(2, i)) &
                     =    new%values(self%point_coordinates%parent_ranges(1, i) : &
                                     self%point_coordinates%parent_ranges(2, i)) &
                       *  self%values(first:last)
                first = last + 1
            end do
        else if (points2%point_coordinates%is_subset_of(self%point_coordinates)) then
            new = Points(self, copy_content = .TRUE.)
            first = 1
            do i = 1, size(points2%point_coordinates%parent_ranges, 2)
                last =   points2%point_coordinates%parent_ranges(2, i) &
                       - points2%point_coordinates%parent_ranges(1, i) + first
                new%values(points2%point_coordinates%parent_ranges(1, i) : &
                           points2%point_coordinates%parent_ranges(2, i)) &
                     =    new%values(points2%point_coordinates%parent_ranges(1, i) : &
                                     points2%point_coordinates%parent_ranges(2, i)) &
                       *  points2%values(first:last)
                first = last + 1
            end do
        else
            print *, "ERROR: The given 'Points' cannot be added together (@Points_product)"
            stop
        end if
    end function
    
    subroutine Points_product_in_place_REAL64(self, factor)
        class(Points), intent(inout)       :: self
        real(REAL64),  intent(in)          :: factor

        self%values = self%values * factor
    end subroutine

    subroutine Points_add_in_place(self, points2)
        class(Points), intent(inout)       :: self
        type(Points), target,   intent(in) :: points2
        logical                            :: coordinates_are_equal 
        integer                         :: i, first, last

        coordinates_are_equal = associated(self%point_coordinates, points2%point_coordinates)
        if (.not. coordinates_are_equal) &
            coordinates_are_equal = self%point_coordinates%are_equal(points2%point_coordinates)

        ! check if the points have the same coordinates, if they do, we can safely add them together
        if (coordinates_are_equal) then
            self%values(:) = self%values(:) + points2%values(:)
        else if (points2%point_coordinates%is_subset_of(self%point_coordinates)) then
            first = 1
            do i = 1, size(points2%point_coordinates%parent_ranges, 2)
                last =   points2%point_coordinates%parent_ranges(2, i) &
                       - points2%point_coordinates%parent_ranges(1, i) + first
                self%values(points2%point_coordinates%parent_ranges(1, i) : &
                            points2%point_coordinates%parent_ranges(2, i)) &
                     =    self%values(points2%point_coordinates%parent_ranges(1, i) : &
                                      points2%point_coordinates%parent_ranges(2, i)) &
                       +  points2%values(first:last)
                first = last + 1
            end do
        else
            print *, "ERROR: The given 'Points' cannot be added together in place (@Points_add_in_place)"
            stop
        end if
    end subroutine

    subroutine Points_overwrite(self, points2)
        class(Points), intent(inout)       :: self
        type(Points), target,   intent(in) :: points2
        logical                            :: coordinates_are_equal 
        integer                         :: i, first, last
        coordinates_are_equal = associated(self%point_coordinates, points2%point_coordinates)
        if (.not. coordinates_are_equal) &
            coordinates_are_equal = self%point_coordinates%are_equal(points2%point_coordinates)

        ! check if the points have the same coordinates, if they do, we can safely add them together
        if (coordinates_are_equal) then
            self%values(:) = points2%values(:)
        else if (points2%point_coordinates%is_subset_of(self%point_coordinates)) then
            first = 1
            do i = 1, size(points2%point_coordinates%parent_ranges, 2)
                last =   points2%point_coordinates%parent_ranges(2, i) &
                       - points2%point_coordinates%parent_ranges(1, i) + first
                self%values(points2%point_coordinates%parent_ranges(1, i) : &
                            points2%point_coordinates%parent_ranges(2, i)) &
                     =  points2%values(first:last)
                first = last + 1
            end do
        else
            print *, "ERROR: The given ('Points') 'self' cannot be owerwritten with given 'points2' (@Points_overwrite)"
            stop
        end if
    end subroutine

#ifdef HAVE_CUDA
    subroutine Points_cuda_init(self)
        class(Points), intent(inout)   :: self

        if (.not. allocated(self%cuda_interface)) then
            self%cuda_interface = Points_init_cuda(self%point_coordinates%get_cuda_interface(), stream_container)
        end if
    end subroutine

    subroutine Points_cuda_destroy(self)
        class(Points), intent(inout)   :: self

        if (self%is_cuda_inited()) then
            call Points_destroy_cuda(self%cuda_interface)
        end if
    end subroutine

    subroutine Points_cuda_set_to_zero(self)
        class(Points), intent(inout)   :: self

        if (self%is_cuda_inited()) then
            call Points_set_to_zero_cuda(self%cuda_interface)
        else
            print *, "ERROR: Trying to set cuda 'Points' to zero without initialized cuda interface (@Points_cuda_set_to_zero)"
            stop
        end if
    end subroutine

    subroutine Points_cuda_download(self) 
        class(Points), intent(inout)       :: self

        if (self%is_cuda_inited()) then
            call Points_download_cuda(self%cuda_interface, self%values)
        else
            print *, "ERROR: Trying to download 'Points' from cuda without initialized cuda interface (@Points_cuda_download)"
            stop
        end if
    end subroutine
#endif
end module