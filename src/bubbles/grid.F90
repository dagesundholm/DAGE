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
!> @file grid.F90
!! Defines Grid1D and Grid3D objects.

!> Grids: Moulds for functions.
!!
!! A grid is an object that contains information about the axis/axes along
!! which a function can be evaluated. The primary attributes of an
!! one-dimensional grid are its length \math\Delta q = q_{max} - q_{min}
!! \math, the number of cells \math N_{cell}\math in which this length is
!! partitioned and the step size \math h_i\math in each cell *i*. A
!! three-dimensional grid is nothing more than a collection of three
!! individual one-dimensional grids.
!!
!! The concept of cells is fundamental to `libdage` because of the way
!! functions are represented with Lagrange interpolation
!! polynomials (LIPs). Since a set of finite-order polynomials can only
!! interpolate a function in a finite interval and since increasing the
!! order of the polynomials hampers accuracy due to Runge's phenomenon, it
!! is necessary to resort to relatively low-order LIPs that only work in
!! a subinterval of the total grid, thus forming a basis in a *cell*.
!!
!! A one-dimensional grid can be constructed explicitly by specifying the
!! minimum value of the grid \math q_{min}\math, the number of cells, the order of
!! the LIPs, \math(N_{lip}-1)\math, and the step size in each cell. Each
!! cell has then \math N_{lip}\math points distributed so that the last point
!! of a cell is the first point of the adjacent cell except for the last
!! cell, yielding \math N_{cell}(N_{lip}-1) + 1\math points in total. The
!! length of the grid is then given by \math \Delta q = \sum_i^{N_{cell}}
!! h_i(N_{lip}-1)\math. The last point \math q_{max}\math can be calculated
!! from \math q_{min} \math and \math \Delta q\math. A non-equidistant
!! grid can be easily constructed by varying \math h_i\math.
!!
!! Two alternative ways to construct an equidistant grid are provided.
!! Firstly, one can specify the range \math[q_{min}, q_{max} ]\math, the
!! order of the LIPs and a threshold value for the step \math h_{max}\math.
!! The resulting grid will have \math N_{cell} = \lceil (\Delta
!! q/h_{max}(N_{lip}-1)\rceil\math cells. The actual step size is
!! calculated from \math \Delta q/(N_{cell}(N_{lip}-1))\math so that it
!! never exceeds \math h_{max}\math. For exact control of the grid
!! parameters, \math h_{max} \math should divide \math \Delta
!! q/(N_{lip}-1)\math evenly so that the ceiling operation does not round
!! up its argument. Secondly, one can specify the range, the number of grid
!! points and optionally the \math N_{lip}\math parameter. The last
!! method exists primarily for such cases when the order of LIPs cannot
!! be specified explicitly in which case a default value is used.
!!
!! ### Usage
!!
!! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.f90}
!!  type(Grid3D), pointer :: grid_eq, grid_neq
!!
!!  ! Some data
!!  qmin = [0.0, 0.0, 0.0]
!!  ranges = reshape([xmin, xmax, ymin, ymax, zmin, zmax], [2,3])
!!  nlip = 7 ! 6th order polynomials
!!  ncell = [3, 3, 3]
!!  stepx = [0.1, 0.2, 0.01]
!!  stepy = [0.1, 0.2, 0.01]
!!  stepz = [0.01, 0.2, 0.01]
!!
!!  ! An equidistant grid with a step no greater than 0.01
!!  grid_eq => Grid3D(ranges, nlip, stepmax=0.01_REAL64)
!!
!!  ! A non-equidistant grid
!!  grid_neq => Grid3D(qmin, ncell, nlip,&
!!                     stepx, stepy, stepz)
!!
!!  ! Shape of grid_eq
!!  print *, grid_eq%get_shape()
!!  ! Accessing the x axis
!!  print *, maxval(grid_neq%axis(1)%get_coord()), grid_neq%axis(1)%qmax
!! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
module Grid_class
    use globals_m
    use LIPBasis_class
    use PBC_class
    use xmatrix_m
    use CudaObject_class
    use GaussQuad_class
#ifdef HAVE_CUDA
    use ISO_C_BINDING
#endif
    implicit none

    public :: Grid1D, Grid1DPointer
    public :: Grid3D, file_exists

    public :: resume_Grid1Ds, store_Grid1Ds
    public :: NLIP_DEFAULT


    ! xwh, get the index of axis grids
    ! in a cubic cell
    public :: axis_index

    ! Convenience function for setting up transformation matrices
    public :: alloc_transformation_matrices

    ! Old functionality via module procedures
    public :: set_grid_slice

    private

    !> LIPs are of (7-1)th order if not specified in a constructor
    integer, parameter  :: NLIP_DEFAULT = 7

    !> A 1D grid.
    type, extends(CudaObject) :: Grid1D
        private
        !> Minimum value of a grid.
        real(REAL64)            :: qmin
        !> Maximum value of a grid.
        real(REAL64)            :: qmax
        !> Number of grid points in the whole grid
        integer                 :: ndim
        !> Number of cells.
        integer                 :: ncell
        !> Grid points (ndim)
        real(REAL64), allocatable :: grid(:)
!        !> Step sizes, for each cell. Ie, the distance between two points in a
!        ! cell, as a function of the cell index (ncell)
!        real(REAL64), allocatable :: cellh(:)
        !> within 'lip' and 'lower_lip' a scaled (typically enlarged) version of
        !a cell is used. cell_scales is the ratio of the real length of a cell and
        !the length used in 'lip'.  Alternatively, (and under the assumption
        !that in 'lip' the distance between two grid points is 1) cell scale is
        !the length of one cell devided by nlip-1 which is the distance between
        !two grid point if the grid is equidistant.
        real(real64), allocatable :: cell_scales(:)
        !> Length of the cells (ncell)
        real(REAL64), allocatable :: cellen(:)
        !> Coordinates where each cell starts (ncell)
        real(REAL64), allocatable :: celld(:)
        !> Total length of the grid in its coordinates
        real(REAL64)              :: delta
        !> Order of LIPs is `nlip-1`.
        integer(int32)            :: nlip
        !> A LIP of order `nlip-1`.
        type(LIPBasis), public    :: lip
        !> A LIP of order `nlip-2`.
        type(LIPBasis), public    :: lower_lip
        !> choice of grid type within each cell. 1 is equidistant,
        !                                        2 is Gauss-Lobatto
        integer(int32)            :: grid_type
        !> If grid is equidistant (ie, all cells have the same size)
        logical, public           :: equidistant_cells
        !> The radius used in initiation of radial grid
        real(REAL64), public      :: cutoff_radius
        !> The radius used in initiation of radial grid
        real(REAL64), public      :: charge, r_max

        ! MPI

        !> Number of slices in a grid.
        integer                 :: slice_n = 1
        !> The rank of the processor.
        integer                 :: slice_rank = 1
        !> Starting index of a slice.
        integer                 :: slice_start = 1
        !> Total size without slicing.
        integer                 :: slice_tot = 1
    contains


        ! Accessors

        ! LIP information
        procedure :: get_lip => Grid1D_get_lip
        procedure :: get_nlip => Grid1D_get_nlip

        ! Grid information
        procedure :: get_qmin  => Grid1D_get_qmin
        procedure :: get_qmax  => Grid1D_get_qmax
        procedure :: get_shape => Grid1D_get_dim
        procedure :: get_coord => Grid1D_get_coord ! returns pointer
        procedure :: get_coordinates => Grid1D_get_coordinates ! returns arrays
        procedure :: get_delta => Grid1D_get_delta
        procedure :: get_ints  => Grid1D_get_ints
        procedure :: is_equidistant => Grid1D_is_equidistant ! ie cells have equal lengths
        procedure :: is_subgrid_of => Grid1D_is_subgrid_of
        procedure :: get_grid_type => Grid1D_get_grid_type

        ! Cell specific
        procedure :: get_ncell => Grid1D_get_ncell
        procedure :: get_cell_scales => Grid1D_get_cell_scales
        procedure :: get_cell_scale  => Grid1D_get_cell_scale
        procedure :: get_cell_starts => Grid1D_get_celld
        procedure :: get_cell_center => Grid1D_get_cell_center
        procedure :: get_cell_deltas => Grid1D_get_cellen

        ! Get part of the grid and initialize a new grid from
        procedure :: get_subgrid => Grid1D_get_subgrid

        ! Workers
        procedure :: get_icell => Grid1D_get_icell
        procedure :: get_icell_equidistant => Grid1D_get_icell_equidistant
        procedure :: get_icell_spherical => Grid1D_get_icell_spherical
        procedure :: coordinate_to_grid_point_coordinate => &
                         Grid1D_coordinate_to_grid_point_coordinate
        procedure, private :: Grid1D_x2cell, Grid1D_x2cell_icell
        generic   :: x2cell    => Grid1D_x2cell, Grid1D_x2cell_icell

        ! Similarity / Equality checks
        procedure :: is_similar => Grid1D_is_similar

        procedure :: is_point_within_range => Grid1D_is_point_within_range

        ! MPI Accessors
        procedure :: get_slice_start
        procedure :: get_slice_rank
        procedure :: get_slice_tot

        ! xwh, \int r^2\chi(r) dr
        procedure :: r2int => Grid1D_r2int

        ! xwh, \int chi(r) dr, nlip for each cell
        procedure :: ints  => Grid1D_ints

        ! xwh, \chi'(x) for all lips in the cell
        ! x is arbitrary
        procedure :: lip_dev => Grid1D_lip_dev

        ! xwh, \chi'(x) for all lips in the cell
        ! x: all the grids in a cell
        procedure :: lip_dev_m => Grid1D_lip_dev_m

        ! xwh, dr/ds = h_i^{-1}, i = grid index
        ! for adjoint points, average the steps
        !
        procedure :: dr_ds => Grid1D_dr_ds


#ifdef HAVE_CUDA
        procedure :: cuda_init => Grid1D_cuda_init
        procedure :: cuda_upload => Grid1D_cuda_upload
        procedure :: cuda_download => Grid1D_cuda_download
        procedure :: cuda_destroy => Grid1D_cuda_destroy
#endif

        ! Destructor
        procedure :: destroy => Grid1D_destroy
    end type

#ifdef HAVE_CUDA

    interface
        type(C_PTR) function Grid1D_init_cuda(ncell, nlip, r_max, h, d, grid_type, gridpoints, lip, derivative_lip, &
                                              lower_derivative_lip, base_integrals, streamContainer) bind(C)
            use ISO_C_BINDING
            integer(C_INT), value :: ncell
            integer(C_INT), value :: nlip
            real(C_DOUBLE), value :: r_max
            real(C_DOUBLE)        :: h(*)
            real(C_DOUBLE)        :: d(*)
            integer(C_INT), value :: grid_type
            real(C_DOUBLE)        :: gridpoints(*)
            real(C_DOUBLE)        :: lip(*)
            real(C_DOUBLE)        :: derivative_lip(*)
            real(C_DOUBLE)        :: lower_derivative_lip(*)
            real(C_DOUBLE)        :: base_integrals(*)
            type(C_PTR),    value :: streamContainer
        end function
    end interface

    interface
        subroutine Grid1D_upload_cuda(grid) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: grid
        end subroutine
    end interface

    interface
        subroutine Grid1D_destroy_cuda(grid) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: grid
        end subroutine
    end interface

#endif

    type :: Grid1DPointer
        type(Grid1D), pointer :: p
    end type

    !> A 3D grid.
    type, extends(CudaObject) :: Grid3D
        private
        !> An array of 1D grids.
        type(Grid1D), public, allocatable         :: axis(:)
        !> Order of LIPs is `nlip-1`.
        integer(int32)               :: nlip
        !> Optimal number of maxlevel in GBFMM
        integer, public              :: gbfmm_maxlevel
        !> A Lagrange interpolation polynomial
        type(LIPBasis), public       :: lip
        !> A LIP of order `nlip-2`.
        type(LIPBasis), public       :: lower_lip
        !> choice of grid type within each cell. 1 is equidistant,
        !                                        2 is Gauss-Lobatto
        integer(int32)               :: grid_type
        !> Periodic boundary conditions.
        type(PBC)                    :: pbc

        !> Grid identifier
        integer                      :: id

        ! type(Grid1D), pointer   :: x, y, z  ! Workaround for ifc
    contains
        ! Accessors

        ! LIP information
        procedure :: get_nlip => Grid3D_get_nlip
        procedure :: get_lip => Grid3D_get_lip

        ! Grid information
        procedure :: get_qmin => Grid3D_get_qmin
        procedure :: get_qmax => Grid3D_get_qmax
        procedure :: get_range => Grid3D_get_range
        procedure :: get_shape => Grid3D_get_dim
        procedure :: get_delta => Grid3D_get_delta
        procedure :: get_pbc => Grid3D_get_pbc
        procedure :: get_id => Grid3D_get_id
        procedure :: get_grid_type => Grid3D_get_grid_type
        procedure :: is_equidistant => Grid3D_is_equidistant
        procedure :: is_subgrid_of => Grid3D_is_subgrid_of
        procedure :: dump          => Grid3D_dump
        procedure :: cartesian_coordinates_to_cell_coordinates => &
                         Grid3D_cartesian_coordinates_to_cell_coordinates
        ! xwh, the index of the first grid in the cell
        procedure :: cartesian_coordinates_to_cube_coordinates => &
                         Grid3D_cartesian_coordinates_to_cube_coordinates

        ! Get part of the grid and initialize a new grid from
        procedure :: get_subgrid => Grid3D_get_subgrid

        ! Cell specific
        procedure :: get_icell => Grid3D_get_icell
        procedure :: get_ncell => Grid3D_get_ncell
        procedure :: coordinates_to_grid_point_coordinates => &
                         Grid3D_coordinates_to_grid_point_coordinates
        procedure :: get_cell_center => Grid3D_get_cell_center

        ! \int chi_i(x) chi_j(y) chi_k(z) d\vec{r}
        procedure :: ints => Grid3D_ints


        ! Similarity / Equality checks
        procedure :: is_equal => Grid3D_is_equal
        procedure :: is_similar => Grid3D_is_similar


        procedure :: is_point_within_range => Grid3D_is_point_within_range
        procedure :: get_all_grid_points => Grid3D_get_all_grid_points

        ! CUDA procedures
#ifdef HAVE_CUDA
        procedure :: cuda_init    => Grid3D_cuda_init
        procedure :: cuda_destroy => Grid3D_cuda_destroy
#endif

        ! Destructor
        procedure :: destroy => Grid3D_grid_destroy
    end type

!------------ Grid3D CUDA Interfaces ----------------- !

#ifdef HAVE_CUDA
    interface
        type(C_PTR) function Grid3D_init_cuda(axis, streamContainer) bind(C)
            use ISO_C_BINDING
            type(C_PTR)           :: axis(*)
            type(C_PTR), value    :: streamContainer
        end function
    end interface

    interface
        type(C_PTR) function Grid3D_cuda_init_ibox(shape, grid_points_x, grid_points_y,&
                                                grid_points_z, ibox, evaluator) bind(C)
            use ISO_C_BINDING
            integer(C_INT)        :: shape(3)
            real(C_DOUBLE)        :: grid_points_x(*)
            real(C_DOUBLE)        :: grid_points_y(*)
            real(C_DOUBLE)        :: grid_points_z(*)
            integer(C_INT), value :: ibox
            type(C_PTR), value    :: evaluator
        end function
    end interface

    interface
        subroutine Grid3D_destroy_cuda(grid) bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: grid
        end subroutine
    end interface

#endif

    !> Constructor interface for 3D grids.
    interface Grid3D
        module procedure :: Grid3D_init_step
        module procedure :: Grid3D_init_equidistant
        module procedure :: Grid3D_init_equidistant_ncell
        module procedure :: Grid3D_init_header
        module procedure :: Grid3D_init_spheres
        module procedure :: Grid3D_init_read
    end interface

    !> Constructor interface for 1D grids.
    interface Grid1D
        module procedure :: Grid1D_init_step
        module procedure :: Grid1D_init_equidistant
        module procedure :: Grid1D_init_radial
        module procedure :: Grid1D_init_spheres
    end interface

contains


! %%%%%%%%%%%%%%%%%%%%%%%% Grid1D constructors %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    !> Constructs a 1D grid from explicit parameters.
    !!
    !!
    function Grid1D_init_step(qmin, ncell, nlip, step, grid_type) result(new)
        real(REAL64),     intent(in)            :: qmin
        integer,          intent(in)            :: ncell
        integer,          intent(in)            :: nlip
        real(REAL64),     intent(in)            :: step(ncell)
        integer(int32),   intent(in)            :: grid_type

        real(real64), allocatable               :: grid_template(:)
        real(real64), allocatable               :: glgrid(:)

        type(Grid1D)                            :: new

        integer                                 :: icell, ipoint, k

        new%nlip  = nlip
        new%ncell = ncell
        new%ndim  = new%ncell*(new%nlip-1) + 1

        allocate(new%cell_scales(new%ncell), source=0.0_REAL64)
        allocate(new%cellen(new%ncell),      source=0.0_REAL64)
        allocate(new%celld(new%ncell),       source=0.0_REAL64)
        allocate(new%grid(new%ndim),         source=0.0_REAL64)

        new%qmin = qmin
        new%cell_scales = step

        new%lip=LIPBasisInit(new%nlip, grid_type)
        select case (grid_type)
            case (1)
                new%lower_lip=LIPBasisInit(new%nlip-1, gridtype=1) ! equidistant, but smaller
            case (2)
                new%lower_lip=LIPBasisInit(new%nlip-1, gridtype=3) ! special type
            case default
        end select
        new%grid_type = grid_type

        ! write(*,*) 'lip ', new%lip%first, new%lip%last
        ! write(*,*) 'lower_lip ', new%lower_lip%first, new%lower_lip%last

        allocate(grid_template(nlip-1))
        select case (grid_type)
            case (1) ! equidistant
                do k=1, nlip-1
                    grid_template(k) = k
                end do

            case (2) ! gauss lobatto
                glgrid = gauss_lobatto_grid(new%nlip, 0.d0, real(nlip-1, 8) ) ! where the second nlip is a length
                write(*,*) 'glgrid: ', glgrid
                do k=1, nlip-1
                    grid_template(k) = glgrid(k+1)
                end do
                deallocate(glgrid)

            case default
        end select

        write(*,*) 'grid_template: ', grid_template

        ! Grid points
        ipoint=1
        new%grid(1)=new%qmin

        ! go through all cells
        do icell=1, new%ncell
            new%celld(icell) = new%grid(ipoint)
            ! each point corresponds to one lagrange interpolation polynomial
            do k=1,new%nlip-1
                ipoint=ipoint+1
                ! grid point is the starting point of the cell plus the step size
                ! times order number of the step in the cells
                new%grid(ipoint)=new%celld(icell)+grid_template(k)*new%cell_scales(icell)
            end do
            new%cellen(icell) = (new%nlip-1)*new%cell_scales(icell)
        end do

        new%cutoff_radius = (new%grid(new%ndim))
        new%qmax = new%grid(new%ndim)
        new%delta = new%qmax - new%qmin

        ! print *, new%grid

        !write(ppbuf, *) "Creating a 1D grid",&
        !    C_NEW_LINE,&
        !    "ndim = ", new%ndim, C_NEW_LINE,&
        !    "nrange = ", new%qmin, new%qmax, C_NEW_LINE,&
        !    "nlip = ", nlip, C_NEW_LINE,&
        !    "ncell = ", new%ncell

        !call pdebug(ppbuf, 1)

        deallocate(grid_template)
    end function

    !> Constructs a 1D grid in a given range and with a fixed step.
    !! Here, the term 'equidistant' refers to the sizes of the cells, not the grid
    !! therein.
    function Grid1D_init_equidistant(ranges, nlip, stepmax, grid_type, step) result(new)
        real(REAL64),           intent(in)     :: ranges(2)
        integer,                intent(in)     :: nlip
        real(REAL64),           intent(in)     :: stepmax
        integer(int32),         intent(in)     :: grid_type
        real(REAL64), optional, intent(inout)  :: step

        type(Grid1D)                           :: new

        integer                                :: ncell
        real(REAL64), allocatable              :: cell_scales(:)
        real(REAL64)                           :: s

        !call pdebug("grid_init_equidistant_1d()",1)

        ncell = ceiling((ranges(2)-ranges(1)) / (stepmax*(nlip-1)))
        allocate(cell_scales(ncell))
        ! Adjust step size so that the range is exactly reproduced
        s = (ranges(2)-ranges(1)) / (ncell*(nlip-1))
        cell_scales = s
        if (present(step)) step = s

        ! call the init_step constructor
        new = Grid1D(ranges(1), ncell, nlip, cell_scales, grid_type)
        new%equidistant_cells = .TRUE.

        deallocate(cell_scales)
    end function

    !> Generate radial grid according to some weird formula :-D
    function Grid1D_init_radial(z, n0, nlip, r_max, grid_type) result(new)
        real(REAL64),  intent(in)   :: z
        integer,       intent(in)   :: n0
        integer,       intent(in)   :: nlip
        real(REAL64),  intent(in)   :: r_max
        integer(int32),intent(in)   :: grid_type

        type(Grid1D)                :: new

        real(REAL64)                :: const
        integer                     :: i, ncell
        real(REAL64), allocatable   :: cell_scales(:)
        real(REAL64)                :: x0,x1,dx

        ! SETUP THE SPHERICAL GRID
        ncell=int(n0*z**0.25d0,INT32)
        allocate(cell_scales(ncell))

        ! Initialize grid, and grab step vector to initialize by hand
        x0=0.d0
        const=8.d0*z**(-1.5d0)
        dx=r_max/ncell
        do i=1,ncell
            x1=const/((r_max+const)/(i*dx) - 1.d0)
            ! Calculate grid point space within cell i
            cell_scales(i)=(x1-x0)/(nlip-1)
            x0=x1
        end do
        new = Grid1D(0.0_REAL64, ncell, nlip, cell_scales, grid_type)
        new%equidistant_cells = .FALSE.
        new%charge = z
        new%r_max = r_max

        deallocate(cell_scales)
    end function

    function Grid1D_init_spheres(centers, radii, step, nlip, grid_type) result(new)
        real(REAL64),   intent(in)         :: centers(:)
        real(REAL64),   intent(in)         :: radii(:)
        real(REAL64),   intent(in)         :: step
        integer,        intent(in)         :: nlip
        integer(int32), intent(in)         :: grid_type

        type(Grid1D)                     :: new

        integer                          :: ncell
        real(REAL64)                     :: box(2)
        real(REAL64)                     :: dx
        real(REAL64)                     :: cell_len
        integer                          :: i

        ! Find minimum box to fit the spheres
        box(1)= minval(centers-radii)
        box(2)= maxval(centers+radii)
        ! Find the minimum number of cells required to fit the box
        cell_len=step*(nlip-1)
        ncell = ceiling((box(2)-box(1))/cell_len)
        ! Expand the box symmetrically so the length it fits a whole number of
        ! cells exactly
        dx = ncell * cell_len - (box(2)-box(1))
        box(1)=box(1)-0.5d0*dx
        box(2)=box(2)+0.5d0*dx

        !call pdebug("Generating default grid from spheres",1)
        !call pdebug("        x0              xmax       ncell      step",1)
        !write(ppbuf,'(2f16.10,i6,f16.10)') box, ncell, step
        !call pdebug(ppbuf,1)

        ! Create equidistant grid
        new=Grid1D(box, nlip, step, grid_type)
    end function

! %%%%%%%%%%%%%%%%%%%%%%%% Grid3D constructors %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    !> Constructs a 3D grid from explicit parameters.
    !!
    !! @param qmin      The starting position of the grid.
    !! @param ncell     The number of cells in along each axis.
    !! @param nlip      The number of LIPs in each cell.
    !! @param stepx     Steps in each cell along the x axis.
    !! @param stepy     Steps in each cell along the y axis.
    !! @param stepz     Steps in each cell along the z axis.
    !! @param pbc       Passed if the system has PBC's.
    function Grid3D_init_step(qmin, ncell, nlip,&
                              stepx, stepy, stepz, grid_type, pbc_string) result(new)
        real(REAL64), intent(in)                :: qmin(3)
        integer, intent(in)                     :: ncell(3)
        integer, intent(in)                     :: nlip
        real(REAL64), intent(in)                :: stepx(ncell(1)),&
                                                   stepy(ncell(2)),&
                                                   stepz(ncell(3))
        integer, intent(in)                     :: grid_type
        character(len=*), intent(in), optional  :: pbc_string

        type(Grid3D)                            :: new

        new%nlip = nlip
        new%lip=LIPBasisInit(new%nlip, grid_type)
        select case (grid_type)
            case (1)
                new%lower_lip=LIPBasisInit(new%nlip-1, gridtype=1) ! equidistant, but smaller
            case (2)
                new%lower_lip=LIPBasisInit(new%nlip-1, gridtype=3) ! special type
            case default
        end select
        new%grid_type = grid_type
        allocate(new%axis(3))
        new%axis(1) = Grid1D(qmin(1), ncell(1), nlip, stepx, grid_type)
        new%axis(1)%equidistant_cells = .TRUE.
        new%axis(2) = Grid1D(qmin(2), ncell(2), nlip, stepy, grid_type)
        new%axis(2)%equidistant_cells = .TRUE.
        new%axis(3) = Grid1D(qmin(3), ncell(3), nlip, stepz, grid_type)
        new%axis(3)%equidistant_cells = .TRUE.
        new%gbfmm_maxlevel = 2

        if (present(pbc_string)) then
            new%pbc=PBC(pbc_string)
        else
            new%pbc=PBC("")
        endif
        new%id=next_grid_id()
    end function

    !> Constructs an equidistant 3D grid in a given range with a fixed step.
    !!
    !! @param ranges    A (2,3)-array specifying the min and max values of
    !!                  coordinates.
    !! @param nlip      Number of LIPs in a cell.
    !! @param stepmax   Threshold value for the step size.
    !! @param step      Passed if the callers wants to store the true step sizes.
    !! @param pbc       Passed if the system has PBC's
    !!
    !! To reproduce the ranges and the step exactly, the parameters should
    !! satisfy `ncell == (ranges(2,i)-ranges(1,i)) / stepmax*(nlip-1)` for
    !! some integer `ncell` for each axis `i`.
    function Grid3D_init_equidistant(ranges, nlip, stepmax, grid_type, step, pbc_string) &
                                                                result(new)
        real(REAL64),               intent(in)      :: ranges(2,3)
        integer,                    intent(in)      :: nlip
        real(REAL64),               intent(in)      :: stepmax
        integer, intent(in)                         :: grid_type
        real(REAL64),     optional, intent(out)     :: step(3)
        character(len=*), optional, intent(in)      :: pbc_string

        type(Grid3D)                                :: new

        real(REAL64)                                :: s(3)
        integer                                     :: ncell
        integer                                     :: i
        real(REAL64), allocatable                   :: cell_scales(:)

        new%nlip = nlip
        new%lip=LIPBasisInit(new%nlip, grid_type)
        select case (grid_type)
            case (1)
                new%lower_lip=LIPBasisInit(new%nlip-1, gridtype=1) ! equidistant, but smaller
            case (2)
                new%lower_lip=LIPBasisInit(new%nlip-1, gridtype=3) ! special type
            case default
        end select
        new%grid_type = grid_type
        allocate(new%axis(3))
        do i=1, 3
            ncell = ceiling((ranges(2,i)-ranges(1,i)) / (stepmax*(nlip-1)))
            allocate(cell_scales(ncell))
            ! Adjust step size so that the range is exactly reproduced
            s(i) = (ranges(2,i)-ranges(1,i)) / (ncell*(nlip-1))
            cell_scales = s(i)

            new%axis(i) = Grid1D(ranges(1,i), ncell, nlip, cell_scales, grid_type)
            deallocate(cell_scales)
        enddo

        if (present(step)) step = s

        if (present(pbc_string)) then
            new%pbc=PBC(pbc_string)
        else
            new%pbc=PBC("")
        endif
        new%id=next_grid_id()
    end function

    !> Constructs an equidistant 3D grid in a given range with a fixed shape.
    function Grid3D_init_equidistant_ncell(ranges, nlip, ncell, grid_type, step, pbc_string) &
                                                                result(new)
        real(REAL64),               intent(in)      :: ranges(2,3)
        integer,                    intent(in)      :: nlip
        integer,                    intent(in)      :: ncell(3)
        integer, intent(in)                         :: grid_type
        real(REAL64),     optional, intent(out)     :: step(3)
        character(len=*), optional, intent(in)      :: pbc_string

        type(Grid3D)                                :: new

        real(REAL64)                                :: s(3)
        integer                                     :: i
        real(REAL64), allocatable                   :: cell_scales(:)

        new%nlip = nlip
        new%lip=LIPBasisInit(new%nlip, grid_type)
        select case (grid_type)
            case (1)
                new%lower_lip=LIPBasisInit(new%nlip-1, gridtype=1) ! equidistant, but smaller
            case (2)
                new%lower_lip=LIPBasisInit(new%nlip-1, gridtype=3) ! special type
            case default
        end select
        new%grid_type = grid_type
        allocate(new%axis(3))

        do i=X_, Z_
            allocate(cell_scales(ncell(i)))
            ! Compute step size so that the range is exactly reproduced
            s(i) = (ranges(2,i)-ranges(1,i)) / (ncell(i)*(nlip-1))
            cell_scales = s(i)
            new%axis(i) = Grid1D(ranges(1,i), ncell(i), nlip, cell_scales, grid_type)
            deallocate(cell_scales)
        enddo

        if (present(step)) step = s

        if (present(pbc_string)) then
            new%pbc=PBC(pbc_string)
        else
            new%pbc=PBC("")
        endif
        new%id=next_grid_id()
    end function

    !> Constructs an equidistant 3D grid in a given range with a fixed number of grid points.
    !!
    !! Unlike with other grid constructors, the `nlip` parameter is optional.
    !! When absent, the default value `NLIP_DEFAULT` is used.
    !!
    !! This routine is mainly intended for IO routines. The caller
    !! must ensure that the `npoints` array satisfies
    !!
    !! `npoints(i) == ncell*(NLIP_DEFAULT-1) + 1`
    !!
    !! for some integer `ncell`.
    function Grid3D_init_header(ranges, npoints, grid_type, nlip, pbc_string) result(new)
        real(REAL64), intent(in)                :: ranges(2,3)
        integer, intent(in)                     :: npoints(3)
        integer, intent(in)                     :: grid_type
        integer, intent(in), optional           :: nlip
        character(len=*), intent(in), optional  :: pbc_string

        type(Grid3D)                            :: new

        integer                                 :: i
        integer                                 :: ncell
        real(REAL64), allocatable               :: cell_scales(:)

        if (present(nlip)) then
            new%nlip = nlip
        else
            new%nlip = NLIP_DEFAULT
        end if
        new%lip=LIPBasisInit(new%nlip, grid_type)
        select case (grid_type)
            case (1)
                new%lower_lip=LIPBasisInit(new%nlip-1, gridtype=1) ! equidistant, but smaller
            case (2)
                new%lower_lip=LIPBasisInit(new%nlip-1, gridtype=3) ! special type
            case default
        end select
        new%grid_type = grid_type
        allocate(new%axis(3))
        do i=1, 3
            ncell = (npoints(i) - 1) / (new%nlip - 1)
            allocate(cell_scales(ncell))
            cell_scales = (ranges(2,i) - ranges(1,i)) / (npoints(i) - 1)
            new%axis(i) = Grid1D(ranges(1,i), ncell, new%nlip, cell_scales, grid_type)
            deallocate(cell_scales)
        enddo

        if (present(pbc_string)) then
            new%pbc=PBC(pbc_string)
        else
            new%pbc=PBC("")
        endif
        new%id=next_grid_id()
    end function



    function Grid3D_init_spheres(centers, radii, step, nlip, grid_type, gbfmm) result(new)
        real(REAL64),      intent(in) :: centers(:,:)
        real(REAL64),      intent(in) :: radii(:)
        real(REAL64),      intent(in) :: step
        !> Number of Lagrange interpolation polynomials per cell
        integer,           intent(in) :: nlip
        integer, intent(in)           :: grid_type
        logical, optional, intent(in) :: gbfmm
        type(Grid3D)                  :: new

        ! grid limits as coordinates with input sphere centers and radii
        real(REAL64)                  :: grid_limits(3, 2), step_sizes(3)
        real(REAL64), allocatable     :: stepsize_x(:), stepsize_y(:), stepsize_z(:)
        integer                       :: box_size(3), maxlevel, step_count(3), &
                                         cell_count(3), cell_modulus
        logical                       :: within_limits(3)

        integer                       :: idim


        ! determine the optimal step size (close to input 'step') that has
        ! box size between 1-300 grid points and is divisible with the
        ! corresponding level
        if (present(gbfmm) .and. gbfmm) then
            do idim = X_, Z_
                grid_limits(idim, 1)= minval(centers(idim, :)-radii(:))
                grid_limits(idim, 2)= maxval(centers(idim, :)+radii(:))
            end do
            grid_limits(:, 1) = minval(grid_limits(:, 1))
            grid_limits(:, 2) = maxval(grid_limits(:, 2))

            ! calculate number of steps for each dimension
            step_count(:) = floor((grid_limits(:, 2) - grid_limits(:, 1)) / step)
            ! get the maxlevel, where all the box sizes are below 300
            maxlevel = 2
            do
                if ( all( [ ( step_count(idim) / (2**maxlevel) < 300, idim = 1, 3) ] ) ) then
                    exit
                else
                    maxlevel = maxlevel + 1
                end if
            end do
            ! initialize all step sizes to the initial 'step'
            step_sizes = step
            ! get the axis step sizes in a way that each axis has cells divisible by 2**maxlevel
            do idim = X_, Z_
                cell_count(idim) = ceiling((grid_limits(idim, 2) - grid_limits(idim, 1)) / &
                                      (((nlip-1) * step_sizes(idim))))

                ! check how many cells away we are from fulfilling the  condition mentioned above
                cell_modulus = mod(cell_count(idim), 2**maxlevel)
                ! adjust the stepsize so that the condition is exactly fulfilled
                if (cell_modulus <= 0.5d0* 2**maxlevel) then
                    cell_count(idim) = (cell_count(idim) - cell_modulus)
                else
                    cell_count(idim) = (cell_count(idim) + 2**maxlevel - cell_modulus)
                end if

                step_sizes(idim) = (grid_limits(idim, 2) - grid_limits(idim, 1)) / (cell_count(idim) * (nlip-1))
                ! calculate the new cell counts per box
                cell_count(idim) = nint((grid_limits(idim, 2) - grid_limits(idim, 1)) &
                                         / ((nlip-1)*step_sizes(idim)))
            end do

             ! allocate the axis stepsizes and give them values
            allocate(stepsize_x(cell_count(X_)))
            allocate(stepsize_y(cell_count(Y_)))
            allocate(stepsize_z(cell_count(Z_)))
            stepsize_x = step_sizes(X_)
            stepsize_y = step_sizes(Y_)
            stepsize_z = step_sizes(Z_)


            ! init grid via Grid3D_init_step constructor
            new = Grid3D(grid_limits(:, 1), cell_count, nlip, stepsize_x, stepsize_y, stepsize_z, grid_type)
            new%gbfmm_maxlevel = maxlevel

            ! deallocate memory
            deallocate(stepsize_x)
            deallocate(stepsize_y)
            deallocate(stepsize_z)

        else
            new%nlip = nlip
            new%lip=LIPBasisInit(new%nlip, grid_type)
            select case (grid_type)
                case (1)
                    new%lower_lip=LIPBasisInit(new%nlip-1, gridtype=1) ! equidistant, but smaller
                case (2)
                    new%lower_lip=LIPBasisInit(new%nlip-1, gridtype=3) ! special type
                case default
            end select
            new%grid_type = grid_type
            new%id=next_grid_id()
            new%pbc=PBC("")

            allocate(new%axis(3))
            new%axis(X_) = Grid1D(centers(X_,:), radii, step, nlip, grid_type)
            new%axis(Y_) = Grid1D(centers(Y_,:), radii, step, nlip, grid_type)
            new%axis(Z_) = Grid1D(centers(Z_,:), radii, step, nlip, grid_type)
        end if
    end function

#ifdef HAVE_CUDA

    subroutine Grid3D_cuda_init(self)
        class(Grid3D), intent(inout)            :: self
        type(C_PTR)                             :: cuda_axis(3)

        if (.not. self%is_cuda_inited()) then
            cuda_axis(X_) = self%axis(X_)%get_cuda_interface()
            cuda_axis(Y_) = self%axis(Y_)%get_cuda_interface()
            cuda_axis(Z_) = self%axis(Z_)%get_cuda_interface()

            self%cuda_interface = Grid3D_init_cuda(cuda_axis, stream_container)
        end if
    end subroutine

    subroutine Grid3D_cuda_destroy(self)
        class(Grid3D), intent(inout) :: self

        if (allocated(self%cuda_interface)) then
            call Grid3D_destroy_cuda(self%cuda_interface)
            deallocate(self%cuda_interface)
        end if
    end subroutine

    subroutine Grid1D_cuda_init(self)
        class(Grid1D), intent(inout) :: self
        type(REAL64_2D), allocatable :: coeffs(:), coeffs_lower(:)
        integer                      :: i

        if (.not. allocated(self%cuda_interface)) then
            coeffs=self%lip%coeffs(1)
            coeffs_lower = self%lower_lip%coeffs(1)
            self%cuda_interface = Grid1D_init_cuda(self%get_ncell(), self%get_nlip(), &
                                  self%cutoff_radius, self%get_cell_scales(), &
                                  self%get_cell_starts(), self%get_grid_type(), &
                                  self%get_coord(), transpose(coeffs(1)%p), &
                                  transpose(coeffs(2)%p), transpose(coeffs_lower(2)%p), &
                                  self%lip%integrals(), stream_container)
            ! the upload was added to the c++ init method
            !call self%cuda_upload()
            do i = 1, size(coeffs)
                deallocate(coeffs(i)%p)
            end do
            deallocate(coeffs)
        end if
    end subroutine

    subroutine Grid1D_cuda_destroy(self)
        class(Grid1D), intent(inout) :: self
        if (allocated(self%cuda_interface)) then
            call Grid1D_destroy_cuda(self%cuda_interface)
            deallocate(self%cuda_interface)
        end if
    end subroutine

    subroutine Grid1D_cuda_upload(self, cuda_interface)
        class(Grid1D),               intent(in) :: self
        type(C_PTR),       optional, intent(in) :: cuda_interface
        if (present(cuda_interface)) then
            call Grid1D_upload_cuda(cuda_interface)
        else if (allocated(self%cuda_interface)) then
            call Grid1D_upload_cuda(self%cuda_interface)
        end if
    end subroutine

    subroutine Grid1D_cuda_download(self)
        class(Grid1D), intent(inout) :: self

        ! do nothing, as there is no such method... (Is this useful anywhere, anytime, remains to be seen)
    end subroutine
#endif
    ! Destructors

    pure subroutine Grid1D_destroy(self)
        class(Grid1D), intent(inout) :: self

        !call pdebug("Destroying a 1D grid!", 1)

        if (allocated(self%grid)) deallocate(self%grid)
        if (allocated(self%cell_scales)) deallocate(self%cell_scales)
        if (allocated(self%cellen)) deallocate(self%cellen)
        if (allocated(self%celld)) deallocate(self%celld)
#ifdef HAVE_CUDA
        if (allocated(self%cuda_interface)) deallocate(self%cuda_interface)
#endif
!        if (allocated(self%lip)) deallocate(self%lip)

        self%ncell= 0
        self%ndim = 0
        self%nlip = 0
    end subroutine

    subroutine Grid3D_grid_destroy(self)
        class(Grid3D), intent(inout) :: self
        integer :: i

!        if (allocated(self%lip)) deallocate(self%lip)
        if (allocated(self%axis)) then
            call self%axis(X_)%destroy()
            call self%axis(Y_)%destroy()
            call self%axis(Z_)%destroy()
            deallocate(self%axis)
        end if
#ifdef HAVE_CUDA
        call self%cuda_destroy()
#endif
    end subroutine

    ! Readers and writers



    !> Inits a new Grid3D object by reading a binary file at folder/filename
    function Grid3D_init_read(folder, filename) result(new)
        character(len=*), intent(in)    :: folder
        character(len=*), intent(in)    :: filename
        logical                         :: file_exists_
        type(Grid3D)                    :: new
        integer                         :: nlip
        integer                         :: nlip_3d(3)
        integer                         :: ncell(3)
        real(REAL64)                    :: qmin(3)
        real(REAL64), allocatable       :: stepx(:), stepy(:), stepz(:)
        integer                         :: grid_type

        file_exists_ = file_exists(folder, filename)
        if (file_exists_) then
            call pdebug("Loading grid from `"//trim(folder)//"/"//trim(filename), 1)

            open(unit=READ_FD, file=trim(folder)//"/"//trim(filename), access='stream')

            read(READ_FD) nlip
            read(READ_FD) grid_type

            read(READ_FD) ncell(1)
            read(READ_FD) nlip_3d(1)
            read(READ_FD) qmin(1)
            allocate(stepx(ncell(1)))
            read(READ_FD) stepx

            read(READ_FD) ncell(2)
            read(READ_FD) nlip_3d(2)
            read(READ_FD) qmin(2)
            allocate(stepy(ncell(2)))
            read(READ_FD) stepy

            read(READ_FD) ncell(3)
            read(READ_FD) nlip_3d(3)
            read(READ_FD) qmin(3)
            allocate(stepz(ncell(3)))
            read(READ_FD) stepz

            if (all(nlip_3d == nlip_3d(1))) then
                new = Grid3D(qmin, ncell, nlip_3d(1), stepx, stepy, stepz, grid_type)
            else
                call perror("Corrupted function! Each axis should have &
                &the same amount of LIPs.")
            endif
            close(unit=READ_FD)
            deallocate(stepx, stepy, stepz)
        end if
        return
    end function

    !> Saves Grid3D in binary format to disk
    subroutine Grid3D_dump(self, folder, gridname)
        class(Grid3D),    intent(in)            :: self
        !> Folder for the binary file
        character(len=*), intent(in)            :: folder
        !> Basename for the binary file
        character(len=*), intent(in)            :: gridname
        integer                                 :: i

        open(unit=WRITE_FD, file=trim(folder)//'/'//trim(gridname)//'.g3d', access='stream')

        write(WRITE_FD) self%nlip
        write(WRITE_FD) self%grid_type
        do i = 1, 3
            write(WRITE_FD) self%axis(i)%ncell
            write(WRITE_FD) self%axis(i)%nlip
            write(WRITE_FD) self%axis(i)%qmin
            write(WRITE_FD) self%axis(i)%cell_scales
        end do
        close(unit=WRITE_FD)
    end subroutine

    ! Accessors

    function Grid3D_get_lip(self) result(lip)
        class(Grid3D), target :: self
        type(LIPBasis), pointer :: lip

        lip=>self%lip
    end function

    function Grid1D_get_lip(self) result(lip)
        class(Grid1D), target :: self
        type(LIPBasis), pointer :: lip

        lip=>self%lip
    end function

    function Grid1D_get_celld(self) result(celld)
        class(Grid1D), target   :: self
        real(REAL64), pointer   :: celld(:)

        celld => self%celld
    end function

    pure function Grid3D_get_dim(self) result(dims)
        class(Grid3D), intent(in) :: self
        integer :: dims(3)
        integer :: i

        dims = [( self%axis(i)%ndim, i=1, 3)]
    end function

    pure function Grid1D_get_dim(self) result(ndim)
        class(Grid1D), intent(in) :: self
        integer       :: ndim
        ndim = self%ndim
    end function

    function Grid3D_get_range(self) result(ranges)
        class(Grid3D) :: self
        real(REAL64) :: ranges(2,3)
        integer :: i

        ! Minimum values for x, y and z
        ranges(1,:) = [( self%axis(i)%qmin, i=1,3 )]
        ! Maximum values for x, y and z
        ranges(2,:) = [( self%axis(i)%qmax, i=1,3 )]
    end function

    function Grid1D_get_cell_scales(self) result(cell_scales)
        class(Grid1D), target :: self
        real(REAL64), pointer :: cell_scales(:)

        cell_scales => self%cell_scales
    end function

    pure function Grid1D_get_cell_scale(self, icell) result(cell_scale)
        class(Grid1D), intent(in) :: self
        integer,       intent(in) :: icell
        real(REAL64)              :: cell_scale

        cell_scale = self%cell_scales(icell)
    end function

    !> Return the index of the cell to which `x` belongs.
    !> @todo Support for PBC.
    !! This will require implementing PBC's for 1D grids, effectively meaning
    !! a total redoing of the PBC module.
    pure function Grid1D_get_icell(self, x) result(icell)
        class(Grid1D),intent(in) :: self
        real(REAL64),intent(in) :: x
        integer(INT32) :: icell, maxi, mini
        logical :: found

        if (x > self%qmax .or. x < self%qmin) then ! outside
            icell=0
        else if (self%celld(self%ncell) <= x ) then ! x in last cell
            icell=self%ncell
        else
            icell = self%ncell/2
            maxi = self%ncell-1
            mini = 1
            found = .false.
            do while (.not. found)
                if (x >= self%celld(icell+1)) then
                    if (x < self%celld(icell+2)) then
                        found=.true.
                        icell=icell+1
                    else
                        mini=icell+2
                        icell=(mini+maxi)/2
                    endif
                else ! (x < self%celld(icell+1))
                    if (x >= self%celld(icell)) then
                        found=.true.
                    else
                        maxi=icell-1
                        icell=(mini+maxi)/2
                    endif
                endif
            enddo
        endif
    end function

    pure function Grid1D_get_icell_spherical(self, x) result(icell)
        class(Grid1D),intent(in) :: self
        real(REAL64), intent(in) :: x
        integer                  :: icell
        real(REAL64)             :: const, dx

        const=8.d0*self%charge**(-1.5d0)
        dx=self%r_max/self%ncell
        ! end
        icell = int(x * (self%r_max + const) / ((const + x)*dx)) +1
    end function

    pure function Grid1D_get_icell_equidistant(self, x) result(icell)
        class(Grid1D),intent(in) :: self
        real(REAL64), intent(in) :: x
        integer                  :: icell

        icell = x * self%ncell / (self%qmax-self%qmin)
    end function

    !> Return the indices of the cells to which `pos` belongs.
    pure function Grid3D_get_icell(self, pos) result(icell)
        class(Grid3D), intent(in) :: self
        real(REAL64), intent(in) :: pos(3)
        integer(INT32) :: icell(3)

        icell(X_) = Grid1D_get_icell(self%axis(X_),pos(X_))
        icell(Y_) = Grid1D_get_icell(self%axis(Y_),pos(Y_))
        icell(Z_) = Grid1D_get_icell(self%axis(Z_),pos(Z_))
    end function

    !> Return the center of cell at 'ix', 'iy', 'iz'
    pure function Grid3D_get_cell_center(self, ix, iy, iz) result(center)
        class(Grid3D), intent(in)  :: self
        integer,       intent(in)  :: ix
        integer,       intent(in)  :: iy
        integer,       intent(in)  :: iz
        real(REAL64)               :: center(3)

        center(X_) = self%axis(X_)%get_cell_center(ix)
        center(Y_) = self%axis(Y_)%get_cell_center(iy)
        center(Z_) = self%axis(Z_)%get_cell_center(iz)
    end function

    pure function Grid3D_get_ncell(self, crd) result(r)
        class(Grid3D), intent(in)  :: self
        integer, intent(in) :: crd
        integer :: r

        r=self%axis(crd)%ncell
    end function

    pure function Grid1D_get_ncell(self) result(r)
        class(Grid1D), intent(in)  :: self
        integer :: r

        r=self%ncell
    end function

    pure function Grid3D_get_nlip(self) result(r)
        class(Grid3D), intent(in)  :: self
        integer(INT32)             :: r
        r=self%nlip
    end function

    pure function Grid1D_get_nlip(self) result(r)
        class(Grid1D), intent(in)  :: self
        integer(INT32) :: r

        r=self%nlip
    end function

    pure function Grid1D_get_delta(self) result(delta)
        class(Grid1D), intent(in)  :: self
        real(REAL64) :: delta

        delta=self%delta
    end function

    pure function Grid1D_get_grid_type(self) result(grid_type)
        class(Grid1D), intent(in) :: self
        integer(int32)            :: grid_type
        grid_type=self%grid_type
    end function

    pure function Grid1D_get_cell_center(self, icell) result(center)
        class(Grid1D), intent(in)  :: self
        integer,    intent(in) :: icell
        real(REAL64)           :: center
        real(REAL64)           :: cell_starts(self%ncell), cell_deltas(self%ncell)

        center = self%celld(icell) + self%cellen(icell) * 0.5
    end function

    pure function Grid3D_get_delta(self) result(delta)
        class(Grid3D), intent(in)  :: self
        integer(INT32)  :: i
        real(REAL64)    :: delta(3)
        delta = [( self%axis(i)%get_delta(), i=1,3)]
    end function

    pure function Grid3D_get_qmin(self) result(qmin)
        class(Grid3D), intent(in)  :: self
        integer(INT32)  :: i
        real(REAL64)    :: qmin(3)
        qmin = [( self%axis(i)%get_qmin(), i=1,3)]
    end function

    pure function Grid3D_get_qmax(self) result(qmax)
        class(Grid3D), intent(in)  :: self
        integer(INT32)  :: i
        real(REAL64)    :: qmax(3)
        qmax = [( self%axis(i)%get_qmax(), i=1,3)]
    end function

    function Grid3D_get_all_grid_points(self) result(grid_points)
        class(Grid3D), intent(in)  :: self
        real(REAL64)                         :: grid_points(3, self%axis(X_)%get_shape()* &
                                                               self%axis(Y_)%get_shape()* &
                                                               self%axis(Z_)%get_shape())

        real(REAL64), dimension(:), pointer  :: xgrid, ygrid, zgrid
        integer                              :: iz, nz, iy, ny, nx

        ! get points on x, y and z axis
        xgrid=>self%axis(X_)%get_coord()
        ygrid=>self%axis(Y_)%get_coord()
        zgrid=>self%axis(Z_)%get_coord()
print *, 'xyzgrid', xgrid, ygrid, zgrid ! remove, lnw 

        nx = self%axis(X_)%get_shape()
        ny = self%axis(Y_)%get_shape()
        nz = self%axis(Z_)%get_shape()
print *, 'xyzshape', nx, ny, nz ! remove, lnw

        do iz=1,nz
            grid_points(Z_, (iz - 1) * (nx * ny) + 1: iz * (nx * ny)) = zgrid(iz)
            forall (iy = 1: ny)
                grid_points(Y_, (iz - 1) * (nx * ny) + (iy - 1) * nx + 1 : &
                        (iz - 1) * (nx * ny) +  iy * nx) = ygrid(iy)
                grid_points(X_, (iz - 1) * (nx * ny) + (iy - 1) * nx + 1 : &
                        (iz - 1) * (nx * ny) +  iy * nx) = xgrid(:)
            end forall

        end do
    end function


    !> Return a vector containing the integrals of the basis functions.

    !> The elements of the resulting vector are given by
    !! \f$ I_i=\int_{-\infty}^{\infty} \chi_i(x) \mathrm{d}x \f$
    pure function Grid1D_get_ints(self, limits) result(integrals)
        class(Grid1D), intent(in)    :: self
        integer, intent(in), optional:: limits(2)
        real(REAL64), allocatable    :: integrals(:)

        integer                      :: icell, start_index, end_index, ncell
        real(REAL64)                 :: base_integrals(self%nlip)

        base_integrals=self%lip%integrals()
        if (present(limits)) then
            start_index = limits(1)
            end_index = limits(2) + 1
        else
            start_index = 0
            end_index = self%ncell
        end if
        ncell = end_index-start_index

        allocate(integrals(ncell*(self%nlip-1) + 1))
        integrals=0.d0
        forall (icell=1 : ncell)
            integrals( (icell-1)*(self%nlip-1) + 1: icell*(self%nlip-1)+1 ) = &
            integrals( (icell-1)*(self%nlip-1) + 1: icell*(self%nlip-1)+1 ) + &
                                            base_integrals * self%cell_scales(icell)
        end forall
    end function

    function Grid1D_get_coord(self, cell_limits) result(r)
        class(Grid1D), target                   :: self
        integer,       intent(in), optional     :: cell_limits(2)
        real(REAL64),  pointer                  :: r(:)
        if (present(cell_limits)) then
            r => self%grid((cell_limits(1)-1)*(self%nlip-1) + 1 : (cell_limits(2))*(self%nlip-1) + 1)
        else
            r => self%grid
        end if
    end function

    pure function Grid1D_get_coordinates(self, cell_limits) result(r)
        class(Grid1D), intent(in), target       :: self
        integer,       intent(in), optional     :: cell_limits(2)
        real(REAL64),  allocatable              :: r(:)
        if (present(cell_limits)) then
            r = self%grid((cell_limits(1)-1)*(self%nlip-1) + 1 : (cell_limits(2))*(self%nlip-1) + 1)
        else
            r = self%grid
        end if
    end function

    function Grid3D_get_subgrid(self, cell_ranges) result(new)
        class(Grid3D), intent(in) :: self
        integer,       intent(in) :: cell_ranges(2, 3)
        integer                   :: axis_range(2)
        type(Grid3D)              :: new
        integer                   :: iaxis

        ! the parameters that do not change upon slicing
        new%nlip = self%nlip
        new%lip=LIPBasisInit(new%nlip, self%grid_type)
        select case (self%grid_type)
            case (1)
                new%lower_lip=LIPBasisInit(new%nlip-1, gridtype=1) ! equidistant, but smaller
            case (2)
                new%lower_lip=LIPBasisInit(new%nlip-1, gridtype=3) ! special type
            case default
        end select
        new%pbc = self%pbc
        new%grid_type = self%grid_type
        allocate(new%axis(3))
        do iaxis = 1, 3 !(x, y, z)
            axis_range = cell_ranges(:, iaxis)
            new%axis(iaxis) = self%axis(iaxis)%get_subgrid(axis_range)
        enddo
        new%id=next_grid_id()
    end function

    function Grid1D_get_subgrid(self, cell_ranges) result(new)
        class(Grid1D), intent(in) :: self
        integer      , intent(in) :: cell_ranges(2)
        type(Grid1D)              :: new
        integer                   :: iaxis, grid_start, grid_end

        ! the parameters that do not change upon slicing
        new%nlip = self%nlip
        new%lip=LIPBasisInit(new%nlip, self%grid_type)
        select case (self%grid_type)
            case (1)
                new%lower_lip=LIPBasisInit(new%nlip-1, gridtype=1) ! equidistant, but smaller
            case (2)
                new%lower_lip=LIPBasisInit(new%nlip-1, gridtype=3) ! special type
            case default
        end select
        new%grid_type = self%grid_type

        ! parameters that have to be recalculated
        new%qmin  = self%celld(cell_ranges(1))
        new%qmax  = self%celld(cell_ranges(2)) + self%cellen(cell_ranges(2))
        new%delta = new%qmax - new%qmin
        new%ncell = cell_ranges(2) - cell_ranges(1) + 1
        new%ndim  = new%ncell * (new%nlip - 1) + 1

        ! parameters that can be sliced
        new%celld = self%celld(cell_ranges(1):cell_ranges(2))
        new%cell_scales = self%cell_scales(cell_ranges(1):cell_ranges(2))
        new%cellen = self%cellen(cell_ranges(1):cell_ranges(2))
        grid_start = ((new%nlip -1) * (cell_ranges(1) - 1)) + 1
        grid_end   = grid_start + new%ndim - 1
        new%grid = self%grid( grid_start : grid_end )
    end function

    function Grid3D_get_pbc(self) result(pbc_ptr)
        class(Grid3D), target :: self
        type(PBC), pointer    :: pbc_ptr

        pbc_ptr=>self%pbc
    end function

    function Grid1D_get_cellen(self) result(cellen)
        class(Grid1D), target :: self
        real(REAL64), pointer :: cellen(:)

        cellen => self%cellen
    end function

    pure function Grid1D_get_qmax(self) result(r)
        class(Grid1D), intent(in)  :: self
        real(REAL64) :: r

        r=self%qmax
    end function

    pure function Grid1D_get_qmin(self) result(r)
        class(Grid1D), intent(in)  :: self
        real(REAL64) :: r

        r=self%qmin
    end function

    pure function get_slice_start(self) result(istart)
        class(Grid1D), intent(in)  :: self
        integer(INT32) :: istart

        istart=self%slice_start
    end function

    pure function get_slice_rank(self) result(rank)
        class(Grid1D), intent(in)  :: self
        integer(INT32) :: rank

        rank=self%slice_rank
    end function

    pure function get_slice_tot(self) result(tot)
        class(Grid1D), intent(in)  :: self
        integer(INT32) :: tot

        tot=self%slice_tot
    end function

    pure function Grid3D_get_id(self) result(id)
        class(Grid3D), intent(in)  :: self
        integer(INT32) :: id
        id=self%id
    end function

    pure function Grid3D_get_grid_type(self) result(grid_type)
        class(Grid3D), intent(in)  :: self
        integer(int32)             :: grid_type
        grid_type=self%grid_type
    end function

    function next_grid_id() result(new_id)
        integer, save :: old_id=1
        integer ::      new_id

        !call pdebug('Assigned new grid id: '//xchar(old_id), 1)
        new_id = old_id
        old_id = old_id + 1
    end function

    ! Grid state
    function Grid1D_is_equidistant(self) result(ok)
        class(Grid1D)   :: self
        logical         :: ok
        ok = self%equidistant_cells
    end function

    ! Check if grid 'self' is a subgrid of 'grid'
    function Grid1D_is_subgrid_of(self, grid) result(is_subgrid)
        class(Grid1D),intent(in)    :: self
        type(Grid1D), intent(in)    :: grid
        logical                     :: is_subgrid

        real(REAL64), parameter     :: TOLERANCE = 1e-7
        integer                     :: start_cell, end_cell
        start_cell = grid%get_icell(self%qmin)
        end_cell = start_cell + self%get_ncell() - 1
        !print *, "start_cell", start_cell, "end_cell", end_cell, "self cell count", size(self%cell_scales)
        if (start_cell < 1 .or. end_cell < 1 .or. end_cell > grid%get_ncell()) then
            is_subgrid = .FALSE.
            return
        end if
        is_subgrid = all([self%nlip == grid%nlip, self%delta < grid%delta, &
                          abs(self%qmin - grid%celld(start_cell)) < TOLERANCE, &
                          abs(self%qmax - grid%celld(end_cell) - grid%cellen(end_cell)) < TOLERANCE, &
                          all(self%cell_scales == grid%cell_scales(start_cell : end_cell))])
    end function

    function Grid3D_is_equidistant(self) result(ok)
        class(Grid3D)   :: self
        logical         :: ok

        logical         :: okx, oky, okz

        okx = self%axis(X_)%is_equidistant()
        oky = self%axis(Y_)%is_equidistant()
        okz = self%axis(Z_)%is_equidistant()
        ok = all([okx, oky, okz])
    end function

    ! Grid state
    function Grid3D_is_subgrid_of(self, grid) result(is_subgrid)
        class(Grid3D),intent(in)    :: self
        type(Grid3D), intent(in)    :: grid
        logical                     :: is_subgrid
        integer                     :: iaxis
        is_subgrid = all([(self%axis(iaxis)%is_subgrid_of(grid%axis(iaxis)), iaxis = X_, Z_)])
    end function


    ! Module procedures


    !> Check if `self` and `grid2` have similar grid, i.e., at the same position.
    !! and with the same shape
    pure function Grid1D_is_similar(self, grid2) result(is_similar)
        class(Grid1D), intent(in) :: self
        class(Grid1D), intent(in) :: grid2
        logical                   :: is_similar
        logical                   :: have_same_shape
        logical                   :: have_same_range
        real(REAL64), parameter   :: tolerance = 1E-10
        integer                   :: shape1, shape2
        real(REAl64)              :: range_difference(2)

        range_difference(1) = self%get_qmin() - grid2%get_qmin()
        range_difference(2) = self%get_qmax() - grid2%get_qmax()

        shape1 = self%get_shape()
        shape2 = grid2%get_shape()

        have_same_shape   =  shape1 == shape2
        have_same_range   =  all([range_difference(1) < tolerance, range_difference(2) < tolerance])
        is_similar        =  all([have_same_shape, have_same_range])

    end function


    !> Check if `self` and `grid2` have the same id.
    pure function Grid3D_is_equal(self, grid2) result(is_equal)
        class(Grid3D), intent(in) :: self
        class(Grid3D), intent(in) :: grid2
        logical                   :: is_equal
        is_equal = (self%id==grid2%id)
    end function

    !> Check if `self` and `grid2` have similar grid, i.e., at the same position.
    !! and with the same shape
    pure function Grid3D_is_similar(self, grid2) result(is_similar)
        class(Grid3D), intent(in) :: self
        type(Grid3D),  intent(in) :: grid2
        logical                   :: is_similar

        is_similar        =  all([self%axis(X_)%is_similar(grid2%axis(X_)), &
                                  self%axis(Y_)%is_similar(grid2%axis(Y_)), &
                                  self%axis(Z_)%is_similar(grid2%axis(Z_))])

    end function

    !> Check if `point` is withing the range of `self`
    pure function Grid1D_is_point_within_range(self, point) result(res)
        class(Grid1D), intent(in) :: self
        real(REAL64),  intent(in) :: point
        logical                   :: res

        res = point >= self%get_qmin() .and. point <= self%get_qmax()

    end function

    !> Check if `point` is withing the range of `self`
    pure function Grid3D_is_point_within_range(self, point) result(res)
        class(Grid3D), intent(in) :: self
        real(REAL64),  intent(in) :: point(3)
        logical                   :: res

        res =       self%axis(X_)%is_point_within_range(point(X_)) &
              .and. self%axis(Y_)%is_point_within_range(point(Y_)) &
              .and. self%axis(Z_)%is_point_within_range(point(Z_))

    end function

    ! xwh
    ! in a cubic cell, return the index
    ! of all the x, y, z axial grids of point (i,j,k)
    ! thus i \in [1,nlip]
    ! all the cubic grids are arranged in 1-D
    function axis_index(nlip, i, j, k)
        integer(int32) :: nlip, i, j, k
        integer(int32), dimension(nlip*3) :: axis_index

        integer(int32) :: l
        do l=1, nlip
            axis_index(l) = (k-1)*nlip**2+(j-1)*nlip+l
            axis_index(nlip+l) = (k-1)*nlip**2+(l-1)*nlip+i
            axis_index(2*nlip+l) = (l-1)*nlip**2+(j-1)*nlip+i
        end do
    end function axis_index


    subroutine set_grid_slice(self,crd,nslices,rank)
        type(Grid3D) :: self
        integer(INT32), intent(in) :: crd
        integer(INT32) :: nslices,rank

        self%axis(crd)%slice_n=nslices
        self%axis(crd)%slice_rank=rank
    end subroutine

    !> Transform x from global coordinates to cell coordinates.
    ! t = (x-x_s)/h,
    ! x_s : left boundary of the cell
    ! t \in [first, last]
    elemental pure function Grid1D_x2cell(self,x) result(res)
        class(Grid1D), intent(in) :: self
        real(REAL64), intent(in)  :: x
        !> If the user passes the cell, it will be assumed that it is valid.
        integer                   :: icell
        real(REAL64)              :: res

        icell=self%get_icell(x)
        ! If out of range, return 0.d0. This check is not performed if the
        ! user passed icell!
        if(icell == 0) then
            res=0.d0
            return
        end if
        res=(x-self%celld(icell))/self%cell_scales(icell)+self%lip%get_first()
    end function

    elemental pure function Grid1D_coordinate_to_grid_point_coordinate(self, coordinate) &
                                                     result(res)
        class(Grid1D), intent(in) :: self
        real(REAL64),  intent(in) :: coordinate
        real(REAL64)              :: res
        res =   self%get_icell(coordinate) * (self%get_nlip()-1) +1 &
              + self%lip%get_first() &
              + self%x2cell(coordinate)
    end function

    pure function Grid3D_coordinates_to_grid_point_coordinates(self, coordinates) &
                                                     result(res)
        class(Grid3D), intent(in) :: self
        real(REAL64),  intent(in) :: coordinates(3)
        real(REAL64)              :: res(3)

        res(X_) = self%axis(X_)%coordinate_to_grid_point_coordinate(coordinates(X_))
        res(Y_) = self%axis(Y_)%coordinate_to_grid_point_coordinate(coordinates(Y_))
        res(Z_) = self%axis(Z_)%coordinate_to_grid_point_coordinate(coordinates(Z_))
    end function

    elemental pure function Grid1D_x2cell_icell(self, x, icell) result(res)
        class(Grid1D), intent(in) :: self
        real(REAL64),  intent(in) :: x
        !> If the user passes the cell, it will be assumed that it is valid.
        integer, intent(in)       :: icell
        real(REAL64)              :: res

        res=(x-self%celld(icell))/self%cell_scales(icell)+self%lip%get_first()
    end function

    !> Transform cartesian coordinates to cell coordinates
    function Grid3D_cartesian_coordinates_to_cell_coordinates(self, coordinates) result(res)
        class(Grid3D) :: self
        real(REAL64), intent(in) :: coordinates(3)
        real(REAL64) :: res(3)

        integer :: ic

        res(X_) = self%axis(X_)%get_icell(coordinates(X_))
        res(Y_) = self%axis(Y_)%get_icell(coordinates(Y_))
        res(Z_) = self%axis(Z_)%get_icell(coordinates(Z_))
    end function

    function Grid3D_cartesian_coordinates_to_cube_coordinates(self, coordinates) result(res)
        class(Grid3D) :: self
        real(REAL64), intent(in) :: coordinates(3)
        real(REAL64) :: res(3)

        integer :: ic

        res = self%cartesian_coordinates_to_cell_coordinates(coordinates)
        res = (res - [1, 1, 1]) * (self%get_nlip() - 1) + 1
    end function

    ! r: (N*M), N: number of cells, M, number of nlips in a cell
    function grid1d_r2int(self)  result(r)
        class(grid1d), intent(in) :: self
        real(real64), allocatable, dimension(:,:) :: r

        ! for each shell, length, middle point coord
        real(real64) :: h, xm
        integer(int32) :: n, i
        type(real64_2d), dimension(1) :: coeff0
        real(real64), allocatable, dimension(:,:) :: coeff1
        real(real64), allocatable, dimension(:,:) :: coeff, coeff2, coeff3, coeff4
        ! (h*t+xm)^2, coeff with t
        real(real64), dimension(3) :: r2
        ! (nlip+2, nlip+1, ..., 1), for integration convenience
        real(real64), dimension(:), allocatable :: sequ

        allocate(r(self%get_ncell(),self%get_nlip()))
        allocate(sequ(self%get_nlip()+2))
        do n = 1, self%get_nlip()+2
            sequ(n) = self%get_nlip()+3-n
        end do

        ! get coeff of 'standard' lips
        coeff0 = self%lip%coeffs(0)
        allocate(coeff1(self%get_nlip(), self%get_nlip()))
        coeff1 = coeff0(1)%p

        do n = 1, self%get_ncell()
            h = self%cell_scales(n)
            xm = self%get_cell_center(n)
            allocate(coeff(self%get_nlip(), self%get_nlip()+2), source = 0.0d0)
            allocate(coeff2(self%get_nlip(), self%get_nlip()+2), source = 0.0d0)
            allocate(coeff3(self%get_nlip(), self%get_nlip()+2), source = 0.0d0)
            allocate(coeff4(self%get_nlip(), self%get_nlip()+2), source = 0.0d0)

            r2 = [h**2, 2.0d0*h*xm, xm**2]
            ! expansion of L(r)*r2, new L(r) after multiply the 1st term of r2
            coeff2(:,1:self%get_nlip()) = coeff1(:,:)*r2(1)
            ! new L(r) after multiply the 2nd term of r2
            coeff3(:,2:self%get_nlip()+1) = coeff1(:,:)*r2(2)
            coeff4(:,3:self%get_nlip()+2) = coeff1(:,:)*r2(3)
            ! h from the variable substitution from x to standard lip variable t
            ! dx -> h*dt
            coeff = (coeff2+coeff3+coeff4)*h
            do i = 1, self%get_nlip()
                r(n,i) = sum( coeff(i,:)*(self%lip%get_last()**sequ(:)-&
                         self%lip%get_first()**sequ(:))/sequ(:) )
            end do
            deallocate(coeff)
            deallocate(coeff2)
            deallocate(coeff3)
            deallocate(coeff4)
        end do

        deallocate(coeff1)
    end function grid1d_r2int

    function Grid1D_dr_ds(self)  result(r)
        class(grid1d), intent(in) :: self
        real(real64), dimension(:), allocatable :: r

        integer(int32) :: grid_shape
        real(REAL64), pointer         :: coord(:)
        ! x(i+1)-x(i)
        real(real64), dimension(:), allocatable :: r_ahead
        ! x(i)-x(i-1)
        real(real64), dimension(:), allocatable :: r_back
        integer(int32) :: p

        coord => self%get_coord()
        grid_shape = self%get_shape()
        allocate(r(grid_shape))
        allocate(r_ahead(grid_shape -2))
        allocate(r_back(grid_shape -2))
        r(1) = coord(2)-coord(1)
        r(grid_shape) = coord(grid_shape)-coord(grid_shape-1)
        do p = 1, grid_shape-2
            r_ahead(p) = coord(p+2)-coord(p+1)
            r_back(p) = coord(p+1)-coord(p)
        end do
        r(2:grid_shape-1) = 0.5d0*(r_ahead+r_back)

        deallocate(r_ahead)
        deallocate(r_back)
    end function Grid1D_dr_ds

    function Grid1D_lip_dev(self,point)  result(r)
        class(grid1d), intent(in) :: self
        real(real64), dimension(:), intent(in) :: point
        real(real64), dimension(:,:), allocatable :: r

        ! nth cell
        integer(int32) :: n
        integer(int32) :: i, j
        ! t coordinate in standard lip
        real(real64) :: t
        ! step
        real(real64) :: h

        ! auxillary to store the standard lip coeff
        ! 1st order derivative
        type(real64_2d), dimension(2) :: coeff0
        real(real64), allocatable, dimension(:,:) :: coeff1
        ! auxillary to store the power of lip
        ! [nlip-2, ..., 0]
        real(real64), dimension(:), allocatable :: sequ

        ! row: points
        allocate(r(size(point),self%get_nlip()))
        allocate(sequ(self%get_nlip()-1))
        do j = 1, size(point)
            n = self%get_icell(point(j))
            h = self%get_cell_scale(n)
            t = self%x2cell(point(j))

            coeff0 = self%lip%coeffs(1)
            coeff1 = coeff0(2)%p
            do n = 1, self%get_nlip()-1
                sequ(n) = self%get_nlip()-1-n
            end do
            ! L'(t) = L'(x) x'(t)
            ! x'(t) = h
            do i = 1, self%get_nlip()
                r(j,i) = sum(coeff1(i,:)*t**sequ(:))/h
            end do
            deallocate(coeff1)
        end do

        deallocate(sequ)
    end function


    ! xwh
    function Grid1D_lip_dev_m(self,icell)  result(r)
        class(grid1d), intent(in) :: self
        ! the index of cell
        integer(int32), intent(in) :: icell
        real(real64), dimension(:,:), allocatable :: r

        real(REAL64), pointer         :: coord(:)
        real(real64), dimension(:), allocatable :: cell_grids

        allocate(r(self%get_nlip(),self%get_nlip()))
        allocate(cell_grids(self%get_nlip()))
        coord => self%get_coord()
        cell_grids = coord((icell-1)*(self%nlip-1)+1:icell*(self%nlip-1)+1)
        ! row is grids index, after this call
        ! column is the lip index
        r=self%lip_dev(cell_grids)

        deallocate(cell_grids)
    end function

    ! xwh
    pure function Grid1D_ints(self, limits) result(integrals)
        class(Grid1D), intent(in)    :: self
        integer, intent(in), optional:: limits(2)
        real(REAL64), allocatable    :: integrals(:)


        integer                      :: icell, start_index, end_index, ncell
        real(REAL64)                 :: base_integrals(self%nlip)

        base_integrals=self%lip%integrals()
        if (present(limits)) then
            start_index = limits(1)
            end_index = limits(2) + 1
        else
            start_index = 0
            end_index = self%ncell
        end if
        ncell = end_index-start_index

        allocate(integrals(ncell*self%nlip))
        integrals=0.d0
        do icell=1, ncell
            integrals( (icell-1)*self%nlip + 1: icell*self%nlip ) = &
            integrals( (icell-1)*self%nlip + 1: icell*self%nlip ) + &
                                            base_integrals * self%cell_scales(icell)
        end do
    end function

    ! xwh
    function Grid3D_ints(self) result(ints)
        class(Grid3D), intent(in)  :: self
        real(REAL64), dimension(:,:), allocatable :: ints

        if (.not. self%is_equidistant()) then
            print *, 'error. only equidistant grid implemented'
        end if
        return

        ! take the first cell from each dimension
        allocate(ints(3,self%get_nlip()))
        ints(1,:) = self%axis(X_)%ints([1,1])
        ints(2,:) = self%axis(Y_)%ints([1,1])
        ints(3,:) = self%axis(Z_)%ints([1,1])
    end function


    !> Allocate matrices for Operator3D subclasses.

    !> Returns allocated, unitialized "`type(REAL64_3D) :: f(3)`" object, with
    !! adequate dimensions for Operator3D objects.
    function alloc_transformation_matrices(gridin, gridout, nt) result(f)
        type(Grid3D), intent(in) :: gridin
        type(Grid3D), intent(in) :: gridout
        integer, intent(in)      :: nt

        type(REAL64_3D)          :: f(3)

        integer                  :: dims_in(3)
        integer                  :: dims_out(3)

        dims_in=gridin%get_shape()
        dims_out=gridout%get_shape()

        allocate(f(X_)%p( dims_out(X_), dims_in (X_), nt ))
        allocate(f(Y_)%p( dims_in (Y_), dims_out(Y_), nt ))
        allocate(f(Z_)%p( dims_in (Z_), dims_out(Z_), nt ))
    end function

    subroutine store_Grid1Ds(grids, folder, filename)
        type(Grid1D),     intent(in)    :: grids(:)
        character(len=*), intent(in)    :: folder
        character(len=*), intent(in)    :: filename
        integer                         :: igrid

        open(unit=WRITE_FD, file=trim(folder)//"/"//trim(filename), access='stream')
        write(WRITE_FD) size(grids)
        do igrid = 1, size(grids)
            write(WRITE_FD) grids(igrid)%get_ncell()
            write(WRITE_FD) grids(igrid)%get_nlip()
            write(WRITE_FD) grids(igrid)%get_cell_scales()
            write(WRITE_FD) grids(igrid)%get_grid_type()
        end do
        close(unit=WRITE_FD)
    end subroutine

    subroutine resume_Grid1Ds(grids, folder, filename)
        type(Grid1D),     allocatable, intent(inout) :: grids(:)
        character(len=*),              intent(in)    :: folder
        character(len=*),              intent(in)    :: filename
        integer                                      :: ngrid, igrid, ncell, nlip
        real(REAL64), allocatable                    :: cell_scales(:)
        logical                                      :: file_exists_
        integer                                      :: grid_type

        file_exists_ = file_exists(folder, filename)
        if (file_exists_) then
            call pdebug("Loading grid1D from `"//trim(folder)//"/"//trim(filename), 1)

            open(unit=READ_FD, file=trim(folder)//"/"//trim(filename), access='stream')
            read(READ_FD) ngrid
            allocate(grids(ngrid))
            do igrid = 1, ngrid
                read(READ_FD) ncell, nlip
                allocate(cell_scales(ncell))
                read(READ_FD) cell_scales
                read(READ_FD) grid_type
                grids(igrid) = Grid1D(0.0d0, ncell, nlip, cell_scales, grid_type)
                deallocate(cell_scales)
            end do
            close(unit=READ_FD)
        end if
        return
    end subroutine

end module

