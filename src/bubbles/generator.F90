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
!> @file generator.F90
!! Base class for all generators (Function3D constructors).

module generator_class
    use ISO_FORTRAN_ENV
    use Globals_m
    use Grid_class
    use Bubbles_class
    use Function3D_class
    use Function3D_types_m
    use ParallelInfo_class
    use PBC_class
    implicit none

    public :: Generator

    private
!> Base class for all generators (Function3D constructors).
!!
!! A generator is an opaque object that stores parameters required to
!! construct functions. The syntax looks like:
!!
!! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.F90}
!! type(DerivedGenerator) :: my_gen
!! type(Function3D)       :: my_fun
!!
!! my_gen=DerivedGenerator( ...some parameters... )
!! my_fun=my_gen%gen()
!! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!!
!! This makes it possible to construct a wide variety of functions
!! without modifications to function construction routines per se, for
!! the parameters stored in a generator are hidden from the caller and
!! are thus arbitrary.
!!
!! `Generator` is an abstract type, and hence derived types must be
!! defined for any practical purpose (see e.g.
!! [GaussGenerator] (@ref gaussgenerator_class::gaussgenerator) and
!! [SlaterGenerator] (@ref slatergenerator_class::slatergenerator)).
!! New derived `Generator` types are built by overriding the following
!! methods:
!!
!! Method     |    Default     |      Description
!! -----------| ---------------| -------------------------------------------
!! [Generator%gen_bubbles()] (@ref generator::gen_bubbles) | @ref gen_bubbles_empty (produces an empty set of bubbles)   | Returns a complete [Bubbles] (@ref bubbles_class::bubbles) object.
!! [Generator%gen_bubf()] (@ref generator::gen_bubf) | @ref gen_bubf_zeros (produces zero-valued radial functions)   | Returns radial functions as a function of `ibub`, `l` and `m`
!! [Generator%gen_cubegrid()] (@ref generator::gen_cubegrid) | None (must be overriden!)   | Returns a cube grid
!! [Generator%gen_cubef()] (@ref generator::gen_cubef) | @ref gen_cubef_zeros (produces a zero-valued cube)   | Returns a cube.
!!
!! Furthermore, [Generator%destroy()] (@ref generator::destroy) must be
!! overriden. The new constructor functions (e.g.
!! [GaussGenerator_init_explicit()]
!! (@ref gaussgenerator_class::gaussgenerator_init_explicit)) should make use
!! of [Generator%set()] (@ref generator::set) to set `type` and `label`.
    type, abstract :: Generator
        private
        character(20)                   :: label="Generated Function3D"
        integer                         :: type=F3D_TYPE_NONE
    contains
        !> The most important method of Generator! Returns a complete
        !! Function3D instance.
        procedure                     :: gen => generator_generate_fun3d
        !> Generates a Bubbles instance.
        procedure                     :: gen_bubbles => gen_bubbles_empty
        !> Generates radial functions for the generated Bubbles.
        procedure                     :: gen_bubf => gen_bubf_zeros
        !> Generates a Grid3D.
        procedure(cubegrid), deferred :: gen_cubegrid
        !> Generates a cube.
        procedure                     :: gen_cubef => gen_cubef_zeros
        !> Set method for Generator%%label and Generator%%type.
        procedure                     :: set => generator_set
        !> Destructor
        procedure(Generator_destroy), deferred :: destroy
    end type

    abstract interface
        function cubegrid(self)
            import Generator, Grid3D, REAL64
            class(Generator), intent(in) :: self
            type(Grid3D)                 :: cubegrid
        end function

        subroutine Generator_destroy(self)
            import Generator
            class(Generator), intent(inout)  :: self
        end subroutine
    end interface

contains

    !> Setter for Generator%type and Generator%label
    subroutine Generator_set(self, type, label)
        class(Generator), intent(inout)    :: self
        integer, intent(in), optional      ::  type
        character(*), intent(in), optional ::  label

        if (present(type))  self%type= type
        if (present(label)) self%label=label
    end subroutine

    !> Returns an empty set of bubbles.
    function gen_bubbles_empty(self) result(bubs)
        class(Generator), intent(in) :: self
        type(Bubbles)                :: bubs

        return
    end function

    !> Returns zero-valued radial functions.
    function gen_bubf_zeros(self, ibub, l, m, r) result(bubf)
        class(Generator), intent(in) :: self
        integer, intent(in)          :: ibub
        integer, intent(in)          :: l
        integer, intent(in)          :: m
        real(REAL64), intent(in)     :: r(:)
        real(REAL64)                 :: bubf(size(r))

        bubf=0.d0
    end function

    !> Returns a zero-valued cube.
    function gen_cubef_zeros(self,x,y,z) result(cubef)
        class(Generator), intent(in) :: self
        !> Grid coordinates along x.
        real(REAL64),     intent(in) :: x(:)
        !> Grid coordinates along y.
        real(REAL64),     intent(in) :: y(:)
        !> Grid coordinates along z.
        real(REAL64),     intent(in) :: z(:)
        real(REAL64)                 :: cubef(size(x), size(y), size(z))

        cubef=0.d0
    end function

    !> Generates a Function3D object based on the Generator parameters.
    function Generator_generate_fun3d(self, grid, bubs) result(new)
        class(Generator), intent(in)                    :: self
        !> If passed, this will be the cube grid of the returned Function3D.
        !! Otherwise, the cube grid is the result of %gen_cubegrid().
        type(Grid3D),     intent(in), optional, target  :: grid
        !> If passed, this will be the bubbles (grids, centers, etc., NOT the
        !! radial functions!) of the returned Function3D. Otherwise, the Bubbles
        !! is the result of %gen_bubbles().
        type(Bubbles),    intent(in), optional, target  :: bubs
        type(Function3D)                                :: new

        type(Grid3D),  pointer :: grid_w
        type(Bubbles), pointer :: bubs_w
        integer                :: ibub, l, m
        integer                :: gdims(3)
        real(REAL64), pointer  :: x(:)
        real(REAL64), pointer  :: y(:)
        real(REAL64), pointer  :: z(:)

        real(REAL64), pointer :: r(:), flm(:)
        real(REAL64), pointer :: fijk(:,:,:)
        real(REAL64), allocatable :: tmp(:,:,:)

        real(REAL64)          :: lattice_v(3)
        real(REAL64)          :: t_v(3)
        type(PBC),   pointer  :: pbc_ptr
        logical               :: pbc_start
        integer               :: pbc_idx

        real(REAL64), parameter :: THRESHOLD=1.d-12

        ! Generate default bubbles unless explicitly given bubs argument
        if(present(bubs)) then
            bubs_w => bubs
        else
            allocate(bubs_w)
            bubs_w=self%gen_bubbles()
        end if

        ! Generate Bubbles radial functions
        do ibub=1,bubs_w%get_nbub()
            r=>bubs_w%gr(ibub)%p%get_coord()
            do l=0,bubs_w%get_lmax()
                do m=-l,l
                    print *, "getting f", ibub, l, m
                    flm=>bubs_w%get_f(ibub,l,m)
                    flm=self%gen_bubf(ibub, l, m, r)
                end do
            end do
        end do

        ! Generate default cube grid unless explicitly given grid argument
        if(present(grid)) then
            grid_w => grid
        else
            allocate(grid_w)
            grid_w=self%gen_cubegrid()
        end if
        
        ! Bootstrap Function3D
        new=Function3D( SerialInfo(grid_w), bubs_w, self%label, self%type )

! %%%%%%%%%        Generate cube     %%%%%%%%%%
        gdims=new%grid%get_shape()
        fijk=>new%get_cube()
        fijk=0.d0
        tmp=fijk

        ! Get grid coordinates
        x=>new%grid%axis(X_)%get_coord()
        y=>new%grid%axis(Y_)%get_coord()
        z=>new%grid%axis(Z_)%get_coord()

        pbc_ptr=>new%grid%get_pbc()
        lattice_v = new%grid%get_delta()

        ! The cube is generated using the PBC's given by the grid. The
        ! %gen_cubef function is called displacing the grid with the
        ! translational vector
        pbcloop: do pbc_idx=0,pbc_ptr%max_3dshell()! Loop over octahedra with total length 2a
            tmp=0.d0
            pbc_start=.TRUE.
            do    ! Loop over cells belonging to the octahedron
                ! We get the i-th translational vector
                t_v=lattice_v*pbc_ptr%next_tv(pbc_idx,pbc_start)
                
                ! If got the last vector in the octahedron, exit loop
                if(pbc_start) exit    
                
                tmp=tmp+self%gen_cubef(x+t_v(X_), y+t_v(Y_), z+t_v(Z_))
            end do
            fijk = fijk+tmp
            if(debug_g>=1) write(ERROR_UNIT, '(a,i6,a,e16.6,a,e16.6)',advance='no') &
                char(13)//"Layer: ", pbc_idx," Max. contrib: ", maxval(abs(tmp))
            if(maxval(abs(tmp))<THRESHOLD) exit
        end do pbcloop
        if(debug_g>=1) write(ERROR_UNIT, *)

        ! Cleanup
        if(.not.present(bubs)) then
            call bubs_w%destroy()
            deallocate(bubs_w)
        end if
        if(.not.present(grid)) then
            call grid_w%destroy()
            deallocate(grid_w)
        end if

    end function

end module
