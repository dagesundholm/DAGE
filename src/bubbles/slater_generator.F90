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
!> @file slater_generator.F90
!! Slater type charge densities and corresponding electrostatic potentials.

!> Slater type charge densities and electrostatic potentials.
module SlaterGenerator_class
    use globals_m
    use Generator_class
    use Grid_class
    use Bubbles_class
    use Function3D_class
    use Function3D_types_m
    use Analytical_m, only: slater_analytical_energy, nucint_analytical_energy
    implicit none

    public :: SlaterGenerator
    public :: SlaterPotGenerator

    private
    !> Generator of Slater type charge densities

    !> An instance of this type is a factory-like object that contains a
    !! method %gen() that returns a Function3D object which represents a
    !! charge density made of Slater-like charge distributions, of the form
    !!
    !! @f[ \rho(\mathbf{r}) =
    !! \sum_{i\in\text{centers}}
    !! \sum_{j\in i}
    !! \frac{2}{\pi}\frac{N_j\zeta_j^{l_j+3}(1+2l_j)}{(l_j+2)!} r^{l_j}
    !! \mathrm{e}^{-2\zeta_j r_i}Y_{l_jm_j}(\theta_i,\phi_i)
    !! @f]
    !!
    !! The resulting function is built exclusively from Bubbles, i.e. the cube
    !! is 0.
    !!
    !! The parameters consist of a list of nuclear centers
    !! (slatergenerator::centers) and their
    !! corresponding
    !! nuclear charges (slatergenerator::z), and a set of shells given by
    !! the nucleus at which each shell is centered (slatergenerator::ibub),
    !! the angular momentum numbers l and m (slatergenerator::l and
    !! slatergenerator::m), the exponents \f$\{\zeta_j\}\f$
    !! (slatergenerator::expos) and the coefficients \f$\{N_j\}\f$
    !! (slatergenerator::coeffs).
    !!
    !! There are two available constructors: either all arrays containing the
    !! parameters of the shells are [passed explicitly]
    !! (@ref slatergenerator::slatergenerator_init_explicit):
    !! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.f90}
    !! rhogen=SlaterGenerator(z      = [1.d0, 1.d0],&
    !!                        centers= reshape(&
    !!                                 [-0.8660254037, -0.5d0, 0.1d0, &
    !!                                   0.8660254037,  0.5d0, 0.1d0],&
    !!                                 [3,2] ),&
    !!                        ibub   = [1, 2], &
    !!                        l      = [0, 0], &
    !!                        m      = [0, 0], &
    !!                        expos  = [1.d0, 1.d0], &
    !!                        coeffs = [1.d0, 1.d0])
    !! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    !! or [the parameters are passed as an array of strings]
    !! (@ref slatergenerator::slatergenerator_init_parse) containing the
    !! parameters:
    !! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.f90}
    !! rhogen=SlaterGenerator([" 2 1.0     1.0   0.5 1.5  ",&
    !!                         "      0  0  1.0 1.0       ",&
    !!                         "      0  0  1.0 2.0       "&
    !!                                                     ,&
    !!                         " 1 1.0     -1.5  3.0  0.5 ",&
    !!                         "      0  0  1.0 1.0       "&
    !!                                                     ,&
    !!                         " 3 2.0     -0.5 -1.5 -1.0 ",&
    !!                         "      0  0  1.0 2.0       ",&
    !!                         "      0  0  1.0 1.0       ",&
    !!                         "      0  0  2.0 3.0       "])
    !! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    type, extends(Generator) :: SlaterGenerator
        private
        !> Atomic numbers, shape (NBUB)
        real(REAL64), allocatable        :: z(:)
        !> Center coordinates, shape (3,NBUB)
        real(REAL64), allocatable        :: centers(:,:)
        !> Shell centers, shape (NSHELL)
        integer, allocatable             :: ibub(:)
        !> Shell l numbers, shape (NSHELL)
        integer, allocatable             :: l(:)
        !> Shell m numbers, shape (NSHELL)
        integer, allocatable             :: m(:)
        !> Shell expos, shape (NSHELL)
        real(REAL64), allocatable        :: expos(:)
        !> Shell coefficients (not including normalization factor), shape (NSHELL)
        real(REAL64), allocatable        :: coeffs(:)
        !> step
        real(REAL64)                :: step=0.2d0
        !> nlip
        integer                     :: nlip=7
        !> Threshold
        real(REAL64)                :: thres=1.d-12
    contains
        procedure :: gen_bubf     => slatergenerator_gen_bubf
        procedure :: gen_bubbles  => slatergenerator_gen_bubbles
        procedure :: gen_cubegrid => slatergenerator_gen_cubegrid
        procedure :: selfint      => slatergenerator_selfint
        procedure :: nucint       => slatergenerator_nucint
        procedure :: destroy      => slatergenerator_destroy
    end type

    !> Electric potential of s-type orbitals
    type, extends(SlaterGenerator) :: SlaterPotGenerator
        contains
            procedure :: gen_bubf => slaterpotgenerator_gen_bubf
    end type

    interface SlaterGenerator
        module procedure :: SlaterGenerator_init_explicit
        module procedure :: SlaterGenerator_init_parse
    end interface

    interface SlaterPotGenerator
        module procedure :: SlaterPotGenerator_init_explicit
        module procedure :: SlaterPotGenerator_init_parse
        module procedure :: SlaterPotGenerator_init_fromSlaterGenerator
    end interface

contains

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SlaterGenerator %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    !> Constructor of SlaterGenerator from explicitly passed parameters.
    function SlaterGenerator_init_explicit(z, centers, ibub, l, m, expos, &
                                       coeffs, step, nlip, thres) result(new)
        real(REAL64), intent(in)           :: z(:)
        real(REAL64), intent(in)           :: centers(3,size(z))
        integer,      intent(in)           :: ibub(:)
        integer,      intent(in)           :: l(size(ibub))
        integer,      intent(in)           :: m(size(ibub))
        real(REAL64), intent(in)           :: expos(size(ibub))
        real(REAL64), intent(in)           :: coeffs(size(ibub))
        real(REAL64), intent(in), optional :: step
        integer,      intent(in), optional :: nlip
        real(REAL64), intent(in), optional :: thres

        type(SlaterGenerator)              :: new

        call new%set(label ="Slater Density",&
                     type  =F3D_TYPE_CUSP )
        new%z       = z
        new%centers = centers
        new%ibub    = ibub
        new%l       = l
        new%m       = m
        new%expos   = expos
        new%coeffs  = coeffs
        if(present(step))  new%step    = step
        if(present(nlip))  new%nlip    = nlip
        if(present(thres)) new%thres   = thres
    end function

    !> Construct from parameters given as text block.

    !> Each center located at `[x,y,z]` and having a nuclear charge
    !! `qnuc` is defined with following syntax:
    !!
    !! ~~~~~~~~~~~~~~~~~~~
    !! nshells qnuc x y z
    !!  l m q zeta
    !!  l m q zeta
    !!  ...
    !! ~~~~~~~~~~~~~~~~~~~
    !!
    !! The `nshells` parameter gives the number of Slater functions at
    !! the center and the following `nshells` lines define the spherical
    !! harmonic indices (l,m), number of electrons `q` and the `zeta`
    !! parameter of each Slater function.
    function SlaterGenerator_init_parse(section, step, nlip, thres) result(new)
        character(len=*), intent(in)           :: section(:)
        real(REAL64),     intent(in), optional :: step
        integer,          intent(in), optional :: nlip
        real(REAL64),     intent(in), optional :: thres

        type(SlaterGenerator)              :: new

        real(REAL64)                       :: z
        real(REAL64)                       :: center(3)
        integer                            :: l
        integer                            :: m
        real(REAL64)                       :: expo
        real(REAL64)                       :: coeff
        integer                            :: iline, icenter, ishell, nshell

        ! The parsing is done by dynamically reallocating the arrays, which is
        ! not very efficient...
        allocate(new%z(0))
        allocate(new%centers(0,0))
        allocate(new%ibub(0))
        allocate(new%l(0))
        allocate(new%m(0))
        allocate(new%expos(0))
        allocate(new%coeffs(0))
        iline=1
        icenter=0
        do
            read(section(iline),*) nshell, z, center
            new%centers=reshape( [ new%centers , center ], &
                                 [ 3, size(new%centers,2)+1 ] )
            new%z=[ new%z, z ]
            iline=iline+1
            icenter=icenter+1
            do ishell=1,nshell
                read(section(iline),*) l, m, coeff, expo
                new%l      = [ new%l,      l ]
                new%m      = [ new%m,      m ]
                new%coeffs = [ new%coeffs, coeff ]
                new%expos  = [ new%expos,  expo ]
                new%ibub   = [ new%ibub, [ icenter ] ]
                iline=iline+1
            end do
            if(iline>size(section)) exit
        end do

        ! Sanity check
        if(any(new%expos <= 0.d0)) then
            call perror("The Slater exponents must be larger than 0")
            stop
        end if
        call new%set(label ="Slater Density",&
                     type  =F3D_TYPE_CUSP )
        if(present(step))  new%step    = step
        if(present(nlip))  new%nlip    = nlip
        if(present(thres)) new%thres   = thres
    end function

    !> Generate set of empty bubbles with adequate parameters.

    !> The Bubbles parameters are generated according to:
    !! . centers: SlaterGenerator::centers
    !! - lmax:    maxval(SlaterGenerator::l)
    !! - z:       SlaterGenerator::z
    !! - grids:   Calling grid_class::grid1dinit_radial, with N0=100 and
    !!   cutoff radius set to 20.0.
    function SlaterGenerator_gen_bubbles(self) result(bubs)
        class(SlaterGenerator), intent(in) :: self
        type(Bubbles)                      :: bubs

        integer :: i
        ! xwh, test
        real(real64) :: step(200) 
        real(real64) :: step0(6) = [0.2d0,0.1d0,0.05d0,0.02d0,0.01d0,0.005d0]
        step = 0.001d0
     
        !bubs=Bubbles(&
        !    lmax=maxval(self%l), &
        !    centers=self%centers, &
        !    global_centers=self%centers, &
! xwh, test
        !    grids=[ ( Grid1D(self%z(i), 100, self%nlip, 20.d0), &
!           grids=[ ( Grid1D(0.0d0, 200, self%nlip, step), &
!                                                  i=1,size(self%z) ) ], &
        !    z=self%z )
    end function

    !> Generates a cubegrid adequate for the resulting charge density.

    !> The resulting grid has its limits at least 8 au from all centers.
    !! Adequate padding is added so that the grid exactly has the desired
    !! step and nlip.
    function SlaterGenerator_gen_cubegrid(self) result(cubegrid)
        class(SlaterGenerator), intent(in)  :: self
        type(Grid3D)                        :: cubegrid

        integer      :: ncell(3)
        real(REAL64) :: box(2,3)
        real(REAL64) :: dx(3)
        integer      :: i

        real(REAL64), allocatable :: extent(:)
        real(REAL64), parameter   :: EXT=8.d0
        ! xwh, test
        real(REAL64) :: step(2)

!        extent=log(abs(self%coeffs)*self%expos**3/PI/self%thres)/&
!                    (2.d0*self%expos)
!        extent=[ (maxval(extent, mask=(self%ibub==i) ), i=1,size(self%z) )]
        extent=[ (EXT, i=1,size(self%z) )]

       cubegrid=Grid3D(centers= self%centers,&
                       radii  = extent,&
                       step   = self%step,&
                       nlip   = self%nlip )
! xwh, test
       ! test point (0,0,1)
       ! step = 0.001d0
       !  cubegrid=Grid3D(qmin= [-0.001d0,-0.001d0,0.999d0],&
       !                  ncell  = [2,2,2],&
       !                  nlip   = self%nlip,&
       !                  stepx  = step,&
       !                  stepy  = step,&
       !                  stepz  = step )
       ! test point (0,0,1/2)
       ! step = 0.001d0
       !  cubegrid=Grid3D(qmin= [-0.001d0,-0.001d0,0.499d0],&
       !                  ncell  = [2,2,2],&
       !                  nlip   = self%nlip,&
       !                  stepx  = step,&
       !                  stepy  = step,&
       !                  stepz  = step )
    end function

    !> Returns the radial parts of Slater type charge densities

    !> The radial function has the form
    !! @f[ \rho(r) =
    !! \frac{2}{\pi}\frac{N^{l+1}\zeta^{l+3}(1+2l)}{(l+2)!} r^{l} \exp(-2\zeta r)
    !! @f]
    !!
    !! This is implemented in an inefficient manner, and it should be only
    !! used for small test systems. The reason is that gen_bubf takes ibub,
    !! l, m and returns the whole radial function, and SlaterGenerator stores
    !! information per shell. Therefore, every call to gen_bubf goes through
    !! all shells to find all shells corresponding to that ibub, l, m.
    function SlaterGenerator_gen_bubf(self, ibub, l, m, r) result(f)
        class(SlaterGenerator), intent(in) :: self
        integer,                intent(in) :: ibub
        integer,                intent(in) :: l
        integer,                intent(in) :: m
        real(REAL64),           intent(in) :: r(:)

        real(REAL64)                       :: f(size(r))
        integer                            :: i

        f=0.d0
        ! Iterate over shells
        do i=1, size(self%expos)
            if((ibub == self%ibub(i)) .and. &
               (l    == self%l(i)   ) .and. &
               (m    == self%m(i)   )        ) then
                f = f + self%coeffs(i)* &
                        slater_dens( self%expos(i), self%l(i) , r )
            end if
        enddo
    end function

    !> \f$1/n!\f$
    pure function factorial(n)
        integer, intent(in) :: n
        real(REAL64)        :: factorial

        integer :: i

        factorial=1.d0
        do i=2, n
            factorial = factorial*i
        enddo
    end function

    elemental function slater_dens(zeta, l, r)
        real(REAL64), intent(in) :: zeta
        integer,      intent(in) :: l
        real(REAL64), intent(in) :: r

        real(REAL64)             :: slater_dens

        real(REAL64)             :: norm_const

        norm_const=2.0d0*zeta**(l+3)*(1.0d0+2*l)/PI/factorial(l+2)
        slater_dens=norm_const * r**l * exp(-2.d0*zeta*r)
    end function

    !> Destructor
    subroutine SlaterGenerator_destroy(self)
        class(SlaterGenerator), intent(inout) :: self

        if(allocated(self%z))        deallocate(self%z)
        if(allocated(self%centers))  deallocate(self%centers)
        if(allocated(self%ibub))     deallocate(self%ibub)
        if(allocated(self%l))        deallocate(self%l)
        if(allocated(self%m))        deallocate(self%m)
        if(allocated(self%expos))    deallocate(self%expos)
        if(allocated(self%coeffs))   deallocate(self%coeffs)
    end subroutine

! %%%%%%%%%%%%%%%%%%%%%%%%%%% SlaterPotGenerator %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    !> Explicit constructor
    function SlaterPotGenerator_init_explicit(z, centers, ibub, l, m, expos, &
                                       coeffs, step, nlip, thres) result(new)
        real(REAL64), intent(in)           :: z(:)
        real(REAL64), intent(in)           :: centers(3,size(z))
        integer,      intent(in)           :: ibub(:)
        integer,      intent(in)           :: l(size(ibub))
        integer,      intent(in)           :: m(size(ibub))
        real(REAL64), intent(in)           :: expos(size(ibub))
        real(REAL64), intent(in)           :: coeffs(size(ibub))
        real(REAL64), intent(in), optional :: step
        integer,      intent(in), optional :: nlip
        real(REAL64), intent(in), optional :: thres

        type(SlaterPotGenerator)           :: new

        call new%set(label ="Slater Density",&
                     type  =F3D_TYPE_SPOT )
        new%z       = z
        new%centers = centers
        new%ibub    = ibub
        new%l       = l
        new%m       = m
        new%expos   = expos
        new%coeffs  = coeffs
        if(present(step))  new%step    = step
        if(present(nlip))  new%nlip    = nlip
        if(present(thres)) new%thres   = thres
    end function

    !> Construct from parameters given as text block.
    function SlaterPotGenerator_init_parse(section, step, nlip, thres) &
                                                                result(new)
        character(len=*), intent(in)           :: section(:)
        real(REAL64),     intent(in), optional :: step
        integer,          intent(in), optional :: nlip
        real(REAL64),     intent(in), optional :: thres

        type(SlaterPotGenerator)               :: new

        new%SlaterGenerator=SlaterGenerator(section)
        if(present(step))  new%step    = step
        if(present(nlip))  new%nlip    = nlip
        if(present(thres)) new%thres   = thres
        call new%set(label ="Slater Density",&
                     type  =F3D_TYPE_SPOT )
    end function

    !> Construct from an already built SlaterGenerator
    function SlaterPotGenerator_init_fromSlaterGenerator(orig) result(new)
        type(SlaterGenerator), intent(in) :: orig
        type(SlaterPotGenerator)          :: new

        new%SlaterGenerator=orig
        call new%set(label ="Slater Density",&
                     type  =F3D_TYPE_SPOT )
    end function

    !> Returns the radial part of s-type Slater electic potential
    function SlaterPotGenerator_gen_bubf(self, ibub, l, m, r) result(f)
        class(SlaterPotGenerator), intent(in) :: self
        integer,                   intent(in) :: ibub
        integer,                   intent(in) :: l
        integer,                   intent(in) :: m
        real(REAL64),              intent(in) :: r(:)

        real(REAL64)                          :: f(size(r))

        integer                               :: i

        f=0.d0
        ! Iterate over shells
        do i=1, size(self%expos)
            if((ibub == self%ibub(i)) .and. &
               (l    == self%l(i)   ) .and. &
               (m    == self%m(i)   )        ) then
                f = f + self%coeffs(i) * &
                        slater_pot( self%expos(i), self%l(i) , r )
            end if
        enddo

    end function

    elemental function slater_pot(zeta, l, r)
        real(REAL64), intent(in) :: zeta
        integer,      intent(in) :: l
        real(REAL64), intent(in) :: r

        real(REAL64)             :: slater_pot

        real(REAL64)             :: g
        real(REAL64)             :: twozr
        integer                  :: j

        if(r==0.d0) then
            if(l==0) then
                slater_pot=zeta
            else
                slater_pot=0.d0
            end if
        else
            twozr=2.d0*zeta*r
            g=1.d0
            do j=1,2*l
                g=g+twozr**j/factorial(j)
            end do
            slater_pot = ( factorial(2+2*l) - &
                           exp(-twozr) * ( g*factorial(2+2*l) + &
                                           (2*l+1.d0)*twozr**(2*l+1)) ) / &
                         factorial(2+l)/twozr**l/r
        end if
    end function

    !> Computes the analytical self-interaction energy of the charge density
    !! described by `self`.
    function SlaterGenerator_selfint(self, grid) result(selfint)
        class(SlaterGenerator), intent(in) :: self
        !> Grid, used for PBC
        type(Grid3D),           intent(in) :: grid
        real(REAL64)                       :: selfint

        selfint=slater_analytical_energy(grid, self%coeffs, self%expos, &
                                         self%centers(:,self%ibub) )
    end function

    !> Computes the analytical interaction energy of the charge density
    !! described by `self` with its set of nuclei.
    function SlaterGenerator_nucint(self, grid) result(selfint)
        class(SlaterGenerator), intent(in) :: self
        !> Grid, used for PBC
        type(Grid3D),           intent(in) :: grid
        real(REAL64)                       :: selfint

        selfint=nucint_analytical_energy(grid,&
                         self%coeffs, self%expos, self%centers(:,self%ibub),&
                         self%z, self%centers)
    end function
end module
