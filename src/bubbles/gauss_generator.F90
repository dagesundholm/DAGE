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
!> @file gauss_generator.F90
!! Gaussian charge densities and corresponding electrostatic potentials.

module GaussGenerator_class
    use Globals_m
    use Generator_class
    use Grid_class
    use Bubbles_class
    use Function3D_class
    use Function3D_types_m
    use Analytical_m, only: gauss_analytical_energy
    implicit none

    public :: GaussGenerator
    public :: GaussBubblesGenerator
    public :: GaussPotGenerator

    private
    !> Generator of Gaussian charge densities.

    !> An instance of this type is a factory-like object that contains a
    !! method %gen() that returns a Function3D object which represents a
    !! charge density made of Gaussian charge distributions, of the form
    !!
    !! @f[ \rho(\mathbf{r}) =
    !! \sum_i q_i
    !! \left(\frac{\alpha_i}{\pi}\right)^{\frac{3}{2}}
    !! \exp(-\alpha_i r^2_i), \quad
    !! r_i = \| \mathbf{R}_i - \mathbf{r} \|
    !! @f]
    !!
    !! The resulting Function3D object has no Bubbles, i.e. the density is
    !! represented purely in the cube.
    !!
    !! There are two GaussGenerator constructors: either
    !! [passing all Gaussian shell parameters explicitly]
    !! (@ref gaussgenerator::gaussgenerator_init_parse):
    !! ~~~~~~~~~~~~~~~~~~~~~~{.f90}
    !! type(GaussGenerator)      :: rhogen
    !! rhogen=GaussGenerator(alpha  = [2.d0,  4.d0],&
    !!                       center = reshape([-1.d0, 0.d0, 0.d0,&
    !!                                          1.d0, 0.d0, 0.d0],[3,2]),&
    !!                       q      = [1.d0, -1.d0])
    !! ~~~~~~~~~~~~~~~~~~~~~~
    !! or alternatively, [from a list of strings]
    !! (@ref gaussgenerator::gaussgenerator_init_explicit) containing
    !! the parameters to be parsed:
    !! ~~~~~~~~~~~~~~~~~~~~~~{.f90}
    !! rhogen=GaussGenerator([" 1.0   2.0  -1.0 0.0 0.0",&
    !!                        "-1.0   4.0   1.0 0.0 0.0"])
    !! ~~~~~~~~~~~~~~~~~~~~~~
    !!
    !! The default generated grid (returned via
    !! [GaussGenerator%gen_cubegrid()] (@ref gaussgenerator::gen_cubegrid)
    !! is such that each individual Gaussian shell is at most
    !! gaussgenerator::thres at the edges of the resulting box. Appropriate
    !! padding is added such that, in each dimension, the grid contains an
    !! integer number of cells measuring
    !! (gaussgenerator::nlip-1)*(gaussgenerator::step) in length.
    !!
    !! @sa [GaussPotGenerator](@ref gaussgenerator_class::gausspotgenerator)
    type, extends(Generator) :: GaussGenerator
        private
        !> Exponents
        real(REAL64), allocatable   :: alpha(:)
        !> Center coordinates
        real(REAL64), allocatable   :: center(:,:)
        !> Charges
        real(REAL64), allocatable   :: q(:)
        !> Step of the default generated grid.
        real(REAL64)                :: step=0.2d0
        !> Number of LIP points of the default generated grid.
        integer(int32)              :: nlip=7
        !> grid_type. 1: eq, 2: Gauss Lobatto
        integer(int32)              :: grid_type = 2
        !> Maximum value of each individual Gaussian shell at the edges
        !! of the default generated grid.
        real(REAL64)                :: thres=1.d-12
    contains
        procedure :: gen_cubegrid    => gaussgenerator_gen_cubegrid
        procedure :: gen_cubef       => gaussgenerator_gen_cubef
        procedure :: selfint         => gaussgenerator_selfint
        procedure :: destroy         => gaussgenerator_destroy
    end type


    type, extends(GaussGenerator) :: GaussBubblesGenerator
        private
        !> Maximum value of each individual Gaussian shell at the edges
        !! of the default generated grid.
        real(REAL64)                :: cutoff=10.d0
        !> Shell centers, shape (NSHELL)
        integer, allocatable             :: ibub(:)
        !> Shell l numbers, shape (NSHELL)
        integer, allocatable             :: l(:)
        !> Shell m numbers, shape (NSHELL)
        integer, allocatable             :: m(:)
        !> Shell expos, shape (NSHELL)
    contains
        procedure :: gen_cubef       => gaussbubblesgenerator_gen_cubef
        procedure :: gen_bubf        => gaussbubblesgenerator_gen_bubf
        procedure :: gen_bubbles     => gaussbubblesgenerator_gen_bubbles
        procedure :: gen_cubegrid    => gaussbubblesgenerator_gen_cubegrid
    end type

    !> Generator of electrostatic potentials caused by Gaussian charge
    !! densities.

    !> Similar to gaussgenerator_class::gaussgenerator, but the generated
    !! Function3D represents the corresponding electrostatic potential,
    !! given by:
    !!
    !! @f[
    !!     V(\mathbf{r}) =
    !!  \sum_i q_i
    !! \frac{2}{\sqrt{\pi}}\frac{\mathrm{erf}(\sqrt{\alpha_i}r_i)}{r_i},
    !! \quad    r_i = \| \mathbf{R}_i - \mathbf{r} \|.
    !! @f]
    !!
    !! The constructors are identical to
    !! gaussgenerator_class::gaussgenerator; additionaly, a GaussPotGenerator
    !! instance can be also constructed from a GaussGenerator instance:
    !! ~~~~~~~~~~~~~~~~~~~~~~{.F90}
    !! type(GaussGenerator)      :: rhogen
    !! type(GaussPotGenerator)   :: potgen
    !! rhogen = GaussGenerator( ... )
    !! ...
    !! potgen = GaussGenerator(rhogen)
    !! ~~~~~~~~~~~~~~~~~~~~~~
    type, extends(GaussGenerator) :: GaussPotGenerator
    contains
        procedure :: gen_cubef    => gausspotgenerator_gen_cubef
    end type

    interface GaussGenerator
        module procedure :: GaussGenerator_init_explicit
        module procedure :: GaussGenerator_init_parse
    end interface

    interface GaussBubblesGenerator
        module procedure :: GaussBubblesGenerator_init_explicit
    end interface

    interface GaussPotGenerator
        module procedure :: GaussPotGenerator_init_explicit
        module procedure :: GaussPotGenerator_init_parse
        module procedure :: GaussPotGenerator_init_from_GaussGenerator
    end interface
contains

    !> Constructor from explicit list of parameters.
    function GaussGenerator_init_explicit(alpha,center,q,step, nlip, thres) &
                                                                 result(new)
        real(REAL64), intent(in)           :: alpha(:)
        real(REAL64), intent(in)           :: center(:,:)
        real(REAL64), intent(in)           :: q(:)
        real(REAL64), intent(in), optional :: step
        integer,      intent(in), optional :: nlip
        real(REAL64), intent(in), optional :: thres

        type(GaussGenerator)  :: new

        call new%set(label ="Gaussian Density",&
                     type  =F3D_TYPE_CUSP )
        new%alpha =alpha
        new%center=center
        new%q     =q
        if (present(step )) new%step  =step
        if (present(nlip )) new%nlip  =nlip
        if (present(thres)) new%thres =thres
    end function

    !> Parses text and populates a generator with correct parameters.
    !!
    !! Each line of the input is of the form
    !!
    !! ~~~~~~~~~~~~~~~~
    !! q alpha x y z
    !! ~~~~~~~~~~~~~~~~
    !!
    !! where `q` is the charge and `alpha` is the exponent
    !! of the Gaussian center located at `[x,y,z]`.
    function GaussGenerator_init_parse(section, step, nlip, thres) result(new)
        !> Array of strings of identical length containing the formatted
        !! parameters of the Gaussian shells.
        character(len=*), intent(in)           :: section(:)
        real(REAL64),     intent(in), optional :: step
        integer,          intent(in), optional :: nlip
        real(REAL64),     intent(in), optional :: thres

        type(GaussGenerator)                   :: new

        integer        :: i
        integer        :: ngau

        ngau = size(section)

        allocate(new%alpha(ngau))
        allocate(new%q(ngau))
        allocate(new%center(3,ngau))

        do i=1, ngau
            read(section(i), *) new%q(i), new%alpha(i), new%center(:,i)
        end do

        ! Sanity check
        if(any(new%alpha <= 0.d0)) then
            call perror("The Gaussian exponents must be larger than 0")
            stop
        end if
        if (present(step )) new%step  =step
        if (present(nlip )) new%nlip  =nlip
        if (present(thres)) new%thres =thres
    end function

    !> Destructor
    subroutine GaussGenerator_destroy(self)
        class(GaussGenerator), intent(inout) :: self

        if(allocated(self%alpha))    deallocate(self%alpha)
        if(allocated(self%center))   deallocate(self%center)
        if(allocated(self%q))        deallocate(self%q)
    end subroutine

    !> Default cube grid  generator for GaussGenerator

    !> We define the extent of each Gaussian shell is the radius beyond which
    !! their value is smaller than GaussGenerator::threshold, given by
    !!
    !! \f$\mathrm{extent}_i=
    !!         \sqrt{ \log\left(|q_i|(\alpha_i/\pi)^{3/2} /
    !!         \mathrm{threshold}\right) / \alpha_i}
    !! \f$
    !!
    !! The returned grid is such that it contains all the shells with
    !! a radius equal to their extent. Adequate padding is added such that
    !! an integer number of cells with the requested nlip and step is
    !! produced in each direction. The padding is added symmetrically, i.e.
    !! ifthe padding needed is x, x/2 is added on each side.
    function GaussGenerator_gen_cubegrid(self) result(cubegrid)
        class(GaussGenerator), intent(in)  :: self
        type(Grid3D)                       :: cubegrid

        integer      :: ncell(3)
        real(REAL64) :: box(2,3)
        real(REAL64) :: extent(size(self%alpha))
        real(REAL64) :: dx(3)
        integer      :: i

        extent=sqrt( log(abs(self%q)*(self%alpha/PI)**1.5d0 / self%thres) / &
                      self%alpha )

        write(*,*) 'using gridtype in slater-gen, used for cube '
        cubegrid=Grid3D(centers   = self%center, &
                        radii     = extent, &
                        step      = self%step, &
                        nlip      = self%nlip, &
                        grid_type = self%grid_type )
    end function

    !> Returns a test charge density consisting of Gaussian functions.

    !> Given the grid points \f$\{x_i\}\f$, \f$\{y_j\}\f$, \f$\{z_k\}\f$,
    !! return the three-dimensional array
    !! \f$\{A_{ijk}=
    !!     \sum_p q_p
    !!     \left(\frac{\alpha_p}{\pi}\right)^{\frac{3}{2}}
    !!     \mathrm{e}^{-\alpha_p \left[( x_i-X_p )^2+( y_j-Y_p )^2+( z_i-Z_p )^2 \right]} \f$
    function GaussGenerator_gen_cubef(self, x, y, z) result(cubef)
        class(GaussGenerator), intent(in) :: self
        real(REAL64),          intent(in) :: x(:)
        real(REAL64),          intent(in) :: y(:)
        real(REAL64),          intent(in) :: z(:)

        real(REAL64)                      :: cubef(size(x), size(y), size(z))

        real(REAL64)                :: ex(size(x))
        real(REAL64)                :: ey(size(y))
        real(REAL64)                :: ez(size(z))
        
        integer                     :: i, j, k, p
        real(REAL64)                :: norm

        print *, "x, y, z", size(x), size(y), size(z)
        cubef=0.d0

        do p=1, size(self%alpha)
            norm = (self%alpha(p)/PI)**(1.5d0)
            ex=exp(-self%alpha(p)*(x-self%center(X_,p))**2)
            ey=exp(-self%alpha(p)*(y-self%center(Y_,p))**2)
            ez=exp(-self%alpha(p)*(z-self%center(Z_,p))**2)
            forall(i=1:size(x), j=1:size(y), k=1:size(z))
                cubef(i,j,k)=cubef(i,j,k) + self%q(p)*norm*ex(i)*ey(j)*ez(k)
            end forall
        end do
    end function

    !> Computes the analytical self-interaction energy of the charge density
    !! described by `self`.
    function GaussGenerator_selfint(self, grid) result(new)
        class(GaussGenerator), intent(in) :: self
        !> Grid, used for PBC
        type(Grid3D),          intent(in) :: grid
        real(REAL64)                      :: new

        new = gauss_analytical_energy(grid, self%q, self%alpha, self%center)
    end function

! %%%%%%%%%%%%%%%%%% GaussPotGenerator %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    !> Build GaussPotGenerator from parameters, explicitly
    function GaussPotGenerator_init_explicit(alpha,center,q,step,nlip,thres) &
                                                                 result(new)
        real(REAL64), intent(in)           :: alpha(:)
        real(REAL64), intent(in)           :: center(:,:)
        real(REAL64), intent(in)           :: q(:)
        real(REAL64), intent(in), optional :: step
        integer,      intent(in), optional :: nlip
        real(REAL64), intent(in), optional :: thres

        type(GaussPotGenerator)            :: new

        call new%set(label ="Slater Density",&
                     type  =F3D_TYPE_SPOT )
        new%alpha =alpha
        new%center=center
        new%q     =q
        if (present(step )) new%step  =step
        if (present(nlip )) new%nlip  =nlip
        if (present(thres)) new%thres =thres
    end function

    !> Construct from parameters given as text block.
    function GaussPotGenerator_init_parse(section, step, nlip, thres) &
                                                                result(new)
        character(len=*), intent(in)           :: section(:)
        real(REAL64),     intent(in), optional :: step
        integer,          intent(in), optional :: nlip
        real(REAL64),     intent(in), optional :: thres

        type(GaussPotGenerator)                :: new

        new%GaussGenerator=GaussGenerator(section)
        if(present(step))  new%step    = step
        if(present(nlip))  new%nlip    = nlip
        if(present(thres)) new%thres   = thres
        call new%set(label ="Gaussian Density",&
                     type  =F3D_TYPE_SPOT )
    end function

    !> Construct from an already built GaussGenerator
    function GaussPotGenerator_init_from_GaussGenerator(orig) result(new)
        type(GaussGenerator), intent(in) :: orig
        type(GaussPotGenerator)          :: new

        new%GaussGenerator=orig
        call new%set(label ="Slater Density",&
                     type  =F3D_TYPE_SPOT )
    end function

    !> Returns the electric potential of a Gaussian charge distribution.
    function GaussPotGenerator_gen_cubef(self, x, y, z) result(cubef)
        class(GaussPotGenerator), intent(in) :: self
        real(REAL64),             intent(in) :: x(:)
        real(REAL64),             intent(in) :: y(:)
        real(REAL64),             intent(in) :: z(:)

        real(REAL64)                         :: cubef(size(x), size(y), size(z))

        integer                     :: i, j, k, p

        cubef=0.d0
        do p=1, size(self%alpha)
            forall(i=1:size(x), j=1:size(y), k=1:size(z))
                cubef(i,j,k) = cubef(i,j,k) + self%q(p) * &
                                   gausspot( self%alpha(p), &
                                             sqrt((self%center(X_,p)-x(i))**2+&
                                                  (self%center(Y_,p)-y(j))**2+&
                                                  (self%center(Z_,p)-z(k))**2) )
            end forall
        end do

    contains

        pure function gausspot(alpha, r)
            real(REAL64), intent(in) :: alpha
            real(REAL64), intent(in) :: r
            real(REAL64)             :: gausspot

            real(REAL64) :: sqrtalpha
            real(REAL64) :: temp
            real(REAL64) :: derf

            sqrtalpha=sqrt(alpha)
            temp=r*sqrtalpha

            if(temp<0.01d0) then
                gausspot=twooversqrtpi*sqrtalpha*&
                        (1.d0-0.33333333333333333d0*temp**2+0.1d0*temp**4)
            else if(temp<10.d0) then
                gausspot=derf(r*sqrtalpha)/r
            else
                gausspot=1.d0/r
            end if
            return
        end function gausspot
    end function


    function GaussBubblesGenerator_init_explicit(q, center, ibub, l, m, alpha, &
                                       step, nlip, thres, cutoff) result(new)
        real(REAL64), intent(in)           :: q(:)
        real(REAL64), intent(in)           :: center(3,size(q))
        integer,      intent(in)           :: ibub(:)
        integer,      intent(in)           :: l(size(ibub))
        integer,      intent(in)           :: m(size(ibub))
        real(REAL64), intent(in)           :: alpha(size(ibub))
        real(REAL64), intent(in), optional :: step
        integer,      intent(in), optional :: nlip
        real(REAL64), intent(in), optional :: thres
        real(REAL64), intent(in), optional :: cutoff

        type(GaussBubblesGenerator)        :: new

        call new%set(label ="Slater Density",&
                     type  =F3D_TYPE_CUSP )
        new%q       = q
        new%center  = center
        new%ibub    = ibub
        new%l       = l
        new%m       = m
        new%alpha   = alpha
        if(present(step))  new%step    = step
        if(present(nlip))  new%nlip    = nlip
        if(present(thres)) new%thres   = thres
        if(present(cutoff))new%cutoff  = cutoff
    end function

    !> Generate set of empty bubbles with adequate parameters.

    !> The Bubbles parameters are generated according to:
    !! . centers: GaussBubblesGenerator::centers
    !! - lmax:    maxval(GaussBubblesGenerator::l)
    !! - z:       GaussBubblesGenerator::z
    !! - grids:   Calling grid_class::grid1dinit_radial, with N0=100 and
    !!   cutoff radius set to 20.0.
    function GaussBubblesGenerator_gen_bubbles(self) result(bubs)
        class(GaussBubblesGenerator), intent(in) :: self
        type(Bubbles)                      :: bubs

        integer :: i
     
        !bubs=Bubbles(&
        !    lmax=maxval(self%l), &
        !    centers=self%center, &
        !    global_centers = self%center, &
        !    grids=[ ( Grid1D(self%q(i), 100, self%nlip, self%cutoff), &
        !                                          i=1,size(self%q) ) ], &
        !    z=self%q )
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
    function GaussBubblesGenerator_gen_bubf(self, ibub, l, m, r) result(f)
        class(GaussBubblesGenerator), intent(in) :: self
        integer,                      intent(in) :: ibub
        integer,                      intent(in) :: l
        integer,                      intent(in) :: m
        real(REAL64),                 intent(in) :: r(:)

        real(REAL64)                       :: f(size(r)), norm
        integer                            :: i

        f=0.d0
        ! Iterate over shells
        do i=1, size(self%alpha)
            norm = (self%alpha(i)/PI)**(1.5d0)
            if((ibub == self%ibub(i)) .and. &
               (l    == self%l(i)   ) .and. &
               (m    == self%m(i)   )        ) then
                f = f + self%q(self%ibub(i))*norm*exp(-self%alpha(i) * r**2)
            end if
        enddo
    end function

    !> Given the grid points \f$\{x_i\}\f$, \f$\{y_j\}\f$, \f$\{z_k\}\f$,
    !! return the three-dimensional array
    !! \f$\{A_{ijk}=
    !!     \sum_p q_p
    !!     \left(\frac{\alpha_p}{\pi}\right)^{\frac{3}{2}}
    !!     \mathrm{e}^{-\alpha_p \left[( x_i-X_p )^2+( y_j-Y_p )^2+( z_i-Z_p )^2 \right]} \f$
    function GaussBubblesGenerator_gen_cubef(self, x, y, z) result(cubef)
        class(GaussBubblesGenerator), intent(in) :: self
        real(REAL64),                 intent(in) :: x(:)
        real(REAL64),                 intent(in) :: y(:)
        real(REAL64),                 intent(in) :: z(:)

        real(REAL64)                             :: cubef(size(x), size(y), size(z))

        real(REAL64)                             :: ex(size(x)), rx(size(x))
        real(REAL64)                             :: ey(size(y)), ry(size(y))
        real(REAL64)                             :: ez(size(z)), rz(size(z))
        
        integer                                  :: i, j, k, p
        real(REAL64)                             :: norm, r

        cubef=0.d0

        do p=1, size(self%center, 2)
            norm = (self%alpha(p)/PI)**(1.5d0)
            rx = x-self%center(X_,p) 
            ry = y-self%center(Y_,p)
            rz = z-self%center(Z_,p)
            ex=exp(-self%alpha(p)*(rx)**2)
            ey=exp(-self%alpha(p)*(ry)**2)
            ez=exp(-self%alpha(p)*(rz)**2)
            do i=1, size(x)
                do j=1, size(y)
                    do k=1, size(z)

                        r = sqrt(rx(i)*rx(i) + ry(j)*ry(j) + rz(k)*rz(k))
                        !print *, i, j, k, r, self%cutoff, r > self%cutoff
                        if (r > self%cutoff) then
                            cubef(i,j,k)=cubef(i,j,k) + self%q(p)*norm*ex(i)*ey(j)*ez(k)

                        end if
                    end do
                end do
            end do
        end do
    end function

    !> Generates a cubegrid adequate for the resulting charge density.

    !> The resulting grid has its limits at least 8 au from all centers.
    !! Adequate padding is added so that the grid exactly has the desired
    !! step and nlip.
    function GaussBubblesGenerator_gen_cubegrid(self) result(cubegrid)
        class(GaussBubblesGenerator), intent(in)  :: self
        type(Grid3D)                              :: cubegrid

        integer      :: ncell(3)
        real(REAL64) :: box(2,3)
        real(REAL64) :: dx(3)
        integer      :: i

        real(REAL64), allocatable :: extent(:)
        real(REAL64), parameter   :: EXT=8.d0

!        extent=log(abs(self%coeffs)*self%expos**3/PI/self%thres)/&
!                    (2.d0*self%expos)
!        extent=[ (maxval(extent, mask=(self%ibub==i) ), i=1,size(self%z) )]
        extent=[ (EXT, i=1,size(self%q) )]

        write(*,*) 'using gridtype in slater-gen, used for cube '
        cubegrid=Grid3D(centers   = self%center, &
                        radii     = extent, &
                        step      = self%step, &
                        nlip      = self%nlip, &
                        grid_type = self%grid_type)
    end function

end module
