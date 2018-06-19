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
! this class is an alternative implementation of an rhf cycle
module alt_RHFCycle_class
    use Function3D_class
    use Function3D_types_m
    use Helmholtz3D_class
    use Laplacian3D_class
    use Globals_m
    use xmatrix_m
    use GaussQuad_class
    use Coulomb3D_class

    implicit none

!    public              :: alt_RHFCycle

!    private

    type                :: alt_RHFCycle
        type(Function3D),public, allocatable     :: orbitals(:)
        real(REAL64),public, allocatable         :: eigenvalues(:)
        integer                                  :: nocc

! electon potential
        class(Function3D), allocatable           :: Ve
! nuclear potential
        type(Function3D)                         :: Vn 
! exchange potentials K \phi_i
        type(Function3D), allocatable            :: K(:)
! helmholtz operators
        type(Helmholtz3D), allocatable           :: helmholtz_operators(:)        
! electron density
        type(Function3D )                         :: electron_density
! laplacian operator for kinetic energy
        class(Laplacian3D), allocatable          :: laplacian_operator
! coulomb operator
        class(Coulomb3D), allocatable            :: coulomb_operator


    contains
        procedure, public    :: exchange_functions    
        procedure, public    :: run => alt_rhfcycle_run
        procedure, public    :: get_rho => get_rhf_rho

    end type

    interface alt_RHFCycle
        module procedure :: alt_RHFCycle_init
    end interface

contains


    function alt_RHFCycle_init(orbitals, nocc) result(new)
        type(Function3D), intent(in)        :: orbitals(:)
        integer, intent(in)                 :: nocc
        type(alt_RHFCycle)                  :: new
        integer                             :: i
        real(REAL64), pointer               :: vals(:,:)


        new%orbitals = orbitals
        new%nocc     = nocc
        allocate(new%laplacian_operator,source=Laplacian3D(new%orbitals(1)%grid)) 
! default values to eigenvalues
        allocate(new%eigenvalues(size(orbitals)))
        new%eigenvalues=-1d0
        allocate(new%coulomb_operator, source=Coulomb3D(new%orbitals(1)%grid))
! allocate electron density
        new%electron_density=Function3D(orbitals(1),type=F3D_TYPE_CUSP)
        new%electron_density=0d0
        do i=1,nocc
            new%electron_density=new%electron_density+2d0*(orbitals(i)*orbitals(i))
        end do
        call new%coulomb_operator%operate_on(new%electron_density,new%Ve)
! initialize nuclear potential, note that Vn lmax=0
        new%Vn=Function3D(orbitals(1)%grid, Bubbles(0,& 
orbitals(1)%bubbles%get_centers(), orbitals(1)%bubbles%get_grid(),&
orbitals(1)%bubbles%get_z(), -1  ) )
        new%Vn = 0d0
        call new%Vn%set_type(F3D_TYPE_NUCP)

        do i=1, orbitals(1)%bubbles%get_nbub()
            vals => new%Vn%bubbles%get_f(i)
            vals = -1d0
        end do        

! allocating helmholtz operators
        allocate(new%helmholtz_operators(size(orbitals)))
        do i=1, size(orbitals)
            new%helmholtz_operators(i) = Helmholtz3D(new%coulomb_operator,-1d0)
        end do

        allocate(new%K(size(orbitals)))



        nullify(vals)

    end function

! function for getting the K \phi s as Function3D objects 
! this scales as ntot x nocc
    function exchange_functions(self) result(ex)
        class(alt_RHFcycle), intent(in) :: self
        integer                         :: i,j
        type(Function3D), allocatable   :: ex(:) 
        class(Function3D), allocatable  :: pot

        ex=self%orbitals
        do i=1, size(self%orbitals)
            ex(i)=0d0
            do j=1, self%nocc
                call self%coulomb_operator%operate_on(&
self%orbitals(i)*self%orbitals(j), pot)
                ex(i) = ex(i) + (self%orbitals(j)*pot)
            end do

        end do

        deallocate(pot)

    end function

    subroutine alt_rhfcycle_run(self)
        class(alt_RHFCycle), intent(inout)  :: self
        integer                             :: i, j
        real(REAL64)                        :: energy

! update electron density and potential by it
        self%electron_density=0d0
        do i=1,self%nocc
            self%electron_density = self%electron_density +&
2d0* (self%orbitals(i)*self%orbitals(i) )
        end do        
        call self%coulomb_operator%operate_on(self%electron_density,self%Ve) 

! evaluate exchanges
        self%K = self%exchange_functions()

! do orbital updates
        do i=1, size(self%orbitals)
write(*,*) i, size(self%orbitals)
            do j=1,5
write(*,*) 'j', j
! test if this works
self%orbitals(i) = self%helmholtz_operators(i).apply.(self%get_rho(&
self%orbitals(i),self%K(i)))

! normalization
self%orbitals(i)=self%orbitals(i)/sqrt(self%orbitals(i) .dot. self%orbitals(i))

! estimate energy
energy = (-0.5d0)*((self%orbitals(i)).dot.(self%laplacian_operator.apply.&
self%orbitals(i))) - ( self%orbitals(i) .dot. self%K(i) ) + &
(self%orbitals(i) .dot. (self%orbitals(i)*self%Ve) ) + &
(self%orbitals(i) .dot. (self%orbitals(i)*self%Vn) )

write(*,*) energy
if (energy > -0d0) energy=-1d0
call self%helmholtz_operators(i)%set_energy(energy)
self%eigenvalues(i) = energy
            end do

write(*,*) 'orbital energy', self%eigenvalues(i)
        end do

! write orbital energies
write(*,*) self%eigenvalues
    end subroutine

    function get_rhf_rho(self, phi_in, K) result(rho)
        class(alt_RHFCycle)       :: self
        type(Function3D)          :: phi_in
        type(Function3D)          :: K
        type(Function3D)          :: rho

        rho = (-1d0/(2*pi)) * ((self%Vn+self%Ve)*phi_in) + (1d0/(2*pi)) *K


    end function

! S
    function get_trial_overlap(self,phi1,phi2) result(S)
        class(alt_RHFCycle)        :: self
        type(Function3D)           :: phi1,phi2
        real(REAL64), allocatable  :: S(:,:)
        allocate(S(2,2))
        S(1,1)=phi1.dot.phi1
        S(2,2)=phi2.dot.phi2
        S(1,2)=phi1.dot.phi2
        S(2,1)=S(1,2)   
    end function

! F
    function get_trial_fock(self, phi1,phi2,K) result(F)
        class(alt_RHFCycle)        :: self
        type(Function3D)           :: phi1,phi2,K
        real(REAL64), allocatable  :: F(:,:)
        allocate(F(2,2))
F(1,1)= (-0.5d0)*((phi1.dot.(self%laplacian_operator.apply.&
phi1)) ) - ( phi1 .dot. K ) + &
(phi1 .dot. (self%Ve*phi1) ) + &
(phi1 .dot. (phi1*self%Vn) )
F(2,2)= (-0.5d0)*((phi2.dot.(self%laplacian_operator.apply.&
phi2)) ) - ( phi2 .dot. K ) + &
(phi2 .dot. (self%Ve*phi2) ) + &
(phi2 .dot. (phi2*self%Vn) )
F(1,2)= (-0.5d0)*((phi1.dot.(self%laplacian_operator.apply.&
phi2)) ) - ( phi1 .dot. K ) + &
(phi1 .dot. (self%Ve*phi2) ) + &
(phi1 .dot. (phi2*self%Vn) )
! note that this assumes that F is symmetric
F(2,1)=F(1,2)


    end function


end module




