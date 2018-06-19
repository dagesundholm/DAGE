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
module Helmholtz3D_class

    use Globals_m
    use Function3D_class
    use Grid_class
    use Bubbles_class
    use potential_class

    use GaussQuad_class
    use Coulomb3D_class
    use Laplacian3D_class

    implicit none

    private

    public    :: Helmholtz3D
    
    !> Does the helmholtz thingie
    type, extends(Operator3D) :: Helmholtz3D
        private
        !> Quadrature used for integration
        type(GaussQuad)                  :: quadrature
        class(Coulomb3D), allocatable    :: coulomb_operator
        real(REAL64)                     :: energy
    contains
        procedure :: operate_on        => Helmholtz3D_operate 
        procedure :: transform_cube    => Helmholtz3D_cube
        procedure :: transform_bubbles => Helmholtz3D_bubbles
        procedure :: destroy           => Helmholtz3D_destroy
        procedure :: set_energy        => Helmholtz3D_set_energy

        procedure :: update            => Helmholtz3D_update
 
    end type

    interface Helmholtz3D
        module procedure :: Helmholtz3D_init
    end interface


contains

    !> Constructor for Helmholtz3D object
    function Helmholtz3D_init(coulomb_operator, energy, quadrature) result(new)
        type(Helmholtz3D)             :: new
        class(Coulomb3D), intent(in)  :: coulomb_operator
        real(REAL64),     intent(in)  :: energy
        !> Gaussian quadrature
        !! If no quadrature is given, then default values used are
        !! nlin=10 and nlog=10. These are the same default values 
        !! used in Coulomb3D
        type(GaussQuad), optional     :: quadrature

        
        
        if(present(quadrature)) then
            new%quadrature = quadrature
        else
            new%quadrature =  GaussQuad()
        endif
        
        allocate(new%coulomb_operator, source = coulomb_operator)
        call new%set_energy(energy)
    end function

    subroutine Helmholtz3D_set_energy(self, energy)
        class(Helmholtz3D), intent(inout)          :: self  
        real(REAL64),       intent(in)             :: energy

        ! Gaussian quadrature weights and tpoints
        real(REAL64), allocatable                  :: weights(:), tpoints(:)
        integer                                    :: i
        
        self%energy = energy

        ! get quadrature weights, and tpoints
        weights = self%quadrature%get_weights()
        tpoints = self%quadrature%get_tpoints()       

        ! multiply quadrature weights with e^[(2 E) / (4 t_i^2)]
        ! this makes the Coulomb3D operator to be a Helmholtz operator
        do i=1,size(tpoints)
            weights(i)=weights(i)*exp(2d0*energy/(4d0*tpoints(i)**2))
        enddo

        ! put the weights to the operator
        call self%coulomb_operator%set_transformation_weights(TWOOVERSQRTPI * weights)

        deallocate(weights)
        deallocate(tpoints)
    end subroutine  

    function Helmholtz3D_operate(self, func, cell_limits, only_cube) result(result_function) 
        class(Helmholtz3D)              :: self 
        !> The input function. Note this should be a multiple of function and
        !! corresponding potential times two
        class(Function3D),  intent(in)  :: func  
        !> Limits in the area in which the operator is applied. NOTE: not used in this operator
        integer, intent(in), optional   :: cell_limits(2, 3)
        !> if only cube is taken into account. NOTE: not used in this operator
        logical, intent(in), optional   :: only_cube

        class(Function3D), allocatable  :: result_function 

        ! Apply the modified coulomb operator
        allocate(result_function, source = self%coulomb_operator .apply. func)
         
        
        ! The result is no longer normalized, thus, we will normalize it
        !allocate(new, source = (result_function * 1d0 / sqrt(result_function .dot. result_function)))
    end function

    !function Helmholtz3D_estimate_energy(self, psi, nuclear_potential) result(E)
    !   class(Helmholtz3D), intent(in)  :: self  
    !   type(Function3D)     :: psi, potential, helper1
    !   type(Function3D)     :: tmp
    !   real(REAL64)         :: E
    !   type(Laplacian3D)    :: laplacian
    !   E = psi .dot. (nuclear_potential * psi)
    !   E=E-0.5d0*(psi .dot. (self%laplacian_operator .apply. psi))
    !end function

    !function estimate_energy2(self, psi) result(E)
    !   class(Helmholtz3D)     :: self
    !   type(Function3D)       :: psi
    !   real(REAL64)           :: E
    !   E= estimate_energy(psi,self%potential)
    !
    !end function

    !function estimate_energy3(psi,potential,psi2) result(E)
    !   type(Function3D)     :: psi, potential, helper1, psi2
    !   class(Function3D), allocatable :: helper2
    !   real(REAL64)         :: E
    !   type(Laplacian3D)    :: laplacian
    !   E=psi.dot.(potential*psi2)
    !   laplacian=Laplacian3D(psi%grid)
    !   allocate(helper2, source=laplacian .apply. psi2)
    !    E=E-0.5d0*(psi.dot.(laplacian.apply.psi))
    !   helper1=psi
    !  helper1%bubbles=helper2%bubbles
    !   call helper1%set_cube(helper2%get_cube())
    !   call helper1%set_type(psi%get_type())
    !   E=E-0.5d0*(psi.dot.helper1)
    !   deallocate(helper2)
    !end function




! testing
! these will be needing actual content

    function Helmholtz3D_cube(self, cubein) result(cubeout)
       class(Helmholtz3D), intent(in) :: self
       real(REAL64), intent(in)      :: cubein(:,:,:)
       real(REAL64),allocatable      :: cubeout(:,:,:)
     
       cubeout=cubein
       return
    end function

    function Helmholtz3D_bubbles(self, bubsin) result(new)
       class(Helmholtz3D), intent(in) :: self
       type(Bubbles), intent(in)     :: bubsin
       type(Bubbles)                 :: new

       new=bubsin
       return
    end function


!> Does an update cycle with the Helmholtz kernel
    function Helmholtz3D_update(self, phi, N0) result(phiout)
       class(Helmholtz3D), intent(in)  :: self
       type(Function3D), intent(in)    :: phi
       type(Function3D)                :: phiout
       integer,optional                :: N0
       integer                         :: N,j
       type(Function3D),allocatable    :: helper(:)
       real(REAL64), allocatable       :: f(:,:),s(:,:),c(:,:)

! checking if number of trial functions is given
       if(present(N0)) then
          N=N0
       else
          N=2
       endif
! allocating the helper table of functions       
       allocate(helper(N))
! allocating the f and s matrices and the coefficient matrix c
       allocate(f(N,N))
       allocate(s(N,N))
       allocate(c(N,N))

       !helper(1)=phi
       !call helper(1)%set_type(2)
       !do j=2,N
! operate with the helmholtz kernel
       !     helper(j)= Helmholtz3D_operate(self, helper(j-1))
       !end do

! fill the matrices
       !f=h_mat(helper,self%potential)
       !s=s_ab(helper)
! calculate the coefficients
       !c=gen_ev(f,s)

! collecting the total wave function
       !phiout=c(1,N)*helper(1)
       !do j=2,N
       !     phiout=phiout+c(j,N)*helper(j)
       !end do
! normalization
       !phiout=phiout*1d0/sqrt(phiout.dot.phiout)
       !call phiout%set_type(phi%get_type())
!write(*,*) c

    end function



!    function auxiliary_density(E,V,phi) result(rho)
    ! E is the guess for the energy eigenvalue
!    real(REAL64)           :: E
    ! V is the potential, phi is the guess for the wave function
    ! rho is the auxiliary density
!    type(Function3D)       :: V, phi, rho
!    rho=E*phi
!    rho=rho-V*phi
!    rho=rho/(2*pi)
!    end function

!    function new_wave_function(rho) result(wf)
!    class(Function3D)      :: rho, wf
!    class(Coulomb3D)       :: op

!    op=Coulomb3D(rho%grid)
!    wf=op.apply.rho
!    end function

! todo: figure out what are the optimal settings for the quadrature
! same as new_wave_function except that a quadrature is also given
!   function new_wave_function_quad(rho,kvad) result(wf)
!    type(Function3D)      :: rho, wf
!    type(Coulomb3D)       :: op
!    type(GaussQuad)       :: kvad
!    op=Coulomb3D(rho%grid,rho%grid,kvad)
!    wf=op.apply.rho
!   end function 




! hamiltonin operaattorin matriisiesitys

    function h_mat(phi, V) result(mat)
    ! list of trial wavefunctions
    type(Function3D)     :: phi(:)
    type(Function3D),allocatable     :: phi2(:)
    type(Function3D), allocatable    :: phi3(:)
    ! potential as Function3D object
    type(Function3D), intent(in)                 :: V
    type(Laplacian3D)                :: op
    integer                          :: i,j,k
    real(REAL64), allocatable        :: mat(:,:)

    op=Laplacian3D(V%grid)
    j=size(phi)

 
    allocate(mat(j,j))
    mat(:,:)=0d0

    do i=1,j
      do k=1,j
       !mat(i,k)=estimate_energy3(phi(i),V,phi(k))
      end do
    end do

    end function

! peittointegraalit
! S is symmetric, and has ones on diagonal
  function s_ab(phi) result(S)
    type(Function3D), intent(in)    :: phi(:)
    integer                          :: i,j,k
    real(REAL64), allocatable        :: S(:,:)
    j=size(phi)
    allocate(S(j,j))
    do i=1,j
      do k=1,j
      S(i,k)=phi(i).dot.phi(k)
      end do
    end do

  end function


  function gen_ev(F,S) result (C)
! todo: lue lapackin dggev-dokumentaatiota
  real(REAL64)               :: F(:,:), S(:,:)
  character(len=1),parameter :: JOBVL='N', JOBVR='V'
  integer                    :: N, LDA, LDB, LDVL, LDVR, LWORK, INFO
  integer, allocatable       :: t2(:)
  real(REAL64), allocatable  :: alphar(:),alphai(:),beta(:), lambda(:), temp(:)
  real(REAL64), allocatable  :: VL(:,:), WORK_ev(:), C(:,:)
! size of matrices
  N=size(F(:,1))
!  parameters for the dggev subroutine of lapack
  LDA=N
  LDB=N
  LDVL=N
  LDVR=N

! 
  LWORK=100
  allocate(WORK_ev(LWORK))
  allocate(alphar(N))
  allocate(alphai(N))
  allocate(beta(N))
  allocate(lambda(N))
  allocate(temp(N))
  allocate(t2(N))
  allocate(VL(N,N))
  allocate(C(N,N))
  
!  calling the lapack subroutine
  call dggev(JOBVL, JOBVR, N, F, LDA, S, LDB, alphar, alphai,beta, VL, LDVL, &
             C, LDVR, WORK_ev, LWORK, INFO)

! generalized eigenvalues are in lambda
  lambda=alphar/beta
! location of the smallest eigenvalue
  t2=minloc(lambda)
! putting the eigenvector of the lowest eigenvalue to the right of the vector
  temp=C(:,N)
  C(:,N)=C(:,t2(1))
  C(:,t2(1))=temp

  ! TODO: figure out why this is needed
  C=-C


  end function

!> destructor for Helmholtz3D object
    subroutine Helmholtz3D_destroy(self)
        class(Helmholtz3D)  :: self
        call self%quadrature%destroy()
        call self%coulomb_operator%destroy()
    end subroutine 



end module
