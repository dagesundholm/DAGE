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
module initial_guess_m
    use Globals_m
    use xmatrix_m
    use Function3D_class
    use Bubbles_class
    use Grid_class
    use Function3D_types_m
    use LCAO_m
    use ParallelInfo_class

    implicit none

    
    !> gaussian lobe functions
    !! g(x,y,z) = c exp(-alpha ((x-Rx)**2+(y-Ry)**2+(z-Rz)**2))
    !GaussianLobe
    type   GaussianLobe
        private
        !> 
        real(REAL64)    :: alpha
        real(REAL64)    :: center(3)
        real(REAL64)    :: c

    end type

    type LobeBasis
        private
        type(GaussianLobe), allocatable     :: lobes(:)
        real(REAL64), allocatable           :: charges(:)
        real(REAL64), allocatable           :: positions(:,:)
        !  how many basis functions there are on atoms
        integer, allocatable                :: functions_on_atom(:)
    contains
        procedure                           :: calculate_overlap_matrix &
                                               => LobeBasis_calculate_overlap_matrix
        procedure                           :: calculate_two_electron_integrals& 
                                               =>LobeBasis_calculate_two_electron_integrals
        procedure                           :: calculate_hamiltonian_matrix&
                                               =>LobeBasis_calculate_hamiltonian_matrix
    end type

    interface LobeBasis
        module procedure :: LobeBasis_init
        module procedure :: LobeBasis_init_from_basis_object
    end interface

    !> cartesian type gaussian function
    !! g(x,y,z) = c (x-Rx)**lx (y-Ry)**ly (z-Rz)**lz 
    !!              * exp(-alpha ((x-Rx)**2+(y-Ry)**2+(z-Rz)**2))
    type AngularGaussian
        private
        integer         :: lx
        integer         :: ly
        integer         :: lz
        real(REAL64)    :: alpha
        real(REAL64)    :: center(3)
        real(REAL64)    :: c
    end type

    !> S_lm *exp(-alpha r^2)
    type SolidGaussian
        private
        integer                               :: l
        integer                               :: m
        real(REAL64)                          :: alpha
        real(REAL64)                          :: center(3)
        type(AngularGaussian), allocatable    :: primitives(:)
        real(REAL64), allocatable             :: coeffs(:)
    end type

    ! basis set of SolidGaussians
    type SolidGaussianBasis
        private
        type(SolidGaussian), allocatable     :: functions(:)
        real(REAL64), allocatable            :: charges(:)
        real(REAL64), allocatable            :: positions(:,:)
        !  how many basis functions there are on atoms
        integer, allocatable                 :: functions_on_atom(:)
    contains
        procedure                           :: calculate_overlap_matrix &
                                               => SolidGaussianBasis_calculate_overlap_matrix
        procedure                           :: calculate_two_electron_integrals& 
                                               =>SolidGaussianBasis_calculate_two_electron_integrals
        procedure                           :: calculate_hamiltonian_matrix&
                                               =>SolidGaussianBasis_calculate_hamiltonian_matrix
        procedure                           :: normalize &
                                               => SolidGaussianBasis_normalize
    end type

    interface SolidGaussianBasis
        module procedure :: SolidGaussianBasis_init_from_basis_object
    end interface

contains
    ! prodcut of s gaussians
    pure function GaussianLobe_product(a,b) result(c)
        type(GaussianLobe), intent(in)       :: a,b
        type(GaussianLobe)                   :: c

        ! prefactor
        c%c = a%c * b%c * exp((-a%alpha*b%alpha)/(a%alpha+b%alpha)*&
            dot_product(a%center-b%center,a%center-b%center))
        ! exponent
        c%alpha = a%alpha + b%alpha
        ! center
        c%center = (a%alpha*a%center+b%alpha*b%center)/(a%alpha+b%alpha)
    end function

    !> overlap between two lobes S_ab
    function GaussianLobe_overlap(a,b) result(overlap)
        type(GaussianLobe)       :: a,b
        real(REAL64)              :: overlap

        overlap = a%c*b%c * exp((-a%alpha*b%alpha)/(a%alpha+b%alpha)*&
            dot_product(a%center-b%center,a%center-b%center))*(pi/(a%alpha+b%alpha))**1.5d0
    end function

    !> kinetic matrix element T_ab
    function GaussianLobe_kinetic_element(a,b) result(kinetic)
        !> Input GaussianLobe types
        type(GaussianLobe), intent(in)      :: a,b
        real(REAL64)                        :: kinetic
        ! formula derived with mathematica
        kinetic = -0.5d0*a%c*b%c/(a%alpha+b%alpha)**3.5d0 * &
            2*a%alpha*b%alpha *pi**1.5d0 * exp(-a%alpha*b%alpha* &
            (dot_product(a%center,a%center)+dot_product(b%center,b%center)-&
            2d0*a%center(1)*b%center(1)-2d0*a%center(2)*b%center(2)-&
            2d0*a%center(3)*b%center(3))/(a%alpha+b%alpha)) * (-3d0*b%alpha +&
            a%alpha*(-3d0+2*b%alpha*(dot_product(a%center,a%center)+dot_product(b%center,b%center)-&
            2d0*a%center(1)*b%center(1)-2d0*a%center(2)*b%center(2)-&
            2d0*a%center(3)*b%center(3))))
    end function

    ! nuclear matrix element
    function GaussianLobe_nuclear_element(a,b,charges,positions) result(vnab)
        type(GaussianLobe), intent(in) :: a,b
        real(REAL64)                   :: charges(:)
        real(REAL64)                   :: positions(:,:)
        real(REAL64)                   :: vnab
        integer                        :: i
        real(REAL64)                   :: t, temp
        type(GaussianLobe)             :: c
        integer                        :: Natom
        real(REAL64)                   :: nucpos(3)
        real(REAL64)                   :: nucchar

        ! initialize things
        vnab=0d0
        c =  GaussianLobe_product(a,b)
        ! shape(Z) = [4,Natom]        
        Natom = size(positions,2)
        do i=1,Natom
            nucchar = charges(i)
            nucpos = positions(:,i)
            t = (a%alpha+b%alpha)*dot_product(c%center-nucpos,c%center-nucpos)
            t = abs(t)
            if(t==0) then
                temp=1d0
            else
                temp = 0.5d0*(pi/t)**0.5d0 *erf(sqrt(t))
            end if            
            vnab = vnab  -2d0*pi*nucchar/(a%alpha+b%alpha)*temp* &
                exp((-a%alpha*b%alpha)/(a%alpha+b%alpha)*&
                dot_product(a%center-b%center,a%center-b%center))
        end do
    end function

    !> two electron integrals from Szabo (A.41)
    pure function two_electron_lobe(a,b,c,d) result(res)
        type(GaussianLobe),intent(in)      :: a,b,c,d
        real(REAL64)                       :: res
        real(REAL64)                       :: t,temp
        type(GaussianLobe)                 :: n1,n2

        n1 = GaussianLobe_product(a,b)
        n2 = GaussianLobe_product(c,d)

        t=((a%alpha+b%alpha)*(c%alpha+d%alpha)*&
            dot_product(n1%center-n2%center,n1%center-n2%center))/&
            (a%alpha+b%alpha+c%alpha+d%alpha)
        if(t==0) then
            temp=1d0
        else
            temp = 0.5d0*(pi/t)**0.5d0 *erf(sqrt(t))
        end if     

        res = temp*2d0*pi**2.5d0/&
            ((a%alpha+b%alpha)*(c%alpha+d%alpha)*(a%alpha+b%alpha+c%alpha+d%alpha))* &
            exp((-a%alpha*b%alpha)/(a%alpha+b%alpha)*&
            dot_product(a%center-b%center,a%center-b%center))* &
            exp((-c%alpha*d%alpha)/(c%alpha+d%alpha)*&
            dot_product(c%center-d%center,c%center-d%center))

    end function

    !> Construct a LobeBasis object based on atom coordinates
    !> and respective charges
    function LobeBasis_init(positions, charges) result (new_basis)
        real(REAL64), intent(in) :: positions(:,:)
        real(REAL64), intent(in) :: charges(:)
        type(LobeBasis)          :: new_basis
        integer                  :: i,j,k, Nbasis, Natoms
        ! element Z has Z+4 lobes associated with it
        Natoms=size(charges)
        Nbasis=FLOOR(sum(charges(:))) + 4*Natoms
        allocate(new_basis%lobes(Nbasis))
        allocate(new_basis%functions_on_atom(Natoms))
       ! go through atoms
       ! TODO: think about finding better exponents
        k=1
        do i=1,Natoms
            ! store the number of basis functions related to atom i
            ! there are Z+4 functions on atom with charge Z
            new_basis%functions_on_atom(i) = floor(charges(i))+4

            new_basis%lobes(k)=GaussianLobe(10d0*charges(i),positions(:,i),1d0)
            k=k+1

            ! test if these larger exponents help
            new_basis%lobes(k)=GaussianLobe(1d2*charges(i),positions(:,i),1d0)
            k=k+1
            new_basis%lobes(k)=GaussianLobe(1d3*charges(i),positions(:,i),1d0)
            k=k+1


            do j=FLOOR(charges(i)),1,-1
                new_basis%lobes(k)=GaussianLobe(j+0d0,positions(:,i),1d0)
                k=k+1
            end do
            ! last lobe
            new_basis%lobes(k)=GaussianLobe(0.1d0,positions(:,i),1d0)
            k=k+1 
        end do

        ! save the atom coordinates
        new_basis%charges = charges
        new_basis%positions     = positions

    end function


    function LobeBasis_init_from_basis_object(positions, charges, basis_object) result (new_basis)
        type(Basis), intent(in)                :: basis_object
        real(REAL64), intent(in)               :: positions(:,:)
        real(REAL64), intent(in)               :: charges(:)
        type(LobeBasis)                        :: new_basis
        integer                                :: i,j,k, Natoms
        real(REAL64), allocatable              :: expos(:)
        real(REAL64), allocatable              :: coeffs(:)
        integer                                :: Nfunctions
! test variables
        integer, allocatable                   :: nonzeros(:)
        integer, allocatable                   :: finfo(:,:)
! this is test ground
        print *, 'basis set type', basis_object%basis_set_type
        print *, 'number of subshells', basis_object%number_of_subshells
        print *, 'number of shells', basis_object%number_of_shells
        print *, 'basis_object%size', basis_object%size
        print *, 'basis_object%atom_type', basis_object%atom_type
        print *, 'local number of subshells', basis_object%local_number_of_subshells
        print *, 'size of exponentials', size(basis_object%exponentials,1),size(basis_object%exponentials,2)
        print *, 'l of subshells', basis_object%l

        print *, basis_object%exponentials(:,1)
        print *, basis_object%exponentials(:,2)
        expos= pack(basis_object%exponentials,basis_object%exponentials/=0d0)
        coeffs= pack(basis_object%coefficients,basis_object%exponentials/=0d0)
        print *,'exponentials', expos
        print *,'coefficients', coeffs

        print *, 'size of expos', size(expos)

        print *, 'tarkastellaan exponentteja'
        allocate(nonzeros(basis_object%number_of_subshells))
        allocate(finfo(basis_object%number_of_subshells,3))
        do i=1, basis_object%number_of_subshells
            nonzeros(i) = count(basis_object%exponentials(:,i)/=0d0)
            print *, 'eksponentteja',count(basis_object%exponentials(:,i)/=0d0)
            print *, i,'eksonentit', basis_object%exponentials(1:nonzeros(i),i)
            ! how many exponentials there are on this subshell
            finfo(i,1)=nonzeros(i)
            ! subshell l
            finfo(i,2)=basis_object%l(i)
            ! number of functions on subshell
            finfo(i,3)=(2*finfo(i,2)+1)*finfo(i,1)
        end do
        print *, 'funktioita kaiken kaikkiaan', sum(finfo(:,3))        

        Natoms = size(charges)
        Nfunctions = size(expos)
        k=1
        ! allocate lobes
        allocate(new_basis%lobes(Nfunctions*Natoms))
        ! allocate 
        allocate(new_basis%functions_on_atom(Natoms))
        do i=1,Natoms
            new_basis%functions_on_atom(i) = Nfunctions
            do j=1,Nfunctions
                 new_basis%lobes(k)=GaussianLobe(expos(j),positions(:,i),1d0)
                 k=k+1
            end do        
        end do
        new_basis%charges   = charges
        new_basis%positions = positions

        !new_basis = LobeBasis_init(positions, charges)
    end function

    !> calculating overlap matrix of a basis
    function LobeBasis_calculate_overlap_matrix(self) result(overlap)
        class(LobeBasis), intent(in)        :: self
        real(REAL64),allocatable            :: overlap(:,:)
        integer                             :: i,j, Nbasis

        Nbasis = size(self%lobes)
        allocate(overlap(Nbasis,Nbasis))
        overlap = 0d0

        do i=1,Nbasis
            do j=1,i
                overlap(i,j) = GaussianLobe_overlap(self%lobes(i), self%lobes(j))
                overlap(j,i) = overlap(i,j)
            end do
        end do

    end function

    !> calculating hamiltonian matrix of basis
    function LobeBasis_calculate_hamiltonian_matrix(self) result(hamiltonian)
        class(LobeBasis), intent(in)       :: self
        real(REAL64),allocatable           :: hamiltonian(:,:)
        integer                            :: i,j, Nbasis

        Nbasis = size(self%lobes)
        allocate(hamiltonian(Nbasis,Nbasis))
        hamiltonian = 0d0

        do i=1,Nbasis
            do j=1,i
                hamiltonian(i,j) = hamiltonian(i,j)+&
                    GaussianLobe_kinetic_element(self%lobes(i),self%lobes(j))
                hamiltonian(i,j) = hamiltonian(i,j)+&
                    GaussianLobe_nuclear_element(self%lobes(i),self%lobes(j),self%charges,self%positions)

                hamiltonian(j,i) = hamiltonian(i,j)
            end do
        end do


    end function

    !> basis two-electron functions
    function LobeBasis_calculate_two_electron_integrals(self) result(res)
        class(LobeBasis), intent(in)         :: self
        real(REAL64),allocatable             :: res(:,:,:,:)
        integer                              :: i,j,k,l, Nbasis

        Nbasis = size(self%lobes)
        allocate(res(Nbasis,Nbasis,Nbasis,Nbasis))
        res=0d0

        forall (i = 1:Nbasis, j = 1:Nbasis, k = 1:Nbasis, l = 1:Nbasis) 
            res(i,j,k,l) = two_electron_lobe(self%lobes(i),self%lobes(j),&
                self%lobes(k),self%lobes(l))
        end forall

    end function


    !> initial guess is a C matrix of form [I_Nocc;0]
    function LobeBasis_make_coefficients (self, Nocc) result(coefficients)
        class(LobeBasis)               :: self
        integer                        :: Nocc
        integer                        :: i, Nbasis
        real(REAL64), allocatable      :: coefficients(:,:)

        Nbasis = size(self%lobes)
        allocate(coefficients(Nbasis,Nbasis))
        coefficients = 0d0

        do i=1,Nbasis
            coefficients(i,i) = 1d0
        end do
    end function

    !> normalize orbitals
    function normalize_orbitals (old_coefficients, overlap_matrix) result(new_coefficients)
        real(REAL64), allocatable       :: old_coefficients(:,:) ,new_coefficients(:,:)
        real(REAL64), allocatable       :: overlap_matrix(:,:)
        integer                         :: i, Nbasis, Norb
        integer                         :: j,k
        real(REAL64)                    :: sum_    

        Nbasis = size(old_coefficients, 1)
        Norb   = size(old_coefficients,2)

        new_coefficients = 0d0*old_coefficients

        do i=1,Norb
            sum_=0d0
            do j=1,Nbasis
                do k=1,Nbasis
                    sum_ = sum_ + old_coefficients(j,i)*old_coefficients(k,i)*overlap_matrix(j,k)
                end do
            end do
            ! normalize
            new_coefficients(:,i) = old_coefficients(:,i)/sqrt(sum_)
!print *, 'old norm', sum_
        end do
    end function

    !> calculate density matrix from coefficient matrix
    function density_matrix_from_coefficients(coefficients,Nocc)result(density_matrix)
        real(REAL64), intent(in)            :: coefficients(:,:)
        real(REAL64), allocatable           :: density_matrix(:,:)
        integer, optional                   :: Nocc
        integer                             :: i,j,k, N, Nbasis
  
        if(present(Nocc)) then
            N = Nocc
        else
            N= size(coefficients,2)
        end if
        ! allocate Nbasis x Nbasis matrix
        Nbasis = size(coefficients,1)
        allocate(density_matrix(Nbasis,Nbasis))
        density_matrix = 0d0

       ! go through occupied orbitals
        do i=1,N
            do j=1,Nbasis
                do k=1,Nbasis
                    density_matrix(j,k) = density_matrix(j,k)&
                        +2d0*coefficients(j,i)*coefficients(k,i)    
                        !+coefficients(j,i)*coefficients(k,i)    
                end do
            end do
        end do

    end function

    !> calculate fock matrix given one particle Hamiltonian H, two-electron
    !! integral tensor G(:,:,:,:) and density matrix D
    function lobe_fock_matrix(hamiltonian_matrix,two_electron_integrals,density_matrix)&
        result(fock_matrix)
        real(REAL64), intent(in)     :: hamiltonian_matrix(:,:), &
                                        two_electron_integrals(:,:,:,:), &
                                        density_matrix(:,:)
        real(REAL64), allocatable    :: fock_matrix(:,:)
        integer                      :: a,b,i,j, Nbasis
        ! initialize fock matrix with one particle hamiltonian
        fock_matrix = 1d0*hamiltonian_matrix
        !fock_matrix = 0.5d0*hamiltonian_matrix
        Nbasis = size(hamiltonian_matrix,2)
        do a=1,Nbasis
            do b=1,Nbasis
                do i=1,Nbasis
                    do j=1,Nbasis
                        fock_matrix(a,b) = fock_matrix(a,b)&
                            + density_matrix(i,j)&
!                            * (two_electron_integrals(a,b,i,j)  -two_electron_integrals(a,j,i,b))
                            * (two_electron_integrals(a,b,i,j)- 0.5d0*two_electron_integrals(a,j,i,b))
!                            * (two_electron_integrals(a,b,i,j) - 0.5d0*two_electron_integrals(a,i,j,b))
!                            * (two_electron_integrals(a,b,i,j)-two_electron_integrals(a,j,b,i))
!                            * (two_electron_integrals(a,b,i,j)-0.5d0*two_electron_integrals(a,j,b,i))
!                            * (two_electron_integrals(a,b,i,j)-0.5d0*two_electron_integrals(a,i,j,b))
!                            * ( two_electron_integrals(a,b,j,i) -0.5d0* two_electron_integrals( a,i,j,b  ) )
!print *,a,b,i,j, ( two_electron_integrals(a,b,i,j) - two_electron_integrals( a,j,i,b  ) )

                    end do
                end do
            end do
        end do        

    end function

    !   nuclear repulsion
    function LobeBasis_nuclear_repulsion(self) result(res)
        class(LobeBasis)         :: self
        real(REAL64)              :: res
        integer                   :: i,j, Natom

        res=0d0
        Natom = size(self%positions,2)
        ! loop over atoms       
        do i=1,Natom-1
            do j=i+1,Natom
                res = res + self%charges(i)*self%charges(j) &
                    /sqrt(dot_product(self%positions(:,i)-self%positions(:,j),&
                                      self%positions(:,i)-self%positions(:,j)))
            end do
        end do 
     
    end function

    ! calculate total energy
    function LobeBasis_total_energy(density_matrix,hamiltonian_matrix, &
                                    two_electron_integrals,nuclear_repulsion) result(res)
        real(REAL64), intent(in)        :: density_matrix(:,:), &
                                           hamiltonian_matrix(:,:), &
                                           two_electron_integrals(:,:,:,:)
        real(REAL64)                    :: nuclear_repulsion,res
        integer                         :: a,b,c,d,N

        res = 2d0*nuclear_repulsion
        N = size(density_matrix,1)
        res = res + sum(density_matrix*hamiltonian_matrix)
        do a=1,N
            do b=1,N
                do c=1,N
                    do d=1,N
                        res = res + 0.5d0*((density_matrix(a,b)* density_matrix(c,d))&
                              *(two_electron_integrals(a,b,c,d)-two_electron_integrals(a,d,c,b)))
                    end do
                end do
            end do
        end do

    end function




    ! SCF iterations
    function lobe_scf(lobe_basis,nocc,imaxin) result(coefficients)
        type(LobeBasis), intent(in)  :: lobe_basis
        integer, intent(in)          :: nocc 
        integer,optional             :: imaxin
        real(REAL64), allocatable    :: coefficients(:,:)
        integer                      :: imax
        ! matrices
        real(REAL64), allocatable    :: overlap_matrix(:,:), &
                                        hamiltonian_matrix(:,:), &
                                        two_electron_integrals(:,:,:,:), &
                                        fock_matrix(:,:)
        real(REAL64), allocatable    :: density_matrix(:,:), &
                                        orthogonalizing_matrix(:,:), &
                                        eigen_vectors(:,:),eigen_values(:)
        ! energy stuff
        real(REAL64), allocatable    :: energies(:), old_energies(:) 
        integer                      :: i,N


        if(present(imaxin)) then
            imax = imaxin
        else
            ! 10 iterations should be enough
            imax = 10
        end if

        ! init
        overlap_matrix = lobe_basis%calculate_overlap_matrix()
        hamiltonian_matrix = lobe_basis%calculate_hamiltonian_matrix()
        two_electron_integrals = lobe_basis%calculate_two_electron_integrals()

        N = size(hamiltonian_matrix,1)
        ! initialize C with core hamiltonian
        allocate(coefficients(N,N))
        call matrix_eigensolver(hamiltonian_matrix,energies,coefficients)
        coefficients = normalize_orbitals(coefficients,overlap_matrix)

        density_matrix = 0d0*coefficients
        fock_matrix = hamiltonian_matrix

        call matrix_eigensolver(overlap_matrix, eigen_values, eigen_vectors)
        orthogonalizing_matrix = 0d0*overlap_matrix
        do i=1,size(overlap_matrix,1)
            orthogonalizing_matrix(i,i) = 1d0/sqrt(eigen_values(i))
        end do
        orthogonalizing_matrix = xmatmul(eigen_vectors, &
            xmatmul(orthogonalizing_matrix,transpose(eigen_vectors)))
        fock_matrix = xmatmul(transpose(orthogonalizing_matrix), &
            xmatmul(fock_matrix,orthogonalizing_matrix))
        call matrix_eigensolver(fock_matrix,energies,coefficients)
        coefficients = xmatmul(orthogonalizing_matrix,coefficients)
        coefficients = normalize_orbitals(coefficients,overlap_matrix)



        ! iterate
        i = 0
        old_energies=energies+1
        ! millihartree convergence is enough
        do while (i<imax .AND. maxval(abs(energies(1:nocc)-old_energies(1:nocc)))>1d-3  )
            old_energies=energies
            density_matrix = density_matrix_from_coefficients(coefficients,nocc)
            fock_matrix =lobe_fock_matrix(hamiltonian_matrix, &
                                          two_electron_integrals, &
                                          density_matrix)
            fock_matrix = xmatmul(transpose(orthogonalizing_matrix), &
                                  xmatmul(fock_matrix,orthogonalizing_matrix))
            call matrix_eigensolver(fock_matrix,energies,coefficients)
            coefficients = xmatmul(orthogonalizing_matrix,coefficients)
            i=i+1

            print *, 'orbital energy differences',i , energies(1:nocc)-old_energies(1:nocc)
            print *, 'orbital energies', energies(1:nocc)
        end do
        
        print *, 'number of iterations done ',i

    end function



    ! mos with structure_object and mould
    function lobe_mos_from_structure(structure_object,mould, norbitals,nocc, basis_object) result(orbitals)
        type(Structure), intent(in)         :: structure_object
        type(Function3D), intent(in)        :: mould
        integer, intent(in)                 :: nocc
        type(Basis), optional               :: basis_object
        type(Function3D), allocatable       :: orbitals(:)
        integer                             :: norbitals
!
        real(REAL64), allocatable           :: coefficients(:,:)
        integer                             :: i,j
        integer                             :: centercounter, fcounter
        real(REAL64), pointer               :: bubvalues(:)
        real(REAL64), pointer               :: r(:)
        type(Grid1D), pointer               :: gridp
        type(LobeBasis)                     :: lobe_basis

        print *, 'begin initial guess generation'


        allocate(orbitals(norbitals))

        if (present(basis_object)) then
            lobe_basis = LobeBasis(structure_object%coordinates, &
                                   structure_object%nuclear_charge, basis_object)
        else
            lobe_basis = LobeBasis(structure_object%coordinates,structure_object%nuclear_charge)
        end if

        print *, 'lobe_basis done'

        coefficients = lobe_scf(lobe_basis, nocc)

        print *,'content to orbitals'

        ! place content of guess to orbitals
        do i=1,norbitals
print *, 'orbital', i
            orbitals(i) = 0d0*mould
            ! loop over bubbles centers
            fcounter=1
            do centercounter=1,size(lobe_basis%positions,2)
                gridp=> orbitals(i)%bubbles%get_grid(centercounter)
                r=>gridp%get_coord()
                bubvalues => orbitals(i)%bubbles%get_f(centercounter,0,0)
                ! loop over related basis functions
!                do j=1,floor(lobe_basis%charges(centercounter)+4)
                do j=1,lobe_basis%functions_on_atom(centercounter)
                    bubvalues=bubvalues+lobe_basis%lobes(fcounter)%c*&
                        exp(-lobe_basis%lobes(fcounter)%alpha*r**2) * coefficients(fcounter,i)
                    fcounter = fcounter +1
                end do
            end do
            ! this is no longer necessary
            !call orbitals(i)%set_type(F3D_TYPE_CUSP)
        end do


        ! cleanup
        nullify(r)
        nullify(bubvalues)
        nullify(gridp)
        print *,'initial guess generation done'

    end function

    ! angular momentum gaussian stuff
    ! following Petersson and Hellsing

    ! auxiliary sum for overlap elements
    function AuxiliarySum_overlap (alpha1,alpha2,l1,l2,diff) result(res)
        real(REAL64), intent(in)                 :: alpha1, alpha2
        integer, intent(in)                      :: l1,l2
        ! diff = (Ax-Bx) or (Ay-By) or (Az-Bz)
        real(REAL64), intent(in)                 :: diff
        real(REAL64)                             :: res
        real(REAL64)                             :: g
        ! loop variables
        integer                                  :: i1, i2, o, omega
        real(REAL64)                             :: multiplier

        g = alpha1 +alpha2
        res = 0d0
        multiplier = (-1)**l1 * gamma(l1+1.d0) * gamma(l2+1.d0)/(g**(l1+l2))

        do i1=0, floor(l1/2 +0d0)
            do i2=0,floor(l2/2 + 0d0)
                omega = l1+l2-2*(i1+i2)
                do o = 0,floor(omega/2 + 0d0)
                    res = res + ((-1)**o * gamma(omega+1.d0)*alpha1**(l2-i1-2*i2-o) * &
                                alpha2**(l1-2*i1-i2-o)/(4**(i1+i2+o)*gamma(i1+1d0)*gamma(i2+1d0)*gamma(o+1d0))*&
                                g**(2*(i1+i2)+o)*diff**(omega-2*o)/ &
                                (gamma(l1-2*i1+1d0)*gamma(l2-2*i2+1d0)*gamma(omega-2*o+1d0)) )
                end do
            end do
        end do

        res = res * multiplier
!        if(l1<0 .OR. l2<0) res =0d0
    end function

    function AngularGaussian_overlap_element(A,B) result(res)
        type(AngularGaussian), intent(in)          :: A,B
        real(REAL64)                               :: res
        real(REAL64)                               :: g, eta
        real(REAL64)                               :: diff(3)
        ! loop variables     
        integer                                    :: i1,i2,o
        integer                                    :: j1,j2,p
        integer                                    :: k1,k2,q
        real(REAL64)                               :: tempx,tempy, tempz, sum_
        real(REAL64)                               :: prefactor
        real(REAL64)                               :: Sx, Sy, Sz
        integer                                    :: omega
        ! test variables
        integer                                    :: itest
       
!TEMP
!        res = AngularGaussian_overlap_element_sp(A,B)
!        return
 

        itest=minval([A%lx,A%ly,A%lz,B%lx,B%ly,B%lz])
        if(itest<0) then
            !print *, 'strange l'
            res = 0d0
            return
        end if
!        print *, 'itest', itest

        g = A%alpha + B%alpha
!print *, 'new expo', g
        eta = A%alpha*B%alpha/g
!print *, 'eta', eta
        diff= A%center - B%center
        !diff= abs( A%center - B%center)
!print *, 'diff', diff
        res =  A%c*B%c*(pi/g)**1.5d0 * exp(-eta*dot_product(diff,diff))
!print *,'prefactor 1', res
        tempx= 0d0
        tempy= 0d0
        tempz= 0d0
        sum_ = 0d0
        Sx = 0d0
        Sy = 0d0
        Sz = 0d0
        prefactor =(-1)**A%lx * gamma(A%lx+1.d0) * gamma(B%lx+1.d0)/(g**(A%lx + B%lx )) 
        prefactor =prefactor * (-1)**A%ly * gamma(A%ly+1.d0) * gamma(B%ly+1.d0)/(g**(A%ly + B%ly )) 
        prefactor =prefactor * (-1)**A%lz * gamma(A%lz+1.d0) * gamma(B%lz+1.d0)/(g**(A%lz + B%lz ))
!print *, 'prefactor 2', prefactor

        do i1=0,floor(A%lx/2d0 )
        do i2=0,floor(B%lx/2d0 )
        omega = A%lx+B%lx -2*(i1+i2)
!print *, i1,i2,'omega', omega
        do o=0,floor(omega/2d0)
            tempx = (-1.d0)**o * gamma(omega+1d0) * (A%alpha)**( B%lx -i1  -2*i2 -o ) * &
( B%alpha )**( A%lx -i2 -2*i1 -o ) * g**( 2*(i1+i2) +o ) *diff(1)**(omega-2*o)
            tempx = tempx/ ( 4**(i1+i2+o) * gamma(i1+1d0) * gamma( i2 +1d0 ) *gamma( o+1d0 ) *&
gamma( A%lx -2*i1 +1d0 ) *gamma( B%lx -2*i2 +1d0 ) * gamma(omega -2*o +1d0) )  

            Sx = Sx + tempx

        end do
        end do
        end do

        do j1=0,floor(A%ly/2d0 )
        do j2=0,floor(B%ly/2d0 )
        omega = A%ly+B%ly -2*(j1+j2)
!print *, j1,j2,'omega', omega
        do p=0,floor(omega/2d0)
            tempy = (-1.d0)**p * gamma(omega+1d0) * (A%alpha)**( B%ly -j1  -2*j2 -p ) * &
( B%alpha )**( A%ly -j2 -2*j1 -p ) * g**( 2*(j1+j2) +p ) *diff(2)**(omega-2*p)
            tempy = tempy/ ( 4**(j1+j2+p) * gamma(j1+1d0) * gamma( j2 +1d0 ) *gamma( p+1d0 ) *&
gamma( A%ly -2*j1 +1d0 ) *gamma( B%ly -2*j2 +1d0 ) * gamma(omega -2*p +1d0) )  

            Sy = Sy + tempy

        end do
        end do
        end do

        do k1=0,floor(A%lz/2d0 )
        do k2=0,floor(B%lz/2d0 )
        omega = A%lz+B%lz -2*(k1+k2)
!print *, k1,k2,'omega', omega
        do q=0,floor(omega/2d0)
            tempz = (-1.d0)**q * gamma(omega+1d0) * (A%alpha)**( B%lz -k1  -2*k2 -q ) * &
( B%alpha )**( A%lz -k2 -2*k1 -q ) * g**( 2*(k1+k2) +q ) *diff(3)**(omega-2*q)
            tempz = tempz/ ( 4**(k1+k2+q) * gamma(k1+1d0) * gamma( k2 +1d0 ) *gamma( q+1d0 ) *&
gamma( A%lz -2*k1 +1d0 ) *gamma( B%lz -2*k2 +1d0 ) * gamma(omega -2*q +1d0) )  


            Sz = Sz + tempz

        end do
        end do
        end do

!print *, res
!print *, prefactor
!print *, 'Sx,Sy,Sz', Sx,Sy,Sz
!print *, diff

        res = res * Sx * Sy *Sz *prefactor

      

!print *, 'summary'
!print *, res
!print *, A%alpha, B%alpha, diff
        
 
!        res = res * AuxiliarySum_overlap(A%alpha,B%alpha,A%lx,B%lx,diff(1)) *&
!                    AuxiliarySum_overlap(A%alpha,B%alpha,A%ly,B%ly,diff(2)) *&
!                    AuxiliarySum_overlap(A%alpha,B%alpha,A%lz,B%lz,diff(3))

!        if(isNaN(res)) res =0d0
!        if(res>huge(res)) res=0d0
    end function


    
    ! 
    recursive function doublefactorial(n) result(res)
        integer      :: n
        real(REAL64) :: res
        if(n==0) then
            res = 1d0
        else if (modulo(n,2)==0) then
            res =2**(n/2) * gamma(n/2d0+1d0)
        else if (n>0) then
            res = gamma(n+1d0)/doublefactorial(n-1)
        else
            res = doublefactorial(n+2)/(n+2)
        end if
    end function


    ! test function for calculating overlap elements for between s and p
    ! orbitals
    function AngularGaussian_overlap_element_sp(A,B) result(res)
        type(AngularGaussian), intent(in)          :: A, B
        !type(AngularGaussian)                      :: Anew, Bnew
        integer                                    :: lA,lB
        integer                                    :: loc1, loc2
        real(REAL64)                               :: res
        real(REAL64)                               :: s0


        ! see the type of function
        lA = maxval(([A%lx,A%ly,A%lz]))
        lB = maxval(([B%lx,B%ly,B%lz]))
        ! see which p function is in question
        loc1 = maxloc([A%lx,A%ly,A%lz],1)
        loc2 = maxloc([B%lx,B%ly,B%lz],1)
        ! prefator
        s0 =  A%c * B%c *(pi/(A%alpha+B%alpha))**1.5d0 * &
exp(-A%alpha*B%alpha*dot_product(A%center-B%center,A%center-B%center)/(A%alpha+B%alpha ))
         

        if (lA == 0) then
            if(lB==0) then
                res = s0
            else
                res = -A%alpha*(B%center(loc2)-A%center(loc2))*s0/( A%alpha+B%alpha )
            end if
        else if (lB==0) then
            res = -B%alpha*( A%center(loc1)-B%center(loc1) ) * s0/( A%alpha+B%alpha )
        else
            res = ( A%center(loc1)-B%center(loc1) ) *( B%center(loc2) -A%center(loc2) )&
 *s0*A%alpha*B%alpha/( A%alpha+B%alpha )**2 
            if (loc1==loc2) then
                res = res + s0/(2*( A%alpha+B%alpha ))
            end if
        end if

    end function

    ! kinetic matrix element for s and p orbitals
    function AngularGaussian_kinetic_element_sp(A,B) result(res)
        type(AngularGaussian), intent(in)          :: A, B
        type(AngularGaussian)                      :: Anew, Bnew
        integer                                    :: lA,lB
        integer                                    :: loc1, loc2
        real(REAL64)                               :: res
        real(REAL64)                               :: s0
        real(REAL64)                               :: k0, ki0,k0j,kij     
        real(REAL64)                               :: si0,s0j, sij
!print *, 'start'

        ! see the type of function
        lA = maxval(([A%lx,A%ly,A%lz]))
        lB = maxval(([B%lx,B%ly,B%lz]))
        ! see which p function is in question
        loc1 = maxloc([A%lx,A%ly,A%lz],1)
        loc2 = maxloc([B%lx,B%ly,B%lz],1)
        ! prefator
        s0 =  A%c * B%c *(pi/(A%alpha+B%alpha))**1.5d0 * &
exp(-A%alpha*B%alpha*dot_product(A%center-B%center,A%center-B%center)/(A%alpha+B%alpha ))
        ! kinetic factor
        k0 = 3d0*A%alpha*B%alpha/( A%alpha + B%alpha ) - &
 2d0 *(A%alpha)**2 *(B%alpha)**2 *  dot_product(A%center-B%center,A%center-B%center) /&
(A%alpha+B%alpha)**2

        if (lA == 0) then
            if(lB==0) then
                ! simple case
                res = s0*k0
            else
                k0j = -2d0*B%alpha*(A%alpha)**2 *(B%center(loc2)-A%center(loc2)) / &
(A%alpha+B%alpha )**2  
                !res = k0j*s0 + k0*AngularGaussian_overlap_element_sp(A,B)

                s0j = -A%alpha*(B%center(loc2)-A%center(loc2))*s0/( A%alpha+B%alpha )
                res = k0j*s0 + k0*s0j

            end if
        else if (lB==0) then
            ki0 = -2d0*A%alpha*(B%alpha)**2*(A%center(loc1)-B%center(loc1)) / &
(A%alpha+B%alpha)**2  
!            res = ki0*s0 + k0*AngularGaussian_kinetic_element_sp(A,B)
            si0 = -B%alpha*( A%center(loc1)-B%center(loc1) ) * s0/( A%alpha+B%alpha )
            res = ki0*s0 + k0*si0

        else
k0j = -2d0*B%alpha*(A%alpha)**2 *(B%center(loc2)-A%center(loc2)) / &
(A%alpha+B%alpha )**2
ki0 = -2d0*A%alpha*(B%alpha)**2 *(A%center(loc1)-B%center(loc1)) / &
(A%alpha+B%alpha)**2   
s0j = -A%alpha*(B%center(loc2)-A%center(loc2))*s0/( A%alpha+B%alpha )
si0 = -B%alpha*( A%center(loc1)-B%center(loc1) ) * s0/( A%alpha+B%alpha )

sij = ( A%center(loc1)-B%center(loc1) ) *( B%center(loc2) -A%center(loc2) )&
 *s0*A%alpha*B%alpha/( A%alpha+B%alpha )**2 
            res = ki0*s0j + k0j*si0 + k0*sij


            if (loc1==loc2 .AND. loc1>0) then
                res = res + s0*A%alpha*B%alpha/( A%alpha + B%alpha )**2
                res = res + s0*k0/(2d0*(A%alpha + B%alpha))
            end if
        end if

        end function




    ! kinetic element, formula three
    function AngularGaussian_kinetic_element(A,B) result(res)
        type( AngularGaussian ), intent(in)  :: A,B
        type( AngularGaussian )              :: Bnew
        real(REAL64)                         :: Ix, Iy, Iz
        real(REAL64)                         :: res 

!TEMP
!        res = AngularGaussian_kinetic_element_sp(A,B)
!        return



        Ix = B%alpha * (2*B%lx+1d0) * AngularGaussian_overlap_element(A,B)
        Bnew = B
        Bnew%lx = B%lx +2
        Ix = Ix -2*(B%alpha)**2 * AngularGaussian_overlap_element(A,Bnew)
        if(B%lx>1) then
            Bnew%lx = B%lx-2
            Ix=Ix-B%lx*(B%lx-1d0)*AngularGaussian_overlap_element(A,Bnew)/2d0
        end if

        Iy = B%alpha * (2*B%ly+1d0) * AngularGaussian_overlap_element(A,B)
        Bnew = B
        Bnew%ly = B%ly +2
        Iy = Iy -2*(B%alpha)**2 * AngularGaussian_overlap_element(A,Bnew)
        if(B%ly>1) then
            Bnew%ly = B%ly-2
            Iy=Iy-B%ly*(B%ly-1d0)*AngularGaussian_overlap_element(A,Bnew)/2d0
        end if

        Iz = B%alpha * (2*B%lz+1d0) * AngularGaussian_overlap_element(A,B)
        Bnew = B
        Bnew%lz = B%lz +2
        Iz = Iz -2*(B%alpha)**2 * AngularGaussian_overlap_element(A,Bnew)
        if(B%lz>1) then
            Bnew%lz = B%lz-2
            Iz=Iz-B%lz*(B%lz-1d0)*AngularGaussian_overlap_element(A,Bnew)/2d0
        end if

        res = Ix + Iy + Iz
        res = res * A%c * B%c

    end function




    pure function Fnu(nu, u) result(res)
        integer, intent(in)       :: nu
        real(REAL64), intent(in)  :: u
        real(REAL64)              :: res
        integer                   :: k
        real(REAL64)              :: temp1, temp2

        if(u<epsilon(0d0)) then
            res = 1d0/(2*nu +1d0)
        else
            res = (gamma(2*nu+1d0))/(2*gamma(nu+1d0))
            temp1 = sqrt(pi)*erf(sqrt(u))/(4d0**(nu)*u**(nu+0.5d0))
            temp2 =0d0
            do k=0, nu-1
                temp2 = temp2+(gamma(nu-k+1d0)/(4**k *(gamma(2*nu-2*k+1d0) *u**(k+1d0) )))
            end do
            temp2 = temp2*exp(-u)
            res = res*(temp1-temp2)
        end if

    end function

    ! nuclear element over all nuclei
    function AngularGaussian_nuclear_element(A,B,charges,positions) result(res)
        type(AngularGaussian), intent(in)           :: A,B
        real(REAL64), intent(in)                    :: charges(:), positions(:,:)
        real(REAL64)                                :: res
        ! variables for combining gaussians
        real(REAL64)                                :: g, eta, P(3),prefactor, dAB(3)
        ! 
        real(REAL64)                                :: diff
        integer                                     :: Natom
        integer                                     :: iatom
        ! loop variables
        real(REAL64)                                :: sum_
        ! indices in x direction
        integer                                     :: i1,i2,o1,o2,r,u
        integer                                     :: mux
        real(REAL64)                                :: Ax
        ! indices in y direction
        integer                                     :: j1,j2,p1,p2,s,v
        integer                                     :: muy
        real(REAL64)                                :: Ay
        ! indices in z direction
        integer                                     :: k1,k2,q1,q2,t,w
        integer                                     :: muz
        real(REAL64)                                :: Az
        ! 
        integer                                     :: nu
        real(REAL64)                                :: dP(3)          

! TEMP
!        res = AngularGaussian_nuclear_element_sp(A,B,charges,positions)
!        return


        res = 0d0
        !dAB = A%center-B%center
        dAB = ( A%center-B%center )
        g = A%alpha + B%alpha
        ! Petersson and Hellsing has a typo in eq. (57)
        eta = A%alpha*B%alpha/g
        P = (A%alpha*A%center+B%alpha*B%center)/g
        prefactor = -2d0 * A%c*B%c*pi*exp(-eta*(dot_product(dAB,dAB ))) /g
        prefactor = prefactor * (-1d0)**(A%lx+B%lx+A%ly+B%ly+A%lz+B%lz) * &
                    gamma(A%lx+1d0) * gamma(B%lx+1d0) * gamma(A%ly+1d0) * & 
                    gamma(B%ly+1d0) * gamma(A%lz+1d0) * gamma(B%lz+1d0)
        Natom=size(charges)
        
        do iatom=1, Natom
            ! start atom loop
            sum_ = 0d0
            diff = g * dot_product(P-positions(:,iatom), P-positions(:,iatom))
            Ax = 0d0
            Ay = 0d0
            Az = 0d0
            dP=P-positions(:,iatom)
            !loops
        ! x loops
        do i1 = 0, floor(A%lx/2d0)
        do i2 = 0, floor(B%lx/2d0)
        do o1 = 0, A%lx - 2*i1
        do o2 = 0, B%lx - 2*i2
        mux   = A%lx + B%lx -2*(i1+i2) - (o1+o2)
        do r  = 0, floor((o1+o2)/2d0)
        do u  = 0, floor(mux/2d0) 
        Ax = (-1d0)**(o2+r) * gamma(o1 + o2 +1d0)/ &
             (4d0**(i1+i2+r)*gamma(i1+1d0)* gamma(i2+1d0) * gamma(o1+1d0)*gamma(o2+1d0)*gamma(r+1d0) )
        Ax = Ax*(A%alpha)**(o2-i1-r)*(B%alpha)**(o1-i2-r)*dAB(1)**(o1+o2-2*r)/&
             (gamma(A%lx-2*i1-o1+1d0)*gamma(B%lx-2*i2-o2+1d0)*gamma(o1+o2-2*r+1d0 ))
        Ax = Ax*(-1d0)**u *gamma(mux+1d0)* ( dP(1) )**(mux-2*u) / &
             (4**u*gamma(u+1d0)*gamma(mux-2*u+1d0)*g**(o1+o2-r+u))
        ! y loops
        do j1 = 0, floor(A%ly/2d0)
        do j2 = 0, floor(B%ly/2d0)
        do p1 = 0, A%ly - 2*j1
        do p2 = 0, B%ly - 2*j2
        muy   = A%ly + B%ly -2*(j1+j2) - (p1+p2)
        do s  = 0, floor((p1+p2)/2d0)
        do v  = 0, floor(muy/2d0)          
        Ay = (-1d0)**(p2+s) * gamma(p1 + p2 +1d0)/ &
             (4d0**(j1+j2+s)*gamma(j1+1d0)* gamma(j2+1d0) * gamma(p1+1d0)*gamma(p2+1d0)*gamma(s+1d0) )
        Ay = Ay*(A%alpha)**(p2-j1-s)*(B%alpha)**(p1-j2-s)*dAB(2)**(p1+p2-2*s)/&
             (gamma(A%ly-2*j1-p1+1d0)*gamma(B%ly-2*j2-p2+1d0)*gamma(p1+p2-2*s+1d0 ))
        Ay = Ay*(-1d0)**v *gamma(muy+1d0)*&
( dP(2) )**(muy-2*v) / &
             (4**v*gamma(v+1d0)*gamma(muy-2*v+1d0)*g**(p1+p2-s+v))
        ! z loops
        do k1 = 0, floor(A%lz/2d0)
        do k2 = 0, floor(B%lz/2d0)
        do q1 = 0, A%lz - 2*k1
        do q2 = 0, B%lz - 2*k2
        muz   = A%lz + B%lz -2*(k1+k2) - (q1+q2)
        do t  = 0, floor((q1+q2)/2d0)
        do w  = 0, floor(muz/2d0)          
        Az = (-1d0)**(q2+t) * gamma(q1 + q2 +1d0)/ &
             (4d0**(k1+k2+t)*gamma(k1+1d0)* gamma(k2+1d0) * gamma(q1+1d0)*gamma(q2+1d0)*gamma(t+1d0) )
        Az = Az*(A%alpha)**(q2-k1-t)*(B%alpha)**(q1-k2-t)*dAB(3)**(q1+q2-2*t)/&
             (gamma(A%lz-2*k1-q1+1d0)*gamma(B%lz-2*k2-q2+1d0)*gamma(q1+q2-2*t+1d0 ))
        Az = Az*(-1d0)**w *gamma(muz+1d0)*&
( dP(3) )**(muz-2*w) / &
             (4**w*gamma(w+1d0)*gamma(muz-2*w+1d0)*g**(q1+q2-t+w))
        ! calculation stuff
        nu = mux + muy + muz -(u + v + w)
        sum_ = sum_ + Ax*Ay*Az*Fnu(nu, diff)        
 
        ! z loops
        end do ! w loop
        end do ! t loop
        end do ! q2 loop
        end do ! q1 loop
        end do ! k2 loop
        end do ! k1 loop
        ! y loops
        end do ! v loop
        end do ! s loop
        end do ! p2 loop
        end do ! p1 loop
        end do ! j2 loop
        end do ! j1 loop
        ! x loops
        end do ! u loop
        end do ! r loop
        end do ! o2 loop
        end do ! o1 loop
        end do ! i2 loop
        end do ! i1 loop

            ! prefactors into account
            sum_ = sum_ * prefactor * charges(iatom)
            res = res + sum_
        end do ! iatom loop
 

    end function

    ! nuclear integral over s and p orbitals
    function AngularGaussian_nuclear_element_sp(A,B,charges,positions) result(res)
        type(AngularGaussian), intent(in)           :: A,B
        real(REAL64), intent(in)                    :: charges(:), positions(:,:)
        real(REAL64)                                :: res
        ! 
        real(REAL64)                                :: s0, si0,s0j,sij
        real(REAL64)                                :: l0, li0,l0j,lij
        ! 
        real(REAL64)                                :: g, P(3), C(3)
        integer                                     :: lA,lB
        integer                                     :: loc1, loc2
        ! 
        integer                                     :: iatom, Natom
        real(REAL64)                                :: diff 

        ! see the type of function
        lA = maxval(([A%lx,A%ly,A%lz]))
        lB = maxval(([B%lx,B%ly,B%lz]))
        ! see which p function is in question
        loc1 = maxloc([A%lx,A%ly,A%lz],1)
        loc2 = maxloc([B%lx,B%ly,B%lz],1)
        ! prefator 


        g=A%alpha + B%alpha
        

        res = 0d0 
        P = (A%alpha*A%center+B%alpha*B%center)/g
        s0 =  A%c * B%c *(pi/(A%alpha+B%alpha))**1.5d0 * &
exp(-A%alpha*B%alpha*dot_product(A%center-B%center,A%center-B%center)/(A%alpha+B%alpha ))

        s0j = -A%alpha*(B%center(loc2)-A%center(loc2))*s0/( A%alpha+B%alpha )

        si0 = -B%alpha*( A%center(loc1)-B%center(loc1) ) * s0/( A%alpha+B%alpha )

        sij = ( A%center(loc1)-B%center(loc1) ) *( B%center(loc2) -A%center(loc2) ) *&
 s0*A%alpha*B%alpha/( A%alpha+B%alpha )**2 
        if(loc1==loc2 .AND. loc1>0) sij = sij + s0/(2*g)


        Natom=size(charges)
        
        do iatom=1, Natom
            ! start atom loop
            C = positions(:,iatom)
            diff = g * dot_product(P-C, P-C)
            ! helper variables
            l0 = Fnu(0,diff)
            li0 = (C(loc1)-P(loc1)) * Fnu(1,diff)
            l0j = (C(loc2)-P(loc2)) * Fnu(1,diff)
            lij = (P(loc1)-C(loc1))* ( P(loc2) -C(loc2) )*Fnu(2,diff)

            if(loc1==loc2 .AND. loc1>0) lij = lij -Fnu(1,diff)/(2*g)

            ! calculation stuff
            if (lA == 0) then
                if(lB==0) then
                    ! simple case
                    res = res + charges(iatom) *s0*l0
                else
                    res = res + charges(iatom) * ( s0j*l0 +s0*l0j )
                end if
            else if (lB==0) then
                res = res + charges(iatom) * ( si0*l0 + s0*li0 )
            else
                res = res + charges(iatom) * ( sij*l0 + si0*l0j + s0j*li0 + s0*lij )
            end if

        end do

        ! scaling
        res = -res*2d0*sqrt(g/pi)
    end function




    ! two electron integral
    function AngularGaussian_two_electron_integral(A,B,C,D) result(res)
    !pure function AngularGaussian_two_electron_integral(A,B,C,D) result(res)
    !pure function AngularGaussian_two_electron_integral(A,C,B,D) result(res)
        type(AngularGaussian), intent(in)           :: A, B, C, D
        real(REAL64)                                :: res
        real(REAL64)                                :: prefactor
        ! variables for combining gaussians
        real(REAL64)                                :: gp, gq
        real(REAL64)                                :: etap, etaq, eta
        real(REAL64)                                :: P(3), Q(3)
        real(REAL64)                                :: temp1, diff
        ! loop variables
        integer                                     :: nu
        ! x indices
        integer                                     :: i1,i2,i3,i4
        integer                                     :: o1,o2,o3,o4 
        integer                                     :: r1,r2, u
        integer                                     :: mux
        real(REAL64)                                :: tempx1, tempx2, Jx
        ! y indices
        integer                                     :: j1,j2,j3,j4
        integer                                     :: p1,p2,p3,p4
        integer                                     :: s1,s2,v
        integer                                     :: muy
        real(REAL64)                                :: tempy1, tempy2,Jy
        ! z indices 
        integer                                     :: k1,k2,k3,k4
        integer                                     :: q1,q2,q3,q4
        integer                                     :: t1,t2,w
        integer                                     :: muz
        real(REAL64)                                :: tempz1,tempz2, Jz


        ! initialize variables
        res=0d0

        gp = A%alpha + B%alpha
        gq = C%alpha + D%alpha
        ! there is a typo in eq 62 of petersson and hellsing
        etap = A%alpha*B%alpha/gp
        etaq = C%alpha*D%alpha/gq
        eta = gp*gq/(gq+gp)
        P = (A%alpha*A%center+B%alpha*B%center)/gp
        Q = (C%alpha*C%center+D%alpha*D%center)/gq
!        diff = eta * dot_product(P-Q,P-Q)
        diff = dot_product(P-Q,P-Q)

        !prefactor = 2d0*A%c*B%c*C%c*D%c*pi**2.5d0/(gq*gp*sqrt(gq+gp))
        prefactor = 2d0*A%c*B%c*C%c*D%c*pi**2.5d0/(gq*gp*sqrt(gq*gp))
        ! this is test
        !prefactor = prefactor * 2.5d0/pi
        
        temp1=dot_product(A%center-B%center,A%center-B%center)
        prefactor = prefactor*exp(-etap*temp1) 
        temp1=dot_product(C%center-D%center,C%center-D%center)
        prefactor = prefactor*exp(-etaq*temp1) 
        ! prefactors from eq 63
        prefactor = prefactor * (-1d0)**(A%lx+B%lx+A%ly+B%ly+A%lz+B%lz) * &
                    gamma(A%lx+1d0) * gamma(B%lx+1d0) * gamma(A%ly+1d0) * & 
                    gamma(B%ly+1d0) * gamma(A%lz+1d0) * gamma(B%lz+1d0)
        prefactor = prefactor / gp**(A%lx+B%lx+A%ly+B%ly+A%lz+B%lz)

        ! start loops
        ! x loops 
        do i1 = 0, floor(A%lx/2d0)
        do i2 = 0, floor(B%lx/2d0)
        do o1 = 0, A%lx -2*i1
        do o2 = 0, B%lx -2*i2
        do r1 = 0, floor((o1+o2)/2d0)
        !
        tempx1 = (-1d0)**(o2+r1)*gamma(o1+o2+1d0)/(4**(i1+i2+r1)*gamma(i1+1d0)*&
                 gamma(i2+1d0)*gamma(o1+1d0)*gamma(o2+1d0)*gamma(r1+1d0))
        tempx1 = tempx1*(A%alpha)**(o2-i1-r1)*(B%alpha)**(o1-i2-r1) * gp**(2*(i1+i2) +r1)*&
                 (A%center(1)-B%center(1))**(o1+o2-2*r1)*gamma(C%lx+1d0)*gamma(D%lx+1d0)/ &
                 (gq**(C%lx+D%lx)*gamma(A%lx-2*i1-o1+1d0)*gamma(B%lx-2*i2-o2+1d0)*&
                 gamma(o1+o2-2*r1+1d0)) 
        do i3 = 0, floor(C%lx/2d0)
        do i4 = 0, floor(D%lx/2d0)
        do o3 = 0, C%lx-2*i3
        do o4 = 0, D%lx-2*i4
        mux = A%lx+B%lx+C%lx+D%lx-2*(i1+i2+i3+i4)-(o1+o2+o3+o4)
        do r2 = 0, floor((o3+o4)/2d0)
        tempx2 = (-1d0)**(o3+r2)*gamma(o3+o4+1d0)/(4**(i3+i4+r2)*gamma(i3+1d0)*&
                 gamma(i4+1d0)*gamma(o3+1d0)*gamma(o4+1d0)*gamma(r2+1d0))
        ! this version is in Bodroski, Vukmirovic, Skrbic
        !tempx2 = tempx2*(C%alpha)**(o4-i3-r2)*(D%alpha)**(o3-i4-r2) * gp**(2*(i3+i4) +r2)*&
        ! this version is in Petersson and Hellsing
        tempx2 = tempx2*(C%alpha)**(o4-i3-r2)*(D%alpha)**(o3-i4-r2) * gq**(2*(i3+i4) +r2)*&
                 (C%center(1)-D%center(1))**(o3+o4-2*r2)/ &
                 (gamma(C%lx-2*i3-o3+1d0)*gamma(D%lx-2*i4-o4+1d0)*gamma(o3+o4-2*r2+1d0)) 
        do u = 0, floor(mux/2d0)
        Jx = (-1)**u * gamma(mux+1d0)*eta**(mux-u) * (P(1)-Q(1))**(mux-2*u)/&
             (4**u * gamma(u+1d0) * gamma(mux-2*u+1d0))
        Jx = Jx * tempx1 * tempx2

        ! y loops 
        do j1 = 0, floor(A%ly/2d0)
        do j2 = 0, floor(B%ly/2d0)
        do p1 = 0, A%ly -2*j1
        do p2 = 0, B%ly -2*j2
        do s1 = 0, floor((p1+p2)/2d0)
        !
        tempy1 = (-1d0)**(p2+s1)*gamma(p1+p2+1d0)/(4**(j1+j2+s1)*gamma(j1+1d0)*&
                 gamma(j2+1d0)*gamma(p1+1d0)*gamma(p2+1d0)*gamma(s1+1d0))
        tempy1 = tempy1*(A%alpha)**(p2-j1-s1)*(B%alpha)**(p1-j2-s1) * gp**(2*(j1+j2) +s1)*&
                 (A%center(2)-B%center(2))**(p1+p2-2*s1)*gamma(C%ly+1d0)*gamma(D%ly+1d0)/ &
                 (gq**(C%ly+D%ly) * gamma(A%ly-2*j1-p1+1d0) * gamma(B%ly-2*j2-p2+1d0)*&
                 gamma(p1+p2-2*s1+1d0) ) 
        do j3 = 0, floor(C%ly/2d0)
        do j4 = 0, floor(D%ly/2d0)
        do p3 = 0, C%ly-2*j3
        do p4 = 0, D%ly-2*j4
        muy = A%ly+B%ly+C%ly+D%ly-2*(j1+j2+j3+j4)-(p1+p2+p3+p4)
        do s2 = 0, floor((p3+p4)/2d0)
        tempy2 = (-1d0)**(p3+s2)*gamma(p3+p4+1d0)/(4**(j3+j4+s2)*gamma(j3+1d0)*&
                 gamma(j4+1d0)*gamma(p3+1d0)*gamma(p4+1d0)*gamma(s2+1d0))
        !tempy2 = tempy2*(C%alpha)**(p4-j3-s2)*(D%alpha)**(p3-j4-s2) * gp**(2*(j3+j4) +s2)*&
        tempy2 = tempy2*(C%alpha)**(p4-j3-s2)*(D%alpha)**(p3-j4-s2) * gq**(2*(j3+j4) +s2)*&
                 (C%center(2)-D%center(2))**(p3+p4-2*s2)/ &
                 (gamma(C%ly-2*j3-p3+1d0)*gamma(D%ly-2*j4-p4+1d0)*gamma(p3+p4-2*s2+1d0)) 
        do v = 0, floor(muy/2d0)
        Jy = (-1)**v * gamma(muy+1d0)*eta**(muy-v) * (P(2)-Q(2))**(muy-2*v)/&
             (4**v * gamma(v+1d0) * gamma(muy-2*v+1d0))
        Jy = Jy * tempy1 * tempy2

        ! z loops
        do k1 = 0, floor(A%lz/2d0)
        do k2 = 0, floor(B%lz/2d0)
        do q1 = 0, A%lz -2*k1
        do q2 = 0, B%lz -2*k2
        do t1 = 0, floor((q1+q2)/2d0)
        !
        tempz1 = (-1d0)**(q2+t1)*gamma(q1+q2+1d0)/(4**(k1+k2+t1)*gamma(k1+1d0)*&
                 gamma(k2+1d0)*gamma(q1+1d0)*gamma(q2+1d0)*gamma(t1+1d0))
        tempz1 = tempz1*(A%alpha)**(q2-k1-t1)*(B%alpha)**(q1-k2-t1) * gp**(2*(k1+k2) +t1)*&
                 (A%center(3)-B%center(3))**(q1+q2-2*t1)*gamma(C%lz+1d0)*gamma(D%lz+1d0)/ &
                 (gq**(C%lz+D%lz)*gamma(A%lz-2*k1-q1+1d0)*gamma(B%lz-2*k2-q2+1d0)*&
                 gamma(q1+q2-2*t1+1d0)) 
        do k3 = 0, floor(C%lz/2d0)
        do k4 = 0, floor(D%lz/2d0)
        do q3 = 0, C%lz-2*k3
        do q4 = 0, D%lz-2*k4
        muz = A%lz+B%lz+C%lz+D%lz-2*(k1+k2+k3+k4)-(q1+q2+q3+q4)
        do t2 = 0, floor((q3+q4)/2d0)
        tempz2 = (-1d0)**(q3+t2)*gamma(q3+q4+1d0)/(4**(k3+k4+t2)*gamma(k3+1d0)*&
                 gamma(k4+1d0)*gamma(q3+1d0)*gamma(q4+1d0)*gamma(t2+1d0))
        !tempz2 = tempz2*(C%alpha)**(q4-k3-t2)*(D%alpha)**(q3-k4-t2) * gp**(2*(k3+k4) +t2)*&
        tempz2 = tempz2*(C%alpha)**(q4-k3-t2)*(D%alpha)**(q3-k4-t2) * gq**(2*(k3+k4) +t2)*&
                 (C%center(3)-D%center(3))**(q3+q4-2*t2)/ &
                 (gamma(C%lz-2*k3-q3+1d0)*gamma(D%lz-2*k4-q4+1d0)*gamma(q3+q4-2*t2+1d0)) 
        do w = 0, floor(muz/2d0)
        Jz = (-1)**w * gamma(muz+1d0)*eta**(muz-w) * (P(3)-Q(3))**(muz-2*w)/&
             (4**w * gamma(w+1d0) * gamma(muz-2*w+1d0))
        Jz = Jz * tempz1 * tempz2


        ! actual calculation 
        nu = mux +muy +muz -(u+v+w)
        res = res + Jx*Jy*Jz*Fnu(nu, eta*diff)  

        ! end z loops
        end do ! w loop
        end do ! t2 loop
        end do ! q4 loop
        end do ! q3 loop
        end do ! k4 loop
        end do ! k3 loop
        end do ! t1 loop
        end do ! q2 loop
        end do ! q1 loop
        end do ! k2 loop
        end do ! k1 loop
        ! end y loops
        end do ! v loop
        end do ! s2 loop
        end do ! p4 loop
        end do ! p3 loop
        end do ! j4 loop
        end do ! j3 loop
        end do ! s1 loop
        end do ! p2 loop
        end do ! p1 loop
        end do ! j2 loop
        end do ! j1 loop

        ! end x loops
        end do ! u loop
        end do ! r2 loop
        end do ! o4 loop
        end do ! o3 loop
        end do ! i4 loop
        end do ! i3 loop
        end do ! r1 loop
        end do ! o2 loop
        end do ! o1 loop
        end do ! i2 loop
        end do ! i1 loop

        ! scale result
        res = res * prefactor

!print *, 'prefactor', prefactor, 2d0*pi**2.5d0 * exp(-A%alpha*B%alpha*&
!dot_product(A%center-B%center,A%center-B%center)/(A%alpha+B%alpha)) * &
!exp(-C%alpha*D%alpha*dot_product(C%center-D%center, C%center-D%center )/&
!(C%alpha+D%alpha)) / ((A%alpha + B%alpha) * ( C%alpha +D%alpha ) * &
!sqrt(A%alpha+B%alpha+C%alpha+D%alpha) )  

    end function




    ! solid gaussian stuff

    ! initializer function for SolidGaussian
    function SolidGaussian_init(l,m,alpha,center) result(new)
        integer, intent(in)       :: l,m
        real(REAL64), intent(in)  :: alpha
        real(REAL64), intent(in)  :: center(3)
        type(SolidGaussian)       :: new
        integer                   :: i, N


        new%alpha  = alpha
        new%center = center
       
        ! note functions are normalized to 1 
        select case (l)
            ! s type function
            case (0)
            !print *, 'spherical function'
            new%l = 0
            new%m = 0
            N=1
            allocate(new%primitives(N))
            allocate(new%coeffs(N))
            new%primitives(1) = AngularGaussian(0,0,0,alpha,center,1d0)
!            new%coeffs(1) = 0.5d0*sqrt(1d0/pi)
            new%coeffs(1) = 1d0

            ! p type function
            case(1)
                new%l = l
                new%m = m
                N=1
                allocate(new%primitives(N))
                allocate(new%coeffs(N))
                select case(m)
                    case(1)
                       new%primitives(1) = AngularGaussian(1,0,0,alpha,center,1d0)
                    case(0)
                       new%primitives(1) = AngularGaussian(0,0,1,alpha,center,1d0)
                    case(-1)
                       new%primitives(1) = AngularGaussian(0,1,0,alpha,center,1d0)
                end select
!                new%coeffs(1) = sqrt(3d0/(4d0*pi))
                new%coeffs(1) = 1d0
            case(2)
                new%l = l
                new%m = m
                select case(m)
                    case(2)
                        allocate(new%primitives(2))
                        allocate(new%coeffs(2))
                        new%primitives(1)=  AngularGaussian(2,0,0,alpha,center,1d0)
                        new%primitives(2)=  AngularGaussian(0,2,0,alpha,center,1d0)
                        new%coeffs(1) = sqrt(3d0)/2d0
                        new%coeffs(2) = -sqrt(3d0)/2d0
                    case(1)
                        allocate(new%primitives(1))
                        allocate(new%coeffs(1))
                        new%primitives(1) = AngularGaussian(1,0,1,alpha,center,1d0)
                        new%coeffs(1) = sqrt(3d0)
                    case(0)
                        allocate(new%primitives(3))
                        allocate(new%coeffs(3))
                        new%primitives(1)=  AngularGaussian(2,0,0,alpha,center,1d0)
                        new%primitives(2)=  AngularGaussian(0,2,0,alpha,center,1d0)
                        new%primitives(3)=  AngularGaussian(0,0,2,alpha,center,1d0)
                        new%coeffs(1) = -0.5d0
                        new%coeffs(2) = -0.5d0
                        new%coeffs(3) = 1d0
                    case(-1)
                        allocate(new%primitives(1))
                        allocate(new%coeffs(1))
                        new%primitives(1) = AngularGaussian(0,1,1,alpha,center,1d0)
                        new%coeffs(1) = sqrt(3d0)
                    case(-2)
                        allocate(new%primitives(1))
                        allocate(new%coeffs(1))
                        new%primitives(1) = AngularGaussian(1,1,0,alpha,center,1d0)
                        new%coeffs(1) = sqrt(3d0)
                end select
            case default
            print *, 'default case: spherical function'
            new%l = 0
            new%m = 0
            N=1
            allocate(new%primitives(N))
            allocate(new%coeffs(N))
            new%primitives(1) = AngularGaussian(0,0,0,alpha,center,1d0)
            !new%coeffs(1) = 0.5d0*sqrt(1d0/pi)
            new%coeffs(1) = 1d0
        end select

    end function

    function SolidGaussian_overlap_element(A,B) result(res)
        type(SolidGaussian), intent(in)        :: A,B 
        real(REAL64)                           :: res
        integer                                :: i,j,N,M

        N = size(A%primitives)
        M = size(B%primitives)
        res = 0d0
        do i=1,N
            do j=1,M
                res = res+A%coeffs(i)*B%coeffs(j)*&
                   AngularGaussian_overlap_element(A%primitives(i), B%primitives(j))
!                   print *, 'overlap coeff', A%coeffs(i), B%coeffs(j)
!                   print *, 'primitive overlap', AngularGaussian_overlap_element(A%primitives(i), B%primitives(j))
            end do
        end do
!        print *,'total overlap', res
    end function

    



    function SolidGaussian_kinetic_element(A,B) result(res)
        type(SolidGaussian), intent(in)        :: A,B 
        real(REAL64)                           :: res
        integer                                :: i,j,N,M

        N = size(A%primitives)
        M = size(B%primitives)
        res = 0d0
        do i=1,N
            do j=1,M
                res = res + A%coeffs(i)*B%coeffs(j)*&
                   AngularGaussian_kinetic_element(A%primitives(i), B%primitives(j))
            end do
        end do
        !if(isNaN(res)) print *, 'NaN on kinetic element'
    end function
    function SolidGaussian_nuclear_element(A,B, charges, positions) result(res)
        type(SolidGaussian), intent(in)        :: A,B 
        real(REAL64)                           :: res
        integer                                :: i,j,N,M
        real(REAL64), intent(in)                    :: charges(:), positions(:,:)

        N = size(A%primitives)
        M = size(B%primitives)
        res = 0d0
        do i=1,N
            do j=1,M
                res = res+  A%coeffs(i)*B%coeffs(j)*&
       AngularGaussian_nuclear_element(A%primitives(i), B%primitives(j), charges, positions)
            end do
        end do
        !if(isNaN(res)) print *, 'NaN on nuclear element'
    end function

    function SolidGaussian_two_electron_integral(A,B,C,D) result(res)
    !pure function SolidGaussian_two_electron_integral(A,B,C,D) result(res)
        type(SolidGaussian), intent(in)        :: A, B, C, D
        real(REAL64)                           :: res 
        integer                                :: i,j,k,l, N, M, O,P

        N = size(A%primitives)
        M = size(B%primitives)
        O = size(C%primitives)
        P = size(D%primitives)

        res = 0d0
!res=AngularGaussian_two_electron_integral(A%primitives(1),B%primitives(1),C%primitives(1) ,D%primitives(1) )
!return
        do i=1,N
            do j=1,M
                do k=1,O
                    do l=1,P
                        res = res + A%coeffs(i)*B%coeffs(j) * C%coeffs(k) * D%coeffs(l) * &
!AngularGaussian_two_electron_integral(A%primitives(1),B%primitives(1),C%primitives(1) ,D%primitives(1) )
AngularGaussian_two_electron_integral(A%primitives(i),B%primitives(j),C%primitives(k) ,D%primitives(l) )
!AngularGaussian_two_electron_integral(A%primitives(i),C%primitives(k),B%primitives(j) ,D%primitives(l) )
                    end do
                end do
            end do
        end do

    end function

    !> calculating overlap matrix of a basis
    function SolidGaussianBasis_calculate_overlap_matrix(self) result(overlap)
        class(SolidGaussianBasis), intent(in)        :: self
        real(REAL64),allocatable                     :: overlap(:,:)
        integer                                      :: i,j, Nbasis

        Nbasis = size(self%functions)
        allocate(overlap(Nbasis,Nbasis))
        overlap = 0d0

        do i=1,Nbasis
            do j=1,i
                overlap(i,j) = SolidGaussian_overlap_element(self%functions(i), self%functions(j))
                !overlap(j,i) = SolidGaussian_overlap_element(self%functions(j), self%functions(i))
!                overlap(j,i) = overlap(i,j)
!if(abs(overlap(i,j))>1d0) then
!    print *, 'HUGE OVERLAP',i,j, overlap(i,j)
!    print *, 'expos', self%functions(i)%alpha,  self%functions(i)%alpha
!    print *, 'coeffs', self%functions(i)%coeffs, self%functions(j)%coeffs
!    print *, 'l', self%functions(i)%l, self%functions(j)%l
!    print *, 'm', self%functions(i)%m, self%functions(j)%m
!    print *, 'dist', self%functions(i)%center - self%functions(j)%center
!    overlap(i,j) = 1d0/overlap(i,j)
!end if
                overlap(j,i) = overlap(i,j)
            end do
            ! regularization for the matrix
            !overlap(i,i) = overlap(i,i)+5d-1
!            print *, 'overlap diagonal element',i,overlap(i,i)
        end do

        print *, 'maxval', maxval(overlap)
        print *, 'minval', minval(overlap)

!print *, 'OVERLAP MATRIX: ', overlap


    end function

    !> basis two-electron functions
    function SolidGaussianBasis_calculate_two_electron_integrals(self) result(res)
        class(SolidGaussianBasis), intent(in)         :: self
        real(REAL64),allocatable                      :: res(:,:,:,:)
        integer                                       :: i,j,k,l, Nbasis

        Nbasis = size(self%functions)
        allocate(res(Nbasis,Nbasis,Nbasis,Nbasis))
        res=0d0

!        forall (i = 1:Nbasis, j = 1:Nbasis, k = 1:Nbasis, l = 1:Nbasis) 
do i=1,Nbasis
!do j=1,Nbasis
do j=1,i
do k=1,Nbasis
!do l=1,Nbasis
do l=1,k
            res(i,j,k,l) = SolidGaussian_two_electron_integral(self%functions(i), & 
                               self%functions(j), self%functions(k),self%functions(l))

res(j,i,k,l) = res(i,j,k,l)
res(i,j,l,k) = res(i,j,k,l)
res(j,i,l,k) = res(i,j,k,l)

end do
end do
end do
end do
!        end forall

    end function

    !> calculating hamiltonian matrix of basis
    function SolidGaussianBasis_calculate_hamiltonian_matrix(self) result(hamiltonian)
        class(SolidGaussianBasis), intent(in)       :: self
        real(REAL64),allocatable                    :: hamiltonian(:,:)
        integer                                     :: i,j, Nbasis
        ! smoothing test
        real(REAL64)                                :: fn
        real(REAL64)                                :: temp

        Nbasis = size(self%functions)
        allocate(hamiltonian(Nbasis,Nbasis))
        hamiltonian = 0d0

        do i=1,Nbasis
            do j=1,i
                hamiltonian(i,j) = hamiltonian(i,j)+&
                    SolidGaussian_kinetic_element(self%functions(i),self%functions(j))
!temp = SolidGaussian_kinetic_element(self%functions(i), self%functions(j))
!print *, 'kinetic matrix element',i,j, temp
                hamiltonian(i,j) = hamiltonian(i,j)+&
                    SolidGaussian_nuclear_element(self%functions(i), self%functions(j),self%charges,self%positions)
                hamiltonian(j,i) = hamiltonian(i,j)
!temp = SolidGaussian_nuclear_element(self%functions(i), self%functions(j),self%charges,self%positions)
!                hamiltonian(j,i) = hamiltonian(i,j)
!print *, 'nuclear matrix element', i,j, temp
            end do
            ! regularization
            !hamiltonian(i,i) = hamiltonian(i,i)-1d-1
        end do

        print *, 'Nbasis', Nbasis
        print *, 'maxval hamiltonian', maxval(hamiltonian), maxloc(hamiltonian)
        print *, 'minval hamiltonian', minval(hamiltonian), minloc(hamiltonian)
        !fn=sum(hamiltonian*hamiltonian)
        !fn = fn/(Nbasis**2)
        !where(abs(hamiltonian)<fn*1d-0) hamiltonian= 0d0

    end function

    function SolidGaussianBasis_init_from_basis_object(positions, charges, basis_object) result (new_basis)
        type(Basis), intent(in)                :: basis_object
        real(REAL64), intent(in)               :: positions(:,:)
        real(REAL64), intent(in)               :: charges(:)
        type(SolidGaussianBasis)               :: new_basis
        integer                                :: i,j,k, Natoms
        integer                                :: Nfunctions
! test variables
        integer, allocatable                   :: nonzeros(:)
        integer, allocatable                   :: finfo(:,:)
        integer                                :: iexp, im
        integer                                :: limplemented
        integer, allocatable                   :: fperatomtype(:)

        !
        limplemented = 1        

        allocate(nonzeros(basis_object%number_of_subshells))
        allocate(finfo(basis_object%number_of_subshells,4))
        print *, 'basis_object%atom_type', basis_object%atom_type
!
        ! functions per atom type array
        allocate(fperatomtype(maxval(basis_object%atom_type)))
        fperatomtype = 0

        do i=1, basis_object%number_of_subshells
            nonzeros(i) = count(basis_object%exponentials(:,i)/=0d0)
            !nonzeros(i) = 1
            !print *, 'eksponentteja',count(basis_object%exponentials(:,i)/=0d0)
            print *, i,'eksponentit', basis_object%exponentials(1:nonzeros(i),i)
            ! exponents on subshell
            finfo(i,1)=nonzeros(i)
            !finfo(i,1)=1
            ! l of subshell
            finfo(i,2)=basis_object%l(i)
            if(finfo(i,2)>limplemented) then
                print *, 'lmax implemented is ', limplemented 
                print *, 'using s function instead'
                finfo(i,2)=0
            end if
            ! number of functions on this subshell
            finfo(i,3)=(2*finfo(i,2)+1)*finfo(i,1)
            ! which atom type is this subshell related to
            finfo(i,4) = basis_object%atom_type(i)
            ! how many functions there are for this atom type
            fperatomtype(basis_object%atom_type(i)) = &
 fperatomtype(basis_object%atom_type(i)) + finfo(i,3) 

        end do
        print *, 'number of functions', sum(finfo(:,3))        



        Natoms = size(charges)
        Nfunctions = 0
        allocate(new_basis%functions_on_atom(Natoms))

        ! loop over atoms
        do i=1,Natoms
            ! increase number of functions by the number 
            ! corresponding to int(charges)
            Nfunctions = Nfunctions + fperatomtype(int(charges(i)))
            ! functions on atom by type
            new_basis%functions_on_atom(i) =  fperatomtype(int(charges(i)))
        end do
        print *, 'Nfunctions', Nfunctions
        
        ! k is a function counter  
        k=1
        ! allocate space for basis functions
        allocate(new_basis%functions(Nfunctions))
        

        do i=1,Natoms
            ! loop over subshells
            do j=1, basis_object%number_of_subshells

                ! if subshell is related to current atomtype,
                ! add functions
                if (int(charges(i))==finfo(j,4) ) then

                do iexp=1,finfo(j,1)
                    do im=-finfo(j,2),finfo(j,2)
                        new_basis%functions(k) = SolidGaussian_init(finfo(j,2),im,&
basis_object%exponentials(iexp,j), positions(:,i)  )
                        k = k +1
                    end do

                end do
                ! end of atom type check
                end if

            ! end of loop over subshells
            end do
        ! end of loop over atoms
        end do
        new_basis%charges   = charges
        new_basis%positions = positions

        print *, 'size charges', size(charges)
        print *, 'size positions', size(positions)

        ! deallocate
        deallocate(nonzeros)
        deallocate(finfo)
        deallocate(fperatomtype)

    end function

    ! SCF iterations with SolidGaussianBasis
    function SolidGaussianBasis_scf(basis,nocc,imaxin) result(coefficients)
        type(SolidGaussianBasis), intent(in)  :: basis
        integer, intent(in)                   :: nocc 
        integer,optional                      :: imaxin
        real(REAL64), allocatable             :: coefficients(:,:)
        integer                               :: imax
        ! matrices
        real(REAL64), allocatable             :: overlap_matrix(:,:), &
                                                 hamiltonian_matrix(:,:), &
                                                 two_electron_integrals(:,:,:,:), &
                                                 fock_matrix(:,:)
        real(REAL64), allocatable             :: density_matrix(:,:), &
                                                 orthogonalizing_matrix(:,:), &
                                                 eigen_vectors(:,:),eigen_values(:)
        ! energy stuff
        real(REAL64), allocatable             :: energies(:), old_energies(:) 
        integer                               :: i,N
        integer                               :: j
        
        ! testing matrix element calculation
        type(LobeBasis)   :: nb
        real(REAL64), allocatable  :: s2(:,:), h2(:,:),d2(:,:,:,:)
        integer                    :: i1,j1,k1,l1
        real(REAL64)               :: delta(24)
        real(REAL64)               :: virhe
        real(REAL64), allocatable  :: c1(:,:),c1old(:,:)
        real(REAL64)               :: cdelta

        if(present(imaxin)) then
            imax = imaxin
        else
            ! 10 iterations should be enough
            imax = 10
        end if

        ! init
        overlap_matrix = basis%calculate_overlap_matrix()
        hamiltonian_matrix = basis%calculate_hamiltonian_matrix()
        two_electron_integrals = basis%calculate_two_electron_integrals()

print *, 'START INTEGRAL COMPARISON'
nb =  LobeBasisFromSHBasis(basis)
s2 = nb%calculate_overlap_matrix()
h2 = nb%calculate_hamiltonian_matrix()
!d2 = nb%calculate_two_electron_integrals()
print *, 'overlap'
print *, 'maxval overlap diff', maxval(overlap_matrix-s2)
print *, 'minval overlap diff', minval(overlap_matrix-s2), &
minloc(overlap_matrix-s2)
print *, 'hamiltonian'
print *, 'maxval  diff', maxval(hamiltonian_matrix-h2), &
 maxloc(hamiltonian_matrix-h2)
print *, 'minval  diff', minval(hamiltonian_matrix-h2), &
 minloc(hamiltonian_matrix-h2)
print *, 'hamiltonian total diff', sum(hamiltonian_matrix-h2), &
sum(hamiltonian_matrix-h2)/size(h2)

print *, 'two_electron_int'
!print *, 'maxval 2 el diff', maxval(two_electron_integrals -d2), &
!maxloc(two_electron_integrals -d2)
!print *, 'minval 2 el diff', minval(two_electron_integrals-d2), &
! minloc(two_electron_integrals -d2)
print *, 'END INTEGRAL COMPARISON'

print *,'a'

        N = size(hamiltonian_matrix,1)
        ! initialize C with core hamiltonian
        allocate(coefficients(N,N))
        call matrix_eigensolver(hamiltonian_matrix,energies,coefficients)
!print *, 'core hamiltonian energies', energies

virhe=0d0
print *, 'overlap matrix element test'
do i=1,N
do j=1,N
!    print *, i,j, AngularGaussian_overlap_element(basis%functions(i)%primitives(1), &
!basis%functions(j)%primitives(1))- AngularGaussian_overlap_element_sp(basis%functions(i)%primitives(1), &
!basis%functions(j)%primitives(1)), basis%functions(i)%primitives(1)%lx, &
!basis%functions(i)%primitives(1)%ly, basis%functions(i)%primitives(1)%lz, &
! basis%functions(j)%primitives(1)%lx, &
!basis%functions(j)%primitives(1)%ly, basis%functions(j)%primitives(1)%lz
virhe = virhe + AngularGaussian_overlap_element(basis%functions(i)%primitives(1), &
basis%functions(j)%primitives(1)) -  AngularGaussian_overlap_element_sp(basis%functions(i)%primitives(1), &
basis%functions(j)%primitives(1))


end do
end do
print *, 'total difference in overlap elements', virhe, virhe/N**2

print *, 'end overlap matrix element test'



print *, 'kinetic energy test'
virhe = 0d0

do i=1,N
do j=1,N

if(i==5 .AND. j==5) then

    print *, i,j, AngularGaussian_kinetic_element(basis%functions(i)%primitives(1), &
basis%functions(j)%primitives(1)), AngularGaussian_kinetic_element_sp(basis%functions(i)%primitives(1), &
basis%functions(j)%primitives(1))

print *, basis%functions(i)%primitives(1)%alpha


end if

virhe = virhe +  AngularGaussian_kinetic_element(basis%functions(i)%primitives(1), &
basis%functions(j)%primitives(1))- AngularGaussian_kinetic_element_sp(basis%functions(i)%primitives(1), &
basis%functions(j)%primitives(1))


!print *,i,j, SolidGaussian_kinetic_element(basis%functions(i), &
!basis%functions(j)), &
! GaussianLobe_kinetic_element(nb%lobes(i), nb%lobes(j)), &
!SolidGaussian_kinetic_element(basis%functions(i), &
!basis%functions(j)) / GaussianLobe_kinetic_element(nb%lobes(i), nb%lobes(j)) , &
!(pi/(basis%functions(i)%alpha+ basis%functions(j)%alpha))**1.5d0, &
!(pi/(basis%functions(i)%alpha+ basis%functions(j)%alpha))**(-1.5d0)
!virhe = virhe + SolidGaussian_kinetic_element(basis%functions(i), &
!basis%functions(j)) - GaussianLobe_kinetic_element(nb%lobes(i), nb%lobes(j))
!print *, i,j, basis%functions(i)%alpha,  basis%functions(j)%alpha, &
!nb%lobes(i)%alpha, nb%lobes(j)%alpha
!if (abs( SolidGaussian_kinetic_element(basis%functions(i),basis%functions(j)) -&
!GaussianLobe_kinetic_element(nb%lobes(i), nb%lobes(j)) ) > 1d-1 ) then
!print *, i,j, 'BIG DIFFERENCE IN KINETIC MATRIX ELEMENT'
!print *, SolidGaussian_kinetic_element(basis%functions(i),basis%functions(j)) -&
!GaussianLobe_kinetic_element(nb%lobes(i), nb%lobes(j)), &
! SolidGaussian_kinetic_element(basis%functions(i),basis%functions(j))  , &
!GaussianLobe_kinetic_element(nb%lobes(i), nb%lobes(j))

!print *, 'exponents', basis%functions(i)%alpha,  basis%functions(j)%alpha, &
!nb%lobes(i)%alpha, nb%lobes(j)%alpha
!print *, 'centers', basis%functions(i)%center, basis%functions(j)%center, &
!nb%lobes(i)%center, nb%lobes(j)%center
!print *, 'dAB**2', dot_product(basis%functions(i)%center-basis%functions(j)%center, &
!basis%functions(i)%center-basis%functions(j)%center), &
!dot_product(nb%lobes(i)%center-nb%lobes(j)%center, nb%lobes(i)%center-nb%lobes(j)%center)
!print *, 'ratio', SolidGaussian_kinetic_element(basis%functions(i),basis%functions(j)) / &
!GaussianLobe_kinetic_element(nb%lobes(i), nb%lobes(j))
!print *, 'alpha + beta', nb%lobes(i)%alpha + nb%lobes(j)%alpha
!print *, '(alpha * beta)/(alpha+beta)', (nb%lobes(i)%alpha * nb%lobes(j)%alpha) / &
!(nb%lobes(i)%alpha + nb%lobes(j)%alpha)
!print *, '(alpha * beta)\(alpha+beta)', (nb%lobes(i)%alpha + nb%lobes(j)%alpha) / &
!(nb%lobes(i)%alpha * nb%lobes(j)%alpha)

 


!end if

end do
end do
print *, "TOTAL ERROR IN KINETIC ERROR:", virhe, virhe/N**2

print *, 'end kinetic energy test'

print *, 'start nuclear matrix element test'

virhe=0d0
do i = 1,N
do j = 1,i
!print *, i, j, SolidGaussian_nuclear_element(basis%functions(i), &
!basis%functions(j), nb%charges, nb%positions), &
!GaussianLobe_nuclear_element(nb%lobes(i),nb%lobes(j), nb%charges, nb%positions) 
!    print *, i,j, AngularGaussian_nuclear_element(basis%functions(i)%primitives(1), &
!basis%functions(j)%primitives(1), basis%charges, basis%positions), &
!AngularGaussian_nuclear_element_sp(basis%functions(i)%primitives(1), &
!basis%functions(j)%primitives(1), basis%charges, basis%positions )
virhe = virhe + AngularGaussian_nuclear_element(basis%functions(i)%primitives(1), &
basis%functions(j)%primitives(1), basis%charges, basis%positions)- &
AngularGaussian_nuclear_element_sp(basis%functions(i)%primitives(1), &
basis%functions(j)%primitives(1), basis%charges, basis%positions )

if (i==5 .AND. j==5) then
print *, i,j, AngularGaussian_nuclear_element(basis%functions(i)%primitives(1), &
basis%functions(j)%primitives(1), basis%charges, basis%positions), &
AngularGaussian_nuclear_element_sp(basis%functions(i)%primitives(1), &
basis%functions(j)%primitives(1), basis%charges, basis%positions ), &
hamiltonian_matrix(i,j), basis%functions(i)%l, basis%functions(i)%m, &
basis%functions(i)%coeffs


end if


end do
end do
print *, 'TOTAL ERROR IN NUCLEAR ERROR', virhe, virhe/N**2

print *, 'end nuclear matrix element test'

print *, 'start two electron integral test'

delta = 0d0
virhe = 0d0

do i1=1,N
do j1=1,N
do k1=1,N
do l1=1,N
!print *, i1,j1,k1,l1, SolidGaussian_two_electron_integral(basis%functions(i1), &
!basis%functions(j1), basis%functions(k1), basis%functions(l1) ) , &
!two_electron_lobe(nb%lobes(i1), nb%lobes(j1), nb%lobes(k1), nb%lobes(l1) ) 
!print *, d2(i1,j1,k1,l1)/two_electron_integrals(i1,j1,k1,l1)
!print *, 'old', d2(i1,j1,k1,l1), 'new', two_electron_integrals(i1,j1,k1,l1), &
!two_electron_integrals(i1,k1,j1,l1), two_electron_integrals(i1,j1,l1,k1), &
!two_electron_integrals(i1,k1,l1,j1), two_electron_integrals(i1,l1,j1,k1), &
!two_electron_integrals(i1,l1,k1,j1), two_electron_integrals(j1,i1,l1,k1), &
!two_electron_integrals(j1,i1,k1,l1), two_electron_integrals(j1,i1,l1,k1)
!delta(1) = delta(1) - d2(i1,j1,k1,l1) + two_electron_integrals(i1,j1,k1,l1)
!delta(2) = delta(2) - d2(i1,j1,k1,l1) + two_electron_integrals(i1,j1,l1,k1)
!delta(3) = delta(3) - d2(i1,j1,k1,l1) + two_electron_integrals(i1,k1,j1,l1)
!delta(4) = delta(4) - d2(i1,j1,k1,l1) + two_electron_integrals(i1,k1,l1,j1)
!delta(5) = delta(5) - d2(i1,j1,k1,l1) + two_electron_integrals(i1,l1,k1,j1)
!delta(6) = delta(6) - d2(i1,j1,k1,l1) + two_electron_integrals(i1,l1,j1,k1)
!
!delta(7) = delta(7) - d2(i1,j1,k1,l1) + two_electron_integrals(j1,i1,k1,l1)
!delta(8) = delta(8) - d2(i1,j1,k1,l1) + two_electron_integrals(j1,i1,l1,k1)
!delta(9) = delta(9) - d2(i1,j1,k1,l1) + two_electron_integrals(j1,k1,i1,l1)
!delta(10) = delta(10) - d2(i1,j1,k1,l1) + two_electron_integrals(j1,k1,l1,i1)
!delta(11) = delta(11) - d2(i1,j1,k1,l1) + two_electron_integrals(j1,l1,k1,i1)
!delta(12) = delta(12) - d2(i1,j1,k1,l1) + two_electron_integrals(j1,l1,i1,k1)
!
!delta(13) = delta(13) - d2(i1,j1,k1,l1) + two_electron_integrals(k1,i1,j1,l1) 
!delta(14) = delta(14) - d2(i1,j1,k1,l1) + two_electron_integrals(k1,i1,l1,j1) 
!delta(15) = delta(15) - d2(i1,j1,k1,l1) + two_electron_integrals(k1,j1,i1,l1) 
!delta(16) = delta(16) - d2(i1,j1,k1,l1) + two_electron_integrals(k1,j1,k1,i1) 
!delta(17) = delta(17) - d2(i1,j1,k1,l1) + two_electron_integrals(k1,l1,i1,j1) 
!delta(18) = delta(18) - d2(i1,j1,k1,l1) + two_electron_integrals(k1,l1,j1,i1) 
! 
!delta(19) = delta(19) - d2(i1,j1,k1,l1) + two_electron_integrals(l1,i1,j1,k1) 
!delta(20) = delta(20) - d2(i1,j1,k1,l1) + two_electron_integrals(l1,i1,k1,j1) 
!delta(21) = delta(21) - d2(i1,j1,k1,l1) + two_electron_integrals(l1,j1,i1,k1) 
!delta(22) = delta(22) - d2(i1,j1,k1,l1) + two_electron_integrals(l1,j1,k1,i1) 
!delta(23) = delta(23) - d2(i1,j1,k1,l1) + two_electron_integrals(l1,k1,i1,j1) 
!delta(24) = delta(24) - d2(i1,j1,k1,l1) + two_electron_integrals(l1,k1,j1,i1) 
!
!print *, d2(i1,j1,k1,l1) - two_electron_integrals( i1,j1,k1,l1 ), 'HMM'
!print *, i1,j1,k1,l1,SolidGaussian_two_electron_integral(basis%functions(i1),&
!basis%functions(j1), basis%functions(k1), basis%functions(l1) ) , &
!two_electron_lobe(nb%lobes(i1), nb%lobes(j1), nb%lobes(k1), nb%lobes(l1) ), &
!SolidGaussian_two_electron_integral(basis%functions(i1),&
!basis%functions(j1), basis%functions(k1), basis%functions(l1) ) - &
!two_electron_lobe(nb%lobes(i1), nb%lobes(j1), nb%lobes(k1), nb%lobes(l1) )
!print *, d2(i1,j1,k1,l1)-two_electron_lobe(nb%lobes(i1), nb%lobes(j1), &
!nb%lobes(k1), nb%lobes(l1) )
!print *, nb%lobes(i1)%c,  nb%lobes(j1)%c,  nb%lobes(k1)%c,  nb%lobes(l1)%c

!print *, i1,j1,k1,l1,SolidGaussian_two_electron_integral(basis%functions(i1),&
!basis%functions(j1), basis%functions(k1), basis%functions(l1) ) , &
!AngularGaussian_two_electron_integral( basis%functions(i1)%primitives(1), &
!basis%functions(j1)%primitives(1), basis%functions(k1)%primitives(1), &
!basis%functions(l1)%primitives(1) ) , &
!two_electron_lobe(nb%lobes(i1), nb%lobes(j1), nb%lobes(k1), nb%lobes(l1) ), &
!'HMMMM'
!print *, d2(i1,j1,k1,l1)- two_electron_integrals(i1,j1,k1,l1)
!virhe= virhe + d2(i1,j1,k1,l1)- two_electron_integrals(i1,j1,k1,l1)
!print *, SolidGaussian_two_electron_integral(basis%functions(i1),&
!basis%functions(j1), basis%functions(k1), basis%functions(l1) ) - &
! two_electron_integrals(i1,j1,k1,l1)

end do
end do
end do
end do
!print *, 'PRODUCTS', product(d2), product(two_electron_integrals)
!print *, 'SUMS', sum(d2), sum(two_electron_integrals), sum(d2)/sum(two_electron_integrals )
!print *, 'SUM DIFF', sum(d2-two_electron_integrals), sum(d2-two_electron_integrals)/N**4
!print *, 'errors 1-24: ', delta( 1:24 )

print *, 'TOTAL ERROR IN 2 EL INTEGRALS', virhe

print *, 'end two electron integral test'

!print *, 'maxval 2 el', maxval(two_electron_integrals), &
!maxloc(two_electron_integrals)
!print *, 'minval 2 el', minval(two_electron_integrals)
!print *, maxval(d2), minval(d2), maxloc(d2)
!print *, 'loc1', maxloc(two_electron_integrals-d2) 
!print *, 'loc2', minloc(two_electron_integrals-d2)

print *,'a2'
        coefficients = normalize_orbitals(coefficients,overlap_matrix)
print *,'a3'

        density_matrix = 0d0*coefficients
        fock_matrix = hamiltonian_matrix

        call matrix_eigensolver(overlap_matrix, eigen_values, eigen_vectors)
print *,'a4'
        orthogonalizing_matrix = 0d0*overlap_matrix
        do i=1,size(overlap_matrix,1)
            print *, eigen_values(i)            
            !eigen_values(i) = 1d-6+ eigen_values(i) - eigen_values(1)
            !if(eigen_values(i)<0d0) eigen_values(i)=1d10
            !orthogonalizing_matrix(i,i) = 1d0/sqrt(abs(eigen_values(i)))
            orthogonalizing_matrix(i,i) = 1d0/sqrt(eigen_values(i))
            !if(eigen_values(i)<0d0) orthogonalizing_matrix(i,i) = 1d20
        end do
print *, 'b'
        orthogonalizing_matrix = xmatmul(eigen_vectors, &
            xmatmul(orthogonalizing_matrix,transpose(eigen_vectors)))
!            xmatmul(orthogonalizing_matrix,matrix_inverse(eigen_vectors)))
print *, 'b2'
        fock_matrix = xmatmul(transpose(orthogonalizing_matrix), &
!        fock_matrix = xmatmul(matrix_inverse(orthogonalizing_matrix), &
            xmatmul(fock_matrix,orthogonalizing_matrix))
print *, 'b3'
        call matrix_eigensolver(fock_matrix,energies,coefficients)
print *, 'core hamiltonian orbital energies', energies
print *,'a5'
        coefficients = xmatmul(orthogonalizing_matrix,coefficients)
        !coefficients = normalize_orbitals(coefficients,overlap_matrix)
print *, 'a6'
!return 
!print *, 'should be identity: ', xmatmul(orthogonalizing_matrix, transpose( orthogonalizing_matrix ))
!print *, 'should be identity: ', xmatmul(orthogonalizing_matrix, matrix_inverse( orthogonalizing_matrix ))
c1=coefficients(:,1:nocc)
cdelta =1d0

        ! iterate
        i = 0
        old_energies=energies+1d0
        ! millihartree convergence is enough
        !do while (i<imax .AND. maxval(abs(energies(1:nocc)-old_energies(1:nocc)))>1d-3  )
        do while (i<200 .AND. (maxval(abs(energies(1:nocc)-old_energies(1:nocc)))>1d-3 .OR.  cdelta>1d-3 ) )
c1old=c1
print *, 'iteration start'
print *, 'orbital energies', energies
            old_energies=energies
            density_matrix = density_matrix_from_coefficients(coefficients,nocc)
print *, 'density matrix done'
            ! NOTICE name of function called
            fock_matrix =lobe_fock_matrix(hamiltonian_matrix, &
                                          two_electron_integrals, &
                                          density_matrix)
print *, 'fock matrix done'
print *, 'fock matrix maxval', maxval(fock_matrix)
print *, 'fock matrix minval', minval(fock_matrix)

            fock_matrix = xmatmul(transpose(orthogonalizing_matrix), &
            !fock_matrix = xmatmul(matrix_inverse(orthogonalizing_matrix), &
                                  xmatmul(fock_matrix,orthogonalizing_matrix))
print *, 'fock matrix transformed'
print *, 'fock matrix maxval', maxval(fock_matrix)
print *, 'fock matrix minval', minval(fock_matrix)
!

            call matrix_eigensolver(fock_matrix,energies,coefficients)
print *, 'fock matrix diagonalized'
            coefficients = xmatmul(orthogonalizing_matrix,coefficients)
print *, 'coefficients transformed'
            !coefficients = normalize_orbitals(coefficients,overlap_matrix)
!print *, 'orbitals normalized'
            i=i+1
c1=coefficients(:,1:nocc)
cdelta=sum(abs(c1-c1old))
print *, 'c1 delta', cdelta
            print *, 'orbital energy differences',i , energies(1:nocc)-old_energies(1:nocc)
            print *, 'orbital energies', energies(1:nocc)
        end do
        
        print *, 'number of iterations done ',i

    end function

    !> guess generation
    ! mos with structure_object and mould
    function SolidGaussian_mos_from_structure(structure_object,mould, norbitals,nocc, basis_object) result(orbitals)
        type(Structure), intent(in)         :: structure_object
        type(Function3D), intent(in)        :: mould
        integer, intent(in)                 :: nocc
        type(Basis)                         :: basis_object
        type(Function3D), allocatable       :: orbitals(:)
        integer                             :: norbitals
!
        real(REAL64), allocatable           :: coefficients(:,:)
        integer                             :: i, j
        integer                             :: l, m 
        integer                             :: centercounter, fcounter
        real(REAL64), pointer               :: bubvalues(:)
        real(REAL64), pointer               :: r(:)
        type(Grid1D), pointer               :: gridp
        type(SolidGaussianBasis)            :: basis_
        integer                             :: fi
        real(REAL64), allocatable           :: temp1(:)
        real(REAL64), allocatable           :: s(:)



        print *, 'begin initial guess generation', nocc, norbitals


        allocate(orbitals(norbitals))

        basis_ = SolidGaussianBasis(structure_object%coordinates,&  
                                   structure_object%nuclear_charge, basis_object)

        !call basis_%normalize()
        print *, 'basis generator done'

        !coefficients = SolidGaussianBasis_scf(basis_, nocc)
        !coefficients = SolidGaussianBasis_scf_v2(basis_, nocc)
        coefficients = SolidGaussianBasis_core_orbitals(basis_, nocc)

        print *,'content to orbitals'

        ! place content of guess to orbitals
        do i=1,norbitals
print *, 'orbital', i
            orbitals(i) = 0d0*mould
            ! loop over bubbles centers
            fcounter=1
            do centercounter = 1, size(basis_%positions,2)
print *, 'centercounter', centercounter
print *, size(basis_%positions,2)
print *, 'center charge', mould%bubbles%get_z(centercounter)

                gridp=> orbitals(i)%bubbles%get_grid(centercounter)
                r=>gridp%get_coord()
                !bubvalues => orbitals(i)%bubbles%get_f(centercounter,0,0)
                ! loop over functions on center
                do fi = 1, basis_%functions_on_atom(centercounter)
                    l = basis_%functions(fcounter)%l
                    m = basis_%functions(fcounter)%m
                    bubvalues => orbitals(i)%bubbles%get_f(centercounter,&
basis_%functions(fcounter)%l, basis_%functions(fcounter)%m)
!print *, 'l', basis_%functions(fcounter)%l
!print *, 'm', basis_%functions(fcounter)%m
!print *, 'center', basis_%positions(:,centercounter)
!print *, 'exponent', basis_%functions(fcounter)%alpha
!print *, 'coefficient', coefficients(fcounter,i) 
!print *, 'r**k, k is: ', orbitals(i)%bubbles%get_k()


!                    bubvalues = bubvalues + exp(-basis_%functions(fcounter)%alpha*r**2) * &
!r**(-l)*coefficients(fcounter,i) / sqrt(4*pi/(2*l+1d0))
                    temp1= exp(-basis_%functions(fcounter)%alpha* r**2) * &
 coefficients(fcounter,i) ! * sqrt(4*pi/(2*l+1d0))
                    if(l>0) temp1 = temp1 * r**l

! scale by charge of nucleus
temp1 = temp1/mould%bubbles%get_z(centercounter)

! 
!temp1 = temp1*sqrt(gamma(l-abs(m)+1d0)/gamma(l+abs(m)+1d0))

                    bubvalues = bubvalues + temp1

                    fcounter = fcounter + 1

!print *, 'maxval bub', maxval(bubvalues)
!print *, 'minval bub', minval(bubvalues)
                    
                end do
                
!                do j=1,lobe_basis%functions_on_atom(centercounter)
!                    bubvalues=bubvalues+lobe_basis%lobes(fcounter)%c*&
!                        exp(-lobe_basis%lobes(fcounter)%alpha*r**2) * coefficients(fcounter,i)
!                    fcounter = fcounter +1
!                end do


            end do
        print *, 'orbital generated'
        !print *, i,'orbital norm squared', orbitals(i).dot.orbitals(i)
        
        end do


        ! cleanup
        nullify(r)
        nullify(bubvalues)
        nullify(gridp)
        print *,'initial guess generation done'

    end function

    ! normalize basis
    subroutine SolidGaussianBasis_normalize(self)
        class(SolidGaussianBasis), intent(inout)    :: self
        integer                                     :: i
        real(REAL64)                                :: temp

        do i=1,size(self%functions)
            temp = SolidGaussian_overlap_element( self%functions(i), &
self%functions(i) ) + 1d-6
!print *, 'temp', temp
            self%functions(i)%coeffs = self%functions(i)%coeffs/sqrt(temp) 
            temp = SolidGaussian_overlap_element( self%functions(i), &
self%functions(i) )
!print *, 'temp', temp


        end do
    end subroutine

    ! test procedures
    function LobeFromSHGasussian(A) result(res)
        type(SolidGaussian), intent(in)   :: A
        type(GaussianLobe)                :: res
        res = GaussianLobe(A%alpha, A%center, 1d0)
    end function
    function LobeBasisFromSHBasis(old) result(new)
        type(SolidGaussianBasis), intent(in)  :: old
        type(LobeBasis)                       :: new
        integer                               :: i
        allocate(new%lobes( size( old%functions ) ))
        do i= 1, size(old%functions)
            new%lobes(i) =  LobeFromSHGasussian( old%functions(i))
        end do
        new%charges = old%charges
        new%positions = old%positions
        new%functions_on_atom = old%functions_on_atom
    end function 

    ! SCF iterations with SolidGaussianBasis
    function SolidGaussianBasis_scf_v2(basis,nocc,imaxin) result(coefficients)
        type(SolidGaussianBasis), intent(in)  :: basis
        integer, intent(in)          :: nocc 
        integer,optional             :: imaxin
        real(REAL64), allocatable    :: coefficients(:,:)
        integer                      :: imax
        ! matrices
        real(REAL64), allocatable    :: overlap_matrix(:,:), &
                                        hamiltonian_matrix(:,:), &
                                        two_electron_integrals(:,:,:,:), &
                                        fock_matrix(:,:)
        real(REAL64), allocatable    :: density_matrix(:,:), &
                                        orthogonalizing_matrix(:,:), &
                                        eigen_vectors(:,:),eigen_values(:)
        ! energy stuff
        real(REAL64), allocatable    :: energies(:), old_energies(:) 
        integer                      :: i,N


        if(present(imaxin)) then
            imax = imaxin
        else
            ! 10 iterations should be enough
            imax = 10
        end if

        ! init
        overlap_matrix = basis%calculate_overlap_matrix()
        hamiltonian_matrix = basis%calculate_hamiltonian_matrix()
        two_electron_integrals = basis%calculate_two_electron_integrals()

        N = size(hamiltonian_matrix,1)
        ! initialize C with core hamiltonian
        allocate(coefficients(N,N))
print *, "N is", N
print *, 'diagonalizing core hamiltonian', hamiltonian_matrix, energies, coefficients
        call matrix_eigensolver(hamiltonian_matrix,energies,coefficients)
        coefficients = normalize_orbitals(coefficients,overlap_matrix)

!            density_matrix = density_matrix_from_coefficients(coefficients,nocc)
!            fock_matrix =lobe_fock_matrix(hamiltonian_matrix, &
!                                          two_electron_integrals, &
!                                          density_matrix)
        !density_matrix = 0d0*coefficients
print *, 'orbital eigenvalues', energies
        fock_matrix = hamiltonian_matrix
print *, 'diagonalizing overlap matrix'

        call matrix_eigensolver(overlap_matrix, eigen_values, eigen_vectors)
        orthogonalizing_matrix = 0d0*overlap_matrix
        do i=1,size(overlap_matrix,1)
            print *, eigen_values(i)
            orthogonalizing_matrix(i,i) = 1d0/sqrt(eigen_values(i))
        end do
        orthogonalizing_matrix = xmatmul(eigen_vectors, &
            xmatmul(orthogonalizing_matrix,transpose(eigen_vectors)))
        fock_matrix = xmatmul(transpose(orthogonalizing_matrix), &
        !fock_matrix = xmatmul(matrix_inverse(orthogonalizing_matrix), &
            xmatmul(fock_matrix,orthogonalizing_matrix))
print *, 'diagonalizing fock matrix'


        call matrix_eigensolver(fock_matrix,energies,coefficients)
        coefficients = xmatmul(orthogonalizing_matrix,coefficients)
        coefficients = normalize_orbitals(coefficients,overlap_matrix)
print *, 'orbital eigenvalues', energies
!return


        ! iterate
        i = 0
        old_energies=energies+1
        ! millihartree convergence is enough
        do while (i<imax .AND. maxval(abs(energies(1:nocc)-old_energies(1:nocc)))>1d-3  )
            old_energies=energies
print *, 'orbital energies', energies
            density_matrix = density_matrix_from_coefficients(coefficients,nocc)
            fock_matrix =lobe_fock_matrix(hamiltonian_matrix, &
                                          two_electron_integrals, &
                                          density_matrix)
            fock_matrix = xmatmul(transpose(orthogonalizing_matrix), &
            !fock_matrix = xmatmul(matrix_inverse(orthogonalizing_matrix), &
                                  xmatmul(fock_matrix,orthogonalizing_matrix))
            call matrix_eigensolver(fock_matrix,energies,coefficients)
            coefficients = xmatmul(orthogonalizing_matrix,coefficients)
            coefficients = normalize_orbitals(coefficients,overlap_matrix)
            i=i+1

            print *, 'orbital energy differences',i , energies(1:nocc)-old_energies(1:nocc)
            print *, 'orbital energies', energies(1:nocc)
        end do
        
        print *, 'number of iterations done ',i

    end function

    ! initialize with core orbitals
    function SolidGaussianBasis_core_orbitals(basis,nocc) result(coefficients)
        type(SolidGaussianBasis), intent(in)  :: basis
        integer, intent(in)          :: nocc 
        real(REAL64), allocatable    :: coefficients(:,:)
        ! matrices
        real(REAL64), allocatable    :: overlap_matrix(:,:), &
                                        fock_matrix(:,:)
        real(REAL64), allocatable    :: orthogonalizing_matrix(:,:), &
                                        eigen_vectors(:,:),eigen_values(:)
        ! energy stuff
        real(REAL64), allocatable    :: energies(:)
        integer                      :: i,N

        ! init
        overlap_matrix = basis%calculate_overlap_matrix()

        N = size(overlap_matrix,1)
        allocate(coefficients(N,N))
        fock_matrix = basis%calculate_hamiltonian_matrix()

        print *, 'diagonalizing overlap matrix'

        call matrix_eigensolver(overlap_matrix, eigen_values, eigen_vectors)
        orthogonalizing_matrix = 0d0*overlap_matrix
        do i=1,size(overlap_matrix,1)
            !print *, eigen_values(i)
            orthogonalizing_matrix(i,i) = 1d0/sqrt(eigen_values(i))
        end do
        orthogonalizing_matrix = xmatmul(eigen_vectors, &
            xmatmul(orthogonalizing_matrix,transpose(eigen_vectors)))
        fock_matrix = xmatmul(transpose(orthogonalizing_matrix), &
        !fock_matrix = xmatmul(matrix_inverse(orthogonalizing_matrix), &
            xmatmul(fock_matrix,orthogonalizing_matrix))
        print *, 'diagonalizing core hamiltonian'


        call matrix_eigensolver(fock_matrix,energies,coefficients)
        coefficients = xmatmul(orthogonalizing_matrix,coefficients)
        coefficients = normalize_orbitals(coefficients,overlap_matrix)
        !print *, 'orbital eigenvalues', energies

    end function

    






end module
