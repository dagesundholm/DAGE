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
!> @file analytical.F90
!! Analytical interaction energies.

!> Analytical interaction energies for selected charge density types.
!!
!! Currently supported charge densities are
!!  - [Gaussians](@ref gaussgenerator_class)
!!  - [Slater s-type functions](@ref slatergenerator_class)
!!
!! The routines implemented here enables one to benchmark numerically
!! obtained values for test systems in which analytical energies can be
!! easily computed.
!!
!! In the case of s-type Slater functions, both self-interaction and
!! nuclear-electronic interaction energies can be computed.
!!
!! \sa [Numerically computed interaction energies](@ref potential_class)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! NOTE ABOUT PBC's                                                         !
! For including PBC's, we add the lattice vectors a number of times. Every !
! loop we add the contribution of a new octahedron/diamond:                !
!                                     X                                    !
!                        X           X X                                   !
! First:    X  Second:  X X  Third: X   X                                  !
!                        X           X X                                   !
!                                     X                                    !
!                                                                          !
! All the points in a given layer k fulfill sum(c_i)=k                     !
! For each cell, we get the translational vector T_i. The contribution of  !
! the i-th cell is sum_jk(I_jk(R_jk+T_i))                                  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module Analytical_m
    use Globals_m
    use PBC_class
    use Grid_class
    implicit none

    private

    public :: slater_analytical_energy
    public :: nucint_analytical_energy
    public :: gauss_analytical_energy

contains

    !> Analytical self-interaction energy of a charge density consisting
    !! of s-type Slater functions.
    function slater_analytical_energy(grid, coeffs, expos, centers) result(E)
        !> A 3D grid containing PBC information
        type(Grid3D), intent(in)        :: grid
        real(REAL64), intent(in)        :: coeffs(:)
        real(REAL64), intent(in)        :: expos(size(coeffs))
        real(REAL64), intent(in)        :: centers(3,size(coeffs))

        real(REAL64)                    :: E

        integer(INT32)                  :: i, j
        real(REAL64)                    :: rij

        ! PBC parameters
        type(PBC), pointer              :: pbc_ptr
        logical                         :: pbc_start
        real(REAL64)                    :: En
        real(REAL64)                    :: lattice_v(3), t_i(3)
        integer(INT32)                  :: pbc_idx
        integer(INT32)                  :: tv(3)

        real(REAL64), parameter         :: cvg=1.d-10

        ! TODO: Check that l == 0 for the whole system

        pbc_ptr => grid%get_pbc()
        E = 0.0d0

        if(sum(coeffs)/=0.d0.and.pbc_ptr%any()) then
            call pprint("The unit cell is not neutral, and you have PBC's in one or more")
            call pprint("dimensions. Your self-interaction energy is infinite!")
            return
        end if

        ! Self-interaction of the original cell
        do i=1, size(coeffs)

            ! Self-interaction for the i-th Slater function
            E = E + Ui(coeffs(i), expos(i))
            do j=i+1, size(coeffs)
                ! The interaction between the i-th and the j-th Slater
                ! TODO: subshell coordinates are the same and Uij -> inf
                rij=sqrt(sum( (centers(:,i)-centers(:,j))**2 ))
                E = E + 2.d0*Uij(expos(i), expos(j),&
                                 coeffs(i), coeffs(j),&
                                 rij)
            end do
        end do

        if(.not.pbc_ptr%any()) return

        lattice_v = grid%get_delta()

        ! First loop is over "layers". pbc_idx=0
        pbcloop: do pbc_idx=1, pbc_ptr%max_3dshell()! Loop over octahedra with total length 2a
            pbc_start=.TRUE.
            En = 0.0d0
            do    ! Loop over cells belonging to the octahedron
                tv=pbc_ptr%next_tv(pbc_idx, pbc_start)! We get the i-th translational vector
                if(pbc_start) exit  ! We got the last vector in the octahedron
                t_i=lattice_v*tv
                do i=1, size(coeffs) !Loop over original functions
                    do j=1, size(coeffs) !Loop over images in the remote cell
                        rij=sqrt(sum( (centers(:,i)-centers(:,j) + t_i)**2 ))
                        En = En + Uij(expos(i), expos(j),&
                                      coeffs(i), coeffs(j),&
                                      rij)
                    end do
                end do
            end do
            E = E + En
            write(ERROR_UNIT, '(a,i6,a,e16.6,a,e16.6)',advance='no') &
                    char(13)//"Layer: ", pbc_idx,"  Contrib: ", En, " Threshold: ", cvg

            if (abs(En)<cvg) exit pbcloop
        end do pbcloop
        write (ERROR_UNIT, *)
    contains

        pure function Ui(q, zeta)
            real(REAL64), intent(in)  :: q, zeta
            real(REAL64)              :: Ui

            Ui = q**2 * 5.d0/8.d0*zeta
        end function

        pure function Uij(zetai, zetaj, qi, qj, rij)
            real(REAL64), intent(in)    :: zetai,&
                                           zetaj,&
                                           qi,&
                                           qj,&
                                           rij

            real(REAL64)                :: Uij

            real(REAL64)                :: ei,&
                                           ej,&
                                           f

            if(rij<=1.d-14) then
                Uij=qi*qj*zetai*zetaj*&
                    (zetai**2 + 3.d0*zetai*zetaj + zetaj**2)/&
                    (zetai+zetaj)**3
            else
                if (zetai == zetaj) then
                    ei = exp(-2.d0*zetai*rij)
                    f = zetai*rij
                    Uij = qi*qj*(1.d0-ei*(1.d0 + 11.d0/8.d0*f +&
                    3.d0/4.d0*f**2 + 1.d0/6.d0*f**3 ))/rij
                else
                    f = 1.d0/(zetai*zetai - zetaj*zetaj)
                    ei = exp(-2.d0*zetai*rij)
                    ej = exp(-2.d0*zetaj*rij)
                    Uij = (qi*qj/rij) * &
                          (1.d0 - zetai**4*zetaj**4*f**2*(ei*(1.d0 + zetai*rij)/zetai**4&
                           - ej*(1.d0 + zetaj*rij)/zetaj**4) -&
                           2.d0*zetai**4*zetaj**4*f**3*(ei/zetai**2 - ej/zetaj**2))
                end if
            end if
        end function
    end function

    !> Analytical nuclear-electron interaction energy between Slater
    !! s-type densities and a set of point charges.
    function nucint_analytical_energy(grid, coeffs, expos, centers, qnuc, ncenters) result(E)
        !> A 3D grid containing PBC information
        type(Grid3D), intent(in)        :: grid
        !> Coefficients of the Slater charge distributions
        real(REAL64), intent(in)        :: coeffs(:)
        !> Exponents of the Slater charge distributions
        real(REAL64), intent(in)        :: expos(size(coeffs))
        !> Centers of the Slater charge distributions
        real(REAL64), intent(in)        :: centers(3,size(coeffs))
        !> Nuclear charges
        real(REAL64), intent(in)        :: qnuc(:)
        !> Nuclear positions
        real(REAL64), intent(in)        :: ncenters(3,size(qnuc))

        real(REAL64)                    :: E

        integer(INT32)                  :: inuc, jrho
        real(REAL64)                    :: rij

        ! PBC parameters
        type(PBC), pointer              :: pbc_ptr
        logical                         :: pbc_start
        real(REAL64)                    :: En
        real(REAL64)                    :: lattice_v(3), t_i(3)
        integer(INT32)                  :: pbc_idx
        integer(INT32)                  :: tv(3)

        real(REAL64), parameter         :: cvg=1.d-10


        pbc_ptr => grid%get_pbc()

        E = 0.0d0

        if(sum(coeffs)/=0.d0.and.pbc_ptr%any()) then
            call pprint("the unit cell is not neutral, and you have pbc's in one or more")
            call pprint("dimensions. your self-interaction energy is infinite!")
            return
        end if

        ! TODO: Check that l == 0 for the whole system

        ! Energy of the original cell
        !Iterate over nuclei
        do inuc=1, size(qnuc)
            ! Iterate over charge distributions
            do jrho=1, size(coeffs)
                ! The interaction energy between the i-th nucleus and the
                ! j-th charge distribution
                rij=sqrt(sum( (ncenters(:,inuc)-centers(:,jrho) )**2 ))
                E = E + Uij(qnuc(inuc), coeffs(jrho), expos(jrho), rij)
            end do
        end do

        if(.not.pbc_ptr%any()) return

        lattice_v = grid%get_delta()

        ! First loop is over "layers". pbc_idx=0
        pbcloop: do pbc_idx=1, pbc_ptr%max_3dshell()! Loop over octahedra with total length 2a
            pbc_start=.TRUE.
            En = 0.0d0
            do    ! Loop over cells belonging to the octahedron
                tv=pbc_ptr%next_tv(pbc_idx, pbc_start)! We get the i-th translational vector
                if(pbc_start) exit  ! We got the last vector in the octahedron
                t_i=lattice_v*tv
                do inuc=1, size(qnuc) !Loop over original functions
                    do jrho=1, size(coeffs) !Loop over images in the remote cell
                        rij=sqrt(sum( (ncenters(:,inuc)-centers(:,jrho) + t_i )**2 ))
                        En = En + Uij(qnuc(inuc), coeffs(jrho), expos(jrho), rij)
                    end do
                end do
            end do
            E = E + En
            write(ERROR_UNIT,'(a,i6,a,e16.6,a,e16.6)',advance='no') &
                    char(13)//"Layer: ", pbc_idx,"  Contrib: ", En, " Threshold: ", cvg

            if (abs(En)<cvg) exit pbcloop
        end do pbcloop
        write (ERROR_UNIT,*)
    contains

        pure function Uij(qnuci, coeffj, expoj, rij)
                real(REAL64), intent(in) :: qnuci
                real(REAL64), intent(in) :: coeffj
                real(REAL64), intent(in) :: expoj
                real(REAL64), intent(in) :: rij

                real(REAL64) :: Uij

                if (rij<=1.d-14) then
                    Uij= expoj
                else
                    Uij = ( 1.d0-exp(-2.d0*expoj*rij)*(1.d0+expoj*rij) ) /rij
                end if
                Uij = -qnuci * coeffj * Uij
        end function
    end function

    !> Self-interaction energy of a charge
    !! density consisting of Gaussian centers.
    !!
    !! Interaction energy of two Gaussian centers:
    !!
    !! @f[ U_{ij} =
    !!   q_i q_j r_{ij}^{-1}\mathrm{erf}(r_{ij}\sqrt{(\alpha_i \alpha_j)/(\alpha_i + \alpha_j)})
    !! @f]
    !!
    !! Interaction energy of a Gaussian center with itself:
    !!
    !! @f[ U_{i} = q_i^2 \sqrt{2\alpha_i/\pi} @f]
    !!
    !! Total potential energy:
    !! @f[ U = \sum_i U_i + 2\sum_{j>i} U_{ij} @f]
    !!
    function gauss_analytical_energy(grid, coeffs, expos, centers) result(E)
        !> A 3D grid containing PBC information
        real(REAL64), intent(in)        :: coeffs(:)
        real(REAL64), intent(in)        :: expos(size(coeffs))
        real(REAL64), intent(in)        :: centers(3,size(coeffs))
        type(Grid3D), intent(in)        :: grid

        real(REAL64)                    :: E

        real(REAL64)                    :: rij
        integer                         :: i, j

        ! PBC params
        real(REAL64)                    :: En
        real(REAL64)                    :: lattice_v(3), t_i(3)
        integer(INT32)                  :: tv(3)

        type(PBC), pointer              :: pbc_ptr
        logical                         :: pbc_start
        integer(INT32)                  :: pbc_idx
        real(REAL64), parameter         :: cvg=1.d-10


        ! Get grid parameters for PBC
        pbc_ptr => grid%get_pbc()
        lattice_v = grid%get_delta()

        E = 0.0d0

        if(sum(coeffs)/=0.d0.and.pbc_ptr%any()) then
            call pprint("the unit cell is not neutral, and you have pbc's in one or more")
            call pprint("dimensions. your self-interaction energy is infinite!")
            return
        end if

        ! Self-interaction energy in the original domain
        do i=1, size(coeffs)
            ! Self-interaction for the i-th gaussian
            E = E + Ui(coeffs(i), expos(i))
            do j=i+1, size(coeffs)
                ! The interaction between the i-th and the j-th gaussians
                rij=sqrt(sum( (centers(:,i)-centers(:,j))**2 ))
                E = E + 2.d0*Uij(expos(i), expos(j),&
                                 coeffs(i), coeffs(j),&
                                 rij)
            end do
        end do

        if(.not.pbc_ptr%any()) return

        ! First loop is over "layers". pbc_idx=0
        pbcloop: do pbc_idx=1, pbc_ptr%max_3dshell()! Loop over octahedra with total length 2a
            pbc_start=.TRUE.
            En = 0.0d0
            do    ! Loop over cells belonging to the octahedron
                tv=pbc_ptr%next_tv(pbc_idx, pbc_start)! We get the i-th trangautional vector
                if(pbc_start) exit  ! We got the last vector in the octahedron
                t_i=lattice_v*tv
                do i=1, size(coeffs) !Loop over original functions
                    do j=1, size(coeffs) !Loop over images in the remote cell
                        rij=sqrt(sum( (centers(:,i)-centers(:,j) + t_i)**2 ))
                        En = En + Uij(expos(i), expos(j),&
                                      coeffs(i), coeffs(j),&
                                      rij)
                    end do
                end do
            end do
            E = E + En
            if(debug_g>=1) write(ERROR_UNIT,'(a,i6,a,e16.6,a,e16.6)',advance='no') &
                    char(13)//"Layer: ", pbc_idx,"  Contrib: ", En, " Threshold: ", cvg

            if (abs(En)<cvg) exit pbcloop
        end do pbcloop
        if(debug_g>=1) write (ERROR_UNIT,*)
    contains
        pure function Ui(q, alpha)
                real(REAL64), intent(in)  :: q,&
                                             alpha

                real(REAL64)              :: Ui
                Ui = q*q*sqrt(2.d0*alpha/PI)
        end function

        function Uij(alphai, alphaj, qi, qj, rij)
                real(REAL64), intent(in)    :: alphai, alphaj
                real(REAL64), intent(in)    :: qi, qj
                real(REAL64), intent(in)    :: rij

                real(REAL64)                :: Uij

                real(REAL64)                :: sqrtmu
                real(REAL64)                :: temp

                sqrtmu = sqrt((alphai*alphaj)/(alphai+alphaj))
                temp = sqrtmu*rij

                if (temp < 0.01_REAL64) then
                    ! Expansion of erf(z) close to z = 0.0
                    Uij = TWOOVERSQRTPI*sqrtmu*&
                            (1.d0-0.33333333333333333d0*temp**2+0.1d0*temp**4)
                else if(temp < 10.0_REAL64) then
                    ! NB: derf is a GNU extension
                    Uij = derf(temp)/rij
                else
                    ! erf(z) tends to 1.0
                    Uij = 1.d0/rij
                end if
                Uij = qi*qj*Uij
        end function
    end function



end module

