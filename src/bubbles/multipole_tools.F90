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
!> @file multipole_tools.F90

!> An auxiliary module for the Function3D class
!!
!! Enables the evaluation of the multipole moments
!! of a Function3D object.
!!
!! Multipole moments are quantities given by
!!
!!  \f[ q_{lm}(\mathbf{P}) = \int
!!  S_l^m(\mathbf{r}-\mathbf{P})\rho(\mathbf{r})\;d^3r \f]
!!
!!  where \f$S_l^m\f$ is a real-valued regular solid harmonic.
module MultiPoleTools_m
    use ISO_FORTRAN_ENV
    use Harmonic_class
    use MultiPole_class
    use Grid_class
    use CartIter_class
    use Evaluators_class
    use RealSphericalHarmonics_class
    use Globals_m
    implicit none
    private

    public :: bubble_multipoles, point_charge_multipoles

contains

    !> Computes the multipole moments of a bubble center
    !!
    !!
    function bubble_multipoles(mp, from, bubbles_lmax, lmax, to) result(q)
        !> The radial integrals of all the radial functions at a bubble center
        real(REAL64), intent(inout) :: mp(:)
        !> The bubble center coordinates
        real(REAL64), intent(in) :: from(3)
        !> Bubbles maximum multipole order
        integer(INT32), intent(in) :: bubbles_lmax
        !> Maximum multipole order
        integer(INT32), intent(in) :: lmax
        !> Location of evaluation coordinates
        real(REAL64), intent(in), optional :: to(3)

        real(REAL64)                    :: q((lmax+1)**2)
        type(TranslationMatrix)         :: W
        integer                         :: l

        q = 0
        ! The scaling of the multipoles of a bubble center.  There are
        ! relatively few of these (up to d-type functions) in chemistry
        ! settings.
        do l=0, bubbles_lmax
            mp(l*l + 1 : l*l + 1 + 2*l) = &
                FOURPI * mp(l*l + 1 : l*l + 1 + 2*l) / (2*l + 1)
        enddo

        q(: (bubbles_lmax+1)**2) = mp
        ! Translate the bubble multipoles to a new center
        ! This translation will typically result in an infinite
        ! amount of non-zero multipole moments; we truncate at lmax.
        if (present(to)) then
            W = TranslationMatrix(lmax)
            call W%apply(q, from=from, to=to)
            call W%destroy()
        end if
    end function

    !> Compute the multipole moments of point charge up to order `lmax`.
    function point_charge_multipoles(lmax, charge, point, reference_point) result(multipole_moments)
        !> Maximum multipole order
        integer,      intent(in)           :: lmax
        !> Charge of the point charge
        real(REAL64), intent(in)           :: charge
        !> Coordinates of the point charge 
        real(REAL64), intent(in)           :: point(3)
        !> Coordinates of the reference point
        real(REAL64), intent(in)           :: reference_point(3)

        !!> Coordinates to which the multipoles are translated to after evaluation
        !real(REAL64), intent(in)           :: translate_to(3)
        ! multipole moments
        real(REAL64)                       :: multipole_moments((lmax+1)**2)
        real(REAL64)                       :: solid_harmonics(1, (lmax+1)**2)
        type(TranslationMatrix)            :: W
 
        
        real(REAL64)                       :: distance, distance_vector(3)

        integer                            :: ndim(3)
        integer                            :: i,j,k, nlip, kappa(3)

        type(RealRegularSolidHarmonics)    :: harmo
        real(REAL64)                       :: coefficient
        integer                            :: l,m
        type(CartIter)                     :: iter


        multipole_moments=0.d0
        harmo = RealRegularSolidHarmonics(lmax)
        solid_harmonics = harmo%eval(reshape(point, [3, 1]))
        !iter=CartIter(ndim=3,d=lmax)
        
        !distance_vector = point - reference_point
 

        ! iterate over all possible vector values, where all the fields with
        ! indices 1-3 are between 0 and lmax.
        do l = 0, lmax
             multipole_moments(l*l + 1 : l*l + 1 + 2*l) =  &
                         charge * solid_harmonics(1, l*l + 1 : l*l + 1 + 2*l)
        end do 
        ! if translate_to is present, transform the multipole moments to these coordinates
        !if (present(translate_to)) then
        W = TranslationMatrix(lmax)
        call W%apply(multipole_moments, from=point, to=reference_point)
        call W%destroy()
        
        !end if 
    end function

end module
