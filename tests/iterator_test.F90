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
!> Test CartIter and MultiPoleIter objects
!! 
!! The CartIter yields exponent triplets (a,b,c) of the monomials
!! x**a * y**b * z**c. The MultiPoleIter object yields all the solid
!! harmonics (l,m) S_l^m that contain the monomial (a,b,c) together with
!! the coefficients (weights).
!!
!! This test checks that the multipole moments of s- and p-type Gaussian
!! are reproduced correctly (unit exponents, Gaussian centered at origin).
!!
!! The spherically symmmetric s-type Gaussians have only the monopole
!! moment (unity in this test)
!!
!! p-type Gaussians have only the dipole moments ([0.5, 0.5, 0.5] in
!! this test).
!!
!! Caveat: The algorithm here is correct but prone to rounding error when
!! l is large. Currently, the tolerance is fairly loose (1e-8).
program iterator_test
    use CartIter_class
    use Globals_m
    use Harmonic_class
    use ISO_FORTRAN_ENV

    implicit none

    integer, parameter :: LMAX = 15 
    real(REAL64), parameter :: tol = 1d-8

    type(HarmonicIter) :: multi
    logical            :: continue_iteration
    type(CartIter) :: iter

    integer :: kappa(3), l, m, i, j, a, b, c
    real(REAL64) :: coeff, qlm_s, qlm_px, qlm_py, qlm_pz
    real(REAL64) :: qvec((LMAX+1)**2)

    qvec = 0.0d0
    iter = CartIter(3, lmax)
    multi = HarmonicIter(lmax)
    call iter%next(kappa, continue_iteration)
    do while(continue_iteration)

        a = kappa(1)
        b = kappa(2)
        c = kappa(3)

        ! These are the Cartesian integrals 
        qlm_s = cartesian_multipole_gauss([a,b,c])
        qlm_px = cartesian_multipole_gauss([a+1,b,c])
        qlm_py = cartesian_multipole_gauss([a,b+1,c])
        qlm_pz = cartesian_multipole_gauss([a,b,c+1])

        call multi%loop_over(kappa)
        ! Conversion to spherical basis
        do while(multi%next(coeff, l, m))

            if (abs(m)>l .or. a+b+c /= l) then
                call perror("ITERATOR TEST FAILED")
                stop
            endif

            i = idx(l,m)
            qvec(i) = qvec(i) + coeff*(qlm_s + qlm_px + qlm_py + qlm_pz)
        enddo
        call iter%next(kappa, continue_iteration)

    enddo
    call multi%destroy()
    call iter%destroy()

    ! Check that the multipoles are correct
    if (qvec(1)/=1.0d0 .or.&
        any(qvec(2:4)/=0.5d0) .or.&
        any(abs(qvec(5:)) > tol)) then

        call perror("ITERATOR TEST FAILED")
        i = maxloc(abs(qvec(5:)), 1)
        print *, lm(i), maxval(abs(qvec(5:)))

        stop

    endif

    call pinfo("ITERATOR TEST PASSED")

contains

    function cartesian_multipole_gauss(kappa) result(I)
        integer, intent(in) :: kappa(3)
        real(REAL64) :: I
        integer :: j, k

        I = 0.0d0
        if (all(mod(kappa,2) == 0)) then
            ! Double factorial
            I = 1
            do k = 1, 3
                do j=1, kappa(k) - 1, 2
                    I = I*j
                enddo
            enddo
            I = I / 2**(sum(kappa)/2)
        endif

    end function

end program
