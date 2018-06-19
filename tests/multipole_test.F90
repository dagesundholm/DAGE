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
!> Test the translation and interaction matrices defined in 
!! MultiPole_class
!!
!!
!! The test suite comprises three parts
!! * Test exclusively interaction matrix (point dipole interaction)
!! * Test exclusively translation matrix (translation of multipole)
!! * Integrated test of both matrices (all elements of the matrices need
!!  to be correct to pass this test)
module multipoles_tests_module
    use MultiPole_class
    use RealSphericalHarmonics_class
    use XMatrix_m
    use ISO_FORTRAN_ENV
    implicit none

contains

    function multipole_test_suite(lmax, output) result(ok)
        integer, intent(in) :: lmax
        logical, intent(in), optional :: output
        logical :: ok
        logical :: output_tmp

        ok = .false.
        if (present(output)) then
            output_tmp = output
        else
            output_tmp = .false.
        endif

        print *, 'interaction test'
        ok = interaction_test(lmax, output_tmp)
        if (.not.ok) return

        print *, 'translation test'
        ok = translation_test(lmax, output_tmp)
        if (.not.ok) return

        print *, 'displaced monopoles test'
        ok = displaced_monopoles_test(lmax, output_tmp)
        if (.not.ok) return

    end function

    !> Interaction matrix works
    !! Rotates a point dipole at around another point dipole in the xy plane
    !! and checks the interaction energy to an analytical expression
    !! (See Atkins, ed. 9, p. 654)
    function interaction_test(lmax, output) result(ok)
        integer, intent(in) :: lmax
        logical, intent(in) :: output
        logical :: ok

        integer, parameter :: seed = 86456
        integer, parameter :: npoints = 200
        real(REAL64), parameter :: PI = 3.14159265358979323846d0, tol = 1d-14

        type(InteractionMatrix) :: T
        real(REAL64) :: r(2,3), dr(3), q(2, (lmax+1)**2), muvec(3), d, theta, U, err
        real(REAL64) :: Tm((lmax+1)**2, (lmax+1)**2)
        integer :: i

        ok = .true.
        call srand(seed)

        r(1,:) = [rand(0), rand(0), rand(0)]
        r(2,:) = [rand(0), rand(0), rand(0)]
        muvec  = [rand(0), rand(0), rand(0)]

        T = InteractionMatrix(lmax)
        Tm = 0.0d0

        ! The dipole moments are parallel
        call make_point_dipole(q(1,:), muvec)
        call make_point_dipole(q(2,:), muvec)

        d = sqrt(sum((r(1,:)-r(2,:))**2))
        dr(3) = r(1,3) - r(2,3)
        do i=0, npoints-1
            theta = 2*PI*i/(npoints-1)
            dr(1) = d*cos(theta)
            dr(2) = d*sin(theta)
            Tm = T%eval(dr)
            U = dot_product(q(2,:), matmul(Tm, q(1,:)))
            err = abs(U-Uref(dr))
            if (output) print *, theta, err
            if (err > tol) then
                print *, 'LMAX', lmax
                print *, 'Error', err
                print '("INTERACTION TEST FAILED!")'
                ok = .false.
                exit
            endif
        enddo

        call T%destroy()
        return 
    contains

        function Uref(dr) result(U)
            real(REAL64), intent(in) :: dr(3)
            real(REAL64) :: U, r, mu, a
            mu = dot_product(muvec, muvec)
            a = dot_product(muvec, dr)
            r = sqrt(sum(dr**2))
            U = mu/r**3 - 3*a*a/r**5
        end function

    end function

    !> Test that multipole translation works
    !! 1. Take the multipoles of a pure monopole delta(r-old) expanded at point 'new'
    !! 2. Translate to 'old', observe that the multipoles are (1.0, 0.0, 0.0, 0.0...0.0)
    function translation_test(lmax, output) result(ok)
        integer, intent(in) :: lmax
        logical, intent(in) :: output
        logical :: ok

        type(TranslationMatrix) :: W
        type(RealRegularSolidHarmonics) :: S
        real(REAL64) :: old(3), new(3), q((lmax+1)**2), Svec((lmax+1)**2)
        integer, parameter :: seed = 83453
        real(REAL64), parameter :: tol = 1d-17

        ok = .true.
        call srand(seed)

        old  = [rand(0), rand(0), rand(0)]
        new  = [rand(0), rand(0), rand(0)]

        W = TranslationMatrix(lmax)
        S = RealRegularSolidHarmonics(lmax)

        ! Multipoles expanded about 'old'
        q = 0.0d0
        q(1) = 1.0d0

        ! Expand about 'new' point
        call W%apply(q, from=old, to=new)

        ! Reference values (regular solid harmonics)
        Svec = reshape(S%eval(reshape(old-new, [3, 1])), [3])

        if (any(abs(Svec-q) > tol)) then
            print '("#TRANSLATION TEST FAILED!")',
            ok = .false.
            goto 99
        endif

        ! Translate back to old
        call W%apply(q, from=new, to=old)
        if (q(1)/=1.0d0 .and. any(q(2:)/=0.0d0)) then
            print '("#TRANSLATION TEST FAILED!")',
            ok = .false.
            goto 99
        endif

        99 call W%destroy() 
        return
  end function

  function displaced_monopoles_test(lmax, output) result(ok)
        integer, intent(in) :: lmax
        logical, intent(in) :: output
        logical :: ok

        ! particles
        real(REAL64), parameter :: rj(3) = [-12.31523266d0,  -2.77677722d0, -13.87351588d0]
        real(REAL64), parameter :: ri(3) = [-13.80289944d0, -12.66934534d0, -8.92620243d0]

        ! boxes
        real(REAL64), parameter :: boxj(3) = [-10.625d0,  -2.125d0, -14.875d0]
        real(REAL64), parameter :: boxi(3) = [-14.875d0, -10.625d0, -10.625d0]

        real(REAL64), parameter :: err_ref(0:15) =  [6.4549767744542658d-003,&
                                                    3.5239702059782635d-003,&
                                                    3.6472898611333848d-005,&
                                                    3.5551144335181639d-004,&
                                                    1.2985977061118203d-004,&
                                                    3.0889425959704231d-005,&
                                                    1.8391214269969369d-006,&
                                                    2.0053722605550828d-006,&
                                                    1.0603102051365276d-006,&
                                                    2.9594595833681847d-007,&
                                                    3.4396637579869882d-008,&
                                                    1.2534625434823532d-008,&
                                                    9.0637385791181302d-009,&
                                                    2.9670064183306977d-009,&
                                                    4.8909977456990106d-010,&
                                                    6.3554106422003542d-011]
        
        type(InteractionMatrix) :: T
        type(TranslationMatrix) :: W
        real(REAL64), allocatable :: Wmx(:), Tmx(:,:), Wout(:,:)
        real(REAL64) :: qi((lmax+1)**2), qj((lmax+1)**2), Uref
        real(REAL64) :: err
        integer :: N, i, j

        ok = .true.

        T = InteractionMatrix(lmax)
        W = TranslationMatrix(lmax)
        qi = 0.0d0; qi(1) = 1.0d0
        qj = 0.0d0; qj(1) = 1.0d0

        N = (LMAX+1)**2
        allocate(Wout(N,N), source=0.0d0)

        Tmx = T%eval(boxj-boxi) 

        if (output) then
            open(32, file='Tmatrix.dat', action='write')
            do i=1, size(Tmx,1)
                write(32, *) Tmx(i,:)
            enddo
            close(32)
        endif

        Wmx = W%eval(rj - boxj) ! old - new
        call dtpmv('L', 'N', 'U', size(qj), Wmx, qj, 1)

        if (output) then
            do j=1, N ! column
                do i=j, N ! row
                    Wout(i,j) = Wmx(i + ((2*N-j)*(j-1)/2))
                enddo
            enddo
            open(32, file='Wmatrix_right.dat', action='write')
            do i=1,N
                write(32,*) Wout(i,:)
            enddo
            close(32)
        endif

        Wout = 0.0d0
        Wmx = W%eval(ri-boxi) ! old - new
        call dtpmv('L', 'N', 'U', size(qi), Wmx, qi, 1)

        if (output) then
            do j=1, N ! column
                do i=j, N ! row
                    Wout(i,j) = Wmx(i + ((2*N-j)*(j-1)/2))
                enddo
            enddo
            open(32, file='Wmatrix_left.dat', action='write')
            do i=1, N
                write(32,*) Wout(i,:)
            enddo
            close(32)
        endif

        Uref = 1.0d0/sqrt(sum((ri-rj)**2))

        if (lmax < size(err_ref)) then
            j = 0
            do i=0, lmax
                j = j + (2*i+1)
                err = abs(dot_product(qi(:j), matmul(Tmx(:j,:j), qj(:j))) - Uref)
                print *, err
                if (err_ref(i) /= err) then
                    print *, err_ref(i)
                    print *, err
                    print '("#DISPLACEMENT TEST FAILED!")',
                    print '("# reference errors not reproduced!")',
                    ok = .false.
                    goto 99
                endif
            enddo
        else
            print '("#DISPLACEMENT TEST FAILED!")',
            print '("# lmax too big!")',
            ok =.false.
            goto 99
        endif

        99 deallocate(Wmx)
           deallocate(Tmx)
           deallocate(Wout)
           call T%destroy()
           call W%destroy()

    end function

    subroutine make_monopole(q)
        real(REAL64), intent(out) :: q(:)

        q(1) = 1.0d0
        q(2:) = 0.0d0

    end subroutine

    subroutine make_point_dipole(q, mu)
        real(REAL64), intent(out) :: q(:)
        real(REAL64), intent(in) :: mu(3)

        q(1) = 0.0d0
        q(2) = mu(2) ! y
        q(3) = mu(3) ! z
        q(4) = mu(1) ! x
        q(5:) = 0.0d0

    end subroutine

    subroutine make_quadrupole(q, mom)
        real(REAL64), intent(out) :: q(:)
        real(REAL64), intent(in) :: mom(5)

        q(1:4) = 0.0d0
        q(5:9) = mom
        q(10:) = 0.0d0

    end subroutine

end module

program multipole_test
    use multipoles_tests_module
    use Globals_m
    implicit none

    if (multipole_test_suite(lmax=15, output=.false.)) then
        call pinfo("ALL TESTS PASSED")
        stop 0
    else
        call perror("SOME TESTS FAILED")
        stop 1
    endif

end program 

