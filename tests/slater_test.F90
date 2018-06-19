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
!> @file slater_test.F90
!! Test SlaterGenerator and SlaterPotGenerator classes.

!> Test SlaterGenerator and SlaterPotGenerator classes.
!! - TEST 1: Explicit SlaterGenerator initialization.
!! - TEST 2: Initialization SlaterGenerator by parsing input, multiple
!!           shells per center, with non-default grid step.
!! - TEST 3: SlaterPotGenerator.
!! - TEST 4: Bubbles IO.
!! - TEST 5: Nuclear interaction.
program slater_test
    use SlaterGenerator_class
    use Function3D_class
    use Coulomb3D_class
    use Grid_class
    use Potential_class
    use Globals_m
    implicit none

    type(SlaterGenerator)   :: rhogen
    type(SlaterPotGenerator):: potgen
    type(Function3D)        :: rho
    type(Function3D)        :: apot
    class(Function3D), allocatable :: pot

    type(Function3D)        :: rho2
    type(Function3D)        :: apot2

    real(REAL64)            :: selfint
    real(REAL64)            :: selfint_exact
    real(REAL64)            :: selfint_apot
    real(REAL64)            :: selfint_io
    real(REAL64)            :: nuc
    real(REAL64)            :: nuc_exact

    logical                 :: part_ok
    logical                 :: all_ok
    real(REAL64), parameter :: TOL=1.d-14 !Tolerance

    character(len=*), parameter :: HEADER=repeat(" ",29)//&
                                    "VALUE             REF              DIFF"

    all_ok =.TRUE.
    part_ok=.TRUE.

    debug_g=0

! %%%%%%%%%%%%%%%%%%%%%%    TEST  1    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    part_ok=.TRUE.
    call pinfo("TEST 1: Testing explicit SlaterGenerator initialization...")

    rhogen=SlaterGenerator(z      = [1.d0, 1.d0],&
                           centers= reshape(&
                                    [-0.8660254037844387d0, -0.5d0, 0.1d0, &
                                      0.8660254037844387d0,  0.5d0, 0.1d0],&
                                    [3,2] ),&
                           ibub   = [1, 2], &
                           l      = [0, 0], &
                           m      = [0, 0], &
                           expos  = [1.d0, 1.d0], &
                           coeffs = [1.d0, 1.d0])
    rho=rhogen%gen()
    
    call sleep(10)

    selfint= rho .dot. ( Coulomb3D(rho%grid, rho%bubbles) .apply. rho )
    selfint_exact=rhogen%selfint(rho%grid)
    call pinfo(HEADER)
    part_ok=compare_and_report(&
        "selfint-numerical",    selfint,        2.1019438083447435d0) < TOL &
        .and. part_ok
    part_ok=compare_and_report(&
        "selfint-analytical",   selfint_exact,  2.1019485856493985d0) < TOL &
        .and. part_ok

    if(part_ok) then
        call pinfo("PASSED!")
    else
        call perror("FAILED!")
    end if
    print*
    all_ok=all_ok .and. part_ok

! %%%%%%%%%%%%%%%%%%%%%%    TEST  2    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    part_ok=.TRUE.
    call pinfo("TEST 2: Testing initialization SlaterGenerator by parsing &
               &input, multiple")
    call pinfo("shells per center, with non-default grid step...")

    rhogen=SlaterGenerator([&
                            " 2 1.0     1.0   0.5 1.5  ",&
                            "      0  0  1.0 1.0       ",&
                            "      0  0  1.0 2.0       "&
                                                        ,&
                            " 1 1.0     -1.5  3.0  0.5 ",&
                            "      0  0  1.0 1.0       "&
                                                        ,&
                            " 3 2.0     -0.5 -1.5 -1.0 ",&
                            "      0  0  1.0 2.0       ",&
                            "      0  0  1.0 1.0       ",&
                            "      0  0  2.0 3.0       "&
                                                            ], step=.4d0)
    rho=rhogen%gen()

    selfint= rho .dot. ( Coulomb3D(rho%grid, rho%bubbles) .apply. rho )
    selfint_exact=rhogen%selfint(rho%grid)
    call pinfo(HEADER)
    part_ok=compare_and_report(&
        "selfint-numerical",    selfint,       31.8808479535843396d0) < TOL &
        .and. part_ok
    part_ok=compare_and_report(&
        "selfint-analytical",   selfint_exact, 31.8912297114622483d0) < TOL &
        .and. part_ok

    if(part_ok) then
        call pinfo("PASSED!")
    else
        call perror("FAILED!")
    end if
    print*
    all_ok=all_ok .and. part_ok

! %%%%%%%%%%%%%%%%%%%%%%    TEST  3    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    part_ok=.TRUE.
    call pinfo("TEST 3: Testing SlaterPotGenerator...")

    potgen=SlaterPotGenerator(rhogen)

    apot=potgen%gen()

    selfint_apot= rho .dot. apot
    selfint_exact=potgen%selfint(apot%grid)

    call pinfo(HEADER)
    part_ok=compare_and_report(&
        "selfint-num-apot",     selfint_apot,  31.8808479535829363d0) < TOL &
        .and. part_ok
    part_ok=compare_and_report(&
        "selfint-analytical",   selfint_exact, 31.8912297114622483d0) < TOL &
        .and. part_ok

    if(part_ok) then
        call pinfo("PASSED!")
    else
        call perror("FAILED!")
    end if
    print*
    all_ok=all_ok .and. part_ok

! %%%%%%%%%%%%%%%%%%%%%%    TEST  4    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    part_ok=.TRUE.
    call pinfo("TEST 4: Testing Bubbles IO...")

    call rho%dump ("rho_slater_test")
    call apot%dump("apot_slater_test")

    rho2= Function3D("rho_slater_test.fun")
    apot2=Function3D("apot_slater_test.fun")

    selfint_apot=rho  .dot. apot
    selfint_io  =rho2 .dot. apot2

    call pinfo(HEADER)
    part_ok=compare_and_report(&
        "selfint-io", selfint_io, selfint_apot) < TOL &
        .and. part_ok

    if(part_ok) then
        call pinfo("PASSED!")
    else
        call perror("FAILED!")
    end if
    print*
    all_ok=all_ok .and. part_ok

! %%%%%%%%%%%%%%%%%%%%%%    TEST  5    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    part_ok=.TRUE.
    call pinfo("TEST 5: Testing nuclear interaction...")

    allocate(pot, source = Coulomb3D(rho%grid, rho%bubbles) .apply. rho)

    nuc=nucint(pot)
    nuc_exact=potgen%nucint(pot%grid)

    call pinfo(HEADER)
    part_ok=compare_and_report(&
        "nucint-numerical",    nuc,       -26.3118089488346101d0) < TOL &
        .and. part_ok
    part_ok=compare_and_report(&
        "nucint-analytical",   nuc_exact, -26.3118089488348232d0) < TOL &
        .and. part_ok

    if(part_ok) then
        call pinfo("PASSED!")
    else
        call perror("FAILED!")
    end if
    print*
    all_ok=all_ok .and. part_ok

! %%%%%%%%%%%%%%%% CLEANUP
    call rho  %destroy()
    call rho2 %destroy()
    call apot %destroy()
    call apot2%destroy()

    open (1, file="rho_slater_test.fun")
    close(1, status="delete")
    open (1, file="apot_slater_test.fun")
    close(1, status="delete")
! %%%%%%%%% FINAL REPORT %%%%%%%%%%%%%%

    if(all_ok) then
        call pinfo("ALL TESTS PASSED")
        stop 0
    else
        call perror("SOME TESTS FAILED")
        stop 1
    end if

contains
    function compare_and_report(what, value, reference) result(diff)
        character(*),  intent(in) :: what
        real(REAL64),  intent(in) :: value
        real(REAL64),  intent(in) :: reference
        real(REAL64)              :: diff
        character(18)             :: tmp

        tmp=what
        diff=abs(value-reference)
        write(ppbuf,'(a18,2f20.16,e12.3)') tmp, value, reference, diff
        call pinfo(ppbuf)
    end function
end program
