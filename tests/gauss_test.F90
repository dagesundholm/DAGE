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
!> @file gauss_test.F90
!! Test GaussGenerator and GaussPotGenerator classes.

!> Test GaussGenerator and GaussPotGenerator classes.

!> Performs the following tests:
!! - TEST 1: Explicit GaussGenerator initialization.
!! - TEST 2: Initialization of GaussGenerator by parsing.
!! - TEST 3: Initialization of GaussGenerator by parsing and generation of
!!           function from non-default grid.
!! - TEST 4: Generation on explicit grid with PBCs.
!! - TEST 5: GaussPotGenerator initialization from GaussGenerator.
!! - TEST 6: GaussPotGenerator explicit initialization.
!! - TEST 7: GaussPotGenerator initialization by parsing.
program gauss_test_new
    use GaussGenerator_class
    use Function3D_class
    use Grid_class
    use Globals_m
    use Coulomb3D_class
    implicit none

    type(GaussGenerator)   :: rhogen        ! Generator for gaussian charge density
    type(GaussPotGenerator):: potgen
    type(Function3D)       :: rho
    type(Function3D)       :: apot
    type(Grid3D)           :: grid

    real(REAL64)           :: selfint       ! Self-Interaction energy
    real(REAL64)           :: selfint_exact ! Analytical Self-Interaction energy
    real(REAL64)           :: selfint_apot  !

    logical                 :: part_ok
    logical                 :: all_ok
    real(REAL64), parameter :: TOL=1.d-14 !Tolerance

    character(len=*), parameter :: HEADER=repeat(" ",29)//&
                                    "VALUE             REF              DIFF"

    all_ok =.TRUE. ! if all the tests are ok
    part_ok=.TRUE. ! if the currently calculated test is ok

! %%%%%%%%%%%%%%%%%%%%%%    TEST  1    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    part_ok=.TRUE.
    call pinfo("TEST 1: Testing explicit GaussGenerator initialization...")

    ! initialize gaussian generator
    rhogen=GaussGenerator(alpha  = [1.d0],&
                          center = reshape([0.d0, 0.d0, 0.d0],[3,1]),&
                          q      = [1.d0])

    ! rhogen%gen() returns a Function3D object representing charge density
    ! (contains only a cube, no bubbles)
    rho=rhogen%gen() 

    ! apply Coulomb operator to the generated charge density and get inner
    ! product with the charge density, i.e., self interaction energy of the
    ! charge density is calculated
    selfint= rho .dot. ( Coulomb3D(rho%grid, rho%bubbles) .apply. rho ) 

    ! get the analytically calculated self interaction energy
    selfint_exact=rhogen%selfint(rho%grid) 
    call pinfo(HEADER)

    ! compare the numerically calculated self interaction energy with the 
    ! number obtained from ???
    part_ok=compare_and_report(&
        "selfint-numerical",    selfint,        0.7979283927253593d0) < TOL &
        .and. part_ok

    ! compare the analytically calculated self interaction energy with the 
    ! number obtained from ???
    part_ok=compare_and_report(&
        "selfint-analytical",   selfint_exact,  0.7978845608028654d0) < TOL &
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
    ! Same test as test1 but the initialization of gaussiangenerator is done
    ! by parsing a string containing 5 different parameters for gaussians
    ! (charge, alpha, x, y, z)
    call pinfo("TEST 2: Testing initialization of GaussGenerator by parsing...")
    rhogen=GaussGenerator(["1.0   3.0   -7.0   0.0   0.0",&
                           "1.0   0.4    0.0   4.0   4.0",&
                           "1.0   2.0    4.0 -10.0  -2.0",&
                           "1.0   1.5    3.0   6.0  -5.0"])

    ! rhogen%gen() returns a Function3D object representing charge density
    ! (contains only a cube, no bubbles)
    rho=rhogen%gen()

    ! calculate self interaction energy
    selfint = rho .dot. ( Coulomb3D(rho%grid, rho%bubbles) .apply. rho )
    ! get the analytically calculated self interaction energy
    selfint_exact = rhogen%selfint(rho%grid)

    call pinfo(HEADER)
    part_ok = compare_and_report(&
        "selfint-numerical",    selfint,        4.9612607728784432d0) < TOL &
        .and. part_ok
    part_ok = compare_and_report(&
        "selfint-analytical",   selfint_exact,  4.9612771741430324d0) < TOL &
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
    call pinfo("TEST 3: Testing initialization of GaussGenerator by parsing,")
    call pinfo("        and generation of function from non-default grid...")
    ! the non-default is obtained by giving different step and thres, and nlip
    ! parameter values
    rhogen=GaussGenerator([" 1.0   2.0  -1.0 0.0 0.0",&
                           "-1.0   4.0   1.0 0.0 0.0"],&
                          step=.1d0, thres=1.d-8, nlip=5)

    rho=rhogen%gen()
    ! same things are done as in previous tests
    selfint= rho .dot. ( Coulomb3D(rho%grid, rho%bubbles) .apply. rho )
    selfint_exact=rhogen%selfint(rho%grid)

    call pinfo(HEADER)
    part_ok=compare_and_report(&
        "selfint-numerical",    selfint,        1.7252361611235862d0) < TOL &
        .and. part_ok
    part_ok=compare_and_report(&
        "selfint-analytical",   selfint_exact,  1.7252391238773686d0) < TOL &
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
    call pinfo("TEST 4: Testing generation on explicit grid with PBCs...")
    grid=Grid3D(ranges=reshape([-4.d0, 4.d0,-8.d0, 8.d0,-8.d0, 8.d0], [2,3]),&
                nlip  =7, stepmax=.1d0, pbc_string="x" )
    rho=rhogen%gen(grid)

    selfint= rho .dot. ( Coulomb3D(grid, rho%bubbles, grid) .apply. rho )
    selfint_exact=rhogen%selfint(grid)

    call pinfo(HEADER)
    part_ok=compare_and_report(&
        "selfint-numerical",    selfint,        1.6898200211055878d0) < TOL &
        .and. part_ok
    part_ok=compare_and_report(&
        "selfint-analytical",   selfint_exact,  1.6855183868782713d0) < TOL &
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
    call pinfo("TEST 5: Testing GaussPotGenerator initialization from GaussGenerator...")
    potgen=GaussPotGenerator(rhogen)
    ! In GaussPotgenerator, only the gen_cubef function is overwritten when compared to
    ! GaussGenerator
    ! Apot should contain electric potential of the gaussian after gen()
    apot=potgen%gen()
   
    ! charge density dot electric potential is self interaction energy? 
    selfint_apot=rho .dot. apot

    ! and finally get the analytical self interaction energy like before
    selfint_exact=potgen%selfint(apot%grid)

    ! and compare the results
    call pinfo(HEADER)
    part_ok=compare_and_report(&
        "selfint-num-apot",     selfint_apot,   1.7252751015070420d0) < TOL &
        .and. part_ok
    part_ok=compare_and_report(&
        "selfint-analytical",   selfint_exact,  1.7252391238773686d0) < TOL &
        .and. part_ok

    if(part_ok) then
        call pinfo("PASSED!")
    else
        call perror("FAILED!")
    end if
    print*
    all_ok=all_ok .and. part_ok
! %%%%%%%%%%%%%%%%%%%%%%    TEST  6    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ! Same as the previous, only with a different init-function
    part_ok=.TRUE.
    call pinfo("TEST 6: Testing GaussPotGenerator explicit initialization...")
    
    potgen=GaussPotGenerator(alpha  = [2.d0,  4.d0],&
                             center = reshape([-1.d0, 0.d0, 0.d0,&
                                                1.d0, 0.d0, 0.d0],[3,2]),&
                             q      = [1.d0, -1.d0], step=.1d0)
    apot=potgen%gen()
    selfint_apot=rho .dot. apot
    selfint_exact=potgen%selfint(apot%grid)

    call pinfo(HEADER)
    part_ok=compare_and_report(&
        "selfint-num-apot",     selfint_apot,   1.7252763154741830d0) < TOL &
        .and. part_ok
    part_ok=compare_and_report(&
        "selfint-analytical",   selfint_exact,  1.7252391238773686d0) < TOL &
        .and. part_ok

    if(part_ok) then
        call pinfo("PASSED!")
    else
        call perror("FAILED!")
    end if
    print*
    all_ok=all_ok .and. part_ok
! %%%%%%%%%%%%%%%%%%%%%%    TEST  7    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ! Same as the previous, only with a different init-function
    part_ok=.TRUE.
    
    call pinfo("TEST 7: Testing GaussPotGenerator initialization by parsing...")
    potgen=GaussPotGenerator([" 1.0   2.0  -1.0 0.0 0.0",&
                              "-1.0   4.0   1.0 0.0 0.0"], step=.1d0)
    apot=potgen%gen()
    selfint_apot=rho .dot. apot
    selfint_exact=potgen%selfint(apot%grid)

    call pinfo(HEADER)
    part_ok=compare_and_report(&
        "selfint-num-apot",     selfint_apot,   1.7252763154741830d0) < TOL &
        .and. part_ok
    part_ok=compare_and_report(&
        "selfint-analytical",   selfint_exact,  1.7252391238773686d0) < TOL &
        .and. part_ok

    if(part_ok) then
        call pinfo("PASSED!")
    else
        call perror("FAILED!")
    end if
    print*
    all_ok=all_ok .and. part_ok


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
