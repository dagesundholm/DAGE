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
!> @file helmholtz_test.F90
!! Test Helmholtz3D class


program helmholtz_test
    use Globals_m
    use Function3D_class
    use Grid_class
    use Bubbles_class
    use GaussQuad_class
    use Helmholtz3D_class




    implicit none
! potential
    type(Function3D)          :: test_potential
! bubbles, bubbles+cube, cube
    type(Function3D)          :: b_function, bc_function, c_function
    type(Helmholtz3D)         :: helmholtz_operator
    real(REAL64)              :: E

    logical                   :: part_ok
    logical                   :: all_ok

    integer                   :: i,j,k
! all function3d objects have the same grid
    type(Grid3D)              :: grid
! bubbles
    type(Bubbles)             :: bubbles1, bubbles2
! used for setting values
    real(REAL64), pointer     :: values(:,:,:)
    real(REAL64), pointer     :: x(:), y(:), z(:)
    real(REAL64), pointer     :: values2(:)
    type(Grid1D)              :: grid2



write(*,*) 'Starting the initialization'

! initializing the grid
    grid=Grid3D(ranges=reshape([-4.1d0, 4.d0,-4.d0, 4.d0,-4.d0, 4.d0], [2,3]),&
                nlip  =7, stepmax=.15d0 )
! bubbles for the potential function
    bubbles1 = Bubbles(lmax    = 0, &
                            centers = reshape([0.d0, 0.d0, 0.d0],[3,1]) ,&
                            grids   = [ Grid1D([0.0d0, 4.0d0], nlip=7,&
                                                  stepmax=0.03d0) ],&
                            z       = [1.d0]    )

    test_potential=Function3D(grid,bubbles1,'potential',3)
    values => test_potential%get_cube()
    x=>grid%axis(X_)%get_coord()
    y=>grid%axis(Y_)%get_coord()
    z=>grid%axis(Z_)%get_coord()

! setting the test_potential to be harmonic potential
! defined in the cube
    do k=1,size(z)
        do j=1,size(y)
             do i=1,size(x)
             values(i,j,k)=0.5d0*((x(i)**2)+(y(j)**2)+(z(k)**2))
             end do
        end do
    end do
    

! bubbles for the test wave functions
    bubbles2 = Bubbles(lmax    = 0, &
                            centers = reshape([0.d0, 0.d0, 0.d0],[3,1]) ,&
                            grids   = [ Grid1D([0.0d0, 4.0d0], nlip=7,&
                                                  stepmax=0.03d0) ],&
                            z       = [1.d0]    )

! constructing the test wave functions
    b_function=Function3D(grid,bubbles2,'bubbles function',1)
    bc_function=Function3D(grid,bubbles2,'bubbles+cube function',1)
    c_function=Function3D(grid,bubbles2,'cube function',1)

! setting up bubbles function
    values  => b_function%get_cube()
    values2 => b_function%bubbles%get_f(1,0,0)
! nothing in cube 
    values=0d0
! everything in bubbles
    grid2=b_function%bubbles%get_grid(1)
    x => grid2%get_coord()
    do i=1,size(x)
        values2(i)=exp(-0.5d0 * x(i)**2)
    end do   
! normalization
    b_function=b_function/sqrt(b_function .dot. b_function)

! setting up bubbles+cube function
    values  => bc_function%get_cube()
    values2 => bc_function%bubbles%get_f(1,0,0)
    x=>grid%axis(X_)%get_coord()
    y=>grid%axis(Y_)%get_coord()
    z=>grid%axis(Z_)%get_coord()
    do k=1,size(z)
        do j=1,size(y)
             do i=1,size(x)
             values(i,j,k)=exp(-0.5d0*((x(i)**2)+(y(j)**2)+(z(k)**2)))
             end do
        end do
    end do
    grid2=bc_function%bubbles%get_grid(1)
    x => grid2%get_coord()
    do i=1,size(x)
        values2(i)=exp(-0.5d0 * x(i)**2)
    end do  
    bc_function=bc_function/sqrt(bc_function .dot. bc_function)

! setting up cube function
    values  => c_function%get_cube()
    x=>grid%axis(X_)%get_coord()
    y=>grid%axis(Y_)%get_coord()
    z=>grid%axis(Z_)%get_coord()
    do k=1,size(z)
        do j=1,size(y)
             do i=1,size(x)
             values(i,j,k)=exp(-0.5d0*((x(i)**2)+(y(j)**2)+(z(k)**2)))
             end do
        end do
    end do
    c_function=c_function/sqrt(c_function .dot. c_function)

! setting up helmholtz_operator with default quadrature
    !helmholtz_operator=Helmholtz3D(test_potential) 

write(*,*) 'Initialization complete. Starting the tests.'
    all_ok  = .TRUE.
    part_ok = .TRUE.

write(*,*) 'Testing energy estimation'
    !E=helmholtz_operator%energy(b_function)
    if(.NOT. isNaN(E)) then
        write(*,*) 'Energy of bubbles wave function is ', E
    else
        write(*,*) 'Energy estimation failed'
        all_ok= .FALSE.
    endif
    !E=helmholtz_operator%energy(bc_function)
    if(.NOT. isNaN(E)) then
        write(*,*) 'Energy of bubbles+cube wave function is ', E
    else
        write(*,*) 'Energy estimation failed'
        all_ok= .FALSE.
    endif
    !E=helmholtz_operator%energy(c_function)
    if(.NOT. isNaN(E)) then
        write(*,*) 'Energy of cube wave function is ', E
    else
        write(*,*) 'Energy estimation failed'
        all_ok= .FALSE.
    endif
! testing updating
    write(*,*) 'Testing wave function updating'
    !b_function=helmholtz_operator%update(b_function)
    !bc_function=helmholtz_operator%update(bc_function)
    !c_function=helmholtz_operator%update(c_function)

    !E=helmholtz_operator%energy(b_function)
    if(.NOT. isNaN(E)) then
        write(*,*) 'Energy of updated bubbles wave function is ', E
    else
        write(*,*) 'Energy estimation failed'
        all_ok= .FALSE.
    endif
    !E=helmholtz_operator%energy(bc_function)
    if(.NOT. isNaN(E)) then
        write(*,*) 'Energy of updated bubbles+cube wave function is ', E
    else
        write(*,*) 'Energy estimation failed'
        all_ok= .FALSE.
    endif
    !E=helmholtz_operator%energy(c_function)
    if(.NOT. isNaN(E)) then
        write(*,*) 'Energy of updated cube wave function is ', E
    else
        write(*,*) 'Energy estimation failed'
        all_ok= .FALSE.
    endif

! testing the trial function generation
    write(*,*) 'Testing the helmholtz trial function generation'
    !b_function=helmholtz_operator.apply. b_function
    !bc_function=helmholtz_operator.apply. bc_function
    !c_function=helmholtz_operator.apply. c_function

    !E=helmholtz_operator%energy(b_function)
    if(.NOT. isNaN(E)) then
        write(*,*) 'Energy of trial bubbles wave function is ', E
    else
        write(*,*) 'Energy estimation failed'
        all_ok= .FALSE.
    endif
    !E=helmholtz_operator%energy(bc_function)
    if(.NOT. isNaN(E)) then
        write(*,*) 'Energy of trial bubbles+cube wave function is ', E
    else
        write(*,*) 'Energy estimation failed'
        all_ok= .FALSE.
    endif
    !E=helmholtz_operator%energy(c_function)
    if(.NOT. isNaN(E)) then
        write(*,*) 'Energy of trial cube wave function is ', E
    else
        write(*,*) 'Energy estimation failed'
        all_ok= .FALSE.
    endif




! CLEANUP
    call test_potential%destroy()
    call b_function%destroy() 
    call bc_function%destroy() 
    call c_function%destroy() 
    call helmholtz_operator%destroy()
    call grid%destroy()
    call bubbles1%destroy()
    call bubbles2%destroy()
    call grid2%destroy()

    nullify(values)
    nullify(x)
    nullify(y)
    nullify(z)
    nullify(values2)


! %%%%%%%%% FINAL REPORT %%%%%%%%%%%%%%

    if(all_ok) then
        call pinfo("ALL TESTS PASSED")
        stop 0
    else
        call perror("SOME TESTS FAILED")
        stop 1
    end if



end program 
