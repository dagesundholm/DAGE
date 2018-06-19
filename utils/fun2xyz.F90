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
!> @file fun2xyz.F90
!> Extract bubble centers from fun file and create xyz file

!> Read a .fun file as command line argument, output an xyz file containing
!> the bubble centers.
!> The atomic number is printed instead of the element symbol...
program fun2xyz
    use globals_m
    use Function3D_class
    implicit none

    type(Function3D) :: func
    character(80) :: fname

    integer :: ibub, nbub
    real(REAL64), pointer :: z(:), crd(:,:)

    call get_command_argument(1,fname)

    call func%load(fname)

    nbub=func%bubbles%get_nbub()
    z=>func%bubbles%get_z()
    crd=>func%bubbles%get_centers()

    print*,nbub
    print*,func%get_label()
    do ibub=1,nbub
        print*,z(ibub),crd(:,ibub)
    end do
end program
