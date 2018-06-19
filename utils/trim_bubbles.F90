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
!> @file trim_bubbles.F90

!> Utility to trim the radial grids of the bubbles of a function read from a
!! binary .fun file, and dumps it to a new binary.fun  file.
program trim_bubbles
    use Globals_m
    use Function3D_class
    use Bubbles_class
    use Grid_class
    implicit none

    type(Function3D) :: funcin
    type(Function3D) :: funcout

    character(len=200) :: input_file, output_file

    real(REAL64), pointer  :: cubein(:,:,:)
    real(REAL64), pointer  :: cubeout(:,:,:)

    type(Grid1D), pointer  :: bubgr(:)

    integer                :: ibub, nbub

    integer, parameter     :: NLIP=7, N0=200
    real(REAL64), parameter:: RMAX=20.d0

    call getarg(1, input_file)
    call getarg(2, output_file)

    call funcin%load(input_file)
    cubein=>funcin%get_cube()
    nbub=funcin%bubbles%get_nbub()

    allocate(bubgr(nbub))
    bubgr=[ (Grid1D(funcin%bubbles%get_z(ibub), N0, NLIP, RMAX), ibub=1,nbub) ]

    funcout=funcin
    funcout%bubbles=funcout%bubbles%project_onto( bubgr )

    call funcin%bubbles%print("orig")
    call funcout%bubbles%print("trimmed")

    call funcout%dump(output_file)

    call funcin%destroy()
    call funcout%destroy()

    do ibub=1,nbub
        call bubgr(ibub)%destroy()
    end do
    deallocate(bubgr)

end program

