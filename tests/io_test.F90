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
program io_test
    use Globals_m
    use Function3D_class
    use Grid_class
    use SlaterGenerator_class
    use Bubbles_class

    real(REAL64), parameter :: DEFAULT_STEP = 0.3d0

    type(Function3D) :: f, g, q
    type(SlaterGenerator) :: s
    type(Grid3D)          :: grid
    type(Grid1D),allocatable :: bubgrid(:)
    real(REAL64),allocatable :: cellh(:)
    real(REAL64), dimension(5) :: stepx, stepy, stepz
    real(REAL64) :: stepmax(3)
    integer :: i

    ! File formats

    ! Create dummy function
    s = SlaterGenerator(z       = [ 1.d0 ], &
                        centers = reshape([0.d0, 0.d0, 0.d0],[3,1]) ,&
                        ibub    = [ 1 ],&
                        l       = [ 0 ],&
                        m       = [ 0 ],&
                        expos   = [ 1.d0 ],&
                        coeffs  = [ 1.d0 ] )

    ! Dummy non-equidistant grid
    stepx = [0.01, 0.2, 0.2, 0.1, 0.1]
    stepy = [0.01, 0.2, 0.2, 0.1, 0.1]
    stepz = [0.01, 0.2, 0.2, 0.1, 0.1]

    f=s%gen( grid = Grid3D( [-5.0d0, -5.0d0, -5.0d0], [5,5,5], 7,&
                            stepx, stepy, stepz ) ,&
             bubs = Bubbles(lmax    = 0, &
                            centers = reshape([0.d0, 0.d0, 0.d0],[3,1]) ,&
                            grids   = [ Grid1D([0.0d0, 20.0d0], nlip=7,&
                                                  stepmax=0.05d0) ],&
                            z       = [1.d0]    ) )

    !********************
    !   Testing writing
    !********************

    !*******************************
    !   Default resolution, no bubbles
    !*******************************
    call f%write('dummy.plt')
    g = Function3D('dummy.plt')

    ! Test: No bubbles were injected
    call assert(all(g%get_cube() == 0.0d0))

    ! Default resolution is set in Function3D_class (currently 0.3 a.u.)
    do i=1,3
        cellh = g%grid%axis(i)%get_cell_steps()
        write(*,"("//xchar(size(cellh))//"F10.5)") cellh
        call assert(all(cellh == cellh(1)))
        call assert(all(cellh <= DEFAULT_STEP))
        stepmax(i) = cellh(1)
    enddo

    !*******************************
    !   Custom resolution, no bubbles
    !*******************************

    call g%write("dummy2.plt", step=stepmax(1))
    q = Function3D("dummy2.plt")
    call assert(all(q%get_cube() == 0.0d0))

    do i=1,3
        cellh = q%grid%axis(i)%get_cell_steps()
        write(*,"("//xchar(size(cellh))//"F10.5)") cellh
        call assert(all(cellh == cellh(1)))
        call assert(all(cellh == stepmax(1)))
    enddo

    !***********************************
    !   Custom resolution, write bubbles
    !************************************

    call f%write("dummy3.cub", step=0.05d0,&
                 write_bubbles=.true.)

    g = Function3D("dummy3.cub")
    call assert(.not. all(g%get_cube() == 0.d0))

    write (*,*) "The test has succeeded!"

end program
