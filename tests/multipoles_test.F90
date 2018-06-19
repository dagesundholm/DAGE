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
program multipoles_test
    use Globals_m
    use Function3D_class
    use GaussGenerator_class
    use CartIter_class
    implicit none

    type(GaussGenerator)      :: rhogen
    type(Function3D)          :: rho
    real(REAL64), allocatable :: mp(:)
    type(CartIter)            :: iter
    integer                   :: kappa(3)
    integer                   :: k
    logical                   :: continue_iteration

    rhogen=GaussGenerator(alpha  = [1.d0],&
                          center = reshape([0.d0, 0.d0, 0.d0],[3,1]),&
                          q      = [1.d0])

    rho=rhogen%gen()
    mp=rho%cube_multipoles(lmax=10, ref= [0.d0,0.d0,0.d0])

    iter=CartIter(3,10)
    k=0
    call iter%next(kappa, continue_iteration)
    do while(continue_iteration)
        k=k+1
        print*, "k, Kappa:", k, kappa
        print*, "Multipole-moment:", mp(k)
        call iter%next(kappa, continue_iteration)
    end do

end program
