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
program interactor

    ! Reads a previously generated potential, and a set of vectors (1, 2 or 3)
    ! (same as $pointval in turbomole) defining a grid.
    ! It gives back the potential values at those grid points in standard input
    ! x y z f(x,y,z)

      use globals_m
      use gopenmol_m
      use lip_class
      use density_m
      use potential_class
      use bubbles_class
      use grid_class
      implicit none

      type(grid_t) :: grid
      type(lip_t) :: lipton
      type(density_t) :: dens1,dens2
      type(potential_t) :: pot1!,pot2

      character(len=100) :: file1,file2

      integer(INT32) :: i,j,k,l,gdims(3),nlip
      real(REAL64),pointer :: box1(:,:,:),box2(:,:,:)
!      real(REAL64) :: E12,E21,S12,S21
      real(REAL64) :: E12,S12


      ! Opening input file and initializing stuff
    open(INPUT_FD,file='GENPOT')
    call new_getkw(input,INPUT_FD)
    if(iargc()>0) then
        call getarg(1,file1)
        call getarg(2,file2)
        file1=trim(file1)
        file2=trim(file2)
    else
        stop 'You have to specify the input densities!'
    end if

    call getkw(input, 'lip_points', nlip)
    call new_lip(lipton, nlip)

    call read_gopenmol_plt_grid(file1, grid)
    call set_grid_slice(grid,SL_COORD,1,1)
    call slice_grid(grid,SL_COORD)

    call new_density(dens1,grid,file1)
    call new_density(dens2,grid,file2)
    
    call new_potential(pot1,grid)
!    call new_potential(pot2,grid)

    call genpot(pot1,dens1,1)
!    call genpot(pot2,dens2,1)

    E12=selfint(pot1,dens2)
!    E21=selfint(pot2,dens1)

    call get_potential(pot1,box1)
    call get_density(dens1,box2)
    box1=box2
    S12=selfint(pot1,dens2)

!    call get_potential(pot2,box1)
!    call get_density(dens2,box2)
!    box1=box2
!    S21=selfint(pot2,dens1)

    print*,'The Coulomb integral:  ',E12
    print*,'The overlap integral:  ',S12
end program

