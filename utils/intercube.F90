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
module intercube_class
    use globals_m
    use grid_class
    use interpol_class
    implicit none

    type intercube_t
        private
        type(interpol_t) :: xlip, ylip, zlip
        type(grid_t), pointer :: grid
        integer(INT32) :: nlip
    end type

    public new_intercube, del_intercube, intercube_t, interpol_cube
    private

contains
    subroutine new_intercube(self, grid)
        type(intercube_t) :: self
        type(grid_t), target :: grid

        real(REAL64), dimension(2) :: rng 
        type(grid1d_t), pointer :: xgrid, ygrid, zgrid

        self%grid=>grid
        call get_grid_coord(grid, X_, xgrid)
        call get_grid_coord(grid, Y_, ygrid)
        call get_grid_coord(grid, Z_, zgrid)
        call new_interpol(self%xlip, xgrid)
        call new_interpol(self%ylip, ygrid)
        call new_interpol(self%zlip, zgrid)
        self%nlip=get_grid_nlip(grid)
    end subroutine

    subroutine del_intercube(self)
        type(intercube_t) :: self
        
        call del_interpol(self%xlip)
        call del_interpol(self%ylip)
        call del_interpol(self%zlip)
    end subroutine

    function interpol_cube(self, f, r) result(y)
        type(intercube_t) :: self
        real(REAL64), dimension(:,:,:), intent(in) :: f
        real(REAL64), dimension(3), intent(in) :: r
        real(REAL64) :: y

        real(REAL64), dimension(self%nlip,self%nlip) :: g
        real(REAL64), dimension(self%nlip) :: h
        integer(INT32) :: i, j, xidx, yidx
        integer(INT32) :: zidx1, zidx2
        integer(INT32) :: ixcell, iycell, izcell

        ixcell=get_cellidx(self%xlip, r(1), xidx)
        iycell=get_cellidx(self%ylip, r(2), yidx)
        izcell=get_cellidx(self%zlip, r(3), zidx1, zidx2)
        xidx=xidx-1
        yidx=yidx-1

        do j=1,self%nlip
            do i=1,self%nlip
                g(i,j)=interpol(self%zlip,izcell, &
                    f(xidx+i,yidx+j,zidx1:zidx2),r(3))
            end do
        end do

        do i=1,self%nlip
            h(i)=interpol(self%ylip, iycell, g(i,:), r(2))
        end do
        
        y=interpol(self%xlip, ixcell, h, r(1))
    end function

    subroutine getcube(self, r,f,cube)
        type(intercube_t) :: self
        real(REAL64), dimension(3), intent(in) :: r
        real(REAL64), dimension(:,:,:), intent(in) :: f
        real(REAL64), dimension(:,:,:), intent(out) :: cube

        integer(INT32) :: i
        integer(INT32) :: ix1, ix2
        integer(INT32) :: iy1, iy2
        integer(INT32) :: iz1, iz2

        i=get_cellidx(self%xlip, r(1), ix1,ix2)
        i=get_cellidx(self%ylip, r(2), iy1,iy2)
        i=get_cellidx(self%zlip, r(3), iz1,iz2)
        cube=f(ix1:ix2,iy1:iy2,iz1:iz2)
!        print "(3('('i3':'i3')'), 3f8.4)", ix1,ix2,iy1,iy2,iz1,iz2, r
    end subroutine

end module
