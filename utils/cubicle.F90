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
program cubicle
    use globals_m
    use gopenmol_m
    use grid_class
    use cubedata_m
    use interpol_class
    use intercube_class
    implicit none

    integer(INT32) :: ierr, i, j, k, n
    character(BUFLEN) :: file1, file2, fname, outfile, bname, fext
    type(grid_t) :: grid1, grid2, outgrid
    real(REAL64), dimension(:,:,:), pointer :: data1, data2
    real(REAL64), dimension(:,:,:), allocatable :: errdata
    real(REAL64), dimension(:,:), allocatable :: odata
    type(intercube_t) :: lip1, lip2

    real(REAL64), dimension(3) :: r
    real(REAL64) :: xstep, ystep, zstep, q1, q2, re, sgn
    real(REAL64), dimension(2) :: xrange, yrange, zrange
    integer(INT32), dimension(3) :: gdims, d1dim
    logical :: diff_p=.false., plt_p=.false., interpol_p=.true.
    logical :: plot_p=.false., convert_p=.false., ascii_p=.false.

    call new_getkw(input)

    call getkw(input, 'mode', fext)
    select case (fext)
        case ('diff')
            diff_p=.true.
        case ('conv')
            convert_p=.true.
        case('plot')
            plot_p=.true.
    end select

    call getkw(input, 'in1', file1)
    if (diff_p) call getkw(input, 'in2', file2)

    ierr=0
!    call getkw('plt_format', ierr)
!    if (ierr /= 0) then
!        plt_p=.true.
!    end if

    call basename(file1,bname)
    fext=filext(file1)
    print *, trim(file1)
    print *, 'base:', trim(bname)
    print *, 'ext:', trim(fext)

    select case (fext)
        case('grd', 'plt')
            if (fext == 'plt') plt_p=.true.
            call readcube(file1, grid1, data1, plt_p)
            plt_p=.false.
            if (diff_p) then
                fext=filext(file2)
                if (fext == 'plt') plt_p=.true.
                call readcube(file2, grid2, data2, plt_p)
            end if
        ! obsolete
        case('gdat')
            open(DDEN_FD, file=trim(bname)//'.gdef', status='old')
            call read_grid(grid1, DDEN_FD)
            close(DDEN_FD)
            d1dim(1)=get_grid_dim(grid1, X_)
            d1dim(2)=get_grid_dim(grid1, Y_)
            d1dim(3)=get_grid_dim(grid1, Z_)
            allocate(data1(d1dim(1), d1dim(2), d1dim(3)))
            open(DDEN_FD, file=trim(file1), access='direct', status='old', &
                recl=product(d1dim(1:2)*2))
            do i=1,d1dim(3)
                read(DDEN_FD, rec=i) data1(:,:,i)
            end do
            close(DDEN_FD)
        case default
            print *, '<<< Unknown file format >>>'
            stop
    end select

    d1dim(1)=get_grid_dim(grid1, X_)
    d1dim(2)=get_grid_dim(grid1, Y_)
    d1dim(3)=get_grid_dim(grid1, Z_)
    xrange=get_grid_range(grid1, X_)
    yrange=get_grid_range(grid1, Y_)
    zrange=get_grid_range(grid1, Z_)
    if (keyword_is_set(input, 'outdim')) then
        call getkw(input, 'outdim', gdims)
    else
        gdims=d1dim
    end if

    call new_grid(outgrid,gdims,xrange,yrange,zrange,7)
    allocate(odata(gdims(1), gdims(2)))
    if (diff_p) then
        allocate(errdata(gdims(1), gdims(2), gdims(3)))
        errdata=0.d0
    end if
    call new_intercube(lip1, grid1)
    if (diff_p) call new_intercube(lip2, grid2)

    xstep=sum(abs(xrange))/(gdims(1)-1)
    ystep=sum(abs(yrange))/(gdims(2)-1)
    zstep=sum(abs(zrange))/(gdims(3)-1)
    if (gdims(1)==d1dim(1) .and. gdims(2)==d1dim(2) .and. & 
        gdims(3)==d1dim(3) ) interpol_p=.false.

    if (diff_p) then
        do k=1,gdims(3)
            r(3)=zrange(1)+dble(k-1)*zstep
            do j=1,gdims(2)
                r(2)=yrange(1)+dble(j-1)*ystep
                do i=1,gdims(1)
                    r(1)=xrange(1)+dble(i-1)*xstep
                    if (interpol_p) then
                        q1=abs(interpol_cube(lip1, data1, r))
                        q2=abs(interpol_cube(lip2, data2, r))
                    else
                        q1=data1(i,j,k)
                        q2=data2(i,j,k)
                    end if
                    odata(i,j)=abs(q1-q2)
                    errdata(i,j,k)=abs(q1-q2)/abs(q1)
                end do
            end do
! write cube
        end do
    else !bongo code, needs fix
        if (keyword_is_set(input, 'outfile')) then
            call getkw(input, 'outfile', outfile)
            call getkw(input, 'ascii', ascii_p)
            print *, 'Writing ascii file', gdims
            open(35, file=trim(outfile))
            do k=1,gdims(3)
                r(3)=zrange(1)+dble(k-1)*zstep
                do j=1,gdims(2)
                    r(2)=yrange(1)+dble(j-1)*ystep
                    do i=1,gdims(1)
                        r(1)=xrange(1)+dble(i-1)*xstep
                        q1=interpol_cube(lip1, data1, r)
                        odata(i,j)=q1
                    end do
                end do
                write(35, '(3d10.3)') odata
            end do
            close(35)
        end if
    end if

!    if (keyword_is_set(input, 'outfile')) then
!        call getkw(input, 'outfile', outfile)
!        call getkw(input, 'ascii', ascii_p)
!        if (ascii_p) then
!            print *, 'Writing ascii file', gdims
!            open(35, file=trim(outfile))
!            do k=1,gdims(3)
!                write(35, '(3d10.3)') odata(:,:,k)
!            end do
!            close(35)
!        else
!            call writecube(file1, outgrid, odata)
!        end if

!        open(35, file=trim(bname)//'.gdef')
!        call write_grid(outgrid, 35)
!        close(35)
!        open(35, file=trim(bname)//'.gdat',access='direct',&
!            recl=gdims(1)*gdims(2)*2)
!        do k=1,gdims(3)
!            write(35, rec=k) odata(:,:,k)
!        end do
!        close(35)
!    end if

    if (keyword_is_set(input, 'plot')) then
        call getkw(input, 'plot', file1)
!        call write_gopenmol(file1, outgrid, odata)
        if (diff_p) call  write_gopenmol('relerr.plt', outgrid, errdata)
    end if

    
    print *, 'Min value: ', minval(data1)
    print *, 'Max value: ', maxval(data1)
    print *
    print *, 'Min value: ', minval(odata)
    print *, 'Max value: ', maxval(odata)
    if (diff_p) then
        print *, 'Min relative error:', minval(errdata)
        print *, 'Max relative error:', maxval(errdata)
    end if
    call del_grid(outgrid)
    call del_intercube(lip1)
    if (diff_p) call del_intercube(lip2)
    deallocate(data1)
    if (diff_p) deallocate(data2)
    if (diff_p) deallocate(errdata)
    deallocate(odata)
    call del_getkw(input)

contains

    function cmpvec(a,b) result (r)
        real(REAL64), dimension(:), intent(in) :: a, b
        logical :: r

        integer(INT32) :: i,n

        n=size(a)
        if (n /= size(b)) then
            r=.false.
            return
        end if
        
        do i=1,n
            if ( a(i) /= b(i) ) then
                r=.false.
                return
            end if
        end do
        r=.true.
    end function

    subroutine testlip(grid3d)
        type(grid_t) :: grid3d

        type(grid1d_t), pointer :: grid
        type(interpol_t) :: lip
        real(8), dimension(:), allocatable :: gdata
        integer(INT32) :: i, n
        real(8) :: x, step
        real(8), dimension(2) :: rng

        call get_grid_coord(grid3d, X_, grid)
        print *, 'new lip'
        call new_interpol(lip, grid)
        n=get_grid_dim(grid)
        allocate(gdata(n))
        rng=get_grid_range(grid)
        step=sum(abs(rng))/(n-1)
        
        do i=1,n
            x=rng(1)+dble(i-1)*step
            gdata(i)=exp(-1.d0*x**2)
            write(98, *) x, gdata(i)
        end do

        do i=1,n
            x=rng(1)+dble(i-1)*step
            write(99, *) x, interpol(lip, gdata, x)
        end do

        do i=2,n-1
            x=rng(1)+0.1+dble(i-1)*step
            write(100, *) x, interpol(lip, gdata, x)
        end do

        call del_interpol(lip)
        deallocate(gdata)
    end subroutine

end program
