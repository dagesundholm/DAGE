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
!> @file gopenmol.F90
!! Implements IO routines for the Gopenmol format

!> IO Backend: gopenmol files.
!!
module gopenmol_m
    use globals_m
    use file_format_class
    implicit none

    public :: read_gopenmol
    public :: write_gopenmol
    private
contains

    !> Reads a rank-three array from a gOpenMol file.
    !!
    !! @returns `.true.` if reading was successful
    function read_gopenmol(filename, gopen) result(ok)
        character(len=*), intent(in)    :: filename
        class(FileFormat)               :: gopen
        logical                         :: ok

        call read_grid_gopenmol(filename, READ_FD,&
                                gopen%gdims,&
                                gopen%ranges,&
                                ok)

        if (ok) then

            allocate(gopen%cube(gopen%gdims(1),&
                                gopen%gdims(2),&
                                gopen%gdims(3)))
            call read_cube_gopenmol(filename, READ_FD,&
                                    gopen%cube,&
                                    ok)
        endif

    end function

    !> Writes a rank-three array to a file
    !!
    !! Existing files are overwritten.
    subroutine write_gopenmol(filename, gopen)
        character(len=*), intent(in) :: filename
        class(FileFormat)            :: gopen

        integer :: pos, p1, p2, p3, i, j, k

        integer(INT16), dimension(3) :: sgdims
        real(REAL32), dimension(2,3) :: sranges
        integer(INT16), parameter :: SURFACE=200, RANK_PLT=3

        sgdims = gopen%gdims
        sranges = AU2A*(gopen%ranges)
        
        open(WRITE_FD, file=trim(filename), form='unformatted', access='direct', recl=RECLEN)

        ! Write header
        write(WRITE_FD, rec=1)  RANK_PLT
        write(WRITE_FD, rec=2)  SURFACE
        write(WRITE_FD, rec=3)  sgdims(3)
        write(WRITE_FD, rec=4)  sgdims(2)
        write(WRITE_FD, rec=5)  sgdims(1)
        write(WRITE_FD, rec=6)  sranges(1,3)
        write(WRITE_FD, rec=7)  sranges(2,3)
        write(WRITE_FD, rec=8)  sranges(1,2)
        write(WRITE_FD, rec=9)  sranges(2,2)
        write(WRITE_FD, rec=10) sranges(1,1)
        write(WRITE_FD, rec=11) sranges(2,1)

        p1=size(gopen%cube, 1)
        p2=size(gopen%cube, 2)
        p3=size(gopen%cube, 3)

        pos=12!+slice_start
        do k=1, p3
            do j=1, p2
                do i=1, p1
                    write(WRITE_FD, rec=pos) real(gopen%cube(i,j,k))
                    pos = pos + 1
                end do
                ! pos=pos+p1_tot-p1
            end do
        end do

        close(WRITE_FD)

    end subroutine

    !> Low-level subroutine to read a grid from a gOpenMol file.
    subroutine read_grid_gopenmol(fname, funit, gdims, dranges, io_ok)
        character(*), intent(in)        :: fname
        integer(INT32), intent(in)      :: funit
        integer(INT32), dimension(3)    :: gdims
        real(REAL64), dimension(2,3)    :: dranges
        logical                         :: io_ok
        
        integer(INT16), dimension(3)    :: sgdims
        real(REAL32), dimension(2,3)    :: sranges

        open(funit,file=trim(fname), form='unformatted',access='direct',recl=RECLEN)
        read(funit,rec=3, err=42) sgdims(3)
        read(funit,rec=4, err=42) sgdims(2)
        read(funit,rec=5, err=42) sgdims(1)
        read(funit,rec=6, err=42) sranges(1,3)
        read(funit,rec=7, err=42) sranges(2,3)
        read(funit,rec=8, err=42) sranges(1,2)
        read(funit,rec=9, err=42) sranges(2,2)
        read(funit,rec=10,err=42) sranges(1,1)
        read(funit,rec=11,err=42) sranges(2,1)

        dranges=sranges*A2AU
        gdims=sgdims
        io_ok = .true.

        close(funit)
        return

42      call perror('Error while reading file '//trim(fname))
        io_ok = .false.
    end subroutine

    !> Low level subroutine to read array data from a gOpenMol file.
    subroutine read_cube_gopenmol(fname, funit, cube, io_ok)
        ! This should also take into account the splitting business

        character(*), intent(in)        :: fname
        integer(INT32)                  :: funit
        real(REAL64), pointer           :: cube(:,:,:)
        logical                         :: io_ok
        
        integer(INT32) :: p1, p2, p3
        integer(INT32) :: i, j, k
        real(REAL32) :: value

        integer(INT32) :: slice_start

        integer :: pos

        p1=size(cube,1)
        p2=size(cube,2)
        p3=size(cube,3)

        open(funit,file=trim(fname), form='unformatted',access='direct',recl=RECLEN)

        ! First value to read
        pos=12!+slice_start
        do k=1,p3
            do j=1,p2
                do i=1,p1
                    read(funit,rec=pos,err=42) value
                    cube(i,j,k)=value
                    pos=pos+1
                end do
                ! pos=pos+p1_tot-p1
            end do
        end do

        io_ok = .true.
        close(funit)
        return

42      call perror('Error while reading file '//trim(fname))
        io_ok = .false.
    end subroutine

end module
