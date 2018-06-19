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
!> @file datadump.F90
!! @todo Short description

!> @todo datadump_m documentation
module datadump_m
    use globals_m
    use ISO_FORTRAN_ENV
    use pprinter
    use mpi_m
    use getkw_class

    public filename_dot_n, genfile, datadump
    private

    interface datadump
        module procedure dump_d_v
        module procedure dump_d_m2
        module procedure dump_d_m3
    end interface

    integer(INT32), parameter :: DUMP_FD=19
contains

    subroutine filename_dot_n(str,n)
        character(*), intent(inout) :: str
        integer(INT32), intent(in) :: n

        write(str, "(a,'.',i3.3)") trim(str), n
    end subroutine

    subroutine genfile(str, fname)
        character(*), intent(in) :: str
        character(*), intent(out) :: fname

        integer :: rank
        
        ! If str is a keyword present in input, fname is set to the 
        ! keyword value. Otherwise, fname is set to str. If this is
        ! an MPI run, '.n' (n=rank) is appended to the name. 
        if (has_keyword(input,str)) then
            call getkw(input, str, fname)
        else
            fname=str
        end if

        rank=0
        if (mpirun_p) then
            rank=get_mpi_rank()
!            if (rank > 0) call filename_dot_n(fname,rank)
            call filename_dot_n(fname,rank)
        end if
    end subroutine

    subroutine dump_d_v(str, dd)
        character(*), intent(in) :: str
        real(8), dimension(:), intent(in) :: dd

        character(LINESZ) :: fname
        
        call genfile(str, fname)
        open(DUMP_FD, file=trim(fname), status='unknown', err=42)
        write(DUMP_FD, *) dd
        close(DUMP_FD)
        return

42      call perror('Could not open file ' // trim(fname))
    end subroutine
        
    subroutine dump_d_m2(str, dd)
        character(*), intent(in) :: str
        real(8), dimension(:,:), intent(in) :: dd

        character(LINESZ) :: fname
        
        call genfile(str, fname)
        open(DUMP_FD, file=trim(fname), status='unknown', err=42)
        write(DUMP_FD, *) dd
        close(DUMP_FD)
        return

42      call perror('Could not open file ' // trim(fname))
    end subroutine

    subroutine dump_d_m3(str, dd)
        character(*), intent(in) :: str
        real(8), dimension(:,:,:), intent(in) :: dd

        character(LINESZ) :: fname
        
        call genfile(str, fname)
        open(DUMP_FD, file=trim(fname), status='unknown', err=42)
        write(DUMP_FD, *) dd
        close(DUMP_FD)
        return

42      call perror('Could not open file ' // trim(fname))
    end subroutine
end module
    
