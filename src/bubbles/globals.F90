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
!> @file globals.F90
!! Global parameters

!> Global parameters
!!
!! This module contains global parameters, utility functions, and
!! automatically makes certain modules available to all sub-modules.
!! @author Jonas Juselius  
!! @date Mon Apr 25 09:35:00 CEST 2005
module globals_m
    use ISO_FORTRAN_ENV
    use ISO_C_BINDING
    use pprinter
    !use getkw_class
    use RealArrays_class
    use Chrono_class
    implicit none 
    
    integer, parameter :: OMP_GLOBAL_THREAD_COUNT=12
    integer, parameter :: BUFLEN=256
    integer, parameter :: LINESZ=128
    integer, parameter :: MAX_LINE_LEN=200
    integer, parameter :: X_=1
    integer, parameter :: Y_=2
    integer, parameter :: Z_=3
    integer, parameter :: T_=4
    integer, parameter :: RECLEN=4 !bytes

    integer, parameter :: SL_COORD=X_ ! Coordinate to slice: X for the moment

    ! math constants
    real(REAL64), parameter :: PI=3.141592653589793D0
    real(REAL64), parameter :: FOURPI=1.2566370614359173D1
    !real(REAL64), parameter :: PI=atan(1.d0)*4.d0
    real(REAL64), parameter :: PI_FAC=6.349363593424098D-002 ! (0.5/PI)**0.75
    real(REAL64), parameter :: TWOOVERSQRTPI=1.1283791670955125739D0
!    real(REAL64), parameter :: PI_INV=3.183098861837907D-001 ! 1/PI
    real(REAL64), parameter :: ZETA=0.5d0
    real(REAL64), parameter :: LN2=LOG(2.0)

    ! physics constants
    real(REAL64), parameter :: C_AU = 137.035999139d0
    real(REAL64), parameter :: ALPHA = 1.d0/C_AU ! 0.0072973525664
    real(REAL64), parameter :: ALPHA2 = ALPHA**2
    ! according to NIST, 2014
    real(REAL64), parameter :: AU2M  = 0.52917721067d-10
    real(REAL64), parameter :: AU2NM = 0.52917721067d-1
    real(REAL64), parameter :: AU2A  = 0.52917721067d0
    real(REAL64), parameter :: AU2PM = 52.917726067d0
    real(REAL64), parameter :: A2AU  = 1.88972612546d0
    real(REAL64), parameter :: NM2AU = 1.88972612546d+1
    real(REAL64), parameter :: PM2AU = 1.88572612546d-2

    ! default filenames
    character(*), parameter :: DEFAULT_INPUT='genpot.inp'
    character(80) :: DDEN_FN='dens.tmp'
    character(80) :: DPOT_FN='pot.tmp'

    ! file descriptors
    integer, parameter :: READ_FD=22
    integer, parameter :: WRITE_FD=42
    integer, parameter :: CUBE_FD=43
    integer, parameter :: INPUT_FD=87
    integer, parameter :: DEBUG_FD=88
    integer, parameter :: OUTPUT_FD=89
    integer, parameter :: GOM_FD=90
    integer, parameter :: DDEN_FD=30
    integer, parameter :: DPOT_FD=31
    integer, parameter :: QPOT_FD=32
    integer, parameter :: QPOT2_FD=33
    integer, parameter :: BUBLIB_FD=34
    integer, parameter :: AXIS_FD=35

    logical :: mpirun_p=.false.
    logical :: master_p=.false.
    logical :: io_node_p=.false.
    integer :: debug_g=0
    integer :: verbo_g=0
    logical :: direct_p=.true.
    logical :: iomode_p=.false.
    logical :: bubbles_p=.false.
    logical :: nuclear_p=.false.
    logical :: errpot_p=.false.
    logical :: selfint_p=.true.
    logical :: spectrum_p=.false.
    integer :: iproc
    integer :: nproc

    integer(INT32) :: io_comm

#ifdef HAVE_CUDA
    type(C_PTR) :: stream_container
    integer     :: STREAMS_PER_DEVICE

    interface
        subroutine StreamContainer_init(streamContainer, streams_per_device)  bind(C)
            use ISO_C_BINDING
            type(C_PTR)           :: streamContainer
            integer(C_INT), value :: streams_per_device

        end subroutine
    end interface


    interface
        type(C_PTR) function StreamContainer_get_subcontainer(streamContainer,  &
                       subcontainer_order_number, total_subcontainers)  bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: streamContainer
            integer(C_INT), value :: subcontainer_order_number
            integer(C_INT), value :: total_subcontainers

        end function
    end interface

    interface
        integer(C_INT) function StreamContainer_get_number_of_devices(streamContainer)  bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: streamContainer

        end function
    end interface

    interface
        integer(C_INT) function StreamContainer_get_streams_per_device(streamContainer)  bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: streamContainer

        end function
    end interface

    interface
        type(C_PTR) function StreamContainer_record_device_event(streamContainer, &
                                             device_order_number)  bind(C)
            use ISO_C_BINDING
            type(C_PTR),    value :: streamContainer
            integer(C_INT), value :: device_order_number
        end function
    end interface
    
    interface
        subroutine StreamContainer_destroy(integrator)  bind(C)
            use ISO_C_BINDING
            type(C_PTR), value    :: integrator

        end subroutine
    end interface
#endif

!    type(getkw_t), save :: input

!    external fseek
!    integer :: fseek

    interface xchar
        module procedure xchar_i, xchar_d
    end interface

contains

    !> @todo Documentation
    function xtrim(str) result(r)
        character(BUFLEN), intent(in) :: str
        character(BUFLEN) :: r
        
        integer(INT32) :: i
        character(1), dimension(BUFLEN) :: tmp

        tmp=transfer(str,tmp)
        
        do i=1,BUFLEN
            if (iachar(tmp(i)) == 0) then
                tmp(i) = ' '
                exit
            end if
        end do
        r=transfer(tmp,r)
        
    end function

    !> Get the number of lines in a file
    function getnlines(fd) result(n)
        !> A file descriptor
        integer(INT32), intent(in) :: fd
        integer(INT32) :: n
        
        n=0
        rewind(fd)
        do 
            read(fd,*,end=100)
            n=n+1
        end do
100     rewind(fd)
    end function

!    function getfsize(fd) result(fs)
!        integer(INT32) :: fd, fs

!        integer(INT32) :: i, ftell

!        call fseek(fd,0,2)
!        i=fseek(fd,0,2)
!        fs=ftell(fd)
!        rewind(fd)
!    end function

    !> Converts integers to formatted strings
    !!
    !! The integer width is hardcoded to 6
    function xchar_i(i) result(s)
        integer(INT32) :: i
        character(20) :: s

        write(s, '(i6)') i
        s=adjustl(s)
    end function

    !> Converts doubles to formatted strings
    !!
    !! The string format is hardcoded to 16 character width
    !! and 8 digit precision.
    function xchar_d(d) result(s)
        real(REAL64) :: d
        character(20) :: s

        write(s, '(ES16.8)') d
        s=adjustl(s)
    end function

    !> Gets the hostname of the system
    function hostname() result(hn)
        character(BUFLEN) :: hn
        integer(INT32) :: hostnm, ierr

        ierr=hostnm(hn)
    end function

    !> @todo Documentation
    subroutine basename(fname,bname)
        character(*), intent(in) :: fname
        character(*), intent(out) :: bname

        integer :: i
        i=index(fname, '.', back=.true.)
        bname=fname(1:i-1)
    end subroutine

    !> Gets the file extension of a filename
    function filext(fname) result(ext)
        character(*), intent(in) :: fname
        character(3) :: ext

        integer :: i
        i=index(fname, '.', back=.true.)
        ext=fname(i+1:)
    end function

    !> Checks if the file at folder/filename exists
    function file_exists(folder, filename) result(file_exists_)
        character(len=*), intent(in)    :: folder
        character(len=*), intent(in)    :: filename
        logical                         :: file_exists_
        
        inquire(file=trim(folder)//"/"//trim(filename), exist=file_exists_)
    end function

    !> Checks if a condition is true.
    !!
    !! Execution is halted if `expr` is not true.
    subroutine assert(expr)
            !> A logical expression to be checked
            logical, intent(in) :: expr
            if (expr) then
                return
            else
                ! Should translate to exit(1)
                write(ERROR_UNIT, *) "Assertion error: going to halt!"
                stop 1
            endif
    end subroutine

    !> Truncates n meaningful numbers of 'x' from the end and
    !! returns the result.
    !!
    !! n: how many meaningful numbers from end is truncated 
    function truncate_number(x, n) result(reslt)
        real(REAL64), intent(in) :: x
        integer,      intent(in) :: n
        integer                  :: exponential
        real(REAL64)             :: reslt, mul
        
        ! determine the magnitude of the number
        exponential = 15-n-floor(log10(abs(x)))
        mul = 10.d0**exponential
        reslt = anint(mul * x) 
        reslt = reslt / mul
    end function

    pure elemental function factorial_int(x) result(reslt)
         integer, intent(in)         :: x
         integer                     :: i
         integer                     :: reslt
         

         reslt = 1
         do i = 1, x
             reslt = reslt * i 
         end do
    end function

    pure elemental function factorial_real(x) result(reslt)
         integer, intent(in)         :: x
         integer                     :: i
         real(REAL64)                :: reslt
         

         reslt = 1
         do i = 1, x
             reslt = reslt * i 
         end do
    end function

end module
