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
!> @file pprinter.F90
!! Handles debug/error messages

!> Pretty printer 
!! @todo Documentation
module pprinter
    use ISO_FORTRAN_ENV
    implicit none

    integer(INT32), parameter :: STDOUT=OUTPUT_UNIT
    integer(INT32), parameter :: STDERR=ERROR_UNIT
    integer(INT32), parameter :: DEVNULL=-1
    integer(INT32), parameter :: NONSTDOUT=7

    integer(INT32), parameter :: PPBUFSZ=8096
    
    character(PPBUFSZ) :: ppbuf 

    integer(INT32) :: pplun=6
    integer(INT32) :: ppwrn=6
    integer(INT32) :: pperr=6
    integer(INT32) :: ppdbg=6
    integer(INT32), private :: level=0

    interface xstr
        module procedure xstr_isp
        module procedure xstr_rdp
        module procedure xstr_isp_vec
        module procedure xstr_rdp_vec
        module procedure xstr_frm_isp
        module procedure xstr_frm_rdp
    end interface
    
    interface pprint
        module procedure pprint_str
        module procedure nl
    end interface
contains
    subroutine pprint_str(str)
        character(*), intent(in) :: str

        if (pplun == DEVNULL) return
        
        write(pplun, 100) trim(str)
100 format(x,a)
    end subroutine

    subroutine pnote(str)
        character(*), intent(in) :: str

        if (pplun == DEVNULL) return

        write(pplun, 100) ' *** ', trim(str)
100 format(a,a)
    end subroutine
    
    subroutine pinfo(str)
        character(*), intent(in) :: str

        if (pplun == DEVNULL) return

!        write(pplun, 100) ' +++ INFO: ', trim(str)
        write(pplun, 100) ' +++ ', trim(str)
100 format(a,a)
    end subroutine

    subroutine pwarn(str)
        character(*), intent(in) :: str

        if (ppwrn == DEVNULL) return

!        write(pplun, 100) ' ### WARNING: ', trim(str), ' ###'
        write(ppwrn, 100) ' >>> ', trim(str) 
100 format(a,a)
    end subroutine

    subroutine pdebug(str, l)
        character(*), intent(in) :: str
        integer, intent(in) :: l

#ifdef _DEBUG
        if (l > _DEBUG) return
        write(STDERR, "('DEBUG:' A)") trim(str)
#endif
        return
    end subroutine

!    subroutine pdebug(str, l)
!        character(*), intent(in) :: str
!        integer(INT32) :: l
!
!        if (level == 0) return
!        if (l > level) return
!        if (ppdbg == DEVNULL) return
!
!!        write(pplun, 100) ' DEBUG: ', trim(str), ' @@@'
!        write(ppdbg, 100) ' @@@ ', trim(str), ' @@@'
!100 format(a,a,a)
!    end subroutine

    subroutine perror(str)
        character(*), intent(in) :: str

        write(STDERR, 102) ' <<<  ERROR: ', trim(str), ' >>>'

102 format(a,a,a)
    end subroutine

    subroutine pcritical(str)
        character(*), intent(in) :: str

        write(STDERR, 101) repeat('>', 70)
        write(STDERR, 100) ' <'
        write(STDERR, 102) ' < ', trim(str)
        write(STDERR, 100) ' <'
        write(STDERR, 101) repeat('>', 70)

100 format(a)
101 format(x,a)
102 format(a,a)
    end subroutine

    subroutine nl
        if (pplun == DEVNULL) return

        write(pplun, *) 
    end subroutine

    subroutine set_pplun(u, chan)
        integer(INT32), intent(in) :: u
        character(*), optional :: chan

        logical :: o_p
        integer(INT32) :: tun
        character(32) :: ch
        
        tun=u
        ch='other'
        if (tun /= DEVNULL) then
            inquire(unit=tun,opened=o_p)
            if (.not.o_p) then
                write(ppbuf, '(a,i3,a)') 'Teletype unit ', tun,' not open'
                call pwarn(ppbuf)
                tun=STDOUT
            end if
        end if

        if (present(chan)) ch=chan

        select case(ch)
            case('warn')
                ppwrn=tun
            case('debug')
                ppdbg=tun
            case default
                pplun=tun
        end select
    end subroutine

    subroutine xpplun(u)
        integer(INT32), intent(out) :: u
        u=pplun
    end subroutine 

    subroutine set_debug_level(l)
        integer(INT32), intent(in) :: l
        level=l
    end subroutine 

    subroutine get_debug_level(l)
        integer(INT32), intent(out) :: l
        l=level
    end subroutine 

    subroutine disable_stdout()
        logical :: o_p

        inquire(STDOUT,opened=o_p)
        if (o_p) then
            close(STDOUT)
        end if
        open(STDOUT, file='/dev/null')

        inquire(NONSTDOUT,opened=o_p)
        if (.not. o_p) then
            open(NONSTDOUT, file='/dev/stdout')
        end if

        if ( pplun == STDOUT ) then
            pplun=NONSTDOUT
        end if
    end subroutine
    
    subroutine enable_stdout()
        logical :: o_p

        inquire(NONSTDOUT,opened=o_p)
        if (o_p) then
            close(NONSTDOUT)
        end if

        inquire(STDOUT,opened=o_p)
        if (.not. o_p) then
            open(STDOUT, file='/dev/stdout')
        end if

        if ( pplun == NONSTDOUT ) then
            pplun=STDOUT
        end if
    end subroutine

    function xstr_isp(arg) result(s)
        integer(INT32), intent(in) :: arg
        character(PPBUFSZ) :: s

        write(s, *) arg
    end function

    function xstr_rdp(arg) result(s)
        real(REAL64), intent(in) :: arg
        character(PPBUFSZ) :: s

        write(s, *) arg
    end function

    function xstr_isp_vec(arg) result(s)
        integer(INT32), dimension(:), intent(in) :: arg
        character(PPBUFSZ) :: s

        write(s, *) arg
    end function

    function xstr_rdp_vec(arg) result(s)
        real(REAL64), dimension(:), intent(in) :: arg
        character(PPBUFSZ) :: s

        write(s, *) arg
    end function

    function xstr_frm_isp(frm,arg) result(s)
        character(*) :: frm
        integer(INT32), intent(in) :: arg
        character(PPBUFSZ) :: s

        write(s, frm) arg
    end function

    function xstr_frm_rdp(frm,arg) result(s)
        character(*) :: frm
        real(REAL64), intent(in) :: arg
        character(PPBUFSZ) ::  s

        write(s, frm) arg
    end function
end module
