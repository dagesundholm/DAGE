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
!> @file timer.F90
!! Convenience routines for timing execution

!> Timers for execution time and a progress bar implementation

module timer_m
    use pprinter
    use ISO_FORTRAN_ENV
    implicit none

    integer(INT32), parameter :: BUFLEN=80
    real(REAL32) :: etime, dtime ! Why not use SP?
    character(BUFLEN) :: fdate
    
    real(REAL32) :: delta_t 

    real(REAL32), dimension(2) :: times 

    public :: stockas_klocka
    public :: swatch
    public :: progress_bar
    public :: pretty_time
    public :: report_time
    private
contains
    subroutine stockas_klocka()
        real(REAL32) :: etime, dtime 
        character(BUFLEN) :: fdate
        delta_t=etime(times) ! It is recomended to call it as a subroutine rather than a function.
        !call pprint(repeat('-', 70))
        write(ppbuf, '(a,f9.2,a)') '   wall time:', delta_t, 'sec'
        call pprint(ppbuf)
        write(ppbuf, '(a,f9.2,a)') '        user:', times(1), 'sec'
        call pprint(ppbuf)
        write(ppbuf, '(a,f9.2,a)') '         sys:', times(2), 'sec'
        call pprint(ppbuf)
        !call pprint(repeat('-', 70))
        !call pprint(fdate())
    end subroutine 

    subroutine swatch(foo)
        integer(INT32), intent(in) :: foo
        real(REAL32) :: etime, dtime
        character(BUFLEN) :: fdate ! Where is this variable used?
        
        delta_t=dtime(times) ! "call dtime(times,delta_t)". Var times is 2D.
        if (foo > 0) then    
            call pprint(repeat('.', 33))
            write(ppbuf, '(a,f9.2,a)') '   wall time:', delta_t, 'sec'
            call pprint(ppbuf)
            write(ppbuf, '(a,f9.2,a)') '        user:', times(1), 'sec'
            call pprint(ppbuf)
            write(ppbuf, '(a,f9.2,a)') '         sys:', times(2), 'sec'
            call pprint(ppbuf)
            call pprint(repeat('.', 33))
        end if
    end subroutine 

    subroutine progress_bar(cur,tot_in)
!        logical,save :: started=.FALSE.
        integer(INT32) :: cur
        integer(INT32), optional :: tot_in
        integer(INT16) :: proglen
        integer(INT16), save :: tot
        integer(INT64), save :: t(2), ticks
        integer(INT64) :: percent,elapsed(2), eta(2), total(2)
        integer(INT32), parameter :: barlen=36, perlen=6,halfbar=(barlen-perlen)/2
        character(40) :: time_string
        character(barlen) :: bar
        character(perlen) :: percent_string
        character(*),parameter :: form_str='(X," Elapsed: ",i2.2,":",i2.2,&
        &" ETA: ",i2.2,":",i2.2," Total: ",i2.2,":",i2.2)'

        if(cur==0) then
            ! tot_in has to be given when initializing the bar
            tot=tot_in
            call system_clock(count_rate=ticks)
            call system_clock(t(1))
            write(0,'(a)',advance='no') "|"//repeat(".",halfbar)//"(  0%)"//&
                repeat(".",halfbar)//"| Elapsed: --:-- ETA: --:--"
        else
            call system_clock(t(2))
            proglen=(cur*barlen)/tot
            percent=cur*100/tot
            write(percent_string,'("(",i3,"%)")') percent
            ! Create bar string
            bar=repeat("=",min(halfbar,proglen))
            if(proglen<halfbar) bar=trim(bar)//">"
            if(proglen<halfbar-1) bar=trim(bar)//repeat(".",halfbar-proglen-1)
            bar=trim(bar)//percent_string
            if(proglen>halfbar+perlen) bar=trim(bar)//repeat("=",proglen-halfbar-perlen)
            if((proglen>halfbar+perlen-1).and.(proglen<barlen)) bar=trim(bar)//">"
            if(proglen<barlen) bar=trim(bar)//repeat(".",min(halfbar,barlen-proglen-1))

            elapsed(1)=    (t(2)-t(1))/ticks/60
            elapsed(2)=mod((t(2)-t(1))/ticks,60)
            eta(1)=(tot-cur)*    (t(2)-t(1))/ticks/60/cur
            eta(2)=mod((tot-cur)*(t(2)-t(1))/ticks/cur,60)
            total(1)=tot*    (t(2)-t(1))/ticks/60/cur
            total(2)=mod(tot*(t(2)-t(1))/ticks/cur,60)
            write(time_string,fmt=form_str) elapsed, eta, total
            ! char(13) is a carriage return! Great!
            write(0,'(a)',advance='no') char(13)//"|"//bar//"|"//time_string
            ! Jump line when done
            if(cur==tot) write(0,*)
        end if

    end subroutine

    !> Unified method to report times
    subroutine report_time(string, time, total)
        character(*)   :: string
        real           :: time
        real, optional :: total

!        write(ppbuf,'("@@@ ",a30,": ",f10.2," s",a)') string, time, pretty_time(time)
        if(present(total)) then
            write(ppbuf,'("@@@ ",a30,": ",f10.2," s",f6.2," %")') string, time, time/total*100.
        else
            write(ppbuf,'("@@@ ",a30,": ",f10.2," s")') string, time
        end if
        call pprint(ppbuf)
    end subroutine

    function pretty_time(time)
        character(40) :: pretty_time
        character(15) :: tmp
        real :: time
        real :: seconds
        integer, parameter :: D=1, H=2, M=3
        character(1), parameter :: ch(4)=['d','h','m','s']
        integer :: t(3)
        integer :: i,j

        seconds=time
        t(M)=seconds/60
        t(H)=t(M)   /60
        t(D)=t(H)   /24
        t(H)=t(H)-24*t(D)
        t(M)=t(M)-60*(24*t(D)+t(H))
        seconds=seconds-60*(60*(24*t(D)+t(H))+t(M))

        do j=1,3
            if (t(j)>0) exit
        end do
        write(tmp,'(*(I2,X,A1,":"),f5.2,A1)') (t(i),ch(i),i=j,3)
        write(pretty_time,'(f5.2,X,A1)') seconds,ch(4)
        pretty_time=trim(tmp)//trim(pretty_time)
    end function

end module

