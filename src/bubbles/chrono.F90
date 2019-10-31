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
!> @file chrono.F90 Nested timers.

!> This module provides the Chrono type, a convenient interface to add nested
!! timers to a code. Its aim is to provide in-program timing reports, and to be
!! used for simple profiling.
!!
!! The two main methods used by Chrono are Chrono::split, which starts timing a
!! new section, and Chrono::stop(), which finished the section. If Chrono::split
!! is called again, before Chrono::stop(), a new subsection is created.
!!
!! The following piece of code illustrates how it is used:
!!
!!~~~~~~~~~~~~~~~~~~~~{.f90}
!!program chrono_example
!!    use Chrono_class
!!    implicit none
!!
!!    bigben=Chrono("chrono_test")
!!    ! Indentation added to facilitate visualization
!!        call bigben%split("A")
!!            call bigben%split("B")
!!                call bigben%split("C")
!!                call bigben%stop
!!                call bigben%split("D")
!!                    call bigben%split("E")
!!                    call bigben%stop
!!                call bigben%stop
!!            call bigben%stop
!!            call bigben%split("F")
!!            call bigben%stop
!!        call bigben%stop
!!        call bigben%split("G")
!!        call bigben%stop
!!    call bigben%stop
!!    call bigben%print()
!!
!!    call bigben%destroy()
!!end program
!!~~~~~~~~~~~~~~~~~~~~
!!The output produced (by `bigben%%print()`) will look like:
!!\verbatim
!! ╚═chrono_test (.000 s)
!!     ╠═A (.000 s, 54.1%)
!!     ║   ╠═B (.000 s, 60.0%)
!!     ║   ║   ╠═C (.000 s, 11.3%)
!!     ║   ║   ╚═D (.000 s, 37.6%)
!!     ║   ║       ╚═E (.000 s, 26.1%)
!!     ║   ╚═F (.000 s, 5.9%)
!!     ╚═G (.000 s, 3.2%)
!!\endverbatim
!!The calls to `split` and `stop` should be able to be present all around the
!!code, the overhead should be negligible. Furthermore, if the main timer is
!!never initialized, those calls return immediately, so the overhead should
!!be minimal.
!!
!!### MPI support ###
!!This module should work also for MPI runs.
!!If the module is preprocessed with the macro `-DHAVE_MPI`, the function
!!`MPI_wtime()` is used for measuring times.
!> @todo Alert when too few/too many timers are stopped.
!> @todo Level parameter to avoid recording events too deep into the tree?

module Chrono_class
    use ISO_FORTRAN_ENV
#ifdef HAVE_MPI
    use mpi
#endif
    implicit none

    public :: Chrono
    public :: bigben

    private
    !> Label length
    integer, parameter :: LABLEN=40
    !> Kind parameter for UTF-8.
    integer, parameter :: ucs4  = selected_char_kind ('ISO_10646')
    !> Number of clock counts in one second.
    integer(INT64)     :: CLOCK_COUNT

    !> Timer for a single section (tree node).
    type :: Timer
        !> Label
        character(len=LABLEN,kind=ucs4)    :: label
        !> Start time
        real(REAL64)         :: t0=-1.d0
        !> End time
        real(REAL64)         :: t1=-1.d0
        !> Pointer to first child
        type(Timer), pointer :: child  =>NULL()
        !> Pointer to next sibling
        type(Timer), pointer :: sibling=>NULL()
        !> Pointer to parent
        type(Timer), pointer :: parent =>NULL()
    contains
        procedure  :: print   => timer_print
        procedure  :: dump    => timer_dump
        procedure  :: get_dt  => timer_get_dt
        procedure  :: destroy => timer_destroy
    end type

    !> Nested timers.

    !> ### Initialization / destruction ###
    !! ~~~{.f90}
    !! bigben=Chrono("chrono_test")
    !! ...
    !! call bigben%destroy()
    !! ~~~
    !!
    !! ### Measuring the time taken by a section of code ###
    !! ~~~{.f90}
    !! call bigben%split("Section")
    !! ! ... Code to be timed ...
    !! call bigben%stop
    !! ~~~
    !!
    !! ### Measuring a section with two subsections ###
    !! ~~~{.f90}
    !! call bigben%split("Section")
    !! ! ...
    !! call bigben%split("Subsection")
    !! ! ...
    !! call bigben%stop
    !! call bigben%split("Subsection")
    !! ! ...
    !! call bigben%stop
    !! ! ...
    !! call bigben%stop
    !! ~~~
    !! ### Caveats ###
    !! Make sure you initialize all timers you start. The symptoms are two:
    !! * The report shows bogus times (e.g. negative).
    !! * The tree structure looks different from what you expected.
    type :: Chrono
        !> The first Timer node.
        type(Timer), pointer     :: first  =>NULL()
        !> Pointer to the most recent running Timer.
        type(Timer), pointer     :: current=>NULL()
        integer              :: iproc
    contains
        procedure   :: split   => chrono_split
        procedure   :: stop    => chrono_stop
        procedure   :: stop_and_print => Chrono_stop_and_print
        procedure   :: print   => chrono_print
        procedure   :: print_current => Chrono_print_current
        procedure   :: dump    => chrono_dump
        procedure   :: destroy => chrono_destroy
    end type

    interface Chrono
        module procedure  :: Chrono_new
    end interface

    interface Timer
        module procedure  :: Timer_new
    end interface

    !> Public instance (broadcasted to be used by all modules).
    type(Chrono) :: bigben
contains
    
    !> Fetch wall time.
    
    !> If using MPI (`HAVE_MPI` macro), use `MPI_wtime()` function. Otherwise
    !! use `system_clock()`.
    function wall_time() result(t)
        real(REAL64)   :: t
#ifdef HAVE_MPI
#ifdef HAVE_OMP
        !$OMP BARRIER
        !$OMP MASTER 
#endif
        t=mpi_wtime()
#ifdef HAVE_OMP
        !$OMP END MASTER
        !$OMP BARRIER
#endif
#else
        integer(INT64) :: t_int

        call system_clock(t_int)
        t=real(t_int,REAL64)/CLOCK_COUNT
#endif
        return
    end function

    !> Constructor
    function Timer_new(label, parent) result(new)
        type(Timer)    :: new
        !> Name of the new subsection.
        character(*)   :: label
        !> Section which the new subsection belongs to.
        type(Timer), target, optional :: parent

        new%label = label
        new%t0    = wall_time()
        if (present(parent)) new%parent => parent
    end function

    !> Pretty-print Timer. Low-level function, `Chrono%%print()` should be used.
    recursive subroutine Timer_print(self, prefix, depth, maxdepth, fd, iproc)
        class(Timer)                            :: self
        !> String to prepend to output (for printing tree structure).
        character(len=*,kind=ucs4), intent(in)  :: prefix
        !> Current depth in the tree.
        integer,                    intent(in)  :: depth
        !> Maximum depth to be printed.
        integer,                    intent(in)  :: maxdepth
        !> File descriptor where to put output.
        integer,                    intent(in)  :: fd
        !> Number of processor
        integer,                    intent(in)  :: iproc

        character(len=len(prefix)+4,kind=ucs4) :: new_prefix
        integer, parameter                     :: LINELEN=80
        character(len=LINELEN,      kind=ucs4) :: line

        real :: dt

        ! Characters for pretty printing tree structure.
        character(len=2,kind=ucs4), parameter:: c1=achar(9568,kind=ucs4)//&!"╠═"
                                                   achar(9552,kind=ucs4)
        character(len=2,kind=ucs4), parameter:: c2=achar(9562,kind=ucs4)//&!"╚═"
                                                   achar(9552,kind=ucs4)
        character(len=2,kind=ucs4), parameter:: c3=achar(9553,kind=ucs4)//&!"║ "
                                                   ucs4_" "
        if(depth>maxdepth) return

        if(associated(self%sibling)) then
            write(line,*) prefix//c1//self%label
            new_prefix=prefix//c3
        else
            write(line,*) prefix//c2//self%label
            new_prefix=prefix//ucs4_"  "
        end if

        dt=self%get_dt()

        write(line,'(a," ","(",f0.5," ","s")') trim(line(:60)), dt
        if(associated(self%parent)) &
            write(line,'(a,","," ",f0.1,"%")') trim(line(:70)), &
                dt/self%parent%get_dt()*100.
        write(line,'(a,")")') trim(line(:79))
        write(fd,'(a, "(iproc: ",i2,")")',         advance='no') line, iproc
        write(fd, *)

        if(associated(self%child)) &
            call self%child%print(new_prefix, depth+1, maxdepth, fd, iproc)
        if(associated(self%sibling)) &
            call self%sibling%print(prefix, depth, maxdepth, fd, iproc)
    end subroutine

    !> Output into machine-parseable format.

    !> Low-level function, Chrono%dump() should be used.
    recursive subroutine Timer_dump(self, depth, fd, start)
        class(Timer)                            :: self
        !> Current depth in the tree.
        integer,                    intent(in)  :: depth
        !> File descriptor where to put output.
        integer,                    intent(in)  :: fd
        !> Starting point of the first timer.
        real(REAL64),               intent(in)  :: start

        character, parameter :: tab=achar(9)

        write(fd,'(i3,a1,f15.6,a1,f15.6,a1,a)') depth, tab, &
            self%t0-start,tab,&
            self%t1-start,tab,trim(self%label)
        if(associated(self%child))   call self%child%dump  (depth+1, fd, start)
        if(associated(self%sibling)) call self%sibling%dump(depth,   fd, start)
    end subroutine

    !> Compute total time elapsed by the (stopped) Timer.
    function Timer_get_dt(self) result(dt)
        class(Timer)   :: self
        real(REAL64)   :: dt

        dt=self%t1-self%t0
    end function

    !> Deallocate siblings and children.
    recursive subroutine Timer_destroy(self)
        class(Timer) :: self
    
        if(associated(self%sibling)) then
            call self%sibling%destroy()
            deallocate(self%sibling)
        end if
        if(associated(self%child)) then
            call self%child%destroy()
            deallocate(self%child)
        end if
    end subroutine

    !> Chrono constructor.
    function Chrono_new(name, iproc) result(new)
        !> Name of the main timer (default "Main").
        character(*), optional, intent(in) :: name
        type(Chrono), target               :: new
        integer, optional, intent(in)      :: iproc
        if (present(iproc)) then
            new%iproc = iproc
        else
            new%iproc = 0
        end if
        call system_clock(count_rate=CLOCK_COUNT)

        allocate(new%first)
        new%current=>new%first
        if(present(name)) then
            new%current=Timer(name)
        else
            new%current=Timer("Main")
        end if

        return
    end function

    !> Pretty-print nested timers.
    subroutine Chrono_print(self, maxdepth, fd)
        class(Chrono)                 :: self
        !> Deepest level printed (default: HUGE).
        integer, intent(in), optional :: maxdepth
        !> File unit for the output (default: `OUTPUT_UNIT`)
        integer, intent(in), optional :: fd

        integer                       :: maxdepth_w
        integer                       :: fd_w

        if(.not.associated(self%first)) return

        if(present(fd)) then
            fd_w=fd
        else
            fd_w=OUTPUT_UNIT
        end if
        ! If fd_w was already open, is this legal? It seems to work...
        open(fd_w,encoding="UTF-8")
        maxdepth_w=huge(maxdepth_w)
        if(present(maxdepth)) then
            maxdepth_w=maxdepth
        else
            maxdepth_w=huge(maxdepth_w)
        end if
        call self%first%print(ucs4_"",depth=0, maxdepth=maxdepth_w, fd=fd_w, iproc=self%iproc)
    end subroutine

    !> Pretty-print nested timers.
    subroutine Chrono_print_current(self, maxdepth, fd)
        class(Chrono)                 :: self
        !> Deepest level printed (default: HUGE).
        integer, intent(in), optional :: maxdepth
        !> File unit for the output (default: `OUTPUT_UNIT`)
        integer, intent(in), optional :: fd

        integer                       :: maxdepth_w
        integer                       :: fd_w

        if(.not.associated(self%first)) return

        if(present(fd)) then
            fd_w=fd
        else
            fd_w=OUTPUT_UNIT
        end if
        ! If fd_w was already open, is this legal? It seems to work...
        open(fd_w,encoding="UTF-8")
        maxdepth_w=huge(maxdepth_w)
        if(present(maxdepth)) then
            maxdepth_w=maxdepth
        else
            maxdepth_w=huge(maxdepth_w)
        end if
        call self%current%print(ucs4_"",depth=0, maxdepth=maxdepth_w, fd=fd_w, iproc=self%iproc)
    end subroutine

    !> Dump `Chrono` into file, in a computer-friendly format.

    !> A python script is shipped somewhere to parse the output of this
    !! subroutine into a convenient struct...
    !! @todo Ship the python script!
    subroutine Chrono_dump(self, fd)
        class(Chrono),          intent(in) :: self
        !> Output file unit
        integer,      optional, intent(in) :: fd

        ! Output to standard output by default
        integer :: fd_w=OUTPUT_UNIT

        if(.not.associated(self%first)) return

        if(present(fd)) fd_w=fd
        call self%first%dump(0, fd_w, self%first%t0)
    end subroutine

    !> Destructor.
    subroutine Chrono_destroy(self)
        class(Chrono)   :: self

        if(.not.associated(self%first)) return

        call self%first%destroy()
        deallocate(self%first)
        return
    end subroutine

    !> Start timing a new section.
    subroutine Chrono_split(self, label)
        class(Chrono),     target     :: self
        !> Name of the new section.
        character(*),      intent(in) :: label

        type(Timer), pointer :: next

        ! Ignore if self was never started.
        if(.not.associated(self%first)) return

        ! Find what is the next timer: either the first child of current, or a
        ! sibling to the youngest child of current
        if(.not.associated(self%current%child)) then
            allocate(self%current%child)
            next=>self%current%child
        else
            next=>self%current%child
            do
                if(.not.associated(next%sibling)) then
                    allocate(next%sibling)
                    next=>next%sibling
                    exit
                end if
                next=>next%sibling
            end do
        end if

        ! Create the new child
        next=Timer(label, self%current)

        ! Make current point to the recently created timer
        self%current=>next
        return

    end subroutine

    !> Stop current section.
    subroutine Chrono_stop(self)
        class(Chrono)     :: self

        ! Ignore if self was never started.
        if(.not.associated(self%first)) return
        ! Add stop time
        self%current%t1=wall_time()
        ! The current timer is now the parent of the timer we just stopped
        if(associated(self%current%parent)) self%current=>self%current%parent
        return
    end subroutine

    subroutine Chrono_stop_and_print(self)
        class(Chrono), intent(inout) :: self
        ! Ignore if self was never started.
        if(.not.associated(self%first)) return
        ! Add stop time
        self%current%t1=wall_time()
        call self%print_current(maxdepth = 0)
        ! The current timer is now the parent of the timer we just stopped
        if(associated(self%current%parent)) self%current=>self%current%parent
    end subroutine 

end module
