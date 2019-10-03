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
!> @file gauss.F90
!! Gaussian quadrature

!> Gaussian quadrature
!!
!! This class deals with integrating a function with the Gaussian quadrature
!! method. According to Gaussian quadrature, an integral is approximated
!! by a series at some carefully selected points @f$t_{i}@f$:
!!
!! @f[ \int\limits_a^b f(x,t)\;dt \approx \sum\limits_{i=1}^{n} \omega_i f(x, t_i) @f]
!!
!! This class uses the 12 point formulas, i.e. we have
!! *n=12*. An arbitrary interval \[a, b\] is divided into smaller intervals,
!! \[a,b\] = \[a1,a2\] U \[a2,a3\] U... U \[a(N-1), aN\], where the Gaussian
!! quadrature is used on each interval.
module GaussQuad_class
    use globals_m
    use mpi_m
    use pprinter
    implicit none
    private

    public :: GaussQuad
    public :: gauss_lobatto_grid
    ! public :: chebychev_grid

    interface GaussQuad
        module procedure :: GaussQuadInit
    end interface

    public GaussQuadInitSpectrum

    type :: GaussQuad
        private
        integer(INT32) :: nlin=10
        integer(INT32) :: nlog=8
        integer(INT32) :: ntot
        real(REAL64)   :: tstart=0.d0
        real(REAL64)   :: tlin=2.d0
        real(REAL64)   :: tlog=1.d3
        real(REAL64), allocatable :: tp(:)
        real(REAL64), allocatable :: tw(:)
!        real(REAL64), dimension(:), pointer :: tk,wk
        real(REAL64), pointer :: tpp(:)
        real(REAL64), pointer :: twp(:)
!        logical :: logtail
!        real(REAL64) :: tmin, tmax
        type(xmpi_scatter_t) :: scatt, scatw
    contains
        procedure :: destroy => GaussQuad_destroy
        procedure :: get_tpoints => GaussQuad_get_tpoints
        procedure :: get_weights => GaussQuad_get_weights
        procedure :: get_npoints => GaussQuad_get_npoints
        procedure :: get_tlog    => GaussQuad_get_tlog
    end type

contains

    function GaussQuadInit(nlin,nlog,tlin,tlog,tstart) result(new)
        type(GaussQuad)       :: new
        integer,optional      :: nlin
        integer,optional      :: nlog
        real(REAL64),optional :: tlin
        real(REAL64),optional :: tlog
        real(REAL64),optional :: tstart

        integer :: i
        integer,parameter :: np=5 ! number of t-points to print per row
        character(len=16) :: fm

        if(present(nlin))   new%nlin=nlin
        if(present(nlog))   new%nlog=nlog
        if(present(tlin))   new%tlin=tlin
        if(present(tlog))   new%tlog=tlog
        if(present(tstart)) new%tstart=tstart

        new%ntot=new%nlin+new%nlog

        allocate(new%tp(new%ntot))
        allocate(new%tw(new%ntot))

        ! Generate t-points for first interval
        call gauleg_int(new%nlin,new%tp(1:new%nlin),new%tw(1:new%nlin),&
            new%tstart,new%tlin)
        ! Generate t-points for logarithmic tail
        call gauleg_int(new%nlog,new%tp(new%nlin+1:),new%tw(new%nlin+1:),&
            new%tlin,new%tlog,log_p=.true.)
!       Master prints t-points
        if(verbo_g>1) then
            write(ppbuf,fmt='(a)') 't-points'
            call pinfo(ppbuf)
            write(fm,'(i2)') np
            fm='('//trim(fm)//'f10.3)'
            do i=0,new%ntot/np
                write(ppbuf,fmt=fm) new%tp(i*np+1:min((i+1)*np,new%ntot))
                call pinfo(ppbuf)
            end do
        end if

    end function

    pure subroutine GaussQuad_destroy(self)
        class(GaussQuad), intent(inout) :: self

        if (allocated(self%tp)) deallocate(self%tp)
        if (allocated(self%tw)) deallocate(self%tw)
        !        if (associated(self%rstart)) deallocate(self%rstart)
        !        if (associated(self%tk)) deallocate(self%tk)
        !        if (associated(self%wk)) deallocate(self%wk)
        if (mpirun_p) then
            !            call del_scatter_vec(self%scatt)
            !            call del_scatter_vec(self%scatw)
        end if
    end subroutine

    function GaussQuad_get_tpoints(self) result(r)
        class(GaussQuad), target :: self
        real(REAL64), dimension(:), pointer :: r

        integer(INT32) :: rank, ierr

        !        if (mpirun_p) then
        !    Why keep this if clause?
        if (.false.) then
            call new_scatter_vec(self%scatt, self%tp)
            rank=get_mpi_rank()
            call xmpi_scatter(self%scatt, self%tpp, ierr)
            if (ierr /= 0) then
                print *, 'get_gauss_tpoints():', ierr
                stop
            end if
            !            print *, "MPI_DEBUG:", rank, size(self%tp)
            r=>self%tpp
        else
            r=>self%tp
        end if
    end function

    function GaussQuad_get_weights(self) result(r)
        class(GaussQuad), target :: self
        real(REAL64), dimension(:), pointer :: r

        integer(INT32) :: rank, ierr

        !        if (mpirun_p) then
        !    Why keep this if clause?
        if (.false.) then
            call new_scatter_vec(self%scatw, self%tw)
            rank=get_mpi_rank()
            call xmpi_scatter(self%scatw, self%twp, ierr)
            if (ierr /= 0) then
                print *, ierr
                stop
            end if
            !            print *, "MPI_DEBUG: tw", rank, size(self%tw)
            r=>self%twp
        else
            r=>self%tw
        end if
    end function

    function GaussQuad_get_npoints(self) result(r)
        class(GaussQuad) :: self
        integer(INT32)  :: r

        r=self%ntot
    end function

    pure function GaussQuad_get_tlog(self) result(tlog)
        class(GaussQuad), intent(in) :: self
        real(REAL64)                 :: tlog

        tlog=self%tlog
    end function

    !    function get_interval_size(n,L) result(k)
    !
    !        ! Gets the factor to divide an interval of length L logarithmically
    !        ! Given a number L, it is divided in n intervals so that
    !        ! 1+k+k^2+k^3+..+k^n=L
    !        ! The first interval is always 1 unit long
    !        ! For instance, 3,7 would yield 2:
    !        ! |-|--|----|
    !        !  1  2  4
    !        !     7
    !        ! Basically the equation k^n-k*L+L-1=0 is solved by Newton's method
    !        ! These equations have only two real positive roots, 1 and the k we are interested in.
    !
    !          real(REAL64) :: k,k_p,L
    !          integer(INT32) :: n
    !
    !          ! We start a little bit to the right of the minimum
    !          k_p=(L/n)**(1.d0/(n-1.d0))
    !          print*,k_p,'k_p pre'
    !          if(k_p>1.d0) then
    !              k_p=k_p+1.d-6
    !          else
    !              k_p=0.5d0*k_p
    !          end if
    !          print*,k_p,'k_p after'
    !          do
    !              k=((n-1.d0)*k_p**n-L+1.d0)/(n*k_p**(n-1.d0)-L)
    !              if(abs(k_p-k)<1.d-12) exit
    !              k_p=k
    !          end do
    !          return
    !    end function
    !
    !    subroutine get_rstart(g)
    !        ! Define the gaussian quadrature intervals starting
    !        ! points.
    !        ! If integral.trange given in the input file consists
    !        ! of two numbers, they are considered as beginning and
    !        ! end. Otherwise, all the points must be given in the
    !        ! file
    !
    !        type(GaussQuad) :: g
    !         real(REAL64), dimension(:),pointer :: trange
    !        integer(INT32) :: i
    !
    !        call getkw_ptr(input, 'integral.trange', trange)
    !
    !        if(size(trange)==g%interv+1) then ! User gave all the points
    !            do i=1,g%interv+1
    !                g%rstart(i)=trange(i)
    !            end do
    !
    !        else if (size(trange)==2) then ! User gave beg. and end
    !            g%rstart(1)=trange(1)
    !
    !            ! NEW "OPTIMIZED" SCHEME
    !            g%rstart(2)=1.0d0
    !            do i=3,g%interv
    !                g%rstart(i)=g%rstart(i-1)+4.9d0/(g%interv-2)
    !            end do
    !    !        g%rstart(g%interv)=100.d0
    !
    !    !      tmax=log(rinfin+1.d0)
    !    !      delt=(tmax-t0)/ncell
    !    !
    !    !      do i=2,ncell
    !    !        t=(i-1)*delt
    !    !        grid%start(i)=exp(t)-1.d0
    !    !        grid%step(i-1)=(grid%start(i)-grid%start(i-1))/(npoint-1)
    !    !      end do
    !    !      grid%step(ncell)=(rinfin-grid%start(ncell))/(npoint-1)
    !    !=====
    !    !       dt=(log(g%tmax)-log(g%tmin+1.d0))/g%interv
    !    !       do i=2,g%interv
    !    !           t=log(g%tmin+1.d0)+dt*(i-1)
    !    !           g%rstart(i)=dexp(t)
    !    !       end do
    !
    !            ! EXPONENTIAL SCHEME
    !    !        k=get_interval_size(2,1.1d0)
    !    !        print*,'k',k
    !    !        do i=2,g%interv
    !    !        do i=2,3
    !    !            g%rstart(i)=g%rstart(i-1)+k**(i-1)
    !    !        end do
    !
    !            g%rstart(g%interv+1)=trange(2)
    !
    !        else if((size(trange)==1).and.(g%nlin==1)) then ! User gave only 1 t-point!
    !            g%rstart(1)=trange(1)
    !        else
    !            call perror('You have specified the integral wrongly!')
    !            call pinfo('trange should have either 2 numbers (beginning')
    !            call pinfo('and end), 1 number (and 1 t-interval, for')
    !            call pinfo('calculating the "t-spectrum"), or tinterv+1,')
    !            call pinfo('if you want to specify all the intervals')
    !            call pinfo('by hand.')
    !            call pprint('')
    !            call pinfo('Aborting...')
    !            stop
    !        end if
    !
    !        nullify(trange)
    !
    !    end subroutine

    ! Sergio's GQ scheme at an arbitrary intervall.
    ! It includes also an log GQ option.
    ! This subroutine uses the official (and stolen) GQ scheme above.
    pure subroutine gauleg_int(n,tp,tw,a,b,log_p)

        integer(INT32),    intent(in)    :: n           ! # of Gauss Points
        real(REAL64),      intent(inout) :: tp(n),tw(n) ! The tp, tp, and their weights, tw
        real(REAL64),      intent(in)    :: a,b
        logical, optional, intent(in)    :: log_p       ! Is the logarithmic calculations also
                                                        ! going to be included?
        call gauleg(n,tp,tw)

        if(present(log_p) .and. log_p) then ! If the logarithmic part is
            tp=a*(b/a)**(0.5d0*(tp+1.d0))   ! activated it will calculate
            tw=0.5d0*log(b/a)*tw*tp         ! quadrature log coordinates.
        else                                !
            tp=0.5d0*( (b+a) + (b-a)*tp)    ! If the logarithmic part is
            tw=0.5d0*(b-a)*tw               ! inactivated, it calculates
        end if                              ! the quadrature normally in
        ! [a;b] intervall.

    end subroutine


    !> Gauss-Legendre abscissae and weights for GQ integration.
    !!
    !! @param ngp Number of quadrature points
    !! @param xabsc Array that will hold the quadrature points
    !! @param weig Array that will store the integration weights
    !!
    !! The abscissae and weights are applicable when integrating a
    !! polynomial on the normalized interval \[-1.0, 1.0\]. Adapted from
    !! <http://www.aeromech.usyd.edu.au/wwwcomp/subrout.html>.
    pure subroutine  gauleg(ngp, xabsc, weig)

        implicit none
        integer,      intent(in) :: ngp
        real(REAL64), intent(out) :: xabsc(ngp), weig(ngp)

        integer :: i, j, m
        real(REAL64) :: p1, p2, p3, pp, z, z1
        real(REAL64),parameter:: EPS=3.0d-15


        m = (ngp + 1) / 2
        !* Roots are symmetric in the interval - so only need to find half of them  */

        do i = 1, m                ! Loop over the desired roots */

            z = cos( pi * (i-0.25d0) / (ngp+0.5d0) )
            !*   Starting with the above approximation to the ith root,
            !*          we enter the main loop of refinement by NEWTON'S method   */
            100         p1 = 1.0d0
            p2 = 0.0d0
            !*  Loop up the recurrence relation to get the Legendre
            !*  polynomial evaluated at z                 */

            do j = 1, ngp
                p3 = p2
                p2 = p1
                p1 = ((2.0d0*j-1.0d0) * z * p2 - (j-1.0d0)*p3) / j
            enddo

            !* p1 is now the desired Legendre polynomial. We next compute pp,
            !* its derivative, by a standard relation involving also p2, the
            !* polynomial of one lower order.      */
            pp = ngp*(z*p1-p2)/(z*z-1.0d0)
            z1 = z
            z = z1 - p1/pp             ! Newton's Method  */

            if (dabs(z-z1) .gt. EPS) GOTO  100

            xabsc(i) =  - z                     ! Roots will be bewteen -1.0 & 1.0 */
            xabsc(ngp+1-i) =  + z               ! and symmetric about the origin  */
            weig(i) = 2.0d0/((1.0d0-z*z)*pp*pp) ! Compute the weight and its       */
            weig(ngp+1-i) = weig(i)             ! symmetric counterpart         */

        end do

    end subroutine

    !> Create a uniformly distributed mesh of t-points to compute "spectra"
    function GaussQuadInitSpectrum(trange) result(new)
        type(GaussQuad) :: new
        real(REAL64), intent(in) :: trange(:)
        integer(INT32) :: i
        integer(INT32),parameter :: np=10
        real(REAL64) :: factor

        factor=10.d0**(1.d0/np)
        new%ntot=np*(log(trange(2))-log(trange(1)))/log(10.)
        allocate(new%tp(new%ntot))
        allocate(new%tw(new%ntot))
        new%tw=1.d0
        new%tp(1)=trange(1)
        do i=2,new%ntot
            new%tp(i)=new%tp(i-1)*factor
        end do

    end function


    ! where the n-point Gauss Lobatto grid in the intervall [-1,1] consists of
    ! the points -1, 1, and the n-2 roots of the first derivative of the n-1st
    ! Legendre Polynomial
    pure function gauss_lobatto_grid(n, left, right) result(tp)
        implicit none
        integer(INT32),  intent(in) :: n           ! # of Gauss-Lobatto Points
        real(REAL64),    intent(in) :: left, right
        real(REAL64)                :: tp(n) ! The points tp


        select case(n)
            ! mathematica: dlp5 = D[LegendreP[5, x], x]
            !              Simplify[Solve[dlp5 == 0, x]]
            case (6)
               tp(1) = -1.d0
               tp(2) = -dsqrt((7.d0 + 2*dsqrt(7.d0))/21.d0)
               tp(3) = -dsqrt((7.d0 - 2*dsqrt(7.d0))/21.d0)
               tp(4) =  dsqrt((7.d0 - 2*dsqrt(7.d0))/21.d0)
               tp(5) =  dsqrt((7.d0 + 2*dsqrt(7.d0))/21.d0)
               tp(6) = 1.d0

            case (7)
               tp(1) = -1.d0
               tp(2) = -dsqrt((15.d0 + 2*dsqrt(15.d0))/33.d0)
               tp(3) = -dsqrt((15.d0 - 2*dsqrt(15.d0))/33.d0)
               tp(4) =  0.d0
               tp(5) =  dsqrt((15.d0 - 2*dsqrt(15.d0))/33.d0)
               tp(6) =  dsqrt((15.d0 + 2*dsqrt(15.d0))/33.d0)
               tp(7) =  1.d0

            case default
    !            write(*,*) 'order not implemented'
    !            flush(6)
    !            call abort()
        end select

        ! scale to intervall
        tp=0.5d0*( (right+left) + (right-left)*tp)
    end function


    ! currently unused
    pure function chebychev_grid(n,left,right) result(tp)
        implicit none
        integer(INT32),    intent(in)    :: n       ! # of Ch Points
        real(REAL64),      intent(in)    :: left, right
        real(REAL64)                     :: tp(n) ! grid points
        integer(int32)                   :: k

        tp(1) = -1
        tp(n) = 1
        do k=1, n-2
          tp(k+1) = cos( (2*k + 1)/(2*(n-2))*PI )
        enddo

        ! scale to intervall
        tp=0.5d0*( (right+left) + (right-left)*tp)
    end function


end module

