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

!> @file pbc.F90
!! Periodic boundary conditions

!> Subroutines to deal with PBCs (periodic boundrary conditions)
!!
module PBC_class
    !
    use globals_m

    public :: PBC

    private

    type :: PBC
        private
        logical :: pbc_dim(3)
        integer(INT32) :: num_pbc
        integer(INT32) :: lims(3,2),first(3),chkpt(3),finnish(3),step(3)
        ! For an interval passing through 0:
        ! a-----0---------------b
        ! lims(crd,:)=(/a,b/)
        ! first=0
        ! chkpt=a   (the one from a and b that has the smallest absolute value)
        ! finnish=b (the one from a and b that has the largest absolute value)
        ! step=sign(1,finnish)
        ! If the interval does not go through 0
        ! lims(crd,:)=(/a,b/)
        ! first=a (the one from a and b that has the smallest absolute value)
        ! finnish=b (the one from a and b that has the largest absolute value)
        ! chkpt=first-1
        ! step=sign(1,finnish)
        !
        ! This is used in get_1d_disp. Check it out to understand this mess ;)

    contains
        procedure :: next_tv       => pbc_next_tv
        procedure :: max_3dshell   => pbc_max_3dshell
        procedure :: any           => pbc_any
        procedure :: at_dim        => pbc_at_dim
        procedure :: contains      => pbc_contains
        procedure :: get_num_pbc   => pbc_get_num_pbc
        procedure :: contains_tv1d => pbc_contains_tv1d
        procedure :: contains_tv3d => pbc_contains_tv3d
        procedure :: get_1d_disp   => pbc_get_1d_disp
    end type

    interface PBC
        module procedure :: PBC_init
    end interface
contains
    !> Initializes pbc_idx from a pbc string
    !
    ! The format of a PBC string is
    ! x[min:max]y[min:max]z[min:max]
    !
    ! For instance:
    ! x-1:100y-1000:1000z-8:0
    ! A missing number means infinity in that direction
    ! x0:y0:z0: is the corner of a cube
    ! The two missing numbers (the colon can then be omitted as well) means infinity in both directions
    ! x0:yz is a surface
    !
    ! The picture below describes the system x0:0y0:0z
    !      ________________________
    !     /_____/_____/_____/_____/|        |x__z
    !     |     |     |     |     ||        /y
    !  ...|_____|_____|_____|_____|/...
    !
    function PBC_init(pbc_string) result(new)
        character(*), intent(in) :: pbc_string
        type(PBC)                :: new

        character(6),parameter :: dimchar="xXyYzZ"
        character(BUFLEN) :: string
        integer(INT32) :: pos,prevpos,crd,colonpos,imin,imax
        integer(INT32) :: tmp

        new%num_pbc=0
        new%pbc_dim=.FALSE.
        new%first  =0
        new%chkpt  =0
        new%finnish=0
        new%step   =0
        new%lims   =0
        if(len(trim(pbc_string))==0) then
            new%lims=0
        else
            prevpos=len(pbc_string)+1
            do
                pos=scan(pbc_string(:prevpos-1),dimchar,.TRUE.) ! BACK=.TRUE., it scans backwards for dimchar characters.
                if(pos==0) exit
                string=pbc_string(pos:prevpos-1)
                prevpos=pos
                select case(string(1:1)) ! This select case clause defines the crd (coordinate).
                    case("x","X")
                        crd=1
                    case("y","Y")
                        crd=2
                    case("z","Z")
                        crd=3
                end select
                new%num_pbc=new%num_pbc+1
                new%pbc_dim(crd)=.TRUE.
                colonpos=scan(string,":")
                if (colonpos==0.or.colonpos==len(trim(string))) then ! Checks whether the colon exists or whether
                    new%lims(crd,2)=huge(new%lims(crd,2))          ! the colon is the last string character.
                else                                            ! Else the upper limit is the last character.
!                    print*,string(2:colonpos-1)
                    read(string(colonpos+1:),*) new%lims(crd,2)
                end if
                if (colonpos==0.or.colonpos==2) then
                    new%lims(crd,1)=-huge(new%lims(crd,1))
                else
!                    print*,string(colonpos+1:)
                    read(string(2:colonpos-1),*) new%lims(crd,1)
                end if

                if(new%lims(crd,1)>new%lims(crd,2)) then
                    call pinfo("The PBC lower limit for one of the axes was higher than the upper limit:")
                    call pinfo(trim(string))
                    call pinfo("I'll swap them and continue... But check you didn't make a mistake!")
                    tmp=new%lims(crd,1)
                    new%lims(crd,1)=new%lims(crd,2)
                    new%lims(crd,2)=tmp
                end if

                imin=minloc(abs(new%lims(crd,:)),1)
                imax=maxloc(abs(new%lims(crd,:)),1)
                new%finnish(crd)=new%lims(crd,imax)
                new%step(crd)   =sign(1,new%finnish(crd))

                if(new%lims(crd,1)<=0.and.0<=new%lims(crd,2)) then
                    new%first(crd)=0
                    new%chkpt(crd)=new%lims(crd,imin)
                else
                    new%first(crd)=new%lims(crd,imin)
                    new%chkpt(crd)=new%first(crd)-new%step(crd)
                end if
            end do
        end if

    end function

    pure function PBC_any(self)
        class(PBC), intent(in) :: self
        logical                :: PBC_any

        PBC_any=any(self%pbc_dim)
    end function

    pure function PBC_at_dim(self,crd)
        class(PBC),     intent(in) :: self
        integer(INT32), intent(in) :: crd
        logical                    :: PBC_at_dim
        PBC_at_dim=self%PBC_dim(crd)
    end function

    pure function PBC_contains(self,i,crd)
        !Returns true if i is a valid displacement for this pbc
        ! i.e., if pbc=x0:
        ! self%contains(0,1)=.true.
        ! self%contains(1,1)=.true.
        ! self%contains(-1,1)=.false.
        ! self%contains(0,2)=.true.
        ! self%contains(1,2)=.false.
        class(PBC),     intent(in) :: self
        integer(INT32), intent(in) :: i
        integer(INT32), intent(in) :: crd
        logical                    :: PBC_contains

        PBC_contains=(i>=self%lims(crd,1).and.i<=self%lims(crd,2))
        return
    end function

    pure function PBC_get_num_pbc(self)
        class(PBC),    intent(in) :: self
        integer(INT32)            :: PBC_get_num_pbc
        PBC_get_num_pbc=self%num_pbc
    end function

     function PBC_next_tv(self,a,flag) result(tv_out)
        class(PBC),     intent(in)      :: self
        integer(INT32), intent(in)      :: a
        integer(INT32), save            :: tv(3)
        integer(INT32)                  :: tv_out(3)
        ! Flag has two purposes:
        !   - If flag is .TRUE. on input, return the first tv for shell a
        !   - Flag will be .TRUE. on output if the last tv was passed already
        logical,        intent(inout)   :: flag

        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! This subroutine is used to get a new translation vector belonging to the a-th octahedron
        ! The input are the last tv used and the octahedron index
        ! For instance, for the second octahedron we will get all the (18) vectors
        ! (+/-2,0,0) (0,+/-2,0) (0,0,+/-2) (+/-1,+/-1,0) (+/-1,0,+/-1) (0,+/-1,+/-1)
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        ! Generate the starting tv (translation vector)
        if(flag) then
            ! PUT SANITY CHECK HERE (a>0, PBC shells for a k>=a exist,etc.)
            flag=.FALSE.
            tv=[a,0,0]
            if (self%contains_tv3d(tv)) then
                tv_out=tv
                return
            end if
        end if

        ! This block generates the next lower tv
        do while(tv(X_)>-a)
            do while(tv(Y_)>-(a-abs(tv(X_))))
                ! Try to flip sign
                if (tv(Z_)>0) then
                    tv(Z_)=-tv(Z_)
                    if (self%contains_tv3d(tv)) then
                        tv_out=tv
                        return
                    end if
                end if
                ! Diminish tv(Y_)
                tv(Y_)=tv(Y_)-1
                if (.not.self%contains_tv1d(Y_,tv(Y_))) cycle
                tv(Z_)=a-abs(tv(X_))-abs(tv(Y_))
                if (self%contains_tv3d(tv)) then
                    tv_out=tv
                    return
                end if
            end do
            ! Diminish tv(X_)
            tv(X_)=tv(X_)-1
            if (.not.self%contains_tv1d(X_,tv(X_))) cycle
            tv(Y_)=a-abs(tv(X_))
            tv(Z_)=a-abs(tv(X_))-abs(tv(Y_))
            if (self%contains_tv3d(tv)) then
                tv_out=tv
                return
            end if
        end do
        flag=.TRUE.

    end function

    subroutine PBC_get_1d_disp(self,crd, initialize,disp,duplicate,thiswaslast)
        class(PBC),     intent(in)    :: self
        integer(INT32), intent(in)    :: crd
        logical,        intent(inout) :: initialize
        integer(INT32), intent(out)   :: disp
        logical,        intent(out)   :: duplicate
        logical,        intent(out)   :: thiswaslast
        ! Returns disp, duplicate and exitsignal
        ! disp is the value of the next displacement
        ! duplicate tells if -disp has to be also considered
        ! exitsignal tells that this was the last contribution

        if(initialize) then
            disp=self%first(crd)
            duplicate=.false.
            initialize=.false.
            thiswaslast=.false.
        else
            disp=disp+self%step(crd)
            if(abs(disp)<=abs(self%chkpt(crd))) then
                duplicate=.true.
            else
                duplicate=.false.
            end if
        end if

        if(abs(disp)>=abs(self%finnish(crd))) thiswaslast=.true.

        return
    end subroutine

    pure function PBC_contains_tv1d(self,dm,tv1d) result(is_contained)
    ! Check if tv1d is a valid point in dimension dm
        class(PBC),     intent(in) :: self
        integer(INT32), intent(in) :: dm
        integer(INT32), intent(in) :: tv1d
        logical                    :: is_contained
        is_contained=((tv1d>=self%lims(dm,1)).and.(tv1d<=self%lims(dm,2)))
        return
    end function

    pure function PBC_contains_tv3d(self,tv3d) result(is_contained)
    ! Check if tv1d is a valid point in dimension dm
        class(PBC),     intent(in) :: self
        integer(INT32), intent(in) :: tv3d(3)
        logical                    :: is_contained
        is_contained=all((tv3d>=self%lims(:,1)).and.(tv3d<=self%lims(:,2)))
        return
    end function

    integer(INT32) function pbc_max_3dshell(self) result(mx)
        class(PBC),intent(in) :: self
        integer(INT32),parameter :: inf=HUGE(1_INT32)
        if (any(abs(self%lims)==inf)) then
            mx=inf
        else
            mx=sum(maxval(abs(self%lims),dim=2))
        end if
        return
    end function
end module
