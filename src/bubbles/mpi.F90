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
!> @file mpi.F90
!! Convenience routines for MPI runs

!> @brief Convenience routines for OpenMPI backend
!!
!! Overrides MPI subroutines to do some stuff easier and allow
!! compatibility with non-MPI installations.
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

!
! DATA TYPES:
!	xmpi_scatter_t:	Defines a vector to be scattered or sent or stuff
!
!			Variables contained:
!			*********************************************************************
!			*	Name		* Type		*	Description                        	*
!			*********************************************************************
!			*	ndim		* integer	*	Number of dimensions of the vector	*
!			*				*			*	to be scattered. It can be 1,2,or 3 *
!			*	chunks		* integer	*	
!			*				*			*	
!			*	ifrom		* integer	*	
!			*				*			*	
!			*	ito			* integer	*	
!			*				*			*	
!			*	send_d[X]	* real(REAL64)	*	Pointer to the ndim-dimensional		*
!			*				*			*	array to be scattered.				*
!			*	recv_d[X]	* real(REAL64)	*	Pointer to the ndim-dimensional		*
!			*				*			*	to receive send_d.					*
!			*********************************************************************
!
! HOW TO USE THIS MODULE
! Example: scattering an array A(17,8)
! 
!	 real(REAL64) :: A(17,8)
!	 real(REAL64),pointer :: r_buffer(:,:)
!	 type(xmpi_scatter_t) :: buffer
!	
!	 call new_scatter_vec(buffer,A)
!	 call xmpi_scatter(buffer,r_buffer)
!	 call del_scatter_vec(buffer)
!
! r_buffer MUST NOT BE ALLOCATED OR ASSIGNED
!
! SUBROUTINES:
! new_scatter_vec(self,send)
! del_scatter_vec(self)
! xmpi_scatter(self, recv, ierr)
! xmpi_sum(self, recv, ierr
! xmpi_sum_ho(self, recv, ierr)
! xmpi_allsum(self, recv, ierr) <= Only for 1d arrays!!!!!!! self is NOT xmpi_scatter_t type
! xmpi_bcast(self, ierr)
! xmpi_send(work, who, tag)
! xmpi_recv(work, who, tag)
! gen_mpi_fname(str, fname) 

module mpi_m
    use globals_m
    use ISO_FORTRAN_ENV
#ifdef HAVE_MPI
    use mpi
#endif
    implicit none

    public start_mpi, stop_mpi, get_mpi_rank, gen_mpi_fname
#ifdef HAVE_MPI
    !include 'mpif.h'
    integer(INT32), public, parameter :: mpi_compiled=1
#else
    integer(INT32), public, parameter :: mpi_compiled=0
#endif

    integer(INT32), private :: ierr
    integer(INT32), private :: rank, world_ncpu=-1
    character(BUFLEN), private :: sys

    integer(INT32), private, parameter :: SCATTER_D1_TAG=1001
    integer(INT32), private, parameter :: SCATTER_D2_TAG=1002
    integer(INT32), private, parameter :: SCATTER_D3_TAG=1003

    type xmpi_scatter_t
        private
        integer(INT32) :: ndim
        integer(INT32), dimension(:), pointer :: chunks
        integer(INT32), dimension(:), pointer :: ifrom, ito
        real(REAL64), dimension(:), pointer :: send_d1, recv_d1
        real(REAL64), dimension(:,:), pointer :: send_d2, recv_d2
        real(REAL64), dimension(:,:,:), pointer :: send_d3, recv_d3
    end type

    interface new_scatter_vec
        module procedure init_scatter_workvec_d1
        module procedure init_scatter_workvec_d2
        module procedure init_scatter_workvec_d3
    end interface

    interface del_scatter_vec
        module procedure del_scatter_workvec
    end interface

    interface xmpi_scatter
        module procedure xmpi_scatter_d1
        module procedure xmpi_scatter_d2
        module procedure xmpi_scatter_d3
    end interface

    interface xmpi_sum
        module procedure xmpi_sum_d1
        module procedure xmpi_sum_d2
        module procedure xmpi_sum_d3
    end interface

    interface xmpi_sum_ho
        module procedure xmpi_sum_ho2
        module procedure xmpi_sum_ho3
        module procedure xmpi_sum_ho3_comm
    end interface

    interface xmpi_allsum
        module procedure xmpi_allsum_d1
    end interface

    interface xmpi_bcast
        module procedure xmpi_bcast_d1
        module procedure xmpi_bcast_d2
        module procedure xmpi_bcast_d3
        module procedure xmpi_bcast_integer
    end interface

    interface xmpi_send
        module procedure xmpi_send_d1
        module procedure xmpi_send_d2
        module procedure xmpi_send_d3
    end interface

    interface xmpi_recv
        module procedure xmpi_recv_d1
        module procedure xmpi_recv_d2
        module procedure xmpi_recv_d3
    end interface

contains
    function start_mpi() result(mpirank)
        integer(INT32) :: mpirank
        integer(INT32) :: hostnm
        
        if (mpi_compiled == 0) then
            print*,'Cool, so mpi did not compile'
            rank=-1
            mpirank=-1
            return
        end if
#ifdef HAVE_MPI
!        ierr=hostnm(sys)
!        call ppnote('Initializing MPI on ' // trim(sys))
!        call nl
        if (mpirun_p) then
            call mpi_init(ierr)
            call mpi_comm_rank(MPI_COMM_WORLD, rank, ierr)
            if (rank == 0) then
                master_p=.true.
            end if
        end if
        mpirank=rank
#endif
    end function
        
    subroutine setup_new_comm_split(color,key,newcomm)
        !Color is the id of the new group
        !key is the rank in newcomm
        integer(INT32),intent(in) :: color, key
        integer(INT32) :: newcomm
        integer(INT32) :: ierr
#ifdef HAVE_MPI
        if ( mpirun_p ) then
            call MPI_COMM_SPLIT(MPI_COMM_WORLD, color, key, newcomm, ierr)
            if(ierr/=0) stop 'Problem creating new communicator'
        else
            newcomm=MPI_COMM_WORLD
        end if
#endif
    end subroutine

    subroutine stop_mpi()
#ifdef HAVE_MPI
        if ( mpirun_p ) then
            call mpi_finalize(ierr)
        end if
#endif
    end subroutine

    function xmpi_ncpu() result(n)
        integer(INT32) :: n
#ifdef HAVE_MPI		
        if (mpirun_p) then
            if (world_ncpu < 0) then
                call mpi_comm_size(MPI_COMM_WORLD, world_ncpu, ierr)
                !OK, I don't see the point of the following line
!                world_ncpu=world_ncpu
            end if
            n=world_ncpu
        else
            n=1
        end if
#else
        n=1
#endif

    end function

    subroutine gen_scatter_workvec(self, nsend)
        type(xmpi_scatter_t) :: self
        integer(INT32), intent(in) :: nsend

        integer(INT32) :: i, nodes, ssz, leftovers, nunit, fidx, tidx

        nodes=xmpi_ncpu()

        allocate(self%chunks(nodes))
        allocate(self%ifrom(nodes))
        allocate(self%ito(nodes))

        leftovers=mod(nsend,nodes)
        nunit=(nsend-leftovers)/nodes
        self%chunks=nunit
!        print *, 'leftovers=', leftovers
!        print *, 'nunit=', nunit

        ! the +1 is to shift work away from the master which has to do all
        ! the communication
        do i=1,leftovers
            self%chunks(i+1)=self%chunks(i+1)+1
        end do
        ! construct scatter indeces
        fidx=0; tidx=0
        do i=1,nodes
            fidx=tidx+1
            tidx=fidx+self%chunks(i)-1
            self%ifrom(i)=fidx
            self%ito(i)=tidx
        end do
!        print *, '@ifrom', self%ifrom, '@'
!        print *, '@ito  ', self%ito, '@'
    end subroutine

    subroutine init_scatter_workvec_d1(self, send)
        type(xmpi_scatter_t) :: self
        real(REAL64), dimension(:), intent(in), target :: send

        integer(INT32) :: chunksz
        
        self%ndim=1
        self%send_d1=>send
        chunksz=size(send)
        call gen_scatter_workvec(self, chunksz)

        chunksz=self%chunks(rank+1) ! rank starts from 0...
        allocate(self%recv_d1(chunksz))
!        print *, 'sefl%chunks=', self%chunks,'@'
!        print *, '@allocated recv', size(self%recv_d1), rank, '@'

    end subroutine

    subroutine init_scatter_workvec_d2(self, send)
        type(xmpi_scatter_t) :: self
        real(REAL64), dimension(:,:), intent(in), target :: send

        integer(INT32) :: chunksz, i

        self%ndim=2
        self%send_d2=>send
        chunksz=size(send(1,:))
        call gen_scatter_workvec(self, chunksz)

        i=size(send(:,1))

        chunksz=self%chunks(rank+1) ! rank starts from 0...
        allocate(self%recv_d2(i,chunksz))
!        print *, 'sefl%chunks=', self%chunks,'@'
!        print *, 'allocated recv', shape(self%recv_d2), rank, '£'
    end subroutine

    subroutine init_scatter_workvec_d3(self, send)
        type(xmpi_scatter_t) :: self
        real(REAL64), dimension(:,:,:), intent(in), target :: send

        integer(INT32) :: chunksz, i,j

        self%ndim=3
        self%send_d3=>send
        chunksz=size(send(1,1,:))
        call gen_scatter_workvec(self, chunksz)

        i=size(send(:,1,1))
        j=size(send(1,:,1))

        chunksz=self%chunks(rank+1) ! rank starts from 0...
        allocate(self%recv_d3(i,j,chunksz))
!        print *, 'sefl%chunks=', self%chunks,'@'
!        print *, 'allocated recv', shape(self%recv_d3), rank, '£'
    end subroutine

    subroutine del_scatter_workvec(self)
        type(xmpi_scatter_t) :: self

        deallocate(self%chunks)
        deallocate(self%ifrom)
        deallocate(self%ito)
        select case(self%ndim)
            case(1)
                deallocate(self%recv_d1)
            case(2)
                deallocate(self%recv_d2)
            case(3)
                deallocate(self%recv_d3)
        end select

    end subroutine

    subroutine xmpi_scatter_d1(self, recv, ierr)
        type(xmpi_scatter_t) :: self
        real(REAL64), dimension(:), pointer :: recv
        integer(INT32), optional :: ierr
        
        integer(INT32) :: i, ifrom, ito

#ifdef HAVE_MPI		
        if (master_p) then
            ifrom=self%ifrom(1)
            ito=self%ito(1)
            self%recv_d1=self%send_d1(ifrom:ito)
            do i=1,world_ncpu-1
                ifrom=self%ifrom(i+1)
                ito=self%ito(i+1)
                if (ito-ifrom < 0) cycle
                call xmpi_send(self%send_d1(ifrom:ito), i, SCATTER_D1_TAG)
            end do
        else
            call xmpi_recv(self%recv_d1, 0, SCATTER_D1_TAG)
        end if
        recv=>self%recv_d1

#endif
        if (present(ierr)) then
            ierr=0
        end if

    end subroutine

    ! send and recv buffers must NOT be the same!!!
    subroutine xmpi_scatter_d2(self, recv, ierr)
        type(xmpi_scatter_t) :: self
        real(REAL64), dimension(:,:), pointer :: recv
        integer(INT32), optional :: ierr
        
        integer(INT32) :: i, ifrom, ito

#ifdef HAVE_MPI		

        if (master_p) then
            ifrom=self%ifrom(1)
            ito=self%ito(1)
            self%recv_d2=self%send_d2(:,ifrom:ito)
            do i=1,world_ncpu-1
                ifrom=self%ifrom(i+1)
                ito=self%ito(i+1)
                if (ito-ifrom < 0) cycle
                call xmpi_send(self%send_d2(:,ifrom:ito), i, SCATTER_D2_TAG)
            end do
        else
            call xmpi_recv(self%recv_d2, 0, SCATTER_D2_TAG)
        end if
        recv=>self%recv_d2

#endif
        if (present(ierr)) then
            ierr=0
        end if

    end subroutine

    subroutine xmpi_scatter_d3(self, recv, ierr)
        type(xmpi_scatter_t) :: self
        real(REAL64), dimension(:,:,:), pointer :: recv
        integer(INT32), optional :: ierr
        
        integer(INT32) :: i, ifrom, ito

#ifdef HAVE_MPI		
        if (master_p) then
            ifrom=self%ifrom(1)
            ito=self%ito(1)
            self%recv_d2=self%send_d2(:,ifrom:ito)
            do i=1,world_ncpu-1
                ifrom=self%ifrom(i+1)
                ito=self%ito(i+1)
                if (ito-ifrom < 0) cycle
                call xmpi_send(self%send_d3(:,:,ifrom:ito), i, SCATTER_D3_TAG)
            end do
        else
            call xmpi_recv(self%recv_d3, 0, SCATTER_D3_TAG)
        end if
        recv=>self%recv_d3

#endif
        if (present(ierr)) then
            ierr=0
        end if

    end subroutine

    subroutine xmpi_sum_d1(send, recv, ierr)
        real(REAL64), dimension(:), intent(in) :: send
        real(REAL64), dimension(:), intent(out) :: recv
        integer(INT32), optional :: ierr

        integer(INT32) :: error, ksz
        ksz=size(send)
        recv(1)=0.d0 !silly
#ifdef HAVE_MPI		
        call mpi_reduce(send, recv, ksz, & 
            MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, error)
#endif
        if (present(ierr)) ierr=error
    end subroutine
    
    subroutine xmpi_sum_d2(send, recv, ierr)
        real(REAL64), dimension(:,:), intent(in) :: send
        real(REAL64), dimension(:,:), intent(out) :: recv
        integer(INT32), optional :: ierr

        integer(INT32) :: j,error, jsz, ksz
        jsz=size(send(1,:))
        ksz=size(send(:,1))
        recv(1,1)=0.d0 !silly
#ifdef HAVE_MPI		
        do j=1,jsz
            call mpi_reduce(send(:,j), recv(:,j), ksz, & 
                MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, error)
        end do
#endif
        if (present(ierr)) ierr=error
    end subroutine

    subroutine xmpi_sum_d3(send, recv, ierr)
        real(REAL64), dimension(:,:,:), intent(in) :: send
        real(REAL64), dimension(:,:,:), intent(out) :: recv
        integer(INT32), optional :: ierr
        integer(INT32) :: i,j,error, isz, jsz, ksz

        isz=size(send(1,1,:))
        jsz=size(send(1,:,1))
        ksz=size(send(:,1,1))
        recv(1,1,1)=0.d0 !silly
#ifdef HAVE_MPI		
        do i=1,isz
            do j=1,jsz
                call mpi_reduce(send(:,j,i), recv(:,j,i), ksz, & 
                    MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, error)
            end do
        end do
#endif

        if (present(ierr)) ierr=error
    end subroutine

    subroutine xmpi_sum_x3(send, recv, ierr)
        real(REAL64), dimension(:,:,:), intent(in) :: send
        real(REAL64), dimension(:,:,:), intent(out) :: recv
        integer(INT32), optional :: ierr
        
        integer(INT32) :: i,j,error, isz, jsz, ksz
        real(REAL64), dimension(:,:), allocatable :: tmp1, tmp2

        isz=size(send(:,1,1))
        jsz=size(send(1,:,1))
        ksz=size(send(1,1,:))
        recv(1,1,1)=0.d0 !silly
#ifdef HAVE_MPI		
        allocate(tmp1(isz,jsz))
        allocate(tmp2(isz,jsz))
        do i=1,ksz
                tmp1=send(:,:,i)
                call mpi_reduce(tmp1, tmp2, isz*jsz, & 
                    MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, error)
                recv(:,:,i)=tmp2
        end do
        deallocate(tmp1, tmp2)
#endif


        if (present(ierr)) ierr=error
    end subroutine

    subroutine xmpi_sum_ho3_comm(comm, send, recv, ierr)
        real(REAL64), dimension(:,:,:), intent(in) :: send
        real(REAL64), dimension(:,:,:), intent(out) :: recv
        integer(INT32), intent(in) :: comm
        integer(INT32), optional :: ierr
        integer(INT32) :: i,j,error, isz, jsz, ksz

        isz=size(send,1)
        jsz=size(send,2)
        ksz=size(send,3)
        recv(1,1,1)=0.d0 !silly
#ifdef HAVE_MPI		
        call mpi_reduce(send, recv, isz*jsz*ksz, & 
            MPI_DOUBLE_PRECISION, MPI_SUM, 0, comm, error)
#endif

        if (present(ierr)) ierr=error
    end subroutine

    subroutine xmpi_sum_ho3(send, recv, ierr)
        real(REAL64), dimension(:,:,:), intent(in) :: send
        real(REAL64), dimension(:,:,:), intent(out) :: recv
        integer(INT32), optional :: ierr
        integer(INT32) :: i,j,error, isz, jsz, ksz

        isz=size(send(1,1,:))
        jsz=size(send(1,:,1))
        ksz=size(send(:,1,1))
        recv(1,1,1)=0.d0 !silly
#ifdef HAVE_MPI		
        call mpi_reduce(send, recv, isz*jsz*ksz, & 
            MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, error)
#endif

        if (present(ierr)) ierr=error
    end subroutine

    subroutine xmpi_sum_ho2(send, recv, ierr)
        real(REAL64), dimension(:,:), intent(in) :: send
        real(REAL64), dimension(:,:), intent(out) :: recv
        integer(INT32), optional :: ierr
        integer(INT32) :: i,j,error, isz, jsz, ksz

        isz=size(send(:,1))
        jsz=size(send(1,:))
        recv(1,1)=0.d0 !silly
#ifdef HAVE_MPI		
        call mpi_reduce(send, recv, isz*jsz, & 
            MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, error)
#endif

        if (present(ierr)) ierr=error
    end subroutine

    subroutine xmpi_bcast_d1(send,ierr)
        real(REAL64), dimension(:), pointer :: send
        integer(INT32), optional :: ierr
        
        integer(INT32) :: ie, ksz
        ksz=size(send)

#ifdef HAVE_MPI		
        call mpi_bcast(send,ksz,MPI_DOUBLE_PRECISION,0, & 
                MPI_COMM_WORLD, ie)

#endif
        if (present(ierr)) then
            ierr=ie
        end if
    end subroutine

    subroutine xmpi_bcast_d2(send,ierr)
        real(REAL64), dimension(:,:), pointer :: send
        integer(INT32), optional :: ierr
        
        integer(INT32) :: j,ie, jsz, ksz
        jsz=size(send(1,:))
        ksz=size(send(:,1))

#ifdef HAVE_MPI		
        do j=1,jsz
            call mpi_bcast(send(:,j),ksz,MPI_DOUBLE_PRECISION,0, & 
                MPI_COMM_WORLD, ie)
        end do

#endif
        if (present(ierr)) then
            ierr=ie
        end if
    end subroutine

    subroutine xmpi_bcast_d3(send,ierr)
        real(REAL64), dimension(:,:,:) :: send
        integer(INT32), optional :: ierr
        
        integer(INT32) :: i,j,ie, isz, jsz, ksz
        ie=0

#ifdef HAVE_MPI		
        isz=size(send(1,1,:))
        jsz=size(send(1,:,1))
        ksz=size(send(:,1,1))
        do i=1,isz
            do j=1,jsz
                call mpi_bcast(send(:,j,i),ksz,MPI_DOUBLE_PRECISION,0, & 
                    MPI_COMM_WORLD, ie)
            end do
        end do

#endif
        if (present(ierr)) then
            ierr=ie
        end if
    end subroutine

    subroutine xmpi_bcast_integer(send,ierr)
        integer(INT32) :: send
        integer(INT32), optional :: ierr
        
        integer(INT32) :: ie
        ie=0

#ifdef HAVE_MPI		
        call mpi_bcast(send,1,MPI_INTEGER,0, & 
                    MPI_COMM_WORLD, ie)
#endif
        if (present(ierr)) then
            ierr=ie
        end if
    end subroutine

    function get_mpi_rank() result(r)
        integer(INT32) :: r
        r=rank
    end function

    subroutine xmpi_send_d1(work, who, tag)
        real(8), dimension(:), intent(in) :: work
        integer(INT32), intent(in) :: who, tag

        integer(INT32) :: n

#ifdef HAVE_MPI
        n=size(work)
        
        call mpi_ssend(work(:),n,MPI_DOUBLE_PRECISION,who,tag,&
            MPI_COMM_WORLD, ierr)
#endif
    end subroutine

    subroutine xmpi_send_d2(work, who, tag)
        real(8), dimension(:,:), intent(in) :: work
        integer(INT32), intent(in) :: who, tag

        integer(INT32) :: n, nsend, k

#ifdef HAVE_MPI
        n=size(work(:,1))
        nsend=size(work(1,:))
        
        do k=1,nsend
            call mpi_ssend(work(:,k),n,MPI_DOUBLE_PRECISION,who,tag,&
                MPI_COMM_WORLD, ierr)
        end do
#endif
    end subroutine

    subroutine xmpi_send_d3(work, who, tag)
        real(8), dimension(:,:,:), intent(in) :: work
        integer(INT32), intent(in) :: who, tag

        integer(INT32) :: n, nsend1, nsend2, k, l

#ifdef HAVE_MPI
        n=size(work(:,1,1))
        nsend1=size(work(1,:,1))
        nsend2=size(work(1,1,:))

        do k=1,nsend1
            do l=1,nsend2
                call mpi_ssend(work(:,k,l),n,MPI_DOUBLE_PRECISION,who,tag,&
                MPI_COMM_WORLD, ierr)
            end do
        end do
#endif
    end subroutine

    subroutine xmpi_recv_d1(work, who, tag)
        real(8), dimension(:), intent(in) :: work
        integer(INT32), intent(in) :: who, tag

#ifdef HAVE_MPI
        integer(INT32) :: n
        integer(INT32), dimension(MPI_STATUS_SIZE) :: stat

        n=size(work)
        
        call mpi_recv(work(:),n,MPI_DOUBLE_PRECISION,who,tag, &
            MPI_COMM_WORLD, stat, ierr)
#endif
    end subroutine

    subroutine xmpi_recv_d2(work, who, tag)
        real(8), dimension(:,:), intent(in) :: work
        integer(INT32), intent(in) :: who, tag

#ifdef HAVE_MPI
        integer(INT32) :: n, nrecv, k
        integer(INT32), dimension(MPI_STATUS_SIZE) :: stat

        n=size(work(:,1))
        nrecv=size(work(1,:))
        
        do k=1,nrecv
            call mpi_recv(work(:,k),n,MPI_DOUBLE_PRECISION,who,tag,&
                MPI_COMM_WORLD, stat, ierr)
        end do
#endif
    end subroutine

    subroutine xmpi_recv_d3(work, who, tag)
        real(8), dimension(:,:,:), intent(in) :: work
        integer(INT32), intent(in) :: who, tag


#ifdef HAVE_MPI
        integer(INT32) :: n, nrecv1, nrecv2, k, l
        integer(INT32), dimension(MPI_STATUS_SIZE) :: stat

        n=size(work(:,1,1))
        nrecv1=size(work(1,:,1))
        nrecv2=size(work(1,1,:))

        do k=1,nrecv1
            do l=1,nrecv2
                call mpi_recv(work(:,k,l),n,MPI_DOUBLE_PRECISION,who,tag,&
                MPI_COMM_WORLD, stat, ierr)
            end do
        end do
#endif
    end subroutine

    subroutine xmpi_allsum_d1(send, recv, ierr)
        real(REAL64), dimension(:), intent(in) :: send
        real(REAL64), dimension(:), intent(out) :: recv
        integer(INT32), optional :: ierr

        integer(INT32) :: error, ksz
        ksz=size(send)
        recv(1)=0.d0 !silly
#ifdef HAVE_MPI		
        call mpi_allreduce(send, recv, ksz, & 
            MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, error)
#endif
        if (present(ierr)) ierr=error
    end subroutine

    subroutine gen_mpi_fname(str, fname) 
        character(*), intent(in) :: str
        character(*), intent(out) :: fname

        if (mpirun_p) then
            write(fname, "(a,'.',i3.3)") trim(str), rank
        else
            fname=str
        end if
    end subroutine

    function x_time()
        real(REAL32) :: x_time
#ifdef HAVE_MPI		
        x_time=mpi_wtime()
#else
        call cpu_time(x_time)
#endif
    end function

subroutine setup_new_comm(newcomm,iammember)
      implicit none
      integer  :: ierr
      integer :: oldgroup, newgroup, newcomm, nmembers, j
      integer :: memberlist(world_ncpu)  !The vector cannot be allocated dynamically...
      logical :: iammember, jismember

#ifdef HAVE_MPI
      nmembers=0
      do j=1,world_ncpu
          jismember=iammember
          call mpi_bcast(jismember,1,mpi_logical,j-1,mpi_comm_world,ierr)
          if(.not.jismember) cycle
          nmembers=nmembers+1
          memberlist(nmembers)=j-1
      end do
      call mpi_comm_group(mpi_comm_world,oldgroup,ierr)
      call mpi_group_incl(oldgroup,nmembers,memberlist(1:nmembers),newgroup,ierr)
      call mpi_comm_create(mpi_comm_world,newgroup,newcomm,ierr)
#else
      newcomm=1
#endif

      return
end subroutine

subroutine sum_boundaries(array,rank,numproc)
        real(REAL64), dimension(:,:,:) :: array
        real(REAL64), dimension(:,:),allocatable :: temp
        integer(INT32) :: rank, numproc, ierr, request, i
        integer(INT32) :: dimx,dimy,dimz

#ifdef HAVE_MPI
        integer(INT32), dimension(MPI_STATUS_SIZE) :: stat
        dimx=size(array,1)
        dimy=size(array,2)
        dimz=size(array,3)

        if(rank<numproc-1) then
            allocate(temp(dimy,dimz))
            call mpi_irecv(temp,dimy*dimz,mpi_double_precision,rank+1,SCATTER_D2_TAG,io_comm,request,ierr)
        end if
        if(rank>0) then
            !call mpi_ssend(array(1,:,:),dimy*dimz,mpi_double_precision,rank-1,SCATTER_D2_TAG,io_comm,stat,ierr)
        end if
        if(rank<numproc-1) then
            call mpi_wait(request, stat, ierr)
            array(dimx,:,:)=array(dimx,:,:)+temp
        end if
        if(allocated(temp)) deallocate(temp)
#endif
        return
    end subroutine


    subroutine xmpi_file_open(fname,handle)
        character(len=*) :: fname
        integer(INT32) :: handle,ierr
#ifdef HAVE_MPI
          call mpi_file_open(io_comm,trim(fname),&
             MPI_MODE_WRONLY+MPI_MODE_CREATE,&
             mpi_info_null,handle,ierr)
#endif
    end subroutine

    subroutine xmpi_file_close(handle)
        integer(INT32) :: handle,ierr
#ifdef HAVE_MPI
        call mpi_barrier(io_comm,ierr)
        call mpi_file_close(handle,ierr)
#endif
    end subroutine

    function xmpi_real_size()
        integer(INT32) :: xmpi_real_size,ierr
#ifdef HAVE_MPI
        call mpi_type_size(MPI_REAL,xmpi_real_size,ierr)
#else
        xmpi_real_size=-1
#endif
    end function

    function xmpi_double_size()
        integer(INT32) :: xmpi_double_size,ierr
#ifdef HAVE_MPI
        call mpi_type_size(MPI_DOUBLE_PRECISION,xmpi_double_size,ierr)
#else
        xmpi_double_size=-1
#endif
        
    end function

    subroutine xmpi_setview_and_write_real(handle,buffer,bufsize,disp)
        integer(INT32) :: handle,ierr,bufsize
        integer(INT64) :: disp
        real(REAL32) :: buffer(:)
#ifdef HAVE_MPI
        integer(kind=mpi_offset_kind) :: disp_mpi
        integer(INT32), dimension(MPI_STATUS_SIZE) :: stat
        disp_mpi=disp
        call mpi_file_set_view(handle,disp_mpi,mpi_real, mpi_real,'native',mpi_info_null,ierr)
        call mpi_file_write(handle,buffer,bufsize,mpi_real,stat,ierr)
#endif
    end subroutine

    subroutine xmpi_setview_and_write_double(handle,buffer,bufsize,disp)
        integer(INT32) :: handle,ierr,bufsize
        integer(INT64) :: disp
        real(REAL64) :: buffer(:,:)
#ifdef HAVE_MPI
        integer(kind=mpi_offset_kind) :: disp_mpi
        integer(INT32), dimension(MPI_STATUS_SIZE) :: stat
        disp_mpi=disp
        call mpi_file_set_view(handle,disp_mpi,MPI_DOUBLE_PRECISION, MPI_DOUBLE_PRECISION,'native',mpi_info_null,ierr)
        call mpi_file_write(handle,buffer,bufsize,MPI_DOUBLE_PRECISION,stat,ierr)
#endif
    end subroutine
end module
