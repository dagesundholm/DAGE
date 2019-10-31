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
!> @file xmatrix.F90
!! Matrix operations

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

!> Matrix operations
!! 
!! Generic interface for handling matrices. The multiplication routines
!! use the BLAS95 or the BLAS library if available during compilation
!! time.
module xmatrix_m
    use ISO_FORTRAN_ENV
    use Globals_m
#ifdef HAVE_BLAS95
    use blas95
#endif
    implicit none
    private

    !> Generic multiplication function.
    !!
    !! Handles matrix multiplication, matrix-vector multiplication and
    !! vector-matrix multiplication. Can use BLAS or BLAS95 as backend.
    interface xmatmul
        module procedure xmatmul_matmat
        module procedure xmatmul_matvec
        module procedure xmatmul_vecmat
        module procedure xmatmul_complex_matvec
        module procedure xmatmul_complex_vecmat
    end interface

    public :: xmatmul
    public :: matrix_inverse
    public :: matrix_eigensolver
    public :: dot3_product
    public :: matrix_generalized_eigensolver
    public :: sort_genev

contains

    !> Matrix multiplication: Matrix times a matrix.
    !!
    !! @param a Any rank-two array
    !! @param b Any rank-two array
    !!
    !! @returns The matrix product of `a` and `b` with shape
    !! `[size(a,1), size(b,2)]`.
    function xmatmul_matmat(a, b) result(c)
        real(REAL64), dimension(:,:), intent(in) :: a,b
        real(REAL64), dimension(size(a,1), size(b,2)) :: c

#ifdef HAVE_BLAS95
        ! We should be really, really careful here! To be safe, we should 
        ! allocate a vector of size n*n, and reshape, since there is no guarantee that
        ! the memory is contiguous (especially if slicing is used)
        c=0.0_REAL64
        call gemm(a,b,c)
#elif HAVE_BLAS
        integer(INT32) :: M, N, K
        c=0.0_REAL64
        M=size(a,1)
        N=size(b,2)
        K=size(a,2)
        call DGEMM('n','n',M,N,K,1.d0,A,M,B,K,0.d0,C,M)
#else
        c = 0.0_REAL64
        c = matmul(a,b)
#endif
    end function

    !> Matrix multplication: Matrix times a vector.
    !!
    !! @param a Any rank-two array
    !! @param b Any rank-one array (vector)
    !!
    !! @returns The vector @f$ \vec{c}=\mathbf{A}\vec{b} @f$.
    function xmatmul_matvec(a,b) result(c)
        real(REAL64), dimension(:,:), intent(in) :: a
        real(REAL64), dimension(:), intent(in) :: b
        real(REAL64), dimension(size(a,1)) :: c

        ! NOTE: A possible bug in pgf90 causes problems so that `c` needs to be
        ! allocated with allocate(c(size(a,1)))

#ifdef HAVE_BLAS95
        !
        ! We should be really, really careful here! To be safe, we should 
        ! allocate a vector of size n*n, and reshape, since there is no guarantee that
        ! the memory is contiguous (especially if slicing is used)
        !
        c=0.0_REAL64
        call gemm(a,b,c)
#elif HAVE_BLAS
        integer(INT32) :: M,N

        M=size(a,1)
        N=size(a,2)
        c=0.0_REAL64
        call DGEMV('n',M,N,1.d0,A,M,B,1,0.d0,C,1)
#else
        c = 0.0_REAL64
        c = matmul(a,b)
#endif
    end function

    !> Matrix multplication: Matrix times a vector.
    !!
    !! @param a Any rank-two array
    !! @param b Any rank-one array (vector)
    !!
    !! @returns The vector @f$ \vec{c}=\mathbf{A}\vec{b} @f$.
    function xmatmul_complex_matvec(a,b) result(c)
        complex*16,   dimension(:,:), intent(in) :: a
        complex*16,   dimension(:),   intent(in) :: b
        complex*16,   dimension(size(a,1))       :: c

        ! NOTE: A possible bug in pgf90 causes problems so that `c` needs to be
        ! allocated with allocate(c(size(a,1)))

#ifdef HAVE_BLAS95
        !
        ! We should be really, really careful here! To be safe, we should 
        ! allocate a vector of size n*n, and reshape, since there is no guarantee that
        ! the memory is contiguous (especially if slicing is used)
        !
        c=0.0
        call zgemm(a,b,c)
#elif HAVE_BLAS
        integer(INT32) :: M,N

        M=size(a,1)
        N=size(a,2)
        c=0.0
        call ZGEMV('n',M,N,1.d0,A,M,B,1,0.d0,C,1)
#else
        c = 0.0_REAL64
        c = matmul(a,b)
#endif
    end function

    !> Matrix multiplication: Vector times a matrix.
    !!
    !! @param a Any rank-one array (vector)
    !! @param b Any rank-two array
    !!
    !! @returns The vector @f$ \vec{c}=\vec{a}\mathbf{B}@f$.
    function xmatmul_vecmat(a,b) result(c)
        real(REAL64), dimension(:), intent(in) :: a
        real(REAL64), dimension(:,:), intent(in) :: b
        real(REAL64), dimension(size(b,2)) :: c

        ! NOTE: A possible bug in pgf90 causes problems so that `c` needs to be
        ! allocated with allocate(c(size(b,2)))

#ifdef HAVE_BLAS95
        !
        ! We should be really, really careful here! To be safe, we should 
        ! allocate a vector of size n*n, and reshape, since there is no guarantee that
        ! the memory is contiguous (especially if slicing is used)
        !
        c=0.0_REAL64
        call gemm(a,b,c)
#elif HAVE_BLAS
        integer(INT32) :: M,N

        M=size(b,1)
        N=size(b,2)
        c=0.0_REAL64
        call DGEMV('t',M,N,1.d0,B,M,A,1,0.d0,C,1)
#else
        c = 0.0_REAL64
        c=matmul(a,b)
#endif
    end function

    !> Matrix multiplication: Vector times a matrix.
    !!
    !! @param a Any rank-one array (vector)
    !! @param b Any rank-two array
    !!
    !! @returns The vector @f$ \vec{c}=\vec{a}\mathbf{B}@f$.
    function xmatmul_complex_vecmat(a1,b) result(c)
        real(REAL64), dimension(:),   intent(in) :: a1
        complex*16,   dimension(:,:), intent(in) :: b
        complex*16,   dimension(size(b,2))       :: c
        complex*16                               :: a(size(a1))
        complex*16                               :: alpha, beta

        

        ! NOTE: A possible bug in pgf90 causes problems so that `c` needs to be
        ! allocated with allocate(c(size(b,2)))

#ifdef HAVE_BLAS95
        !
        ! We should be really, really careful here! To be safe, we should 
        ! allocate a vector of size n*n, and reshape, since there is no guarantee that
        ! the memory is contiguous (especially if slicing is used)
        !
        a = a1

        c=0.0
        call gemm(a,b,c)
#elif HAVE_BLAS
        integer(INT32) :: M,N

        a = a1
        alpha = complex(1.0d0, 0.0d0)
        beta = complex(0.0d0, 0.0d0)
        M=size(b,1)
        N=size(b,2)
        c=0.0
        call ZGEMV('t',M,N,alpha,B,M,A,1,beta,C,1)
#else
        c = 0.0
        c=matmul(a,b)
#endif
    end function

    !> Inner product of two arrays of rank three.
    function dot3_product(x, y) result(val)
        real(REAL64), dimension(:,:,:)  :: x, y
        real(REAL64)                    :: val

        val=sum(x*y)
    end function dot3_product

    !> Inverts a matrix (requires LAPACK)
    !!
    !! @param a Any rank-two array
    !!
    !! @returns The matrix @f$\mathbf{A}^{-1}: \mathbf{A}^{-1}\mathbf{A} = \mathbf{I} @f$. 
    function matrix_inverse(a) result(b)
        implicit none
        real(REAL64), intent(in) :: a(:,:)
        real(REAL64)             :: b(size(a,1), size(a,2))

#ifdef HAVE_LAPACK
        integer(INT32)           :: n
        integer                  :: ipiv(size(a,1)), info
        real(REAL64)             :: wrk(size(a,1)**2)

        n=size(a,1)
        b=a
        call dgetrf(n,n,b,n,ipiv,info)
        call dgetri(n,b,n,ipiv,wrk,n*n,info)
#else
        call perror('NO LAPACK FOUND')
        stop
#endif
    end function

    !> Eigensolver for real symmetric matrices. Only the lower triangular matrix
    !! is required.
    subroutine matrix_eigensolver(a,e,u)
        real(REAL64), intent(in)    :: a(:,:)
        real(REAL64), intent(inout) :: e(:)
        real(REAL64), intent(inout) :: u(:,:)

        real(REAL64), allocatable :: wrk(:)
        integer                   :: n
        integer                   :: info

        u=a
        n=size(a,1)
        allocate(wrk((n+2)*n))

        n=size(a,1)
        if(size(a,1)/=size(a,2)) then
            call perror("Cannot diagonalize non-square matrices...")
            stop
        end if
!#ifdef HAVE_BLAS95
        call DSYEV("V", "U", n, u, n, e, wrk, size(wrk), info )
!#endif
        if(info<0) then
            write(ppbuf,'("DSYEV: Error in argument #",i1)') -info
            call perror(ppbuf)
            stop
        else if(info>0) then
            call perror("DSYEV: Eigensolver did not converge.")
            stop
        end if
    end subroutine

    !> Solver for the generalized eigenvalue problem FC = SCe
    subroutine matrix_generalized_eigensolver(F,S,C,e)
        real(REAL64), intent(in)      :: F(:,:),S(:,:)
        character(len=1),parameter    :: JOBVL='N', JOBVR='V'
        integer                       :: N, LDA, LDB, LDVL, LDVR, LWORK, INFO
        integer, allocatable          :: t2(:)
        real(REAL64), allocatable     :: alphar(:),alphai(:),beta(:), temp(:)
        real(REAL64), allocatable, intent(inout) :: e(:)
        real(REAL64), allocatable     :: VL(:,:), WORK_ev(:)
        real(REAL64), allocatable, intent(inout) :: C(:,:)
        real(REAL64), allocatable     :: Ftemp(:,:), Stemp(:,:)


    ! allocating stuff
        N=size(F(:,1))
        LDA=N
        LDB=N
        LDVL=N
        LDVR=N
        LWORK=80*N

        allocate(WORK_ev(LWORK))
        allocate(alphar(N))
        allocate(alphai(N))
        allocate(beta(N))
        allocate(temp(N))
        allocate(t2(N))
        allocate(VL(N,N))

    ! not sure if F and S need to be conserved
        Ftemp=F
        Stemp=S


!#ifdef HAVE_BLAS95
        ! calling dggev
        call dggev(JOBVL, JOBVR, N, Ftemp, LDA, Stemp, LDB, alphar, alphai, &
            beta, VL,  LDVL, C, LDVR, WORK_ev, LWORK, INFO)
!#endif
        ! the eigenvalues go to e
        e=alphar/beta

        t2=minloc(e,rank(e))
        write(*,*) 'minloc: ',t2, 'size',shape(t2)

    end subroutine


    !> subroutine for sorting coefficients C and eigenvalues e
    recursive subroutine sort_genev(C,e)
        real(REAL64), allocatable, intent(inout) :: C(:,:), e(:)
        real(REAL64),allocatable                 :: temp1(:), e1
        integer,allocatable                                  :: i1(:)
        real(REAL64),allocatable                 :: Ctemp(:,:),etemp(:)
        if(size(C,2)>1) then
    ! swap eigenvalue to left
            i1=minloc(e)
            e1=e(1)
            e(1)=e(i1(1))
            e(i1)=e1
    ! swap corresponding coefficients
            temp1=C(:,1)
            C(:,1)=C(:,i1(1))
            C(:,i1(1))=temp1
            Ctemp=C(:,2:size(e))
            etemp=e(2:size(e))
            call sort_genev(Ctemp,etemp)
            C(:,2:size(e))=Ctemp
            e(2:size(e))=etemp
        endif
    end subroutine




end module
