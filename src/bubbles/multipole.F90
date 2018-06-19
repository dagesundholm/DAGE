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
module MultiPole_class
    use ISO_FORTRAN_ENV
    use RealSphericalHarmonics_class
    use Harmonic_class, only : lm
    use XMatrix_m
    implicit none
    private

    !> Provides the translation and interaction matrices for the Fast Multipole Method

    type, public :: InteractionMatrix
        private
        integer(INT32) :: lmax = 0
        type(RealRegularSolidHarmonics) :: solid_harmonics
        real(REAL64), allocatable :: I(:)
        real(REAL64), allocatable :: c(:,:)
        integer(INT32), allocatable :: larr(:,:)
        integer(INT32), allocatable :: marr(:,:)
    contains
        procedure, private :: compute_coeffs => InteractionMatrix_compute_coeffs
        procedure :: destroy => InteractionMatrix_destroy
        procedure, private :: InteractionMatrix_eval_many, InteractionMatrix_eval_pick
        generic :: eval => InteractionMatrix_eval_many, InteractionMatrix_eval_pick
        procedure :: apply => InteractionMatrix_apply
    end type

    type, public :: TranslationMatrix
        private
        integer(INT32) :: lmax = 0
        type(RealRegularSolidHarmonics) :: yb 
        real(REAL64), allocatable :: S(:)
        real(REAL64), allocatable :: c(:,:)
        integer(INT32), allocatable :: larr(:,:)
        integer(INT32), allocatable :: marr(:,:)
        logical :: trans = .false.
    contains
        procedure, private :: compute_coeffs => TranslationMatrix_compute_coeffs
        procedure :: eval => TranslationMatrix_eval
        procedure :: apply => TranslationMatrix_apply
        procedure :: destroy => TranslationMatrix_destroy 
        procedure :: transpose => TranslationMatrix_transpose
    end type


    interface TranslationMatrix
        module procedure :: TranslationMatrix_init
    end interface

    interface InteractionMatrix
        module procedure :: InteractionMatrix_init
    end interface

contains

    !!!!!!!!!!!!!!!!!!!!!!
    ! INTERACTION MATRIX !
    !!!!!!!!!!!!!!!!!!!!!!

    function InteractionMatrix_init(lmax) result(new)
        integer(INT32), intent(in) :: lmax
        integer :: N
        type(InteractionMatrix) :: new
        new%lmax = lmax
        new%solid_harmonics =  RealRegularSolidHarmonics(2*lmax)
        N = (lmax+1)**2
        allocate(new%larr(N*(N+1)/2,2), source=-1)
        allocate(new%marr(N*(N+1)/2,2), source=0)
        allocate(new%c(N*(N+1)/2,2), source=0.0d0)
        allocate(new%I((2*lmax+1)**2), source=0.0d0)
        call new%compute_coeffs()
    end function

    subroutine InteractionMatrix_destroy(self)
        class(InteractionMatrix) :: self
        if (allocated(self%I)) deallocate(self%I)
        if (allocated(self%c)) deallocate(self%c)
        if (allocated(self%larr)) deallocate(self%larr)
        if (allocated(self%marr)) deallocate(self%marr)
        call self%solid_harmonics%destroy()
    end subroutine

    subroutine InteractionMatrix_compute_coeffs(self)
        class(InteractionMatrix) :: self
        integer :: i, s, N, lmv(2), l, m, j, k, r
        real(REAL64) :: ij(0:2)
        real(REAL64) :: c

        N = (self%lmax+1)**2
        ij = [1.0d0, 0.5d0, 0.25d0]
        do s=1, N ! column index
            lmv = lm(s)
            j = lmv(1)
            k = lmv(2)
            do i=s, N ! row index
                lmv = lm(i)
                l = lmv(1)
                m = lmv(2)

                ! Common prefactor
                c = 2*(-1)**j*ij(delta(m,0)+delta(k,0))

                r = i+(2*N-s)*(s-1)/2

                if (m >= 0 .and. k >= 0) then
                    ! cc

                    self%larr(r,1) = l+j
                    self%marr(r,1) = abs(m+k)
                    self%c(r,1) = Cct(l+j, m+k)

                    self%larr(r,2) = l+j
                    self%marr(r,2) = abs(m-k)
                    self%c(r,2) = (-1)**k*Cct(l+j, m-k)

                elseif (m >= 0 .and. k < 0) then
                    ! cs

                    self%larr(r,1) = l+j
                    self%marr(r,1) = -abs(m+k)
                    self%c(r,1) = Cst(l+j, m+k)

                    self%larr(r,2) = l+j
                    self%marr(r,2) = -abs(m-k)
                    self%c(r,2) = -(-1)**k*Cst(l+j, m-k)

                    self%c(r,:) = -self%c(r,:)*(-1)**k

                elseif (m < 0 .and. k >= 0) then
                    ! sc

                    self%larr(r,1) = l+j
                    self%marr(r,1) = -abs(m+k)
                    self%c(r,1) = Cst(l+j, m+k)

                    self%larr(r,2) = l+j
                    self%marr(r,2) = -abs(m-k)
                    self%c(r,2) = (-1)**k*Cst(l+j, m-k)

                    self%c(r,:) = -self%c(r,:)*(-1)**m



                elseif (m < 0 .and. k < 0) then
                    ! ss

                    self%larr(r,1) = l+j
                    self%marr(r,1) = abs(m+k)
                    self%c(r,1) = -Cct(l+j, m+k)

                    self%larr(r,2) = l+j
                    self%marr(r,2) = abs(m-k)
                    self%c(r,2) = (-1)**k*Cct(l+j, m-k)

                    self%c(r,:) = self%c(r,:)*(-1)**(m+k)
                endif

                self%c(r,:) =  self%c(r,:)*c

                ! Scaling
                self%c(r,:) = (-1)**m*self%c(r,:)/D(l,m)
                self%c(r,:) = (-1)**k*self%c(r,:)/D(j,k)
            enddo
        enddo

    end subroutine

    !> Construct the interaction matrix of FMM theory in a set of points
    !!
    !! The results is a 3D array of shape ((LMAX+1)**2, (LMAX+1)**2, Npoints)
    !!
    !! The tensor is not defined at the origin (i.e., at the coalescence
    !! point of charged particles)
    function InteractionMatrix_eval_many(self, dR) result(TT)
        class(InteractionMatrix), intent(in) :: self
        !> The points as a matrix of shape (3, Npoints)
        real(REAL64),             intent(in) :: dR(:,:)
        real(REAL64)                    :: TT((self%lmax + 1)**2,(self%lmax + 1)**2,size(dR, 2))

        real(REAL64)                    :: r(size(dR,2)), S(size(dR, 2), (2*self%lmax+1)**2), &
                                           irregular_harmonics(size(dR, 2), (2*self%lmax+1)**2)
        real(REAL64)                    :: w(size(self%c, 1), size(dR, 2))
        !real(REAL64)                    :: dR_tmp(4,size(dR,1))
        integer :: N,M,i,j,k,l,jc,col,lmv(2), Npoints

        Npoints = size(dR, 2)
        N = (self%lmax + 1)**2
        M = N*(N+1)/2

        TT = 0.0d0
        !allocate(w(M), source=0.0d0)
        ! Evaluate irregular harmonics in the points dR
        !dR_tmp(1,:) = dR(:,1)
        !dR_tmp(2,:) = dR(:,2)
        !dR_tmp(3,:) = dR(:,3)
        !dR_tmp(4,:) = r
        r = sqrt(sum(dR**2, 1))
        S = self%solid_harmonics%eval(dR)
        

        forall (k=1 : Npoints)

            ! Spherical harmonics to irregular solid harmonics
            forall (l=0 : 2*self%lmax)
                ! Irregular solid harmonics in terms of regular ones
                ! Note the exponent 2l instead of l.
                ! I_lm = S_lm*r^{-(2l +1 )}
                irregular_harmonics(k, l*l+1 : (l+1)**2) = S(k, l*l+1 : (l+1)**2)/r(k)**(2*l+1)
            end forall

            ! Lower triangular part including diagonal
            w(:, k) = self%c(:, 1) * irregular_harmonics(k, lm_map(self%larr(:,1), self%marr(:,1)))
            w(:, k) = w(:, k) + self%c(:, 2)*irregular_harmonics(k, lm_map(self%larr(:,2), self%marr(:,2)))

            ! Construct the matrix
            forall (col= 1 : N) ! Column index
                !lmv = lm(s)
                !j = lmv(1)
                forall (i= col : N) ! Row index
                    !lmv = lm(i)
                    !l = lmv(1)
                    TT(i,col,k) = w( i+(2*N-col)*(col-1)/2, k)
                    TT(col,i,k) = TT(i,col,k)
                end forall
            end forall

            ! The T matrix is "almost symmetric", satisfying the
            ! relation T_jk,lm = (-1)**(l-j)*T_lm,jk
            forall (l=0 : self%lmax)
                forall (j=l+1 : self%lmax)
                    TT(l*l + 1 : l*l + 1 + 2*l,&
                      j*j + 1 : j*j + 1 + 2*j, k) = (-1)**(l-j)*TT(l*l + 1 : l*l + 1 + 2*l,&
                                                           j*j + 1 : j*j + 1 + 2*j, k)
                end forall
            end forall

        end forall

        !deallocate(w)

    end function

    !> Construct the interaction matrix of FMM theory in a single point dR
    !!
    !! The matrix is not defined at the origin (i.e., at the coalescence of
    !! charged particles)
    function InteractionMatrix_eval_pick(self, dR) result(T)
        class(InteractionMatrix)                :: self
        !> A point (dX, dY, dZ)
        real(REAL64), intent(in)                :: dR(3)

        real(REAL64), allocatable               :: w(:)
        integer                                 :: l, j, lmv(2), N, M, k
        integer                                 :: i, s, jc
        real(REAL64)                            :: r, dR_tmp(4,1)

        real(REAL64), allocatable               :: T(:,:)

        N = (self%lmax + 1)**2
        M = N*(N+1)/2

        allocate(w(M), source=0.0d0)
        allocate(T(N,N), source=0.0d0)

        ! Evaluate irregular solid harmonics
        r = sqrt(sum(dR**2))
        !dR_tmp = reshape([dR(1),dR(2),dR(3),r],[4,1])
        
        self%I = reshape(self%solid_harmonics%eval(reshape(dR, [3, 1])),&
                         shape(self%I))
        k = 0
        do j=0, 2*self%lmax
            self%I(k+1:k+(2*j+1)) = self%I(k+1:k+(2*j+1))/r**(2*j+1)
            k = k+(2*j+1)
        enddo

        ! Lower triangular part including diagonal
        w = self%c(:,1)*self%I(lm_map(self%larr(:,1), self%marr(:,1)))
        w = w + self%c(:,2)*self%I(lm_map(self%larr(:,2), self%marr(:,2)))

        ! Construct the matrix
        do s=1, N ! column index
            lmv = lm(s)
            j = lmv(1)
            do i=s, N ! row index
                lmv = lm(i)
                l = lmv(1)
                jc = i+(2*N-s)*(s-1)/2
                T(i,s) = w(jc)
                T(s,i) = (-1)**(l-j)*T(i,s)
            enddo
        enddo

        deallocate(w)

    end function

    function InteractionMatrix_apply(self, Qvec, dueto, at) result(V)
        class(InteractionMatrix) :: self
        real(REAL64), intent(in) :: Qvec(:), dueto(3), at(3)
        real(REAL64), allocatable :: Tmx(:,:)
        real(REAL64) :: V(size(Qvec))

        if (size(Qvec) > (self%lmax + 1)**2) then
            print *, size(Qvec), (self%lmax + 1)**2
            write(ERROR_UNIT, *) "Multipole vector too big."
            stop
            return
        else
            V = 0.0d0
            Tmx = self%eval(dueto-at)
            V = xmatmul(Tmx, Qvec)
            deallocate(Tmx)
        endif

        ! q(P)V()
    end function

    !!!!!!!!!!!!!!!!!!!!!!
    ! TRANSLATION MATRIX !
    !!!!!!!!!!!!!!!!!!!!!!

    function TranslationMatrix_init(lmax) result(new)
        integer(INT32), intent(in) :: lmax
        integer :: N
        type(TranslationMatrix) :: new

        new%lmax = lmax
        new%yb = RealRegularSolidHarmonics(lmax)
        N = (lmax+1)**2
        allocate(new%larr(N*(N+1)/2,2), source=-1)
        allocate(new%marr(N*(N+1)/2,2), source=0)
        allocate(new%c(N*(N+1)/2,2), source=0.0d0)
        allocate(new%S(N), source=0.0d0)
        call new%compute_coeffs()
    end function

    subroutine TranslationMatrix_destroy(self)
        class(TranslationMatrix) :: self
        if (allocated(self%S)) deallocate(self%S)
        if (allocated(self%c)) deallocate(self%c)
        if (allocated(self%larr)) deallocate(self%larr)
        if (allocated(self%marr)) deallocate(self%marr)
        call self%yb%destroy()
    end subroutine

    !> Compute parameters for the lower diagonal part of the
    !TranslationMatrix
    subroutine TranslationMatrix_compute_coeffs(self) 
        class(TranslationMatrix) :: self
        integer :: N,i,s,l,m,j,k,r,lmv(2)
        real(REAL64) :: ij(0:1) 

        N = (self%lmax+1)**2
        ij = [1.0d0, 0.5d0]
        do s=1, N !column index
            lmv = lm(s)
            j = lmv(1)
            k = lmv(2)
            do i=s, N !row index
                lmv = lm(i)
                l = lmv(1)
                m = lmv(2)

                r = i+(2*N-s)*(s-1)/2

                if (m>=0 .and. k>=0) then
                    ! cc

                    if (-(l-j) <= abs(m-k) .and. abs(m-k) <= (l-j)) then
                        self%larr(r,1) = l-j
                        self%marr(r,1) = abs(m-k)
                        self%c(r,1) = Cc(l-j, m-k)*ij(delta(0,k))
                    endif

                    if (-(l-j) <= abs(m+k) .and. abs(m+k) <= (l-j)) then
                        self%larr(r,2) = l-j
                        self%marr(r,2) = abs(m+k)
                        self%c(r,2) = (-1)**k*Cc(l-j, m+k)*ij(delta(0,k))
                    endif

                elseif (m >= 0 .and. k < 0) then
                    ! cs

                    if (-(l-j) <= -abs(m-k) .and. -abs(m-k) <= (l-j)) then
                        self%larr(r,1) = l-j
                        self%marr(r,1) = -abs(m-k)
                        self%c(r,1) = Cs(l-j, m-k)*(-1)**k
                    endif

                    if (-(l-j) <= -abs(m+k) .and. -abs(m+k) <= (l-j)) then

                        self%larr(r,2) = l-j
                        self%marr(r,2) = -abs(m+k)
                        self%c(r,2) = -Cs(l-j, m+k)
                    endif

                elseif (m < 0 .and. k >= 0) then

                    ! sc
                    if (-(l-j) <= -abs(m-k) .and. -abs(m-k) <= (l-j)) then
                        self%larr(r,1) = l-j
                        self%marr(r,1) = -abs(m-k)
                        self%c(r,1) = -Cs(l-j, m-k)*(-1)**m*ij(delta(0,k))
                    endif

                    if (-(l-j) <= -abs(m+k) .and. -abs(m+k) <= (l-j)) then
                        self%larr(r,2) = l-j
                        self%marr(r,2) = -abs(m+k)
                        self%c(r,2) = -Cs(l-j,m+k)*ij(delta(0,k))*(-1)**(m+k)
                    endif

                elseif (m < 0 .and. k < 0) then
                    ! ss

                    if (-(l-j) <= abs(m-k) .and. abs(m-k) <= (l-j)) then
                        self%larr(r,1) = l-j
                        self%marr(r,1) = abs(m-k)
                        self%c(r,1) = Cc(l-j, m-k)*(-1)**(m-k)
                    endif

                    if (-(l-j) <= abs(m+k) .and. abs(m+k) <= (l-j)) then
                        self%larr(r,2) = l-j
                        self%marr(r,2) = abs(m+k)
                        self%c(r,2) = -Cc(l-j, m+k)*(-1)**m
                    endif

                endif

                ! Scaling
                self%c(r,:) = self%c(r,:)*D(l,m)*(-1)**(m-k)/D(j,k)

            enddo
        enddo

    end subroutine

    !> The translation matrix in packed storage mode
    function TranslationMatrix_eval(self, dR) result(w)
        class(TranslationMatrix)            :: self
        real(REAL64), intent(in)            :: dR(3)

        integer                             :: N, M, j, k
        !real(REAL64)                        :: r !, dR_tmp(4,1)

        real(REAL64), allocatable           :: w(:)

        N = (self%lmax + 1)**2
        M = N*(N+1)/2
        allocate(w(M), source=0.0d0)

        !r = sqrt(sum(dR**2))
        !dR_tmp = reshape([dR(1),dR(2),dR(3),r],[4,1])
        self%S = reshape(self%yb%eval(reshape(dR, [3,1])), [N])

        ! Evaluate regular solid harmonics
        !k = 0
        !do j=0, self%lmax
        !    self%S(k+1:k+(2*j+1)) = self%S(k+1:k+(2*j+1))*r**j
        !    k = k+(2*j+1)
        !enddo

        ! Lower triangular part including diagonal
        where (self%larr(:,1) /= -1)
            w = self%c(:,1)*self%S(lm_map(self%larr(:,1), self%marr(:,1)))
        endwhere

        where (self%larr(:,2) /= -1)
            w = w + self%c(:,2)*self%S(lm_map(self%larr(:,2), self%marr(:,2)))
        endwhere

        return
    end function

    subroutine TranslationMatrix_apply(self, q, from, to)
        class(TranslationMatrix) :: self
        real(REAL64), intent(inout) :: q(:)
        real(REAL64), intent(in) :: from(3), to(3)
        real(REAL64), allocatable :: W(:)

        if (size(q) > (self%lmax + 1)**2) then
            write(ERROR_UNIT, *) "Multipole vector too big for translation."
            return
        else
            W = self%eval(from-to)
            if (self%trans) then
                call dtpmv('L', 'T', 'U', size(q), W, q, 1)
            else
                call dtpmv('L', 'N', 'U', size(q), W, q, 1)
            endif
            deallocate(W)
        endif

    end subroutine

    subroutine TranslationMatrix_transpose(self)
        class(TranslationMatrix) :: self
        self%trans = .not. self%trans
    end subroutine


    !!!!!!!!!!!!!!!!!!!!!!
    ! UTILITIES          !
    !!!!!!!!!!!!!!!!!!!!!!

    recursive function Cc(l,m) result(Clm)
        integer, intent(in) :: l,m
        integer :: i
        real(REAL64) :: n(2)
        real(REAL64) :: Clm
        n = 0
        if ( m == 0 ) then
            n(1) = product([(1.0d0*i, i=1,l)])
            Clm = 1.0d0/n(1)
        elseif (m > 0) then
            n(1) = product([(1.0d0*i, i=1,l-m)])
            n(2) = product([(1.0d0*i, i=1,l+m)])
            Clm = (-1)**m/sqrt(2.0d0*n(1)*n(2))
        else
            Clm = (-1)**m*Cc(l,-m)
        endif
    endfunction

    recursive function Cs(l,m) result(Clm)
        integer, intent(in) :: l,m
        real(REAL64) :: n(2)
        integer :: i
        real(REAL64) :: Clm
        n = 0
        if ( m == 0 ) then
            Clm = 0.0d0
        elseif (m > 0) then
            n(1) = product([(1.0d0*i, i=1,l-m)])
            n(2) = product([(1.0d0*i, i=1,l+m)])
            Clm = (-1)**m/sqrt(2.0d0*n(1)*n(2))
        else
            Clm = -(-1)**m*Cs(l,-m)
        endif
    endfunction

    ! Should these be scaled?
    function Cst(l,m) result(Clm)
        integer, intent(in) :: l,m
        real(REAL64) :: n(2)
        integer :: i
        real(REAL64) :: Clm
        n(1) = product([(1.d0*i,i=1,l-m)])
        n(2) = product([(1.d0*i,i=1,l+m)])
        Clm = n(1)*n(2)*Cs(l,m)
    end function

    function Cct(l,m) result(Clm)
        integer, intent(in) :: l,m
        real(REAL64) :: n(2)
        integer :: i
        real(REAL64) :: Clm
        n(1) = product([(1.0d0*i,i=1,l-m)])
        n(2) = product([(1.0d0*i,i=1,l+m)])
        Clm = n(1)*n(2)*Cc(l,m)
    end function

    real(REAL64) function D(l,m)
        integer, intent(in) :: l,m
        real(REAL64) :: n(2)
        integer :: i

        n(1) = product([(1.d0*i,i=1,l+m)])
        n(2) = product([(1.d0*i,i=1,l-m)])
        D = n(1)*n(2)*(2.0d0-delta(m,0))
        D = sqrt(D)
    endfunction

    integer function delta(i,j)
        integer, intent(in) :: i, j
        delta = merge(1, 0, i==j)
    endfunction

end module

