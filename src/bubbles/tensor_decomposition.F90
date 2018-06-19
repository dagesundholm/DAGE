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
!> @file tensor_decomposition.F90
!! operations related to tensor decompositions
module tensor_decomposition_m
    use Globals_m
    use xmatrix_m

    implicit none
    private

    public   :: tucker_compress
    public   :: tucker_decompress

    contains

    function three_index_tensor_to_matrix(T,n) result(m)
        ! three index tensor
        real(REAL64)              :: T(:,:,:)
        ! matrix version of T
        real(REAL64), allocatable :: m(:,:)
        ! which index is 
        integer                   :: n
        integer,allocatable       :: sizes(:)
        integer                   :: i,j


        ! dimensions of tensor
        sizes = shape(T)
        ! there might be a smarter way to do this...
        if(n==1) then
            ! allocate matrix
            allocate(m(sizes(1), sizes(2)*sizes(3)))
            do i=1,sizes(1)
                do j=1, sizes(3)
                    m(i,(j-1)*sizes(2)+1:(j-1)*sizes(2)+sizes(2)) = &
                        T(i,:,j)
                end do
            end do
        else if(n==2) then
            ! allocate matrix
            allocate(m(sizes(2),sizes(1)*sizes(3)))
            do i=1,sizes(2)
                do j=1, sizes(3)
                    m(i,(j-1)*sizes(1)+1:(j-1)*sizes(1)+sizes(1)) = &
                        T(:,i,j)
                end do
            end do
        else if(n==3) then 
            allocate(m(sizes(3),sizes(1)*sizes(2)))
            do i=1,sizes(3)
                do j=1, sizes(2)
                    m(i,(j-1)*sizes(1)+1:(j-1)*sizes(1)+sizes(1)) = &
                        T(:,j,i)
                end do
            end do

        else
            call perror('n should be 1, 2 or 3')
            stop
        end if
    end function

    function first_singular_vectors(m,N) result(UN)
        real(REAL64), intent(in)        :: m(:,:)
        integer, intent(in)             :: N 
        real(REAL64), allocatable       :: UN(:,:)
        real(REAL64), allocatable       :: U(:,:)
        integer, allocatable            :: sizes(:)
        real(REAL64), allocatable       :: S(:,:)            
        integer                         :: LWORK
        real(REAL64), allocatable       :: WORK(:)
        integer                         :: INFO
        real(REAL64)                    :: V(1,1)

        sizes = shape(m)
            
        allocate(S(sizes(1),sizes(1)))

        LWORK = 2* maxval([1, 3*sizes(1)+sizes(2), 5*sizes(1)  ])
        allocate(WORK(LWORK)) 
        allocate(U(sizes(1),sizes(1)))

        call dgesvd('S','N', sizes(1), sizes(2), m, &
                    max(1,sizes(1)), S, U, sizes(1), V,1, WORK, LWORK, INFO  )

        ! allocate result matrix for N left singular vectors
        allocate(UN(sizes(1),N))
        UN = U(:,1:N)

        ! cleanup
        deallocate(S, WORK, sizes,U)

    end function


    ! contract T with m at index ind
    function contract_tensor_with_matrix(T,m, ind) result(Tnew)
        real(REAL64), intent(in)    :: T(:,:,:)
        real(REAL64), intent(in)    :: m(:,:)
        integer                     :: ind
        integer                     :: i
        integer, allocatable        :: st(:),sm(:),stmp(:)
        real(REAL64), allocatable   :: Tnew(:,:,:)
        real(REAL64), allocatable   :: tmp(:,:,:)
        real(REAL64), allocatable   :: tmp2(:,:)
        real(REAL64), allocatable   :: tmp3(:,:)

        st = shape(T)
        sm = shape(m)

        if (ind == 1) then
            tmp = reshape(T,[2,3,1])
        else if(ind == 2) then
            tmp = reshape(T,[1,3,2])
        else if(ind == 3) then
            tmp = reshape(T,[1,2,3])
        else
            call perror('n should be 1, 2 or 3')
            stop
        end if

        ! shape of tmp
        stmp = shape(tmp)

        ! tensor tmp to matrix tmp2
        tmp2 = reshape(tmp,[stmp(1)*stmp(2),stmp(3)])
        deallocate(tmp)
        tmp3 = xmatmul(tmp2, m)
        deallocate(tmp2)

        ! reshape matrix tmp3 to tensor tmp
        tmp = reshape(tmp3,[stmp(1),stmp(2),sm(2)])
        deallocate(tmp3)

        ! reshape tmp to Tnew
        ! here we assume that reshape(T, order) works the same 
        ! way as Octave's permute
        if (ind==1) then
            Tnew = reshape(tmp, [3,1,2])
        else if(ind==2) then
            Tnew = reshape(tmp, [1,3,2])
        else if(ind==3) then
            Tnew = reshape(tmp, [1,2,3])
        end if

        deallocate(tmp)
    end function

    ! compress to rankin tucker vectors
    ! default rank is 40
    ! A,B,C and r are calculated from T
    subroutine tucker_compress(T, A, B, C, r, rankin) 
        real(REAL64), intent(inout)    :: T(:,:,:)
        real(REAL64), allocatable      :: A(:,:)
        real(REAL64), allocatable      :: B(:,:)
        real(REAL64), allocatable      :: C(:,:)
        real(REAL64), allocatable      :: r(:,:,:)
        ! rank 
        integer, optional              :: rankin
        integer                        :: rank
        ! worker variables
        real(REAL64), allocatable      :: Tw(:,:)        
        real(REAL64), allocatable      :: tmp(:,:,:)
        real(REAL64), allocatable      :: tmp2(:,:,:)

        ! choose rank of compression
        if(present(rankin)) then
            rank = rankin
        else
            rank = 40
        end if

        Tw = three_index_tensor_to_matrix(T,1)
        A = first_singular_vectors(Tw, rank)
        deallocate(Tw)
        Tw = three_index_tensor_to_matrix(T,2)
        B = first_singular_vectors(Tw, rank)
        deallocate(Tw)
        Tw = three_index_tensor_to_matrix(T,3)
        C = first_singular_vectors(Tw, rank)
        deallocate(Tw) 

        tmp = contract_tensor_with_matrix(T,A,1)
        tmp2= contract_tensor_with_matrix(tmp,B,2)
        r=contract_tensor_with_matrix(tmp2,C,3)

        deallocate(tmp,tmp2)

    end subroutine

    subroutine tucker_decompress(T,A,B,C,r)
        real(REAL64), allocatable :: T(:,:,:)
        real(REAL64)              :: A(:,:)
        real(REAL64)              :: B(:,:)
        real(REAL64)              :: C(:,:)
        real(REAL64)              :: r(:,:,:)
        ! workers
        real(REAL64), allocatable :: tmp(:,:,:)
        real(REAL64), allocatable :: tmp2(:,:,:)

        tmp = contract_tensor_with_matrix(r,transpose(A),1)
        tmp2= contract_tensor_with_matrix(tmp,transpose(B),2)
        T = contract_tensor_with_matrix(tmp2,transpose(C),3)

        ! cleanup
        deallocate(tmp, tmp2)

    end subroutine




end module
