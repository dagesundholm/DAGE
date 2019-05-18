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
!> @file lip.F90
!! Defines a Lagrange interpolation polynomial object

!> Representation of Lagrange interpolation polynomials.
!!
!! This module defines the `LIPBasis` type (Lagrange Interpolated Polynomial
!! type), which describes an Interpolated Polynomial in Lagrange
!! form.
!!
!! #### Maths
!!
!! Any function *y* that is continuous on the interval [*a*, *b*] can be
!! interpolated by a Lagrange interpolation polynomial *L(x)* given a
!! set of *n* grid points \f$(x_{1}, y_{1})...(x_{n},y_{n})\f$ in the
!! interval. *L(x)* satisfies by definition \f$L(x_{i})=y(x_i)=y_{i}\f$ so that the
!! interpolation is exact in at the grid points. Evaluation points
!! intervening two grid points should yield a good approximation to the
!! original function so that overall
!!
!! @f[ L(x) \approx y(x) \quad a \le x \le b. @f]
!!
!! This is achieved by defining *n* basis functions of degree *n-1* that
!! have the property of being one at one of the grid points and zero at all
!! others. Clearly, such a function for the grid point \f$(x_j, y_j)\f$
!! is given by
!!
!! @f[l_j(x) = \prod\limits_{i\neq j}^n \frac{x-x_j}{x_i-x_j} @f]
!!
!! and *L(x)* is consequently
!!
!! @f[ L(x) = \sum\limits_{j=1}^n y_j l_j(x). @f]
!!
!! #### Notes
!!
!! Storagewise, each basis function, \f$l_{j}(x)\f$, can by represented as a sum
!!
!! @f[ l_{j}(x) = \sum\limits_{i=0}^n c_{ij}x^i, @f]
!!
!! and the expansion coefficients \c c(i,j) can be stored in a matrix.
!!

module LIPBasis_class
    use Globals_m
    use Xmatrix_m
    use GaussQuad_class
    use pprinter
    implicit none

    public LIPBasisInit
    public overlap
    public eval_polys
    private

    !> Representation of Lagrange Interpolation Polynomial basis
    type, public :: LIPBasis
        private
        !> Number of points, or polynomial order +1
        integer(INT32) :: nlip
        !> choice of grid: 0 -> equidistant, 1 -> gauss-lobatto
        integer(INT32) :: grid_type
        !> Beginning of the cell (in its own scale)
        real(REAL64) :: first
        !> End of the cell
        real(REAL64) :: last
    contains
        ! Accessors
        procedure :: get_nlip  => LIPBasis_get_nlip
        procedure :: get_gridtype  => LIPBasis_get_gridtype
        procedure :: get_first => LIPBasis_get_first
        procedure :: get_last  => LIPBasis_get_last

        ! Coefficient constructors
        !> Coefficients of the polynomials and their derivatives
        procedure :: coeffs            => LIPBasis_coeffs
        !> Coefficients of the integrals of the polynomials
        procedure :: integral_coeffs   => LIPBasis_integral_coeffs
        !> Integrals of the polynomials within the cell
        procedure :: integrals         => LIPBasis_integrals
        !> Inward integrals
        procedure :: inward_integrals  => LIPBasis_inward_integrals
        !> Outward integrals
        procedure :: outward_integrals => LIPBasis_outward_integrals
        !> Overlap integrals
        procedure :: overlaps          => LIPBasis_overlaps
        !> Derivative constants?
        procedure :: der_const         => LIPBasis_der_const
    end type

contains

! %%%%%%%%%%% Constructor %%%%%%%%%%%%%%
    pure function LIPBasisInit(nlip, gridtype) result(new)
        integer, intent(in) :: nlip, gridtype
        type(LIPBasis) :: new

        new=LIPBasis( nlip=nlip, grid_type=gridtype, first=-(nlip-1)/2, last=nlip/2 )
    end function

! %%%%%%%%%%% Accessors %%%%%%%%%%%%%%
    pure function LIPBasis_get_nlip(self) result(nlip)
        class(LIPBasis), intent(in) :: self
        integer :: nlip

        nlip=self%nlip
    end function

    pure function LIPBasis_get_gridtype(self) result(gridtype)
        class(LIPBasis), intent(in) :: self
        integer :: gridtype

        gridtype=self%grid_type
    end function

    pure function LIPBasis_get_first(self) result(first)
        class(LIPBasis), intent(in) :: self
        real(REAL64) :: first

        first=self%first
    end function

    pure function LIPBasis_get_last(self) result(last)
        class(LIPBasis), intent(in) :: self
        real(REAL64) :: last

        last=self%last
    end function

! %%%%%%%%%%% Workers %%%%%%%%%%%%%%
    !> @todo This should be moved to grid.F90 or similar
    subroutine overlap(self,s,cellh) ! cellh used as scaling lnw
        type(LIPBasis) :: self
        real(REAL64), dimension(:,:), intent(out) :: s
        real(REAL64), dimension(:), intent(in) :: cellh

        real(REAL64), allocatable :: ints(:,:)
        integer(INT32) :: ncell,icell,i,j,k,l, ndim

        s=0.d0
        ncell=size(cellh)
        ndim=size(s(:,1))

        ints=self%overlaps(0)

        do icell=1,ncell
            do i=1,self%nlip
                do j=1,self%nlip
                    k=(icell-1)*(self%nlip-1)+i
                    l=(icell-1)*(self%nlip-1)+j
                    s(k,l)=s(k,l)+ints(i,j)*cellh(icell)
                end do
            end do
        end do

        if(debug_g > 1) then
            write(6,'(a)') ' '
            write(6,'(a)') ' '
            write(6,'(a)') &
            ' The overlap matrix '
            write(6,'(a)') repeat('=', 70)
            do i=1,ndim
                write(6,'(5(D14.8,1x))') (s(i,j),j=1,ndim)
            end do
            write(6,'(a)') repeat('=', 70)
            write(6,'(a)') ' '
            write(6,'(a)') ' '
        end if

        deallocate(ints)
    end subroutine overlap

    !> Multiplies one polynomial with another bundle of polynomials

    !> The coefficients of the resulting polynomials are given in one row
    !! per polynomial, in decreasing power order:
    !!
    !! \f$ A_i(x)=B(x) C_i(x)=\sum_{j=0}^{N} \f$ \c mult_pol(i,j) \f$ x^{N-j}\f$
    pure function mult_pol(pol1,pol2) result(res)
        !> The coefficients of the first polynomial must be given in
        !! decreasing power order:
        !!
        !! \f$ B(x)=\sum_{i=0}^{N} \f$ \c pol1(i) \f$ x^{N-j}\f$
        real(REAL64),intent(in) :: pol1(:)
        !> The coefficients of the bundle of polynomials must be given in one
        !! row per polynomial, in decreasing power order:
        !!
        !! \f$ C_i(x)=\sum_{j=0}^{N} \f$ \c pol2(i,j) \f$ x^{N-j}\f$
        real(REAL64),intent(in) :: pol2(:,:)
        integer                 ::  i,m,n
        real(REAL64) :: res(size(pol2,1),size(pol1)+size(pol2,2)-1)

        m=size(pol1)
        n=size(pol2,2)
        res=0.d0
        do i=1,m
            res(:,i:i+n-1)=res(:,i:i+n-1)+pol1(i)*pol2(:,:)
        end do
    end function mult_pol

    !> Return the LIPBasis coefficients.
    !! \f$\partial_x^k P_i(x)=\sum_{j=0}^{N-k} \f$ \c coeffs(k)%%p(i,j) \f$ x^{N-j}\f$\n

    !> The result is given as an array of \c real64ptr_2d objects. The k-th
    !! derivative is given as the k-th element of the returned array. For
    !! instance, for \f$N=7\f$ (6-th order), \c coeffs(1)%%p(5,4) is the
    !! coefficient of the 2nd (6-4) order term (\f$x^2\f$) of the 1st
    !! derivative of the 5th basis polynomial.
    !!
    !! For instance the 2nd order LIP's (centered at -1, 0 and 1) are given
    !! by:
    !! \f[ P_1(x)=\frac{1}{2}x-\frac{1}{2}x^2 \f]
    !! \f[ P_2(x)=1-x^2 \f]
    !! \f[ P_3(x)=\frac{1}{2}x+\frac{1}{2}x^2 \f]
    !! \c LIPBasis%%coeffs(2) will return the following arrays:
    !!
    !! \c coeffs(0)%%p(:,:) =
    !! \f$
    !!   \left( \begin{array}{ccc}
    !!   -0.5 &  0.5 &  0.0 \\\
    !!   -1.0 &  0.0 &  1.0 \\\
    !!    0.5 &  0.5 &  0.0
    !! \end{array} \right)
    !! \f$
    !!
    !! \c coeffs(1)%%p(:,:) =
    !! \f$
    !!   \left( \begin{array}{cc}
    !!   -1.0 &  0.5 \\\
    !!   -2.0 &  0.0 \\\
    !!    1.0 &  0.5
    !! \end{array} \right)
    !! \f$
    !!
    !! \c coeffs(2)%%p(:,:) =
    !! \f$
    !!   \left( \begin{array}{c}
    !!   -1.0\\\
    !!   -2.0\\\
    !!    1.0
    !! \end{array} \right)
    !! \f$
    pure function LIPBasis_coeffs(self, der) result(coeffs)
        implicit none
        class(LIPBasis), intent(in) :: self
        integer,         intent(in) :: der
        type(REAL64_2D)             :: coeffs(1:der+1)
        real(REAL64)                :: glgrid(self%nlip)

        integer(INT32):: i,j,k,n
        real(REAL64) :: x_i,dx

        !if (der>=self%nlip) then
            !write(ppbuf,&
            !'("Requested derivative order larger than LIPBasis order (",&
            !    &i2,">",i2,")")') der, self%nlip-1
            !call perror(ppbuf)
            !stop
        !end if

        do k=0,der
            allocate(coeffs(k+1)%p(self%nlip,self%nlip-k))
            coeffs(k+1)%p=0.d0
        end do

        n=self%nlip

        ! Generate coefficients of the base polynomials
        select case (self%grid_type)

            case (1) ! equidistant
                coeffs(1)%p(:,n)=1.d0
                do i=1,self%nlip               !Iterate over points
                    x_i=i-(self%nlip+1)/2
                    do j=1,self%nlip           !Iterate over polynomials
                        if(i==j) cycle         !Skip x_i=x_j
                        dx=j-i
                        coeffs(1)%p(j,1:n-1)=(coeffs(1)%p(j,2:n)-coeffs(1)%p(j,1:n-1)*x_i)/dx
                        coeffs(1)%p(j,n)=-coeffs(1)%p(j,n)*x_i/dx
                    end do
                end do

            case (2) ! gauss lobatto
                glgrid = gauss_lobatto_grid(self%nlip, self%first, self%last)
                coeffs(1)%p(:,n)=1.d0
                do i=1,self%nlip               !Iterate over points
                    x_i=glgrid(i)
                    do j=1,self%nlip           !Iterate over polynomials
                        if(i==j) cycle         !Skip x_i=x_j
                        dx=glgrid(j)-glgrid(i)
                        coeffs(1)%p(j,1:n-1) = (coeffs(1)%p(j,2:n)-coeffs(1)%p(j,1:n-1)*x_i)/dx
                        coeffs(1)%p(j,n)=-coeffs(1)%p(j,n)*x_i/dx
                    end do
                end do

            case default
        end select

        ! Generate derivative coefficients
        do k=1,der
            do j=0,n-k-1
                coeffs(k+1)%p(:,n-j-k)=coeffs(k)%p(:,n-j-k) * (j+1)
            end do
        end do
        return
    end function

    !> Compute the coefficients of the indefinite integrals of the
    !! polynomials.
    !! \f$\int P_i(x)\mathrm{d}x =\sum_{j=0}^{N}\f$ \c integral_coeffs(i,j) \f$x^{N+1-j}\f$
    pure function LIPBasis_integral_coeffs(self) result(int_coeffs)
        class(LIPBasis), intent(in) :: self
        real(REAL64)                :: int_coeffs(self%nlip, self%nlip + 1)

        type(REAL64_2D), allocatable :: coeffs(:)

        integer(INT32):: i,n

        n=self%nlip
        int_coeffs(:,n+1)=0.d0
        coeffs=self%coeffs(1)
        forall (i=0 : n-1)
            int_coeffs(:,n-i)=coeffs(1)%p(:,n-i) / (i+1)
        end forall

        deallocate(coeffs(1)%p)
        deallocate(coeffs)
    end function

    !> Compute the integrals of the polynomials over all space.\n
    !> \c integrals(i) \f$ = \int_{-\infty}^{\infty}
    !! \chi_i(x)\mathrm{d}x \f$
    ! xwh, actually, integrate with the domain, not the whole R,
    ! or in another way, lips are all 0 outside the domain,
    ! if not, probably the integration does not converge!
    pure function LIPBasis_integrals(self) result(integrals)
        class(LIPBasis), intent(in) :: self
        real(REAL64)                :: integrals(self%nlip)

        real(REAL64)                :: coeffs(self%nlip, self%nlip + 1)
        real(REAL64)                :: tmp1(self%nlip), tmp2(self%nlip)
        integer :: i

        coeffs=self%integral_coeffs()

        tmp1=0.d0
        do i=1, self%nlip
            tmp1=(tmp1+coeffs(:,i)) * self%first
        end do
        tmp2=0.d0
        do i=1,self%nlip
            tmp2=(tmp2+coeffs(:,i)) * self%last
        end do
        integrals=tmp2-tmp1
    end function

    !> Compute the inward integrals.
    !> \c inward_integrals(i,j) \f$ = \int_{x_j}^{\infty}
    !! \chi_i(x)\mathrm{d}x \f$
    !! For domain grids, x0, ..., xn
    !! res(1,1): integral of lip1 in domain x0-xn
    !! res(1,2): integral of lip1 in domain x1-xn
    !! res(1,3): integral of lip1 in domain x2-xn
    !! ...
    pure function LIPBasis_inward_integrals(self) result(res)
        class(LIPBasis), intent(in) :: self
        real(REAL64)                :: res(self%nlip,self%nlip-1)

        real(REAL64)                :: coeffs(self%nlip, self%nlip + 1)
        real(REAL64)                :: tmp1(self%nlip), tmp2(self%nlip), x
        integer                     :: i,j


        coeffs=self%integral_coeffs()

        tmp2=0.d0
        do i=1,self%nlip
            tmp2=(tmp2+coeffs(:,i))*self%last
        end do

        x=self%first
        do j=1,self%nlip-1
            tmp1=0.d0
            do i=1,self%nlip
                tmp1=(tmp1+coeffs(:,i))*x
            end do
            res(:,j)=tmp2-tmp1
            x=x+1
        end do
    end function


    !> Compute the outward integrals.
    !> \c outward_integrals(i,j) \f$ = \int_{-\infty}^{x_j}
    !! \chi_i(x)\mathrm{d}x \f$
    !! For domain grids, x0, ..., xn
    !! res(1,1): integral of lip1 in domain x0-x1
    !! res(1,2): integral of lip1 in domain x0-x2
    !! res(1,3): integral of lip1 in domain x0-x3
    !! ...
    pure function LIPBasis_outward_integrals(self) result(res)
        class(LIPBasis), intent(in) :: self
        real(REAL64)                :: res(self%nlip,self%nlip -1)

        real(REAL64)                :: coeffs(self%nlip, self%nlip + 1)
        real(REAL64)                :: tmp1(self%nlip), tmp2(self%nlip), x
        integer                     :: i,j


        coeffs=self%integral_coeffs()

        tmp1=0.d0
        do i=1,self%nlip
            tmp1=(tmp1+coeffs(:,i))*self%first
        end do

        x=self%first
        do j=1,self%nlip-1
            x=x+1.d0
            tmp2=0.d0
            do i=1,self%nlip
                tmp2=(tmp2+coeffs(:,i))*x
            end do
            res(:,j)=tmp2-tmp1
        end do
    end function


    !> Compute the overlap integrals.
    !> \c overlaps(i,j) \f$ = \int_{-\infty}^{\infty}
    !!  \chi_i(x)\chi_j(x)\mathrm{d}x \f$
    pure function LIPBasis_overlaps(self,d) result(res)
        class(LIPBasis), intent(in) :: self
        integer,         intent(in) :: d
        real(REAL64)                :: res(self%nlip,self%nlip)
        type(REAL64_2D), allocatable :: coeffs(:)

        real(REAL64) :: poly(self%nlip,2*( self%nlip-d ))
        integer :: i,k


        coeffs=self%coeffs(d)

        do i=1,self%nlip
            poly(:,2:)=mult_pol(coeffs(d+1)%p(i,:), coeffs(d+1)%p(:,:))
            do k=1,2*( self%nlip-d )-1
                poly(:,k)=poly(:,k+1)/(2*( self%nlip-d )-k)
            end do
            poly(:,2*(self%nlip-d))=0.d0
            ! if(i==1) then
            ! end if
            res(:,i)=eval_polys(poly, self%last) -&
                     eval_polys(poly, self%first)
        end do

        deallocate(coeffs(1)%p)
        deallocate(coeffs)
    end function


    pure function LIPBasis_der_const(self) result(res)
        class(LIPBasis), intent(in) :: self
        real(REAL64)                :: res(self%nlip,self%nlip)

        type(REAL64_2D), allocatable :: coeffs(:)
        real(REAL64), dimension(self%nlip) :: points
        real(REAL64), dimension(self%nlip-1) :: order

        integer :: i,k

        do i=1, self%nlip
            points(i) = -1*(self%nlip-1)/2 + i - 1
        end do

        do i=1, self%nlip-1
            order(i) = (self%nlip-1)-i
        end do

        coeffs=self%coeffs(1)

        do i=1,self%nlip
           do k = 1, self%nlip
               res(i,k) = sum(coeffs(2)%p(i,:)*points(k)**order(:))
           end do
        end do

        deallocate(coeffs(1)%p)
        deallocate(coeffs)
    end function LIPBasis_der_const



    !> Evaluate \c m polynomials (\f$\{p_i(x)\}\f$) of order \c n-1 whose
    !! coefficientes are given in array \c coeffs(m,n) at point \c x. Returns
    !! an array \c p with \c m elements given by \c p(i)= \f$p_i(\f$x\f$)\f$.
    pure function eval_polys(coeffs,x) result(res)
        !> Polynomial coefficients
        !! \f$p_{i}(x)=\sum_{j=0}^n\f$ \c pcoeffs(i,j) \f$x^{n-j}\f$
        real(REAL64), intent(in) :: coeffs(:,:)
        !> Point at which the polynomials will be evaluated
        real(REAL64), intent(in) :: x

        integer :: k
        real(REAL64) :: res(size(coeffs,1))

        res=coeffs(:,1)
        do k=2,size(coeffs,2)
            res=(res*x)+coeffs(:,k)
        end do
    end function
end module

