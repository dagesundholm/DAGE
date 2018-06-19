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
! this module contains stuff related to modified spherical
! Bessel functions 
module modified_bessel_m
    use Globals_m

    implicit none

    type           ::    mod_spherical_Bessel_first
        integer,allocatable    ::    sinh_coeffs(:)
        integer,allocatable    ::    cosh_coeffs(:)
        contains
        procedure  :: eval        => evaluate_first
        procedure  :: destroy     => destroy_first
    end type

    type           ::    mod_spherical_Bessel_second
        integer, allocatable   ::    coeffs(:)
    
        contains
        procedure  :: eval        => evaluate_second
        procedure  :: destroy     => destroy_second
    end type

! collection of modified spherical bessel functions
    type           ::   Bessel_collection
        type(mod_spherical_Bessel_first),  allocatable  :: first(:)
        type(mod_spherical_Bessel_second), allocatable  :: second(:)

        contains  
        procedure :: destroy     => destroy_bessel_collection
        procedure :: eval        => evaluate_bessel_collection
    end type




    contains


    !> this function returns the coefficients of a 
    !! modified spherical Bessel function of the first kind.
    !! Recursion relation is
    !! \f[
    !!     g_n= g_{n-2} - \frac{2n-1}{r} g_{n-1}
    !! \f]
    !! Modified function is
    !! \f[
    !!     \hat{I}_{n+½}(r) = g_n(r) sinh(r) + g_{-(n+1)}(r) cosh(r) 
    !! \f]
    recursive function first_kind_coeff(l,lmax) result(res)
        integer, intent(in)     :: l
        integer, intent(in)     :: lmax
        integer, allocatable    :: res(:)
        integer                 :: i
        integer, allocatable    :: tmp(:)

        allocate(res(lmax+1))
        allocate(tmp(lmax+1))
        res=0

 
        if(l>=0) then
            if(l==0) then
                res(1)=1
            end if
            if(l==1) then
                res(2)=-1
            end if
            if(l>=2) then
                res=first_kind_coeff(l-2,lmax)
                tmp=first_kind_coeff(l-1,lmax)
                do i=1,lmax
                    res(i+1)=res(i+1)-(2*l-1)*tmp(i)
                end do
            endif
            else
                res=first_kind_coeff(l+2,lmax)
                tmp=first_kind_coeff(l+1,lmax)
                do i=1,lmax
                    res(i+1)=res(i+1)+(2*l+3)*tmp(i)
                end do
        endif

    end function

    !> this function returns the coefficients of a 
    !! modified spherical Bessel function of the second kind.
    !! Recursion relation is:
    !! \f[
    !!     g_n= g_{n-2} + \frac{2n-1}{r} g_{n-1}
    !! \f]
    !! Modified function is
    !! \f[
    !!     \hat{K}_{n+½}(r) = g_n(r) \frac{2}{\pi} e^{-r}
    !! \f]

    recursive function second_kind_coeff(l,lmax) result(res)
        integer, intent(in)     :: l
        integer, intent(in)     :: lmax
        integer, allocatable    :: res(:)
        integer                 :: i,j
        integer, allocatable    :: tmp(:)

        if(l>=0) then
            j=l
        else
            j=-l
        endif

        allocate(res(lmax+1))
        allocate(tmp(lmax+1))
        res=0
        if(j==0) then
            res(1)=1
        end if
        if(j==1) then
            res(1)=1
            res(2)=1
        end if
        if(j>=2) then
            res=second_kind_coeff(l-2,lmax)
            tmp=second_kind_coeff(l-1,lmax)
            do i=1,lmax
                res(i+1)=res(i+1)+(2*l-1)*tmp(i)
            end do

        endif

    end function

    function evaluate_second(self,distances) result(res)
        class(mod_spherical_Bessel_second)  :: self
        real(REAL64)                        :: distances(:)
        real(REAL64)                        :: res(size(distances))
        integer                             :: i,j

        res=0d0
        do i=1,size(distances)
            if(distances(i)>epsilon(0d0)) then
                do j=1,size(self%coeffs)
                    if(distances(i)**(j)>epsilon(0d0))then
                    res(i)=res(i)+self%coeffs(j)*distances(i)**(-j)
                    endif
                end do
                res(i)=res(i)*exp(-distances(i)) *pi/2
            end if
        end do

    end function

    function second_kind_init(l) result(new)
        integer                           :: l
        type(mod_spherical_Bessel_second) :: new
        new%coeffs = second_kind_coeff(l,l+1)
    end function

    function first_kind_init(l) result(new)
        integer                           :: l
        type(mod_spherical_Bessel_first) :: new
        new%sinh_coeffs = first_kind_coeff(l,l+1)
        new%cosh_coeffs = first_kind_coeff(-l-1,l+1)
    end function

    function evaluate_first(self, distances) result(res)
        class(mod_spherical_Bessel_first)   :: self
        real(REAL64)                        :: distances(:)
        real(REAL64)                        :: res(size(distances))
        integer                             :: i,j

        res=0d0
        do i=1,size(distances)
            if(distances(i)>epsilon(0d0)) then
! notice that size(self%sinh_coeffs) == size(self%cosh_coeffs)
                do j=1,size(self%sinh_coeffs)
                    res(i)=res(i)   +   &
self%sinh_coeffs(j)*distances(i)**(-j)*sinh(distances(i)) +  &
self%cosh_coeffs(j)*distances(i)**(-j)*cosh(distances(i))
                end do
            end if
        end do
    end function

    subroutine destroy_first(self)
        class(mod_spherical_Bessel_first)  :: self
        deallocate(self%sinh_coeffs)
        deallocate(self%cosh_coeffs)
    end subroutine

    subroutine destroy_second(self)
        class(mod_spherical_Bessel_second)  :: self
        deallocate(self%coeffs)
    end subroutine

    subroutine destroy_bessel_collection(self)
        class(Bessel_collection)            :: self
        deallocate(self%first)
        deallocate(self%second)
    end subroutine

    function evaluate_bessel_collection(self,r1,r2,i) result(res)
        class(Bessel_collection)        :: self
        real(REAL64)                    :: r1,r2
        integer                         :: i
        real(REAL64)                    :: res
        real(REAL64)                    :: tmp(1)
        real(REAL64)                    :: s1,s2

        res = 0d0

        if(r1>r2) then
            s1=r1
            s2=r2
        else
            s1=r2
            s2=r1
        endif


        tmp=self%first(i+1)%eval( [ s1 ]  ) *  & 
            self%second(i+1)%eval( [ s2 ] )
        res=tmp(1)
    end function

    function Bessel_collection_init(l)  result(new)
        integer                       :: l
        type(Bessel_collection)       :: new
        integer                       :: i
        type(mod_spherical_Bessel_first), allocatable   :: first(:)
        type(mod_spherical_Bessel_second), allocatable  :: second(:)
        allocate(first(l+1))
        allocate(second(l+1))

        do i=0,l
            first(i+1)  = first_kind_init(i)
            second(i+1) = second_kind_init(i)
        end do
        new%first=first
        new%second=second
    end function





end module
