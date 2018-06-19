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
!> @file harmonics.F90 Real-valued spherical harmonics.

!> This file is expected to supersede harmonic.F90 (notice the similarity in the
!! names! Let that be a motivation to remove harmonic.F90 ASAP ;) ). Things
!! missing:
!! @todo [YProduct](@ref harmonic_class::yproduct)
!! @todo [assign_c_pointers](@ref harmonic_class::assign_c_pointers)
!! @todo [eval](@ref realsphericalharmonics_eval) must be able to evaluate
!! derivatives (see [Ybundle%eval](@ref harmonic_class::eval))
!! @todo [Cart2Sph_Iter_t](@ref harmonic_class::cart2sph_iter_t) and rename as
!! Cart2SphProjector.
module RealSphericalHarmonics_class
    use ISO_FORTRAN_ENV
    use Globals_m
    use CartIter_class
    use Grid_class
#ifdef HAVE_OMP
    use omp_lib
#endif
#ifdef HAVE_CUDA
    use Cuda_m
    use ISO_C_BINDING
#endif
    implicit none

    public :: RealRegularSolidHarmonics, RealSphericalHarmonics, ComplexSphericalHarmonics, &
              RealSphericalCubeHarmonics
    public :: lm_map

    private


!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!      3D polynomial, used to construct RealRegularSolidHarmonics  %
!          using recursion relations.                              %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    type :: Poly3D
        private
        integer,      allocatable :: expos(:,:)
        real(REAL64), allocatable :: coeffs(:)
    contains
        procedure :: print   => Poly3D_print
        procedure :: destroy => Poly3D_destroy

        procedure, private ::          Poly3D_plus_Poly3D
        generic :: operator(+) =>      Poly3D_plus_Poly3D

        procedure, private ::          Poly3D_minus_Poly3D
        generic :: operator(-) =>      Poly3D_minus_Poly3D

        procedure, private          :: Poly3D_times_Poly3D
        procedure, private          :: Poly3D_times_REAL64
        procedure, private, pass(p) :: REAL64_times_Poly3D
        generic :: operator(*) =>      Poly3D_times_Poly3D, &
                                       Poly3D_times_REAL64, &
                                       REAL64_times_Poly3D
        procedure, private          :: get_derivative => Poly3D_get_derivative
    end type
    
    interface Poly3D
        module procedure Poly3D_init_empty
        module procedure Poly3D_init_explicit
    end interface

    !> Arrays of Poly3D objects.
    type :: Poly3D_array
        private
        type(Poly3D), allocatable :: m(:)
    contains
        procedure :: print   => Poly3D_array_print
        procedure :: destroy => Poly3D_array_destroy
    end type

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!      Real-valued solid harmonics                                 %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    !> The real spherical harmonics can be written as:
    !! \f[
    !!     Y_{lm}=\sum c_{ijk}^{lm} \hat{x}^i \hat{y}^j \hat{z}^k
    !! \f]
    !!
    !! realsphericalharmonics is a sparse representation of the matrix
    !! \f$c_{ijk}^{lm}\f$. The methods
    !! [eval](@ref realsphericalharmonics_eval) and
    !! [cart2sph](@ref realsphericalharmonics_cart2sph)
    !! exploit this sparsity for efficiency.
    type :: RealRegularSolidHarmonics
        private
        !> Minimum angular momentum number.
        integer                   :: lmin
        !> Maximum angular momentum number.
        integer                   :: lmax
        !> Number of (l,m) terms to which a given triplet of exponents (i,j,k)
        !! contributes.
        integer,      allocatable :: number_of_terms(:, :)
        !> Coefficients.
        real(REAL64), allocatable :: coefficients(:, :)
        !> l numbers.
        integer,      allocatable :: l(:, :)
        !> m numbers.
        integer,      allocatable :: m(:, :)
        !> l,m  order numbers.
        integer,      allocatable :: lm(:, :)
        !> Internal term counter.
        integer,      allocatable :: k(:)
        !> Internal exponent triplet counter.
        integer,      allocatable :: ijk(:)
        !> Type of normalization, (1=Racah's normalization), (2=Conventional normalization)
        integer                   :: normalization = 1
        !> The unformatted coefficient matrix
        type(Poly3D_array), allocatable :: polynomials(:, :)
#ifdef HAVE_CUDA
        type(C_PTR)               :: cuda_interface
#endif
    contains
        procedure          :: next_lm           => RealRegularSolidHarmonics_next_lm
        procedure          :: next_lm_          => RealRegularSolidHarmonics_next_lm_
        procedure          :: reset             => RealRegularSolidHarmonics_reset
        procedure          :: eval              => RealRegularSolidHarmonics_eval
        procedure          :: eval_grid         => RealRegularSolidHarmonics_eval_grid
        procedure          :: cart2sph          => RealRegularSolidHarmonics_cart2sph
        procedure          :: print             => RealRegularSolidHarmonics_print
        procedure          :: destroy           => RealRegularSolidHarmonics_destroy
        procedure          :: evaluate_gradients=> RealRegularSolidHarmonics_evaluate_gradients
        procedure, private :: convert_poly3d_array_to_internal &
                                                => RealRegularSolidHarmonics_convert_poly3d_array_to_internal
#ifdef HAVE_CUDA
        procedure, private :: assign_c_pointers => RealRegularSolidHarmonics_assign_c_pointers
        procedure, private :: eval_grid_cuda    => RealRegularSolidHarmonics_eval_grid_cuda
        procedure, private :: init_cuda         => RealRegularSolidHarmonics_init_cuda
        procedure          :: get_cuda_interface => RealRegularSolidHarmonics_get_cuda_interface
#endif
        
    end type

    
    interface RealRegularSolidHarmonics
        module procedure RealRegularSolidHarmonics_init
    end interface

!------------ CUDA Interfaces ----------------- !

#ifdef HAVE_CUDA
!     interface
!         type(C_PTR) function RealRegularSolidHarmonics_cuda_init(lmin, lmax, number_of_terms, normalization, &
!                                                  exponentials, coefficients, number_of_lm_terms, new_coefficients, &
!                                                  lm_indices, ijk_max)  bind(C)
!             use ISO_C_BINDING
!             integer(C_INT), value :: lmin
!             integer(C_INT), value :: lmax
!             integer(C_INT)        :: number_of_terms(*)
!             integer(C_INT), value :: normalization
!             integer(C_INT)        :: exponentials(*)
!             real(C_DOUBLE)        :: coefficients(*)
!             integer(C_INT)        :: number_of_lm_terms(*)
!             real(C_DOUBLE)        :: new_coefficients(*)
!             integer(C_INT)        :: lm_indices(*)
!             integer(C_INT), value :: ijk_max
! 
!         end function
!     end interface
! 
!     interface
!         subroutine RealRegularSolidHarmonics_cuda_evaluate_grid(harmonics, &
!                 grid, center, cubes) bind(C)
!             use ISO_C_BINDING
!             type(C_PTR), value    :: harmonics
!             type(C_PTR), value    :: grid
!             real(C_DOUBLE)        :: center(3)
!             real(C_DOUBLE)        :: cubes(*)
!         end subroutine
!     end interface
! 
!     interface
!         subroutine RealRegularSolidHarmonics_cuda_destroy(harmonics) bind(C)
!             use ISO_C_BINDING
!             type(C_PTR), value    :: harmonics
!         end subroutine
!     end interface

    
#endif

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!      Real-valued spherical harmonics (Racah's normalization)     %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    type :: RealSphericalHarmonics
        private
        !> Regular Solid Harmonics - Object to help evaluate the values of spherical harmonics
        !! as solid harmonics are related to spherical harmonics by a multiplier of 1/(r^l). 
        type(RealRegularSolidHarmonics) :: solid_harmonics
    contains
        procedure          :: eval                         => RealSphericalHarmonics_eval
        procedure          :: evaluate_gradients           => RealSphericalHarmonics_evaluate_gradients
        
        procedure          :: eval_grid                    => RealSphericalHarmonics_eval_grid
        procedure          :: destroy                      => RealSphericalHarmonics_destroy
    end type


    interface RealSphericalHarmonics
        module procedure RealSphericalHarmonics_init
    end interface

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!      Real-valued spherical harmonics in cube                     %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    type, extends(RealSphericalHarmonics) :: RealSphericalCubeHarmonics
#ifdef HAVE_CUDA
        type(C_PTR), allocatable  :: cuda_interface
#endif
    contains
        procedure          :: eval_grid                    => RealSphericalCubeHarmonics_eval_grid
        procedure, private :: eval_grid_cpu                => RealSphericalCubeHarmonics_eval_grid_cpu
        procedure          :: destroy                      => RealSphericalCubeHarmonics_destroy
        
#ifdef HAVE_CUDA
        procedure, private :: eval_grid_cuda               => RealSphericalCubeHarmonics_eval_grid_cuda
#endif
    end type


    interface RealSphericalCubeHarmonics
        module procedure RealSphericalCubeHarmonics_init
    end interface

#ifdef HAVE_CUDA
    interface
         type(C_PTR) function RealSphericalCubeHarmonics_init_cuda(lmin, &
                 lmax, normalization, shape, streamContainer) bind(C)
             use ISO_C_BINDING
             integer(C_INT), value :: lmin
             integer(C_INT), value :: lmax
             integer(C_INT), value :: normalization
             integer(C_INT)        :: shape(3)
             type(C_PTR), value    :: streamContainer
         end function
     end interface

    interface
         subroutine RealSphericalCubeHarmonics_destroy_cuda( &
                 harmonics) bind(C)
             use ISO_C_BINDING
             type(C_PTR), value    :: harmonics
         end subroutine
     end interface

    
    interface
         subroutine RealSphericalCubeHarmonics_evaluate_cuda( &
                 harmonics, grid, center) bind(C)
             use ISO_C_BINDING
             type(C_PTR), value    :: harmonics
             type(C_PTR), value    :: grid
             real(C_DOUBLE)        :: center(3)
         end subroutine
     end interface

    interface 
         subroutine RealSphericalCubeHarmonics_download_cuda( &
                 harmonics, host_results, host_results_shape) bind(C)
             use ISO_C_BINDING
             type(C_PTR), value    :: harmonics
             real(C_DOUBLE)        :: host_results(*)
             integer(C_INT)        :: host_results_shape(4)
         end subroutine
     end interface

     interface 
         subroutine  RealCubeHarmonics_register_result_array_cuda( &
                 harmonics, host_results, host_results_shape) bind(C)
             use ISO_C_BINDING
             type(C_PTR), value    :: harmonics
             real(C_DOUBLE)        :: host_results(*)
             integer(C_INT)        :: host_results_shape(4)
         end subroutine
     end interface

    interface 
         subroutine RealCubeHarmonics_unregister_result_array_cuda( &
                 harmonics, host_results) bind(C)
             use ISO_C_BINDING
             type(C_PTR), value    :: harmonics
             real(C_DOUBLE)        :: host_results(*)
         end subroutine
     end interface
     
#endif

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!      Complex-valued spherical harmonics (Racah's normalization)     %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    type :: ComplexSphericalHarmonics
        private
        !> Real Spherical Harmonics - Object to help evaluate the values of complex spherical harmonics
        !! as the real spherical harmonics can be used to calculate the complex spherical harmonics and the
        !! calculation of real spherical harmonics is fast. 
        type(RealSphericalHarmonics) :: real_spherical_harmonics
    contains
        procedure :: eval        => ComplexSphericalHarmonics_eval
        procedure :: destroy     => ComplexSphericalHarmonics_destroy
    end type


    interface ComplexSphericalHarmonics
        module procedure ComplexSphericalHarmonics_init
    end interface

contains

! %%%%%%%%%%%%%%%%%%%%%% Poly3D %%%%%%%%%%%%%%%%%%%%%%%%%%
    subroutine Poly3D_print(self)
        class(Poly3D), intent(in) :: self
        integer :: i

        do i=1,size(self%coeffs)
            print'("[",3i3,"]",e12.5)',self%expos(:,i),self%coeffs(i)
        end do
        print*,repeat("-",20)
    end subroutine

    subroutine Poly3D_array_print(self)
        class(Poly3D_array), intent(in) :: self

        integer :: m

        do m=lbound(self%m,1), ubound(self%m,1)
            call self%m(m)%print()
        end do
        print*,repeat("=",20)
    end subroutine

    pure subroutine Poly3D_array_destroy(self)
        class(Poly3D_array), intent(inout) :: self

        integer :: m

        do m=lbound(self%m,1), ubound(self%m,1)
            call self%m(m)%destroy()
        end do
        deallocate(self%m)
    end subroutine

    pure function Poly3D_init_empty(length) result(new)
        integer, intent(in) :: length
        type(Poly3D)        :: new

        allocate(new%expos(3,length))
        allocate(new%coeffs(length))
        new%expos=0
        new%coeffs=0.d0
    end function

    pure function Poly3D_init_explicit(expos, coeffs) result(new)
        type(Poly3D)               :: new
        integer,      intent(in)   :: expos(:,:)
        real(REAL64), intent(in)   :: coeffs(:)

        if(size(expos,1)==3 .and. size(expos,2)==size(coeffs)) then
            new%expos=expos
            new%coeffs=coeffs
        end if
    end function

    pure subroutine Poly3D_destroy(self)
        class(Poly3D), intent(inout) :: self

        if (allocated(self%expos))  deallocate(self%expos)
        if (allocated(self%coeffs)) deallocate(self%coeffs)
    end subroutine

    !> Poly3D addition.

    !> Adds two Poly3D to return another one, preserving lexicographical order.
    !! All monomials with coefficient below 1.d-15 are expunged.
    pure function Poly3D_plus_Poly3D(p1, p2) result(p_result)
        class(Poly3D), intent(in) :: p1, p2
        type (Poly3D)             :: p_out, p_result

        integer :: i1, i2, i_out
        integer :: l1, l2

        l1=size(p1%coeffs)
        l2=size(p2%coeffs)
        p_out=Poly3D(l1+l2)
        i1=1
        i2=1
        i_out=1
        do i1=1, l1
            ! Insert as many elements from p2 as we can
            if (i2 <= l2 .and. expos_gt(p1%expos(:,i1), p2%expos(:,i2)) ) then
                do i2=i2,l2
                    if (expos_lt(p2%expos(:,i2), p1%expos(:,i1)) ) then
                        p_out%expos(:,i_out)=p2%expos(:,i2)
                        p_out%coeffs (i_out)=p2%coeffs (i2)
                        if(abs(p_out%coeffs (i_out))>1.d-15) i_out=i_out+1
                    else
                        exit
                    end if
                end do
            end if

            ! If equal expos, add the coefficients and append
            if (i2 <= l2 ) then
                if (all(p1%expos(:,i1) == p2%expos(:,i2))) then
                    p_out%expos(:,i_out)=p1%expos(:,i1)
                    p_out%coeffs (i_out)=p1%coeffs (i1)+p2%coeffs (i2)
                    if(abs(p_out%coeffs (i_out))>1.d-15) i_out=i_out+1
                    i2=i2+1
                else
                    p_out%expos(:,i_out)=p1%expos(:,i1)
                    p_out%coeffs (i_out)=p1%coeffs (i1)
                    if(abs(p_out%coeffs (i_out))>1.d-15) i_out=i_out+1
                end if
            ! Append monomials from p1
            else
                p_out%expos(:,i_out)=p1%expos(:,i1)
                p_out%coeffs (i_out)=p1%coeffs (i1)
                if(abs(p_out%coeffs (i_out))>1.d-15) i_out=i_out+1
            end if
        end do
        ! Append the rest of p2
        do i2=i2,l2
            p_out%expos(:,i_out)=p2%expos(:,i2)
            p_out%coeffs (i_out)=p2%coeffs (i2)
            if(abs(p_out%coeffs (i_out))>1.d-15) i_out=i_out+1
        end do

        p_result=Poly3D(expos =p_out%expos(:,:i_out-1),&
                        coeffs=p_out%coeffs (:i_out-1))
        call p_out%destroy()
        return
    end function

    pure function Poly3D_minus_Poly3D(p1, p2) result(p_out)
        class(Poly3D), intent(in) :: p1, p2
        type (Poly3D)             :: p_out

        p_out = p1 + ( (-1.d0) * p2 )
    end function

    !> Poly3D multiplication.

    !> Multiply two Poly3D to return another one, preserving lexicographical
    !! order. All monomials with coefficient below 1.d-15 are expunged.
    pure function Poly3D_times_Poly3D(p1, p2) result(p_out)
        class(Poly3D), intent(in) :: p1
        class(Poly3D), intent(in) :: p2
        type (Poly3D)             :: p_out

        type(Poly3D)              :: p_tmp, p_tmp2
        integer :: i

        p_out=Poly3D(1)

        ! Add, one by one, the product of p1 with each one of the monomials of
        ! p2.
        do i=1,size(p2%coeffs)
            p_tmp=Poly3D(p1%expos, p1%coeffs)
            ! Multiply monomials == Add exponents
            p_tmp%expos =p_tmp%expos +spread(p2%expos(:,i),2,size(p1%coeffs))
            ! Multiply coefficients
            p_tmp%coeffs=p_tmp%coeffs*p2%coeffs(i)
            ! Add to output
            p_tmp2=p_out+p_tmp
            call p_tmp%destroy()
            call p_out%destroy()
            p_out = p_tmp2
            call p_tmp2%destroy()
        end do
    end function

    pure function Poly3D_times_REAL64(p, r) result(p_out)
        class(Poly3D), intent(in) :: p
        real(REAL64),  intent(in) :: r
        type (Poly3D)             :: p_out

        p_out=Poly3D(expos =p%expos,&
                     coeffs=p%coeffs * r)
    end function

    !> get a derivative of Poly3D object to direction of 'direction'nth component of polynomial
    !! and returns the result as a new Poly3D. 
    !! 
    !! NOTE: this function assumes that the components of the polynomial are independent of
    !!       each other (for example cartesian coordinates)
    function Poly3D_get_derivative(self, direction) result(derivative)
        class(Poly3D), intent(in) :: self
        integer,       intent(in) :: direction
        type (Poly3D)             :: derivative
        real(REAL64)              :: new_coefficients(size(self%coeffs))
        integer                   :: new_exponentials(size(self%expos, 1), &
                                                      size(self%expos, 2)), i, counter
        counter = 0
        do i = 1, size(self%coeffs)
            ! if the exponential of the direction is 0, then we get zero monomial, which
            ! is not added to the result polynomial
            if (self%expos(direction, i) > 0) then
                counter = counter + 1
                new_coefficients(counter) = self%coeffs(i) * self%expos(direction, i)
                new_exponentials(:, counter) = self%expos(:, i)
                new_exponentials(direction, counter) = self%expos(direction, i) - 1 
            end if
        end do

        ! if there are no terms in the new one, let's make it to be zero in order to
        ! have something to evaluate
        if (counter == 0) then
            new_exponentials(:, 1) = [0, 0, 0]
            new_coefficients(1) = 0.0d0
            counter = 1
        end if

        derivative=Poly3D(expos  = new_exponentials(:, : counter),&
                          coeffs = new_coefficients(: counter))
    end function

    pure function REAL64_times_Poly3D(r, p) result(p_out)
        real(REAL64),  intent(in) :: r
        class(Poly3D), intent(in) :: p
        type (Poly3D)             :: p_out

        p_out=p * r
    end function

    !> Lesser than (<) comparison of integer triplets in lexicographical order
    !! (cannot overload operator for integer arrays).
    pure function expos_lt(expos1, expos2)
        integer, intent(in) :: expos1(3), expos2(3)
        logical             :: expos_lt

        expos_lt = (expos1(1) <  expos2(1)) .or. &
                   ( (expos1(1) == expos2(1)) .and. &
                     ( (expos1(2) <  expos2(2)) .or. &
                       ( (expos1(2) == expos2(2)) .and. &
                         ( (expos1(3) <  expos2(3)) ) ) ) )
        return
    end function

    !> Greater than (>) comparison of integer triplets in lexicographical order
    !! (cannot overload operator for integer arrays).
    pure function expos_gt(expos1, expos2)
        integer, intent(in) :: expos1(3), expos2(3)
        logical             :: expos_gt

        expos_gt = (expos1(1) >  expos2(1)) .or. &
                   ( (expos1(1) == expos2(1)) .and. &
                     ( (expos1(2) >  expos2(2)) .or. &
                       ( (expos1(2) == expos2(2)) .and. &
                         ( (expos1(3) >  expos2(3)) ) ) ) )
        return
    end function

! %%%%%%%%%%%%%%%%%%%%%% Spherical harmonics generation %%%%%%%%%%%%%%%%%%%%%%%
    !> Generate spherical harmonics as an array of Poly3D arrays using the
    !! recursion relations of Helgaker, Jørgensen and Olsen (ref. missing).

    !> This function is not for the end user
    function spherical_harmonics_poly3d(lmax) result(harmo)
        integer,            intent(in)   :: lmax
        type(Poly3D_array)               :: harmo(0: lmax)

        integer :: l,m

        type(Poly3D) :: s, temp, temp2, temp3
        type(Poly3D) :: x
        type(Poly3D) :: y
        type(Poly3D) :: z
        type(Poly3D) :: r2

        x=Poly3D(expos =reshape([1,0,0],[3,1]), coeffs=[1.d0])
        y=Poly3D(expos =reshape([0,1,0],[3,1]), coeffs=[1.d0])
        z=Poly3D(expos =reshape([0,0,1],[3,1]), coeffs=[1.d0])
        s=Poly3D(expos =reshape([0,0,0],[3,1]), coeffs=[1.d0])
        
        r2=x*x+y*y+z*z

        if(lmax<0) then
            call perror("lmax must be >=0")
            stop
        end if

        ! #### L=0 ####
        l=0
        allocate(harmo(l)%m(-l:l))
        harmo(0)%m(0)=s

        if(lmax==0) return
        ! #### L=1 ####
        l=1
        allocate(harmo(l)%m(-l:l))
        harmo(l)%m(:)=[y,z,x]

        ! #### L>=2 ####
        do l=2,lmax
            temp = y*harmo(l-1)%m(l-1)
            temp2 = x*harmo(l-1)%m(-(l-1))
            temp3 = temp + temp2
            allocate(harmo(l)%m(-l:l))
            harmo(l)%m(-l)=sqrt((2*l-1.d0)/(2.d0*l)) * temp3
            call temp%destroy()
            call temp2%destroy()
            call temp3%destroy()
            do m=-(l-1), l-1
                temp = z*harmo(l-1)%m(m)
                harmo(l)%m(m)=(2*l-1.d0) / sqrt( 1.d0*(l+m)*(l-m) ) * temp
                call temp%destroy()
                if(l-abs(m)>1) then 
                    temp = r2*harmo(l-2)%m(m)
                    temp2 = ( (sqrt( (l+m-1.d0)*(l-m-1.d0) / ( (l+m)*(l-m) )) ) * temp)
                    harmo(l)%m(m)=harmo(l)%m(m)- temp2
                        
                    call temp%destroy()
                    call temp2%destroy()
                end if
            end do
            temp = x*harmo(l-1)%m(l-1)
            temp2 = y*harmo(l-1)%m(-(l-1))
            temp3 = (temp - temp2)
            harmo(l)%m(l)=sqrt((2*l-1.d0)/(2.d0*l)) * &
                      temp3
            call temp%destroy()
            call temp2%destroy()
            call temp3%destroy()
        end do
        call s%destroy()
        call y%destroy()
        call z%destroy()
        call x%destroy()
        call r2%destroy()
    end function

    !> Get the derivative polynomials from regular polynomials
    function spherical_harmonics_poly3d_derivatives(harmo, lmax) result(derivative_polynomials)
        type(Poly3D_array), intent(in)   :: harmo(0:lmax)
        integer,            intent(in)   :: lmax
        type(Poly3D_array)               :: derivative_polynomials(0 : lmax, 3)

        integer                          :: l, m
    
        do l = 0, lmax
            allocate(derivative_polynomials(l, X_)%m(-l:l))
            allocate(derivative_polynomials(l, Y_)%m(-l:l))
            allocate(derivative_polynomials(l, Z_)%m(-l:l))
            do m = -l, l
                derivative_polynomials(l, X_)%m(m) = harmo(l)%m(m)%get_derivative(X_)
                derivative_polynomials(l, Y_)%m(m) = harmo(l)%m(m)%get_derivative(Y_)
                derivative_polynomials(l, Z_)%m(m) = harmo(l)%m(m)%get_derivative(Z_)
            end do
        end do
    end function


! %%%%%%%%%%%%%%%%%%%%%%% RealRegularSolidHarmonics %%%%%%%%%%%%%%%%%%%%%

    !> Constructor
    function RealRegularSolidHarmonics_init(lmax, lmin, normalization, init_derivatives) result(new)
        !> Maximum angular momentum quantum number l generated
        integer, intent(in)              :: lmax
        !> Minimum angular momentum quantum number l generated
        integer, intent(in), optional    :: lmin
        !> Type of normalization (1=racah's normalization (default), 2=conventional normalization)
        integer, intent(in), optional    :: normalization
        !> If the capability to evaluate derivatives is initialized (default: .FALSE.)
        logical, intent(in), optional    :: init_derivatives
        type(RealRegularSolidHarmonics)  :: new
        integer                          :: nt_tot, multiplicity, i, l, m
        logical                          :: init_derivatives_

        new%lmax = lmax

        if (present(lmin)) then
            new%lmin = lmin
        else
            new%lmin = 0
        end if

        if (present(init_derivatives)) then
            init_derivatives_ = init_derivatives
        else
            init_derivatives_ = .FALSE.
        end if

        if (present(normalization)) then
            new%normalization = normalization
        end if 
        if (init_derivatives_) then
            allocate(new%polynomials(0 : new%lmax, 4))
            new%polynomials(:, 1)   = spherical_harmonics_poly3d(new%lmax)
            new%polynomials(:, 2:4) = spherical_harmonics_poly3d_derivatives(new%polynomials(:, 1), new%lmax)
            multiplicity = 4
        else
            allocate(new%polynomials(0 : new%lmax, 1))
            new%polynomials(:, 1)   = spherical_harmonics_poly3d(new%lmax)
            multiplicity = 1
        end if
    
        allocate(new%number_of_terms((new%lmax+1)*(new%lmax+2)*(new%lmax+3)/6, multiplicity))
        new%number_of_terms(1, :)=0
        nt_tot=sum( &
            [( sum( [( size(new%polynomials(l, 1)%m(m)%coeffs), m=-l,l )] ), &
                                               l=new%lmin,new%lmax )]  )

        allocate(new%k(multiplicity), source = 0)
        allocate(new%ijk(multiplicity), source = 1)
        allocate(new%l(nt_tot, multiplicity))
        allocate(new%m(nt_tot, multiplicity))
        allocate(new%lm(nt_tot, multiplicity))
        allocate(new%coefficients(nt_tot, multiplicity))
        
        do i = 1, multiplicity
            call new%convert_poly3d_array_to_internal(i)
        end do
        
#ifdef HAVE_CUDA
        call new%init_cuda()
#endif
    end function

    subroutine RealRegularSolidHarmonics_convert_poly3d_array_to_internal(self, order_number)
        class(RealRegularSolidHarmonics), intent(inout) :: self
        integer,                          intent(in)    :: order_number
        integer                                         :: l, m, ijk, i, k, kappa(3)

        type(CartIter)                                  :: cart
        logical                                         :: continue_iteration
        k=0
        ijk=0
        cart=CartIter(ndim=3, maximum_modulus=self%lmax)
        call cart%next(kappa, continue_iteration)

        ! loop over all monomials from [i = 0, j = 0, k = 0] to 
        ! all [i, j, k] where i+j+k <= lmax
        do while(continue_iteration)
            ijk=ijk+1

            ! initialize the number of terms as the number of
            ! terms in the previous i, j, k value
            if(ijk>1) self%number_of_terms(ijk, order_number)=self%number_of_terms(ijk-1, order_number)

            ! get the angular momentum l = i + j + k
            l=sum(kappa)
            if (order_number > 1) l = l +1
            if (l > self%lmax) continue
            ! evaluate only 
            if (l >= self%lmin .and. l <= self%lmax) then
                ! loop over all m values
                do m=-l,l
                    ! loop over all terms of the value
                    do i=1,size(self%polynomials(l, order_number)%m(m)%coeffs)
                        ! check if the exponentials of the term match with the monomial 
                        ! integers i, j, and k
                        if(all(kappa==self%polynomials(l, order_number)%m(m)%expos(:,i))) then
                            ! if this is the case store the term
                            ! k is the total number of monomial evaluations
                            ! in the entire array
                            k=k+1

                            ! add the 'number of terms that contain ijk monomial' counter by one
                            self%number_of_terms(ijk, order_number)=self%number_of_terms(ijk, order_number)+1
                            
                            ! coeffs in 'c'
                            self%coefficients(k, order_number)=self%polynomials(l, order_number)%m(m)%coeffs(i)

                            ! multiply the coefficient with normalization factor if we are using
                            ! the conventional normalization
                            if (self%normalization == 2) then
                                self%coefficients(k, order_number) = self%coefficients(k, order_number) * sqrt((2*l+1)/dble(4*pi))
                            end if 

                            ! also store the l, m angular quantum numbers
                            self%l(k, order_number)=l
                            self%m(k, order_number)=m
                            self%lm(k, order_number)=lm_map(l, m) 
                            exit
                        end if
                    end do
                end do
            end if
            call cart%next(kappa, continue_iteration)
        end do
    end subroutine

#ifdef HAVE_CUDA
    subroutine RealRegularSolidHarmonics_init_cuda(self)
        class(RealRegularSolidHarmonics), intent(inout) :: self
        integer(C_INT), pointer                         :: number_of_terms(:)
        integer(C_INT), pointer                         :: exponentials(:,:)
        real(C_DOUBLE), pointer                         :: coefficients(:)

        call self%assign_c_pointers(number_of_terms, exponentials, coefficients)
        !self%cuda_interface = RealRegularSolidHarmonics_cuda_init(self%lmin, self%lmax, &
        !    number_of_terms, self%normalization, exponentials, coefficients, self%number_of_terms,  &
        !    self%coefficients, self%lm, size(self%number_of_terms)) 

        deallocate(number_of_terms, exponentials, coefficients)
    end subroutine

    !> Assign C-pointers used in CUDA-calculations
    !! @TODO: Change the structure in CUDA-evaluation to be like the Serial-implementation
    !! This would save a lot of execution time.
    pure subroutine RealRegularSolidHarmonics_assign_c_pointers(self, number_of_terms, &
                                                                exponentials, coefficients)
        class(RealRegularSolidHarmonics), intent(in) :: self
        integer(C_INT), intent(out),   pointer  :: number_of_terms(:), exponentials(:,:)
        real(C_DOUBLE), intent(out),   pointer  :: coefficients(:)

        integer :: k, kt, i, j, l, m


        ! calculate the ending index in the coefficient, exponential arrays
        ! related to each l,m pair
        allocate(number_of_terms((self%lmax+1)**2))
        number_of_terms(1) = size(self%polynomials(0, 1)%m(0)%coeffs)
        j = 2
        do l=1,self%lmax
            do m = -l, l
                number_of_terms(j) = number_of_terms(j-1) + size(self%polynomials(l, 1)%m(m)%coeffs)
                j = j + 1
            end do
        end do

        ! get the total number of terms evaluated
        kt = number_of_terms((self%lmax+1)**2)

        ! allocate the arrays
        allocate(coefficients(kt))
        allocate(exponentials(3,kt))

        ! populate the predefined arrays
        j=1
        do l = 0, self%lmax
            do m = -l, l
                do i=1,size(self%polynomials(l, 1)%m(m)%coeffs)
                    coefficients(j)   = self%polynomials(l, 1)%m(m)%coeffs(i)
                    exponentials(:,j) = self%polynomials(l, 1)%m(m)%expos(:, i)
                    j=j+1
                end do
            end do 
        end do
        
    end subroutine

    function RealRegularSolidHarmonics_eval_grid_cuda(self, grid, center) result(result_cubes)
        class(RealRegularSolidHarmonics), intent(in)    :: self
        type(Grid3D),                     intent(inout) :: grid
        real(REAL64),                     intent(in)    :: center(3)
        type(C_PTR)                                     :: cuda_grid
        real(REAL64)                                    :: result_cubes(self%lmin**2 + 1 : (self%lmax+1)**2, &
                                                                         grid%axis(X_)%get_shape(), &
                                                                         grid%axis(Y_)%get_shape(), &
                                                                         grid%axis(Z_)%get_shape())                                 
        

        ! initialize the cuda grid
        !call RealRegularSolidHarmonics_cuda_evaluate_grid(self%cuda_interface, &
        !        grid%get_cuda_interface(), center, result_cubes)
    end function

    pure function RealRegularSolidHarmonics_get_cuda_interface(self) result(cuda_interface)
        class(RealRegularSolidHarmonics), intent(in)    :: self
        type(C_PTR)                                     :: cuda_interface
        
        cuda_interface = self%cuda_interface
    end function
    
#endif

    !> Reset counters for iter_lm.
    subroutine RealRegularSolidHarmonics_reset(self)
        class(RealRegularSolidHarmonics) :: self

        self%ijk(:)=1
        self%k(:)=0
    end subroutine

    !> Return the next lm term for the current ijk triplet, or setup the
    !! iterator for the next ijk triplet. Basis for
    !! [eval](@ref realsphericalharmonics_eval) and
    !! [cart2sph](@ref realsphericalharmonics_cart2sph).

    !> Use in combination with CartIter:
    !! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    !! iter=CartIter(3,lmax)
    !! harmo=RealRegularSolidHarmonics(lmax)
    !!
    !! do while(iter%next(kappa))
    !!     do while(harmo%next_lm(l,m,c))
    !!         ... Do something with l, m, kappa and c ...
    !!     end do
    !! end do
    !! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    !! Note: Use in any other situation is useless
    pure subroutine RealRegularSolidHarmonics_next_lm(self, l, m, c, continue, order_number)
        class(RealRegularSolidHarmonics), intent(inout) :: self
        !> Angular quantum number l will be placed to this variable
        integer,                       intent(out)   :: l
        !> Angular quantum number m will be placed to this variable
        integer,                       intent(out)   :: m
        !> The coefficient for the current monomial [i, j, k] for the 
        !! pair of quantum numbers l, m will be put to this variable 
        real(REAL64),                  intent(out)   :: c
        !> Is there any quantum number pairs for the current monomial [i, j, k]
        logical,                       intent(out)   :: continue
        !> Angular quantum number m will be placed to this variable
        integer,                       intent(in)    :: order_number

        call self%next_lm_(l, m, c, continue, order_number, self%k, self%ijk)
    end subroutine

    !> Return the next lm term for the current ijk triplet, or setup the
    !! iterator for the next ijk triplet. Basis for
    !! [eval](@ref realsphericalharmonics_eval) and
    !! [cart2sph](@ref realsphericalharmonics_cart2sph).

    !> Use in combination with CartIter:
    !! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    !! iter=CartIter(3,lmax)
    !! harmo=RealRegularSolidHarmonics(lmax)
    !!
    !! do while(iter%next(kappa))
    !!     do while(harmo%next_lm(l,m,c))
    !!         ... Do something with l, m, kappa and c ...
    !!     end do
    !! end do
    !! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    !! Note: Use in any other situation is useless
    pure subroutine RealRegularSolidHarmonics_next_lm_(self, l, m, c, continue, order_number, k, ijk)
        class(RealRegularSolidHarmonics), intent(in) :: self
        !> Angular quantum number l will be placed to this variable
        integer,                       intent(out)   :: l
        !> Angular quantum number m will be placed to this variable
        integer,                       intent(out)   :: m
        !> The coefficient for the current monomial [i, j, k] for the 
        !! pair of quantum numbers l, m will be put to this variable 
        real(REAL64),                  intent(out)   :: c
        !> Is there any quantum number pairs for the current monomial [i, j, k]
        logical,                       intent(out)   :: continue
        !> Angular quantum number m will be placed to this variable
        integer,                       intent(in)    :: order_number
        !> The k-trackers
        integer,                       intent(inout) :: k(:)
        !> The monomial tracker
        integer,                       intent(inout) :: ijk(:)

        ! Check if there are any l,m pairs that have contribution from
        ! the current monomial [i, j, k]
        continue = k(order_number) < self%number_of_terms(ijk(order_number), order_number)
        ! If yes, fill the values for l, m, and c and return TRUE
        if(continue) then
            k(order_number)=k(order_number)+1
            l=self%l(k(order_number), order_number)
            m=self%m(k(order_number), order_number)
            c=self%coefficients(k(order_number), order_number)
        ! If not, get ready for the next monomial and return FALSE  
        else
            ijk(order_number)=ijk(order_number)+1
        end if
        return
    end subroutine

    !> Evaluate all spherical harmonics at the list of points `pos`.

    !> Returns array of shape `(<number of points>, (lmax+1)^2)`. The cost of
    !! the call is proportional to \f$N(L+1)^{3.5}\f$.
    function RealRegularSolidHarmonics_eval(self, pos) result(res)
        class(RealRegularSolidHarmonics)         :: self
        !> List of points. Shape is `(3, <number of points>)`.
        real(REAL64),              intent(in)    :: pos(:,:)
        real(REAL64)                             :: res( size(pos,2),&
                                                       (self%lmin**2 + 1) : (self%lmax+1)**2 )
        type(CartIter)                           :: iter
        real(REAL64),   allocatable              :: tmp(:, :)
        integer                                  :: kappa(3)
        integer                                  :: flipdim
        integer                                  :: axis
        integer                                  :: l, m, first_point, last_point
        real(REAL64)                             :: c

        integer                                  :: i,j, id
        logical                                  :: continue_iteration, continue_lm_iteration, omp_parallelize
        integer,       allocatable               :: k(:), ijk(:)

        call self%reset()
        
        
        res=0.d0
#ifdef HAVE_OMP
        omp_parallelize = size(pos,2) > 1000
        !$OMP PARALLEL IF(size(pos,2) > 1000) &
        !$OMP& PRIVATE(first_point, last_point, iter, tmp, continue_iteration) &
        !$OMP& PRIVATE(continue_lm_iteration, axis, id, l, m, c, kappa, k, ijk, flipdim) 
        first_point = size(pos,2) / omp_get_num_threads() * (omp_get_thread_num()) + &
                      min(omp_get_thread_num()+1, mod(size(pos,2), omp_get_num_threads()))
        last_point = size(pos,2) / omp_get_num_threads() * (omp_get_thread_num()+1) + &
                      min(omp_get_thread_num()+2, mod(size(pos,2), omp_get_num_threads())) - 1
        if (omp_get_thread_num()+1 == omp_get_num_threads()) last_point = last_point +1
        
        ! if the openmp parallelization is not u
        if (size(pos,2) <= 1000) then
            first_point = 1
            last_point = size(pos, 2)
        end if
#else
        first_point = 1
        last_point =  size(pos,2)
#endif
        allocate(tmp(last_point-first_point + 1, 3), source = 1.0d0)
        iter = CartIter(3, self%lmax)
        k = self%k
        ijk = self%ijk
        call iter%next(kappa, continue_iteration, flipdim)
        do while(continue_iteration)
            ! multiply tmp with the flipdim component of changed axis
            ! so that tmp's x component is x^i, y component is y^j and
            ! z component is z^k, where i, j, and k are kappa(1), kappa(2)
            ! and kappa(3), respectively. x, y, and z are pos(1), pos(2) and
            ! pos(3)  
            if(.not.all(kappa==0)) &
                tmp(:, flipdim) = tmp(:,flipdim)*pos(flipdim, first_point:last_point)

            forall (axis = flipdim+1 : Z_)
                tmp(:,axis)=tmp(:,flipdim)
            end forall

            call self%next_lm_(l, m, c, continue_lm_iteration, 1, k, ijk)
            do while(continue_lm_iteration)
                id = lm_map(l,m)
                res(first_point:last_point, id) = res(first_point:last_point, id)+ c*tmp(:, Z_)
                call self%next_lm_(l, m, c, continue_lm_iteration, 1, k, ijk)
            end do
            call iter%next(kappa, continue_iteration, flipdim)
        end do
        deallocate(tmp, k, ijk)
#ifdef HAVE_OMP
        !$OMP END PARALLEL
#endif
    end function

    !> Evaluate gradient of regular solid harmonics at the list of points `pos`.

    !> Returns array of shape `(<number of points>, (lmax+1)^2, 3)`. The cost of
    !! the call is proportional to \f$N(L+1)^{3.5}\f$.
    function RealRegularSolidHarmonics_evaluate_gradients(self, pos) result(res)
        class(RealRegularSolidHarmonics)         :: self
        !> List of points. Shape is `(3, <number of points>)`.
        real(REAL64),              intent(in)    :: pos(:,:)
        real(REAL64)                             :: res( size(pos,2),&
                                                       (self%lmin**2 + 1) : (self%lmax+1)**2, 3)
        type(CartIter) :: iter
        real(REAL64)   :: tmp( size(pos,2), 3)
        integer        :: kappa(3)
        integer        :: flipdim
        integer        :: axis
        integer        :: l, m
        real(REAL64)   :: c

        integer :: i,j,k, id
        logical :: continue_iteration, continue_lm_iteration

        call self%reset()
        
        
        res=0.d0
        tmp =1.d0
        iter=CartIter(3,self%lmax)
        
        call iter%next(kappa, continue_iteration, flipdim)
        do while(continue_iteration)
            ! multiply tmp with the flipdim component of changed axis
            ! so that tmp's x component is x^i, y component is y^j and
            ! z component is z^k, where i, j, and k are kappa(1), kappa(2)
            ! and kappa(3), respectively. x, y, and z are pos(1), pos(2) and
            ! pos(3)  
            if(.not.all(kappa==0)) &
                tmp(:,flipdim)=tmp(:,flipdim)*pos(flipdim,:)

            forall (axis = flipdim+1 : Z_)
                tmp(:,axis)=tmp(:,flipdim)
            end forall
            
            call self%next_lm(l, m, c, continue_lm_iteration, 2)
            do while(continue_lm_iteration)
                id = lm_map(l,m)
                res(:, id, 1) = res(:, id, 1)+ c*tmp(:,Z_)
                call self%next_lm(l, m, c, continue_lm_iteration, 2)
            end do
            
            call self%next_lm(l, m, c, continue_lm_iteration, 3)
            do while(continue_lm_iteration)
                id = lm_map(l,m)
                res(:, id, 2) = res(:, id, 2)+ c*tmp(:,Z_)
                call self%next_lm(l, m, c, continue_lm_iteration, 3)
            end do
            
            call self%next_lm(l, m, c, continue_lm_iteration, 4)
            do while(continue_lm_iteration)
                id = lm_map(l,m)
                res(:, id, 3) = res(:, id, 3)+ c*tmp(:,Z_)
                call self%next_lm(l, m, c, continue_lm_iteration, 4)
            end do
            call iter%next(kappa, continue_iteration, flipdim)
        end do
    end function

    function RealRegularSolidHarmonics_eval_grid(self, grid, center) result(reslt)
        class(RealRegularSolidHarmonics)   :: self
        type(Grid3D), intent(in)           :: grid
        real(REAL64), intent(in)           :: center(3)
         
        real(REAL64)                       :: reslt(        grid%axis(X_)%get_shape(), &
                                                            grid%axis(Y_)%get_shape(), &
                                                            grid%axis(Z_)%get_shape(), &
                                                            self%lmin**2 + 1 : (self%lmax+1)**2 )
        real(REAL64)                       :: cube_pos(  3, grid%axis(X_)%get_shape(), &
                                                            grid%axis(Y_)%get_shape(), &
                                                            grid%axis(Z_)%get_shape()) 
        integer                            :: q, p, o, lm

! #ifdef HAVE_CUDA
!         real(REAL64)                       :: result_cubes(self%lmin**2 + 1 : (self%lmax+1)**2, &
!                                                             grid%axis(X_)%get_shape(), &
!                                                             grid%axis(Y_)%get_shape(), &
!                                                             grid%axis(Z_)%get_shape())  
!         type(Grid3D)                       :: temp_grid
! 
!         temp_grid = grid
!        
!         
!         result_cubes = self%eval_grid_cuda(temp_grid, center)
!           
!         ! switch the result array ordering
!         forall (lm = self%lmin**2 + 1 : (self%lmax+1)**2)
!             reslt(:, :, :, lm) = result_cubes(lm, :, :, :) 
!         end forall
! 
!         call temp_grid%destroy()
!         
!         
! #else
        real(REAL64)                       :: gridpoints_x(grid%axis(X_)%get_shape()), &
                                              gridpoints_y(grid%axis(Y_)%get_shape()), &
                                              gridpoints_z(grid%axis(Z_)%get_shape())

        gridpoints_x = grid%axis(X_)%get_coord() - center(X_)
        gridpoints_y = grid%axis(Y_)%get_coord() - center(Y_)
        gridpoints_z = grid%axis(Z_)%get_coord() - center(Z_)

        ! evaluate kappa r and positions relative to center of the farfield box for each grid point
        forall(o = 1 : grid%axis(X_)%get_shape(), p = 1 : grid%axis(Y_)%get_shape(), q = 1 : grid%axis(Z_)%get_shape())
            cube_pos(X_, o, p, q) = gridpoints_x(o)
            cube_pos(Y_, o, p, q) = gridpoints_y(p)
            cube_pos(Z_, o, p, q) = gridpoints_z(q)
        end forall  

        ! evaluate spherical harmonics at each cube point and reshape the result to be of desired shape
        reslt = reshape( &
                    self%eval(reshape( cube_pos, [3, size(gridpoints_z) * size(gridpoints_y) * size(gridpoints_x)] ) ), &
                    [size(gridpoints_x), size(gridpoints_y), size(gridpoints_z), (self%lmax+1)**2 - (self%lmin)**2] &
                ) 
!#endif
    end function

    !> Rotate "Cartesian vector" into "Spherical vector".
    function RealRegularSolidHarmonics_cart2sph(self, cart) result(sph)
        class(RealRegularSolidHarmonics)         :: self
        real(REAL64),              intent(in) :: cart(:)
        real(REAL64)                          :: sph( (self%lmax+1)**2 )
        type(CartIter) :: iter
        integer        :: kappa(3)
        integer        :: l, m, ijk
        real(REAL64)   :: c

        integer :: i,j,k
        logical :: continue_iteration, continue_lm_iteration

        call self%reset()

        sph=0.d0
        iter=CartIter(3,self%lmax)
        ijk=0
        call iter%next(kappa, continue_iteration)
        do while(continue_iteration)
            ijk=ijk+1
            call self%next_lm(l, m, c, continue_lm_iteration, 1)
            do while(continue_lm_iteration)
                sph(lm_map(l,m))=sph(lm_map(l,m))+c*cart(ijk)
                call self%next_lm(l, m, c, continue_lm_iteration, 1)
            end do
            call iter%next(kappa, continue_iteration)
        end do
    end function

    !> Pretty print.
    subroutine RealRegularSolidHarmonics_print(self)
        class(RealRegularSolidHarmonics) :: self

        integer        :: l, m
        integer        :: l1, m1
        integer        :: expos(3)
        integer        :: axis
        real(REAL64)   :: c
        type(CartIter) :: iter
        character      :: strs(3)=["x","y","z"]
        logical        :: continue_iteration, continue_lm_iteration

        logical :: first

        write(*,'(a)') repeat("#",42)
        do l=0,self%lmax
            write(*,'(a)') repeat("+",42)
            do m=-l,l
                write(*,'(a)') repeat("-",42)
                write(*,'(3x,"Y{",i2,",",i2,"} = ")',advance='no') l,m
                call self%reset()
                iter=CartIter(3,self%lmax)
                first=.TRUE.
            ! Due to the sparse-matrix format of RealRegularSolidHarmonics, the
            ! looping here is very awkward and inefficient. ALL terms must be
            ! checked for EACH AND EVERY spherical harmonic function...
                call iter%next(expos, continue_iteration)
                do while (continue_iteration)
                    call self%next_lm(l1,m1, c, continue_lm_iteration, 1)
                    do while(continue_lm_iteration)
                        if(l==l1 .and. m==m1) then
                            if(.not.first) then
                                write(*,*) "+"
                                write(*,'(14x)',advance='no')
                            else
                                first=.FALSE.
                            end if
                            write(*,'(e12.5)',advance='no') c
                            do axis=X_,Z_
                                if(expos(axis)>0) then
                                    write(*,'(" ",a)',advance='no') strs(axis)
                                    if(expos(axis)>1) &
                                        write(*,'("^",i0)',advance='no') &
                                            expos(axis)
                                end if
                            end do
                        end if
                        call self%next_lm(l1,m1, c, continue_lm_iteration, 1)
                    end do
                    call iter%next(expos, continue_iteration)
                end do
                write(*,*)
            end do
            write(*,'(a)') repeat("-",42)
        end do
        write(*,'(a)') repeat("+",42)
        write(*,'(a)') repeat("#",42)
    end subroutine

    subroutine RealRegularSolidHarmonics_destroy(self)
        class(RealRegularSolidHarmonics), intent(inout) :: self
        integer                                         :: l, i
        if(allocated(self%number_of_terms)) deallocate(self%number_of_terms)
        if(allocated(self%coefficients))  deallocate(self%coefficients)
        if(allocated(self%l))  deallocate(self%l)
        if(allocated(self%m))  deallocate(self%m)
        if(allocated(self%lm)) deallocate(self%lm)
        if(allocated(self%ijk)) deallocate(self%ijk)
        if(allocated(self%k)) deallocate(self%k)
        if(allocated(self%polynomials)) then
            do i = 1, size(self%polynomials, 2)
                do l = 0, self%lmax
                    call self%polynomials(l, i)%destroy()
                end do
            end do
            deallocate(self%polynomials)
        end if
#ifdef HAVE_CUDA
        !call RealRegularSolidHarmonics_cuda_destroy(self%cuda_interface)
#endif
    end subroutine

! %%%%%%%%%%%%%%%%%%%%%%% RealSphericalHarmonics %%%%%%%%%%%%%%%%%%%%%

    function RealSphericalHarmonics_init(lmax, lmin, normalization, init_derivatives) result(new)
        !> Maximum angular momentum quantum number l generated
        integer, intent(in)           :: lmax
        !> Minimum angular momentum quantum number l generated
        integer, intent(in), optional :: lmin
        !> Type of normalization (1=racah's normalization (default), 2=conventional normalization)
        integer, intent(in), optional    :: normalization
        !> If the capability to evaluate derivatives is initialized (default: .FALSE.)
        logical, intent(in), optional    :: init_derivatives
        !> Result object
        type(RealSphericalHarmonics)  :: new
      
        new%solid_harmonics = RealRegularSolidHarmonics(lmax, lmin = lmin, normalization = normalization, &
                                                              init_derivatives = init_derivatives)
    end function

    !> Evaluates values of Real Spherical Harmonics with angular momentum 
    !! quantum number 'l' values 0 - self%lmax at points given as cartesian 
    !! (x, y, z coordinates) relative to the center ('positions').
    !! 
    !! Evaluation is done using Real Solid Harmonics (defined as r^l * Y_lm).
    !! Despite the definition, the solid harmonics are easier to evaluate and
    !! thus the evaluation of RealSphericalHarmonics is done in this way.
    function RealSphericalHarmonics_eval(self, positions, distances) result(res)
        class(RealSphericalHarmonics), intent(in)    :: self
        !> List of points. Shape is `(3, <number of points>)`.
        real(REAL64),              intent(in)        :: positions(:,:)
        !> List of distances from origo. Shape is (<number of points>).
        real(REAL64), optional,    intent(in)        :: distances(size(positions, 2))
        real(REAL64)                                 :: res( size(positions,2), &
                                                             (self%solid_harmonics%lmin**2 + 1) : &
                                                             (self%solid_harmonics%lmax+1)**2 ), &
                                                        r(size(positions, 2))
        integer                                      :: i, l
        real(REAL64), parameter                      :: THRESHOLD = epsilon(0.d0)

        call bigben%split("RealSphericalHarmonics - eval ")
        if (present(distances)) then
            r = distances
        else 
#ifdef HAVE_OMP
            !$OMP PARALLEL DO
#endif
            do i = 1, size(positions, 2)
                r(i) = sqrt(sum(positions(:, i) * positions(:, i)))   
            end do
#ifdef HAVE_OMP
            !$OMP END PARALLEL DO
#endif
        end if
     
        ! evaluate solid harmonics (r^l Y_lm)
        res = self%solid_harmonics%eval(positions)

#ifdef HAVE_OMP
        !$OMP PARALLEL DO PRIVATE(l)
#endif
        ! divide res by r^l to obtain (Y_lm)
        do i = 1, size(r)
            if (r(i) > THRESHOLD) then
                forall (l = self%solid_harmonics%lmin : self%solid_harmonics%lmax)
                    res(i, l*l + 1 : l*l + 1 + 2*l) = res(i, l*l + 1 : l*l + 1 + 2*l) / (r(i) ** l)
                end forall
            end if
        end do
#ifdef HAVE_OMP
        !$OMP END PARALLEL DO
#endif
        call bigben%stop()

    end function

    !> Evaluates gradients of Real Spherical Harmonics with angular momentum 
    !! quantum number 'l' values 0 - self%lmax at points given as cartesian 
    !! (x, y, z coordinates) relative to the center ('positions').
    !! 
    !! Evaluation is done using Real Solid Harmonics (defined as r^l * Y_lm).
    !! Despite the definition, the solid harmonics are easier to evaluate and
    !! thus the evaluation of RealSphericalHarmonics is done in this way.
    function RealSphericalHarmonics_evaluate_gradients(self, positions, distances) result(res)
        class(RealSphericalHarmonics), intent(in)    :: self
        !> List of points. Shape is `(3, <number of points>)`.
        real(REAL64),              intent(in)        :: positions(:,:)
        !> List of distances from origo. Shape is (<number of points>).
        real(REAL64), optional,    intent(in)        :: distances(size(positions, 2))
        real(REAL64)                                 :: res( size(positions,2), &
                                                             (self%solid_harmonics%lmin**2 + 1) : &
                                                             (self%solid_harmonics%lmax+1)**2, 3 ), &
                                                        solid_harmonics_values(size(positions,2), &
                                                             (self%solid_harmonics%lmin**2 + 1) : &
                                                             (self%solid_harmonics%lmax+1)**2), &
                                                        r(size(positions, 2)), &
                                                        r_gradient(size(positions, 2), 3), &
                                                        pos(size(positions, 1), size(positions, 2))
        integer                                      :: i, l, offset
        real(REAL64), parameter                      :: THRESHOLD = epsilon(0.d0)


        if (present(distances)) then
            r = distances
        else 
            forall(i = 1 : size(positions, 2))
                r(i) = sqrt(sum(positions(:, i) * positions(:, i)))   
            end forall
        end if
     
        res =                    self%solid_harmonics%evaluate_gradients(pos)
        solid_harmonics_values = self%solid_harmonics%eval(pos)
        r_gradient(:, :) = 0.0d0
        do l = self%solid_harmonics%lmin, self%solid_harmonics%lmax
            r_gradient(:, X_) = -l*positions(X_, :) / (r**(l+2))
            r_gradient(:, Y_) = -l*positions(Y_, :) / (r**(l+2))
            r_gradient(:, Z_) = -l*positions(Z_, :) / (r**(l+2))
            do offset = 0, 2*l
                res(:, l*l+1+offset, X_) = res(:, l*l+1+offset, X_) / (r**l) &
                    + solid_harmonics_values(:, l*l+1+offset) * r_gradient(:, X_)  
                res(:, l*l+1+offset, Y_) = res(:, l*l+1+offset, Y_) / (r**l) &
                    + solid_harmonics_values(:, l*l+1+offset) * r_gradient(:, Y_)
                res(:, l*l+1+offset, Z_) = res(:, l*l+1+offset, Z_) / (r**l) &
                    + solid_harmonics_values(:, l*l+1+offset) * r_gradient(:, Z_)  
            end do
        end do
    end function

    function RealSphericalHarmonics_eval_grid(self, grid, center, no_cuda) result(reslt)
        class(RealSphericalHarmonics)      :: self
        type(Grid3D), intent(in)               :: grid
        real(REAL64), intent(in)               :: center(3)
        logical,      intent(in), optional     :: no_cuda
         
        real(REAL64)                           :: reslt(    grid%axis(X_)%get_shape(), &
                                                            grid%axis(Y_)%get_shape(), &
                                                            grid%axis(Z_)%get_shape(), &
                                                            self%solid_harmonics%lmin**2 + 1 : &
                                                            (self%solid_harmonics%lmax+1)**2 )

        reslt = 0.0d0
    end function

    subroutine RealSphericalHarmonics_destroy(self)
        class(RealSphericalHarmonics), intent(inout)    :: self
        call self%solid_harmonics%destroy()
    end subroutine 

! %%%%%%%%%%%%%%%%%%%%%%% RealSphericalCubeHarmonics %%%%%%%%%%%%%%%%%%%%%
    
    function RealSphericalCubeHarmonics_init(lmin, lmax, normalization, shape) result(new)
        !> Maximum angular momentum quantum number l generated
        integer, intent(in)           :: lmax
        !> Minimum angular momentum quantum number l generated
        integer, intent(in)           :: lmin
        !> Type of normalization (1=racah's normalization (default), 2=conventional normalization)
        integer, intent(in)           :: normalization
        !> Type of normalization (1=racah's normalization (default), 2=conventional normalization)
        integer, intent(in)           :: shape(3)
        !> Result object
        type(RealSphericalCubeHarmonics)  :: new
        new%solid_harmonics = RealRegularSolidHarmonics(lmax, lmin = lmin, normalization = normalization)
#ifdef HAVE_CUDA
        new%cuda_interface = RealSphericalCubeHarmonics_init_cuda(lmin, lmax, normalization, shape, stream_container)
#endif
    end function


    function RealSphericalCubeHarmonics_eval_grid(self, grid, center, no_cuda) result(reslt)
        class(RealSphericalCubeHarmonics)      :: self
        type(Grid3D), intent(in)               :: grid
        real(REAL64), intent(in)               :: center(3)
        logical,      intent(in), optional     :: no_cuda
         
        real(REAL64)                           :: reslt(    grid%axis(X_)%get_shape(), &
                                                            grid%axis(Y_)%get_shape(), &
                                                            grid%axis(Z_)%get_shape(), &
                                                            self%solid_harmonics%lmin**2 + 1 : &
                                                            (self%solid_harmonics%lmax+1)**2 )

#ifdef HAVE_CUDA
        type(Grid3D)                       :: temp_grid

        if (present(no_cuda)) then
            if (no_cuda) then
                call self%eval_grid_cpu(grid, center, reslt)
            else
                temp_grid = grid
                call self%eval_grid_cuda(temp_grid, center, reslt)
                call temp_grid%destroy()
            end if
        else 
            temp_grid = grid
            call self%eval_grid_cuda(temp_grid, center, reslt)
            call temp_grid%destroy()
        end if
        
        
#else
        call self%eval_grid_cpu(grid, center, reslt)
#endif
    end function

    subroutine RealSphericalCubeHarmonics_eval_grid_cpu(self, grid, center, reslt)
        class(RealSphericalCubeHarmonics),    intent(in)    :: self
        type(Grid3D),                     intent(in)    :: grid
        real(REAL64),                     intent(in)    :: center(3)
        real(REAL64),                     intent(inout) :: reslt(grid%axis(X_)%get_shape(), &
                                                                 grid%axis(Y_)%get_shape(), &
                                                                 grid%axis(Z_)%get_shape(), &
                                                                 self%solid_harmonics%lmin**2 + 1 : &
                                                                 (self%solid_harmonics%lmax+1)**2 )
        real(REAL64)                       :: gridpoints_x(grid%axis(X_)%get_shape()), &
                                              gridpoints_y(grid%axis(Y_)%get_shape()), &
                                              gridpoints_z(grid%axis(Z_)%get_shape())
        integer                            :: q, p, o
        
        real(REAL64), allocatable          :: cube_pos(:, :, :, :), r(:, :, :)

        gridpoints_x = grid%axis(X_)%get_coord() - center(X_)
        gridpoints_y = grid%axis(Y_)%get_coord() - center(Y_)
        gridpoints_z = grid%axis(Z_)%get_coord() - center(Z_)
       
        allocate(r (           grid%axis(X_)%get_shape(), &
                               grid%axis(Y_)%get_shape(), &
                               grid%axis(Z_)%get_shape()) )

        allocate(cube_pos(  3, grid%axis(X_)%get_shape(), &
                               grid%axis(Y_)%get_shape(), &
                               grid%axis(Z_)%get_shape()) )

        ! evaluate r and positions relative to center of the farfield box for each grid point
        forall(o = 1 : grid%axis(X_)%get_shape(), p = 1 : grid%axis(Y_)%get_shape(), q = 1 : grid%axis(Z_)%get_shape())
            r(o, p, q) = sqrt(gridpoints_x(o) ** 2 + gridpoints_y(p) ** 2 + gridpoints_z(q) ** 2 )
            cube_pos(X_, o, p, q) = gridpoints_x(o)
            cube_pos(Y_, o, p, q) = gridpoints_y(p)
            cube_pos(Z_, o, p, q) = gridpoints_z(q)
        end forall  

        ! evaluate spherical harmonics at each cube point and reshape the result to be of desired shape
        reslt = reshape( &
                    self%eval(reshape( cube_pos, [3, size(gridpoints_z) * size(gridpoints_y) * size(gridpoints_x)] ), &
                              reshape( r, [size(gridpoints_z) * size(gridpoints_y) * size(gridpoints_x)] )),           &            
                    [size(gridpoints_x), size(gridpoints_y), size(gridpoints_z), &
                     (self%solid_harmonics%lmax+1)**2 - (self%solid_harmonics%lmin)**2]  &
                ) 
        deallocate(cube_pos, r)
    end subroutine 


#ifdef HAVE_CUDA
    subroutine RealSphericalCubeHarmonics_eval_grid_cuda(self, grid, center, result_cubes)
        class(RealSphericalCubeHarmonics),    intent(in)    :: self
        type(Grid3D),                         intent(inout) :: grid
        real(REAL64),                         intent(in)    :: center(3)
        real(REAL64),                         intent(inout) :: result_cubes( grid%axis(X_)%get_shape(), &
                                                                         grid%axis(Y_)%get_shape(), &
                                                                         grid%axis(Z_)%get_shape(), &
                                                                         self%solid_harmonics%lmin**2 + 1 : &
                                                                        (self%solid_harmonics%lmax+1)**2)       
        integer                                         :: total_number_of_terms, ijk_max

        call RealCubeHarmonics_register_result_array_cuda(self%cuda_interface, result_cubes, shape(result_cubes))
        call RealSphericalCubeHarmonics_evaluate_cuda(self%cuda_interface, grid%get_cuda_interface(), center)
        call RealSphericalCubeHarmonics_download_cuda(self%cuda_interface, result_cubes, shape(result_cubes))
        call CUDASync_all()
        call RealCubeHarmonics_unregister_result_array_cuda(self%cuda_interface, result_cubes)
    end subroutine
    
#endif

    subroutine RealSphericalCubeHarmonics_destroy(self)
        class(RealSphericalCubeHarmonics), intent(inout)    :: self
        call self%solid_harmonics%destroy()
#ifdef HAVE_CUDA
        call RealSphericalCubeHarmonics_destroy_cuda(self%cuda_interface)
        deallocate(self%cuda_interface)
#endif
    end subroutine 

! %%%%%%%%%%%%%%%%%%%%%%% ComplexSphericalHarmonics %%%%%%%%%%%%%%%%%%%%%

    function ComplexSphericalHarmonics_init(lmax, lmin, normalization) result(new)
        !> Maximum angular momentum quantum number l generated
        integer, intent(in)           :: lmax
        !> Minimum angular momentum quantum number l generated
        integer, intent(in), optional :: lmin
        !> Type of normalization (1=racah's normalization (default), 2=conventional normalization)
        integer, intent(in), optional    :: normalization
        !> Result object
        type(ComplexSphericalHarmonics)  :: new
      
        new%real_spherical_harmonics = RealSphericalHarmonics(lmax, lmin = lmin, normalization = normalization)
    end function

    !> Evaluates values of Complex Spherical Harmonics with angular momentum 
    !! quantum number 'l' values 0 - self%lmax at points given as cartesian 
    !! (x, y, z coordinates) relative to the center ('positions').
    !! 
    !! Evaluation is done using Real Sherical Harmonics.
    function ComplexSphericalHarmonics_eval(self, positions, distances) result(res)
        class(ComplexSphericalHarmonics), intent(in)    :: self
        !> List of points. Shape is `(3, <number of points>)`.
        real(REAL64),              intent(in)        :: positions(:,:)
        !> List of distances from origo. Shape is (<number of points>).
        real(REAL64), optional,    intent(in)        :: distances(size(positions, 2))
        complex*16                                   :: res( size(positions,2), &
                                                             (self%real_spherical_harmonics%solid_harmonics%lmin**2 + 1) : &
                                                             (self%real_spherical_harmonics%solid_harmonics%lmax+1)**2 )
        real(REAL64)                                 :: real_harmonics( size(positions,2), &
                                                             (self%real_spherical_harmonics%solid_harmonics%lmin**2 + 1) : &
                                                             (self%real_spherical_harmonics%solid_harmonics%lmax+1)**2 ), &
                                                        r(size(positions, 2))
        integer                                      :: i, l, m
        real(REAL64), parameter                      :: THRESHOLD = epsilon(0.d0)
     
        ! evaluate solid harmonics (r^l Y_lm)
        real_harmonics = self%real_spherical_harmonics%eval(positions, distances)
     
        do l = self%real_spherical_harmonics%solid_harmonics%lmin, self%real_spherical_harmonics%solid_harmonics%lmax
            do m = -l, -1
                forall (i = 1 : size(positions, 2))
                    res(i, lm_map(l, m)) = 1.0d0 / dsqrt(2.0d0) * &
                        complex(real_harmonics(i, lm_map(l, -m)), -real_harmonics(i, lm_map(l, m)))
                end forall
            end do
            forall (i = 1 : size(positions, 2))
                res(i, lm_map(l, 0)) = complex(real_harmonics(i, lm_map(l, 0)), 0.0d0)
            end forall
            do m = 1, l 
                forall (i = 1 : size(positions, 2))
                    res(i, lm_map(l, m)) = (-1.0d0)**m / dsqrt(2.0d0) * &
                        complex(real_harmonics(i, lm_map(l, m)), real_harmonics(i, lm_map(l, -m)))
                end forall
            end do
        end do

    end function

    subroutine ComplexSphericalHarmonics_destroy(self)
        class(ComplexSphericalHarmonics), intent(inout)    :: self
        call self%real_spherical_harmonics%destroy()
    end subroutine 

! %%%%%%%%%%%%%%%%%%%%%%% MISC functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    !> Assign unique integer to (l,m) pair in order of l, then m.
    elemental pure function lm_map(l,m)
        integer, intent(in) :: l
        integer, intent(in) :: m
        integer             :: lm_map
        lm_map=l*(l+1)+m+1
    end function
end module
