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
!> @file harmonic.F90
!> Representation of spherical harmonics.

!> @todo harmonic_class docs
module harmonic_class
    use globals_m
    use xmatrix_m
    use CartIter_class
#ifdef HAVE_CUDA
    use ISO_C_BINDING
#endif
    implicit none

    public :: YBundle
    public :: Y_t
    public :: YProduct
    public :: Cart2SphIter
    public :: HarmonicIter

    public :: dvpos
#ifdef HAVE_CUDA
    public :: assign_c_pointers
#endif

    public :: idx, lm

    public :: rminus1_t, rminus1_init, rminus1_eval

    private
    type mono3D_t
        ! Define a monomial
        ! c * x^a * y^b * z^c * r^d ; r=(x^2+y^2+z^2)^(1/2)
        ! coeff* x**(expos(1)) * y**(expos(2)) * z**(expos(3)) * r**(expos(4))
        ! a>=b>=c>=0
        ! d can be smaller than 0
        private
        real(REAL64) :: coeff=0.d0
        integer(INT32) :: expos(4)=0
    contains
        procedure, pass(self) :: eval=>mono3D_eval
    end type
    
    interface mono3d_t
        module procedure mono3D_copy
    end interface

    type poly3D_t
        private
        type(mono3D_t),allocatable :: terms(:)
        integer(INT32) :: num_terms=0
    contains
        procedure :: destroy  =>poly3D_destroy
        procedure :: eval=>poly3D_eval
        procedure :: extend=>poly3D_extend
        procedure :: print=>poly3D_print
    end type
    
    interface poly3d_t
        module procedure poly3D_copy
        module procedure poly3D_init
    end interface

    type Y_t
        ! Describes a real spherical harmonic
        ! poly is the polynomial expression. For instance,
        ! Y(1,1) = x^1*r^-1
        ! Y(2,0) = 1.5*z^2*r^-2 - 0.5
        private
        integer(INT32) :: l,m
        type(poly3D_t) :: poly
    contains
        procedure, private    :: Y_eval_coeffs, Y_eval_coeffs_simple 
        procedure             :: get_coeff => Y_get_coeff
        generic,   public     :: eval_coeffs => Y_eval_coeffs, Y_eval_coeffs_simple
    end type

    type YProduct
        ! Stores the expansion coefficients C_{l_1,m_1,l_2,m_2}^{l,m} 
        ! Y_{l_1,m_1} Y_{l_2,m_2} =
        !            \sum_{l,m} C_{l_1,m_1,l_2,m_2}^{l,m} Y_{l,m}
        private
        integer(INT32) :: nmax
        integer(INT32), allocatable, public :: pos(:),number_of_terms(:), result_order_numbers(:)
        real(REAL64),  allocatable, public :: coefficients(:)
    contains
        procedure :: get_coeffs => YProduct_get_coeffs
        procedure :: destroy => YProduct_destroy
    end type

    type :: Cart2SphIter
        ! In order to transform from Cartesian to Spherical we do:
        ! f_{lm} = \sum_\kappa C_\kappa^{lm} f_\kappa * r**(|\kappa|-l)
        ! that is we want to reorganize this:
        ! c(1) * 1 + c(2) * x + c(3) * y + c(4) * z + c(5) *x^2 ...
        ! as 
        ! c(1) * s + c(2) * r * p_x + ... + (2*c(10)-c(5)-c(8)) * r^2 * d_z^2...
        ! But most C_\kappa^{lm} elements are zeros, so it's more efficient to
        ! implement as sparse matrix

        ! HOW TO USE:
        ! Depending of which operation is more costly, it is possible to
        ! iterate over the Cartesians that contribute to each spherical
        ! harmonic:

!            do l=0,self%lmax
!                do m=-l,l
!                    call C2S%init_loop_over_cart(l,m)
!                    done=.FALSE.
!                    do while(.not.done)
!                        call C2S%next_cart(coeff,kappa,done)
!                        k=sum(kappa)
!                        cout(idx(l,m))=cout(idx(l,m)) + &
!                                       coeff * cin(dvpos(kappa))
!                    end do
!                end do
!            end do

        ! or to iterate over all spherical harmonics that a given Cartesian
        ! contributes to:

!            do k=0,self%lmax
!                call cart%init(k)
!                done_out=.FALSE.
!                do while(.not.done_out)
!                    i=i+1
!                    kappa=cart%next(done_in)
!                    call C2S%init_loop_over_sph(kappa)
!                    done=.FALSE.
!                    do while(.not.done_in)
!                        call C2S%next_sph(coeff,l,m,done)
!                        cout(idx(l,m))=cout(idx(l,m)) + &
!                                       coeff * cin(dvpos(kappa))
!                    end do
!                end do
!            end do
        private
        integer(INT32) :: lmax, nsph, ncart, nelem
        integer(INT32) :: next, last
        real(REAL64), allocatable :: crs_val(:), ccs_val(:)
        integer(INT32), allocatable :: crs_kappa(:,:), ccs_lm(:,:), ccs_lm_id(:)
        integer(INT32), allocatable :: crs_ptr(:), ccs_ptr(:)
    contains
        procedure :: destroy             => Cart2Sph_Iter_destroy
        procedure :: init_loop_over_cart => Cart2Sph_Iter_init_loop_over_cart
        procedure :: init_loop_over_sph  => Cart2Sph_Iter_init_loop_over_sph
        procedure :: next_cart           => Cart2Sph_Iter_next_cart 
        procedure :: next_sph            => Cart2Sph_Iter_next_sph  
    end type

    !> Spherical harmonics
    type :: YBundle
        private
        integer(INT32) :: lmax, numY=0
        type(Y_t),pointer :: Y(:)=>NULL()
   contains
        procedure :: get_lmax=> YBundle_get_lmax
        procedure :: extend => YBundle_extend
        procedure :: eval    => YBundle_eval
        procedure :: pick    => YBundle_pick
        procedure :: print   => YBundle_print
        procedure :: destroy => YBundle_destroy
    end type

    !> An iterator object for constructing multipole moments \f$q_{lm}\f$
    !! from Cartesian multipole moments.
    !!
    !! Multipole moment evaluation requires information about regular
    !! solid harmonics \f$S_{l}^m\f$. These are homogeneous polynomials
    !! of degree l. They can be therefore written as
    !!
    !!  \f[ S_{l}^{m} = \sum_{a,b,c} c_{abc}x^a y^b z^c, \quad a+b+c = l. \f]
    !!
    !! When integration is performed in Cartesian coordinates, the above
    !! decomposition is used. The `HarmonicIter` object enables one
    !! to map an arbitrary Cartesian monomial, given by the exponent
    !! triplet (a,b,c), to those solid harmonic functions (l,m) that
    !! contain that monomial. In addition, the iterator yields the
    !! coefficient \f$c_{abc}\f$. This way the Cartesian intermediate
    !! integrals can be pieced together to yield the spherical
    !! multipole moments.
    !!
    !! Usage:
    !! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.f90} 
    !! program harmoniciter_test
    !!     use Harmonic_class
    !!     implicit none
    !!     type(HarmonicIter) :: multi
    !!     integer, parameter :: lmax=15, expo(3) = [2,2,2]
    !!     real(REAL64) :: coeff
    !!     integer :: l,m
    !!
    !!     multi=HarmonicIter(lmax)
    !!     call multi%loop_over(expo)
    !!     do while(multi%next(coeff, l, m))
    !!         print *, 'The monomial ', expo, ' is in '
    !!         print *, l,m, 'with weight', coeff
    !!     end do
    !!     call multi%destroy()
    !! end program
    !! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    !
    ! Some extra info:
    !
    ! The cache field of the HarmonicIter object is a matrix that
    ! is used to store the coefficients. The row index of the matrix is 
    ! the combined index (l,m)-> i. Similarly, the column index is the
    ! combined index of the exponents (a,b,c) -> j.
    !
    ! The algorithm for recovering all the solid harmonics that contain
    ! the monomial (a,b,c):
    !  
    !  1. Select the cache column j <- (a,b,c)
    !  2. Select those cache rows i <- (l,m), l=a+b+c, |m|<=l that constitute the l-shell.
    !     There are 2l+1 solid harmonics/rows that can possibly contain the
    !     monomial (a,b,c); In practice, only a fraction of the these
    !     will actually contain the monomial (a,b,c).
    !  3. Go over the rows.
    !     Yield the coefficients (l,m,coeff) one by one whenever a cache
    !     element (i,j) is non-zero.
    !  4. When the whole shell has been processed, terminate iteration.
    type :: HarmonicIter
        private
        integer :: lmax = 0, l = 0
        real(REAL64), public, pointer :: cache(:,:) => NULL()
        real(REAL64), pointer :: cache_line(:) => NULL()
    contains
        procedure :: loop_over => HarmonicIter_loop_over
        procedure :: next => HarmonicIter_next
        procedure :: destroy => HarmonicIter_destroy
    end type

    interface HarmonicIter
        module procedure HarmonicIter_init
    end interface

    type rminus1_t
        ! Contains the Cartesian derivatives of 1/r
        private
        integer(INT32) :: nmax, sz
        type(poly3D_t),pointer :: polys(:)
    end type

    interface operator(*)
        module procedure mono3D_times_mono3D
        module procedure poly3D_times_mono3D
        module procedure poly3D_times_poly3D
        module procedure real_times_mono3D
        module procedure real_times_poly3D
    end interface

    interface operator(**)
        module procedure poly3D_pow
    end interface

    interface operator(+)
        module procedure poly3D_plus_poly3D
    end interface

    interface operator(-)
        module procedure poly3D_minus_poly3D
    end interface

    interface YBundle
        module procedure YBundle_init
    end interface

    interface YProduct
        module procedure YProduct_init
    end interface

    interface Cart2SphIter
        module procedure Cart2SphIter_init
    end interface

contains

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!! POLYNOMIAL                                                !!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ! Destructor
    pure subroutine poly3D_destroy(self)
        class(poly3D_t), intent(inout) :: self
        if(allocated(self%terms)) deallocate(self%terms)
        self%num_terms=0
        return
    end subroutine

    pure function mono3D_copy(orig) result(new) 
        type(mono3D_t),intent(in) :: orig
        type(mono3D_t)            :: new
        
        new = mono3D_t( coeff=orig%coeff,expos=orig%expos )
        return
    end function

    pure function poly3D_copy(orig) result(new)
        type(poly3D_t),intent(in) :: orig
        type(poly3D_t)            :: new
        new=poly3D_t(terms=orig%terms, num_terms=orig%num_terms)
        return
    end function

    pure function poly3D_init(num_terms) result(new)
        integer(INT32), intent(in) :: num_terms
        type(poly3D_t) :: new

        allocate(new%terms(num_terms))
        new%num_terms=num_terms
        return
    end function

    pure function real_times_mono3D(k,m_in) result(m_out)
        type(mono3D_t), intent(in) :: m_in
        real(REAL64), intent(in) :: k
        type(mono3D_t) :: m_out

        m_out=mono3D_t( coeff=k * m_in%coeff,&
                        expos=m_in%expos )
        return
    end function

    pure function real_times_poly3D(k,p_in) result(p_out)
        type(poly3D_t), intent(in) :: p_in
        real(REAL64), intent(in) :: k
        type(poly3D_t) :: p_out
        integer(INT32) :: i

        p_out=poly3D_t(terms=&
                [(k*p_in%terms(i),i=1,p_in%num_terms)],&
                   num_terms=p_in%num_terms)
        return
    end function

    pure function mono3D_times_mono3D(m1_in,m2_in) result(m_out)
        type(mono3D_t), intent(in) :: m1_in, m2_in
        type(mono3D_t) :: m_out

        m_out=mono3D_t( coeff=m1_in%coeff * m2_in%coeff,&
                        expos=m1_in%expos + m2_in%expos )
        return
    end function

    pure function poly3D_times_mono3D(p_in,m_in) result(p_out)
        type(poly3D_t), intent(in) :: p_in
        type(mono3D_t), intent(in) :: m_in
        type(poly3D_t) :: p_out

        integer(INT32) :: i

        ! According to Fortran 2003, there is no need to allocate; the
        ! allocatable component should be allocated appropriately
!        call init_poly3D(p_in%num_terms)
        p_out=poly3D_t(terms=&
                [(p_in%terms(i)*m_in,i=1,p_in%num_terms)],&
                   num_terms=p_in%num_terms)
        return
    end function

    pure function poly3D_plus_poly3D(p1,p2) result(pout)
        type(poly3D_t), intent(in) :: p1,p2
        type(poly3D_t) :: ptmp,pout

        integer(INT32) :: i,k,counter,nout
        integer(INT32) :: maxsize

        if (p1%num_terms==0) then
            pout=p2
            return
        else if (p2%num_terms==0) then
            pout=p1
            return
        end if

        maxsize=p1%num_terms+p2%num_terms
        ptmp = poly3D_t(maxsize)
        ptmp%terms(1:p1%num_terms)=p1%terms
        counter=p1%num_terms

        out: do i=1,p2%num_terms
            ! Find equal entry
            do k=1,counter
                ! If found, sum coeff
                if( all(ptmp%terms(k)%expos==p2%terms(i)%expos) ) then
                        ptmp%terms(k)%coeff=ptmp%terms(k)%coeff+&
                                              p2%terms(i)%coeff
                    cycle out
                end if
            end do
            ! If not found, create new entry
            counter=counter+1
            ptmp%terms(counter)=mono3D_t( coeff=p2%terms(i)%coeff,&
                                          expos=p2%terms(i)%expos)
        end do out

        !Find out how many valid terms came out
        nout=count(abs(ptmp%terms(:)%coeff)>epsilon(0.d0))
        pout = poly3D_t(nout)
        counter=1
        do i=1,ptmp%num_terms
            if (abs(ptmp%terms(i)%coeff)<=epsilon(0.d0)) cycle
            pout%terms(counter)%coeff=ptmp%terms(i)%coeff
            pout%terms(counter)%expos=ptmp%terms(i)%expos
            counter=counter+1
        end do
        return
    end function

    pure function poly3D_minus_poly3D(p1,p2) result(pout)
        type(poly3D_t), intent(in) :: p1,p2
        type(poly3D_t) :: pout
        pout=p1+(-1.d0)*p2
        return
    end function

    pure function poly3D_times_poly3D(p1,p2) result(pout)
        type(poly3D_t), intent(in) :: p1,p2
        type(poly3D_t) :: pout

        integer(INT32) :: i

        do i=1,p2%num_terms
            pout=pout+ p1 * p2%terms(i)
        end do
    end function

    pure function poly3D_pow(p,k) result(pout)
        type(poly3D_t), intent(in) :: p
        integer(INT32),intent(in) :: k
        type(poly3D_t) :: pout

        integer(INT32) :: i

        if(k==0) then
            pout=poly3d_t(terms=[mono3D_t(coeff=1.d0,expos=[0,0,0,0])],&
                                num_terms=1)
        else
            pout=p
            do i=1,k-1
                pout=pout*p
            end do
        end if
    end function

    pure function poly3D_extend(self,k) result(pout)
        ! Extend polynomial so that all terms are of order k. The difference
        ! between the order of p and k must be an even number.
        
        ! Examples:

        ! p is 1
        ! p%extend(2) returns
        ! x^2/r^2 + y^2/r^2 + z^2/r^2

        ! p is xy/r^2
        ! p%extend(6) returns
        ! ( x^5y + 2x^3y^3 + xy^5 + 2x^3yz^2 + 2xy^3z^2 + xyz^4 ) / r^6

        class(poly3D_t), intent(in) :: self
        integer(INT32),intent(in) :: k
        type(poly3D_t) :: pout, r2

        integer(INT32) :: i

        r2=poly3D_t(terms=[mono3D_t(coeff=1.d0,expos=[2,0,0,-2]),&
                           mono3D_t(coeff=1.d0,expos=[0,2,0,-2]),&
                           mono3D_t(coeff=1.d0,expos=[0,0,2,-2])],&
                           num_terms=3)

        pout=poly3D_t(terms=[mono3D_t(coeff=0.d0,expos=[0,0,0,0])],num_terms=0)

        do i=1,self%num_terms
            pout=pout+(r2**((k+self%terms(i)%expos(4))/2))*self%terms(i)
        end do
    end function

    pure function mono3D_eval(self,points, distances) result(res)
        class(mono3D_t),intent(in):: self
        real(REAL64),intent(in):: points(:,:) !shape is (3,npoints)
        real(REAL64),intent(in):: distances(:) !shape is (npoints)
        real(REAL64) :: res(size(points,2))
        real(REAL64), parameter :: THRESHOLD = epsilon(0.d0)
        integer(INT32) :: i


        if(all(self%expos==0)) then
            res=self%coeff
        else
            res = 0.0d0
            ! Return 0 if r==0
            do i = 1, size(points, 2)
                if (distances(i) > THRESHOLD) then
                    res(i) = product(points(:,i)**self%expos(: 3))              &
                        * distances(i) ** self%expos(4)*self%coeff
                else if (self%expos(4) == 0) then
                    res(i) = product(points(:,i)**self%expos(: 3)) * self%coeff
                end if
            end do
        end if
        return
    end function

    pure function poly3D_eval(self,points, distances) result(res)
        class(poly3D_t),intent(in):: self
        real(REAL64),intent(in) :: points(:,:) !shape is (3,npoints)
        real(REAL64),intent(in) :: distances(:)
        real(REAL64)            :: res(size(points,2))
        integer(INT32) :: i

        res=0.d0
        do i=1,self%num_terms
            res=res+mono3D_eval(self%terms(i),points, distances)
        end do
        return
    end function

    pure function poly3D_der(p_in,d) result(p_out)
        type(poly3D_t),intent(in)  :: p_in
        integer(INT32), intent(in) :: d
        type(poly3D_t) :: p_out
        integer(INT32) :: exp1(4),exp2(4) ! 
        integer(INT32) :: i! d is the direction of the derivative

        ! Derivation can be rewritten as polynomial multiplication
        ! d_x P(x;a,b,c,d)=  a* P(x;a-1,b,c,d) + d P(x;a+1,b,c,d-2) =
        !     P(x;a,b,c,d) * ( a/x + d*x/r^2 )
        exp1=[(0,i=1,d-1), 1,(0,i=d+1,3),-2] ! [xyz]*r^-2
        exp2=[(0,i=1,d-1),-1,(0,i=d+1,3), 0] ! [xyz]^-1
        do i=1,p_in%num_terms
            
            p_out=p_out+&
                  (poly3D_t(num_terms=1,terms=&
                     [mono3D_t(coeff=p_in%terms(i)%expos(4),expos=exp1)]) + &
                   poly3D_t(num_terms=1,terms=&
                     [mono3D_t(coeff=p_in%terms(i)%expos(d),expos=exp2)]))*&
                              p_in%terms(i)
        end do
        return
    end function

    pure function poly3D_multider(p_in,dv) result(p_out)
        type(poly3D_t), intent(in) :: p_in
        integer(INT32), intent(in) :: dv(3)
        type(poly3D_t) :: p_out
        integer(INT32) :: i,j ! dv(i) is the derivation order in direction i

        p_out=p_in
        do i=1,3
            do j=1,dv(i)
                p_out=poly3D_der(p_out,i)
            end do
        end do
        return
    end function

    subroutine poly3D_print(self)
        class(poly3D_t),intent(in):: self
        integer(INT32) :: i

        print*,'¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤'
        print'(f12.5,4i5)',(self%terms(i)%coeff,self%terms(i)%expos,&
                                                    i=1,self%num_terms)
        print*
    end subroutine
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!! SPHERICAL HARMONICS                                       !!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    !> Returns the index for the (l,m)-th spherical harmonic, if they are
    !! ordered by (l,m):
    !! (0,0) -> 1 ; (1,-1) -> 2 ; (1,0) -> 3 ; (1,1) -> 4 ; (2,-2) -> 5 ; etc
    elemental pure function idx(l,m)
        integer, intent(in) :: l,m
        integer             :: idx
  
        idx=1+l*(l+1)+m
        return 
    end function

    pure function lm(idx)
        ! Inverse of the function 'idx'
        integer(INT32), intent(in) :: idx
        integer(INT32)             :: lm(2)
  
        lm(1)=int(sqrt(idx-1.d0),kind=INT32)
        lm(2)=idx-lm(1)*(lm(1)+1)-1
        return 
    end function

    pure function YBundle_init(lmax) result(new)
        ! Iterative generation of spherical harmonics
        ! Helgaker, Jørgensen and Olsen, Molecular Electronic-Structure
        ! Theory, Wiley, 2000, p. 218
        integer(INT32), intent(in) :: lmax
        type(YBundle) :: new
!        integer(INT32), intent(in) :: lmax

        integer(INT32) :: l,m
        real(REAL64) :: f1
        type(mono3D_t),parameter :: x=mono3D_t(coeff=1.d0,expos=[1,0,0,-1]),&
                                    y=mono3D_t(coeff=1.d0,expos=[0,1,0,-1]),&
                                    z=mono3D_t(coeff=1.d0,expos=[0,0,1,-1]),&
                                   r2=mono3D_t(coeff=1.d0,expos=[0,0,0, 0])

        new%lmax=lmax
        new%numY=(new%lmax+1)**2 
        allocate(new%Y( new%numY ))
        ! Y_{0,0} (s)
        new%Y(idx(0,0))=Y_t(l=0, m=0,poly=&
             poly3D_t(terms=&
                    [mono3D_t(coeff=1.d0,expos=[0,0,0,0])],&
                  num_terms=1))
        ! l=1 shell, if it is needed
        if (new%lmax>0) then
            ! Y_{1,-1}
            new%Y(idx(1,-1))=Y_t(l=1,m=-1,poly=&
                               new%Y(idx(0,0))%poly * y )
            ! Y_{1, 0}
            new%Y(idx(1, 0))=Y_t(l=1,m= 0,poly=&
                               new%Y(idx(0,0))%poly * z )
            ! Y_{1, 1}
            new%Y(idx(1, 1))=Y_t(l=1,m= 1,poly=&
                               new%Y(idx(0,0))%poly * x )
        end if
        ! Remaining shells with l>=2
        do l=2,new%lmax
            ! Y_{l,-l}
            f1=sqrt(1.d0-0.5d0/l)
            new%Y(idx(l,-l))=Y_t(l=l,m=-l,poly=&
             f1*(new%Y(idx(l-1,l-1))%poly*y+new%Y(idx(l-1,-(l-1)))%poly*x))

            ! Y_{l,-(l-1)} to Y_{l,l-1} 
            do m=-(l-1),l-1
                new%Y(idx(l, m))=Y_t(l=l,m=m,poly=&
                (2.d0*l-1.d0)* new%Y(idx(l-1,m))%poly * z )
                if (l-abs(m)>1) &
                    new%Y(idx(l, m))%poly=new%Y(idx(l, m))%poly - &
                           sqrt((l+m-1.d0)*(l-m-1.d0))* &
                           new%Y(idx(l-2,m))%poly * r2
                new%Y(idx(l, m))%poly=&
                    (sqrt(1.d0*(l-m)*(l+m))**(-1)*new%Y(idx(l, m))%poly)
            end do

            ! Y_{l,l}
            new%Y(idx(l, l))=Y_t(l=l,m=l,poly=&
            f1*(new%Y(idx(l-1,l-1))%poly*x-new%Y(idx(l-1,-(l-1)))%poly*y))
        end do

        
        return
    end function

    pure subroutine YBundle_destroy(self)
        class(YBundle), intent(inout) :: self
        integer          :: l,m

        do l=0,self%lmax
            do m=-l,l
                call self%Y(idx(l,m))%poly%destroy()
            end do
        end do
        deallocate(self%Y)

    end subroutine

    pure function YBundle_get_lmax(self) result(lmax)
        class(YBundle),intent(in):: self
        integer :: lmax

        lmax=self%lmax
        return
    end function


    !> Returns an array with the values or derivatives of {Y_{l,m}} at the
    !! coordinates given in 'points'.
    pure function YBundle_eval(self, points, distances, dv, lmin) result(res)

        class(YBundle),intent(in)           :: self
        real(REAL64),intent(in)             :: points(:,:) !shape is (3,npoints)
        real(REAL64),intent(in)             :: distances(:) !shape is (npoints)
        integer(INT32),optional, intent(in) :: dv(3)
        integer(INT32),optional, intent(in) :: lmin
        real(REAL64)                        :: res(size(points,2),self%numY)
        integer(INT32)                      :: i, start_index

        if (present(lmin)) then
            start_index = lmin*lmin + 1
        else
            start_index = 1
        end if

        if(present(dv)) then
            forall (i=start_index : self%numY)
                res(:,i)=poly3D_eval(poly3D_multider(self%Y(i)%poly,dv), &
                                     points, distances)
            end forall
        else
            forall (i=start_index : self%numY)
                res(:,i)=poly3D_eval(self%Y(i)%poly,points, distances)
            end forall
        end if
        return
    end function

    function YBundle_pick(self,l,m) result(Y)
        class(YBundle), intent(in), target :: self
        integer(INT32), intent(in)         :: l,m
        type(Y_t),pointer :: Y

        Y=>self%Y(idx(l,m))
        return
    end function

    subroutine YBundle_print(self)
        class(YBundle) :: self
        integer(INT32) :: l,m
        do l=0,self%lmax
            print*,repeat("+",60)
            do m=-l,l
                print*,l,m
                call self%Y(idx(l,m))%poly%print()
            end do
        end do
        print*,repeat("+",60)
    end subroutine

    pure subroutine YBundle_extend(self)
        class(YBundle), intent(inout) :: self
        integer :: i
        do i=1, self%numY
            self%Y(i)%poly = self%Y(i)%poly%extend(self%Y(i)%l)
        enddo
    end subroutine

    !> Returns the coefficient `coeff' multiplying the monomial 
    !! x**a * y**b * z**c with expo = (a,b,c).
    pure function Y_get_coeff(self, expo) result(coeff)
        class(Y_t),     intent(in) :: self
        integer(INT32), intent(in) :: expo(3)
        real(REAL64) :: coeff
        integer :: i

        coeff = 0.0d0
        do i=1, self%poly%num_terms
            if (all(self%poly%terms(i)%expos(1:3)==expo)) then
                coeff = self%poly%terms(i)%coeff
                return
            endif
        enddo

    end function
    
    pure function Y_eval_coeffs_simple(self, points, distances) result(res)
        class(Y_t),intent(in)     :: self
        real(REAL64),intent(in)   :: points(:,:)  !shape is (3,npoints)
        real(REAL64),intent(in)   :: distances(:) !shape is (npoints)
        real(REAL64)              :: res(size(points,2))
        
        res(:) = poly3D_eval(self%poly, points, distances)
    end function

    pure function Y_eval_coeffs(self,points, distances, dv) result(res)
        ! Given Y_{lm}, which is a spherical harmonic that multiplies a radial
        ! function f(r), returns array A, whose elements a_ij are
        ! a_ij = P_i(point(j))

        ! dv represents the derivation orders:
        ! d^{dv(1)}/dx^{dv(1)} d^{dv(2)}/dx^{dv(2)} d^{dv(3)}/dx^{dv(3)}

        ! P_i(x,y,z,r) is the term that multiplies the i-th derivative of f
        ! in the dv-th derivative of Y_{l,m}*f(r):
        !                                vvvvvvvvvvvv 
        ! d_{dv} Y_{l,m}*f(r) = \sum_i { P_i(x,y,z,r) * f^{(i)}(r) }
        !                                ^^^^^^^^^^^^ 

        class(Y_t),intent(in)     :: self
        real(REAL64),intent(in)   :: points(:,:) !shape is (3,npoints)
        real(REAL64),intent(in)   :: distances(:) !shape is (npoints)
        integer(INT32),intent(in) :: dv(3)

        real(REAL64)              :: res(size(points,2), 0:sum(dv))

        integer(INT32)            :: ider, maxd, dm, i, ipol
        ! We will need one derivative per degree
        type(poly3D_t)            :: derpols(0:sum(dv)), prevpols(0:sum(dv))
        integer(INT32)            :: expos(4)
        type(mono3D_t)            :: term

        maxd=sum(dv)
        ! Construct derivative polynomials
        ! The starting polynomial (degree 0) is the original one
        prevpols(0) =  poly3D_t(self%poly)
        ider=0
        do dm=X_,Z_
            !      |<-------- x ------------>| /r
            expos=[(0,i=1,dm-1),1,(0,i=dm+1,3),-1]
            term=mono3D_t(coeff=1.d0,expos=expos)
            do i=1,dv(dm)
                ! Derivate each polynomial
                do ipol=0,ider
                    ! If a,b,c and d are the exponents of the polynomial:
                    ! d_x f(r) P(x;a,b,c,d)=
                    !     f(r) * d_x(P(x;a,b,c,d))
                    derpols(ipol)=derpols(ipol)+poly3D_der(prevpols(ipol),dm)
                    !   + f'(r)*     P(x;a+1,b,c,d-1)
                    derpols(ipol+1)=derpols(ipol+1)+prevpols(ipol)*term
                end do
                ! New polynomials become old polynomials
                prevpols=derpols
                ! Set derivated polynomials to 0
                do ipol=0,ider+1
                    call derpols(ipol)%destroy()
                end do
                ider=ider+1
            end do
        end do

        ! Evaluate each polynomial at each point
        do i=0,maxd
            res(:,i)=poly3D_eval(prevpols(i),points, distances)
            call prevpols(i)%destroy()
        end do

        return
    end function

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!! REAL SPHERICAL HARMONICS PRODUCTS                         !!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    pure function prod_idx(Yprod,idx1,idx2)
    ! Returns the index for the (idx1-th,idx2-th)-th spherical harmonic, as if
    ! looking up the following table:
    !               idx>
    !          1    2   3   4
    !      1   1    2   3   4
    !idx<  2        5   6   7
    !      3            8   9
    !      4               10
        type(YProduct), intent(in) :: Yprod
        integer(INT32), intent(in) :: idx1,idx2
        integer(INT32)             :: prod_idx,is,il
        is=min(idx1,idx2)
        il=max(idx1,idx2)
        prod_idx=(is-1)*(2*Yprod%nmax-is)/2+il
        return 
    end function

    pure function idx_idx(Yprod,idx)
        ! Inverse of the previous function
        type(YProduct), intent(in) :: Yprod
        integer(INT32), intent(in) :: idx
        integer(INT32) :: idx_idx(2)
  
        idx_idx(1)=int(0.5*( -sqrt(9.d0-8*idx+4*Yprod%nmax*(Yprod%nmax+1))+&
                             3+2*Yprod%nmax), kind=INT32)
        idx_idx(2)=idx-(idx_idx(1)-1)*(2*Yprod%nmax-idx_idx(1)+2)/2+idx_idx(1)-1
        return 
    end function

    pure function YProduct_init(lmax, result_lmax) result(new)
        integer(INT32), intent(in)  :: lmax
        integer(INT32), intent(in)  :: result_lmax
        type(YProduct)              :: new
        integer(INT32)              :: sz
        integer(INT32)              :: order_numbers((lmax+1)**6) ! Large enough... I think
        real(REAL64)                ::   coefficients((lmax+1)**6)
        integer(INT32)              :: l1,m1,l2,ms,m2,l,m
        integer(INT32)              :: i,number_of_result_terms,sgn

        new%nmax=(lmax+1)**2

        ! Number of pairs Y_1*Y_2
        sz=new%nmax*(new%nmax+1)/2
        allocate(new%pos(sz))
        allocate(new%number_of_terms(sz))
        i=0
        number_of_result_terms=0
        new%pos(1)=1
        do l1=0,lmax
            do m1=-l1,l1
                do l2=l1,lmax
                    if(l1==l2) then
                        ms=m1
                    else
                        ms=-l2
                    end if
                    do m2=ms,l2
                        i=i+1
                        sgn=sign(1,m1)*sign(1,m2)
                        do l=abs(l1-l2),min(result_lmax,l1+l2),2
                            m=sgn*abs(m1-m2)
                            if(abs(m)>l) cycle
                            ! increase the counter for found final result terms
                            number_of_result_terms=number_of_result_terms+1
                            
                            ! calculate and store the coefficient used in final calculation
                            coefficients(number_of_result_terms)  =prod_coeff(l1,m1,l2,m2,l,m)
                            ! store the order number of result array in result bubbles
                            order_numbers(number_of_result_terms) = idx(l,m)
                            if((m1*m2/=0).and.( (sgn==1).or.(m1+m2)/=0 )) then
                                m=sgn*abs(m1+m2)
                                if(abs(m)>l) cycle
                                ! increase the counter for found final result terms
                                number_of_result_terms=number_of_result_terms+1
                                ! calculate and store the coefficient used in final calculation
                                coefficients(number_of_result_terms) = prod_coeff(l1,m1,l2,m2,l,m)
                                ! store the order number of result array in result bubbles
                                order_numbers(number_of_result_terms)  = idx(l,m)
                            end if
                        end do
                        ! store the number of l,m pairs affected by l1,l2,m1,m2
                        new%number_of_terms(i)=number_of_result_terms-new%pos(i)+1
                        if(i<sz) new%pos(i+1)=new%pos(i)+new%number_of_terms(i)
                        if(new%number_of_terms(i)==0) new%pos(i)=0
                    end do
                end do
            end do
        end do
        ! allocate memory for the final coefficients
        allocate(new%coefficients(number_of_result_terms))
        new%coefficients=coefficients(:number_of_result_terms)

        ! allocate the memory for result indices
        allocate(new%result_order_numbers(number_of_result_terms))
        new%result_order_numbers=order_numbers(:number_of_result_terms)
        return
    end function

    pure subroutine YProduct_destroy(self)
        class(YProduct), intent(inout) :: self

        if (allocated(self%pos)) deallocate(self%pos)
        if (allocated(self%number_of_terms)) deallocate(self%number_of_terms)
        if (allocated(self%result_order_numbers)) deallocate(self%result_order_numbers)
        if (allocated(self%coefficients)) deallocate(self%coefficients)
        return
    end subroutine

    pure subroutine YProduct_get_coeffs(self,l1,m1,l2,m2,sz, order_numbers, coefficients)
        !Retrieve the ids and coefficients of the Y's to which the product
        ! Y1*Y2 contributes
        ! For instance
        ! s * s = s              ->       nout=[1];coeff=[1]
        ! px* px= (1/3)*s - sqrt(1/3) * dz^2 + sqrt(2/3) * dx^2-y^2   ->
        !                      nout=[1,7,9];coeff=[0.333,-0.578,0.816]
        ! sz becomes the size of nout and coeff. If it's 0, Y1 and Y2 don't
        ! couple
        class(YProduct), intent(inout), target :: self
        integer(INT32),  intent(in)            :: l1,m1,l2,m2
        integer(INT32),  intent(out)           :: sz
        integer(INT32),  pointer, intent(out)  :: order_numbers(:)
        real(REAL64),    pointer, intent(out)  :: coefficients(:)
        integer(INT32) :: n

        ! Unique pair ID
        n=prod_idx(self,idx(l1,m1),idx(l2,m2))
        ! get the number of terms caused by the 4 index combination l1, m1, l2, m2
        sz=self%number_of_terms(n)
        if(sz==0) return
        ! get the order numbers of the result arrays in the result bubbles
        order_numbers => self%result_order_numbers(self%pos(n):self%pos(n)+self%number_of_terms(n)-1)
        ! get the order coefficients used for the terms in the result bubbles
        coefficients  => self%coefficients(self%pos(n):self%pos(n)+self%number_of_terms(n)-1)
        return
    end subroutine

    pure function cg(l1,m1,l2,m2,l,m)
!    Clebsch-Gordan calculator
!    Stolen (and adapted) from
!    http://www.davidgsimpson.com/software/cg_f90.txt
!    Returns <l_1m_1l_2m_2;lm>
!    It only works for integer l's!
        integer(INT32), intent(in) :: l1, l2, l, m1, m2, m
        real(REAL64) :: cg

        integer(INT32) :: I, K
        real(REAL64) :: SUMK, TERM
        real(REAL64), DIMENSION(0:99) :: FACT

        fact(0)=1.d0
        do i=1,size(fact)-1
            fact(i)=fact(i-1)*i
        end do

!         Check for conditions that give cg = 0.
        IF ( (l < ABS(l1-l2)) .OR. (l > (l1+l2)) .OR. &
             (ABS(m1)> l1)    .OR. (ABS(m2)> l2) .OR. (ABS(m) > l) .or. &
             (m /= m1+m2)) THEN
            cg = 0.0D0
        ELSE
!         Compute Clebsch-Gordan coefficient.
            cg = SQRT((2*l+1.d0)/FACT(l1+l2+l+1) * &
                     FACT(l1+l2-l)*FACT(l2+l-l1)*FACT(l+l1-l2) * &
                     FACT(l1+m1)*FACT(l1-m1)*FACT(l2+m2)*&
                     FACT(l2-m2)*FACT(l+m)*FACT(l-m))
            SUMK = 0.0D0
            DO K = max(l1+m2-l,l2-m1-l,0),min(l1+l2-l,l1-m1,l2+m2)
                TERM = FACT(l1+l2-l-K)*FACT(l-l1-m2+K)*FACT(l-l2+m1+K)*&
                       FACT(l1-m1-K)*FACT(l2+m2-K)*FACT(K)
                IF (MOD(K,2) .EQ. 1) TERM = -TERM
                SUMK = SUMK + 1.0D0/TERM
            END DO
            cg = cg * SUMK
        END IF

        return
    end function

    pure function cp(m)
        integer(INT32), intent(in) :: m
        real(REAL32) :: cp

        select case(m)
            case(:-1)
                cp=0.5
            case(0)
                cp=0
            case(1:)
                cp=m
        end select
        return
    end function

    pure function cm(m)
        integer(INT32), intent(in) :: m
        real(REAL32) :: cm

        select case(m)
            case(:-1)
                cm=m-0.5
!            case(0) !This should never happen!!!
            case(1:)
                cm=0
        end select
        return
    end function

    pure function prod_coeff(l1,m1,l2,m2,l,m) result(res)
        integer(INT32), intent(in) :: l1, l2, l, m1, m2, m
        real(REAL64) :: res
        real(REAL32) :: f

        if( mod(l-l1-l2,2)/=0 .or. &
            sign(1,m1)*sign(1,m2)*sign(1,m) < 0 ) then! Only 0 or 2 negative m's
            res=0.d0
        else
            if(m==m1+m2) then
                f=cp(m1)+cp(m2)-cp(m)
                res=cg(l1,0,l2,0,l,0)*cg(l1, m1,l2, m2,l,m)*(-1)**nint(f)
            else if(m== m1-m2) then
                f=cp(m1)+cm(m2)-cp(m)
                res=cg(l1,0,l2,0,l,0)*cg(l1, m1,l2,-m2,l,m)*(-1)**nint(f)
            else if(m==-m1+m2) then
                f=cm(m1)+cp(m2)-cp(m)
                res=cg(l1,0,l2,0,l,0)*cg(l1,-m1,l2, m2,l,m)*(-1)**nint(f)
            else if(m==-m1-m2) then
                f=cm(m1)+cm(m2)-cp(m)
                res=cg(l1,0,l2,0,l,0)*cg(l1,-m1,l2,-m2,l,m)*(-1)**nint(f)
            else ! m1 m2 and m don't couple
                res=0.d0
            end if
            if(m1*m2*m/=0) res=res/sqrt(2.d0) ! Extra factor if all are non-zero
        end if
        return
    end function

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!  Cartesian To Spherical conversion                   !!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    function Cart2SphIter_init(lmax) result(new)
        integer, intent(in) :: lmax
        type(Cart2SphIter)  :: new

        type(YBundle) :: yb
        type(Y_t),pointer :: Y
        type(poly3D_t) :: poly_w
        integer(INT32) :: k,l,m,i,ilm,icart,iout
        real(REAL64), allocatable,target :: tmp(:,:), cart_to_sph(:,:)
        real(REAL64), pointer :: val
        type(CartIter) :: iter
        integer(INT32) :: kappa(3),numk
        logical :: continue_iteration

        new%lmax=lmax
        yb=YBundle(new%lmax)

        k=new%lmax+1
        new%ncart=k*(k+1)*(k+2)/6
        new%nsph=k*k
        allocate(cart_to_sph(new%nsph, new%ncart))
        cart_to_sph=0.d0
        ! Collect coefficients for each k-shell
        do k=0,new%lmax
            numk=(k+1)*(k+2)/2
            allocate(tmp(numk,numk))
            tmp=0.d0
            do l=mod(k,2),k,2
                do m=-l,l
                    Y=>yb%pick(l,m)
                    poly_w=Y%poly%extend(k)
                    do i=1,poly_w%num_terms
                        val=>tmp(dvpos_shell(poly_w%terms(i)%expos(1:3)),&
                                 (l+1)*(l+2)/2-l+m)
                        val=poly_w%terms(i)%coeff
                    end do
                end do
            end do
            ! Invert matrix
            tmp=matrix_inverse(tmp)
            ! Put the coefficients in to the final matrix
            do l=mod(k,2),k,2
                i=max(l*(l-1)/2+1,1)
                cart_to_sph( l**2+1 : (l+1)**2, &
                             k*(k+1)*(k+2)/6+1 : (k+1)*(k+2)*(k+3)/6 )= &
                             tmp( i:i+2*l, :)
            end do
            deallocate(tmp)
        end do

        ! Find number of non-zero elements
        new%nelem=count(abs(cart_to_sph)>1.d-16)

        ! Store matrix using Contracted Row Storage format
        allocate(new%crs_ptr(new%nsph+1))
        allocate(new%crs_val(new%nelem))
        allocate(new%crs_kappa(3,new%nelem))
        new%crs_ptr(1)=1
        ilm=0
        iout=0
        do l=0,new%lmax
            do m=-l,l
                ilm=ilm+1
                icart=0
                iter=CartIter(3,new%lmax)
                call iter%next(kappa, continue_iteration)
                do while(continue_iteration)
                    icart=icart+1
                    if(abs(cart_to_sph(ilm,icart))>1.d-16) then
                        iout=iout+1
                        new%crs_val(iout)=cart_to_sph(ilm,icart)
                        new%crs_kappa(:,iout)=kappa
                    end if
                    call iter%next(kappa, continue_iteration)
                end do
                new%crs_ptr(ilm+1)=iout+1
                call iter%destroy()
            end do
        end do
        ! Store matrix using Contracted Column Storage format
        allocate(new%ccs_ptr(new%ncart+1))
        allocate(new%ccs_val(new%nelem))
        allocate(new%ccs_lm_id(new%nelem))
        allocate(new%ccs_lm(2,new%nelem))
        new%ccs_ptr(1)=1
        icart=0
        iout=0
            iter=CartIter(3,new%lmax)
            call iter%next(kappa, continue_iteration)
            do while(continue_iteration)
                icart=icart+1
                ilm=0
                do l=0,new%lmax
                    do m=-l,l
                        ilm=ilm+1
                        if(abs(cart_to_sph(ilm,icart))>1.d-16) then
                            iout=iout+1
                            new%ccs_val(iout)=cart_to_sph(ilm,icart)
                            new%ccs_lm(:,iout)=[l,m]
                            new%ccs_lm_id(iout) = ilm
                        end if
                    end do
                end do
                call iter%next(kappa, continue_iteration)
                new%ccs_ptr(icart+1)=iout+1
        end do
        call iter%destroy()
        call yb%destroy()

        deallocate(cart_to_sph)
!        do l=0,YBundle%lmax
!            do m=-l,l
!                print'("<<<< ",i3,", ",i3," >>>>")', l,m
!                i=idx(l,m)
!                do k=new%crs_ptr(i),new%crs_ptr(i+1)-1
!                    print*,new%crs_kappa(:,k),new%crs_val(k)
!                end do
!            end do
!        end do
!        i=0
!        do k=0,YBundle%lmax
!            call cart%init(k)
!            do while(cart%next(kappa))
!                i=i+1
!                print'("<<<< ",3i3," >>>>")', kappa
!                do ilm=new%ccs_ptr(i),new%ccs_ptr(i+1)-1
!                    print*,new%ccs_lm(:,ilm),new%ccs_val(ilm)
!                end do
!            end do
!        end do
    end function

    pure subroutine Cart2Sph_Iter_destroy(self)
        class(Cart2SphIter), intent(inout) :: self

        deallocate(self%crs_val)
        deallocate(self%ccs_val)
        deallocate(self%crs_kappa)
        deallocate(self%ccs_lm)
        deallocate(self%ccs_lm_id)
        deallocate(self%crs_ptr)
        deallocate(self%ccs_ptr)
    end subroutine

    pure subroutine Cart2Sph_Iter_init_loop_over_sph(self,kappa)
        class(Cart2SphIter), intent(inout) :: self
        integer(INT32),      intent(inout) :: kappa(3)
        integer(INT32)                     :: i

        i=dvpos(kappa)
        self%next=self%ccs_ptr(i)-1
        self%last=self%ccs_ptr(i+1)
    end subroutine

    pure subroutine Cart2Sph_Iter_next_sph(self,val,l,m, continue)
        class(Cart2SphIter), intent(inout) :: self
        real(REAL64), intent(out)          :: val
        integer(INT32), intent(out)        :: l,m
        logical, intent(out)               :: continue

        self%next=self%next+1
        if(self%next==self%last) then
            continue=.FALSE.
            return
        else
            continue=.TRUE.
        end if

        val=self%ccs_val(self%next)
        l=self%ccs_lm(1,self%next)
        m=self%ccs_lm(2,self%next)
    end subroutine

    pure subroutine Cart2Sph_Iter_init_loop_over_cart(self,l,m)
        class(Cart2SphIter), intent(inout) :: self
        integer(INT32),      intent(in)    :: l,m
        integer(INT32)                     :: i

        i=l**2+l+m+1
        self%next=self%crs_ptr(i)-1
        self%last=self%crs_ptr(i+1)
    end subroutine

    function Cart2Sph_Iter_next_cart(self,val,kappa) result(continue)
        class(Cart2SphIter), intent(inout) :: self
        real(REAL64),        intent(out)   :: val
        integer(INT32),      intent(out)   :: kappa(3)
        logical                            :: continue

        self%next=self%next+1
        if(self%next==self%last) then
            continue=.FALSE.
            return
        else
            continue=.TRUE.
        end if

        continue=.TRUE.
        val=self%crs_val(self%next)
        kappa=self%crs_kappa(:,self%next)
        return
    end function


    subroutine rminus1_init(self,nmax)
        type(rminus1_t), intent(out) :: self
        integer(INT32),  intent(in)  :: nmax
        integer(INT32)               :: ider
        integer(INT32)               :: dv(3)

        type(poly3D_t) :: orig
        
        type(CartShellIter) :: cart
        logical :: continue_iteration

        orig=poly3D_t(num_terms=1,&
                terms=[mono3D_t(coeff=1.d0,expos=[0,0,0,-1])])
                                                                    
        self%sz=(nmax+1)*(nmax+2)*(nmax+3)/6
        allocate(self%polys(self%sz))

        do ider=0,nmax
            cart=CartShellIter(3,ider)
            call cart%next(dv, continue_iteration)
            do while(continue_iteration)
                self%polys(dvpos(dv))=poly3D_multider(orig,dv)
                call cart%next(dv, continue_iteration)
            end do
        end do
    end subroutine

    !> Evaluates all derivatives of 1/r at points given in pos
    pure function rminus1_eval(self, points, distances) result(res)
        type(rminus1_t), intent(in) :: self
        real(REAL64), intent(in)    :: points(:,:), distances(:)
        real(REAL64)                :: res(self%sz,size(points,2))
        integer(INT32) :: i

        do i=1,self%sz
            res(i,:)=self%polys(i)%eval(points, distances)
        end do
    end function

#ifdef HAVE_CUDA
    pure subroutine assign_c_pointers(self,nt,c,e)
        type(YBundle), intent(inout), target :: self
        integer(c_int),pointer, intent(out) :: nt(:),e(:,:)
        real(c_double),pointer, intent(out) :: c(:)

        integer :: k,kt,i,j

        allocate(nt(self%numY))
        nt(1)=self%Y(1)%poly%num_terms
        do k=2,self%numY
            nt(k)=nt(k-1)+self%Y(k)%poly%num_terms
        end do
        kt=nt(self%numY)
        allocate(c(kt))
        allocate(e(3,kt))
        j=1
        do k=1,self%numY
            do i=1,self%Y(k)%poly%num_terms
                c(j)=self%Y(k)%poly%terms(i)%coeff
                e(:,j)=self%Y(k)%poly%terms(i)%expos(1:3)
                j=j+1
            end do
        end do
        return
    end subroutine
#endif

    !> @todo Would it be possible to eliminate these two functions \c dvpos
    !! and \c dvpos_shell?
    pure function dvpos(dv) result(pos)
        !Return the position of dv in the series
        ![0 0 0],[0 0 1],[0 1 0],[1 0 0],[0 0 2],[0 1 1],[0 2 0],[1 0 1]...
        integer(INT32), intent(in) :: dv(3)
        integer(INT32)             :: pos
        pos=dvpos_shell(dv) + sum(dv)**2
        return
    end function

    pure function dvpos_shell(dv) result(pos)
        !Return the position of dv in its own sub-shell
        integer(INT32), intent(in) :: dv(3)
        integer(INT32) :: pos,l
        l=sum(dv)
        pos=(dv(1)*(3+2*l-dv(1))+2)/2 + dv(2)
        return
    end function

    !> Constructs a HarmonicIter instance.
    !!
    function HarmonicIter_init(lmax) result(new)
        !> The highest multipole shell considered
        integer, intent(in) :: lmax
        
        type(HarmonicIter) :: new

        integer :: e(3), nrows, ncol
        integer :: l, m, i, j
        type(CartIter) :: iter
        real(REAL64) :: coeff
        type(YBundle) :: yb
        type(Y_t), pointer :: Y
        logical :: continue_iteration

        new%lmax = lmax

        yb = YBundle(new%lmax)
        call yb%extend()

        iter = CartIter(3, new%lmax)
        nrows = (new%lmax + 1)**2
        ncol = iter%get_num_iter()

        ! Using lmax = 15 consumes 176.0 kB of memory
        allocate(new%cache(nrows, ncol), source=0.0d0)
        call iter%next(e, continue_iteration)
        do while(continue_iteration)
            j = lexicopos(e, new%lmax)
            l = e(1) + e(2) + e(3)
            do m = -l, l
                i = idx(l,m)
                Y => yb%pick(l,m)
                coeff = Y%get_coeff(e)
                new%cache(i, j) = coeff
            enddo
            call iter%next(e, continue_iteration)
        enddo

    end function

    !> Destroys a HarmonicIter instance
    pure subroutine HarmonicIter_destroy(self)
        class(HarmonicIter), intent(inout) :: self
        if (associated(self%cache)) deallocate(self%cache)
        self%cache => NULL()
        self%cache_line => NULL()
    end subroutine

    !> Selects the monomial (a,b,c) that we are interested in.
    !!
    !! This method must be called before iterating with the next method.
    !! See the example at the beginning of this file.
    pure subroutine HarmonicIter_loop_over(self, expo)
        class(HarmonicIter), intent(inout) :: self
        !> An exponent triplet (a,b,c)
        integer(INT32), intent(in) :: expo(3)

        integer(INT32) :: j, l

        self%l = sum(expo)
        j = lexicopos(expo, self%lmax)
        ! Select the l-shell
        l = self%l
        self%cache_line => self%cache(l*l + 1 : l*l + 1 + 2*l, j)
        self%cache_line (0:) => self%cache_line
    end subroutine

    !> The next element of the iterator.
    function HarmonicIter_next(self, coeff, l, m) result(cont)
        class(HarmonicIter), intent(in) :: self
        integer(INT32), intent(out) ::  l, m
        real(REAL64), intent(out) :: coeff
        
        logical :: cont
        integer, save :: j = -1

        do while (j < 2*self%l)
            j = j + 1
            coeff = self%cache_line(j)
            if (coeff /= 0.0) then
                l = self%l
                m = -self%l + j
                cont = .true.
                return
            endif
        enddo
        j = -1
        cont = .false.

    end function

    !>Return the lexicographic position of kappa in the series
    !! [0 0 0],[0 0 1],[0 0 2],[0 1 1],[0 2 0],[0 1 1],[0 2 0],[1 0 0]...[lmax,0,0]
    pure function lexicopos(kappa,lmax) result(pos)
        integer, intent(in) :: kappa(3), lmax
        integer :: pos
        integer :: a, b, c
        a = kappa(1)
        b = kappa(2)
        c = kappa(3)
        pos = 6+11*a-6*a*a+a*a*a+9*b-6*a*b-3*b*b+6*c
        pos = pos + 12*a*lmax - 3*a*a*lmax + 6*b*lmax + 3*a*lmax*lmax
        pos = pos/6
    end function

end module
