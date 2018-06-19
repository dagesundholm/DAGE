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
module HelmholtzMultipoleTools_class
    use ISO_FORTRAN_ENV
    use Helmholtz3D_class            ! Use Helmholtz3D_class to get the Modified Bessel Functions
    use RealSphericalHarmonics_class ! to get the lm_map function
    use Globals_m
#ifdef HAVE_OMP
    use omp_lib
#endif

    implicit none

    private
    
    public  :: HelmholtzMultipoleTranslator, HelmholtzMultipoleConverter, factorial

    type, abstract             ::  HelmholtzExpansionTranslator
        !> The maximum 'l' quantum number in input multipole expansion
        integer                                    :: input_lmax
        !> The maximum 'l' quantum number in output multipole expansion
        integer                                    :: lmax
        real(REAL64)                               :: kappa
        real(REAL64)                               :: new_scaling_factor = 1.0d0
        real(REAL64)                               :: scaling_factor = 1.0d0
        integer                                    :: normalization = 1
        class(ModSphBesselCollection), allocatable :: bessel_collection
    contains
        procedure, private :: HelmholtzExpansionTranslator_translate_complex
        procedure, private :: HelmholtzExpansionTranslator_translate_real
        procedure, private :: translate_aligned_real_multipoles => &
                                  HelmholtzExpansionTranslator_translate_aligned_real_multipoles
        procedure, private :: translate_aligned_cmplx_multipoles => &
                                  HelmholtzExpansionTranslator_translate_aligned_cmplx_multipoles
        generic, public    :: translate                    => &
                                  HelmholtzExpansionTranslator_translate_complex, &
                                  HelmholtzExpansionTranslator_translate_real
        
        procedure, public  :: destroy                      => &
                                  HelmholtzExpansionTranslator_destroy
        procedure(get_al_trans_coefficients), private, deferred :: &
                                  get_aligned_translation_coefficients
    end type

    abstract interface
       pure function get_al_trans_coefficients(self, bessel_values) result(coefficients)
           import HelmholtzExpansionTranslator, REAL64
           class(HelmholtzExpansionTranslator), intent(in) :: self
           real(REAL64), intent(in)                        :: bessel_values(0 : self%input_lmax+1)
           real(REAl64)                                    :: coefficients(0:self%lmax,  &
                                                                    0:self%input_lmax, 0:self%lmax)
       end function
    end interface
 


    !> Transfers Helmholtz Multipole from one center to another. The evaluation
    !! point for the multipole must be outside a sphere that contains the 
    !! all the charges translated to its center.
    !!
    !! The corresponding conventional multipole tool is the TranslationMatrix
    type, extends(HelmholtzExpansionTranslator) :: HelmholtzMultipoleTranslator
    contains
        procedure, private :: get_aligned_translation_coefficients => &
                                  HelmholtzMultipoleTranslator_get_al_trans_coefficients
    end type

    interface HelmholtzMultipoleTranslator
        module procedure :: HelmholtzMultipoleTranslator_init
    end interface

    !> Converts Helmholtz Multipole Expansion to Local Helmholtz Expansion
    !!
    !! The corresponding conventional multipole tool is the InteractionMatrix.
    type, extends(HelmholtzExpansionTranslator) :: HelmholtzMultipoleConverter
    contains
        procedure, private :: get_aligned_translation_coefficients => &
                                  HelmholtzMultipoleConverter_get_al_trans_coefficients
    end type

    interface HelmholtzMultipoleConverter
        module procedure :: HelmholtzMultipoleConverter_init
    end interface
 
    !> Evaluates Local Expansion at the neighbouring points of the center of the  
    !! expansion 
    !!
    !! There is no corresponding tool for the conventional multipoles.
    type HelmholtzLocalTranslator
    end type
contains

    

!-------------------------------------------------------------------------------!
! HelmholtzExpansionTranslator methods                                          !
!-------------------------------------------------------------------------------!

    subroutine HelmholtzExpansionTranslator_destroy(self)
        class (HelmholtzExpansionTranslator), intent(inout) :: self
        if (allocated(self%bessel_collection)) then
            call self%bessel_collection%destroy()
            deallocate(self%bessel_collection)
        end if
    end subroutine  

    function HelmholtzExpansionTranslator_translate_complex(self, multipoles, coordinates, destination) result(reslt_multipoles)
        class (HelmholtzExpansionTranslator), intent(in) :: self
        complex*16,                           intent(in) :: multipoles(:, :)
        real(REAL64),                         intent(in) :: coordinates(3, size(multipoles, 2))
        real(REAL64),                         intent(in) :: destination(3)
        complex*16                                       :: new_multipoles((self%input_lmax+1)**2, size(multipoles, 2)), &
                                                            reslt_multipoles((self%lmax+1)**2, size(multipoles, 2))
        real(REAL64)                                     :: bessel_values(size(multipoles, 2), 0 : self%input_lmax+1)
        real(REAL64)                                     :: distances(size(coordinates, 2))
        real(REAL64)                                     :: alpha, beta, gamma
        real(REAL64)                                     :: real_multipoles(size(multipoles, 1))
        integer                                          :: i, l

        forall (i = 1 : size(coordinates, 2))
            ! calculate the distance from the destination
            distances(i) = sqrt((coordinates(1, i)-destination(1))**2 + &
                (coordinates(2, i)-destination(2))**2 + (coordinates(3, i)-destination(3))**2)
        end forall 

        bessel_values = self%bessel_collection%eval(self%kappa * distances)

        do i = 1, size(multipoles, 2) 
            ! calculate euler angles for coordinate system rotation from normal to a system where 
            ! z-vector is aligned with the destination coordinate vector
            call get_euler_angles_for_new_z_axis((destination - coordinates(:, i)) / distances(i), alpha, beta, gamma)
  
            ! rotate the coordinate system so that the z-axis is aligned with the vector from
            ! origo to the multipole coordinates
            new_multipoles(:, i) = rotate_complex_spherical_harmonic_expansion(multipoles(:, i), alpha, beta, gamma, self%lmax)

            ! translate the multipoles with the formula for the z-aligned multipoles
            reslt_multipoles(:, i) =  self%translate_aligned_cmplx_multipoles(new_multipoles(:, i), &
                                      bessel_values(i, :), self%kappa * distances(i))

 
            ! rotate back, the new axis system should be aligned with the N-vector (see Euler angle wikipedia
            ! for definition) and thus, there is no need to rotate beforehand (gamma = 0), -alpha as (gamma) rotates
            ! the x and y-axis back to the original positions and -beta rotates the z-vector back to its original
            ! position (cartesian)
            reslt_multipoles(:, i) = rotate_complex_spherical_harmonic_expansion(reslt_multipoles(:, i), -gamma, -beta, &
                                                                                -alpha, self%lmax)

        end do
    end function 

    function HelmholtzExpansionTranslator_translate_real(self, multipoles, coordinates, destination) result(reslt_multipoles)
        class (HelmholtzExpansionTranslator)             :: self
        real(REAL64),                         intent(in) :: multipoles(:, :)
        real(REAL64),                         intent(in) :: coordinates(3, size(multipoles, 2))
        real(REAL64),                         intent(in) :: destination(3)
        real(REAL64)                                     :: new_multipoles(size(multipoles, 1), size(multipoles, 2)), &
                                                            multipoles2(size(multipoles, 1)), &
                                                            reslt_multipoles((self%lmax+1)**2, size(multipoles, 2))
        complex*16                                       :: complex_multipoles(size(multipoles, 1))
        real(REAL64)                                     :: bessel_values(size(multipoles, 2), 0 : self%input_lmax + 1)
        real(REAL64)                                     :: distances(size(coordinates, 2))
        real(REAL64)                                     :: alpha(size(coordinates, 2)), beta(size(coordinates, 2)), &
                                                            gamma(size(coordinates, 2))
        integer                                          :: i, l

        
        if (self%normalization /= 2) then
            forall(l = 0 : self%input_lmax)
                new_multipoles(l*l + 1 : l*l + 1 + 2*l, :) = sqrt((2*l+1)/dble(4*pi)) &
                     * multipoles(l*l + 1 : l*l + 1 + 2*l, :)
            end forall
        else
            new_multipoles = multipoles
        end if


#ifdef HAVE_OMP
        !$OMP PARALLEL DO
#endif
        do i = 1, size(coordinates, 2)
            ! calculate the distance from the destination
            distances(i) = dsqrt((coordinates(1, i)-destination(1))**2 + &
                (coordinates(2, i)-destination(2))**2 + (coordinates(3, i)-destination(3))**2)
        end do
#ifdef HAVE_OMP
        !$OMP END PARALLEL DO
#endif

        !self%bessel_collection%scaling_factor = self%kappa*distances(1)
        bessel_values = self%bessel_collection%eval(self%kappa * distances)

        if (any(isNan(bessel_values))) then
            print *, "bessels has nans"
        end if

        if (any(isNan(new_multipoles))) then
            print *, "scaled input multipoles has nans"
        end if

        !call bigben%split("Translate")
#ifdef HAVE_OMP
            !$OMP PARALLEL DO &
            !$OMP& PRIVATE(l) 
#endif
        do i = 1, size(multipoles, 2)
            ! calculate euler angles for coordinate system rotation from normal to a system where 
            ! z-vector is aligned with the destination coordinate vector
            !call bigben%split("Rotate in")
            call get_euler_angles_for_new_z_axis((destination - coordinates(:, i))  &
                 / distances(i), alpha(i), beta(i), gamma(i))
 
            ! rotate the coordinate system so that the z-axis is aligned with the vector from
            ! origo to the multipole coordinates
            new_multipoles(:, i) = rotate_real_spherical_harmonic_expansion(new_multipoles(:, i), &
                                       alpha(i), beta(i), gamma(i), self%input_lmax)
            !call bigben%stop_and_print()

            !if (any(isNan(new_multipoles(:, i)))) then
            !    print *, "rotated multipoles has nans", i, distances(i)
            !end if
  
            !call bigben%split("Translate")
            ! translate the multipoles with the formula for the z-aligned multipoles
            reslt_multipoles(:, i) = self%translate_aligned_real_multipoles(new_multipoles(:, i), &
                                    bessel_values(i, :), self%kappa * distances(i))
            

            !if (any(isNan(reslt_multipoles(:, i)))) then
            !    print *, "rotated & translated multipoles has nans", i, distances(i)
            !end if

            ! rotate back, the new axis system should be aligned with the N-vector (see Euler angle wikipedia
            ! for definition) and thus, there is no need to rotate beforehand (gamma = 0), -alpha as (gamma) rotates
            ! the x and y-axis back to the original positions and -beta rotates the z-vector back to its original
            ! position (cartesian)
            
            reslt_multipoles(:, i) = rotate_real_spherical_harmonic_expansion(reslt_multipoles(:, i), &
                                        -gamma(i), -beta(i), -alpha(i), self%lmax)
            !call bigben%stop_and_print()
            !if (any(isNan(reslt_multipoles(:, i)))) then
            !    print *, "rotated & translated & rerotated multipoles has nans", i, distances(i)
            !end if

            
            if (self%normalization /= 2) then
                forall(l = 0 : self%lmax)
                    reslt_multipoles(l*l + 1 : l*l + 1 + 2*l, i) = &
                        sqrt((4*pi)/dble(2*l+1)) * reslt_multipoles(l*l + 1 : l*l + 1 + 2*l, i)
                end forall
            end if

        end do
#ifdef HAVE_OMP
        !$OMP END PARALLEL DO
#endif
        !call bigben%stop()
    end function 

    pure function HelmholtzExpansionTranslator_translate_aligned_cmplx_multipoles(self, multipoles, bessel_values, kappa_distance) &
                      result(new_multipoles)
        class (HelmholtzExpansionTranslator), intent(in) :: self
        complex*16,   intent(in)                         :: multipoles((self%input_lmax+1)**2)
        real(REAL64), intent(in)                         :: bessel_values(0 : self%input_lmax+1)
        real(REAL64), intent(in)                         :: kappa_distance
        complex*16                                       :: new_multipoles((self%lmax+1)**2)
        real(REAL64)                                     :: translation_coefficients(0:self%lmax, 0:self%input_lmax, 0:self%lmax)
        integer                                          :: id, id2, l, m, n, input_lmax


        translation_coefficients = self%get_aligned_translation_coefficients(bessel_values)
        !if (any(isNan(translation_coefficients))) then
        !    print *, "translation coefficients has nans"
        !end if 
        do l = 0, self%lmax
            do m = -l, l
                id = lm_map(l, m)
                new_multipoles(id) = translation_coefficients(l, abs(m), abs(m))              &
                                        * multipoles(lm_map(abs(m), m)) * self%scaling_factor**abs(m) &
                                        * self%new_scaling_factor ** (-l) 

                
                do n = abs(m)+1, self%input_lmax
                    new_multipoles(id) = new_multipoles(id) + &
                                    translation_coefficients(l, n, abs(m))   &
                                    * multipoles(lm_map(n, m)) * self%scaling_factor**n  &
                                    * self%new_scaling_factor ** (-l) 
                end do  
            end do
        end do 
    end function

    pure function HelmholtzExpansionTranslator_translate_aligned_real_multipoles(self, multipoles, bessel_values, kappa_distance) &
                      result(new_multipoles)
        class (HelmholtzExpansionTranslator), intent(in) :: self
        real(REAL64), intent(in)                         :: multipoles((self%input_lmax+1)**2)
        real(REAL64), intent(in)                         :: bessel_values(0 : 3*self%lmax)
        real(REAL64), intent(in)                         :: kappa_distance
        real(REAL64)                                     :: new_multipoles((self%lmax+1)**2)
        complex*16                                       :: complex_multipoles(size(multipoles)), &
                                                            new_complex_multipoles((self%lmax+1)**2)

        ! convert the aligned real multipole expansion to complex 
        complex_multipoles = convert_real_spherical_harmonic_expansion_to_complex(multipoles, self%input_lmax)
 
        ! and translate it along the z-axis 
        new_complex_multipoles = self%translate_aligned_cmplx_multipoles(complex_multipoles, bessel_values, kappa_distance)
        
        ! convert the translated multipole expansion back to real
        new_multipoles = convert_complex_spherical_harmonic_expansion_to_real(new_complex_multipoles, self%lmax)
    end function


!-------------------------------------------------------------------------------!
! HelmholtzMultipoleTranslator methods                                          !
!-------------------------------------------------------------------------------!

    function HelmholtzMultipoleTranslator_init(lmax, kappa, input_lmax, &
                                               normalization, scaling_factor) result(new)
        !> Maximum angular momentum quantum number 'l' value 
        integer,      intent(in)           :: lmax
        !> The Helmholtz operator 'kappa' parameter
        real(REAL64), intent(in)           :: kappa
        !> Maximum angular momentum quantum number 'l' value in input expansion 
        integer,      intent(in)           :: input_lmax
        !> The type of normalization used in the converter
        integer,      intent(in), optional :: normalization
        !> The scaling factor used in input expansion
        real(REAL64), intent(in), optional :: scaling_factor
        type(HelmholtzMultipoleTranslator) :: new

        integer                            :: l

        
        new%lmax = lmax
        new%input_lmax = input_lmax
        new%kappa = kappa
        allocate(new%bessel_collection, source = FirstModSphBesselCollection(input_lmax+1))

        ! 1 is the Racah normalization and 2 is the conventional normalization
        if (present(normalization)) then
            new%normalization = normalization
        end if

        if (present(scaling_factor)) then
            new%scaling_factor = scaling_factor
            new%new_scaling_factor = scaling_factor
        end if
    end function    

    pure function HelmholtzMultipoleTranslator_get_al_trans_coefficients(self, bessel_values) &
                 result(coefficients)
        class(HelmholtzMultipoleTranslator), intent(in) :: self
        real(REAL64),                        intent(in) :: bessel_values(0 : self%input_lmax+1)
        real(REAL64)                                    :: coefficients(0:self%lmax,  &
                                                                    0:self%input_lmax, 0:self%lmax)
     
        coefficients = get_all_aligned_translation_coefficients(bessel_values, self%lmax, self%input_lmax)     
    end function 

!-------------------------------------------------------------------------------!
! HelmholtzMultipoleConverter methods                                           !
!-------------------------------------------------------------------------------!

    function HelmholtzMultipoleConverter_init(lmax, kappa, input_lmax, &
                                              normalization, scaling_factor) result(new)
        !> Maximum angular momentum quantum number 'l' value 
        integer,      intent(in)           :: lmax
        !> The Helmholtz operator 'kappa' parameter
        real(REAL64), intent(in)           :: kappa
        !> Maximum angular momentum quantum number 'l' value in input expansion 
        integer,      intent(in)           :: input_lmax
        !> The type of normalization used in the converter
        integer,      intent(in), optional :: normalization
        !> The scaling factor used in input expansion
        real(REAL64), intent(in), optional :: scaling_factor
        !> The result object
        type(HelmholtzMultipoleConverter)  :: new

        integer                            :: l

        
        new%lmax = lmax
        new%input_lmax = input_lmax
        new%kappa = kappa
        allocate(new%bessel_collection, source = SecondModSphBesselCollection(input_lmax+1))
        ! 1 is the Racah normalization and 2 is the conventional normalization
        if (present(normalization)) then
            new%normalization = normalization
        end if

        if (present(scaling_factor)) then
            new%new_scaling_factor = 1 / scaling_factor
            new%scaling_factor = scaling_factor
        end if
    end function   

    pure function HelmholtzMultipoleConverter_get_al_trans_coefficients(self, bessel_values) &
                 result(coefficients)
        class(HelmholtzMultipoleConverter), intent(in) :: self
        real(REAL64), intent(in)                       :: bessel_values(0 : self%input_lmax+1)
        real(REAL64)                                   :: coefficients(0:self%lmax,  &
                                                                    0:self%input_lmax, 0:self%lmax)

        coefficients = get_all_aligned_ml_translation_coefficients(bessel_values, self%lmax, self%input_lmax)        
    end function

    pure function get_all_aligned_ml_translation_coefficients(bessel_values, lmax, input_lmax) result(result_coefficients)
        !> Array of values of modified spherical bessels of 1st or second kind for value 'kappa*distance'
        !! In multipole to local expansion translation these are of 2nd kind
        real(REAL64), intent(in) :: bessel_values(0 : input_lmax+1)
        !> The maximum 'l' quantum number in output multipole expansion
        integer,      intent(in) :: lmax
        !> The maximum 'l' quantum number in input multipole expansion
        integer,      intent(in) :: input_lmax
        
        real(REAL64)             :: coefficients(-1:input_lmax+1, -1:input_lmax+1, -1:input_lmax+1), &
                                    result_coefficients(0:lmax, 0:input_lmax, 0:lmax), &
                                    a_coefficients(-1:input_lmax+1, -input_lmax-1:input_lmax+1), &
                                    b_coefficients(-1:input_lmax+1, -input_lmax-1:input_lmax+1)
        integer                  :: l, m, n

        
        coefficients = 0.0d0
        a_coefficients = 0.0d0
        b_coefficients = 0.0d0
        ! Generate all translation coefficients for conventional Spherical Harmonics normalization
        
        forall(n = 0:input_lmax+1, m=-input_lmax-1:input_lmax+1)
            a_coefficients(n, m) = a_coefficient(n, m)
            b_coefficients(n, m) = b_coefficient(n, m) 
        end forall


       
        coefficients(0, 0, 0) = bessel_values(0)

        forall (n=1:input_lmax+1)
            coefficients(n, 0, 0) =  (-1)**n * sqrt(dble(2*n+1))* bessel_values(n)
            coefficients(0, n, 0) =  sqrt(dble(2*n+1))* bessel_values(n)
        end forall 

        
        ! now we have (l, 0, 0) and (0, n, 0)
        l = 0
        m = 0
        do n = 1, input_lmax
            ! the order of the calculations must be this, (l+1, n, m) before (l, n+1, m) 
            coefficients(1, n, 0) =  (-a_coefficients(n, m)   * coefficients(l, n+1, m)  &
                                      -a_coefficients(n-1, m) * coefficients(l, n-1, m))  &
                                       / (a_coefficients(l, m))
            
            coefficients(n, 1, 0) = (-1)**(1+n) * coefficients(1, n, 0)
        end do

        n = 0
        ! now we have (l, 0, 0), (0, n, 0), (l, 1, 0) and (1, n, 0)
        do m = 0, input_lmax
            ! calculate all other (l, n, m) coefficients for fixed m
            ! for the first round we need (l-1, m, m), (l+1, m, m) and (l, m-1, m)
            ! we have these due to the last loop in this loop and the ones before this.
            do n = m, input_lmax
                ! for the round we need (l+1, n, m), (l-1, n, m), (l, n-1, m)
                ! we get (l, n+1, m)

                do l = m, input_lmax
                    coefficients(l, n+1, m) =  ( - a_coefficients(l, m)   * coefficients(l+1, n, m)  &
                                                 - a_coefficients(l-1, m) * coefficients(l-1, n, m)  &
                                               -  a_coefficients(n-1, m) * coefficients(l, n-1, m)) &
                                               / (a_coefficients(n, m))
                end do
            end do
            ! advance with respect to m, i.e., get the (l, m, m) for the next round
            do l = m, input_lmax
                coefficients(l, m+1, m+1) = ( - b_coefficients(l, -m-1) * coefficients(l-1, m, m)     &
                                               - b_coefficients(l+1, m)  * coefficients(l+1, m, m)) &
                                            / ( b_coefficients(m+1, -m-1) )

            end do 
        end do 
 
        !do n = 0, lmax, 2
        !    do l = 1, lmax, 2 
        !        coefficients(n, l, :) = (-1) * coefficients(n, l, :) 
        !    end do 
        !end do


        result_coefficients = coefficients(0:lmax, 0:input_lmax, 0:lmax)
         
    end function


    pure function get_all_aligned_translation_coefficients(bessel_values, &
                lmax, input_lmax) result(result_coefficients)
        !> Array of values of modified spherical bessels of 1st or second kind for value 'kappa*distance'
        !! In multipole to multipole translation these are of 1st kind
        real(REAL64), intent(in) :: bessel_values(0 : input_lmax+1)
        !> The maximum 'l' quantum number in output multipole expansion
        integer,      intent(in) :: lmax
        !> The maximum 'l' quantum number in input multipole expansion
        integer,      intent(in) :: input_lmax
        
        real(REAL64)             :: coefficients(-1:input_lmax+1, -1:input_lmax+1, -1:input_lmax+1), &
                                    result_coefficients(0:lmax, 0:input_lmax, 0:lmax), &
                                    a_coefficients(-1:input_lmax+1, -input_lmax-1:input_lmax+1), &
                                    b_coefficients(-1:input_lmax+1, -input_lmax-1:input_lmax+1)
        integer                  :: l, m, n

        
        coefficients = 0.0d0
        ! Generate all translation coefficients for conventional Spherical Harmonics normalization
        
        forall(n = 0:input_lmax+1, m=-input_lmax-1:input_lmax+1)
            a_coefficients(n, m) = a_coefficient(n, m)
            b_coefficients(n, m) = b_coefficient(n, m) 
        end forall


        forall (n=0:input_lmax+1)
            coefficients(n, 0, 0) =  (-1)**(n) * sqrt(dble(2*n+1))* bessel_values(n)
            coefficients(0, n, 0) = (-1)**(n) * coefficients(n, 0, 0)
        end forall 


        
        ! now we have (l, 0, 0) and (0, n, 0)
        l = 0
        m = 0
        do n = 1, input_lmax
            ! the order of the calculations must be this, (l+1, n, m) before (l, n+1, m) 
            coefficients(1, n, 0) = (a_coefficients(n, 0)   * coefficients(0, n+1, 0)  &
                                     + a_coefficients(n-1, 0) * coefficients(0, n-1, 0))  &
                                       / a_coefficients(l, 0)
            
            coefficients(n, 1, 0) = (-1)**(1+n) * coefficients(1, n, 0)
            
        end do

        n = 0
        
        ! now we have (l, 0, 0), (0, n, 0), (l, 1, 0) and (1, n, 0)
        do m = 0, input_lmax
            ! calculate all other (l, n, m) coefficients for fixed m
            ! for the first round we need (l-1, m, m), (l+1, m, m) and (l, m-1, m)
            ! we have these due to the last loop in this loop and the ones before this.
            do n = m, input_lmax
                ! for the round we need (l+1, n, m), (l-1, n, m), (l, n-1, m)
                ! we get (l, n+1, m)
                do l = m, input_lmax
                    coefficients(l, n+1, m) = (a_coefficients(l, m)   * coefficients(l+1, n, m)  &
                                               + a_coefficients(l-1, m) * coefficients(l-1, n, m)  &
                                               - a_coefficients(n-1, m) * coefficients(l, n-1, m)) &
                                               / a_coefficients(n, m)
          
                end do
            end do

            ! advance with respect to m, i.e., get the (l, m, m) for the next round
            do l = m, input_lmax
                coefficients(l, m+1, m+1) = (b_coefficients(l, -m-1) * coefficients(l-1, m, m)     &
                                              + b_coefficients(l+1, m)  * coefficients(l+1, m, m)) &
                                            / b_coefficients(m+1, -m-1)
            end do 
        end do 

        result_coefficients = coefficients(0:lmax, 0:input_lmax, 0:lmax)
        
    end function

    pure function a_coefficient(n, m)
        integer, intent(in)  :: n, m
        real(REAL64)         :: a_coefficient 
        a_coefficient = dsqrt(dble((n+1+abs(m))*(n+1-abs(m)))/((2*n+1)*(2*n+3)))
    end function

    pure function b_coefficient(n, m)
        integer, intent(in)  :: n, m
        real(REAL64)         :: b_coefficient 

        b_coefficient = dsqrt(dble((n-m-1)*(n-m))/((2*n-1)*(2*n+1)))
        if (abs(m) > n) then
            b_coefficient = 0
        else if (m < 0) then
            b_coefficient = -b_coefficient
        end if
    end function

    pure function a_coefficient_racah(n, m) result(a_coefficient)
        integer, intent(in)  :: n, m
        real(REAL64)         :: a_coefficient 
        !a_coefficient = dsqrt(dble((n+1+abs(m))*(n+1-abs(m)))/((2*n+1)*(2*n+3))) &
        !                 * dsqrt(dble(2*n+1)/(2*n+3))
        a_coefficient = dsqrt(dble((n+1-abs(m))*(n+1+abs(m))))/dble(2*n+3)
    end function

    pure function b_coefficient_racah(n, m) result(b_coefficient)
        integer, intent(in)  :: n, m
        real(REAL64)         :: b_coefficient 
        !b_coefficient = dsqrt(dble((n-m-1)*(n-m))/((2*n-1)*(2*n+1))) &
        !                * dsqrt(dble(2*n-1) / (2*n+3))
        b_coefficient = dsqrt(dble((n-m-1)*(n-m)))/(2*n+1)
        if (abs(m) > n) then
            b_coefficient = 0
        else if (m < 0) then
            b_coefficient = -b_coefficient
        end if
    end function
        

    function get_aligned_function_mm_translation_coefficient(l, n, m, kappa_distance, bessel_values, lmax) result(coefficient)
        integer,      intent(in) :: l, n, m, lmax
        !> Helmholtz parameter times the translation distance for the current point
        real(REAL64), intent(in) :: kappa_distance
        !> Array of values of modified spherical bessels of xth kind for value 'kappa*distance'
        real(REAL64), intent(in) :: bessel_values(0 : 3*lmax)
        integer                  :: k
        real(REAL64)             :: coefficient
        real(REAL64)             :: fact_l_m_m, fact_n_p_m

        coefficient = 0.0d0


        ! get values for (n-m)! and (n+m)!
        fact_l_m_m = factorial(l - m)
        fact_n_p_m = factorial(n + m)
        do k = m, min(l, n)
            coefficient = coefficient + 0.5d0**(k) * (-1.0d0)**(n+l) * (2.0d0*l + 1.0d0)                     &
                           * dble(fact_l_m_m * fact_n_p_m * factorial(2*k)) * (kappa_distance) ** (-k)     &
                           * bessel_values(l+n-k)                                                     &
                           / dble( factorial(k+m) * factorial(k-m) * factorial(n-k) * factorial(l-k) * &
                               factorial(k) ) 
        end do
    end function

    function get_aligned_function_ll_translation_coefficient(l, n, m, kappa_distance, bessel_values) result(coefficient)
        integer,      intent(in) :: l, n, m
        !> Helmholtz parameter times the distance for the current point
        real(REAL64), intent(in) :: kappa_distance
        !> Array of values of modified spherical bessels of xth kind for value 'kappa*distance'
        real(REAL64), intent(in) :: bessel_values(:)
        integer                  :: k
        real(REAL64)             :: coefficient
        real(REAL64)             :: fact_n_m_m, fact_l_p_m

        coefficient = 0.0d0

        ! get values for (n-m)! and (n+m)!
        fact_n_m_m = factorial(n - m)
        fact_l_p_m = factorial(l + m)
        do k = m, min(l, n)
            coefficient = coefficient + 0.5d0**k * (2.0d0*n+1.0d0)                                     &
                           * dble(fact_n_m_m * fact_l_p_m * factorial(2*k)) * (kappa_distance) ** (-k) &
                           * bessel_values(l+n-k)                                                     &
                           / dble( factorial(k+m) * factorial(k-m) * factorial(n-k) * factorial(l-k) * &
                               factorial(k) ) 
        end do
    end function

    
    pure elemental function factorial(x) result(reslt)
         integer, intent(in)         :: x
         integer(kind=16)            :: i
         real(REAL64)                :: reslt
         

         reslt = 1
         do i = 1, x
             reslt = reslt * i 
         end do
    end function

    pure subroutine get_euler_angles_for_new_z_axis(z_vector, alpha, beta, gamma)
        real(REAL64), intent(in)    :: z_vector(3)
        real(REAL64), intent(out)   :: alpha, beta, gamma
        real(REAL64)                :: normalized_vector(3), xy_projection(3)
       
        beta = acos(z_vector(3))
        alpha = atan(z_vector(2), z_vector(1))
        gamma = 0.0d0
    end subroutine

    pure function cross(a, b)
        real(REAL64)        :: cross(3)
        real(REAL64), INTENT(IN) :: a(3), b(3)

        cross(1) = a(2) * b(3) - a(3) * b(2)
        cross(2) = a(3) * b(1) - a(1) * b(3)
        cross(3) = a(1) * b(2) - a(2) * b(1)
    end function

    pure function rotate_complex_spherical_harmonic_expansion(expansion_values, alpha, beta, gamma, lmax) result(reslt)
        !> Complex Spherical Harmonics expansion values, can be for instance (helmholtz) multipole moments
        !! First index is l, and the second is the center
        complex*16, intent(in)      :: expansion_values(:)
        !> The euler rotation angles, see Euler-angle wikipedia for meanings
        real(REAL64), intent(in)    :: alpha, beta, gamma
        !> The maximum angular momentum number in the expansion
        integer,      intent(in)    :: lmax
        complex*16                  :: reslt(size(expansion_values))
        complex*16                  :: sph_harm_rot_coefficients(-lmax:lmax, -lmax:lmax, 0:lmax)
        integer                     :: id, l, m, mp

        sph_harm_rot_coefficients = get_complex_spherical_harmonic_rotation_coefficients(alpha, beta, gamma, lmax)
        reslt = 0.0d0

        do l = 0, lmax
            do mp = -l, l
                id = lm_map(l, mp)
                do m = -l, l
                    reslt(id) = reslt(id) + sph_harm_rot_coefficients(m, mp, l) * expansion_values(lm_map(l, m))
                end do
            end do
        end do
    end function

    pure function rotate_real_spherical_harmonic_expansion(expansion_values, alpha, beta, gamma, lmax) result(reslt)
        !> Real Spherical Harmonics expansion values, can be for instance (helmholtz) multipole moments
        !! First index is l, and the second is the center
        real(REAL64), intent(in)    :: expansion_values(:)
        real(REAL64), intent(in)    :: alpha, beta, gamma
        integer,      intent(in)    :: lmax
        real(REAL64)                :: reslt(size(expansion_values))
        real(REAL64)                :: sph_harm_rot_coefficients(-lmax:lmax, -lmax:lmax, 0:lmax)
        integer                     :: l, m, mp, id
        
        sph_harm_rot_coefficients = get_real_spherical_harmonic_rotation_coefficients(alpha, beta, gamma, lmax)
        !call verify_real_sph_harmo_rot_coeff(sph_harm_rot_coefficients, alpha, beta, gamma, lmax)
        reslt = 0.0d0

        do l = 0, lmax
            forall (m = -l : l)
                reslt(lm_map(l, m)) = sum(sph_harm_rot_coefficients(m, -l:l, l) * expansion_values(l*l+1:(l+1)*(l+1))) 
            end forall
        end do


    end function

    pure function convert_real_spherical_harmonic_expansion_to_complex(expansion_values, lmax) &
                      result(complex_expansion)
        real(REAL64), intent(in) :: expansion_values((lmax+1)**2)
        integer, intent(in)      :: lmax
        complex*16               :: complex_expansion((lmax+1)**2)
        integer                  :: m, l
 
        do l = 0, lmax
            do m = -l, -1
                complex_expansion(lm_map(l, m)) = 1.0d0 / dsqrt(2.0d0) * &
                    complex(expansion_values(lm_map(l, -m)), -expansion_values(lm_map(l, m)))
            end do
            complex_expansion(lm_map(l, 0)) = complex(expansion_values(lm_map(l, 0)), 0.0d0)
            do m = 1, l 
                complex_expansion(lm_map(l, m)) = (-1.0d0)**m / dsqrt(2.0d0) * &
                    complex(expansion_values(lm_map(l, m)), expansion_values(lm_map(l, -m)))
            end do
        end do
        
    end function

    pure function convert_complex_spherical_harmonic_expansion_to_real(expansion_values, lmax) &
                      result(real_expansion)
        complex*16, intent(in) :: expansion_values((lmax+1)**2)
        integer,    intent(in) :: lmax
        real(REAL64)           :: real_expansion((lmax+1)**2)
        integer                :: m, l

        do l = 0, lmax
            do m = -l, -1
                real_expansion(lm_map(l, m)) = -1/dsqrt(2.0d0) * ( IMAGPART(expansion_values(lm_map(l, m))) &
                    - (-1.0d0)**m * IMAGPART(expansion_values(lm_map(l, -m))))
            end do
            real_expansion(lm_map(l, 0)) = REALPART(expansion_values(lm_map(l, 0))) 
           
            do m = 1, l
                real_expansion(lm_map(l, m)) = 1/dsqrt(2.0d0) * ( REALPART(expansion_values(lm_map(l, -m))) + &
                    (-1.0d0)**m  * REALPART(expansion_values(lm_map(l, m))))
            end do
        end do
    end function

    !> Gets the coefficient matrix 'D^l_{mm'}' used in rotation of complex spherical harmonics. The 
    !! result will be a 3 dimensional array where the first index is for 'l', second for 'm' and third
    !! for 'm'', i.e., 'mp'.
    pure function get_complex_spherical_harmonic_rotation_coefficients(alpha, beta, gamma, lmax) result(reslt)
        !> The euler angles for rotation in radians
        real(REAL64), intent(in)   :: alpha, beta, gamma
        !> The maximum angular momentum quantum number l
        integer,      intent(in)   :: lmax
        ! the angular momentum degree l, and corresponding order numbers m and m' 
        integer                    :: l, m, mp
        complex*16                 :: reslt(-lmax:lmax, -lmax:lmax, 0:lmax)
        real(REAL64)               :: rot_coeff_matrices(-lmax:lmax, 0:lmax, 0:lmax)

        rot_coeff_matrices = get_rotation_coefficient_matrices(beta, lmax)
        do l = 0, lmax
            ! go through all m values
            do m = -l, -1
                ! go through all mp values
                forall (mp = -l : -1)
                    reslt(l, m, mp) =   exp(complex(0.0d0, -m*alpha))           &
                                      * rot_coeff_matrices(-m, -mp, l) &
                                      * exp(complex(0.0d0, -mp*gamma))
                end forall
  
                reslt(l, m, 0)     =  exp(complex(0.0d0, -m*alpha))           &
                                      * rot_coeff_matrices(-m, 0, l)             
                forall (mp = 1 : l)
                    reslt(l, m, mp) =   exp(complex(0.0d0, -m*alpha))            &
                                      * (-1)**(m+mp) * rot_coeff_matrices(m, mp, l) &
                                      * exp(complex(0.0d0, -mp*gamma))
                end forall
            end do

            do m = 0, l
                ! go through all mp values
                forall (mp = -l : l)
                    reslt(l, m, mp) =   exp(complex(0.0d0, -m*alpha))           &
                                      * rot_coeff_matrices(mp, m, l) &
                                      * exp(complex(0.0d0, -mp*gamma))
                end forall
            end do
        end do 
        
        
    end function

    subroutine verify_real_sph_harmo_rot_coeff(reslt, alpha, beta, gamma, lmax)
        real(REAL64), intent(in) :: reslt(0:lmax, -lmax:lmax, -lmax:lmax)
        real(REAL64), intent(in) :: alpha, beta, gamma
        integer,      intent(in) :: lmax
        real(REAL64)             :: THRESHOLD = 1e-10, value

        print *, "verifying real sph harmo rot coeffs"
 
        value = dcos(alpha) * dcos(gamma) - dsin(alpha)*dsin(gamma)*cos(beta)
        print *, abs(reslt(1, -1, -1) - value), value, reslt(1, -1, -1) 
        if (abs(reslt(1, -1, -1) - value) > THRESHOLD) then
            print *, "DELTA rot coeff error (1, -1, -1)", value, reslt(1, -1, -1) 
        end if

        
        value = dsin(alpha) * dsin(beta)
        print *, abs(reslt(1, -1, 0) - value), value, reslt(1, -1, 0)
        if (abs(reslt(1, -1, 0) - value) > THRESHOLD) then
            print *, "DELTA rot coeff error (1, -1, 0)", value, reslt(1, -1, 0) 
        end if

        value = dcos(alpha) * dsin(gamma) + dsin(alpha) * dcos(gamma) * dcos(beta)
        print *, abs(reslt(1, -1, 1) - value), value, reslt(1, -1, 1)
        if (abs(reslt(1, -1, 1) - value) > THRESHOLD) then
            print *, "DELTA rot coeff error (1, -1, 1)", value, reslt(1, -1, 1) 
        end if

       value = dsin(gamma) * dsin(beta)
        print *, abs(reslt(1, 0, -1) - value), value, reslt(1, 0, -1)
        if (abs(reslt(1, 0, -1) - value) > THRESHOLD) then
            print *, "DELTA rot coeff error (1, 0, -1)", value, reslt(1, 0, -1) 
        end if

        value = dcos(beta)
        print *, abs(reslt(1, 0, 0) - value), value, reslt(1, 0, 0)
        if (abs(reslt(1, 0, 0) - value) > THRESHOLD) then
            print *, "DELTA rot coeff error (1, 0, 0)", value, reslt(1, 0, 0) 
        end if

        value = -dcos(gamma) * dsin(beta)
        print *, abs(reslt(1, 0, 1) - value), value, reslt(1, 0, 1)
        if (abs(reslt(1, 0, 1) - value) > THRESHOLD) then
            print *, "DELTA rot coeff error (1, 0, 1)", value, reslt(1, 0, 1) 
        end if

        value = -dcos(alpha) * dsin(gamma) * dcos(beta) - dsin(alpha) * dcos(gamma)
        print *, abs(reslt(1, 1, -1) - value), value, reslt(1, 1, -1)
        if (abs(reslt(1, 1, -1) - value) > THRESHOLD) then
            print *, "DELTA rot coeff error (1, 1, -1)", value, reslt(1, 1, -1) 
        end if

        value = dcos(alpha) * dsin(beta) 
        print *, abs(reslt(1, 1, 0) - value), value, reslt(1, 1, 0)
        if (abs(reslt(1, 1, 0) - value) > THRESHOLD) then
            print *, "DELTA rot coeff error (1, 1, 0)", value, reslt(1, 1, 0) 
        end if

        value = dcos(alpha) * dcos(gamma) * dcos(beta) - dsin(alpha) * dsin(gamma)
        print *, abs(reslt(1, 1, 1) - value), value, reslt(1, 1, 1)
        if (abs(reslt(1, 1, 1) - value) > THRESHOLD) then
            print *, "DELTA rot coeff error (1, 1, 1)", value, reslt(1, 1, 1) 
        end if
 
    end subroutine


    !> In the rotation of real spherical harmonics expansion, the rotated expansion coefficients of degree
    !! 'l' can be represented as a linear combination of the original expansion components of degree 'l'.
    !! This function calculates the coefficients for each expansion coefficient and returns them as an
    !! 3 dimensional array with indices in order: l, m, m', where m is the original angular momentum quantum
    !! number and m' is the new.
    pure function get_real_spherical_harmonic_rotation_coefficients(alpha, beta, gamma, lmax) result(reslt)
        !> The euler angles for rotation in radians
        real(REAL64), intent(in)   :: alpha, beta, gamma
        !> The maximum angular momentum quantum number l
        integer,      intent(in)   :: lmax
        ! the angular momentum degree l, and corresponding order numbers m and m' 
        integer                    :: l, m, mp
        real(REAL64)               :: reslt(-lmax:lmax, -lmax:lmax, 0:lmax)
        real(REAL64)               :: rot_coeff_matrices(-lmax:lmax, 0:lmax, 0:lmax), &
                                      cos_malpha(0:lmax), sin_malpha(0:lmax), cos_mgamma(0:lmax), &
                                      sin_mgamma(0:lmax)
        real(REAL64), parameter    :: sqrt_2 = sqrt(2.0d0)

        forall (m = 0 : lmax)
            cos_malpha(m) = cos(m * alpha)
            sin_malpha(m) = sin(m * alpha)
            cos_mgamma(m) = cos(m * gamma)
            sin_mgamma(m) = sin(m * gamma)
        end forall

        !call bigben%split("Generate rota coeffs")
        rot_coeff_matrices = get_rotation_coefficient_matrices(beta, lmax)
        !call bigben%stop_and_print()
        !call verify_rotation_coefficients(rot_coeff_matrices, beta, lmax)
        
        !call bigben%split("Apply them")
        do l = 0, lmax
            
            ! go through negative m values
            do m = -l, -1
                ! negative m' values
                forall (mp = -l: -1)
                    reslt(mp, m, l) = -2.0d0 * sin_malpha(-m) * sin_mgamma(-mp) *         &
                        (rot_coeff_matrices(-m, -mp, l) + (-1)**(-m) * rot_coeff_matrices(mp, -m, l)) / 2.0d0  &
                        + 2.0d0 * cos_malpha(-m) * cos_mgamma(-mp) *                                     &
                        (rot_coeff_matrices(-m, -mp, l) - (-1)**(-m) * rot_coeff_matrices(mp, -m, l)) / 2.0d0  
                end forall
                ! m'=0
                reslt(0, m, l) = sqrt_2 * sin_malpha(-m) * &
                    (rot_coeff_matrices(-m, 0, l) + (-1)**(-m) * rot_coeff_matrices(0, -m, l)) / 2.0d0   &
                    + sqrt_2 * cos_malpha(-m) *                                                      &
                    (rot_coeff_matrices(-m, 0, l) - (-1)**(-m) * rot_coeff_matrices(0, -m, l)) / 2.0d0  

                ! positive m' values
                forall (mp = 1: l)
                    reslt(mp, m, l) = +2.0d0 * sin_malpha(-m) * cos_mgamma(mp) *         &
                        (rot_coeff_matrices(-m, mp, l) + (-1)**(-m) * rot_coeff_matrices(-mp, -m, l)) / 2.0d0  &
                        + 2.0d0 * cos_malpha(-m) * sin_mgamma(mp) *                                      &
                        (rot_coeff_matrices(-m, mp, l) - (-1)**(-m) * rot_coeff_matrices(-mp, -m, l)) / 2.0d0  
                end forall
            end do

            ! go through m= 0 values
            forall (mp = -l : -1)
                reslt(mp, 0, l) = -sqrt_2 * sin_mgamma(-mp) *         &
                    (rot_coeff_matrices(0, -mp, l) +  rot_coeff_matrices(mp, 0, l)) / 2.0d0  &
                    - sqrt_2 * cos_mgamma(-mp) *                                     &
                    (rot_coeff_matrices(0, -mp, l) - rot_coeff_matrices(mp, 0, l)) / 2.0d0  
            end forall
            reslt(0, 0, l) = 1.0d0 * &
                (rot_coeff_matrices(0, 0, l) + rot_coeff_matrices(0, 0, l)) / 2.0d0   
            forall (mp = 1: l)
                reslt(mp, 0, l) = sqrt_2 *  cos_mgamma(mp) *         &
                    (rot_coeff_matrices(0, mp, l) + rot_coeff_matrices(-mp, 0, l)) / 2.0d0  &
                    - sqrt_2 * sin_mgamma(mp) *                                      &
                    (rot_coeff_matrices(0, mp, l) - rot_coeff_matrices(-mp, 0, l)) / 2.0d0  
            end forall

            ! go through positive m values
            do m = 1, l
                forall (mp = -l: -1)
                    reslt(mp, m, l) = -2.0d0 * cos_malpha(m) * sin_mgamma(-mp) *         &
                        (rot_coeff_matrices(m, -mp, l) + (-1)**m * rot_coeff_matrices(mp, m, l)) / 2.0d0  &
                        - 2.0d0 * sin_malpha(m) * cos_mgamma(-mp) *                                     &
                        (rot_coeff_matrices(m, -mp, l) - (-1)**m * rot_coeff_matrices(mp, m, l)) / 2.0d0  
                end forall
                reslt(0, m, l) = sqrt_2 * cos_malpha(m) * &
                    (rot_coeff_matrices(m, 0, l) + (-1)**m * rot_coeff_matrices(0, m, l)) / 2.0d0   &
                    - sqrt_2 * sin_malpha(m) *                                                      &
                    (rot_coeff_matrices(m, 0, l) - (-1)**m * rot_coeff_matrices(0, m, l)) / 2.0d0  
                forall (mp = 1: l)
                    reslt(mp, m, l) = 2.0d0 * cos_malpha(m) * cos_mgamma(mp) *         &
                        (rot_coeff_matrices(m, mp, l) + (-1)**m * rot_coeff_matrices(-mp, m, l)) / 2.0d0  &
                        - 2.0d0 * sin_malpha(m) * sin_mgamma(mp) *                                      &
                        (rot_coeff_matrices(m, mp, l) - (-1)**m * rot_coeff_matrices(-mp, m, l)) / 2.0d0  
                end forall
            end do
        end do 
        !call bigben%stop_and_print()
        
    end function

    !> This function calculates the small d-matrix coefficients in
    !! spherical harmonics rotation. The coefficients can be used in rotation of both
    !! complex and real spherical harmonics. The result matrix is of shape 
    !! (0:lmax, 0:lmax, -lmax:lmax).
    !!
    !! This function calculates the coefficients
    !! using the algorithm found in publication: "Evaluation of the rotation matrices 
    !! in the basis of real spherical harmonics", Miguel A. Blanco, M. Flórez, M. Bermejo,
    !! Journal of Molecular Structure (Theochem) 419 (1997) 19-27
    !! 
    !! TODO: optimize the structure of the result matrix (contains plenty of empty cells) and
    !! the order is not optimal for the iterations (reverse l, m, m' order)
    pure function get_rotation_coefficient_matrices(beta, lmax) result(reslt)
        real(REAL64), intent(in)   :: beta
        integer,      intent(in)   :: lmax
        
        ! the angular momentum degree l, and corresponding order numbers m and m' 
        integer                    :: l, m, mp

        real(REAL64)               :: reslt(-lmax:lmax, 0:lmax, 0:lmax), cos_beta, cos_beta_p2, &
                                      sin_beta_p2, sin_beta, tan_beta_p2
        real(REAL64), parameter    :: sqrt_2 = sqrt(2.0)
        cos_beta = cos(beta)
        sin_beta = sin(beta)
        cos_beta_p2 = cos(0.5d0 * beta)
        sin_beta_p2 = sin(0.5d0 * beta)
        tan_beta_p2 = tan(0.5d0 * beta)
        reslt = 0.0d0
        ! obtain the value for l=0, m=0, m'=0, reference: eq. 48
        reslt(0, 0, 0) = 1
        ! orbtain value for l=1, m=0, m'=0, reference: eq.49
        reslt(0, 0, 1) = cos_beta
        ! obtain value for l=1, m=1, m'=-1, reference: eq.50
        reslt(-1, 1, 1) = sin_beta_p2 ** 2.0d0
        !obtain result for l=1, m=1, m'=0, reference: eq.51
        reslt(0, 1, 1) = -1.0d0 / sqrt_2 * sin_beta
        reslt(-1, 0, 1) = reslt(0, 1, 1)
        reslt(1, 0, 1) = -1.0d0 * reslt(0, 1, 1)
        !obtain result for l=1, m=1, m'=1, reference: eq.52
        reslt(1, 1, 1) = cos_beta_p2 ** 2
           
        if ((pi - abs(beta)) > 1e-5) then

            do l = 2, lmax
                do m = 0, l-2
                    do mp = -m, 0
                        reslt(mp, m, l) = l * (2 * l - 1) / sqrt(dble((l*l - m*m)*(l*l - mp*mp))) * &
                            ((reslt(0, 0, 1) - m*mp/dble(l*(l-1)))*reslt(mp, m, l-1) -                &
                            sqrt(dble(((l-1)*(l-1) - m*m) * ((l-1)*(l-1) - mp*mp))) * reslt(mp, m, l-2) / ((l-1)*(2*l-1)))
                        reslt(-m, -mp, l) = reslt(mp, m, l)
                    end do

                    reslt(m, 0, l) = (-1.0d0)**(m) * reslt(0, m, l)
                    ! the below loop is the same as above loop, but contains different copy
                    do mp = 1, m
                        reslt(mp, m, l) = l * (2 * l - 1) / sqrt(dble((l*l - m*m)*(l*l - mp*mp))) * &
                            ((reslt(0, 0, 1) - m*mp/dble(l*(l-1)))*reslt(mp, m, l-1) -                &
                            sqrt(dble(((l-1)*(l-1) - m*m) * ((l-1)*(l-1) - mp*mp))) * reslt(mp, m, l-2) / ((l-1)*(2*l-1)))
                        reslt(m, mp, l) = (-1.0d0)**(mp+m) * reslt(mp, m, l)
                    end do
                end do
                !if (any(isNan(reslt(l, 0:l, -l:l)))) then
                !    print *, l, " a rotation coefficients has nans", beta
                !end if 
                ! obtain result for l=m=m' reference: eq. 65
                reslt(l, l, l) = reslt(1, 1, 1) * reslt(l-1, l-1, l-1)
                ! obtain result for m=mp=l-1 reference: eq. 66
                reslt(l-1, l-1, l) = (l * reslt(0, 0, 1) - l + 1) * reslt(l-1, l-1, l-1)    

                
                do mp = l, 1, -1
                    reslt(mp-1, l, l) = -1.0d0 * sqrt(dble(l+mp) / dble(l-mp+1)) * tan_beta_p2 * reslt(mp, l, l)
                    reslt(l, mp-1, l) = (-1.0d0)**(l+mp-1) * reslt(mp-1, l, l)
                end do
            
                reslt(-l, 0, l) = reslt(0, l, l)
                reslt(l, 0, l) = (-1.0d0)**(l) * reslt(0, l, l)

    
                !if (any(isNan(reslt(l, 0:l, -l:l)))) then
                !    print *, l, " c rotation coefficients has nans", beta
                !end if 
                do mp = 0, -l+1, -1
                    reslt(mp-1, l, l) = -1.0d0 * sqrt(dble(l+mp) / dble(l-mp+1)) * tan_beta_p2 * reslt(mp, l, l)
                    reslt(-l, -(mp-1), l) = reslt(mp-1, l, l)
                end do
                
                reslt(-l+1, l, l) = sqrt(dble(factorial(2*l)) / (factorial(l-l+1) * factorial(l+l-1)) ) &
                                * (cos_beta_p2)**(l-l+1) * (-sin_beta_p2)**(l+l-1)
        
                
                !if (any(isNan(reslt(l, 0:l, -l:l)))) then
                !    print *, l, " d rotation coefficients has nans", beta
                !end if 
                ! obtain result for l=m, reference: eq: 68
                do mp = l-1, 1, -1
                    if (abs(reslt(mp, l-1, l)) > epsilon(0.0d0)) then
                        reslt(mp-1, l-1, l) = -1.0d0 * (l * cos_beta - mp + 1) / (l * cos_beta - mp) * &
                                        sqrt(dble(l+mp) / dble(l-mp+1)) * tan_beta_p2 * reslt(mp, l-1, l)
                    else
                        reslt(mp-1, l-1, l) = 0.0d0
                    end if
                    !if (isNan(reslt(l, l-1, mp-1))) then
                    !    print *,  -1.0d0 * (l * cos(beta) - mp + 1), "/ (", l, "*", cos(beta), "-", - mp, &
                    !                    ") *", sqrt(dble(l+mp) / dble(l-mp+1)) * tan(0.5d0*beta), &
                    !                    reslt(l, l-1, mp)
                    !end if
                    reslt(l-1, mp-1, l) = (-1.0d0)**(l-1+mp-1) * reslt(mp-1, l-1, l)
                end do

                
                !if (any(isNan(reslt(l, 0:l, -l:l)))) then
                !    print *, l, " e rotation coefficients has nans", beta
                !end if 


                reslt(-(l-1), 0, l) = reslt(0, l-1, l)
                reslt((l-1), 0, l) = (-1.0d0)**(l-1) * reslt(0, l-1, l)

                
                ! obtain result for l=m, reference: eq: 68
                !do mp = -l+1, 2-l, -1
                    
                !    print *, "i", l, l-1, mp-1, ",", l, -(mp-1), -(l-1)
                !    reslt(l, l-1, mp-1) = -1.0d0 * (l * dcos(beta) - mp + 1) / (l * dcos(beta) - mp) * &
                !                        dsqrt(dble(l+mp) / dble(l-mp+1)) * dtan(0.5d0*beta) * reslt(l, l-1, mp)
                !    
                !    reslt(l, -(mp-1), -(l-1)) = reslt(l, l-1, mp-1)
                !      
                !end do

                !reslt(l, l-1, -l+1) = (l*dcos(beta)+l-1) * dsqrt(dble(factorial(2*l-1)/(factorial(l-l+1)*factorial(l+l-1)))) & 
                !                    * (dcos(beta/2)**(l-1-l+1)) * (-sin(beta/2))**(l-1+l-1)
                ! obtain result for l=m, reference: eq: 67, positive m' values
                do mp = -1, -l+1, -1
                    reslt(mp, l-1, l) = (l*cos_beta - mp) * sqrt(dble(factorial(2*l-1)) / (factorial(l+mp) * factorial(l-mp)) ) &
                                * (cos_beta_p2)**(l+mp-1) * (-sin_beta_p2)**(l-mp-1)
                    reslt(-(l-1), -mp, l) = reslt(mp, l-1, l)
                end do
            end do
        else
            do l = 2, lmax
                do m = 0, l-2
                    do mp = -m, 0
                        reslt(mp, m, l) = l * (2 * l - 1) / sqrt(dble((l*l - m*m)*(l*l - mp*mp))) * &
                            ((reslt(0, 0, 1) - m*mp/dble(l*(l-1)))*reslt(mp, m, l-1) -                &
                            sqrt(dble(((l-1)*(l-1) - m*m) * ((l-1)*(l-1) - mp*mp))) * reslt(mp, m, l-2) / ((l-1)*(2*l-1)))
                        reslt(-m, -mp, l) = reslt(mp, m, l)
                    end do

                    reslt(m, 0, l) = (-1.0d0)**(m) * reslt(0, m, l)
                    ! the below loop is the same as above loop, but contains different copy
                    do mp = 1, m
                        reslt(mp, m, l) = l * (2 * l - 1) / sqrt(dble((l*l - m*m)*(l*l - mp*mp))) * &
                            ((reslt(0, 0, 1) - m*mp/dble(l*(l-1)))*reslt(mp, m, l-1) -                &
                            sqrt(dble(((l-1)*(l-1) - m*m) * ((l-1)*(l-1) - mp*mp))) * reslt(mp, m, l-2) / ((l-1)*(2*l-1)))
                        reslt(m, mp, l) = (-1.0d0)**(mp+m) * reslt(mp, m, l)
                    end do
                end do
                !if (any(isNan(reslt(l, 0:l, -l:l)))) then
                !    print *, l, " a rotation coefficients has nans", beta
                !end if 
                ! obtain result for l=m=m' reference: eq. 65
                reslt(l, l, l) = reslt(1, 1, 1) * reslt(l-1, l-1, l-1)
                ! obtain result for m=mp=l-1 reference: eq. 66
                reslt(l-1, l-1, l) = (l * reslt(0, 0, 1) - l + 1) * reslt(l-1, l-1, l-1)   

                
                reslt(:, l, l)     = 0
                reslt(-l, :, l)    = 0
                reslt(:, l-1, l)   = 0
                reslt(-(l-1), :, l)= 0
                reslt(-l, l, l)    = 1 
                reslt(1-l, l-1, l) = -1
                reslt(-(l-1), -(1-l), l) = -1

                ! obtain result for l=m, reference: eq: 67, positive m' values
                do mp = -1, -l+1, -1
                    reslt(mp, l-1, l) = (l*cos_beta - mp) * sqrt(dble(factorial(2*l-1)) / (factorial(l+mp) * factorial(l-mp)) ) &
                                * (cos_beta_p2)**(l+mp-1) * (-sin_beta_p2)**(l-mp-1)
                    reslt(-(l-1), -mp, l) = reslt(mp, l-1, l)
                end do

            end do
        end if
            
        !if (any(isNan(reslt))) then
        !    print *, "rotation coefficients has nans"
        !end if
    end function

    subroutine verify_rotation_coefficients(rotation_coefficients, beta, lmax)
        real(REAL64)               :: rotation_coefficients(0:lmax, 0:lmax, -lmax:lmax)
        real(REAL64), intent(in)   :: beta
        integer,      intent(in)   :: lmax
        real(REAL64)               :: THRESHOLD = 1e-10, value
        
        ! the angular momentum degree l, and corresponding order numbers m and m' 
        integer                    :: l, m, mp, k
        print *, "verifying rotation coefficients"
        print *, "---------------------------------"
        do l = 0, lmax
            do m = 0, l
                do mp = -l, l
                    ! Test the opposites etc:
                    ! d_mm' = d_-m'-m = (-1)**(m+m') d_-m,-m' = (-1)**m+m' d_m'm
                    if (-mp >= 0) then
                        if (abs(rotation_coefficients(l, m, mp)-rotation_coefficients(l, -mp, -m)) &
                          > THRESHOLD) then
                            print *, "opposite error at", l, m, mp, "->", l, -mp, -m
                        end if
                    end if

                    if (mp >= 0) then
                        if (abs(rotation_coefficients(l, m, mp)- (-1)**(mp+m) * rotation_coefficients(l, mp, m)) &
                          > THRESHOLD) then
                            print *, "opposite error at", l, m, mp, "-> (-1)**(m+mp)", l, mp, m
                        end if
                    end if

                    if (m == 0) then
                          
                        if (abs(rotation_coefficients(l, m, mp) - (-1)**(mp+m) * rotation_coefficients(l, -m, -mp)) &
                          > THRESHOLD) then
                            print *, "opposite error at", l, m, mp, "-> (-1)**(m+mp)", l, mp, m
                        end if
                    end if

                    if (m == l) then
                        value = dsqrt(dble(factorial(2*l)) / (factorial(l+mp) * factorial(l-mp)) ) &
                                * (dcos(beta/2))**(l+mp) * (-dsin(beta/2))**(l-mp)
                        if (abs(rotation_coefficients(l, m, mp) - value) > THRESHOLD) then
                            print *, "value error at", l, m, mp, "calculated value", value, &
                                         "recursion value", rotation_coefficients(l, m, mp), &
                                         "parent recursion value", rotation_coefficients(l, m, mp+1), &
                                         "tan (beta/2)", dtan(beta/2)  
                        end if
                
                    

                    else if (m == l-1) then
                        value = (l*cos(beta) - mp) * dsqrt(dble(factorial(2*l-1)) / (factorial(l+mp) * factorial(l-mp)) ) &
                                * (dcos(beta/2))**(l+mp-1) * (-dsin(beta/2))**(l-mp-1)
                        if (abs(rotation_coefficients(l, m, mp) - value) > THRESHOLD) then
                            print *, "value error at", l, m, mp, "calculated value", value, &
                                         "recursion value", rotation_coefficients(l, m, mp), &
                                         "tan (beta/2)", dtan(beta/2)  
                        end if
                
                    end if       
                    
                    
                end do
            end do
        end do

    end subroutine    
end module