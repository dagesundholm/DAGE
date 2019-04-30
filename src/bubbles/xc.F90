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
!> @file xc.F90 File containing routines for evaluation of Exchange and 
!! correlation potential for Function3D electron density.

module XC_class
    use ISO_FORTRAN_ENV
    use globals_m
    use lipbasis_class
    use function3d_class
    use Function3D_types_m
    use Grid_class 
    use bubbles_class
    use Evaluators_class
    use RealSphericalHarmonics_class
    use Laplacian3D_class
    use CoreEvaluator_class
    !> interface to libxc library
    use xc_f90_types_m 
    use xc_f90_lib_m 
    !> spherical integration
    use xc_lebedev
    use MemoryLeakChecker_m
    use Points_class
#ifdef HAVE_CUDA
    use CUDA_m
#endif

    implicit none


    type XC
        !> (ex, cor), select exchange & correlation functional.
        !! positive interger, values the same with libxc library.
        !! 0 means not used.
        !> default: lda-x, lda-pw.
        integer(INT32)               :: exchange_functional_type = 1
        integer(INT32)               :: correlation_functional_type = 12
        !> energy
        real(REAL64)                 :: xc_e_int !> energy. Exc
        !> closed shell only for now. 
        ! logical :: polarized = .false.
        !> the xc energy density, Exc = \int dr \rho * func.
        !> not the kernel, where Exc = \int dr func
        type(Function3D)             :: energy_per_particle 
        type(Function3D)             :: potential
        type(Points), allocatable    :: becke_weights(:)

        !> Function3D - evaluator used to evaluate density
        type(Function3DEvaluator), allocatable    :: bubbles_points_evaluators(:)
        type(Function3DEvaluator), allocatable    :: grid_points_evaluator
        type(PointCoordinates),    allocatable    :: density_evaluation_points(:)
        integer,                   allocatable    :: radial_integration_orders(:)
        real(REAL64),              allocatable    :: radial_order_distances(:)
        integer,                   allocatable    :: radial_first_points(:, :)
        integer,                   allocatable    :: radial_lmax(:)
        integer(INT32),            allocatable    :: points_per_string(:, :)
        class(Laplacian3D),         pointer       :: laplacian_operator
        type(CoreEvaluator),       pointer        :: core_evaluator
        logical                                   :: orbitals_density_evaluation

        !> The pointers to the libxc objects
        TYPE(xc_f90_pointer_t)       :: exchange_function_pointer
        TYPE(xc_f90_pointer_t)       :: exchange_info_pointer
        TYPE(xc_f90_pointer_t)       :: correlation_function_pointer
        TYPE(xc_f90_pointer_t)       :: correlation_info_pointer
      
    contains
        !> updates xc with the projection method. 
        !! restriction: only for lda.
        procedure :: eval        => XC_eval  
        !> Do the evaluation of input points, called in the eval functions
        procedure, private :: evaluate_points => XC_evaluate_points
        !> project values to bubbles
        procedure :: project_to_bubbles => XC_project_to_bubbles
        !> updates xc with the derivative on grid approach
        !! restriction: lda and gga.
        procedure :: eval_g      => XC_eval_grid  
        !> updates xc. not finished. not promising.
        procedure :: eval_taylor => XC_eval_taylor
        procedure, private :: get_divergence_from_strings & 
                                => XC_get_divergence_from_strings
        procedure, private :: calculate_density_evaluation_points &
                                 => XC_calculate_density_evaluation_points
        procedure, private :: evaluate_bubbles   => XC_evaluate_bubbles
        procedure, private :: evaluate_divergence_potential_term   => XC_evaluate_divergence_potential_term
        procedure, private :: evaluate_cube      => XC_evaluate_cube
        procedure, private :: evaluate_cores     => XC_evaluate_cores
        procedure, private :: evaluate_electron_density_laplacian => XC_evaluate_electron_density_laplacian
        procedure, private :: reevaluate_electron_density => XC_reevaluate_electron_density
        procedure, private :: evaluate_electron_density_cube => XC_evaluate_electron_density_cube
        procedure, private :: evaluate_electron_density_bubbles_points => XC_evaluate_electron_density_bubbles_points
        procedure, private :: preevaluate_becke_weights => XC_preevaluate_becke_weights
        procedure          :: destroy            => XC_destroy 
    end type xc 
    
    !> initialization.
    interface XC
       module procedure xc_init_borrow
    end interface

    contains
    

    !> Initialize XC object from input orbital.
    !! xc most quantities initialized to 0, 
    !! NOT cnosistent with the input density yet.
    function XC_init_borrow(orbital, exchange_functional_type, correlation_functional_type, &
                            lmax, laplacian_operator, core_evaluator, orbitals_density_evaluation) &
                            result(new)
        type(function3d),   intent(in)             :: orbital
        integer(INT32),     intent(in)             :: exchange_functional_type  
        integer(INT32),     intent(in)             :: correlation_functional_type  
        !> max l for xc bubbles.
        integer(INT32),     intent(in)             :: lmax
        class(Laplacian3D), intent(inout), target  :: laplacian_operator
        type(CoreEvaluator),intent(inout), target  :: core_evaluator
        logical,            intent(in)             :: orbitals_density_evaluation
        logical                                    :: evaluate_gradients
        integer                                    :: i, j, p
        real(REAL64), pointer                      :: coord(:)
        type(Function3D)                           :: temp

        !> The initialized XC-object
        type(XC), target                           :: new
       
        new%laplacian_operator => laplacian_operator
        new%core_evaluator => core_evaluator
        new%exchange_functional_type = exchange_functional_type
        new%correlation_functional_type = correlation_functional_type
        new%orbitals_density_evaluation = orbitals_density_evaluation
        new%xc_e_int = 0.0d0

        !> initialize xc function3d. 
        call new%energy_per_particle%init_copy(orbital, type = F3D_TYPE_CUSP, &
                                     lmax = orbital%bubbles%get_lmax())
        !> assign the potential to be the same with energy per particle.
        call new%potential%init_copy(orbital, type = F3D_TYPE_CUSP, &
                                     lmax = orbital%bubbles%get_lmax())

        
        if (new%exchange_functional_type /= 0)       &
            call xc_f90_func_init(new%exchange_function_pointer, new%exchange_info_pointer, &
                                  new%exchange_functional_type, XC_UNPOLARIZED)
        if (new%correlation_functional_type /= 0) &
            call xc_f90_func_init(new%correlation_function_pointer, new%correlation_info_pointer, &
                                  new%correlation_functional_type, XC_UNPOLARIZED)

        evaluate_gradients = & 
            xc_f90_info_family(new%exchange_info_pointer) == XC_FAMILY_GGA .or.  &
            xc_f90_info_family(new%exchange_info_pointer) == XC_FAMILY_HYB_GGA .or. &
            xc_f90_info_family(new%correlation_info_pointer) == XC_FAMILY_GGA .or.  &
            xc_f90_info_family(new%correlation_info_pointer) == XC_FAMILY_HYB_GGA

        
        !> radial integration order
        !new%radial_integration_orders = [38] 
        !new%radial_order_distances    = [0.0d0] 
        !new%radial_lmax               = [12]
        new%radial_integration_orders = [110,       110,   110,    110,     110,     110,     110, &
                                         110,   110,   110,   110] 
        new%radial_order_distances    = [0.0d0, 1.00d-2, 2.60d-1, 3.33d-1, 5.459d-1, 8.008d-1, 1.502d0, &
                                         2.672d0, 5.015d0, 7.364d0, 9.2d0] 
        new%radial_lmax               = [12,        12,    12,      12,     12,      12,       12, &
                                         12,        12,    12,     12]
        allocate(new%radial_first_points(size(new%radial_integration_orders), orbital%bubbles%get_nbub()))
        do i = 1, orbital%bubbles%get_nbub()
            coord => orbital%bubbles%gr(i)%p%get_coord()
            new%radial_first_points(1, i) = 1
            p = 1
            j = 2
            do while(j <= size(new%radial_integration_orders))
                if (coord(p) > new%radial_order_distances(j)) then
                    new%radial_first_points(j, i) = p
                    j = j + 1
                end if
                p = p + new%potential%grid%get_nlip()-1
            end do
        end do
        call new%calculate_density_evaluation_points(orbital)
        call new%preevaluate_becke_weights(orbital%bubbles)
        allocate(new%bubbles_points_evaluators(orbital%bubbles%get_nbub()))
        call temp%init_copy(new%potential, lmax = new%potential%bubbles%get_lmax() + 1)
        new%grid_points_evaluator    = Function3DEvaluator(.FALSE.)
        do i = 1, orbital%bubbles%get_nbub()
            new%bubbles_points_evaluators(i) = Function3DEvaluator(.FALSE.) !, ibubs = [i])
        end do
        call temp%destroy()
       
    end function XC_init_borrow

    subroutine XC_destroy(self)
        class(XC) :: self
        integer   :: i

        if (allocated(self%bubbles_points_evaluators)) then
            do i = 1, size(self%bubbles_points_evaluators)
                call self%bubbles_points_evaluators(i)%destroy()
            end do
            deallocate(self%bubbles_points_evaluators)
        end if
        
        if (allocated(self%density_evaluation_points)) then
            do i = 1, size(self%density_evaluation_points)
                call self%density_evaluation_points(i)%destroy()
            end do
            deallocate(self%density_evaluation_points)
        end if
        
        if (allocated(self%radial_integration_orders)) then
            deallocate(self%radial_integration_orders)
        end if
        
        if (allocated(self%radial_order_distances)) then
            deallocate(self%radial_order_distances)
        end if
        
        if (allocated(self%radial_first_points)) then
            deallocate(self%radial_first_points)
        end if
        
        if (allocated(self%radial_lmax)) then
            deallocate(self%radial_lmax)
        end if
        
        if (allocated(self%points_per_string)) then
            deallocate(self%points_per_string)
        end if
        
        if (allocated(self%becke_weights)) then
            do i = 1, size(self%becke_weights)
                call self%becke_weights(i)%destroy()
            end do
            deallocate(self%becke_weights)
        end if
        
        if (allocated(self%grid_points_evaluator)) then
            call self%grid_points_evaluator%destroy()
            deallocate(self%grid_points_evaluator)
        end if
        call self%potential%destroy()
        call self%energy_per_particle%destroy()
    end subroutine XC_destroy



    

    subroutine XC_calculate_density_evaluation_points(self, orbital)
        class(XC),        intent(inout) :: self
        type(Function3D), intent(in)    :: orbital
        type(REAL64_2D), allocatable    :: points(:)
        integer,         allocatable    :: number_of_points(:)
        real(REAL64)                    :: angular_unit(sum(self%radial_integration_orders), 4)
        integer                         :: i, offset, p, j, isection, first_point, last_point, point_count
        ! cartesian coordinates of a bubble center
        real(REAL64)                    :: tmp_center(3), becke_weights(orbital%bubbles%get_nbub(), 1), point(3)
        ! the radial grid coorinates and xc values for bubbles
        real(REAL64), pointer           :: tmp_coord(:)
        
        logical                         :: found_zero

        allocate(self%points_per_string(sum(self%radial_integration_orders), &
                                        orbital%bubbles%get_nbub()), source = 0)

        !> obtain the cartesian coordinates and weights from xc_lebedev module.
        allocate(number_of_points(orbital%bubbles%get_nbub()))
        allocate(points(orbital%bubbles%get_nbub()))
        allocate(self%density_evaluation_points(orbital%bubbles%get_nbub()))
        do i = 1, orbital%bubbles%get_nbub()
            number_of_points(i) = 0
            first_point = 1
            do isection = 1, size(self%radial_integration_orders)
                last_point = sum(self%radial_integration_orders(:isection))
                call ld_by_order(self%radial_integration_orders(isection), &
                        angular_unit(first_point:last_point, 1), &
                        angular_unit(first_point:last_point, 2), &
                        angular_unit(first_point:last_point, 3), &
                        angular_unit(first_point:last_point, 4))
                if (isection == size(self%radial_integration_orders)) then
                    point_count =  orbital%bubbles%gr(i)%p%get_shape() - self%radial_first_points(isection, i) + 1
                else
                    point_count =  self%radial_first_points(isection+1, i) - self%radial_first_points(isection, i) + 1
                end if
                number_of_points(i) =   number_of_points(i) &
                                   +   self%radial_integration_orders(isection) &
                                     * point_count
                first_point = last_point + 1
            end do
            allocate(points(i)%p(3, number_of_points(i)))
        end do
        do i = 1,  orbital%bubbles%get_nbub()  
            tmp_center = orbital%bubbles%get_centers(i) 
            tmp_coord => orbital%bubbles%gr(i)%p%get_coord()

            j = 1
            offset = 0
            do isection = 1, size(self%radial_integration_orders)
                first_point = self%radial_first_points(isection, i)
                if (isection == size(self%radial_integration_orders)) then
                    last_point = size(tmp_coord)
                else
                    last_point = self%radial_first_points(isection+1, i)
                end if
                
                do while (j <= sum(self%radial_integration_orders(:isection)))
                    self%points_per_string(j, i) = 0 
                    found_zero = .FALSE.
                    !> loop over each grid point.
                    do p = first_point, last_point  

                        !> coordinates on the spherical integration sphere. 
                        point(1) = angular_unit(j, 1) * tmp_coord(p) + tmp_center(1) 
                        point(2) = angular_unit(j, 2) * tmp_coord(p) + tmp_center(2)
                        point(3) = angular_unit(j, 3) * tmp_coord(p) + tmp_center(3)

                        !> becke weights for each point.
                        call becke_partition(orbital%bubbles%get_nbub(), 1, &
                            orbital%bubbles%get_centers(), &
                            ! orbital%bubbles%get_z(), &
                            [point], becke_weights)

                

                        ! add to the offset and to the number of points per points
                        !if (becke_weights(i, 1) > 1e-7) then
                            offset = offset + 1
                            self%points_per_string(j, i) = self%points_per_string(j, i) + 1
                            points(i)%p(:, offset) = point
                        !else if (.not. found_zero) then
                        !    exit
                        !end if
                        
                    end do
                    j = j + 1
                end do
            end do
            self%density_evaluation_points(i) = PointCoordinates(points(i)%p(:, :offset))
            deallocate(points(i)%p)
        end do
        deallocate(points)
        deallocate(number_of_points)
        
    end subroutine


    subroutine XC_get_divergence_from_strings(self, result_bubbles, point_values_x, point_values_y, &
                                              point_values_z, lmax, use_becke_weights)
        class(XC),        intent(inout)         :: self
        type(Bubbles),    intent(inout)         :: result_bubbles
        type(REAL64_1D),  intent(in), target    :: point_values_x(:)
        type(REAL64_1D),  intent(in), target    :: point_values_y(:)
        type(REAL64_1D),  intent(in), target    :: point_values_z(:)
        integer,          intent(in), optional  :: lmax
        logical,          intent(in), optional   :: use_becke_weights
        
        real(REAL64)                            :: center
        real(REAL64), pointer                   :: coord(:) 
        real(REAL64), pointer                   :: bubbles_val(:, :)   
        !> cartesian coordiates of points on a sphere for integration.
        !! radius = 1, or = bubble grid. 
        real(REAL64)                            :: unitary_coordinates(3, sum(self%radial_integration_orders))
        !> spherical integration weights
        real(REAL64), allocatable               :: spherical_weights(:)
        !> evaluation arrays for derivation
        real(REAL64), allocatable               :: values(:, :, :)
        ! temporary array for the spherical harmonics for the evaluation points
        real(REAL64), allocatable               :: spherical_harmonics_points(:, :)
        ! temporary array for the spherical harmonics derivatives for the evaluation points
        real(REAL64), allocatable               :: spherical_harmonics_gradients(:, :, :)
        real(REAL64), allocatable               :: temp(:, :)
        !> becke partition weight.
        real (REAL64), allocatable              :: becke_weights(:, :)   
        real(REAL64), pointer                :: value_pointer(:)
        ! loop variables and offset
        integer                              :: i, j, n, l, offset, isection, first_point, last_point, &
                                                nlip, first_cell, last_cell, section_lmax, previous_section_lmax, &
                                                lmax_
        type(Grid1D)                         :: subgrid
        type(Interpolator1D)                 :: interpolator_
        !> spehrical harmonics evaluator
        type(RealSphericalHarmonics)         :: harmonics
        real(REAL64),             pointer    :: gradients_x(:), gradients_y(:), gradients_z(:)
        logical                              :: use_becke_weights_

        if (present(lmax)) then
            lmax_ = min(lmax, result_bubbles%get_lmax())
        else
            lmax_ = result_bubbles%get_lmax()
        end if

        if (present(use_becke_weights)) then
            use_becke_weights_ = use_becke_weights
        else
            use_becke_weights_ = .FALSE.
        end if

        first_point = 1
        allocate(spherical_weights(sum(self%radial_integration_orders)), source = 0.0d0)
        !> obtain the cartesian coordinates and weights from xc_lebedev module.
        do isection = 1, size(self%radial_integration_orders)
            last_point = sum(self%radial_integration_orders(: isection))
            call ld_by_order(self%radial_integration_orders(isection), &
                            unitary_coordinates(1, first_point:last_point), &
                            unitary_coordinates(2, first_point:last_point), &
                            unitary_coordinates(3, first_point:last_point), &
                            spherical_weights(first_point : last_point) ) 
            first_point = last_point + 1
        end do
        !spherical_weights = spherical_weights * 4 * pi

        !> init the SphericalHarmonics with derivative evaluation on.
        harmonics = RealSphericalHarmonics(lmax_, normalization = 1, init_derivatives = .TRUE.)
        allocate(spherical_harmonics_points(sum(self%radial_integration_orders), (lmax_+1)**2))
        allocate(spherical_harmonics_gradients(sum(self%radial_integration_orders), (lmax_+1)**2, 3))

        ! evaluate spherical harmonics and their gradients for all the points defined previously
        spherical_harmonics_points(:, :) = &
            harmonics%eval(unitary_coordinates)
        spherical_harmonics_gradients(:, :, :) = &
            harmonics%evaluate_gradients(unitary_coordinates)
        call harmonics%destroy()

        ! the 2*l+1 belongs inside the loop, but we are doing it here because we can
        forall (l = 1 : lmax_)
        
             spherical_harmonics_points(:, l*l+1:(l+1) * (l+1)) = & 
                 spherical_harmonics_points(:, l*l+1:(l+1) * (l+1)) * (2*l+1)
             spherical_harmonics_gradients(:, l*l+1:(l+1) * (l+1), :) = &
                 spherical_harmonics_gradients(:, l*l+1:(l+1) * (l+1), :) * (2*l+1)
         end forall

        
        call result_bubbles%set_k(-1)
        do i = 1,  result_bubbles%get_nbub()  
            coord => result_bubbles%gr(i)%p%get_coord()
            nlip = result_bubbles%gr(i)%p%get_nlip()
            bubbles_val => result_bubbles%get_f(i)
            bubbles_val = 0.0d0

            allocate(temp(size(bubbles_val, 1), size(bubbles_val, 2)), source = 0.0d0)
            
            ! init the offset to 0
            offset = 0
            j = 1
            do isection = 1, size(self%radial_integration_orders)
                section_lmax = min(lmax_, self%radial_lmax(isection))
                if (isection == 1) then
                    previous_section_lmax = section_lmax
                    !section_lmax = 0
                end if
                ! get the first and last grid point of the section
                first_point = self%radial_first_points(isection, i)
                if (isection == size(self%radial_integration_orders)) then
                    last_point = size(coord) 
                else
                    last_point = self%radial_first_points(isection + 1, i) 
                end if

                ! the sections overlap, thus we are re-initializing the first point to zero
                bubbles_val(first_point, :) = 0.0d0
                temp(first_point, :)        = 0.0d0

                do while (j <= sum(self%radial_integration_orders(:isection)))
                    allocate(becke_weights(result_bubbles%get_nbub(), self%points_per_string(j, i)), source = 0.0d0)
                    !> becke weights for each point.
                    if (use_becke_weights_) then
                        call becke_partition_atomic_size(result_bubbles%get_nbub(), self%points_per_string(j, i), &
                            result_bubbles%get_centers(), result_bubbles%get_z(),  &
                            self%density_evaluation_points(i)%coordinates(:, offset+1:offset+self%points_per_string(j, i)), &
                            becke_weights)
                    else
                        becke_weights = 1.0d0
                    end if

                    ! add the points string (with one spherical_harmonics value)
                    ! multiplied with the correct coefficients to the result bubbles and the temporary array
                    ! NOTE: we are using the bubbles_val, i.e., the result bubbles as a temporary storage 
                    !       for the f^r part
                    forall (n = 1 : (section_lmax+1)**2)
                        bubbles_val(first_point:first_point+self%points_per_string(j, i)-1, n) = &
                            bubbles_val(first_point:first_point+self%points_per_string(j, i)-1, n) &
                                +   spherical_harmonics_points(j, n) * spherical_weights(j) &
                                  * becke_weights(i, :)  &
                                  *  (    point_values_x(i)%p(offset+1:offset+self%points_per_string(j, i)) &
                                        * unitary_coordinates(X_, j) &
                                      +   point_values_y(i)%p(offset+1:offset+self%points_per_string(j, i)) &
                                        * unitary_coordinates(Y_, j) &
                                      +   point_values_z(i)%p(offset+1:offset+self%points_per_string(j, i)) &
                                         * unitary_coordinates(Z_, j))

                        temp(first_point:first_point+self%points_per_string(j, i)-1, n) = &
                              temp(first_point:first_point+self%points_per_string(j, i)-1, n) &
                            !+   !coord(first_point : last_point) &
                            +   spherical_weights(j) &
                              * becke_weights(i, :) &
                              * (    point_values_x(i)%p(offset+1:offset+self%points_per_string(j, i)) &
                                   * spherical_harmonics_gradients(j, n, X_) &
                                   !* gradients_x(offset+1:offset+self%points_per_string(j, i))    &
                                 +   point_values_y(i)%p(offset+1:offset+self%points_per_string(j, i)) & 
                                   * spherical_harmonics_gradients(j, n, Y_) &
                                   !* gradients_y(offset+1:offset+self%points_per_string(j, i))    &
                                 +   point_values_z(i)%p(offset+1:offset+self%points_per_string(j, i)) &
                                   * spherical_harmonics_gradients(j, n, Z_))
                                   !* gradients_z(offset+1:offset+self%points_per_string(j, i))) 
                    end forall
                    offset = offset + self%points_per_string(j, i)
                    j = j + 1
                    deallocate(becke_weights)
                end do
                !if (previous_section_lmax /= section_lmax) &
                !    bubbles_val(first_point, (previous_section_lmax+1)**2 + 1 :) = 0.0d0
            end do
            
            !call result_bubbles%extrapolate_origo(2, lmax = 0)
            !print *, "temp bubbles"
            !print *, "---------------------"
            !call result_bubbles%print_out(lmax_ = 1)
            
            ! get the interpolator that is used in evaluation of the derivatives
            interpolator_ = Interpolator1D(result_bubbles%gr(i)%p, 1, ignore_first = .TRUE.)
        
            ! derivate the f^r part
            values = interpolator_%eval(bubbles_val, coord)

            ! multiply derivatives of f^r with r
            forall (n = 1 : (lmax_+1)**2)
                !bubbles_val(:, n) = bubbles_val(:, n) / coord
                !temp(:, n)      = temp(:, n) / coord
                values(:, 2, n) = values(:, 2, n) * coord
            end forall

            !print *, "values", values(1:15, 2, 1)
            !print *, "temp 1", temp(1:15, 1)
            call interpolator_%destroy()
            ! get the final bubbles
            ! NOTE: we are setting the bubbles k to -1, thus the formula is slightly changed
            !       when comparing with the original one
            forall (l = 0 : lmax_)
                bubbles_val(:, l*l+1:(l+1)*(l+1)) = &
                                values(:, 2, l*l+1:(l+1)*(l+1)) &
                              + 2.0d0   * bubbles_val(:, l*l+1:(l+1)*(l+1)) &
                              - temp(:, l*l+1:(l+1)*(l+1))
            end forall

            
            !do l = 0, result_bubbles%get_lmax()
            !    bubbles_val(:, l*l+1:(l+1)*(l+1)) = &
            !          bubbles_val(:, l*l+1:(l+1)*(l+1)) &
            !        * sqrt(dble(4*pi)/(2*l+1))
            !end do
            !call result_bubbles%extrapolate_origo(2, lmax = 0)
            bubbles_val(1, 2:) = 0.0d0
            deallocate(values, temp)
        end do ! loop of bubbles
        deallocate(spherical_weights, spherical_harmonics_points, spherical_harmonics_gradients)
    end subroutine
    
    subroutine XC_preevaluate_becke_weights(self, prototype_bubbles)
        class(XC),     intent(inout)                :: self
        type(Bubbles), intent(in)                   :: prototype_bubbles
        integer                                     :: i, first_point, last_point, number_of_points
        real(REAL64), allocatable                   :: temp_becke(:, :)
        
        allocate(self%becke_weights(prototype_bubbles%get_nbub()))
                    
        !> becke weights for each point.
        do i = 1,  size(self%density_evaluation_points) 
            number_of_points = size(self%density_evaluation_points(i)%coordinates, 2)
            self%becke_weights(i) = Points(self%density_evaluation_points(i))
            allocate(temp_becke(prototype_bubbles%get_nbub(), number_of_points), source = 0.0d0)
#ifdef HAVE_OMP
            !$OMP PARALLEL &
            !$OMP& PRIVATE(first_point, last_point) 
            first_point = number_of_points &
                        / omp_get_num_threads() * (omp_get_thread_num()) + &
                       min(omp_get_thread_num()+1, mod(number_of_points, omp_get_num_threads()))
            last_point = number_of_points / omp_get_num_threads() * (omp_get_thread_num()+1) + &
                        min(omp_get_thread_num()+2, mod(number_of_points, omp_get_num_threads())) - 1
            if (omp_get_thread_num()+1 == omp_get_num_threads()) last_point = last_point +1
#else
            first_point = 1
            last_point = number_of_points
#endif
            call becke_partition_atomic_size &
                (prototype_bubbles%get_nbub(), last_point-first_point+1, &
                prototype_bubbles%get_centers(), &
                prototype_bubbles%get_z(),  &
                self%density_evaluation_points(i)%coordinates(:, first_point:last_point), &
                temp_becke(:, first_point:last_point))
            self%becke_weights(i)%values(first_point:last_point) = &
                temp_becke(i, first_point:last_point)
#ifdef HAVE_OMP
            !$OMP END PARALLEL
#endif
            deallocate(temp_becke)
        end do
    end subroutine

    subroutine XC_project_to_bubbles(self, point_values, result_bubbles, extrapolate_origo, lmax, use_becke_weights)
        class(XC),     intent(in)                :: self
        type(Points),  intent(in)                :: point_values(:)
        type(Bubbles),    intent(inout), target  :: result_bubbles
        logical,          intent(in), optional   :: extrapolate_origo
        integer,          intent(in), optional   :: lmax
        logical,          intent(in), optional   :: use_becke_weights
        real(REAL64),     pointer                :: bubbles_val(:, :)
        real(REAL64)                             :: center, sum_becke
        real(REAL64),     pointer                :: coord(:)  
        !> cartesian coordiates of points on a sphere for integration.
        !! radius = 1, or = bubble grid. 
        real(REAL64), allocatable                :: unitary_coordinates(:, :)
        !> spherical integration weights
        real(REAL64), allocatable                :: spherical_weights(:)
        !> spehrical harmonics evaluator
        type(RealSphericalHarmonics)             :: harmonics
        ! temporary array for the spherical harmonics for the evaluation points
        real(REAL64), allocatable                :: spherical_harmonics_points(:, :), spherical_harmonics(:, :)
        ! loop variables and offset
        integer                                  :: i, j, n, m(1), offset, l, lmax_, isection, first_point, last_point, &
                                                    section_lmax, previous_section_lmax, max_n, number_of_points
        logical                                  :: use_becke_weights_
        !> becke partition weights.
        type(Points), allocatable                :: becke_weights(:)

        if (present(lmax)) then
            lmax_ = min(lmax, result_bubbles%get_lmax())
        else
            lmax_ = result_bubbles%get_lmax()
        end if

        if (present(use_becke_weights)) then
            use_becke_weights_ = use_becke_weights
        else
            use_becke_weights_ = .FALSE.
        end if


        !> init the SphericalHarmonics.
        harmonics = RealSphericalHarmonics(lmax_, normalization = 2)
        allocate(spherical_harmonics_points(sum(self%radial_integration_orders), (lmax_+1)**2))
        allocate(spherical_weights(sum(self%radial_integration_orders)), source = 0.0d0)
        first_point = 1
        !> obtain the cartesian coordinates and weights from xc_lebedev module.
        do isection = 1, size(self%radial_integration_orders)
            last_point = sum(self%radial_integration_orders(: isection))
            allocate(unitary_coordinates(3, self%radial_integration_orders(isection)), source = 0.0d0)
            call ld_by_order(self%radial_integration_orders(isection), unitary_coordinates(1, :), &
                            unitary_coordinates(2, :), unitary_coordinates(3, :), &
                            spherical_weights(first_point : last_point) ) 
            spherical_harmonics = harmonics%eval(unitary_coordinates)
            spherical_harmonics_points(first_point : last_point, :) = &
                spherical_harmonics(:, :)
            deallocate(spherical_harmonics)
            deallocate(unitary_coordinates)
            first_point = last_point + 1
        end do
        call harmonics%destroy()
        ! the 2*l+1 belongs inside the loop, but we are doing it here because we can
         forall (l = 1 : lmax_)
              spherical_harmonics_points(:, l*l+1:(l+1) * (l+1)) = & 
                  spherical_harmonics_points(:, l*l+1:(l+1) * (l+1)) * (2*l+1)
         end forall
        forall (n = 1 : (lmax_+1)**2)
             spherical_harmonics_points(:, n) = spherical_harmonics_points(:, n) * spherical_weights(:)
        end forall
        
        
            

        allocate(becke_weights(result_bubbles%get_nbub()))
        do i = 1,  result_bubbles%get_nbub()  
            ! init the offset to 0
            offset = 0
            
            coord => result_bubbles%gr(i)%p%get_coord()
            bubbles_val => result_bubbles%get_f(i)
            bubbles_val(:, :) = 0.0d0
            becke_weights(i) = Points(self%becke_weights(i), copy_content = use_becke_weights_)
            if (.not. use_becke_weights_) &
                becke_weights(i)%values = 1.0d0
            
            j = 1
            do isection = 1, size(self%radial_integration_orders)
                section_lmax = min(lmax_, self%radial_lmax(isection))
                if (isection == 1) previous_section_lmax = section_lmax
                first_point = self%radial_first_points(isection, i)
                if (isection == size(self%radial_integration_orders)) then
                    last_point = result_bubbles%gr(i)%p%get_shape()
                else
                    last_point = self%radial_first_points(isection+1, i)
                end if
                max_n = (section_lmax+1)**2
                ! the sections overlap, thus we are re-initializing the first point to zero
                bubbles_val(first_point, :) = 0.0d0
                do while (j <= sum(self%radial_integration_orders(:isection)))
                    ! add the points string (with one spherical_harmonics value)
                    ! multiplied with the correct coefficients to the result bubbles
                    do n = 1, max_n
                        bubbles_val(first_point:first_point+self%points_per_string(j, i)-1, n) = &
                            bubbles_val(first_point:first_point+self%points_per_string(j, i)-1, n) &
                                + spherical_harmonics_points(j, n) * &
                                point_values(i)%values(offset+1:offset+self%points_per_string(j, i)) * &
                                becke_weights(i)%values(offset+1:offset+self%points_per_string(j, i))
                    end do
                    !print '(i3, "value at 0.75", e11.4, "input value:", e11.4)', j,  &
                    !    bubbles_val(m(1), 1), point_values(i)%p(offset+1+m(1))
                    offset = offset + self%points_per_string(j, i)
                    j = j + 1
                end do
                !qif (previous_section_lmax /= section_lmax) &
                !    bubbles_val(first_point, (previous_section_lmax+1)**2 + 1 :) = 0.0d0
            end do

            !bubbles_val = bubbles_val * 4.0d0 * pi
            do l = 0, result_bubbles%get_lmax()
                bubbles_val(:, l*l+1:(l+1)*(l+1)) = &
                      bubbles_val(:, l*l+1:(l+1)*(l+1)) &
                    * sqrt(dble(4*pi)/(2*l+1))
            end do
            bubbles_val(1, 1:) = 0.0d0
            call becke_weights(i)%destroy()
        end do ! loop of bubbles
        deallocate(becke_weights)
        ! if we choose to interpolate the value at the origo, then do the interpolation
        if (present(extrapolate_origo)) then
            if (extrapolate_origo) then
                call result_bubbles%extrapolate_origo(6, lmax = 0)
                
            end if
        end if
        !call scale_bubbles(result_bubbles)
        deallocate(spherical_weights)
        deallocate(spherical_harmonics_points)
    end subroutine

    !> Evaluates the potential and energy per particle for the selected XC-functional.
    !! In practice, this subroutine hides the LibXC-calls and the OpenMP parallelization 
    !! of them.
    !!
    !! This subroutine takes in density and contracted gradients (possibly not allocated),
    !! and allocates and fills the 'energy_per_particle' and 'potential' arrays. 
    subroutine XC_evaluate_points(self, density,  &
                                  energy_per_particle, potential_density, &
                                  contracted_gradients, potential_contracted_gradients)
        class(XC),        intent(in)                  :: self
        real(REAL64),     intent(in)                  :: density(:)
        real(REAL64),     intent(inout)                 :: energy_per_particle(:)
        !> The derivative of the result energy per particle with respect to density
        real(REAL64),     intent(inout)               :: potential_density(:)
        real(REAL64),     intent(in),    optional     :: contracted_gradients(:)
        !> The derivative of the result energy per particle with respect to contracted gradients
        real(REAL64),     intent(inout), optional     :: potential_contracted_gradients(:)
        logical                                       :: gga_correlation, gga_exchange
        real(REAL64),                    allocatable  :: correlation_energy(:), &
                                                         correlation_potential_density(:), &
                                                         correlation_potential_gradients(:), &
                                                         temp_density(:)
        integer                                       :: first_point, last_point, number_of_points, i, counter



        gga_correlation = &
            xc_f90_info_family(self%exchange_info_pointer) == XC_FAMILY_GGA
        gga_exchange = &
            xc_f90_info_family(self%correlation_info_pointer) == XC_FAMILY_GGA

        ! allocate the energy_per_particle and potential_density -arrays
        allocate(correlation_energy(size(density)), source = 0.0d0)
        allocate(correlation_potential_density(size(density)), source = 0.0d0)
        if (gga_exchange .or. gga_correlation) &
            allocate(correlation_potential_gradients(size(density)), source = 0.0d0)
        counter = 0
        do i = 1, size(density)
            if (density(i) < 0.0) then
                counter = counter + 1
            end if
            
        end do
#ifdef HAVE_OMP
        !$OMP PARALLEL &
        !$OMP& PRIVATE(first_point, last_point, number_of_points) &
        !$OMP& SHARED(correlation_energy, correlation_potential_density, correlation_potential_gradients) 
        first_point = size(density) / omp_get_num_threads() * (omp_get_thread_num()) + &
                      min(omp_get_thread_num()+1, mod(size(density), omp_get_num_threads()))
        last_point = size(density) / omp_get_num_threads() * (omp_get_thread_num()+1) + &
                      min(omp_get_thread_num()+2, mod(size(density), omp_get_num_threads())) - 1
        if (omp_get_thread_num()+1 == omp_get_num_threads()) last_point = last_point +1
#else
        first_point = 1
        last_point = size(density) 
#endif
        number_of_points = last_point-first_point + 1
        
        ! evaluate the correct exchange
        if (gga_exchange) then
            call xc_f90_gga_exc_vxc(self%exchange_function_pointer, number_of_points, &
                                    density(first_point), contracted_gradients(first_point),   &
                                    energy_per_particle(first_point), potential_density(first_point), &
                                    potential_contracted_gradients(first_point))
        else
            call xc_f90_lda_exc_vxc(self%exchange_function_pointer, number_of_points, &
                                    density(first_point), energy_per_particle(first_point), &
                                    potential_density(first_point))
        end if
        
        
        ! evaluate the correct correlation
        if (gga_correlation) then
            call xc_f90_gga_exc_vxc(self%correlation_function_pointer, number_of_points, &
                                density(first_point), contracted_gradients(first_point),           &
                                correlation_energy(first_point), correlation_potential_density(first_point), &
                                correlation_potential_gradients(first_point))
        
            ! add the stuff together
            if (gga_exchange) then
                potential_contracted_gradients(first_point:last_point) = &
                      potential_contracted_gradients(first_point:last_point) &
                    + correlation_potential_gradients(first_point:last_point)
            else
                potential_contracted_gradients(first_point:last_point) = &
                    correlation_potential_gradients(first_point:last_point)
            end if
        else
            call xc_f90_lda_exc_vxc(self%correlation_function_pointer, number_of_points, &
                                density(first_point),  correlation_energy(first_point), &
                                correlation_potential_density(first_point))
        end if
        energy_per_particle(first_point:last_point) =   energy_per_particle(first_point:last_point) &
                                                      + correlation_energy(first_point:last_point)
        potential_density(first_point:last_point)   =   potential_density(first_point:last_point) &
                                                      + correlation_potential_density(first_point:last_point)
        !energy_per_particle(first_point:last_point) = -1.0d0 *  sign&
        !        (energy_per_particle(first_point:last_point), density(first_point:last_point))
        !potential_density(first_point:last_point) = -1.0d0 * sign&
        !        (potential_density(first_point:last_point), density(first_point:last_point))
#ifdef HAVE_OMP
        !$OMP END PARALLEL
#endif
        if (allocated(correlation_potential_gradients)) deallocate(correlation_potential_gradients)
        deallocate(correlation_potential_density, correlation_energy)
    end subroutine
    
    
    
    subroutine scale_bubbles(bubbls)
        type(Bubbles), intent(inout)      :: bubbls
        
        integer                           :: i, l, n, m
        real(REAL64),  pointer            :: coord(:)
        real(REAL64), allocatable         :: factors(:)
        real(REAL64), parameter           :: steepness = 5.0d0
        real(REAL64)                      :: distance_threshold
        
        do i = 1, bubbls%get_nbub()
            coord => bubbls%gr(i)%p%get_coord()
            do l = 4, bubbls%get_lmax()
                distance_threshold = 0.6d0
                factors =  (erf(steepness * (distance_threshold - coord))+1.0d0)/2.0d0
                factors = 1.0d0 - factors
                do m = 1, 2*l+1
                    bubbls%bf(i)%p(:, l*l+m) = bubbls%bf(i)%p(:, l*l+m) * factors(:)
                end do
                deallocate(factors)
            end do
            
        end do
    end subroutine
    
    subroutine scale(bubbls, overlapping)
        type(Bubbles), intent(inout)      :: bubbls
        logical                           :: overlapping
        integer                           :: i, j, n, m(1)
        real(REAL64),  pointer            :: centers(:, :), coord(:)
        real(REAL64)                      :: min_distance
        real(REAL64),  allocatable        :: factors(:), dfactors(:)
        real(REAL64), parameter           :: steepness = 8.0d0
        
        do i = 1, bubbls%get_nbub()
            min_distance = 60.d0
            centers => bubbls%get_centers()
            do j = 1, i-1
                min_distance = min(sqrt(sum((centers(:, j) - centers(:, i))**2)), min_distance)
            end do
            do j = i+1, bubbls%get_nbub()
                min_distance = min(sqrt(sum((centers(:, j) - centers(:, i))**2)), min_distance)
            end do
            min_distance = min_distance
            coord => bubbls%gr(i)%p%get_coord()
            if (overlapping) then
                factors  =  w(coord)
            else
                factors = w2(coord)
            end if
            dfactors = factors(1:size(coord)-1) - factors(2:size(coord))
            forall (n = 1 : (bubbls%get_lmax() + 1)**2 )
                bubbls%bf(i)%p(:, n) = bubbls%bf(i)%p(:, n) * factors
            end forall
            deallocate(factors, dfactors)
        end do
    end subroutine
    
     elemental function w(r)
        real(REAL64), intent(in) :: r
        real(REAL64) :: w, b

        b = r+0.d0
        if(r>epsilon(r)) then
            w=0.5d0*erfc(4.2d0  * b-1.1d0/(b))
        else
            w=1.d0
        end if
    end function
    
    elemental function w2(r)
        real(REAL64), intent(in) :: r
        real(REAL64) :: w2, b

        b = r+0.d0
        if(r>epsilon(r)) then
            w2=0.5d0*erfc(6.2d0  * b-1.3d0/(b))
        else
            w2=1.d0
        end if
    end function
    
    subroutine scale_cube(bubbls, cube, grid)
        type(Bubbles), intent(in)      :: bubbls
        real(REAL64),  intent(inout)   :: cube(:, :, :)
        type(Grid3D),  intent(in)      :: grid   
        !> The threshold from bubbles where the contribution from bubbles
        !! and cube is the same
        real(REAL64), parameter        :: distance_threshold = 1.00d0
        !> The threshold from bubbles where the contribution from bubbles
        !! and cube is the same
        real(REAL64), parameter        :: steepness = 5.0d0
        real(REAL64)                   :: factor_cube(size(cube, 1), size(cube, 2), size(cube, 3)), &
                                          factor_cube2(size(cube, 1), size(cube, 2), size(cube, 3))
        real(REAL64)                   :: gridpoints_x(grid%axis(X_)%get_shape()), &
                                          gridpoints_y(grid%axis(Y_)%get_shape()), &
                                          gridpoints_z(grid%axis(Z_)%get_shape())
        real(REAL64)                   :: center(3)
        real(REAL64), allocatable      :: bubble_factors(:)
        type(Grid1D), pointer          :: bubble_grid
        integer                        :: i, n, o, p, q

        ! get the minimum distance to any bubble for each grid point and scale the bubbles
        do i = 1, bubbls%get_nbub()
            center = bubbls%get_centers(i)
            gridpoints_x = grid%axis(X_)%get_coord() - center(X_)
            gridpoints_y = grid%axis(Y_)%get_coord() - center(Y_)
            gridpoints_z = grid%axis(Z_)%get_coord() - center(Z_)
            ! evaluate r and positions relative to center of the farfield box for each grid point
            if (i == 1) then
                forall(o = 1 : grid%axis(X_)%get_shape(), p = 1 : grid%axis(Y_)%get_shape(), q = 1 : grid%axis(Z_)%get_shape())
                    factor_cube(o, p, q) = sqrt(gridpoints_x(o) ** 2 + gridpoints_y(p) ** 2 + gridpoints_z(q) ** 2 )
                end forall 
            else
                forall(o = 1 : grid%axis(X_)%get_shape(), p = 1 : grid%axis(Y_)%get_shape(), q = 1 : grid%axis(Z_)%get_shape())
                    factor_cube(o, p, q) = &
                        min(factor_cube(o, p, q), sqrt(gridpoints_x(o) ** 2 + gridpoints_y(p) ** 2 + gridpoints_z(q) ** 2 ))
                end forall 
            end if
        end do


        ! get the factor 1.0d0 - (erf(distance_threshold - r)+1)/2
        factor_cube  =  (erf(steepness * (distance_threshold - factor_cube))+1.0d0)/2.0d0
        factor_cube2 = 1.0d0 - factor_cube
        
        cube = cube * factor_cube2
    end subroutine

    subroutine merge_cube_and_bubbles_cube(bubbls, cube, bubbles_cube, grid)
        type(Bubbles), intent(in)      :: bubbls
        real(REAL64),  intent(inout)   :: cube(:, :, :)
        real(REAL64),  intent(in)      :: bubbles_cube(:, :, :)
        type(Grid3D),  intent(in)      :: grid   
        !> The threshold from bubbles where the contribution from bubbles
        !! and cube is the same
        real(REAL64), parameter        :: distance_threshold = 0.40d0
        !> The threshold from bubbles where the contribution from bubbles
        !! and cube is the same
        real(REAL64), parameter        :: steepness = 7.0d0
        real(REAL64)                   :: factor_cube(size(cube, 1), size(cube, 2), size(cube, 3)), &
                                          factor_cube2(size(cube, 1), size(cube, 2), size(cube, 3))
        real(REAL64)                   :: gridpoints_x(grid%axis(X_)%get_shape()), &
                                          gridpoints_y(grid%axis(Y_)%get_shape()), &
                                          gridpoints_z(grid%axis(Z_)%get_shape())
        real(REAL64)                   :: center(3)
        real(REAL64), allocatable      :: bubble_factors(:)
        type(Grid1D), pointer          :: bubble_grid
        integer                        :: i, n, o, p, q

        ! get the minimum distance to any bubble for each grid point and scale the bubbles
        do i = 1, bubbls%get_nbub()
            center = bubbls%get_centers(i)
            gridpoints_x = grid%axis(X_)%get_coord() - center(X_)
            gridpoints_y = grid%axis(Y_)%get_coord() - center(Y_)
            gridpoints_z = grid%axis(Z_)%get_coord() - center(Z_)
            ! evaluate r and positions relative to center of the farfield box for each grid point
            if (i == 1) then
                forall(o = 1 : grid%axis(X_)%get_shape(), p = 1 : grid%axis(Y_)%get_shape(), q = 1 : grid%axis(Z_)%get_shape())
                    factor_cube(o, p, q) = sqrt(gridpoints_x(o) ** 2 + gridpoints_y(p) ** 2 + gridpoints_z(q) ** 2 )
                end forall 
            else
                forall(o = 1 : grid%axis(X_)%get_shape(), p = 1 : grid%axis(Y_)%get_shape(), q = 1 : grid%axis(Z_)%get_shape())
                    factor_cube(o, p, q) = &
                        min(factor_cube(o, p, q), sqrt(gridpoints_x(o) ** 2 + gridpoints_y(p) ** 2 + gridpoints_z(q) ** 2 ))
                end forall 
            end if
        end do


        ! get the factor 1.0d0 - (erf(distance_threshold - r)+1)/2
        factor_cube  =  (erf(steepness * (distance_threshold - factor_cube))+1.0d0)/2.0d0
        factor_cube2 = 1.0d0 - factor_cube
        cube(:, :, :) = cube(:, :, :) *  factor_cube2(:, :, :)
        cube(:, :, :) = cube(:, :, :) +  factor_cube(:, :, :) * bubbles_cube(:, :, :)

    end subroutine

    subroutine merge_cube_and_bubbles(bubbls, cube, grid)
        type(Bubbles), intent(inout)   :: bubbls
        real(REAL64),  intent(inout)   :: cube(:, :, :)
        type(Grid3D),  intent(in)      :: grid   
        !> The threshold from bubbles where the contribution from bubbles
        !! and cube is the same
        real(REAL64), parameter        :: distance_threshold = 12.0d0
        !> The threshold from bubbles where the contribution from bubbles
        !! and cube is the same
        real(REAL64), parameter        :: steepness = 3.0d0
        real(REAL64)                   :: factor_cube(size(cube, 1), size(cube, 2), size(cube, 3))
        real(REAL64)                   :: gridpoints_x(grid%axis(X_)%get_shape()), &
                                          gridpoints_y(grid%axis(Y_)%get_shape()), &
                                          gridpoints_z(grid%axis(Z_)%get_shape())
        real(REAL64)                   :: center(3)
        real(REAL64), allocatable      :: bubble_factors(:)
        type(Grid1D), pointer          :: bubble_grid
        integer                        :: i, n, o, p, q

        ! get the minimum distance to any bubble for each grid point and scale the bubbles
        do i = 1, bubbls%get_nbub()
            center = bubbls%get_centers(i)
            gridpoints_x = grid%axis(X_)%get_coord() - center(X_)
            gridpoints_y = grid%axis(Y_)%get_coord() - center(Y_)
            gridpoints_z = grid%axis(Z_)%get_coord() - center(Z_)
            ! evaluate r and positions relative to center of the farfield box for each grid point
            if (i == 1) then
                forall(o = 1 : grid%axis(X_)%get_shape(), p = 1 : grid%axis(Y_)%get_shape(), q = 1 : grid%axis(Z_)%get_shape())
                    factor_cube(o, p, q) = sqrt(gridpoints_x(o) ** 2 + gridpoints_y(p) ** 2 + gridpoints_z(q) ** 2 )
                end forall 
            else
                forall(o = 1 : grid%axis(X_)%get_shape(), p = 1 : grid%axis(Y_)%get_shape(), q = 1 : grid%axis(Z_)%get_shape())
                    factor_cube(o, p, q) = &
                        min(factor_cube(o, p, q), sqrt(gridpoints_x(o) ** 2 + gridpoints_y(p) ** 2 + gridpoints_z(q) ** 2 ))
                end forall 
            end if

            bubble_grid => bubbls%get_grid(i)
            bubble_factors = (erf(steepness * (distance_threshold - bubble_grid%get_coord()))+1.0d0)/2.0d0 
            forall (n = 1 : (bubbls%get_lmax()+1)**2)
                bubbls%bf(i)%p(:, n) = bubble_factors(:) * bubbls%bf(i)%p(:, n)
            end forall
            deallocate(bubble_factors)
            nullify(bubble_grid)
        end do

        ! get the factor 1.0d0 - (erf(distance_threshold - r)+1)/2
        factor_cube = 1.0d0 - (erf(steepness * (distance_threshold - factor_cube))+1.0d0)/2.0d0
        cube(:, :, :) = cube(:, :, :) * factor_cube(:, :, :)
        
    end subroutine

    

    subroutine smoothen_cube(bubbls, cube, grid)
        type(Bubbles), intent(inout)   :: bubbls
        real(REAL64),  intent(inout)   :: cube(:, :, :)
        type(Grid3D),  intent(in)      :: grid   
        real(REAL64)                   :: center(3)
        integer                        :: i, x, y, z, nlip, ix, iy, iz

        nlip = grid%get_nlip()

        ! get the minimum distance to any bubble for each grid point and scale the bubbles
        do i = 1, bubbls%get_nbub()
            center = bubbls%get_centers(i)
            ix =   grid%axis(X_)%get_icell(center(X_)) * (nlip-1) +1 &
                + grid%lip%get_first()
            iy =   grid%axis(Y_)%get_icell(center(Y_)) * (nlip-1) +1 &
                + grid%lip%get_first()
            z = grid%axis(Z_)%get_icell(center(Z_)) * (nlip-1) +1 &
                 + grid%lip%get_first()

            x = nint(ix + grid%axis(X_)%x2cell(center(X_)))

            do x = ix-4, ix+4

            do y = iy-4, iy+4
            cube(x, y, z) = &
                (cube(x, y, z-4) + cube(x, y, z+4)) / 2.0d0 
            cube(x, y, z-2) = &
                (cube(x, y, z-4) + cube(x, y, z)) / 2.0d0 
            cube(x, y, z+2) = &
                (cube(x, y, z) + cube(x, y, z+4)) / 2.0d0 
            cube(x, y, z-1) = &
                (cube(x, y, z-2) + cube(x, y, z)) / 2.0d0 
            cube(x, y, z+1) = &
                (cube(x, y, z) + cube(x, y, z+2)) / 2.0d0 
            cube(x, y, z-3) = &
                (cube(x, y, z-4) + cube(x, y, z-2)) / 2.0d0 
            cube(x, y, z+3) = &
                (cube(x, y, z+4) + cube(x, y, z+2)) / 2.0d0 
            end do
            end do

            x = nint(ix + grid%axis(X_)%x2cell(center(X_)))

            print *, "after", iy-3, cube(x, iy-3, z-3: z+3)

        end do
        

    end subroutine

    subroutine XC_evaluate_bubbles(self, &
                                   evaluate_gradients, input_density, &
                                   energy_per_particle, potential, &
                                   potential_gradients_x_bubbles, &
                                   potential_gradients_y_bubbles, &
                                   potential_gradients_z_bubbles, &
                                   divergence_bubbles, &
                                   derivative_x, derivative_y, derivative_z, &
                                   energy_density, &
                                   only_diagonal_bubbles, &
                                   core_density, &
                                   occupied_orbitals)
        class(XC),        intent(inout),        target     :: self
        logical,                            intent(in)     :: evaluate_gradients
        type(Function3D),                   intent(in)     :: input_density
        type(Bubbles),                      intent(inout)  :: energy_per_particle
        type(Bubbles),                      intent(inout)  :: potential
        type(Bubbles),                      intent(inout)  :: potential_gradients_x_bubbles
        type(Bubbles),                      intent(inout)  :: potential_gradients_y_bubbles
        type(Bubbles),                      intent(inout)  :: potential_gradients_z_bubbles
        type(Bubbles),    optional,         intent(inout)  :: energy_density 
        type(Bubbles),    optional,         intent(inout)  :: divergence_bubbles
        type(Function3D), optional,         intent(in)     :: derivative_x, derivative_y, derivative_z
        logical, optional,                  intent(in)     :: only_diagonal_bubbles
        type(Function3D), optional,         intent(inout)  :: core_density(:)
        !> The occupied orbitals in an array
        type(Function3D), optional,         intent(in)     :: occupied_orbitals(:)
        type(Points),             allocatable              :: potential_density(:), &
                                                              potential_contracted_gradients(:), &
                                                              energy_per_particle_(:), &
                                                              evaluated_density(:), &
                                                              contracted_gradients(:), &
                                                              gradients_x(:), gradients_y(:), gradients_z(:), &
                                                              potential_gradients_x(:),  &
                                                              potential_gradients_y(:), &
                                                              potential_gradients_z(:), factors(:), energy_density_(:), temp(:), &
                                                              divergence(:)
        type(BubblesEvaluator), pointer                    :: bubbles_evaluator
                                                              
        integer                                            ::  i, j, counter
        logical                                            :: only_diagonal_bubbles_
        integer, allocatable                               :: ibubs(:)
        type(Function3D),         allocatable              :: core_derivative_x(:), core_derivative_y(:), &
                                                              core_derivative_z(:)
                                                              
        if (present(core_density)) then
            call self%core_evaluator%evaluate_core_gradient(core_density, core_derivative_x, &
                                                            core_derivative_y, core_derivative_z)
        end if
        
        only_diagonal_bubbles_ = .FALSE.
        if (present(only_diagonal_bubbles)) only_diagonal_bubbles_ = only_diagonal_bubbles
        
        call bigben%split('XC-bubbles')
        allocate(contracted_gradients(input_density%bubbles%get_nbub()))
        allocate(gradients_x(input_density%bubbles%get_nbub()))
        allocate(gradients_y(input_density%bubbles%get_nbub()))
        allocate(gradients_z(input_density%bubbles%get_nbub()))
        allocate(potential_contracted_gradients(input_density%bubbles%get_nbub()))
        allocate(potential_gradients_x(input_density%bubbles%get_nbub()))
        allocate(potential_gradients_y(input_density%bubbles%get_nbub()))
        allocate(potential_gradients_z(input_density%bubbles%get_nbub()))
        allocate(evaluated_density(input_density%bubbles%get_nbub()))
        allocate(potential_density(input_density%bubbles%get_nbub()))
        allocate(energy_per_particle_(input_density%bubbles%get_nbub()))
        if (present(energy_density)) allocate(energy_density_(input_density%bubbles%get_nbub()))
        allocate(temp(input_density%bubbles%get_nbub()))
        counter = 0
        do i = 1, input_density%bubbles%get_nbub()
            bubbles_evaluator => self%bubbles_points_evaluators(i)%get_bubbles_evaluator()
            if (only_diagonal_bubbles_) then
                ibubs = [input_density%bubbles%get_ibub(i)]
            else
                ibubs = input_density%bubbles%get_ibubs()
            end if
            
            evaluated_density(i) = Points(self%density_evaluation_points(i))
            energy_per_particle_(i) = Points(self%density_evaluation_points(i))
            potential_density(i) = Points(self%density_evaluation_points(i))
                   
            ! evaluate all the gga stuff, if we have a gga functional
            if (evaluate_gradients) then
                gradients_x(i) = Points(self%density_evaluation_points(i))
                gradients_y(i) = Points(self%density_evaluation_points(i))
                gradients_z(i) = Points(self%density_evaluation_points(i))
                if (self%orbitals_density_evaluation) then
                    call self%evaluate_electron_density_bubbles_points(occupied_orbitals, &
                             self%bubbles_points_evaluators(i), &
                             evaluated_density(i), &
                             evaluate_gradients, gradients_x(i), gradients_y(i), gradients_z(i))
                else
                    call self%bubbles_points_evaluators(i)%evaluate_points(input_density, &
                                                                        evaluated_density(i), &
                                                                        derivative_x = derivative_x,  &
                                                                        derivative_y = derivative_y,  &
                                                                        derivative_z = derivative_z, &
                                                                        derivative_points_x = gradients_x(i), &
                                                                        derivative_points_y = gradients_y(i), &
                                                                        derivative_points_z = gradients_z(i), &
                                                                        ignore_cube = only_diagonal_bubbles_, &
                                                                        ibubs = ibubs)
                end if
!                  print *, "DENSITY", i
                if (present(core_density)) then
                    call self%core_evaluator%evaluate_points_and_collapse &
                            (core_density, evaluated_density(i), input_density%bubbles, .TRUE.)
                    call self%core_evaluator%evaluate_points_and_collapse &
                            (core_derivative_x, gradients_x(i), derivative_x%bubbles, .FALSE.)
                    call self%core_evaluator%evaluate_points_and_collapse &
                            (core_derivative_y, gradients_y(i), derivative_y%bubbles, .FALSE.)
                    call self%core_evaluator%evaluate_points_and_collapse &
                            (core_derivative_z, gradients_z(i), derivative_z%bubbles, .FALSE.)
                end if
                
                ! calculate the contracted gradients
                contracted_gradients(i) = gradients_x(i) * gradients_x(i)
                temp(i)                 = gradients_y(i) * gradients_y(i)
                call contracted_gradients(i)%add_in_place(temp(i))
                call temp(i)%destroy()
                temp(i)                 = gradients_z(i) * gradients_z(i)
                call contracted_gradients(i)%add_in_place(temp(i))
                call temp(i)%destroy()

                
                potential_contracted_gradients(i) = Points(self%density_evaluation_points(i))

                !> call libxc routines to get the energy per particle and its derivatives with respect to
                !! density and contracted gradients at the points required to project bubbles.
                call self%evaluate_points(evaluated_density(i)%values, &
                                          energy_per_particle_(i)%values, &
                                          potential_density(i)%values, &
                                          contracted_gradients = contracted_gradients(i)%values, &
                                          potential_contracted_gradients = potential_contracted_gradients(i)%values)
    
            else
                
                if (self%orbitals_density_evaluation) then
                    call self%evaluate_electron_density_bubbles_points(occupied_orbitals, &
                             self%bubbles_points_evaluators(i), &
                             evaluated_density(i), &
                             evaluate_gradients)
                else
                    call self%bubbles_points_evaluators(i)%evaluate_points(input_density, &
                                                                        evaluated_density(i), &
                                                                        ignore_cube = only_diagonal_bubbles_, &
                                                                        ibubs = ibubs)
                end if
                
                call self%evaluate_points(evaluated_density(i)%values, &
                                          energy_per_particle_(i)%values, &
                                          potential_density(i)%values)
            end if
            deallocate(ibubs)
            
            if (.FALSE.) then     
                
                ibubs = [input_density%bubbles%get_ibub(i)]
                ! re-evaluate the gradients only for the diagonal bubbles
                call self%bubbles_points_evaluators(i)%evaluate_points(input_density, &
                                                                       evaluated_density(i), &
                                                                       derivative_x = derivative_x, &
                                                                       derivative_y = derivative_y, &
                                                                       derivative_z = derivative_z, &
                                                                       derivative_points_x = gradients_x(i), &
                                                                       derivative_points_y = gradients_y(i), &
                                                                       derivative_points_z = gradients_z(i), &
                                                                       ignore_cube = .TRUE., &
                                                                       ibubs = ibubs)
              
            end if
            
            if (present(energy_density)) energy_density_(i)    = evaluated_density(i) * energy_per_particle_(i)
            call evaluated_density(i)%destroy() 
            
            if (evaluate_gradients) then
                potential_gradients_x(i) = potential_contracted_gradients(i) * gradients_x(i)
                potential_gradients_y(i) = potential_contracted_gradients(i) * gradients_y(i)
                potential_gradients_z(i) = potential_contracted_gradients(i) * gradients_z(i)
                call potential_contracted_gradients(i)%destroy()
                call gradients_x(i)%destroy()
                call gradients_y(i)%destroy()
                call gradients_z(i)%destroy()
                call contracted_gradients(i)%destroy()
            end if
            call self%bubbles_points_evaluators(i)%destroy_stored_objects()
        end do
        if (present(core_density)) then
            call destroy_core_functions(core_derivative_x)
            call destroy_core_functions(core_derivative_y)
            call destroy_core_functions(core_derivative_z)
        end if
        deallocate(evaluated_density)
        deallocate(gradients_x, gradients_y, gradients_z)
        deallocate(contracted_gradients, potential_contracted_gradients)
        
        ! project the energy density to bubbles
        if (present(energy_density)) call self%project_to_bubbles(energy_density_, energy_density, &
                                     extrapolate_origo = .TRUE., use_becke_weights = .TRUE.)
        
        ! project the energy per particle to bubbles
        call self%project_to_bubbles(energy_per_particle_, energy_per_particle, &
                                     extrapolate_origo = .TRUE., use_becke_weights = .TRUE.)

        ! project the derivative of energy with respect to the density to bubbles
        call self%project_to_bubbles(potential_density, potential, extrapolate_origo = .TRUE., use_becke_weights = .TRUE.)

        ! project the derivative of energy with respect to the contracted gradients to bubbles
        if (evaluate_gradients) then
            if (present(divergence_bubbles)) then
                allocate(divergence(input_density%bubbles%get_nbub()))
                call self%project_to_bubbles(potential_gradients_x, potential_gradients_x_bubbles, &
                                            extrapolate_origo = .TRUE., use_becke_weights = .FALSE.)
                call self%project_to_bubbles(potential_gradients_y, potential_gradients_y_bubbles, &
                                            extrapolate_origo = .TRUE., use_becke_weights = .FALSE.)
                call self%project_to_bubbles(potential_gradients_z, potential_gradients_z_bubbles, &
                                            extrapolate_origo = .TRUE., use_becke_weights = .FALSE.)
                bubbles_evaluator => self%grid_points_evaluator%get_bubbles_evaluator()
                call bubbles_evaluator%evaluate_divergence_as_bubbles(potential_gradients_x_bubbles, &
                    potential_gradients_y_bubbles, potential_gradients_z_bubbles, divergence_bubbles) 
                    
                do i = 1, input_density%bubbles%get_nbub()
                    divergence(i) = Points(self%density_evaluation_points(i))
                    bubbles_evaluator => self%bubbles_points_evaluators(i)%get_bubbles_evaluator()
                    ibubs = [input_density%bubbles%get_ibub(i)]
                    call bubbles_evaluator%evaluate_points(divergence_bubbles, &
                                                           divergence(i), &
                                                           ibubs = ibubs)
                end do
                call self%project_to_bubbles(divergence, divergence_bubbles, &
                                        extrapolate_origo = .TRUE., use_becke_weights = .TRUE.)
                do i = 1, input_density%bubbles%get_nbub()
                    call divergence(i)%destroy()
                end do
                deallocate(divergence)
            end if
            
            call self%project_to_bubbles(potential_gradients_x, potential_gradients_x_bubbles, &
                                        extrapolate_origo = .TRUE., use_becke_weights = .TRUE.)
            call self%project_to_bubbles(potential_gradients_y, potential_gradients_y_bubbles, &
                                        extrapolate_origo = .TRUE., use_becke_weights = .TRUE.)
            call self%project_to_bubbles(potential_gradients_z, potential_gradients_z_bubbles, &
                                        extrapolate_origo = .TRUE., use_becke_weights = .TRUE.)
            !call smoothen_bubbles(potential_gradients_x_bubbles)
            !call smoothen_bubbles(potential_gradients_y_bubbles)
            !call smoothen_bubbles(potential_gradients_z_bubbles)
        end if
        
        do i = 1, input_density%bubbles%get_nbub()
            call potential_density(i)%destroy()
            call energy_per_particle_(i)%destroy()
            if (present(energy_density)) call energy_density_(i)%destroy()
            if (evaluate_gradients) then
                call potential_gradients_x(i)%destroy()
                call potential_gradients_y(i)%destroy()
                call potential_gradients_z(i)%destroy()
            end if
        end do
        deallocate(potential_density, energy_per_particle_, potential_gradients_x, &
                   potential_gradients_y, potential_gradients_z, temp)
        if (present(energy_density)) deallocate(energy_density_)
        call bigben%stop()
    end subroutine 

    subroutine XC_evaluate_cores(self, evaluate_gradients, core_density, &
                                 energy_per_particle, potential, &
                                 derivative_x, derivative_y, derivative_z, &
                                 potential_gradients_x, potential_gradients_y, potential_gradients_z , &
                                 energy_density, divergence)
        class(XC),                target,   intent(inout)  :: self
        type(Function3D),                   intent(in)     :: core_density(:)
        logical,                            intent(in)     :: evaluate_gradients
        type(Function3D),                   intent(inout)  :: energy_per_particle
        type(Function3D),                   intent(inout)  :: potential
        type(Function3D),         optional, intent(inout)  :: divergence
        type(Function3D),         optional, intent(inout)  :: energy_density
        type(Function3D),         optional, intent(in)     :: derivative_x, derivative_y, derivative_z
        type(Function3D),         optional, intent(in)     :: potential_gradients_x, potential_gradients_y, potential_gradients_z
        type(Function3D),         allocatable              :: core_derivative_x(:), core_derivative_y(:), &
                                                              core_derivative_z(:),   &
                                                              core_potential_gradients_x(:), &
                                                              core_potential_gradients_y(:), &
                                                              core_potential_gradients_z(:), &
                                                              core_energy_density(:), &
                                                              core_energy_per_particle(:), &
                                                              core_potential(:), &
                                                              core_divergence(:)
        integer                                            :: i
        type(Function3DEvaluator), pointer                 :: cube_evaluator

        call self%core_evaluator%evaluate_core_gradient(core_density, core_derivative_x, &
                                                        core_derivative_y, core_derivative_z)
        if (present(energy_density)) call self%core_evaluator%init_core_functions(core_energy_density)
        call self%core_evaluator%init_core_functions(core_potential_gradients_x)
        call self%core_evaluator%init_core_functions(core_potential_gradients_y)
        call self%core_evaluator%init_core_functions(core_potential_gradients_z)
        call self%core_evaluator%init_core_functions(core_energy_per_particle)
        call self%core_evaluator%init_core_functions(core_potential)
        call self%core_evaluator%init_core_functions(core_divergence)

        do i = 1, size(core_density)
            cube_evaluator => self%core_evaluator%core_grid_evaluators(i)
            if (evaluate_gradients) then
                core_derivative_x(i)%bubbles = Bubbles(derivative_x%bubbles, copy_content = .TRUE.)
                core_derivative_y(i)%bubbles = Bubbles(derivative_y%bubbles, copy_content = .TRUE.)
                core_derivative_z(i)%bubbles = Bubbles(derivative_z%bubbles, copy_content = .TRUE.)
                
                call self%evaluate_cube(cube_evaluator, evaluate_gradients, core_density(i), &
                                core_density(i)%grid, core_energy_per_particle(i)%cube, &
                                core_potential(i)%cube, core_potential_gradients_x(i)%cube, &
                                core_potential_gradients_y(i)%cube, core_potential_gradients_z(i)%cube, &
                                core_derivative_x(i), core_derivative_y(i), core_derivative_z(i), &
                                energy_density = core_energy_density(i)%cube &
                                )
                call core_derivative_x(i)%bubbles%destroy()
                call core_derivative_y(i)%bubbles%destroy()
                call core_derivative_z(i)%bubbles%destroy()
            else
                
                call self%evaluate_cube(cube_evaluator, evaluate_gradients, core_density(i), &
                                core_density(i)%grid, core_energy_per_particle(i)%cube, &
                                core_potential(i)%cube, core_potential_gradients_x(i)%cube, &
                                core_potential_gradients_y(i)%cube, core_potential_gradients_z(i)%cube, &
                                energy_density = core_energy_density(i)%cube &
                                )
            end if

            if (evaluate_gradients) then
                call core_potential_gradients_x(i)%inject_bubbles_to_cube(potential_gradients_x%bubbles, factor = -1.0d0)
                call core_potential_gradients_y(i)%inject_bubbles_to_cube(potential_gradients_y%bubbles, factor = -1.0d0)
                call core_potential_gradients_z(i)%inject_bubbles_to_cube(potential_gradients_z%bubbles, factor = -1.0d0)
                call core_potential(i)%inject_bubbles_to_cube(potential%bubbles, factor = -1.0d0)
                if (present(energy_density)) &
                    call core_energy_density(i)%inject_bubbles_to_cube(energy_density%bubbles, factor = -1.0d0)
                call core_energy_per_particle(i)%inject_bubbles_to_cube(energy_per_particle%bubbles, factor = -1.0d0)

                call cube_evaluator%evaluate_divergence_as_Function3D(core_potential_gradients_x(i), &
                                                                      core_potential_gradients_y(i), &
                                                                      core_potential_gradients_z(i), &
                                                                      core_divergence(i), ignore_bubbles = .TRUE.)
                 core_divergence(i)%cube = -2.0d0 * core_divergence(i)%cube
            end if
            nullify(cube_evaluator)
        end do
        
        if (present(energy_density)) call self%core_evaluator%collapse_core_functions(energy_density, core_energy_density)
        call self%core_evaluator%collapse_core_functions(energy_per_particle, core_energy_per_particle)
        call self%core_evaluator%collapse_core_functions(potential, core_potential)
        call self%core_evaluator%collapse_core_functions(divergence, core_divergence)
        call destroy_core_functions(core_derivative_x)
        call destroy_core_functions(core_derivative_y)
        call destroy_core_functions(core_derivative_z)
        call destroy_core_functions(core_potential_gradients_x)
        call destroy_core_functions(core_potential_gradients_y)
        call destroy_core_functions(core_potential_gradients_z)
        if (present(energy_density)) call destroy_core_functions(core_energy_density)
        call destroy_core_functions(core_energy_per_particle)
        call destroy_core_functions(core_divergence)
    end subroutine

    
    subroutine XC_evaluate_cube(self, density_evaluator, &
                                evaluate_gradients, input_density, output_grid, &
                                energy_per_particle, potential, &
                                potential_gradients_x_cube, &
                                potential_gradients_y_cube, &
                                potential_gradients_z_cube, &
                                derivative_x, derivative_y, derivative_z, &
                                energy_density, occupied_orbitals)
        class(XC),        intent(inout),        target     :: self
        type(Function3DEvaluator), pointer, intent(inout)  :: density_evaluator
        logical,                            intent(in)     :: evaluate_gradients
        type(Function3D),                   intent(in)     :: input_density
        type(Grid3D),                       intent(in)     :: output_grid
        real(REAL64),                       intent(inout)  :: energy_per_particle(:, :, :)
        real(REAL64),                       intent(inout)  :: potential(:, :, :)
        real(REAL64),                       intent(inout)  :: potential_gradients_x_cube(:, :, :)
        real(REAL64),                       intent(inout)  :: potential_gradients_y_cube(:, :, :)
        real(REAL64),                       intent(inout)  :: potential_gradients_z_cube(:, :, :)
        type(Function3D),         optional, intent(in)     :: derivative_x, derivative_y, derivative_z
        real(REAL64),             optional, intent(inout)  :: energy_density(:, :, :)
        type(Function3D),         optional, intent(in)     :: occupied_orbitals(:)
        real(REAL64),              allocatable             :: potential_density(:), &
                                                              potential_contracted_gradients(:), &
                                                              energy_per_particle_(:), &
                                                              contracted_gradients(:), evaluated_density_1d(:)
        real(REAL64),   allocatable, target                :: gradients_x(:, :, :), gradients_y(:, :, :), &
                                                              gradients_z(:, :, :), evaluated_density(:, :, :)
        real(REAL64),   contiguous,  pointer               :: gradients_x_pointer(:, :, :), gradients_y_pointer(:, :, :), &
                                                              gradients_z_pointer(:, :, :), evaluated_density_pointer(:, :, :), &
                                                               gradients_x_1d_pointer(:), &
                                                              gradients_y_1d_pointer(:), gradients_z_1d_pointer(:)
        integer                                            :: output_shape(3), i, j, l(1), counter
        type(Function3D)                                   :: temp

        call bigben%split('XC-cube')
        output_shape = output_grid%get_shape()
#ifdef HAVE_CUDA
        evaluated_density_pointer => CudaCube_init_page_locked_cube(output_shape)
#else
        allocate(evaluated_density(output_shape(1), output_shape(2), output_shape(3)))
        evaluated_density_pointer => evaluated_density
#endif
        allocate(energy_per_particle_(product(output_shape)))
        allocate(potential_density(product(output_shape)))
        if (evaluate_gradients) then
            
            call bigben%split('Density Evaluation: cube')
            allocate(potential_contracted_gradients(product(output_shape)))
#ifdef HAVE_CUDA
            gradients_x_pointer => CudaCube_init_page_locked_cube(output_shape)
            gradients_y_pointer => CudaCube_init_page_locked_cube(output_shape)
            gradients_z_pointer => CudaCube_init_page_locked_cube(output_shape)
#else
            allocate(gradients_x(output_shape(1), output_shape(2), output_shape(3)))
            allocate(gradients_y(output_shape(1), output_shape(2), output_shape(3)))
            allocate(gradients_z(output_shape(1), output_shape(2), output_shape(3)))

            gradients_x_pointer => gradients_x
            gradients_y_pointer => gradients_y
            gradients_z_pointer => gradients_z
#endif
            gradients_x_1d_pointer(1:size(gradients_x_pointer)) => gradients_x_pointer(:, :, :)
            gradients_y_1d_pointer(1:size(gradients_y_pointer)) => gradients_y_pointer(:, :, :)
            gradients_z_1d_pointer(1:size(gradients_z_pointer)) => gradients_z_pointer(:, :, :)
            
            if (self%orbitals_density_evaluation) then
                call self%evaluate_electron_density_cube( &
                         occupied_orbitals, evaluated_density_pointer, evaluate_gradients, &
                         gradients_x_pointer, gradients_y_pointer, gradients_z_pointer)
            else
                call density_evaluator%evaluate_grid(input_density, &
                                                    result_cube = evaluated_density_pointer, &
                                                    derivative_x = derivative_x, &
                                                    derivative_y = derivative_y, &
                                                    derivative_z = derivative_z, &
                                                    derivative_cube_x = gradients_x_pointer, &
                                                    derivative_cube_y = gradients_y_pointer, &
                                                    derivative_cube_z = gradients_z_pointer)
            end if
            
            contracted_gradients =   gradients_x_1d_pointer * gradients_x_1d_pointer &
                                   + gradients_y_1d_pointer * gradients_y_1d_pointer &
                                   + gradients_z_1d_pointer * gradients_z_1d_pointer

            
            evaluated_density_1d  = reshape(evaluated_density_pointer, [product(output_shape)])
                     
            
            ! get the density 
            call bigben%stop()
            call bigben%split('Evaluate points: cube')

            ! evaluate the exchange and correlation with libxc at all cube points
            call self%evaluate_points(evaluated_density_1d, &
                                    energy_per_particle_, potential_density, &
                                    contracted_gradients = contracted_gradients, &
                                    potential_contracted_gradients = potential_contracted_gradients)
            call bigben%stop()
        else
            if (self%orbitals_density_evaluation) then
                call self%evaluate_electron_density_cube( &
                         occupied_orbitals, evaluated_density_pointer, evaluate_gradients)
            else
                call density_evaluator%evaluate_grid(input_density, result_cube = evaluated_density_pointer)
            end if
            
            
            
            evaluated_density_1d  = reshape(evaluated_density_pointer, [product(output_shape)])
            ! evaluate the exchange and correlation with libxc at all cube points
            call self%evaluate_points(evaluated_density_1d, &
                                    energy_per_particle_, potential_density)
        end if
        deallocate(evaluated_density_1d)



        counter = 0
!         do j = 1, size(contracted_gradients)
!             if (      abs(contracted_gradients(j)) * abs(evaluated_density(j)) < 1e-10 &
!                 .and. abs(evaluated_density(j)) < 1e-6 ) then
!                 potential_contracted_gradients(j) = 0.0d0
!                 counter = counter + 1
!             end if
!         end do
        
        if (allocated(contracted_gradients)) deallocate(contracted_gradients)

        ! reshape the 1d arrays to 3d-cubes
        if (present(energy_density)) then
            energy_density(:, :, :)      = reshape(energy_per_particle_ , output_shape)
            energy_density(:, :, :)      = energy_density(:, :, :) * evaluated_density_pointer
        end if
#ifdef HAVE_CUDA
        call CudaCube_destroy_page_locked_cube(evaluated_density_pointer)
#else
        deallocate(evaluated_density)
#endif
        energy_per_particle(:, :, :) = reshape(energy_per_particle_, output_shape)
        potential(:, :, :)           = reshape(potential_density,    output_shape)
        if (evaluate_gradients) then
            potential_gradients_x_cube(:, :, :) = reshape(  potential_contracted_gradients, output_shape)
            potential_gradients_x_cube(:, :, :) = potential_gradients_x_cube(:, :, :) * gradients_x_pointer(:, :, :)
            potential_gradients_y_cube(:, :, :) = reshape(  potential_contracted_gradients, output_shape)
            potential_gradients_y_cube(:, :, :) = potential_gradients_y_cube(:, :, :) * gradients_y_pointer(:, :, :)
            potential_gradients_z_cube(:, :, :) = reshape(  potential_contracted_gradients, output_shape)
            potential_gradients_z_cube(:, :, :) = potential_gradients_z_cube(:, :, :) * gradients_z_pointer(:, :, :)
#ifdef HAVE_CUDA
            call CudaCube_destroy_page_locked_cube(gradients_x_pointer)
            call CudaCube_destroy_page_locked_cube(gradients_y_pointer)
            call CudaCube_destroy_page_locked_cube(gradients_z_pointer)
#else
            deallocate(gradients_x, gradients_y, gradients_z)
#endif
            deallocate(potential_contracted_gradients)
        end if

        
        ! clean up
        deallocate(potential_density, energy_per_particle_)
        call bigben%stop()
    end subroutine

    subroutine smoothen_bubbles(bubbls)
        type(Bubbles), intent(inout)                        :: bubbls
        real(REAL64), pointer                               :: coord(:) 
        real(REAL64), allocatable                           :: centers(:, :)
        integer                                             :: i, j, k, closest_point(1), icell, first, last, nlip, point
        real(REAL64)                                        :: distance, difference_per_grid_point, total_distance
        real(REAL64), allocatable                           :: diff(:)
        centers = bubbls%get_centers()
        do i = 1,bubbls%get_nbub()
            coord => bubbls%gr(i)%p%get_coord()
            nlip = bubbls%gr(i)%p%get_nlip()
            do j = 1, bubbls%get_nbub()
                if (j /= i) then
                    distance = sqrt(sum((centers(:, i) - centers(:, j))**2))
                    closest_point = minloc(abs(coord-distance))
                    icell = bubbls%gr(i)%p%get_icell(distance)
                    first = closest_point(1) - 12
                    last =  closest_point(1) + 12
                    total_distance = coord(last) - coord(first)
                    diff = (bubbls%bf(i)%p(last, :) - bubbls%bf(i)%p(first, :))
                    do point = first, last
                        bubbls%bf(i)%p(point, :) =   bubbls%bf(i)%p(first, :) &
                                                + (coord(point) - coord(first)) / total_distance * diff
                    end do
                    
                    deallocate(diff)
                end if
            end do
        end do
        deallocate(centers)
    end subroutine

    !> Evaluate the effect to the xc-potential caused by the derivative of energy per particle with respect to
    !! the contracted gradients.
    subroutine XC_evaluate_divergence_potential_term(self, density_evaluator, &
                                                     divergence_f3d, potential_gradients_x, &
                                                     potential_gradients_y, potential_gradients_z, &
                                                     ignore_bubbles)
        class(XC),                 target,   intent(inout)  :: self
        type(Function3DEvaluator), pointer,  intent(inout)  :: density_evaluator
        type(Function3D),                    intent(inout)  :: divergence_f3d
        type(Function3D),                    intent(in)     :: potential_gradients_x, potential_gradients_y, &
                                                               potential_gradients_z
        logical,                   optional, intent(in)     :: ignore_bubbles
        integer                                             :: i, j, closest_point(1)
        real(REAL64)                                        :: distance
        type(Function3D)                                    :: temp
        logical                                             :: ignore_bubbles_
        
        ignore_bubbles_ = .FALSE.
        if (present(ignore_bubbles)) ignore_bubbles_ = ignore_bubbles
 
         call density_evaluator%evaluate_divergence_as_Function3D(potential_gradients_x, potential_gradients_y, &
                                                                  potential_gradients_z, &
                                                                  temp, ignore_bubbles = ignore_bubbles_)
        !call self%core_evaluator%evaluate_core_divergence_and_collapse(potential_gradients_x, potential_gradients_y, &
        !                                                          potential_gradients_z, &
        !                                                          temp)
        if (ignore_bubbles_) then
            call temp%bubbles%destroy()
            temp%bubbles = Bubbles(divergence_f3d%bubbles, copy_content = .TRUE.)
        end if
        call divergence_f3d%destroy()
        divergence_f3d = (-2.0d0) * temp
        call temp%destroy()

        
    end subroutine
    
    subroutine XC_evaluate_electron_density_laplacian(self, occupied_orbitals, laplacian)
        class(XC),        intent(inout), target     :: self
        !> Input orbitals that are occupied
        type(Function3D), intent(in)                :: occupied_orbitals(:)
        !> Output laplacian
        type(Function3D), intent(inout)             :: laplacian
        integer                                     :: i
        class(Function3D),         allocatable      :: kin   
        type(Function3D)                            :: temp, derivative_x, derivative_y, derivative_z

        do i = 1, size(occupied_orbitals)
            
            call self%laplacian_operator%operate_on(occupied_orbitals(i), kin)
            temp = kin * occupied_orbitals(i)
            call temp%product_in_place_REAL64(4.0d0)
            call laplacian%add_in_place(temp)
            call kin%destroy()
            call temp%destroy()
            deallocate(kin)
            
            call self%grid_points_evaluator%evaluate_gradients_as_Function3Ds &
                     (occupied_orbitals(i), derivative_x, derivative_y, derivative_z)
            temp = derivative_x * derivative_x
            call temp%product_in_place_REAL64(4.0d0)
            call laplacian%add_in_place(temp)
            call temp%destroy()
            call derivative_x%destroy()
            
            temp = derivative_y * derivative_y
            call temp%product_in_place_REAL64(4.0d0)
            call laplacian%add_in_place(temp)
            call temp%destroy()
            call derivative_y%destroy()
            
            temp = derivative_z * derivative_z
            call temp%product_in_place_REAL64(4.0d0)
            call laplacian%add_in_place(temp)
            call temp%destroy()
            call derivative_z%destroy()
            
        end do 
    end subroutine
    
    subroutine XC_evaluate_electron_density_bubbles_points(self, occupied_orbitals,  &
                                                 bubbles_points_evaluator, &
                                                 electron_density, &
                                                 evaluate_gradients,&
                                                 derivative_x, &
                                                 derivative_y, &
                                                 derivative_z)
        class(XC),        intent(inout), target     :: self
        !> The occupied orbitals in an array
        type(Function3D), intent(in)                :: occupied_orbitals(:)
        !> The evaluator used in evaluation of the electron density at the points of interest
        type(Function3DEvaluator), intent(inout)    :: bubbles_points_evaluator 
        !> If the gradients are evaluated
        logical,          intent(in)                :: evaluate_gradients
        !> output electron density as cubes
        type(Points),     intent(inout)             :: electron_density
        !> output derivatives as cubes
        type(Points),     intent(inout), optional   :: derivative_x, &
                                                       derivative_y, &
                                                       derivative_z
        type(Function3D)                            :: temp_x, temp_y, temp_z, temp_orbital
        type(Points)                                :: temp_points, temp_points2, temp_points_x, temp_points_y, temp_points_z
        type(Function3D),          allocatable      :: temp
        integer                                     :: i

        

        do i = 1, size(occupied_orbitals)

            
            if (evaluate_gradients) then
                temp_points = Points(electron_density)
                temp_points_x = Points(electron_density)
                temp_points_y = Points(electron_density)
                temp_points_z = Points(electron_density)
                call self%grid_points_evaluator%evaluate_gradients_as_Function3Ds &
                        (occupied_orbitals(i), temp_x, temp_y, temp_z)
                call bubbles_points_evaluator%evaluate_points(occupied_orbitals(i), &
                                                                       temp_points, &
                                                                       derivative_x = temp_x,  &
                                                                       derivative_y = temp_y,  &
                                                                       derivative_z = temp_z, &
                                                                       derivative_points_x = temp_points_x, &
                                                                       derivative_points_y = temp_points_y, &
                                                                       derivative_points_z = temp_points_z)
                                                                       
                
                call temp_x%destroy()
                call temp_y%destroy()
                call temp_z%destroy()
                
                temp_points2 = temp_points * temp_points
                call temp_points2%product_in_place_REAL64(2.0d0)
                call electron_density%add_in_place(temp_points2)
                call temp_points2%destroy()
                
                temp_points2 = temp_points * temp_points_x
                call temp_points2%product_in_place_REAL64(4.0d0)
                call derivative_x%add_in_place(temp_points2)
                call temp_points2%destroy()
                
                temp_points2 = temp_points * temp_points_y
                call temp_points2%product_in_place_REAL64(4.0d0)
                call derivative_y%add_in_place(temp_points2)
                call temp_points2%destroy()
                
                temp_points2 = temp_points * temp_points_z
                call temp_points2%product_in_place_REAL64(4.0d0)
                call derivative_z%add_in_place(temp_points2)
                call temp_points2%destroy()
                
                call temp_points%destroy()
                call temp_points_x%destroy()
                call temp_points_y%destroy()
                call temp_points_z%destroy()
            else
                temp_points = Points(electron_density)
                call bubbles_points_evaluator%evaluate_points(occupied_orbitals(i), &
                                                              temp_points)
                temp_points2 = temp_points * temp_points
                call temp_points2%product_in_place_REAL64(2.0d0)
                call electron_density%add_in_place(temp_points2)
                call temp_points2%destroy()
                call temp_points%destroy()
            end if
        end do
    end subroutine
    
    subroutine XC_evaluate_electron_density_cube(self, occupied_orbitals,  &
                                                 electron_density, &
                                                 evaluate_gradients,&
                                                 derivative_x, &
                                                 derivative_y, &
                                                 derivative_z)
        class(XC),        intent(inout), target     :: self
        !> The result v_xc potential object
        type(Function3D), intent(in)                :: occupied_orbitals(:)
        !> If the gradients are evaluated
        logical,          intent(in)                :: evaluate_gradients
        !> output electron density as cubes
        real(REAL64),     intent(out)               :: electron_density(:, :, :)
        !> output derivatives as cubes
        real(REAL64),     intent(out), optional     :: derivative_x(:, :, :), &
                                                       derivative_y(:, :, :), &
                                                       derivative_z(:, :, :)
        type(Function3D)                            :: temp_x, temp_y, temp_z, temp_orbital
        type(Function3D),          allocatable      :: temp
        integer                                     :: i, output_shape(3)

        output_shape = occupied_orbitals(1)%grid%get_shape()
        electron_density = 0.0d0
        
        if (evaluate_gradients) then
            derivative_x = 0.0d0
            derivative_y = 0.0d0
            derivative_z = 0.0d0
        end if
        
        do i = 1, size(occupied_orbitals)
            call temp_orbital%init_copy(occupied_orbitals(i), copy_content = .TRUE.)
            call temp_orbital%inject_bubbles_to_cube(temp_orbital%bubbles)
            
            electron_density = electron_density + 2.0d0 * temp_orbital%cube * temp_orbital%cube
            
            if (evaluate_gradients) then
                call self%grid_points_evaluator%evaluate_gradients_as_Function3Ds &
                        (occupied_orbitals(i), temp_x, temp_y, temp_z)
                call temp_x%inject_bubbles_to_cube(temp_x%bubbles)
                derivative_x = derivative_x + 4.0d0 * temp_orbital%cube * temp_x%cube 
                
                call temp_y%inject_bubbles_to_cube(temp_y%bubbles)
                derivative_y = derivative_y + 4.0d0 * temp_orbital%cube * temp_y%cube
                
                call temp_z%inject_bubbles_to_cube(temp_z%bubbles)
                derivative_z = derivative_z + 4.0d0 * temp_orbital%cube * temp_z%cube 
                
                call temp_x%destroy()
                call temp_y%destroy()
                call temp_z%destroy()
            end if
            call temp_orbital%destroy()
        end do


    end subroutine

    subroutine XC_reevaluate_electron_density(self, input_electron_density, output_electron_density)
        class(XC),        intent(inout), target     :: self
        !> The result v_xc potential object
        type(Function3D), intent(in)                :: input_electron_density
        !> The result v_xc potential object
        type(Function3D), intent(out)               :: output_electron_density
        integer                                     :: i, output_shape(3)
        type(Points),             allocatable       :: evaluated_density(:)

        
        call output_electron_density%init_copy(input_electron_density, lmax = 9)
        allocate(evaluated_density(input_electron_density%bubbles%get_nbub()))
        do i = 1, input_electron_density%bubbles%get_nbub()
            evaluated_density(i) = Points(point_coordinates = self%density_evaluation_points(i))
            call self%bubbles_points_evaluators(i)%evaluate_points(input_electron_density, evaluated_density(i))
        end do

        ! project the energy density to bubbles
        call self%project_to_bubbles(evaluated_density, output_electron_density%bubbles, &
                                     extrapolate_origo = .TRUE., use_becke_weights = .TRUE.)
        
        do i = 1, input_electron_density%bubbles%get_nbub()
            call evaluated_density(i)%destroy()
        end do
        deallocate(evaluated_density)
        call self%grid_points_evaluator%evaluate_grid(input_electron_density, output_electron_density%cube)
        call output_electron_density%inject_bubbles_to_cube(output_electron_density%bubbles, factor = -1.0d0)
        
    end subroutine
    

    subroutine XC_eval(self, density, potential, energy_per_particle, energy, occupied_orbitals)
        class(XC),        intent(inout), target     :: self
        !> The input electron density
        type(Function3D), intent(inout)             :: density
        !> The result v_xc potential object
        type(Function3D), intent(out)               :: potential
        !> The result energy density object
        type(Function3D), intent(out)               :: energy_per_particle
        !> The result energy
        real(REAL64), intent(out), optional         :: energy
        !> Input occupied orbitals
        type(Function3D), intent(in), optional      :: occupied_orbitals(:)
       
        ! dimension S, S: order of spherical integration
        real(REAL64), dimension(:), allocatable     :: tmp_exc, tmp_vxc, &
                                                       contracted_gradients,&
                                                       vsigma, gradients_x, gradients_y, gradients_z
        real(REAL64),                 allocatable   :: injection(:, :, :), integrals(:)
        type(Function3D)                            :: potential_contracted_gradients_x, &
                                                       potential_contracted_gradients_y, &
                                                       potential_contracted_gradients_z, &
                                                       temp, temp2
        real(REAL64), pointer                       :: exc_cube(:,:,:), vxc_cube(:,:,:)
        integer, dimension(3)                       :: xc_cube_shape
        type(Function3DEvaluator), pointer          :: evaluator
        type(BubblesEvaluator),    pointer          :: bubbles_evaluator
        type(CubeEvaluator),       pointer          :: cube_evaluator
        type(Function3D)                            :: input_density,  divergence_f3d, &
                                                       derivative_x, derivative_y, derivative_z, energy_density
        logical                                     :: first, evaluate_gradients
        integer                                     :: i, n_gridpoints
        type(Bubbles)                               :: temp_bubbles, temp_bubbles2, temp_bubbles3, temp_bubbles4
        type(Points), allocatable                   :: becke_points(:)
        type(Function3D), allocatable               :: core_density(:)

        call potential%init_copy(density, lmax = density%bubbles%get_lmax())
        potential = 0.0d0
        call potential_contracted_gradients_x%init_copy(density, lmax = density%bubbles%get_lmax())
        call potential_contracted_gradients_y%init_copy(density, lmax = density%bubbles%get_lmax())
        call potential_contracted_gradients_z%init_copy(density, lmax = density%bubbles%get_lmax())
        call energy_density%init_copy(density, lmax = density%bubbles%get_lmax())
        call energy_per_particle%init_copy(density, lmax = density%bubbles%get_lmax())
        call divergence_f3d%init_copy(density, lmax = density%bubbles%get_lmax())
        !call self%reevaluate_electron_density(density, temp)
        
        
        evaluate_gradients = &
            xc_f90_info_family(self%exchange_info_pointer) == XC_FAMILY_GGA .or.  &
            xc_f90_info_family(self%exchange_info_pointer) == XC_FAMILY_HYB_GGA .or. &
            xc_f90_info_family(self%correlation_info_pointer) == XC_FAMILY_GGA .or.  &
            xc_f90_info_family(self%correlation_info_pointer) == XC_FAMILY_HYB_GGA

        
        
        !> cube.
        exc_cube => energy_per_particle%get_cube()
        vxc_cube => potential%get_cube()
        exc_cube = 0.0d0
        vxc_cube = 0.0d0
        
        
        !call self%core_evaluator%evaluate_core_electron_density(occupied_orbitals, core_density)
        !call self%core_evaluator%init_core_functions(core_density, density)
        !call self%core_evaluator%collapse_core_functions(density, core_density)
        
        if (evaluate_gradients) then
            !call self%evaluate_electron_density_gradient(occupied_orbitals, derivative_x, derivative_y, derivative_z)
            call self%grid_points_evaluator%evaluate_gradients_as_Function3Ds &
                     (density, derivative_x, derivative_y, derivative_z)

            !call self%core_evaluator%evaluate_core_gradient_and_collapse(core_density, &
            !                                                derivative_x, derivative_y, derivative_z)
            
        end if
        evaluator => self%grid_points_evaluator 
    
        call self%evaluate_cube(evaluator, evaluate_gradients, density, &
                                density%grid, exc_cube, vxc_cube, potential_contracted_gradients_x%cube, &
                                potential_contracted_gradients_y%cube, potential_contracted_gradients_z%cube, &
                                derivative_x = derivative_x, derivative_y = derivative_y, derivative_z = derivative_z,&
                                energy_density = energy_density%cube, occupied_orbitals = occupied_orbitals &
                                )
                                
        nullify(evaluator)
        call self%evaluate_bubbles(evaluate_gradients, density, &
                                   energy_per_particle%bubbles, &
                                   potential%bubbles, potential_contracted_gradients_x%bubbles, &
                                   potential_contracted_gradients_y%bubbles, &
                                   potential_contracted_gradients_z%bubbles, &
                                   derivative_x = derivative_x, &
                                   derivative_y = derivative_y, &
                                   derivative_z = derivative_z, &
                                   energy_density = energy_density%bubbles, &
                                   only_diagonal_bubbles = .FALSE., &
                                   occupied_orbitals = occupied_orbitals &
                                   ) !, &
                                   !core_density = core_density)


        ! deduct the energy density and potential bubbles from corresponding cubes
        call energy_density%inject_bubbles_to_cube(energy_density%bubbles, factor = -1.0d0)

        ! deduct the energy density and potential bubbles from corresponding cubes
        call energy_per_particle%inject_bubbles_to_cube(energy_per_particle%bubbles, factor = -1.0d0)
        !energy_per_particle%cube = 0.0d0
        call potential%inject_bubbles_to_cube(potential%bubbles, factor = -1.0d0)
        !potential%cube = 0.0d0

        
         evaluator => self%grid_points_evaluator   

        if (evaluate_gradients) then

            ! inject the potential from contracted gradients bubbles to a cube
            call potential_contracted_gradients_x%inject_bubbles_to_cube &
               (potential_contracted_gradients_x%bubbles, factor = -1.0d0)

            call potential_contracted_gradients_y%inject_bubbles_to_cube &
               (potential_contracted_gradients_y%bubbles, factor = -1.0d0)

            call potential_contracted_gradients_z%inject_bubbles_to_cube &
               (potential_contracted_gradients_z%bubbles, factor = -1.0d0)
               
            
            
            
            !potential_contracted_gradients%taylor_series_bubbles = 0.0d0
            ! evaluate the term caused to the potential by the gradients and add it to the result
            call self%evaluate_divergence_potential_term(evaluator, divergence_f3d, &
                                                        potential_contracted_gradients_x, potential_contracted_gradients_y, &
                                                        potential_contracted_gradients_z, ignore_bubbles = .FALSE.)


        end if
        !call self%evaluate_cores(evaluate_gradients, core_density, &
        !                         energy_per_particle, potential, &
        !                         derivative_x, derivative_y, derivative_z, &
        !                         potential_contracted_gradients_x, potential_contracted_gradients_y, &
        !                         potential_contracted_gradients_z, &
        !                         divergence = divergence_f3d, &
        !                         energy_density = energy_density)
        !call destroy_core_functions(core_density)
                                
        if (evaluate_gradients) then
            call derivative_x%destroy()
            call derivative_y%destroy()
            call derivative_z%destroy()
            
            call potential_contracted_gradients_x%destroy()
            call potential_contracted_gradients_y%destroy()
            call potential_contracted_gradients_z%destroy()
        end if
                                 
        if (evaluate_gradients) then
            
            !call scale_cube(divergence_f3d%bubbles, divergence_f3d%cube, &
            !                divergence_f3d%grid)
            !call potential%print_out_centers()
            call potential%add_in_place(divergence_f3d)
            call potential%bubbles%extrapolate_origo(6, lmax = 0)

            call divergence_f3d%destroy()
            !call scale_cube(potential%bubbles, potential%cube, &
            !                potential%grid)
            !call scale_cube(energy_density%bubbles, energy_density%cube, &
            !               energy_density%grid)
        end if
        
        !call potential%inject_extra_bubbles(lmax = 6)
        
        call evaluator%destroy_stored_objects()
        nullify(evaluator)

        !do i = 1, energy_density%bubbles%get_nbub()
        !    energy_density%bubbles%bf(i)%p(1:6, :) = 0.0d0
        !    energy_density%bubbles%bf(i)%p(1:6, 2:) = 0.0d0
        !end do
        
        !call smoothen_bubbles(potential%bubbles)
        !call smoothen_bubbles(energy_density%bubbles)

        ! set the taylor bubbles of xc potential and energy_density to zero, as it only causes
        ! difficulties (and due to projection the taylor bubbles are not really neede) 
        !allocate(energy_density%bubbles_contaminants((energy_density%taylor_order+1)*(energy_density%taylor_order+2)*&
        !                          (energy_density%taylor_order+3)/6, energy_density%bubbles%get_nbub_global()), &
        !                           source = 0.0d0)
        !allocate(energy_density%cube_contaminants((energy_density%taylor_order+1)*(energy_density%taylor_order+2)*&
        !                          (energy_density%taylor_order+3)/6, energy_density%bubbles%get_nbub_global()), &
        !                           source = 0.0d0)
        !call energy_per_particle%precalculate_taylor_series_bubbles(non_overlapping = .TRUE.)
        !energy_density%taylor_series_bubbles = 0.0d0
        !allocate(potential%bubbles_contaminants((potential%taylor_order+1)*(potential%taylor_order+2)*&
        !                          (potential%taylor_order+3)/6, potential%bubbles%get_nbub_global()), &
        !                           source = 0.0d0)
        !allocate(potential%cube_contaminants((potential%taylor_order+1)*(potential%taylor_order+2)*&
        !                          (potential%taylor_order+3)/6, potential%bubbles%get_nbub_global()), &
        !                           source = 0.0d0)
        !call potential%precalculate_taylor_series_bubbles(non_overlapping = .TRUE.)
        !potential%taylor_series_bubbles = 0.0d0
        !potential%cube_contaminants = 0.0d0

        
        ! Calculate the total Exchange correlation energy, if it is needed
        if (present(energy)) then
#ifdef HAVE_CUDA    
            ! make sure that we are all finished before communicating the electron density between
            ! computational nodes
            call CUDASync_all()
#endif
            energy = energy_per_particle .dot. density
            !energy_density%cube = 0.0d0
            energy = energy_density%integrate()
        end if
        call energy_density%destroy()

        !call input_density%destroy()
        
    end subroutine XC_eval

    !> update xc.  
    !! restriction: xc have only s bubble. use the same grid as 
    !! the input density. cube equi-distant for each dimension.

    subroutine XC_eval_grid(self, density, op_w, op_n)
        class(XC) :: self
        type(function3d), intent(in) :: density
        ! options to calculate the integration weight
        integer(INT32), intent(in), optional :: op_w
        ! options for order of polynomials to calculate 
        ! the density gradient
        ! (n+1) point formula
        integer(INT32), intent(in), optional :: op_n


        integer(INT32) :: w_idx
        integer(INT32) :: n_idx
        ! dummy index
        integer(INT32) :: p, q, r, s, i, j, k, l, m, n
        real(real64), dimension(:), allocatable :: cell_density
        ! |\nabla \rho|^2
        real(real64), dimension(:), allocatable :: cell_density_gradient
        real(real64), dimension(:), pointer :: fr
        ! d\rho_r /dr
        real(real64), dimension(:), allocatable :: d_fr
        integer(int32) :: nlip
    
        real(real64), dimension(:), pointer :: e
        real(real64), dimension(:), pointer :: v
        real(real64), dimension(:), allocatable :: vrho
        real(real64), dimension(:), allocatable :: vsigma
        real(real64), dimension(:,:), allocatable :: w
        ! test the new weight <<<<<<<<<<<<<<<<<<<<<<<<<
        real(real64), dimension(:), allocatable :: w_tmp
        real(real64) :: step
        integer(int32) :: center
        real(real64), dimension(:), allocatable :: dr_ds
        real(real64), dimension(:), pointer :: coord
        real(real64), dimension(:), pointer :: coord_start
        real(real64), dimension(:,:), allocatable :: der_const
!       integer(int32), parameter :: nlip2 = 3
!       real(real64), dimension(nlip2,nlip2) :: der_const2
        real(real64), dimension(:,:), allocatable :: der_const2
        type(lipbasis) :: small_lip
        real(real64), dimension(:), allocatable :: d_fr2
        real(real64), dimension(:), allocatable :: vrho_c
        real(real64), dimension(:), allocatable :: vsigma_c
        real(real64), dimension(:), allocatable :: e_c
        real(real64) :: a, b, c, aa, bb, cc
        real(real64), dimension(:,:,:), allocatable :: t_vrho
        real(real64), dimension(:,:,:), allocatable :: t_vsigma
        real(real64), dimension(:,:,:), allocatable :: t_den
        real(real64), dimension(:,:,:), allocatable :: t_sigma
        real(real64), dimension(:,:,:), allocatable :: t_sigmax
        real(real64), dimension(:,:,:), allocatable :: t_sigmay
        real(real64), dimension(:,:,:), allocatable :: t_sigmaz
        real(real64), dimension(:,:,:), allocatable :: t_dsigma_drho
        real(real64), dimension(:,:,:,:), allocatable :: t_const
        type(grid1d) :: t_grid1d
        real(real64), dimension(:,:), allocatable :: t_r2int
        ! >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        real(real64), dimension(:), allocatable :: g_der_den
        real(real64), dimension(:,:), allocatable :: lip_dev

        ! interface to libxc
        TYPE(xc_f90_pointer_t) :: xc_func
        TYPE(xc_f90_pointer_t) :: xc_info
        ! add lda part
        ! TYPE(xc_f90_pointer_t) :: xc_func_lda
        ! TYPE(xc_f90_pointer_t) :: xc_info_lda
        ! for correlation
        TYPE(xc_f90_pointer_t) :: xc_func_c
        TYPE(xc_f90_pointer_t) :: xc_info_c

        ! cube
        real(real64), dimension(:,:,:), pointer :: e_3
        real(real64), dimension(:,:,:), pointer :: v_3
        real(real64), dimension(:), allocatable :: e_3_cell
        real(real64), dimension(:), allocatable :: v_3_cell
        ! coord of points in a cubic cell
        real(real64), dimension(:,:), allocatable :: points
        real(real64), dimension(:), pointer :: grid_x
        real(real64), dimension(:), pointer :: grid_y
        real(real64), dimension(:), pointer :: grid_z
        real(real64), dimension(:), allocatable :: den_3
        ! |\nabla \rho|^2, (\nabla f)_x, ...
        real(real64), dimension(:,:), allocatable :: den_3_d
        real(real64), dimension(:), allocatable :: den_3_d_norm
        real(real64), dimension(:), allocatable :: den_3_d_x
        real(real64), dimension(:), allocatable :: den_3_d_y
        real(real64), dimension(:), allocatable :: den_3_d_z
        real(real64), dimension(:), allocatable :: w_3
        ! matrix of \int chi(p) dp
        real(real64), dimension(:,:), allocatable :: w_3_m
        ! for loop over the cube axis
        ! for dg/d\rho
        ! index of grids on the axis
        integer(int32), dimension(:), allocatable :: axis_idx

        real(real64), dimension(:,:), allocatable :: lip_dev_x
        real(real64), dimension(:,:), allocatable :: lip_dev_y
        real(real64), dimension(:,:), allocatable :: lip_dev_z


        integer(int32) :: succed

        print *, '+++ evaluation(update) of xc (grid approach) being called ...'

        if(present(op_w)) then
            w_idx = op_w
        else
            w_idx = 1
        endif

        if(present(op_n)) then
            n_idx = op_n
        else
            n_idx = 3
        endif
        
        !> temporary.
        if (w_idx /= 1 .or. n_idx /= 3 ) print *, 'error! not implemented.'


        if (self%exchange_functional_type /= 0)    &
             call xc_f90_func_init(xc_func, xc_info, self%exchange_functional_type, XC_UNPOLARIZED)
        if(self%correlation_functional_type /= 0)  &
             call xc_f90_func_init(xc_func_c, xc_info, self%correlation_functional_type, XC_UNPOLARIZED)

        nlip = n_idx 

        !> derivative constant of lip basis
        small_lip=LIPBasisInit(nlip, 0)
        der_const2 = small_lip%der_const()
        !print *, 'der_const2 ', der_const2(2,:)

        !> update bubble.
        !> loop over each s bubble.

        loop_bubble: do i = 1,  self%energy_per_particle%bubbles%get_nbub()  
            print *, 'bubbles: ', i
            allocate(vrho(density%bubbles%gr(i)%p%get_shape()))
            allocate(vrho_c(density%bubbles%gr(i)%p%get_shape()))
            allocate(vsigma(density%bubbles%gr(i)%p%get_shape()))
            allocate(vsigma_c(density%bubbles%gr(i)%p%get_shape()))
            allocate(e_c(density%bubbles%gr(i)%p%get_shape()))
            allocate(w_tmp(density%bubbles%gr(i)%p%get_shape()))

            e => self%energy_per_particle%bubbles%get_f(i,0,0)
            v => self%potential%bubbles%get_f(i,0,0)
            e = 0.0d0
            v = 0.0d0
            coord=>density%bubbles%gr(i)%p%get_coord()
            dr_ds = density%bubbles%gr(i)%p%dr_ds()

            !> weight of numerical integration of Exc.
            w_tmp = dr_ds*coord**2

            !> another option for weight.
            !! different formula for potential.
            !w = density%bubbles%gr(1)%r2int()
            !> fetch input density info.
            fr => density%bubbles%get_f(i,0,0)
            d_fr = density%bubbles%radial_derivative(i)
            d_fr2 = d_fr**2
            call xc_f90_gga_exc_vxc(xc_func, size(fr), fr(1), &
                d_fr2(1), e(1), vrho(1), vsigma(1))
            call xc_f90_gga_exc_vxc(xc_func_c, size(fr), fr(1), &
              d_fr2(1), e_c(1), vrho_c(1), vsigma_c(1))
            e = e+e_c
            vrho=vrho+vrho_c
            vsigma=vsigma+vsigma_c
            !> for combined x and c functionals.
            if (self%exchange_functional_type == self%correlation_functional_type) then
                e = e/2.0d0
                vrho = vrho/2.0d0 
                vsigma = vsigma/2.0d0
            end if
            
            !> loop over each grid point.
            do j = 2,  size(fr)-1
                !> r2int as weight.
                !t_grid1d = grid1d(coord(j-1),1,3,[coord(j)-coord(j-1)])
                !t_r2int = t_grid1d%r2int()
                !print *, 'r2int, weight ', t_r2int(1,:)
                !> r2int as weight.
                !a = (t_r2int(1,1)/t_r2int(1,2))*vsigma(j-1)*2.0d0*d_fr(j-1)*der_const2(2,1)/dr_ds(j-1)
                aa = (w_tmp(j-1)/w_tmp(j))*vsigma(j-1)*2.0d0*d_fr(j-1)*der_const2(3,2)/dr_ds(j-1)
                b = vsigma(j)*2.0d0*d_fr(j)*der_const2(2,2)/dr_ds(j)
                !> r2int as weight.
                !c = (t_r2int(1,3)/t_r2int(1,2))*vsigma(j+1)*2.0d0*d_fr(j+1)*der_const2(2,3)/dr_ds(j+1)
                cc = (w_tmp(j+1)/w_tmp(j))*vsigma(j+1)*2.0d0*d_fr(j+1)*der_const2(1,2)/dr_ds(j+1)
                v(j) = aa+b+cc+vrho(j)
                !print *, 'v(j) ', coord(j),  v(j)
            end do
            !> special point. first grid point.
            t_grid1d = grid1d(coord(1),1,3,[coord(2)-coord(1)])
            t_r2int = t_grid1d%r2int()
            aa = vsigma(1)*2.0d0*d_fr(1)*der_const2(1,1)/dr_ds(1)
            b = (t_r2int(1,2)/t_r2int(1,1))*vsigma(2)*2.0d0*d_fr(2)*der_const2(1,2)/dr_ds(2)
            cc = (t_r2int(1,3)/t_r2int(1,1))*vsigma(3)*2.0d0*d_fr(3)*der_const2(1,3)/dr_ds(3)
            v(1) = aa + b + cc
        end do loop_bubble
        deallocate(vrho,vrho_c,vsigma,vsigma_c,e_c,w_tmp,d_fr,d_fr2)
        e => null()
        v => null()
        fr => null()
        
        !> cube 
        print *, 'start cube '
        e_3 => self%energy_per_particle%get_cube()
        v_3 => self%potential%get_cube()
        e_3 = 0.0d0
        v_3 = 0.0d0
        self%xc_e_int = self%energy_per_particle .dot. density 
        print *, 'xc energy (bubble only) ', self%xc_e_int
        if(density%bubbles%get_nbub()==1) print *, 'for one atom, the cube omitted.'
        return

        allocate(vrho(nlip**3))
        allocate(vrho_c(nlip**3))
        allocate(vsigma(nlip**3))
        allocate(vsigma_c(nlip**3))
        allocate(e_3_cell(nlip**3))
        allocate(points(3,nlip**3))
        allocate(t_const(nlip,nlip,nlip,nlip))
        allocate(t_dsigma_drho(nlip,nlip,nlip))
        grid_x => density%grid%axis(X_)%get_coord()
        grid_y => density%grid%axis(Y_)%get_coord()
        grid_z => density%grid%axis(Z_)%get_coord()
        step = grid_x(2)-grid_x(1)
        !> loop over grids (i,j,k)
        !! temporary. omit the terminal points
        do i = 2, size(grid_x) - 1
            do j = 2, size(grid_y) - 1
                 do k = 2, size(grid_z) - 1
                     !> coordinates of all points in the derivative cell. 
                     do l = 1, nlip
                         do m = 1, nlip
                             do n = 1, nlip
                                 ! first move along the X-axis 
                                 points(1,(l-1)*nlip**2+(m-1)*nlip+n)=&
                                 grid_x(i-2+n)
                                 ! then Y-axis 
                                 points(2,(l-1)*nlip**2+(m-1)*nlip+n)=&
                                 grid_y(j-2+m)
                                 points(3,(l-1)*nlip**2+(m-1)*nlip+n)=&
                                 grid_z(k-2+l)
                             end do
                         end do
                     end do
                     den_3=density%evaluate(points)
                     den_3_d=density%square_gradient_norm(points)
                     den_3_d_norm=den_3_d(1,:)
                     den_3_d_x=den_3_d(2,:)
                     den_3_d_y=den_3_d(3,:)
                     den_3_d_z=den_3_d(4,:)
                     vrho = 0.0d0
                     vrho_c = 0.0d0
                     vsigma = 0.0d0
                     vsigma_c = 0.0d0
                     call xc_f90_gga_exc_vxc(xc_func, size(den_3), den_3(1), &
                          den_3_d_norm(1), e_3_cell(1), vrho(1), vsigma(1))
                     call xc_f90_gga_exc_vxc(xc_func_c, size(den_3), den_3(1), &
                          den_3_d_norm(1), e_3_cell(1), vrho_c(1), vsigma_c(1))
                     vrho=vrho+vrho_c
                     vsigma=vsigma+vsigma_c
                     t_den =  reshape(den_3,(/nlip,nlip,nlip/))
                     t_vrho =  reshape(vrho,(/nlip,nlip,nlip/))
                     t_vsigma =  reshape(vsigma,(/nlip,nlip,nlip/))
                     t_sigma =  reshape(den_3_d_norm,(/nlip,nlip,nlip/))
                     t_sigmax =  reshape(den_3_d_x,(/nlip,nlip,nlip/))
                     t_sigmay =  reshape(den_3_d_y,(/nlip,nlip,nlip/))
                     t_sigmaz =  reshape(den_3_d_z,(/nlip,nlip,nlip/))
                     do l = 1, nlip
                        do m = 1, nlip
                           do n = 1, nlip
                               t_const(l,m,n,1) = der_const2(4-l,2)
                               t_const(l,m,n,2) = der_const2(4-m,2)
                               t_const(l,m,n,3) = der_const2(4-n,2)
                               t_dsigma_drho(l,m,n) = 2.0d0*(t_sigmax(l,m,n)*t_const(l,m,n,1)+&
                                                             t_sigmay(l,m,n)*t_const(l,m,n,2) + &
                                                             t_sigmaz(l,m,n)*t_const(l,m,n,3))/step
                               !a = t_vsigma(l,m,n)*t_dsigma_drho(l,m,n) 
                           end do
                        end do
                     end do
                     b = t_vsigma(1,2,2)*t_dsigma_drho(1,2,2) + t_vsigma(3,2,2)*t_dsigma_drho(3,2,2) + &
                         t_vsigma(2,1,2)*t_dsigma_drho(2,1,2) + t_vsigma(2,3,2)*t_dsigma_drho(2,3,2) + &
                         t_vsigma(2,2,1)*t_dsigma_drho(2,2,1) + t_vsigma(2,2,3)*t_dsigma_drho(2,2,3) 
                     v_3(i,j,k) = b + t_vrho(2,2,2)
                     !> the boundary points are overwritten by the next cubic cell.
                     e_3(i-(nlip-1)/2:i+(nlip-1)/2,&
                     j-(nlip-1)/2:j+(nlip-1)/2,&
                     k-(nlip-1)/2:k+(nlip-1)/2) = &
                     reshape(e_3_cell,(/nlip,nlip,nlip/))
                 end do
             end do
         end do
         !> cube as the remainder.
         e_3 = e_3-self%energy_per_particle%bubbles&
               %eval_3dgrid(self%energy_per_particle%grid)
         v_3 = v_3-self%potential%bubbles&
               %eval_3dgrid(self%potential%grid)

         deallocate(vrho,vsigma,e_3_cell,points,t_const)
         deallocate(den_3,den_3_d,den_3_d_norm,den_3_d_x,den_3_d_y,den_3_d_z)
         grid_x => null()
         grid_y => null()
         grid_z => null()
         !> Exc
         self%xc_e_int = self%energy_per_particle .dot. density 
         print *, 'end xc update'
    end subroutine XC_eval_grid
    
    subroutine XC_eval_taylor(self,density)  
        ! lda:
        ! # 1 cut the density.
        ! 2 eval bubble part of exc, vxc.
        ! 3 eval total exc, vxc, get cube. 
        class(xc), intent(inout) :: self
        type(function3d), intent(in) :: density
        type(function3d) :: density_mod
        integer, dimension(3) :: xc_cube_shape
        real(REAL64), dimension(:), allocatable :: tmp_exc, tmp_exc_x, tmp_exc_c, tmp_vxc, tmp_vxc_x, tmp_vxc_c 
        real(REAL64), pointer :: exc_cube(:,:,:), vxc_cube(:,:,:), den_cube(:,:,:)
        real(REAL64), pointer :: tmp_coord(:), tmp_den(:), tmp_exc_val(:, :), tmp_vxc_val(:, :)
        real(REAL64), pointer :: den_bubble(:,:)
        TYPE(xc_f90_pointer_t)                      :: xc_func_x
        TYPE(xc_f90_pointer_t)                      :: xc_info_x
        TYPE(xc_f90_pointer_t)                      :: xc_func_c
        TYPE(xc_f90_pointer_t)                      :: xc_info_c
        integer(INT32) :: p, l, i, j, k, m, n
        integer(INT32) :: bubble_size
        real(real64), dimension(:), allocatable :: den_cube_1d, exc_inject_1d, vxc_inject_1d, exc_cube_1d, vxc_cube_1d
        real(real64), dimension(:,:,:), allocatable :: exc_inject, vxc_inject, den_func3d(:,:,:)
        integer(INT32), dimension(1) :: radius
        real(real64), dimension(:,:), allocatable :: z_points

        ! analysis of density
        print *, '--- the density used to update the xc'
        print *, 'lmax: ', density%bubbles%get_lmax()
        print *, 'total e: ', density%integrate()
        print *, 'e in the bubbles: ', density%bubbles%integrate()
        density_mod = density
        do i = 1, density%bubbles%get_nbub()
            print *, i, 'th bubble', 'total e: ', density%bubbles%integrate(i)
            tmp_den => density_mod%bubbles%get_f(i,0,0)
            tmp_den = 0.0d0
            print *, 's part: ', density%bubbles%integrate(i)-density_mod%bubbles%integrate(i)
            tmp_den => null()
        end do
        print *, 'for atom. press density into the s bubble'
        density_mod = density
        den_bubble => density_mod%bubbles%get_f(1)
        den_bubble  = 0.0d0
        tmp_den => density_mod%bubbles%get_f(1,0,0)
        tmp_coord=>density%bubbles%gr(1)%p%get_coord()
        allocate(z_points(3,size(tmp_coord)))
        z_points(1,:) = 0.0d0
        z_points(2,:) = 0.0d0
        z_points(3,:) = tmp_coord
        tmp_den = density%evaluate(z_points)
        print *, 'e in the s bubble: ', density_mod%bubbles%integrate(1)
         
        den_bubble => null()
        tmp_coord=> null()
        tmp_den => null()
        deallocate(z_points)
        print *, '-- test density only. no update of xc.'
        return
        
        if (self%exchange_functional_type /= 0)    &
            call xc_f90_func_init(xc_func_x, xc_info_x, self%exchange_functional_type, XC_UNPOLARIZED)
        if (self%correlation_functional_type /= 0) &
            call xc_f90_func_init(xc_func_c, xc_info_c, self%correlation_functional_type, XC_UNPOLARIZED)

        bubble_size = self%energy_per_particle%bubbles%gr(1)%p%get_shape()
        allocate(tmp_exc(bubble_size))
        allocate(tmp_exc_x(bubble_size))
        allocate(tmp_exc_c(bubble_size))
        allocate(tmp_vxc(bubble_size))
        allocate(tmp_vxc_x(bubble_size))
        allocate(tmp_vxc_c(bubble_size))

        print *,  'lmax of density', density%bubbles%get_lmax()
        do i = 1,  self%energy_per_particle%bubbles%get_nbub()  
            tmp_exc_x = 0.0d0
            tmp_vxc_x = 0.0d0
            tmp_exc_c = 0.0d0
            tmp_vxc_c = 0.0d0

            tmp_coord=>self%energy_per_particle%bubbles%gr(i)%p%get_coord()
            ! just for h2 (1.4 bohr)
            radius = maxloc(1.0d0/(0.7d0-tmp_coord),mask=tmp_coord/=0.7d0)
            print *, 'number of points in the atom: ', radius
            print *, 'total number of points in the bubble (default): ', bubble_size 
            tmp_den => density%bubbles%get_f(i,0,0)
            print *, 'number of density in a bubble: ', density%bubbles%integrate(i)
            tmp_exc_val => self%energy_per_particle%bubbles%get_f(i)
            tmp_exc_val = 0.0d0
            tmp_vxc_val => self%potential%bubbles%get_f(i)
            tmp_vxc_val = 0.0d0
            call xc_f90_lda_exc_vxc(xc_func_x, radius(1), tmp_den(1), tmp_exc_x(1), tmp_vxc_x(1))
            call xc_f90_lda_exc_vxc(xc_func_c, radius(1), tmp_den(1), tmp_exc_c(1), tmp_vxc_c(1))
            tmp_exc = tmp_exc_x + tmp_exc_c  
            tmp_vxc = tmp_vxc_x + tmp_vxc_c
            tmp_exc_val(:, lm_map(0,0)) = tmp_exc
            tmp_vxc_val(:, lm_map(0,0)) = tmp_vxc
        end do
        deallocate(tmp_exc)
        deallocate(tmp_exc_x)
        deallocate(tmp_exc_c)
        deallocate(tmp_vxc)
        deallocate(tmp_vxc_x)
        deallocate(tmp_vxc_c)


        exc_cube => self%energy_per_particle%get_cube()
        vxc_cube => self%potential%get_cube()
        exc_cube = 0.0d0
        vxc_cube = 0.0d0
        call bigben%split("Dot product")
        self%xc_e_int = self%energy_per_particle .dot. density
        print *, 'Exc (bubble only) = ', self%xc_e_int
        call bigben%stop() 
        xc_cube_shape = self%energy_per_particle%grid%get_shape()
        m = xc_cube_shape(1)*xc_cube_shape(2)*xc_cube_shape(3)
        allocate(tmp_exc(m))
        allocate(tmp_exc_x(m))
        allocate(tmp_exc_c(m))
        allocate(tmp_vxc(m))
        allocate(tmp_vxc_x(m))
        allocate(tmp_vxc_c(m))
        den_cube => density%get_cube()
        den_func3d = den_cube + density%inject_bubbles()
        den_cube_1d=reshape(den_func3d,(/m/))
        
       call xc_f90_lda_exc_vxc(xc_func_x, m, den_cube_1d(1), tmp_exc_x(1), tmp_vxc_x(1))
       call xc_f90_lda_exc_vxc(xc_func_c, m, den_cube_1d(1), tmp_exc_c(1), tmp_vxc_c(1))
       tmp_exc = tmp_exc_x + tmp_exc_c  
       tmp_vxc = tmp_vxc_x + tmp_vxc_c

       exc_inject=self%energy_per_particle%inject_bubbles()
       vxc_inject=self%potential%inject_bubbles()
       exc_inject_1d=reshape(exc_inject,(/m/))
       vxc_inject_1d=reshape(vxc_inject,(/m/))
       exc_cube_1d=tmp_exc-exc_inject_1d
       vxc_cube_1d=tmp_vxc-vxc_inject_1d
       exc_cube=reshape(exc_cube_1d,(/xc_cube_shape(1),xc_cube_shape(2),xc_cube_shape(3)/))
       vxc_cube=reshape(vxc_cube_1d,(/xc_cube_shape(1),xc_cube_shape(2),xc_cube_shape(3)/))

       self%xc_e_int = self%energy_per_particle .dot. density 
       deallocate(tmp_exc)
       deallocate(tmp_exc_x)
       deallocate(tmp_exc_c)
       deallocate(tmp_vxc)
       deallocate(tmp_vxc_x)
       deallocate(tmp_vxc_c)
       deallocate(den_cube_1d, exc_inject_1d, vxc_inject_1d, exc_cube_1d, vxc_cube_1d)
       deallocate(exc_inject, vxc_inject)
       print *, 'end xc update'
    end subroutine XC_eval_taylor  


    !> lda exchange. energy density & potential.
    function lda_x_get_e(point, density) result(lda_x_e)
      real(REAL64), dimension(:,:), allocatable, intent(in) :: point
      type(Function3D), intent(in) :: density
      real(REAL64), dimension(:,:), allocatable :: lda_x_e
      real(REAL64), parameter :: lda_x_cst = -1*0.75d0*(3.0d0/PI)**(1.0d0/3.0d0)
      
      real(REAL64), dimension(:), allocatable :: tmp_den

      allocate(tmp_den(size(point,2)))
      allocate(lda_x_e(2,size(point,2)))
      tmp_den = density%evaluate(point)
      
      lda_x_e(1,:) = lda_x_cst * tmp_den**(1.0d0/3.0d0)
      lda_x_e(2,:) = -(3.0d0*tmp_den/pi)**(1.0d0/3.0d0)
      deallocate(tmp_den,lda_x_e)
    end function lda_x_get_e

    !> lda_x_get_e2 evaluates xc with only point density, 
    !! not the density function3d.
    !! calling is easier.
    function lda_x_get_e2(tmp_den) result(lda_x_e)
      real(REAL64), dimension(:), intent(in) :: tmp_den
      real(REAL64), dimension(:,:), allocatable :: lda_x_e
      real(REAL64), parameter :: lda_x_cst = -1*0.75d0*(3.0d0/PI)**(1.0d0/3.0d0)
      
      allocate(lda_x_e(2,size(tmp_den)))
      
      lda_x_e(1,:) = lda_x_cst * tmp_den**(1.0d0/3.0d0)
      lda_x_e(2,:) = -(3.0d0*tmp_den/pi)**(1.0d0/3.0d0)
      deallocate(lda_x_e)
    end function lda_x_get_e2

    !> lda correlation. pw tpye. 
    function lda_c_pw_get_e(point, density) result(lda_c_pw_e)
      real(REAL64), dimension(:,:), allocatable, intent(in) :: point
      type(Function3D), intent(in) :: density
      real(REAL64), dimension(:,:), allocatable :: lda_c_pw_e
      real(REAL64), dimension(:), allocatable :: tmp_den
      real(REAL64), dimension(:), allocatable :: wigner_seitz_radius, rs, q0, q1, q1d
      real(REAL64), parameter :: p = 1.00d0
      real(REAL64), parameter :: a = 0.031091d0
      real(REAL64), parameter :: alpha1 = 0.21370d0
      real(REAL64), parameter :: beta1 = 7.5957d0
      real(REAL64), parameter :: beta2 = 3.5876d0
      real(REAL64), parameter :: beta3 = 1.6382d0
      real(REAL64), parameter :: beta4 = 0.49294d0
    
      allocate(tmp_den(size(point,2)))
      allocate(wigner_seitz_radius(size(point,2)))
      allocate(rs(size(point,2)))
      allocate(q0(size(point,2)))
      allocate(q1(size(point,2)))
      allocate(q1d(size(point,2)))
      allocate(lda_c_pw_e(2,size(point,2)))
      tmp_den = density%evaluate(point)
      wigner_seitz_radius = (3.0d0/(4.0d0*pi*tmp_den))**(1.0d0/3.0d0)
      rs = wigner_seitz_radius 
      lda_c_pw_e(1,:) = -2.0d0*a*(1+alpha1*rs)* &
      log(1+1/(2.0d0*a*(beta1*sqrt(rs)+beta2*rs+beta3*rs**1.5d0+beta4*rs**(p+1))))
    
      rs = wigner_seitz_radius 
      q0 = -2.0d0*a*(1+alpha1*rs)
      q1 = 2.0d0*a*(beta1*sqrt(rs)+beta2*rs+beta3*rs**1.5d0+beta4*rs**(p+1))
      q1d = a*(beta1*rs**(-0.5d0) + 2.0d0*beta2 + 3.0d0*beta3*rs**0.5d0 + 2*(p+1)*beta4*rs**p)
      lda_c_pw_e(2,:) = lda_c_pw_e(1,:) - (rs/3.0d0)*(-2.0d0*a*alpha1*log(1+1.0d0/q1)-q0*q1d/(q1**2+q1))
      deallocate(tmp_den,wigner_seitz_radius,rs,q0,q1,q1d,lda_c_pw_e)
    end function lda_c_pw_get_e
    
    !> becke partition function.
    !! without atomic size adjustments.
    subroutine becke_partition(n, m, coord_N, coord, w)
        implicit none
        integer, intent(in) :: n, m !> number of nuclus = n, points = m
        real(REAL64), intent(in) :: coord_N(3,n) !> coordinates of the nuclea
        real(REAL64), intent(in) :: coord(3,m) !> coordinates of the points
        real(REAL64) :: w(n,m) !> the weight of point m referring to nuclear n

        !> \mu value refering to two nuclea. the diagonal elements are set 
        !> to zero, no meaning. 
        real(REAL64) :: mu(n,n,m)
        real(REAL64) :: f1(n,n,m)
        real(REAL64) :: f2(n,n,m)
        real(REAL64) :: f3(n,n,m)
        real(REAL64) :: f4(n,n,m)
        real(REAL64) :: s(n,n,m)
        real(REAL64) :: p(n,m)
        real(REAL64) :: r_en(n,m)
        real(REAL64) :: r_nn(n,n)

        integer :: i, j, k
        
        !> special case, atom
        if(n==1) then 
          w = 1.0d0
          return
        end if

        forall (i = 1 : m, j = 1 : n)
            r_en(j,i) = sqrt( (coord(1,i) - coord_N(1,j))**2 + & 
                              (coord(2,i) - coord_N(2,j))**2 + &
                              (coord(3,i) - coord_N(3,j))**2 )
        end forall
!       write(*,*) 'r_en = ', r_en

        forall (i = 1 : n, j = 1 : n)
            r_nn(i,j) = sqrt( (coord_N(1,i) - coord_N(1,j))**2 + & 
                              (coord_N(2,i) - coord_N(2,j))**2 + &
                              (coord_N(3,i) - coord_N(3,j))**2 )
        end forall
!       write(*,*) 'r_nn = ', r_nn

        forall (i = 1 : n, j = 1: n, k = 1 : m)
            mu(i,j,k) = (r_en(i,k) - r_en(j,k))/r_nn(i,j)
        end forall
!       write(*,*) 'mu = ', mu

        f1 = (3.0d0/2.0d0) * mu - (1.0d0/2.0d0) * mu**3
        f2 = (3.0d0/2.0d0) * f1 - (1.0d0/2.0d0) * f1**3
        f3 = (3.0d0/2.0d0) * f2 - (1.0d0/2.0d0) * f2**3
        !f4 = (3.0d0/2.0d0) * f3 - (1.0d0/2.0d0) * f3**3
        s = (1.0d0/2.0d0) * (1-f3)
!       write(*,*) 's = ', s
        p = 1.0d0
        do i = 1, n
            do j = 1, m
                do k = 1, i-1
                    p(i, j) = p(i, j) * s(i, k, j)
                end do
                do k = i+1, n
                    p(i, j) = p(i, j) * s(i, k, j)
                end do
            end do
        end do
!       write(*,*) 'p = ', p

        forall (i = 1 : n, j = 1 : m)
            w(i,j) = p(i,j)/sum(p(:,j)) 
        end forall
    end subroutine becke_partition

    subroutine becke_partition_atomic_size(n, m, coord_N, nu_charge, coord, w)
        !> atomic size adjustments
        !> ref: becke1988multicenter, doi: 10.1063/1.454033.
        implicit none
        integer, intent(in)       :: n, m !> number of nuclei = n, points = m
        real(REAL64), intent(in)  :: coord_N(3,n) !> coordinates of the nuclea
        real(REAL64), intent(in)  :: coord(3,m) !> coordinates of the points
        real(REAL64), intent(out) :: w(n,m) !> the weight of point m referring to nuclear n
        real(real64), intent(in)  :: nu_charge(n)

        !> \mu value refering to two nuclea. the diagonal elements are set 
        !> to zero, no meaning. 
        real(REAL64) :: mu(n,n,m)
        real(REAL64) :: nu(n,n,m)
        real(REAL64) :: f1(n,n,m)
        real(REAL64) :: f2(n,n,m)
        real(REAL64) :: f3(n,n,m)
        real(REAL64) :: f4(n,n,m)
        real(REAL64) :: s(n,n,m)
        real(REAL64) :: p(n,m)
        real(REAL64) :: r_en(n,m)
        real(REAL64) :: r_nn(n,n)
        real(REAL64) :: a_nn(n,n)
        !> bragg_slater radii. ref https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page).
        !! for first two rows only. note he and ne undefined.
        !! each column: nuclear charge, radius
        real(REAL64) :: bragg_slater(2,8)
        real(REAL64) :: r(n)
        real(REAL64) :: a(n,n)

        integer :: i, j, k
        
        !> special case, atom
        if(n==1) then 
          w = 1.0d0
          return
        end if
        
        bragg_slater(1,:)=[1.0d0,3.0d0,4.0d0,5.0d0,6.0d0,7.0d0,8.0d0,9.0d0]
        bragg_slater(2,:)=[0.35d0,1.45d0,1.05d0,0.85d0,0.70d0,0.65d0,0.60d0,0.50d0]

        r = 0.0d0
        do i = 1, n
            do j = 1, size(bragg_slater,2)
                if (nu_charge(i) == bragg_slater(1,j)) r(i) = bragg_slater(2,j)
            end do
        end do
        if (.not. all(r/=0.0d0))  then
            print *, 'error, bragg-slater radii unassigned'
            return
        end if
        
        !> the adjustment parameter
        a = 0.0d0
        do i = 1, n
            do j = 1, n
                if (i==j) then
                    ! arbitrary assigned. not used.
                    a(i,i) = 1.0d0
                else
                    a(i,j) = ((r(i)/r(j)-1)/(r(i)/r(j)+1))/(((r(i)/r(j)-1)/(r(i)/r(j)+1))**2-1)
                    if (a(i,j)-0.5d0>1.0d-6) a(i,j) = 0.5d0
                    if (a(i,j)+0.5d0<1.0d-6) a(i,j) = -0.5d0
                end if
            end do
        end do

        ! calculate the distances between the nuclei and the evaluated points
        forall (i = 1 : m, j = 1 : n)
            r_en(j,i) = sqrt( (coord(1,i) - coord_N(1,j))**2 + & 
                              (coord(2,i) - coord_N(2,j))**2 + &
                              (coord(3,i) - coord_N(3,j))**2 )
        end forall

        ! calculate the distances between nuclei
        forall (i = 1 : n, j = 1 : n)
            r_nn(i,j) = sqrt( (coord_N(1,i) - coord_N(1,j))**2 + & 
                              (coord_N(2,i) - coord_N(2,j))**2 + &
                              (coord_N(3,i) - coord_N(3,j))**2 )
        end forall

        forall (i = 1 : n, j = 1: n, k = 1 : m)
            mu(i,j,k) = (r_en(i,k) - r_en(j,k))/r_nn(i,j)
            nu(i,j,k) = mu(i,j,k) + a(i,j)*(1-mu(i,j,k)**2)
        end forall

        f1 = (3.0d0/2.0d0) * nu - (1.0d0/2.0d0) * nu**3
        f2 = (3.0d0/2.0d0) * f1 - (1.0d0/2.0d0) * f1**3
        f3 = (3.0d0/2.0d0) * f2 - (1.0d0/2.0d0) * f2**3
        !f4 = (3.0d0/2.0d0) * f3 - (1.0d0/2.0d0) * f3**3
        s = (1.0d0/2.0d0) * (1-f3)

        p = 1.0d0
        do i = 1, n
            do j = 1, m
                do k = 1, i-1
                    p(i, j) = p(i, j) * s(i, k, j)
                end do
                do k = i+1, n
                    p(i, j) = p(i, j) * s(i, k, j)
                end do
            end do
        end do

        forall (i = 1 : n, j = 1 : m)
            w(i,j) = p(i,j)/sum(p(:,j)) 
        end forall
    end subroutine becke_partition_atomic_size

end module xc_class

