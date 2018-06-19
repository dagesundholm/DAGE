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
program multipole_translation_test
    use RealSphericalHarmonics_class
    use HelmholtzMultipoleTools_class
    use Helmholtz3D_class
    use Globals_m
    use Core_class
    use Function3D_class
    use GBFMMParallelInfo_class
    use ParallelInfo_class
    use GBFMMHelmholtz3D_class
    use GaussQuad_class
    use SCFCycle_class
  
    implicit none

    real(REAL64) :: position_set1(3, 1)
    real(REAL64) :: position_set2(3, 1)
    real(REAL64) :: position_set3(3, 1)
    real(REAL64) :: position_set4(3, 1)
    real(REAL64) :: position_set5(3, 1)
    real(REAL64) :: position_set6(3, 6)
    real(REAL64) :: position_set7(3, 6)
    real(REAL64) :: position_set8(3, 6)
    real(REAL64) :: position_set9(3, 8)
    real(REAL64) :: position_set10(3, 1)
    real(REAL64) :: position_set11(3, 1)
    real(REAL64) :: position_set12(3, 1)
    real(REAL64) :: position_set13(3, 1)
    real(REAL64) :: position_set14(3, 1)
    real(REAL64) :: position_set15(3, 1)
    real(REAL64) :: position_set16(3, 1)
    real(REAL64) :: position_set17(3, 1)
    real(REAL64) :: multipole_centers(3, 19)
    integer      :: i, lmax, input_lmax, normalization
    real(REAL64) :: evaluation_point(3), multipole_center(3), evaluation_center(3), kappa
    logical      :: passed
    type(Core)         :: core_object
    type(Function3D), allocatable, target :: mos(:)
    class(ParallelInfo), pointer  :: parallelization_info
    character(100)                :: program_name = ""

    ! position set 1 includes one position at origo
    position_set1(:, 1) = [0.0d0, 0.0d0, 0.0d0]

    ! position set 2 includes one position along z axis at distance 2
    position_set2(:, 1) = [0.3d0, 1.5d0, 1.5d0]

    ! position set 3 includes one position along z axis at distance -0.2
    position_set3(:, 1) = [0.0d0, 0.0d0, 1.0d0]

    ! position set 3 includes one position along z axis at distance -0.2
    position_set4(:, 1) = [0.0d0, 0.0d0, -2.0d0]

    ! position set 3 includes one position along z axis at distance 3.0
    position_set5(:, 1) = [3.0d0, 0.0d0, 0.0d0]

    ! position set 4 includes mixed positions withing 1.0 radius of origin
    position_set6(:, 1) = [0.4d0, 0.6d0, 0.1d0]
    position_set6(:, 2) = [0.7d0, 0.3d0, 0.4d0]
    position_set6(:, 3) = [-0.2d0, -0.2d0, -0.1d0]
    position_set6(:, 4) = [0.0d0, -0.2d0, 0.8d0]
    position_set6(:, 5) = [0.1d0, -0.5d0, 0.4d0]
    position_set6(:, 6) = [0.3d0, -0.5d0, -0.1d0]

    ! position set 4 includes mixed positions withing 3.0 radius of origin
    position_set7(:, 1) = [0.4d0, 2.0d0, 0.1d0]
    position_set7(:, 2) = [0.7d0, -2.8d0, 1.4d0]
    position_set7(:, 3) = [-0.2d0, -0.2d0, -2.8d0]
    position_set7(:, 4) = [0.2d0, 2.8d0, 0.2d0]
    position_set7(:, 5) = [2.8d0, -0.2d0, 0.2d0]
    position_set7(:, 6) = [-0.2d0, 0.2d0, 2.8d0]

    ! position set 4 includes mixed positions withing 3.0 radius of origin
    position_set8(:, 1) = [0.4d0, 3.8d0, 0.1d0]
    position_set8(:, 2) = [0.7d0, 0.3d0, -3.8d0]
    position_set8(:, 3) = [3.8d0, -0.2d0, -0.1d0]
    position_set8(:, 4) = [0.2d0, -3.8d0, 0.2d0]
    position_set8(:, 5) = [-3.8d0, -0.2d0, 0.2d0]
    position_set8(:, 6) = [-0.2d0, 0.2d0, 3.8d0]

    ! corner set, scale with the box size
    position_set9(:, 1) = [1.0d0, 1.0d0, 1.0d0]
    position_set9(:, 2) = [1.0d0, 1.0d0, -1.0d0]
    position_set9(:, 3) = [1.0d0, -1.0d0, 1.0d0]
    position_set9(:, 4) = [1.0d0, -1.0d0, -1.0d0]
    position_set9(:, 5) = [-1.0d0, 1.0d0, 1.0d0]
    position_set9(:, 6) = [-1.0d0, 1.0d0, -1.0d0]
    position_set9(:, 7) = [-1.0d0, -1.0d0, 1.0d0]
    position_set9(:, 8) = [-1.0d0, -1.0d0, -1.0d0]

    position_set10(:, 1) = [1.0d0, 1.0d0, 1.0d0]
    position_set11(:, 1) = [1.0d0, 1.0d0, -1.0d0]
    position_set12(:, 1) = [1.0d0, -1.0d0, 1.0d0]
    position_set13(:, 1) = [1.0d0, -1.0d0, -1.0d0]
    position_set14(:, 1) = [-1.0d0, 1.0d0, 1.0d0]
    position_set15(:, 1) = [-1.0d0, 1.0d0, -1.0d0]
    position_set16(:, 1) = [-1.0d0, -1.0d0, 1.0d0]
    position_set17(:, 1) = [-1.0d0, -1.0d0, -1.0d0]


    ! multipole_centers
    multipole_centers(:, 1) = [0.0d0, 0.0d0, 2.0d0]
    multipole_centers(:, 2) = [0.0d0, 0.0d0, 0.1d0]
    multipole_centers(:, 3) = [0.0d0, 2.0d0, 0.0d0]
    multipole_centers(:, 4) = [0.0d0, 0.1d0, 0.1d0]
    multipole_centers(:, 5) = [2.0d0, 0.0d0, 0.0d0]
    multipole_centers(:, 6) = [0.1d0, 0.0d0, 0.1d0]
    multipole_centers(:, 7) = [1.0d0, 0.2d0, 4.2d0]
    multipole_centers(:, 8) = [5.0d0, 0.4d0, 4.0d0]
    multipole_centers(:, 9) = [0.0d0, 9.0d0, 0.4d0]
    multipole_centers(:, 10)= [0.0d0, 0.0d0, 7.0d0]
    multipole_centers(:, 11)= [0.0d0, 8.0d0, 0.0d0]
    multipole_centers(:, 12)= [6.0d0, 0.0d0, 0.0d0]
    multipole_centers(:, 13)= [0.0d0, 0.0d0, 15.0d0]
    multipole_centers(:, 14)= [0.0d0, 13.0d0, 0.0d0]
    multipole_centers(:, 15)= [8.0d0, 7.0d0, 0.0d0]
    multipole_centers(:, 16)= [3.8d0, 3.3d0, 2.6d0]
    multipole_centers(:, 17)= [0.0d0, 0.0d0, 3.0d0]
    multipole_centers(:, 18)= [0.0d0, 0.0d0, 8.0d0]
    multipole_centers(:, 19)= [1.0d0, 1.0d0, 1.0d0]
 

    evaluation_point = [2.0d0, 2.0d0, 2.0d0]
    evaluation_center = [0.0d0, 0.0d0, 0.0d0]
    lmax = 2
    input_lmax = 25
    normalization = 2
    multipole_center = [3.5d0, 8.0d0, 0.0d0]
    kappa = sqrt(-2.0d0 * (-0.50d0))
    passed = potential_test(lmax, kappa, 20, 1.0d0)
    passed = first_bessel_test(.TRUE., lmax, kappa) 
    !passed = second_bessel_test(.TRUE., lmax, kappa) 

    program_name = "multipole_translation_test"
    !core_object = read_command_line_input(program_name)
    !mos = core_object%get_input_orbitals()

    !call ml_test_set(lmax, input_lmax, normalization)

!     if (settings(1)%gbfmm) then
!         parallelization_info => mos(1)%parallelization_info
!         select type(parallelization_info)
!             type is (GBFMMParallelInfo)
!                 !call ml_boxed_test(parallelization_info, lmax, input_lmax, normalization, 1, position_set9, position_set9, 64)
!         end select
!         call f3d_ml_test(mos, settings(1), lmax, input_lmax, normalization, 1, position_set9, 1)   
!     else
!         !passed = second_bessel_test(.TRUE., lmax)   
!         call real_multipole_to_local_conversion_test(position_set6, multipole_centers(:, 7), &
!                                                    [0.0d0, 0.0d0, 0.0d0], lmax, input_lmax, 2)
!         passed = real_potential_ml_test(2.2*position_set10,  &
!                             multipole_centers(:, 15), evaluation_center, 2.2 *position_set13, lmax, input_lmax, normalization, 2)
!     end if

    !passed = real_potential_ml_test(position_set8,  &
    !                        multipole_centers(:, 15), evaluation_center, evaluation_point, lmax, input_lmax, normalization, 2)
    !call real_multipole_to_local_conversion_test(position_set6, multipole_centers(:, 7), [0.0d0, 0.0d0, 0.0d0], lmax, input_lmax, 2)
    !passed = real_potential_ml_test(position_set7, multipole_centers(:, 9), [0.0d0, 0.0d0, 0.0d0], [1.5d0, 1.2d0, 2.0d0], lmax, 2)
    !call real_multipole_translation_test(position_set7, multipole_centers(:, 7), [0.0d0, 0.0d0, 0.0d0], lmax, input_lmax, 2)
    !call real_potential_ml_test(position_set7, multipole_centers(:, 12), [0.0d0, 0.0d0, 0.0d0], lmax, 2)
    !call complex_multipole_translation_test(4)
contains

    subroutine f3d_ml_test(mos, settings, lmax, input_lmax, normalization, ibox, &
                           evaluation_points, output_mode)
        type(Function3D), allocatable, intent(in)  :: mos(:)
        type(ProgramSettings),   intent(in) :: settings
        integer,                 intent(in) :: lmax, input_lmax, normalization, ibox
        real(REAL64),            intent(in) :: evaluation_points(:, :)
        integer,                 intent(in) :: output_mode
        type(GBFMMHelmholtz3D)              :: helmholtz_operator
        type(GaussQuad)                     :: quadrature
        type(RHFCycle)                      :: rhf_cycle
        type(Function3D),       allocatable :: potentials(:)
        real(REAL64)                        :: multipole_moments((input_lmax+1)**2, 1), &
                                               local_expansion((lmax+1)**2, 1), &
                                               converted_local_expansion((lmax+1)**2, 1)
        type(HelmholtzMultipoleConverter)   :: translator
        real(REAL64)                        :: box_center_farfield(3), box_center(3), normalization_factor, &
                                               scaling_factor, farfield_centers(3, 1), box_size(3),  &
                                               potential, comparison_potential, comparison_potential2, &
                                               evaluation_distance, translation_distance, comp
        integer                             :: i, j, m, l, other_box, jbox, box_limits(2, 3), box_limits_farfield(2, 3)
        real(REAL64)                        :: evaluation_positions( &
                                                   size(evaluation_points, 1), size(evaluation_points, 2)), &
                                               fbessel_values(1, 0:lmax), spherical_harmonics(1, (lmax+1)**2), &
                                               kappa, sbessel_values(1, 0:lmax), spherical_harmonics2(1, (lmax+1)**2)
        type(FirstModSphBesselCollection)   :: first_bessels
        type(SecondModSphBesselCollection)  :: second_bessels
        type(RealSphericalHarmonics)        :: harmo
        integer, allocatable                :: farfield_indices(:)

        scaling_factor = 0.2d0

        
       
        

        
        !rhf_cycle = RHFCycle(mos, size(mos),  &
        !quadrature = GaussQuad(NLIN=20, NLOG=16), use_gbfmm = settings%gbfmm)
        !call rhf_cycle%run(update_orbitals = .FALSE.)
        !potentials = rhf_cycle%get_orbital_potentials()
        !helmholtz_operator = GBFMMHelmholtz3D( rhf_cycle%coulomb_operator, &
        !                rhf_cycle%eigen_values(1), quadrature = rhf_cycle%quadrature, &
        !                farfield_potential_lmax = lmax, farfield_potential_input_lmax = input_lmax, &
        !                normalization = normalization)

        kappa = sqrt(-2.0d0*rhf_cycle%eigen_values(1))
        
        ! calculate multipole moments
        ! evaluate box center and cell limits
        box_center = helmholtz_operator%parallel_info%get_box_center(ibox, helmholtz_operator%parallel_info%maxlevel)
        box_limits = helmholtz_operator%parallel_info%get_box_cell_index_limits(ibox,  &
                        helmholtz_operator%parallel_info%maxlevel, global = .FALSE.)

        farfield_indices = helmholtz_operator%parallel_info%get_local_farfield_indices &
                               (ibox, helmholtz_operator%parallel_info%maxlevel)

        
        first_bessels = FirstModSphBesselCollection(lmax, scaling_factor = scaling_factor)
        second_bessels = SecondModSphBesselCollection(lmax, scaling_factor = scaling_factor)

        do jbox = 1, size(farfield_indices)
            other_box = farfield_indices(jbox)
            box_limits_farfield = helmholtz_operator%parallel_info%get_box_cell_index_limits(other_box,  &
                            helmholtz_operator%parallel_info%maxlevel, global = .FALSE.)
            
            box_center_farfield = helmholtz_operator%parallel_info%get_box_center(other_box, &
                                      helmholtz_operator%parallel_info%maxlevel)

            farfield_centers(:, 1) = box_center_farfield

            harmo = RealSphericalHarmonics(input_lmax, normalization = normalization)
    
            ! evaluate Helmholtz type cube multipole moments for the box AT box center
            !multipole_moments(:, 1) = helmholtz_operator%get_cube_multipoles(potentials(1), harmo, &
            !                                                box_center_farfield, jbox, box_limits_farfield)
            ! evaluate Helmholtz type local expansion relative to box center
            local_expansion(:, 1) = helmholtz_operator%get_cube_local_expansion(potentials(1), harmo, &
                                                            box_center, box_limits_farfield)

            call harmo%destroy()

            ! get the evaluation positions
            box_size = (helmholtz_operator%parallel_info%ranges(2, :) - helmholtz_operator%parallel_info%ranges(1, :)) &
                        / (helmholtz_operator%parallel_info%get_grid_limit(helmholtz_operator%parallel_info%maxlevel) + 0.0d0)
            do j = 1, size(evaluation_points, 2)
                evaluation_positions(:, j) = evaluation_points(:, j) * box_size/2.0d0 * 1.00d0
            end do

            
            comp = helmholtz_operator%get_cube_comparison_potential(potentials(1), box_limits_farfield, &
                        box_center+ evaluation_positions(:, 1))


            ! translate 'original_multipoles' multipole expansion from 'multipole_center'
            ! to 'evaluation_center'.
            translator = HelmholtzMultipoleConverter(lmax, kappa, input_lmax, normalization = normalization, &
                                                    scaling_factor = scaling_factor)

            converted_local_expansion = translator%translate(multipole_moments, farfield_centers, box_center)
            call translator%destroy()

            ! initialize objects used to evaluate spherical harmonics and bessel functions
            harmo = RealSphericalHarmonics(lmax, normalization = normalization)
            

            do i = 1, size(evaluation_points, 2) 
                evaluation_distance = sqrt(sum(evaluation_positions(:, i)**2))
                translation_distance = sqrt(sum((evaluation_positions(:, i) + box_center - box_center_farfield)**2))

                ! evaluate second modified spherical bessel functions and spherical harmonics at evaluation point
                fbessel_values = first_bessels%eval(kappa * [evaluation_distance])
                sbessel_values = second_bessels%eval(kappa * [translation_distance])
                spherical_harmonics = harmo%eval(reshape(evaluation_positions(:, i), [3, 1]))
                spherical_harmonics2 = harmo%eval(reshape(evaluation_positions(:, i) + box_center - box_center_farfield, [3, 1]))

                ! set the result values to 0
                potential = 0.0d0
                comparison_potential = 0.0d0
                comparison_potential2 = 0.0d0
                normalization_factor = 1.0d0
                do l = 0, lmax
                    if (normalization /= 2) then
                        normalization_factor = (2*l+1)/dble(4*pi)
                    end if
                    do  m = -l, l
                        ! Calculate the translated potential via using the original evaluation point and corresponding
                        ! spherical harmonics + the translated multipoles
                        potential = potential +  fbessel_values(1, l) * normalization_factor  &
                                                * spherical_harmonics(1, lm_map(l, m)) &
                                                * converted_local_expansion(lm_map(l, m), 1)

                        ! Calculate the translated potential using the original evaluation point and corresponding
                        ! spherical harmonics + the explicitly evaluated multipoles
                        comparison_potential = comparison_potential +   normalization_factor * fbessel_values(1, l) &
                                                * spherical_harmonics(1, lm_map(l, m)) &
                                                * local_expansion(lm_map(l, m), 1)

                        comparison_potential2 = comparison_potential2 + normalization_factor * sbessel_values(1, l) &
                                                * spherical_harmonics2(1, lm_map(l, m)) &
                                                * multipole_moments(lm_map(l, m), 1)

                        
                        if (output_mode >= 2) then
                            print *, i, l, m,  8*kappa*potential, 8*kappa*comparison_potential, 8*kappa*comparison_potential2, & !, &
                                !converted_local_expansion(lm_map(l, m), 1), local_expansion(lm_map(l, m), 1), &
                                converted_local_expansion(lm_map(l, m), 1) / local_expansion(lm_map(l, m), 1)
                                !multipole_moments(lm_map(l, m), 1)
                        end if
                    end do
                end do
                if (output_mode >= 1) then
                    write(*, '("farfield_box:", i4, ", evaluation point:", i4, ", value: ", e14.6, & 
                            &"rel. error local: ", e14.6, ", rel. error mult:", e14.6, "")') &
                        other_box, i, 8 * kappa * potential, &
                        (8*kappa*potential - 8*kappa*comparison_potential) / ( 8*kappa*comparison_potential) , &
                        (8*kappa*potential - 8*kappa*comparison_potential2) / ( 8*kappa*comparison_potential2)
                end if
                
            end do
            
            ! destroy all stuff
            call harmo%destroy()
        end do

        
        do j = 1, size(potentials)
            call potentials(j)%destroy()
        end do
            
        call first_bessels%destroy()
        call second_bessels%destroy()
        deallocate(potentials)
        deallocate(farfield_indices)
                   
    end subroutine

    subroutine ml_boxed_test(parallel_info, lmax, input_lmax, normalization, ibox, evaluation_points, charge_points, other_box)
        type(GBFMMParallelInfo), intent(in) :: parallel_info
        integer,                 intent(in) :: lmax, input_lmax, normalization, ibox
        real(REAL64),            intent(in) :: evaluation_points(:, :), charge_points(:, :)
        integer,    optional,    intent(in) :: other_box

        integer, allocatable                :: farfield_indices(:)
        real(REAL64)                        :: box_center_farfield(3), box_center(3)
        logical                             :: passed
        integer                             :: j, jbox
        real(REAL64)                        :: evaluation_positions( &
                                                   size(evaluation_points, 1), size(evaluation_points, 2)), &
                                               charge_positions(     &
                                                   size(charge_points, 1), size(charge_points, 2)), box_size(3)

        box_size = (parallel_info%ranges(2, :) - parallel_info%ranges(1, :)) &
                    / (parallel_info%get_grid_limit(parallel_info%maxlevel) + 0.0d0)
        do j = 1, size(evaluation_points, 2)
            evaluation_positions(:, j) = evaluation_points(:, j) * box_size/2.0d0 * 1.00d0
        end do

        do j = 1, size(charge_points, 2)
            charge_positions(:, j) = charge_points(:, j) * box_size/2.0d0 * 1.00d0
        end do
        
        farfield_indices = parallel_info%get_local_farfield_indices(ibox, parallel_info%maxlevel)
        box_center = parallel_info%get_box_center(ibox, parallel_info%maxlevel)
        print *, box_center
   
        if (present(other_box)) then
            ! get the center of the farfield box
            box_center_farfield = parallel_info%get_box_center(other_box, parallel_info%maxlevel)
            print *, "--------------------------------------------"
            print *, "Box number", other_box, box_center_farfield-box_center, box_center_farfield
            print *, "--------------------------------------------"
            passed = real_potential_ml_test(charge_positions, box_center_farfield, &
                            box_center, evaluation_positions, lmax, input_lmax, normalization, 2)
            call real_multipole_to_local_conversion_test(charge_positions, box_center_farfield-box_center, &
                                                   [0.0d0, 0.0d0, 0.0d0], lmax, input_lmax, normalization)
        else

            do j = 1, size(farfield_indices)
                jbox = farfield_indices(j)
                ! get the center of the farfield box
                box_center_farfield = parallel_info%get_box_center(jbox, parallel_info%maxlevel)
                print *, "--------------------------------------------"
                print *, "Box number", jbox, box_center_farfield-box_center, box_center_farfield
                print *, "--------------------------------------------"
                passed = real_potential_ml_test(charge_positions, box_center_farfield, &
                                box_center, evaluation_positions, lmax, input_lmax, normalization, 1)
            end do
        end if
    end subroutine 

    subroutine ml_test_set(lmax, input_lmax, normalization)
        integer, intent(in)           :: lmax, input_lmax, normalization
        integer                       :: i
        

        write (*, '("Running multipole translation test with lmax=",i3, &
               & ", input lmax=",i3)') &
               lmax, input_lmax

        !call real_multipole_translation_test(position_set1, multipole_centers(:, 8), [0.0d0, 0.0d0, 0.0d0], 14, 2)
        do i = 1, size(multipole_centers, 2)
            print *, "-----------------------" 
            write (*, '("Translation ",i4, ": From [",f6.2,",",f6.2,",", f6.2, "] to [",f6.2,",",f6.2,",",f6.2, "], r=",f6.2)')  &
                i, evaluation_center(1), evaluation_center(2), evaluation_center(3),  &
                   multipole_centers(1, i), multipole_centers(2, i), multipole_centers(3, i), &
                   sqrt(sum((multipole_centers(:, i)-evaluation_center)**2)) 
            passed = real_potential_ml_test(position_set8, multipole_centers(:, i), &
                            evaluation_center, position_set9 , lmax, input_lmax, normalization, 1)
            if (.not. passed) then
                write (*, '("---Translation test failed. Trying with more tightly packed charge cluster.")')
                passed = real_potential_ml_test(position_set7,  &
                        multipole_centers(:, i), evaluation_center, position_set9, lmax, input_lmax, normalization, 1)

                if (.not. passed) then
                    write (*, '("---Translation test failed even with the more tighly packed charge cluster. &
                            &Trying the same charge at the origo.")')
                    passed = real_potential_ml_test(position_set1, multipole_centers(:, i), evaluation_center, &
                        position_set9, lmax, input_lmax, normalization, 1)
    
                    if (.not. passed) then
                        write (*, '("---Translation test did not pass even with the single charge cluster. &
                            &at the origo. This indicates major problems that could lie with the Bessel& 
                            &functions: The accuracy is not good enough. Other possibility could be that &
                            &the maximum angular momentum value is not large enough.")')
                    else
                        write (*, '("---Translation test passed with the single charge cluster. &
                            &This indicates that the problem could be with rotations of the clusters in the&
                            &translation. However, the Bessel function accuracy can not be ruled out. However,&
                            &it is not very likely. Also it could be that &
                            &the maximum angular momentum value is not large enough.")')
                    end if 
                else 
                    write (*, '("---Translation test passed with the more tighly packed charge cluster. &
                            &This indicates a problem with the Bessel functions: The accuracy is not &
                            &good enough for the translation distance. Other possibility could be that &
                            &the maximum angular momentum value is not large enough.")')
                end if 
            end if
        end do

    end subroutine

    subroutine mm_test_set(evaluation_point, lmax, input_lmax)
        integer, intent(in)           :: lmax, input_lmax
        !> The point where the potential is evaluated
        real(REAL64), intent(in)        :: evaluation_point(3)

        write (*, '("Running multipole translation test with lmax=",i3, &
               & ", input lmax=",i3,". The evaluation point is  [",f6.2,",", f6.2,",", f6.2,"]")') &
               lmax, input_lmax, evaluation_point(1), evaluation_point(2), evaluation_point(3)

        !call real_multipole_translation_test(position_set1, multipole_centers(:, 8), [0.0d0, 0.0d0, 0.0d0], 14, 2)
        do i = 1, size(multipole_centers, 2)
            print *, "-----------------------" 
            write (*, '("Translation ",i4, ": [",f6.2,",", f6.2,",", f6.2, "], r=",f6.2)')  &
        
                i, multipole_centers(1, i), multipole_centers(2, i), multipole_centers(3, i), sqrt(sum(multipole_centers(:, i)**2)) 
            passed = real_potential_mm_test(position_set7, multipole_centers(:, i), &
                            [0.0d0, 0.0d0, 0.0d0], evaluation_point , lmax, input_lmax, 2, 1)
            if (.not. passed) then
                write (*, '("---Translation test failed. Trying with more tightly packed charge cluster.")')
                passed = real_potential_mm_test(position_set6, multipole_centers(:, i), &
                        [0.0d0, 0.0d0, 0.0d0], evaluation_point, lmax, input_lmax, 2, 1)

                if (.not. passed) then
                    write (*, '("---Translation test failed even with the more tighly packed charge cluster. &
                            &Trying the same charge at the origo.")')
                    passed = real_potential_mm_test(position_set1, multipole_centers(:, i), [0.0d0, 0.0d0, 0.0d0], &
                        evaluation_point, lmax, input_lmax, 2, 1)
    
                    if (.not. passed) then
                        write (*, '("---Translation test did not pass even with the single charge cluster. &
                            &at the origo. This indicates major problems that could lie with the Bessel& 
                            &functions: The accuracy is not good enough. Other possibility could be that &
                            &the maximum angular momentum value is not large enough.")')
                    else
                        write (*, '("---Translation test passed with the single charge cluster. &
                            &This indicates that the problem could be with rotations of the clusters in the&
                            &translation. However, the Bessel function accuracy can not be ruled out. However,&
                            &it is not very likely. Also it could be that &
                            &the maximum angular momentum value is not large enough.")')
                    end if 
                else 
                    write (*, '("---Translation test passed with the more tighly packed charge cluster. &
                            &This indicates a problem with the Bessel functions: The accuracy is not &
                            &good enough for the translation distance. Other possibility could be that &
                            &the maximum angular momentum value is not large enough.")')
                end if 
            end if
        end do

    end subroutine

    function potential_test(lmax, kappa, number_of_points, max_r) result(passed)
        integer,      intent(in)      :: lmax, number_of_points
        real(REAL64), intent(in)      :: kappa, max_r
        real(REAL64)                  :: fbessel_values(number_of_points, 0:lmax), &
                                         sbessel_values(number_of_points, 0:lmax), &
                                         res, max_error, error, max_error_distance, &
                                         max_error_comparison, max_error_value, &
                                         r(number_of_points)
        integer                       :: i, j, l
        logical                       :: passed
        type(SecondModSphBesselCollection)  :: second_bessels
        type(FirstModSphBesselCollection)   :: first_bessels
        second_bessels = SecondModSphBesselCollection(lmax)
        first_bessels = FirstModSphBesselCollection(lmax)

        r(1) = 0.000d0 
        r(2) = max_r  / number_of_points
        do i = 3, number_of_points
            r(i) = r(i-1) + r(2) 
        end do

        fbessel_values = first_bessels%eval(kappa*r)
        sbessel_values = second_bessels%eval(kappa * r)
        do i = 1, number_of_points
            max_error = 0.0d0
            do j=i+1, number_of_points
                res = 0.0d0
                do l=0, lmax
                    res = res + (2*l+1)* fbessel_values(i, l) * sbessel_values(j, l)  
                end do
                error = abs(2.0d0*kappa/PI * res - exp(-kappa*(r(j)-r(i))) / (r(j)-r(i)))
                print *, "relative error", error/exp(-kappa*(r(j)-r(i))) / (r(j)-r(i))
                if (error > max_error .or. j == i+1) then
                    max_error = error
                    max_error_distance  =  r(j) - r(i)
                    max_error_comparison = exp(-kappa*(r(j)-r(i))) / (r(j)-r(i))
                    max_error_value = 2.0d0*kappa/PI * res
                end if 
            end do
            write(*, "('Maximum error for r:', f14.6, ' found at distance: ', f14.6, ' and was: ', e14.6,&
                       &' value:', e14.6, 'comparison:', e14.6)") &
                    r(i), max_error_distance, max_error, max_error_value, max_error_comparison
        end do
        passed = .TRUE.
    end function 

    function first_bessel_test(scaled, lmax, kappa) result(test_passed)
        logical,      intent(in)      :: scaled
        integer,      intent(in)      :: lmax
        real(REAL64), intent(in)      :: kappa
        logical                       :: test_passed
        real(REAL64)                  :: r(30), error, fbessel_values(30, 0:lmax), &
                                         scaling_factor
        real(REAL128)                 :: comparison_value, z, scal, ez(25), emz(25)
        real(REAL128)                 :: zero = 0.0d0
        integer                       :: i, l
        type(FirstModSphBesselCollection)  :: first_bessels
        ez = 0.0d0
        emz = 0.0d0
        scaling_factor = 1.0d0
        print *, scaling_factor
        first_bessels = FirstModSphBesselCollection(lmax, scaling_factor = scaling_factor)
        r(1) = 0.005d0 
        do i = 2, 25
            r(i) = 1.5 * r(i-1) 
        end do

        fbessel_values = first_bessels%eval(kappa*r)
        scal = 1.0d0
        do l = 0, lmax
            write(*, "('l:', i3, e14.6)"), l, epsilon(zero)
            do i = 1, 25
                z = kappa * r(i)
                ez(i) = ez(i) + (2*l+1) * fbessel_values(i, l)
                emz(i) = emz(i) + (-1)**l * (2*l+1) * fbessel_values(i, l)
                comparison_value = series_first_bessel(z, l, scaling_factor = scal)
                write(*, "('kappa:', f14.6, ' r:', f14.6, ' Comparison:', e14.6, ' Collection:', e14.6, ' Error:', e14.6)") &
                    kappa, r(i), comparison_value, fbessel_values(i, l), comparison_value - fbessel_values(i, l) 
            end do
        end do
        do i = 1, 25
            write (*, "('r:', f14.6, ',sum(bessels):', e14.6, ', real e^z: ', e14.6, ', difference to e^z': e14.6)"), &
                   r(i), ez(i), exp(kappa*r(i)), abs(ez(i)-exp(kappa*r(i))) 
             write (*, "('r:', f14.6, ',sum(-bessels):', e14.6, ', real e^-z: ', e14.6, ', difference to e^-z': e14.6)"), &
                   r(i), emz(i), exp(-kappa*r(i)), abs(emz(i)-exp(-kappa*r(i))) 
        end do
    end function

    function second_bessel_test(scaled, lmax, kappa) result(test_passed)
        logical,      intent(in)      :: scaled
        integer,      intent(in)      :: lmax
        real(REAL64), intent(in)      :: kappa
        logical                       :: test_passed
        real(REAL64)                  :: r(30), error, fbessel_values(30, 0:lmax)
        real(REAL64)                 :: comparison_value
        integer                       :: i, l
        type(SecondModSphBesselCollection)  :: second_bessels

        
        second_bessels = SecondModSphBesselCollection(lmax)
        r(1) = 0.0005d0
        do i = 2, 30
            r(i) = 1.5 * r(i-1)
        end do

        fbessel_values = second_bessels%eval(kappa * r)

        do l = 0, lmax
            write(*, "('l:', i3)"), l 
            do i = 1, 30
                comparison_value = series_second_bessel(kappa * r(i), l, 1.0d0)
                write(*, "('kappa:', f14.6, ' r:', f14.6, ' Comparison:', e14.6, ' Collection:', e14.6, ' Error:', e14.6)") &
                    kappa, r(i), comparison_value, fbessel_values(i, l), comparison_value - fbessel_values(i, l) 
            end do
        end do
    end function

    function series_first_bessel(z, n, scaling_factor) result(reslt)
        real(REAL128), intent(in)   :: z, scaling_factor
        integer,      intent(in)    :: n
        real(REAL128)               :: prefactor, reslt, divider
        integer                     :: i
        integer, parameter          :: TERM_COUNT = 200

        
        prefactor = z / scaling_factor
        prefactor = prefactor**n
        divider = 1  
        do i = 1, 2*n+1, 2
            divider = divider * i 
        end do

        prefactor = prefactor / divider

        reslt = 1.0d0
        divider = 1
        do i = 1, TERM_COUNT
            divider = divider * (2*n + 2*i + 1)
            reslt = reslt + (0.5d0*z**2) ** i / (factorial_real(i) * divider )
        end do
        reslt = prefactor * reslt

    end function

    function series_second_bessel(z, n, scaling_factor) result(reslt)
        real(REAL64),                       intent(in)   :: z, scaling_factor
        integer,                            intent(in)    :: n
        real(REAL64)                                     :: reslt, prefactor1, prefactor2, divider, divider2
        integer                                          :: i
        integer, parameter                                :: TERM_COUNT = 30

        
        prefactor1 = scaling_factor**n * z ** n  
        prefactor2 = scaling_factor**n / ((-1)**(n) * z**(n+1))
        divider = 1  
        do i = 1, 2*n-1, 2
            divider = divider * i 
        end do

        prefactor1 = prefactor1 / (divider * (2*n+1))
        prefactor2 = prefactor2 * divider

        reslt = prefactor1 - prefactor2
        divider = 1
        divider2 = 1
        do i = 1, TERM_COUNT
            divider = divider * (2*n + 2*i + 1)
            divider2 = divider2 * (-2*n + 2*i - 1)
            reslt = reslt + (0.5d0*z**2) ** i / factorial_real(i) * &
                                   ( prefactor1 / divider - prefactor2 / divider2) 
        end do
        reslt = pi/2 * (-1)**(n+1) * reslt

    end function

    function real_potential_mm_test(original_positions, translation, multipole_center, &
                                    evaluation_point, lmax, input_lmax, normalization, output_mode) result(test_passed)
        integer, intent(in)             :: lmax
        integer, intent(in)             :: input_lmax
        !> The type of normalization used in spherical harmonics
        integer,      intent(in)        :: normalization
        !> The original positions of charges without 'multipole_center' offset
        real(REAL64), intent(in)        :: original_positions(:, :)
        !> Vector representing the change in the position of the multipole expansion 
        !! that is applied 
        real(REAL64), intent(in)        :: translation(3)
        !> The offset applied to the charge positions given in 'original_positions' 
        real(REAL64), intent(in)        :: multipole_center(3)
        !> The point where the potential is evaluated
        real(REAL64), intent(in)        :: evaluation_point(3)
        !> The way we are printing, 0: only errors, 1: also success reason, 2: verbose
        integer,      intent(in)        :: output_mode
        !> The result object marking if the test has been passed
        logical                         :: test_passed
        real(REAL64)                    :: original_multipoles((input_lmax+1)**2, 1), &
                                           translated_multipoles((lmax+1)**2, 1), &
                                           comparison_multipoles((lmax+1)**2, 1), &
                                           positions(3, size(original_positions, 2)), &
                                           translated_positions(3, size(original_positions, 2)), &
                                           mc(3, 1), &
                                           potential, comparison_potential, comparison_potential2, &
                                           spherical_harmonics(1, (lmax+1)**2), &
                                           spherical_harmonics2(1, (lmax+1)**2),&
                                           fbessel_values(1, 0:lmax+2), sbessel_values(1, 0:lmax+2), r, &
                                           r_ij, r_ec, r_kl, sbessel_nontranslated(1, 0:lmax+2), &
                                           comparison_potential3, THRESHOLD, multipole_radius, &
                                           translation_distance, evaluation_distance
        integer                         :: l, m, i
        type(RealSphericalHarmonics)    :: harmo
        type(SecondModSphBesselCollection) :: second_bessels


        type(HelmholtzMultipoleTranslator) :: translator

        !> The accuracy threshold is set to 1 %
        THRESHOLD = 1e-2

 
        ! add the multipole_center, offset to the original positions 
        forall(i = 1 : size(original_positions, 2))
            positions(:, i) = original_positions(:, i) + multipole_center
        end forall

        ! calculate the positions that the charges would be if the coordinate system
        ! was translated by 'translation'
        forall (i = 1 : size(original_positions, 2))
            translated_positions(:, i) = positions(:, i) - translation
        end forall 

        ! check if our input parameters are valid for this translation:
        ! in mm. translation the evaluation point must be outside the sphere enclosing the 
        ! original and translated multipole charges

        ! calculate max distance of charge from origo (multipole radius)
        do i = 1, size(positions, 2)
            ! the distance of the charge from the origo
            r = sqrt(sum(positions(:, i)**2))
            if (i == 1 .or. r > multipole_radius) then
                multipole_radius = r
            end if
        end do
        translation_distance = sqrt(sum(translation**2))
        evaluation_distance = sqrt(sum(evaluation_point**2))
        
        if (output_mode == 2) then
            print *, "Translation distance", translation_distance
            print *, "Multipole radius", multipole_radius
        end if 

        if (evaluation_distance < translation_distance + multipole_radius) then            
            if (output_mode >= 1) then
                print *, "Test PASSED due to input fail: Evaluation point is ", &
                        "within the circle of the translated multipole. ",  &
                        "This is not allowed."
                print *, "Evaluation distance:", evaluation_distance, &
                        "Translated multipole radius",  translation_distance + multipole_radius
            end if

            test_passed = .TRUE.
            return
        end if

        ! get multipole moments of the untranslated positions, centered at multipole_center
        original_multipoles(:, 1) = get_point_charge_real_multipoles(positions, &
                                        kappa, &
                                        input_lmax, normalization = normalization, &
                                        scaling_factor = kappa * translation_distance)

        ! get multipole moments, centered at origin, for translated positions
        comparison_multipoles(:, 1) = get_point_charge_real_multipoles(translated_positions, &
                                        kappa, &
                                        lmax, normalization = normalization, &
                                        scaling_factor = kappa * translation_distance)
        mc(:, 1) = [0.0d0, 0.0d0, 0.0d0]

        ! translate 'original_multipoles' multipole expansion by vector 'translation' from origin
        ! to 'translation'. This should be the same as translating the positions by -translation
        translator = HelmholtzMultipoleTranslator(lmax, kappa, input_lmax, normalization = normalization, &
                                                  scaling_factor = kappa * translation_distance)
        translated_multipoles = translator%translate(original_multipoles, mc, translation)
        call translator%destroy()

        r_ec = sqrt(sum((evaluation_point+translation)**2)) 
        ! initialize objects used to evaluate spherical harmonics and bessel functions
        harmo = RealSphericalHarmonics(lmax, normalization = normalization)
        second_bessels = SecondModSphBesselCollection(lmax+2, scaling_factor = kappa * translation_distance)


        ! evaluate second modified spherical bessel functions and spherical harmonics at evaluation point
        r = sqrt(sum((evaluation_point)**2))
        sbessel_values = second_bessels%eval(kappa *  [evaluation_distance])
        spherical_harmonics = harmo%eval(reshape(evaluation_point, [3, 1]))

        if (output_mode >= 2) then
            print *, "second bessels", sbessel_values
        end if
        
        ! evaluate the second modified spherical bessel functions and spherical harmonics
        ! at translated evaluation point. 
        !
        ! These are used in the opposite of the actual operation 
        ! (the evaluation point moves, not the multipoles)
        r_ec = sqrt(sum((evaluation_point+translation)**2)) 
        sbessel_nontranslated = second_bessels%eval(kappa * [r_ec])
        spherical_harmonics2 = harmo%eval(reshape([evaluation_point+translation], [3, 1]))

        ! set the result values to 0
        potential = 0.0d0
        comparison_potential = 0.0d0
        comparison_potential2 = 0.0d0

        do l = 0, lmax
            do  m = -l, l
                
                ! Calculate the translated potential via using the original evaluation point and corresponding
                ! spherical harmonics + the translated multipoles
                potential = potential + sbessel_values(1, l) * spherical_harmonics(1, lm_map(l, m)) &
                                         * translated_multipoles(lm_map(l, m), 1)
                
                ! Calculate the translated potential via using the original evaluation point and corresponding
                ! spherical harmonics + the explicitly evaluated multipoles
                comparison_potential = comparison_potential + sbessel_values(1, l) * spherical_harmonics(1, lm_map(l, m)) &
                                         * comparison_multipoles(lm_map(l, m), 1)
                 
                ! Calculate the translated potential via using the translated evaluation point and corresponding
                ! spherical harmonics + the original multipoles (opposite evaluation)
                comparison_potential2 = comparison_potential2 + sbessel_nontranslated(1, l)  &
                                         * spherical_harmonics2(1, lm_map(l, m)) &
                                         * original_multipoles(lm_map(l, m), 1)

                if (output_mode >= 2) then
                    print *, 8*kappa*potential, 8*kappa*comparison_potential, 8*kappa*comparison_potential2
                end if
            end do
        end do
        r_ij = sqrt(sum((positions(:, 1) - evaluation_point)**2))

        ! calculate the pointwise potential by using the formula e^(-kappa * r) / r
        ! for each position
        comparison_potential3 = 0.0d0
        do i = 1, size(positions, 2) 
            r_kl = sqrt(sum((-positions(:, i) + evaluation_point + translation)**2))
            comparison_potential3 = comparison_potential3 + exp(-kappa * r_kl) / (r_kl)
        end do 
        

        !> Check if the translated potential is within the limits  
        if (abs((8*kappa*potential - comparison_potential3) / comparison_potential3) > THRESHOLD) then
            write (*, "('MM-translation test FAILED. Error-%:', en20.10, ', Abs error:' e20.10)") &
                100 * abs((8*kappa*potential - comparison_potential3) / comparison_potential3), &
                8*kappa*potential - comparison_potential3
            print *, "Potential, Comparison, Eval-point translation , Direct"
            write (*, "(e20.10, e20.10, e20.10, e20.10)") 8*kappa*potential, 8*kappa*comparison_potential, &
                     8*kappa*comparison_potential2, comparison_potential3
            test_passed = .FALSE.
        else if (output_mode >= 1) then
            write (*, "('MM-translation test PASSED. Error-%:', f20.10, ', Abs error:' e20.10)") &
                100 * abs((8*kappa*potential - comparison_potential3) / comparison_potential3), &
                8*kappa*potential - comparison_potential3
            print *, "Potential, Comparison, Eval-point translation , Direct"
            write (*, "(e20.10, e20.10, e20.10, e20.10)") 8*kappa*potential, 8*kappa*comparison_potential, &
                     8*kappa*comparison_potential2, comparison_potential3
            test_passed = .TRUE. 
        end if
        call second_bessels%destroy()
        return
    end function

    function real_potential_ml_test(original_positions, multipole_center, evaluation_center, &
                 evaluation_points, lmax, input_lmax, normalization, output_mode) result(test_passed)
        integer, intent(in)             :: lmax
        integer, intent(in)             :: input_lmax
        !> The type of normalization used in spherical harmonics
        integer,      intent(in)        :: normalization
        !> The original positions of charges without 'multipole_center' offset
        real(REAL64), intent(in)        :: original_positions(:, :)
        !> The center of the multipole moments
        real(REAL64), intent(in)        :: multipole_center(3)
        !> The center of the local expansion
        real(REAL64), intent(in)        :: evaluation_center(3)
        !> The point where the potential is evaluated relative to the new translation center
        real(REAL64), intent(in)        :: evaluation_points(:, :)
        !> The way we are printing, 0: only errors, 1: also success reason, 2: verbose
        integer,      intent(in)        :: output_mode
        !> The result object marking if the test has been passed
        logical                         :: test_passed
        real(REAL64)                    :: original_multipoles((input_lmax+1)**2, 1)
        real(REAL64)                    :: translated_multipoles((lmax+1)**2, 1)
        real(REAL64)                    :: comparison_multipoles((lmax+1)**2, 1)
        real(REAL64)                    :: positions(3, size(original_positions, 2))
        real(REAL64)                    :: translated_positions(3, size(original_positions, 2))
        real(REAL64)                    :: mc(3, 1)
        real(REAL64)                    :: potential, comparison_potential, comparison_potential2, &
                                           spherical_harmonics(1, (lmax+1)**2), &
                                           spherical_harmonics2(1, (lmax+1)**2),&
                                           fbessel_values(1, 0:lmax), sbessel_values(1, 0:lmax), r, &
                                           r_ij, r_ec, r_kl, sbessel_nontranslated(1, 0:lmax), &
                                           comparison_potential3, THRESHOLD, multipole_radius, &
                                           translation_distance, evaluation_distance, normalization_factor, &
                                           scaling_factor
        integer                         :: l, m, i, j
        type(RealSphericalHarmonics)    :: harmo
        type(FirstModSphBesselCollection) :: first_bessels
        type(SecondModSphBesselCollection) :: second_bessels


        type(HelmholtzMultipoleConverter) :: translator

        !> The accuracy threshold is set to 1 %
        THRESHOLD = 1e-1
 
        scaling_factor = 5.0d0 !kappa * sqrt(sum((multipole_center -evaluation_center)**2))       

        ! calculate the positions that the charges would be if the coordinate system
        ! was translated by 'translation'
        forall (i = 1 : size(original_positions, 2))
            translated_positions(:, i) = -evaluation_center + multipole_center + original_positions(:, i)
        end forall 

        ! check if our input parameters are valid for this translation:
        ! in mm. translation the evaluation point must be outside the sphere enclosing the 
        ! original and translated multipole charges

        ! calculate max distance of charge from origo (multipole radius)
        do i = 1, size(original_positions, 2)
            ! the distance of the charge from the multipole center
            r = sqrt(sum(original_positions(:, i)**2))
            if (i == 1 .or. r > multipole_radius) then
                multipole_radius = r
            end if
        end do

        translation_distance = sqrt(sum((multipole_center - evaluation_center)**2))
        if (translation_distance <= 2 * multipole_radius) then
            if (output_mode >= 1) then
                print *, "Test PASSED due to input fail: The converted local expansion and ", &
                        "the multipole expansion overlap. ",  &
                        "This is not allowed."
                
                print *, "Translation distance:", translation_distance, &
                        "Multipole expansion radius: ", multipole_radius
            end if
            test_passed = .TRUE.
            return
        end if

        
        
        !if (evaluation_distance > multipole_radius) then
        !    test_passed = .TRUE.
        !    write(*, "('Test PASSED due to input fail: The evaluation point (r=',f10.6,') is not within &
        !               & the local expansion sphere (r=',f10.6,'). This is not allowed.')")     &
        !              evaluation_distance, multipole_radius 
        !    return
        !end if
        ! get multipole moments of the untranslated positions, centered at multipole_center
        original_multipoles(:, 1) = get_point_charge_real_multipoles(original_positions, &
                                        kappa, &
                                        input_lmax, normalization = normalization, &
                                        scaling_factor = scaling_factor)

        ! get local expansion moments, centered at 'evaluation_center', for translated positions
        comparison_multipoles(:, 1) = get_point_charge_real_local_expansion(translated_positions, &
                                        kappa, &
                                        lmax, normalization = normalization, &
                                        scaling_factor = scaling_factor)
        mc(:, 1) = multipole_center

        ! translate 'original_multipoles' multipole expansion from 'multipole_center'
        ! to 'evaluation_center'.
        translator = HelmholtzMultipoleConverter(lmax, kappa, input_lmax, normalization = normalization, &
                                                 scaling_factor = scaling_factor)

        translated_multipoles = translator%translate(original_multipoles, mc, evaluation_center)
        call translator%destroy()
        
        ! initialize objects used to evaluate spherical harmonics and bessel functions
        harmo = RealSphericalHarmonics(lmax, normalization = normalization)
        first_bessels = FirstModSphBesselCollection(lmax, scaling_factor = scaling_factor)
        second_bessels = SecondModSphBesselCollection(lmax, scaling_factor = scaling_factor)

        test_passed = .TRUE.
        do i = 1, size(evaluation_points, 2) 
            evaluation_distance = sqrt(sum(evaluation_points(:, i)**2))

            !if (evaluation_distance > multipole_radius) then
            !    if (output_mode >= 1) then
            !        print *, "Test PASSED due to input fail: The evaluation point is ", &
            !                "not within the converted local expansion. ",  &
            !                "This is not allowed."
            !        
            !        print *, "Evaluation distance:", evaluation_distance, &
            !                "Local expansion radius: ", multipole_radius
            !    end if
            !    test_passed = .TRUE. 
            !    return
            !end if 
            if (output_mode >= 2) then
                print *, "Translation distance", translation_distance
                print *, "Multipole radius", multipole_radius
                print *, "Evaluation distance", evaluation_distance
            end if 

            ! evaluate second modified spherical bessel functions and spherical harmonics at evaluation point
            fbessel_values = first_bessels%eval(kappa * [evaluation_distance])
            spherical_harmonics = harmo%eval(reshape(evaluation_points(:, i), [3, 1]))
            sbessel_values = second_bessels%eval(kappa *  [translation_distance])
            spherical_harmonics2 = harmo%eval(reshape(evaluation_points(:, i), [3, 1]))

            ! set the result values to 0
            potential = 0.0d0
            comparison_potential = 0.0d0
            comparison_potential2 = 0.0d0
            normalization_factor = 1.0d0
            do l = 0, lmax
                if (normalization /= 2) then
                    normalization_factor = (2*l+1)/dble(4*pi)
                end if
                do  m = -l, l
                    ! Calculate the translated potential via using the original evaluation point and corresponding
                    ! spherical harmonics + the translated multipoles
                    potential = potential +  fbessel_values(1, l) * normalization_factor  * spherical_harmonics(1, lm_map(l, m)) &
                                            * translated_multipoles(lm_map(l, m), 1)

                    ! Calculate the translated potential via using the original evaluation point and corresponding
                    ! spherical harmonics + the explicitly evaluated multipoles
                    comparison_potential = comparison_potential +   normalization_factor * fbessel_values(1, l) &
                                            * spherical_harmonics(1, lm_map(l, m)) &
                                            * comparison_multipoles(lm_map(l, m), 1)

                    comparison_potential2 = comparison_potential2 + sbessel_values(1, l) * normalization_factor  &
                                            * spherical_harmonics2(1, lm_map(l, m)) &
                                            * original_multipoles(lm_map(l, m), 1)
                    
                    if (output_mode >= 2) then
                        print *, 8*kappa*potential, 8*kappa*comparison_potential, 8*kappa*comparison_potential2
                    end if
                end do
            end do

            ! calculate the pointwise potential by using the formula e^(-kappa * r) / r
            ! for each position
            comparison_potential3 = 0.0d0
            do j = 1, size(translated_positions, 2) 
                !print *, translated_positions(:, j) - evaluation_points(:, i)
                r_kl = sqrt(sum((translated_positions(:, j) - evaluation_points(:, i))**2))
                comparison_potential3 = comparison_potential3 + 0.1d0 * exp(-kappa * r_kl) / (r_kl)
            end do 
            

            !> Check if the translated potential is within the limits  
            if (abs((8*kappa*potential - comparison_potential3) / comparison_potential3) > THRESHOLD) then
                if (output_mode >= 2) then
                    print *, "Potential, Comparison, Eval-point translation , Direct"
                    write (*, "(e20.10, e20.10, e20.10, e20.10)") 8*kappa*potential, 8*kappa*comparison_potential, &
                            comparison_potential3
                end if
                
 
                if (abs((8*kappa*comparison_potential - comparison_potential3) / comparison_potential3) > THRESHOLD) then
                    write (*, "('ML-translation test FAILED. Error-%:', e12.4, ', Abs error:' e12.4, ', Correct value:', e12.4, &
                           &'. REASON: OUTPUT LMAX is too SMALL.')") &
                    100 * abs((8*kappa*potential - comparison_potential3) / comparison_potential3), &
                    8*kappa*potential - comparison_potential3, comparison_potential3
                else
                    write (*, "('ML-translation test FAILED. Error-%:', e12.4, ', Abs error:' e12.4, ', Correct value:', e12.4, &
                           &'. REASON: INPUT LMAX is too SMALL.')") &
                    100 * abs((8*kappa*potential - comparison_potential3) / comparison_potential3), &
                    8*kappa*potential - comparison_potential3, comparison_potential3
                end if
                
                write (*, "('Local Expansion potential Error-%:', e12.4, ', Abs error:' e12.4, ', Correct value:', e12.4)") &
                    100 * abs((8*kappa*comparison_potential - comparison_potential3) / comparison_potential3), &
                    8*kappa*comparison_potential - comparison_potential3, comparison_potential3
                test_passed = .FALSE.
            else if (output_mode >= 1) then
                write (*, "('ML-translation test passed. Error-%:', e12.4, ', Abs error:' e12.4, ', Correct value:', e12.4)") &
                    100 * abs((8*kappa*potential - comparison_potential3) / comparison_potential3), &
                    8*kappa*potential - comparison_potential3, comparison_potential3
                !print *, "Potential, Comparison, Eval-point translation , Direct"
                !write (*, "(e20.10, e20.10, e20.10, e20.10)") 8*kappa*potential, 8*kappa*comparison_potential, &
                !        comparison_potential3
            end if
        end do
        call first_bessels%destroy()
        call second_bessels%destroy()

        return
    end function

    subroutine real_multipole_to_local_conversion_test(original_positions, translation, &
                                         multipole_center, lmax, input_lmax, normalization)
        integer, intent(in)             :: lmax, input_lmax
        integer,      intent(in)        :: normalization
        real(REAL64), intent(in)        :: original_positions(:, :), translation(3), multipole_center(3)
        real(REAL64)                    :: original_multipoles((input_lmax+1)**2, 1), &
                                           translated_multipoles((lmax+1)**2, 1), &
                                           comparison_multipoles((lmax+1)**2, 1), &
                                           positions(3, size(original_positions, 2)), &
                                           translated_positions(3, size(original_positions, 2)), &
                                           mc(3, 1), scaling_factor
        integer                         :: l, m, i

        type(HelmholtzMultipoleConverter)     :: translator


        scaling_factor = 1.0d0 ! kappa * sqrt(sum(translation**2))
        translator = HelmholtzMultipoleConverter(lmax, kappa, input_lmax, normalization = normalization, &
                                                 scaling_factor = scaling_factor)
        translated_positions = original_positions

        forall(i = 1 : size(original_positions, 2))
            positions(:, i) = original_positions(:, i) + multipole_center
        end forall

        forall (i = 1 : size(original_positions, 2))
            translated_positions(:, i) = positions(:, i) - translation
        end forall 

        original_multipoles(:, 1) = get_point_charge_real_multipoles(positions, &
                                        kappa, &
                                        input_lmax, normalization = normalization, &
                                        scaling_factor = scaling_factor)
        comparison_multipoles(:, 1) = get_point_charge_real_local_expansion(translated_positions, &
                                        kappa, &
                                        lmax, normalization = normalization, scaling_factor = scaling_factor)
        mc(:, 1) = multipole_center 
        translated_multipoles = translator%translate(original_multipoles, mc, translation)
        
        call print_results(original_multipoles, translated_multipoles, comparison_multipoles, lmax)
    end subroutine
       
    subroutine print_results(original_multipoles, translated_multipoles, comparison_multipoles, lmax)
        integer,      intent(in)        :: lmax
        real(REAL64), intent(in)        :: original_multipoles((lmax+1)**2, 1), &
                                           translated_multipoles((lmax+1)**2, 1), &
                                           comparison_multipoles((lmax+1)**2, 1)
        integer                         :: l, m
        print *, "REAL MULTIPOLES"
        print *, "(l, m), original, translated, direct, error, error-%"
        print *, "-----------------------------------------------------------"
        do l = 0, lmax
            do m = -l, l
                write(*, '(i4, i4, f14.6, e14.6, e14.6, e14.6, f14.6)') l, m, original_multipoles(lm_map(l, m), 1), &
                        translated_multipoles(lm_map(l, m), 1), &
                        comparison_multipoles(lm_map(l, m), 1), &
                        (translated_multipoles(lm_map(l, m), 1)-comparison_multipoles(lm_map(l, m), 1)), &
                        100*(translated_multipoles(lm_map(l, m), 1)-comparison_multipoles(lm_map(l, m), 1)) &
                         / comparison_multipoles(lm_map(l, m), 1)
            end do
        end do
    end subroutine

    subroutine real_multipole_translation_test(original_positions, translation, &
                     multipole_center, lmax, input_lmax, normalization)
        integer,      intent(in)        :: lmax, input_lmax
        integer,      intent(in)        :: normalization
        real(REAL64), intent(in)        :: original_positions(:, :), translation(3), multipole_center(3)
        real(REAL64)                    :: original_multipoles((input_lmax+1)**2, 1), &
                                           translated_multipoles((lmax+1)**2, 1), &
                                           comparison_multipoles((lmax+1)**2, 1), &
                                           positions(3, size(original_positions, 2)), &
                                           translated_positions(3, size(original_positions, 2)), &
                                           mc(3, 1), r
        integer                         :: l, m, i

        type(HelmholtzMultipoleTranslator) :: translator


        r = sqrt(sum(translation)**2)
        translator = HelmholtzMultipoleTranslator(lmax, kappa, input_lmax, &
             normalization = normalization, scaling_factor = kappa * r)
        translated_positions = original_positions

        forall(i = 1 : size(original_positions, 2))
            positions(:, i) = original_positions(:, i) + multipole_center
        end forall

        forall (i = 1 : size(original_positions, 2))
            translated_positions(:, i) = positions(:, i) - translation
        end forall 

        original_multipoles(:, 1) = get_point_charge_real_multipoles(positions, &
                                        kappa, &
                                        input_lmax, normalization = normalization, &
                                        scaling_factor = kappa * r)
        comparison_multipoles(:, 1) = get_point_charge_real_multipoles(translated_positions, &
                                        kappa, &
                                        lmax, normalization = normalization, &
                                        scaling_factor = kappa * r)
        mc(:, 1) = [0.0d0, 0.0d0, 0.0d0]
        translated_multipoles = translator%translate(original_multipoles, mc, translation)
        
        call print_results(original_multipoles, translated_multipoles, comparison_multipoles, lmax)
    end subroutine

    subroutine complex_multipole_translation_test(lmax, input_lmax, normalization)
        integer, intent(in)             :: lmax, input_lmax
        integer,      intent(in)        :: normalization
        complex*16                      :: original_multipoles((input_lmax+1)**2, 1), &
                                           translated_multipoles((lmax+1)**2, 1), &
                                           comparison_multipoles((lmax+1)**2, 1)
        real(REAL64)                    :: original_positions(3, 3), translated_positions(3, 3), translation(3), &
                                           origo(3, 1), rotated_positions(3, 3)
        integer                         :: l, m, i

        type(HelmholtzMultipoleTranslator)     :: translator

        translator = HelmholtzMultipoleTranslator(lmax, kappa, input_lmax, normalization = normalization)
        origo = 0.0d0
        original_positions(:, 1) = [0.0d0, 1.0d0, 1.0d0]
        original_positions(:, 2) = [1.0d0, 0.0d0, 0.0d0]
        original_positions(:, 3) = [-1.0d0, 2.0d0, 4.0d0]
        translation = [1.0d0, 0.0d0, 1.0d0]
        translated_positions = original_positions
        forall (i = 1:3)
            translated_positions(:, i) = translated_positions(:, i) - translation
        end forall 
        forall(i = 1:3)
            rotated_positions(1, i) = original_positions(1, i)
            rotated_positions(2, i) = -original_positions(3, i)
            rotated_positions(3, i) = original_positions(2, i)
        end forall
        original_multipoles(:, 1) = get_point_charge_complex_multipoles(original_positions, kappa, &
                                        input_lmax, normalization = normalization)
        comparison_multipoles(:, 1) = get_point_charge_complex_multipoles(translated_positions, kappa, &
                                        lmax, normalization = normalization)
        translated_multipoles = translator%translate(original_multipoles, origo, translation)
        print *, "translated positions", translated_positions
        print *, "COMPLEX MULTIPOLES"
        print *, "(l, m), original, translated, direct"
        print *, "-----------------------------------------------------------"
        do l = 0, lmax
            do m = -l, l
                write(*, '(i4,",", i4,",", f4.2, f4.2, f4.2, f4.2)')l, m, original_multipoles(lm_map(l, m), 1), &
                        translated_multipoles(lm_map(l, m), 1), &
                        comparison_multipoles(lm_map(l, m), 1)
            end do
        end do
    end subroutine

    function get_point_charge_real_multipoles(positions, kappa, lmax, normalization, scaling_factor) result(multipoles)
        real(REAL64), intent(in)        :: positions(:, :)
        real(REAL64), intent(in)        :: kappa
        integer,      intent(in)        :: lmax 
        integer,      intent(in)        :: normalization
        real(REAL64), intent(in)        :: scaling_factor
        type(FirstModSphBesselCollection) :: bessels
        type(FirstModSphBessel)         :: bess
        real(REAL64)                    :: multipoles((lmax+1)**2)
        type(RealSphericalHarmonics)    :: harmo
        real(REAL64)                    :: values(1), spherical_harmonics(size(positions, 2), (lmax+1)**2), &
                                           bessel_values(1, 0:lmax)
        integer                         :: i, l, m

        harmo = RealSphericalHarmonics(lmax, normalization = normalization)

        spherical_harmonics = harmo%eval(positions)
        bessels = FirstModSphBesselCollection(lmax, scaling_factor = scaling_factor)
        multipoles = 0.0d0
        do i = 1, size(positions, 2)
            bessel_values = bessels%eval(kappa *  [dsqrt(positions(1, i)**2 + &
                                                        positions(2, i)**2 + &
                                                        positions(3, i)**2)])
            do l = 0, lmax
                do m = -l, l  
                    multipoles(lm_map(l, m)) = multipoles(lm_map(l, m)) + bessel_values(1, l) &
                                  * spherical_harmonics(i, lm_map(l, m)) 
                end do
            end do
        end do
    end function 

    function get_point_charge_real_local_expansion(positions, kappa, lmax, normalization, scaling_factor) result(multipoles)
        real(REAL64), intent(in)        :: positions(:, :)
        real(REAL64), intent(in)        :: kappa
        integer,      intent(in)        :: lmax 
        integer,      intent(in)        :: normalization
        real(REAL64), intent(in)        :: scaling_factor
        type(SecondModSphBesselCollection) :: bessels
        real(REAL64)                    :: multipoles((lmax+1)**2)
        type(RealSphericalHarmonics)    :: harmo
        real(REAL64)                    :: values(1), spherical_harmonics(size(positions, 2), (lmax+1)**2), &
                                           bessel_values(1, 0:lmax)
        integer                         :: i, l, m

        harmo = RealSphericalHarmonics(lmax, normalization = normalization)
        spherical_harmonics = harmo%eval(positions)
        bessels = SecondModSphBesselCollection(lmax, scaling_factor = scaling_factor)
        multipoles = 0.0d0
        do i = 1, size(positions, 2)
            bessel_values = bessels%eval(kappa *  [dsqrt(positions(1, i)**2 + &
                                                        positions(2, i)**2 + &
                                                        positions(3, i)**2)])
            do l = 0, lmax    
                do m = -l, l 
                    multipoles(lm_map(l, m)) = multipoles(lm_map(l, m)) + bessel_values(1, l) &
                        * spherical_harmonics(i, lm_map(l, m))   
                end do
            end do
        end do
    end function 

    function get_point_charge_complex_multipoles(positions, kappa, lmax, normalization) result(multipoles)
        real(REAL64), intent(in)        :: positions(:, :)
        real(REAL64), intent(in)        :: kappa
        integer,      intent(in)        :: lmax
        integer,      intent(in)        :: normalization
        type(FirstModSphBessel)         :: bessels
        complex*16                      :: multipoles((lmax+1)**2), spherical_harmonics(size(positions, 2), (lmax+1)**2)
        type(ComplexSphericalHarmonics) :: harmo
        real(REAL64)                    :: values(1)
        integer                         :: i, l, m

        harmo = ComplexSphericalHarmonics(lmax, normalization = normalization)
        spherical_harmonics = harmo%eval(positions)
        multipoles = complex(0.0d0,0.0d0)
        do i = 1, size(positions, 2)
            do l = 0, lmax
                bessels = FirstModSphBessel(l)
                values = bessels%eval(kappa * [dsqrt(positions(1, i)**2 + positions(2, i)**2 + positions(3, i)**2)])
                do m = -l, l 
                    multipoles(lm_map(l, m)) = multipoles(lm_map(l, m)) + values(1) * spherical_harmonics(i, lm_map(l, m))    
                end do
            end do
        end do
    end function
end program