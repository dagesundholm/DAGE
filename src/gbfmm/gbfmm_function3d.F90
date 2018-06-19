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
! Assistance class for the Grid-based Fast Multipole Method. Contains
! the GBFMMFunction3D object derived from Function3D to allow Coulomb3D-like 
! use of the GBFMM method. Now the GBFMM can be used as 
!
!     allocate(V, source = GBFMMCoulomb3D(grid=grid,         &
!                                         maxlevel=maxlevel, &
!                                         lmax=lmax))
!     rho .dot. (V .apply. rho) 
!
! where rho is the charge density, of type Function3D, and V is of type
! class(Coulomb3D). Grid is a Grid3D object, maxlevel is the number of recursive 
! box splits and lmax is the number of TODO taken into account in the multipole calculation.
! 
! Pauli Parkkinen, 2015

module GBFMMFunction3D_class
#ifdef HAVE_MPI
    use MPI
#endif
    use ISO_FORTRAN_ENV
    use Function3D_class
    use Bubbles_class
    use Function3D_types_m
    use GBFMMParallelInfo_class
    use Multipole_class
    use Grid_class
    use Globals_m
    use MultiPoleTools_m
    use RealSphericalHarmonics_class

    implicit none
    
    public      :: GBFMMFunction3D
    public      :: get_box_cell_index_limits
    public      :: get_box_center
    public      :: para_range
    public      :: morton
    public      :: unmorton
    public      :: assignment(=)

    private

    

    type, public, extends(Function3D) :: GBFMMFunction3D
        !> parallel info containing information about box splitting and processor ranges
        type(GBFMMParallelInfo), pointer:: parallel_info
        !> Level of splitting in boxes
        integer                         :: maxlevel = 2
        !> Maximum multipole taken into account
        integer                         :: lmax = 7
        !> If the downward pass has been performed for the potential
        logical                         :: downward_pass = .FALSE.
       
        real(REAL64), allocatable       :: farfield_potential(:,:) 
        real(REAL64), allocatable       :: bubbles_farfield_potential(:,:) 
        type(Function3D), allocatable   :: nearfield_potential(:)
        
    contains     
        procedure                       :: init                         => &
                                               GBFMMFunction3D_init_sub
        procedure                       :: destroy                      => &
                                               GBFMMFunction3D_destroy
        !procedure                       :: left_dot_product             => &
        !                                       GBFMMFunction3D_left_dot_product
        procedure                       :: translate_multipoles         => &
                                               GBFMMFunction3D_translate_multipoles
        procedure                       :: evaluate                     => &
                                               GBFMMFunction3D_evaluate
        procedure                       :: evaluate_farfield_potential_grid => &
                                               GBFMMFunction3D_evaluate_farfield_potential_grid
        procedure                       :: evaluate_point_charge_energy => &
                                               GBFMMFunction3D_evaluate_point_charge_energy
        !procedure                       :: multiply                     => &
        !                                       GBFMMFunction3D_product
    end type   

    
    ! constructors
    interface GBFMMFunction3D
        module procedure :: GBFMMFunction3D_init
    end interface 
    
    interface assignment(=)
        module procedure :: Function3D_assign_GBFMMFunction3D
    end interface

contains

    subroutine GBFMMFunction3D_init_sub(self, parallel_info, lmax, type, downward_pass, bubble_count)
        class(GBFMMFunction3D),  intent(inout)       :: self
        !> parallel info containing information about box splitting and processor ranges
        type(GBFMMParallelInfo), intent(in), target  :: parallel_info
        !> the maximum number of multipoles taken into account
        integer,          intent(in)                 :: lmax
        !> The type of the function, in terms of Function3D_type
        integer,          intent(in)                 :: type
        !> If the downward pass has been performed for the potential
        logical,          intent(in)                 :: downward_pass
        !> The type of the function, in terms of Function3D_type
        integer,          intent(in)                 :: bubble_count
    
        integer                                      :: domain(2), total_box_count 

        self%parallel_info => parallel_info
        self%parallelization_info => parallel_info 
        domain = parallel_info%get_domain(parallel_info%maxlevel)
        total_box_count = parallel_info%get_total_box_count()
        self%grid => self%parallelization_info%get_grid()

        ! allocate memory for multipole moments and potential
        allocate(self%farfield_potential((lmax + 1)**2, total_box_count), source=0.0d0)
        allocate(self%bubbles_farfield_potential((lmax + 1)**2, bubble_count), source=0.0d0)
        allocate(self%nearfield_potential(domain(1):domain(2)))
        
        call self%init_cube()
        self%cube = 0.0d0
        self%downward_pass = downward_pass
        self%maxlevel = parallel_info%maxlevel
        self%lmax = lmax
        self%type = type
        
        self%label = ""
    end subroutine

    function GBFMMFunction3D_init(parallel_info, lmax, type, downward_pass, bubble_count) result(new)
        !> parallel info containing information about box splitting and processor ranges
        type(GBFMMParallelInfo), intent(in), target  :: parallel_info
        !> the maximum number of multipoles taken into account
        integer,          intent(in)                 :: lmax
        !> The type of the function, in terms of Function3D_type
        integer,          intent(in)                 :: type
        !> If the downward pass has been performed for the potential
        logical,          intent(in)                 :: downward_pass
        !> The type of the function, in terms of Function3D_type
        integer,          intent(in)                 :: bubble_count
        type(GBFMMFunction3D)                        :: new

        call new%init(parallel_info, lmax, type, downward_pass, bubble_count)
        return
    end function
    
    subroutine Function3D_assign_GBFMMFunction3D(function1, function2)
        class(Function3D), intent(out), allocatable :: function1
        type(GBFMMFunction3D), intent(in)                :: function2

        allocate(function1, source = function2)
    end subroutine

    !> Evaluate the potential at positions included in 'points'
    function GBFMMFunction3D_evaluate(self, points, add_bubbles) result(results)
        class(GBFMMFunction3D), intent(in) :: self
        !> the points in which the potential is evaluated
        real(REAL64), intent(in)           :: points(:,:)
        !> multiply the potential of each point with corresponding multiplier, if not present
        !! each potential is multiplied with 1
        !real(REAL64), optional  :: multipliers(:)
        !> here only because of the inteface at Function3D
        logical, optional, intent(in)      :: add_bubbles
        ! The result array
        real(REAL64)                       :: res(size(points,2)), results(size(points,2))
        integer                            :: ipoint, ierr
        res = 0.0d0
        results = 0.0d0

        do ipoint = 1, size(points, 2)
            ! as this function does not take the charge into account, the charge is set to 1
            ! note: only the one processor containing the point in its domain, calculates
            ! anything
            res(ipoint) = self%evaluate_point_charge_energy(1.0d0, points(:, ipoint))
        end do

#ifdef HAVE_MPI
        
        ! Get the contributions from all processors, every other processor
        ! except the one that calculates the potential, adds zero
        call mpi_allreduce(res, results, &
                        size(res), MPI_DOUBLE_PRECISION, MPI_SUM,  &
                        MPI_COMM_WORLD, ierr)
#else
        results = res
#endif
    end function 

    function GBFMMFunction3D_evaluate_point_charge_energy(self, charge, charge_position) result(point_charge_energy)
        class(GBFMMFunction3D), intent(in) :: self
        !> charge of the point charge (floating point number to allow) other future uses
        real(REAL64),           intent(in) :: charge
        !> the position of the point charge
        real(REAL64)                       :: charge_position(3)

        real(REAL64)                       :: charge_positions(3, 1)
        type(Function3D)                   :: box_potential
        ! result variable
         
        real(REAL64)                       :: point_charge_energy, nearfield_energy(1)
        real(REAL64)                       :: multipole_moments((self%lmax + 1)**2), box_center(3)
        integer                            :: ibox, domain(2)
        type(RealRegularSolidHarmonics)    :: solid_harmonics

        ! Find the maxlevel box in which the point charge is in
        ibox = self%parallel_info%get_box_index_from_coordinates(charge_position, self%parallel_info%maxlevel)
        ! Get the computational domain of the current processor
        domain = self%parallel_info%get_domain(self%maxlevel)     
   
        ! If the box belongs to this processors domain, evaluate energy
        if (ibox >= domain(1) .and. ibox <= domain(2)) then

            ! Evaluate the farfield potential of the point charge
            
            ! get the center of the box 'ibox'
            !box_center = self%parallel_info%get_box_center(ibox, self%maxlevel)      

            ! Evaluate the multipole moment of the point charge at 
            ! the center of the box containing the point charge
            !multipole_moments = point_charge_multipoles(self%lmax, charge, charge_position, box_center)
         
            ! evaluate the farfield energy by multiplying the point charge multipoles (at box_center)
            ! with the potential at the point
            point_charge_energy = dot_product(self%farfield_potential(:,ibox),&
                                          multipole_moments)

            ! Evaluate and add the nearfield potential of the point charge
            charge_positions(:, 1) = charge_position

            solid_harmonics = RealRegularSolidHarmonics(self%lmax)
            box_potential = self%nearfield_potential(ibox)
            !box_potential%cube = box_potential%cube +  &
            !    self%evaluate_farfield_potential_grid(solid_harmonics, &
            !                                          box_potential%grid, &
            !                                          ibox)
            nearfield_energy = box_potential%evaluate(charge_positions)
            !print *, "nearfield nuc-el energy", nearfield_energy, point_charge_energy
            point_charge_energy = charge * nearfield_energy(1)
            call box_potential%destroy()
            call solid_harmonics%destroy()
        else 
            point_charge_energy = 0.0d0
        end if


        
    end function 

    

    function GBFMMFunction3D_product(self, f2, cell_limits, part_of_dot_product, result_bubbles) result(result_function)
        class(GBFMMFunction3D), intent(in), target :: self
        class(Function3D),      intent(in), target :: f2
        type(Function3D)                           :: result_function
        !> Limits in the area in which the operator is applied 
        !! if the f2 contains the entire function
        integer, intent(in), optional              :: cell_limits(2, 3)
        !> If this is set to .true. only cube part's multipole moments are evaluated
        logical, intent(in), optional              :: part_of_dot_product
        type(Bubbles), intent(in), optional        :: result_bubbles
        type(Bubbles)                              :: box_bubbles
        type(Bubbles), pointer                     :: resulting_bubbles
        
        ! Temporary variables
        type(Function3D)                           :: energy_density_ibox, box_potential, temp

        ! domain of calculated boxes
        integer                                    :: domain(2)

        integer                                    :: ibox, box_cell_limits(2, 3), cube_limits(2, 3)

        type(Bubbles)                              :: farfield_bubbles
        !type(RealRegularSolidHarmonics)            :: solid_harmonics

        !solid_harmonics = RealRegularSolidHarmonics(self%lmax)
         
        call bigben%split("GBFMMFunction3D - Function3D -product")
        ! copy the original input function to be basis of the result_function
        select type (f2)
            type is (Function3D)
                result_function = Function3D(f2, type=f2%type)
                result_function%cube = 0.0d0
            class default
                print *, "ERROR: In GBFMMFunction3D: f2 is not Function3D like it should be"
                stop
        end select

        ! get processor box-index-domain on the maximum level
        domain = self%parallel_info%get_domain(self%maxlevel)

        ! calculate result bubbles
        call bigben%split("Bubbles product")
        call self%multiply_bubbles(f2, result_function%bubbles)
        call bigben%stop_and_print() 
 
        
        do ibox = domain(1), domain(2)
            ! get the starting and ending cells indexes of the box
            box_cell_limits = self%parallel_info%get_box_cell_index_limits(ibox, self%maxlevel, global = .FALSE.)

            ! get the cube limits for the box
            cube_limits = self%parallel_info%get_cube_ranges(box_cell_limits)

            ! convert the farfield potential to bubbles representation
            !farfield_bubbles = self%get_farfield_bubbles(ibox)
            
            ! get the nearfield potential of the box
            box_potential = self%nearfield_potential(ibox)

            ! inject the farfield bubbles to the cube of the box potential
            !box_potential%cube = box_potential%cube + farfield_bubbles%eval_3dgrid(box_potential%grid)
            !box_potential%cube = box_potential%cube +  &
            !    self%evaluate_farfield_potential_grid(solid_harmonics, &
            !                                          box_potential%grid, &
            !                                          ibox)

            ! multiply the 'box_potential' and f2 (most likely the charge density)
            energy_density_ibox = box_potential%multiply(f2, box_cell_limits, &
                                      result_bubbles = result_function%bubbles)

            !box_bubbles = self%bubbles%get_sub_bubbles(self%parallel_info%get_box_limits(ibox))

            !if (box_bubbles%get_nbub() > 0) then
            !    energy_density_ibox = box_potential%multiply(f2, box_cell_limits, &
            !                              only_cube = .FALSE.)
            !else
            !    energy_density_ibox = box_potential%multiply(f2, box_cell_limits, &
            !                              only_cube = .FALSE., result_bubbles = result_function%bubbles)
            !end if

            
            

            !temp = box_potential%multiply(f2, box_cell_limits, &
            !                                       only_cube = .TRUE.)
            !print *, "maxval minval cube", maxval(temp%cube), minval(temp%cube)
            !energy_density_ibox%cube = energy_density_ibox%cube + temp%cube
            !call temp%destroy()
            
            ! after the multiplication, set the cube of the  'energy_density' to be the corresponding
            ! part of result_function
            result_function%cube(cube_limits(1, X_) : cube_limits (2, X_),   &
                                 cube_limits(1, Y_) : cube_limits (2, Y_),   &
                                 cube_limits(1, Z_) : cube_limits (2, Z_)) = &
                energy_density_ibox%cube

            ! deallocate memory
            call farfield_bubbles%destroy()
            call box_potential%destroy()
            call energy_density_ibox%destroy()
        end do
        !call solid_harmonics%destroy()
        call bigben%stop_and_print() 
    end function

    function GBFMMFunction3D_evaluate_farfield_potential_grid(self, solid_harmonics, grid, ibox) result(potential_cube)
        class(GBFMMFunction3D),          intent(in)    :: self
        integer,       intent(in)                      :: ibox
        !> Solid Harmonics object, Note: must be Racah normalized
        type(RealRegularSolidHarmonics), intent(in)    :: solid_harmonics
        !> Grid to which we are injecting the potential
        type(Grid3D),                    intent(in)    :: grid
        real(REAL64)                                   :: potential_cube(grid%axis(X_)%get_shape(), &
                                                                         grid%axis(Y_)%get_shape(), &
                                                                         grid%axis(Z_)%get_shape())
        integer                                        :: l, m
        real(REAL64)                                   :: solid_harmonics_cube(grid%axis(X_)%get_shape(), &
                                                                               grid%axis(Y_)%get_shape(), &
                                                                               grid%axis(Z_)%get_shape(), &
                                                                               (self%lmax+1)**2)
        real(REAL64)                                  :: box_center(3)

        call bigben%split("Solid Harmonics evaluate grid")
        ! Evaluate the center of the box
        box_center = self%parallel_info%get_box_center(ibox, self%parallel_info%maxlevel)
        
        ! evaluate real solid harmonics in Racah's normalization 
        solid_harmonics_cube = solid_harmonics%eval_grid(grid, box_center)
        call bigben%stop_and_print()

        call bigben%split("Evaluate multipoles to cube")
        potential_cube = 0.0d0
        do l=0, self%lmax 
            do m=-l,l
                
                potential_cube = potential_cube + &
                    solid_harmonics_cube(:, :, :, l*(l+1)+m+1) * self%farfield_potential(l*(l+1)+m+1, ibox)
            end do
        end do
        call bigben%stop_and_print()
    end function

    !> Interaction energy between the electric potential `self` and
    !! the charge density `f2`
    !
    ! Calculation is in direction f2 .dot. self, thus the naming.
    function GBFMMFunction3D_left_dot_product(self, f2) result(total_interaction_energy)
        class(GBFMMFunction3D), intent(in) :: self
        !> Input charge density
        class(Function3D),      intent(in) :: f2
        real(REAL64)                     :: total_interaction_energy

        ! Boxed multipole moments of rho
        real(REAL64), allocatable        :: multipole_moments(:, :), bubble_multipole_moments(:, :), &
                                            temp_bubble_multipole_moments(:, :)

        ! Temporary variables
        type(Function3D)                 :: energy_density_ibox, box_potential
        real(REAL64)                     :: nearfield_interaction_energy, & 
                                            total_nearfield_interaction_energy
        real(REAL64)                     :: farfield_interaction_energy, &
                                            total_farfield_interaction_energy
        type(Bubbles)                    :: result_bubbles
 
        ! domain of calculated boxes
        integer                         :: domain(2)
        type(Bubbles)                   :: sub_bubbles, box_bubbles
 
        ! first level for which the energy contraction is done
        ! depends if downward pass has been performed
        integer                         :: first_level

        ! box limits as integers
        integer                          :: box_limits(2, 3), total_box_count
        ! maximum derivative order in cube contaminant calculations
        integer, parameter               :: max_derivative_order = 2

        real(REAL64)                     :: box_center(3)
        integer                          :: ibox, ierr, bubble_limits(2), ilevel, offset, ibub, i

        real(REAL64)                     :: tic, toc
        type(RealRegularSolidHarmonics)  :: solid_harmonics

        solid_harmonics = RealRegularSolidHarmonics(self%lmax)
        call bigben%split("Calculate energy")
        total_box_count = self%parallel_info%get_total_box_count()

        ! allocate multipole moments, total_box_count is the number of boxes
        ! in all levels
        allocate(multipole_moments((self%lmax + 1) ** 2, total_box_count))
        
        
        ! set all energy variables to 0
        total_interaction_energy = 0.0d0
        farfield_interaction_energy = 0.0d0
        nearfield_interaction_energy = 0.0d0

        call bigben%split("Calculate energy: Local farfield")
        call bigben%split("Energy: Multipole moments")
        domain = self%parallel_info%get_domain(self%maxlevel)
        offset = self%parallel_info%get_level_offset(self%maxlevel)

        farfield_interaction_energy = 0.0d0

        ! The cube multipole evaluation
        do ibox = domain(1), domain(2)
            ! evaluate box center and cell limits (when size of a cell is (1,1,1))
            box_center = self%parallel_info%get_box_center(ibox, self%maxlevel)
            box_limits = self%parallel_info%get_box_cell_index_limits(ibox, self%maxlevel, global = .FALSE.)

            ! Calculate multipole moments at each box center 
            multipole_moments(:,offset + ibox) = f2%cube_multipoles(self%lmax, box_center, &
                                                 box_limits)
        enddo
       
        ! if downward pass is not made in the potential calculation, multipole 
        ! moments in all levels must be calculated
        if (.not. self%downward_pass) then
            ! send the own contributions of multipole moments to neighbor processors
            ! and receive the needed contributions from neighbors
            call self%parallel_info%communicate_matrix(multipole_moments, self%maxlevel)

            ! Translate multipoles to coarser level
            do ilevel = self%maxlevel-1, self%parallel_info%start_level, -1
                call self%translate_multipoles(multipole_moments, ilevel)

                ! Make sure that each processor has the correct multipole moments
                ! for the next steps
                call self%parallel_info%communicate_matrix(multipole_moments, ilevel)
            end do

            ! set the first level for the energy calculation
            first_level = 2
        else
            ! if downward pass has been performed, no other energy contributions
            ! must be calculated than the maximum level
            first_level = self%maxlevel
        end if 
        
        call bigben%stop()
        
        ! the first level depends if the downward pass has been executed
        ! if not, first_level = 2, if yes first_level = self%maxlevel
        call bigben%split("Dot product")
        do ilevel = first_level, self%maxlevel
            domain = self%parallel_info%get_domain(ilevel)
            offset = self%parallel_info%get_level_offset(ilevel)
            ! The far-field interaction energy contribution
            do ibox = domain(1), domain(2)
                farfield_interaction_energy = farfield_interaction_energy &
                                          + dot_product(self%farfield_potential(:,offset + ibox),&
                                          multipole_moments(:, offset + ibox))
            enddo
        end do
        call bigben%stop()
        ! stop local farfield timing
        call bigben%stop()
        deallocate(multipole_moments)
        !------------------------------------------------!
        ! Evaluate the interaction energy at near-field  !
        !------------------------------------------------!
 

        ! U_{ab}^(nn(P)) = \int \rho_b^{(P)} (r) V_a^{(P)} (r) d^3 r
        ! Get the indices of the boxes the current processor handles
        ! and store them to ista and iend
         
        domain = self%parallel_info%get_domain(self%maxlevel)
        call self%multiply_bubbles(f2, result_bubbles)
        !write(*, '("Bubbles-energy: ", f16.10 )') result_bubbles%integrate() 

        call bigben%split("Nearfield") 
        do ibox = domain(1), domain(2)
            !call cpu_time(tic)
            ! get the starting and ending cells indexes of the box
            box_limits = self%parallel_info%get_box_cell_index_limits(ibox, self%maxlevel, global = .FALSE.)
            !box_bubbles = self%bubbles%get_sub_bubbles(self%parallel_info%get_box_limits(ibox))
            box_potential = self%nearfield_potential(ibox)
            !box_potential%cube = box_potential%cube +  &
            !    self%evaluate_farfield_potential_grid(solid_harmonics, &
            !                                          box_potential%grid, &
            !                                          ibox)
            
            
            ! do multiplication only for the density inside box_limits
            ! At this step, before multiplication, we append new bubbles for the
            ! far-field potential. One bubble per local far-field box
            ! The multiplication is carrided out exactly as it is here
            !if (box_bubbles%get_nbub() > 0) then
            !    energy_density_ibox = self%nearfield_potential(ibox)%multiply(f2, box_limits, &
            !                              only_cube = .FALSE.)
            !else
                energy_density_ibox = box_potential%multiply(f2, box_limits, &
                                           result_bubbles = result_bubbles)
            !end if
            call box_potential%destroy()
            ! Now the energy density should be integrated SELECTIVELY,
            ! Only the cube and e.g. those bubbles whose centers are within this
            ! box
            nearfield_interaction_energy = nearfield_interaction_energy &
                 + energy_density_ibox%integrate(only_bubbles_within_cube = .TRUE.)

            call energy_density_ibox%destroy()
            !call cpu_time(toc)
            !call box_bubbles%destroy()
            !print*, "Box number", ibox, "E:", nearfield_interaction_energy, &
            !         "time:", toc-tic, "nearfield sizes", shape(self%nearfield_potential(ibox)%cube), result_bubbles%get_lmax()
        enddo
        call bigben%stop()
        call bigben%stop()

        !write (*, '("FF-energy", f16.10, ", NF-energy:", f16.10)') farfield_interaction_energy, nearfield_interaction_energy
        total_interaction_energy = nearfield_interaction_energy !+ farfield_interaction_energy
        call self%parallel_info%sum_real(total_interaction_energy)



    end function

    ! If a multipole moment is known with respect to some center, the
    ! corresponding moment in some other center can be known without recomputing the multipole
    ! integral.
    ! 
    ! That is, one can translate multipole moments between expansion centers. 
    ! This transformation is most compactly expressed as a matrix-vector product
    !
    !  q(N) = W(O-N) q(O)
    !
    !> This function translates previously calculated multipoles to the parent level 
    subroutine GBFMMFunction3D_translate_multipoles(self, multipole_moments, numlevel)
        class(GBFMMFunction3D)      :: self
        !> multipole moments in a matrix that contains has (self%lmax + 1)**2 rows (first index)
        !! and the number of levels on all boxes in the columns
        real(REAL64), intent(inout) :: multipole_moments(:, :)   
        !> number of level that is translated  
        integer, intent(in)         :: numlevel

        type(TranslationMatrix)     :: translation_matrix
        ! Temporary variable to store child multipole moments
        real(REAL64)                :: child_multipole_moments((self%lmax + 1)**2)

        integer                     :: ibox
        integer                     :: child_box
        integer                     :: parent_box
        integer                     :: domain(2)

        real(REAL64)                :: child_position(3), child_size(3)
        real(REAL64)                :: parent_position(3), parent_size(3)

        translation_matrix = TranslationMatrix(self%lmax)

        ! starting and ending indices of this processors domain
        domain = self%parallel_info%get_domain(numlevel + 1)
         
        do ibox = domain(1), domain(2)
            ! the position of the center of the child box
            child_position = self%parallel_info%get_box_center(ibox, numlevel + 1) 
            ! the position of the center of the parent box
            parent_position = self%parallel_info%get_box_center( &
                              self%parallel_info%get_parent_index(ibox, numlevel + 1), numlevel)        

            ! child box index with offset
            child_box = self%parallel_info%get_level_offset(numlevel + 1) + ibox
            ! parent box index with offset
            parent_box = self%parallel_info%get_level_offset(numlevel)  &
                         + self%parallel_info%get_parent_index(ibox, numlevel + 1)

            ! copy existing child box multipole moment to the variable
            child_multipole_moments = multipole_moments(:, child_box)
    
            ! translate the potential of the child box to the position
            ! of the parent box
            call translation_matrix%apply(child_multipole_moments, from=child_position,  &
                                          to=parent_position)
            ! add the value of the child box multipole moment to the parent box
            ! multipole moment   
            multipole_moments(:, parent_box) = multipole_moments(:,parent_box) &
                                                    + child_multipole_moments

        enddo
        call translation_matrix%destroy()
        return
    end subroutine


    !> Deallocates the memory of GBFMMFunction3D
    subroutine GBFMMFunction3D_destroy(self)
        class(GBFMMFunction3D), intent(inout) :: self
        ! index of the box
        integer                          :: ibox, domain(2)

        call self%destroy_cube()
        nullify(self%parallelization_info)
        nullify(self%grid)
        call self%bubbles%destroy()
        if (allocated(self%farfield_potential)) deallocate(self%farfield_potential)

        if (allocated(self%bubbles_farfield_potential)) deallocate(self%bubbles_farfield_potential)
        if (allocated(self%cube_contaminants)) deallocate(self%cube_contaminants)
        if (allocated(self%nearfield_potential)) then
            
            domain = self%parallel_info%get_domain(self%parallel_info%maxlevel)
            do ibox = domain(1), domain(2)
                call self%nearfield_potential(ibox)%destroy()
            end do
            deallocate(self%nearfield_potential)
        end if
        
        if(allocated(self%taylor_series_bubbles)) then
            call self%taylor_series_bubbles%destroy()
            deallocate(self%taylor_series_bubbles)
        end if
        nullify(self%parallel_info)
        

    end subroutine

!------------------------------------------------------------------
!   Below: Deprecated functions used by old routines              !
!------------------------------------------------------------------

    !> get x,y,z starting and ending cell integer coordinates of box with number 'ibox',
    !! i.e., coordinates when cell size is (1, 1, 1)
    function get_box_cell_index_limits(box_size, ibox) result(limits)
        !> size of a box as number of cells
        integer,             intent(in)     :: box_size(3)
        !> order number of box
        integer,             intent(in)     :: ibox
        
        ! box position as integers
        integer                             :: box_vector(3)
        ! result matrix
        integer                             :: limits(2, 3)
        ! morton indexes begin from 0 and box vectors begin from (0,0,0)
        box_vector = unmorton(ibox-1)
      
        
        limits(1, :) = (box_vector  * box_size)
        
        limits(2, :) = limits(1, :) + box_size
        limits(1, :) = limits(1, :) + (/1, 1, 1/)
    end function

    
    !> get box center x,y,z coordinates
    ! deprecated, functions of GBFMMParallelInfo should be used
    function get_box_center(box_size, ibox, grid) result(center)
        integer,             intent(in)     :: box_size(3)
        integer,             intent(in)     :: ibox
        type(Grid3D),        intent(in)     :: grid
        
        integer                             :: box_vector(3)
        ! index of the cell
        integer                             :: icell
        
        ! center of the box, (returned)
        real(REAL64)                        :: center(3)
        ! array to store cell starting points
        real(REAL64), allocatable           :: cell_starts(:)
        
        
        box_vector = unmorton(ibox -1)
        
        
        ! get x-axis cell center
        icell = box_vector(X_) * box_size(X_) + box_size(X_) / 2 + 1
        if (mod(box_size(X_), 2) == 1) then
            center(X_) = grid%axis(X_)%get_cell_center(icell)
        else
            cell_starts =  grid%axis(X_)%get_cell_starts()
            center(X_) = cell_starts(icell)
        end if 

        
        icell = box_vector(Y_) * box_size(Y_) + box_size(Y_) / 2 + 1
        ! get y-axis cell center 
        if (mod(box_size(Y_), 2) == 1) then
            center(Y_) = grid%axis(Y_)%get_cell_center(icell)
        else
            icell = box_vector(Y_) * box_size(Y_) + box_size(Y_) / 2 + 1
            cell_starts =  grid%axis(Y_)%get_cell_starts()
            center(Y_) = cell_starts(icell)
        end if 

        
        icell = box_vector(Z_) * box_size(Z_) + box_size(Z_) / 2 + 1
        ! get z-axis cell center 
        if (mod(box_size(Z_), 2) == 1) then
            center(Z_) = grid%axis(Z_)%get_cell_center(icell)
        else
            cell_starts = grid%axis(Z_)%get_cell_starts()
            center(Z_) = cell_starts(icell)
        end if 
    end function

    ! deprecated, functions of GBFMMParallelInfo should be used
    subroutine para_range(first_box, last_box, nproc, irank, ista, iend)
        integer(INT32), intent(in)  :: first_box, last_box 
        integer(INT32), intent(out) :: ista, iend 
        integer(INT32), intent(in)  :: irank ! iproc (rank)
        integer(INT32), intent(in)  :: nproc ! # cores
        integer :: i, j
        
        i = ( last_box - first_box + 1 ) / nproc 
        j = mod( last_box - first_box + 1, nproc)

        ! first box that belongs to processor irank
        ista = irank * i + first_box + min(irank, j) 
        ! last box that belongs to processor irank
        iend = ista + i - 1 
        if ( j > irank) iend = iend + 1
        return
    end subroutine

    

   

    ! Returns 3D box index from box index, 
    ! deprecated, functions of GBFMMParallelInfo should be used
    function unmorton(boxIndex) 
        integer, intent(in) :: boxIndex
        integer :: unmorton(3)
        integer :: i, j, k, n, mortonIndex3D(0:2)

        mortonIndex3D = 0
        n = boxIndex
        k = 0
        i = 0
        do while( n /= 0 ) 
            j = 2 - k
            mortonIndex3D(j) = mortonIndex3D(j) + mod(n,2)*ishft(1, i)
            n = n/2
            k = mod((k+1),3)
            if( k == 0 ) i = i + 1 
        enddo
        unmorton(1) = mortonIndex3D(1)
        unmorton(2) = mortonIndex3D(2)
        unmorton(3) = mortonIndex3D(0)

    end function

    ! Generate 1D index from integer indices at any level
    ! deprecated, functions of GBFMMParallelInfo should be used
    function morton(boxIndex3D, numLevel)
        integer, intent(in) :: boxIndex3D(3)
        integer, intent(in) :: numLevel
        integer             :: boxIndex3D_tmp(3)
        integer :: i,nx,ny,nz
        integer :: morton
        morton = 0 
        boxIndex3D_tmp = boxIndex3D
        do i=0, numLevel-1
            nx = mod(boxIndex3D_tmp(1),2)
            boxIndex3D_tmp(1) = ishft(boxIndex3D_tmp(1),-1)
            morton = morton + nx*ishft(1, (3*i+1))

            ny = mod(boxIndex3D_tmp(2),2)
            boxIndex3D_tmp(2) = ishft(boxIndex3D_tmp(2),-1)
            morton = morton + ny*ishft(1,(3*i))

            nz = mod(boxIndex3D_tmp(3),2)
            boxIndex3D_tmp(3) = ishft(boxIndex3D_tmp(3), -1) 
            morton = morton + nz*ishft(1, (3*i+2))
        enddo

        ! C to Fortran indexing
        morton = morton + 1
    end function


end module