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
module DIIS_class
    use Function3D_class
    use SCF_class
    use SCFCycle_class
    use xmatrix_m
    use Globals_m

    implicit none
    public DIIS, HelmholtzDIIS, LCMODIIS
 
    private

!--------------------------------------------------------------------!
!        DIIS interface definition                                   !
!--------------------------------------------------------------------!
! Abstract type to define all Direct Inversion of the Iterative      !
! Subspace (and similar, namely, KAIN) methods.                                           !
!                                                                    !
! The main functionality is the 'optimize' function, which fulfills  !
! the SCFOptimizer - interface.                                      !
!--------------------------------------------------------------------! 

    type, abstract, extends(SCFOptimizer) :: DIIS
        !> DIIS 'B' matrix
        real(REAL64), allocatable     :: diis_matrix(:, :) 
        !> DIIS vector
        real(REAL64), allocatable     :: diis_vector(:)
        !> The maximum number of guesses used 
        integer                       :: needed_guess_count = 1
        !> Number of guesses needed for evaluation
        integer                       :: used_guess_count = 4
        !> The error threshold at which the DIIS can start
        real(REAL64)                  :: initialization_threshold = -5.0d0
        integer                       :: guess_count = 0
        integer                       :: next_index = 1
   contains 
        procedure(update_data), private, deferred   :: update_data
        procedure(update),      private, deferred   :: update
        
        procedure, private                          :: get_new_coefficients => &
                                                           DIIS_get_new_coefficients 
        procedure, public                           :: optimize             => &
                                                           DIIS_optimize
        procedure, private                          :: get_first_guess_index => &
                                                           DIIS_get_first_guess_index
        procedure, private                          :: get_previous_guess_index => &
                                                           DIIS_get_previous_guess_index
        procedure, private                          :: get_next_guess_index => &
                                                           DIIS_get_next_guess_index
        procedure, private                          :: store_update => &
                                                           DIIS_store_update
        procedure, private                          :: store_guess => &
                                                           DIIS_store_guess
   end type 

   abstract interface 
       subroutine update_data(self)
           import DIIS
           class(DIIS), intent(inout)       :: self
       end subroutine

       subroutine update(self, coefficients)
           import DIIS, REAL64
           class(DIIS), intent(inout), target :: self
           real(REAL64), intent(in)           :: coefficients(:)
       end subroutine

   end interface


!--------------------------------------------------------------------!
!        Helmholtz DIIS definition                                   !
!--------------------------------------------------------------------!

   type, extends(DIIS) :: HelmholtzDIIS
        !> Matrix to store previous guesses
        type(Function3D), allocatable         :: previous_guesses(:, :)
        !> Matrix to store the updates of previous guesses
        type(Function3D), allocatable         :: previous_updates(:, :)
        !> Matrix to store the energy updates of previous guesses
        real(REAL64), allocatable             :: previous_energy_updates(:, :)
        !> 
        real(REAL64), allocatable             :: previous_energies(:, :)
   contains 
        procedure, private                    :: update_data => HelmholtzDIIS_update_data
        procedure, private                    :: update      => HelmholtzDIIS_update
        procedure, public                     :: destroy     => HelmholtzDIIS_destroy
        procedure, private                    :: store_guess => HelmholtzDIIS_store_guess
   end type

   interface HelmholtzDIIS
        module procedure              :: HelmholtzDIIS_init
    end interface

!--------------------------------------------------------------------!
!        LCMO DIIS definition                                        !
!--------------------------------------------------------------------!
        
   type, extends(DIIS) :: LCMODIIS
        !> Matrix to store hamiltonian matrices of guesses
        real(REAL64), allocatable        :: hamiltonian_matrices(:, :, :)
        !> Error matrices (number of orbitals, number of orbitals, number of guesses)
        real(REAL64), allocatable        :: error_matrices(:, :, :)
        !> Pointer to the actual scf_cycle with the lcmo type
        class(LCMOSCFCycle), allocatable :: lcmo_scf_cycle
    contains
        procedure, private                    :: update_data      => LCMODIIS_update_data
        procedure, private                    :: update           => LCMODIIS_update
        procedure, private                    :: get_error_matrix => LCMODIIS_get_error_matrix
        procedure, public                     :: destroy          => LCMODIIS_destroy
    end type


    interface LCMODIIS
        module procedure              :: LCMODIIS_init
    end interface

contains

!--------------------------------------------------------------------!
!        DIIS interface functions and subroutines
!--------------------------------------------------------------------!    


    !> Get the index of the oldest guess still stored to the results
    pure function DIIS_get_first_guess_index(self) result(first_guess_index)
        class(DIIS), intent(in)           :: self
        integer                           :: first_guess_index
        if (self%guess_count == self%used_guess_count) then
             first_guess_index = self%next_index
        else
            first_guess_index = 1
        end if
    end function 

    
    pure function DIIS_get_previous_guess_index(self, guess_index) result(previous_index)
        class(DIIS), intent(in)           :: self
        integer, intent(in)               :: guess_index
        integer                           :: first_guess_index, previous_index

        
        if (guess_index == 1) then
            if (self%guess_count == self%used_guess_count) then
                previous_index = self%used_guess_count
            else
                previous_index = -1
            end if
        else
            previous_index = guess_index - 1 
         end if
    end function 

    pure function DIIS_get_next_guess_index(self, guess_index) result(next_index)
        class(DIIS), intent(in)           :: self
        integer, intent(in)               :: guess_index
        integer                           :: first_guess_index, next_index

        if (guess_index == self%used_guess_count) then
            if (self%next_index == 1) then
                next_index = -1
            else
                next_index = 1
            end if
        else if (guess_index == self%next_index-1) then
            next_index = -1
        else if (guess_index == self%guess_count) then
            next_index = -1
        else
            next_index = guess_index + 1 
        end if
    end function 

    subroutine DIIS_optimize(self)
        class(DIIS)                      :: self
        real(REAL64), allocatable        :: coefficients(:)
        real(REAL64)                     :: energy
        logical                          :: initialized 
        integer                          :: iteration_number
 
        initialized = .FALSE.
        iteration_number = 2
        energy = 0.0d0

        print *, "-------------------------------------"
        print *, "Calculating DIIS initial guess energy"
        print *, "-------------------------------------"
        call self%scf_cycle%calculate_hamiltonian_matrix(.FALSE.)
        call self%update_data()
        call self%store_guess()
        
        do 
            call self%scf_cycle%update(1.0d0)
            call self%store_update()
            call self%scf_cycle%gs_orthogonalize_orbitals()
            print *, "-------------------------------------"
            print *, "Starting DIIS iteration number", iteration_number
            print *, "-------------------------------------"
            call self%scf_cycle%calculate_hamiltonian_matrix(.FALSE.)
            ! update the  diis matrix (B), error, and other
            ! method specific data
            call self%update_data()

            print *, "Error is", self%error, "Initialization threshold", self%initialization_threshold, &
                     "Convergence threshold", self%total_energy_convergence_threshold
            if (self%is_converged()) then
                print *, " ** DIIS converged **"
                return
            else if (initialized .or. ( &
                     self%error < self%initialization_threshold .and. & 
                    self%guess_count >= self%needed_guess_count)) then
                if (.not. initialized .and. self%error < self%initialization_threshold & 
                    .and. self%guess_count >= self%needed_guess_count) then
                    print *, " ** Reached initialization threshold **"
                    initialized = .TRUE.
                end if 
                print *, "guess count", self%guess_count
                allocate(coefficients(self%guess_count))
                ! get new coefficients by doing DIIS
                coefficients = self%get_new_coefficients()

                ! update (hamiltonian matrix / orbitals) using 
                ! the diis method with calculated coefficients
                ! the update method should contain store_guess at the
                ! end
                print *, "Updating via DIIS"
                call self%update(coefficients)
                deallocate(coefficients)
            else 
                ! update (hamiltonian matrix/orbitals) with scf cycle
                print *, "Updating via SCF Cycle"
                call self%store_guess()
            end if
            print *, "Energy is", self%scf_cycle%energy

            if (iteration_number == self%max_iterations) then
                print *, "Reached DIIS iteration limit:", self%max_iterations
                return
            end if  
            write(*, '("-----------------------------------------------------------------------")')
            write(*, '("in diis")')
            write(*, '("Iteration ",i4," completed")'), iteration_number
            write(*, '("Total energy:      ", f24.16,"")'), self%scf_cycle%energy + self%nuc_rep_energy
            write(*, '("Electronic energy: ", f24.16,"")'), self%scf_cycle%energy
            write(*, '("Nuclear repulsion: ", f24.16,"")'), self%nuc_rep_energy
            write(*, '("Energy change:     ", f24.16,"")'),  self%scf_cycle%energy - energy
            write(*, '("-----------------------------------------------------------------------")')
            energy = self%scf_cycle%energy
            iteration_number = iteration_number + 1
        end do
        
    end subroutine

    
    function DIIS_get_new_coefficients(self) result(result_coefficients)
        class(DIIS), intent(in)       :: self
        real(REAL64), allocatable     :: matrix(:, :)
        real(REAL64), allocatable     :: vector(:), &
                                         coefficients(:), matrix2(:, :), result_coefficients(:)
        integer                       :: current_guess_index, iguess, i, jguess, j, previous_guess_index
        
        ! get the index of the previous guess (not included in the matrix and vector)
        previous_guess_index = self%get_previous_guess_index(self%next_index)

        allocate(matrix(self%guess_count-1, self%guess_count-1))
        allocate(vector(self%guess_count-1))
        i = 1
        do iguess = 1, self%guess_count
            if (iguess /= previous_guess_index) then
                j = 1
                do jguess = 1, self%guess_count
                    if (jguess /= previous_guess_index) then
                        matrix(i, j) = self%diis_matrix(iguess, jguess) 
                        j = j+1
                    end if
                end do
                vector(i) = self%diis_vector(iguess)
                i = i + 1
            end if
        end do
        !vector(self%guess_count) = -1
        print *, "Matrix", matrix
        print *, "diis vector", vector
        allocate(coefficients(self%guess_count-1))
        
        ! calculate the multipliers for each 
        allocate(matrix2(self%guess_count-1, self%guess_count-1))
        
        matrix2 = matrix_inverse(matrix)
        print *, "size mat", shape(matrix2), "2", shape(matrix), "shape vec", shape(vector), &
                 "shape coeffs", shape(coefficients)
        coefficients = xmatmul(matrix2, vector) 
        deallocate(matrix2)
        deallocate(matrix)
        deallocate(vector)
        
        allocate(result_coefficients(self%guess_count))
        
        i = 1
        do iguess = 1, self%guess_count
            if (iguess /= previous_guess_index) then
                result_coefficients(iguess) = coefficients(i)
                i = i + 1
            else
                result_coefficients(iguess) = 0.0d0
            end if
        end do

        ! remove the lambda parameter from the coefficient vector
        !coefficients = coeff
        print *, "Result coefficients are", result_coefficients
        deallocate(coefficients)
    end function 
    
    subroutine DIIS_store_update(self)
        class(DIIS), intent(inout)             :: self
    end subroutine

    subroutine DIIS_store_guess(self)
        class(DIIS), intent(inout)             :: self
    end subroutine

    

!--------------------------------------------------------------------!
!        Helmholtz DIIS functions & subroutines
!--------------------------------------------------------------------!

    function HelmholtzDIIS_init(scf_cycle, needed_guess_count, used_guess_count, &
             total_energy_convergence_threshold, eigenvalue_convergence_threshold, &
             initialization_threshold, maximum_iterations) result(new)
        class(HelmholtzSCFCycle), target   :: scf_cycle
        !> Number of guesses needed for evaluation
        integer, intent(in), optional      :: needed_guess_count
        !> The maximum number of guesses used 
        integer, intent(in), optional      :: used_guess_count
        !> The threshold at which the DIIS has been converged
        real(REAL64), intent(in), optional :: total_energy_convergence_threshold
        !> The threshold at which the DIIS has been converged
        real(REAL64), intent(in), optional :: eigenvalue_convergence_threshold
        !> The error threshold at which the DIIS process starts
        real(REAL64), intent(in), optional :: initialization_threshold
        !> Maximum iteration count
        integer, intent(in), optional      :: maximum_iterations
        type(HelmholtzDIIS), target        :: new

        if (present(needed_guess_count)) then
            new%needed_guess_count = needed_guess_count
        end if 

        if (present(used_guess_count)) then
            new%used_guess_count = used_guess_count
        end if 

        if (present(total_energy_convergence_threshold)) then
            new%total_energy_convergence_threshold = total_energy_convergence_threshold
        end if 
        
        if (present(eigenvalue_convergence_threshold)) then
            new%eigenvalue_convergence_threshold = eigenvalue_convergence_threshold
        end if 
        
        if (present(initialization_threshold)) then
            new%initialization_threshold = initialization_threshold
        end if 

        if (present(maximum_iterations)) then
            new%max_iterations = maximum_iterations
        end if
        
        new%scf_cycle => scf_cycle
        new%orbital_count = new%scf_cycle%total_orbital_count
        
        allocate(new%previous_guesses(new%orbital_count, new%used_guess_count))
        allocate(new%previous_updates(new%orbital_count, new%used_guess_count))
        allocate(new%previous_energies(new%orbital_count, new%used_guess_count), source = 0.0d0)
        allocate(new%previous_energy_updates(new%orbital_count, new%used_guess_count), source = 0.0d0)
        allocate(new%diis_matrix(new%used_guess_count, new%used_guess_count), source = 0.0d0)
        allocate(new%diis_vector(new%used_guess_count), source = 0.0d0)
    end function


    

    !> Update the orbitals and corresponding eigen values according to 
    subroutine HelmholtzDIIS_update(self, coefficients)
        class(HelmholtzDIIS), target, intent(inout)    :: self
        real(REAL64), intent(in)                       :: coefficients(:)
        integer                                        :: iorbital, iguess
        type(Function3D), pointer                      :: update_n, update_i
        type(Function3D)                               :: orbital_update, temp
        type(Function3D), allocatable                  :: new_orbitals(:)
        type(Function3DPointer), pointer               :: latest_guess(:)
        real(REAL64)                                   :: latest_energies(self%orbital_count), &
                                                          e_difference(self%orbital_count), &
                                                          e_update_n(self%orbital_count), &
                                                          e_update_i, normalization_factor, &
                                                          energy_update, new_eigen_values(self%orbital_count)
        integer                                        :: current_guess_index, previous_guess_index, i, next_guess_index

        ! get the latest guess from the scf cycle and corresponding energies
        latest_guess => self%scf_cycle%all_orbitals
        latest_energies = self%scf_cycle%get_orbital_energies()


        ! get the index of previous completed iteration
        previous_guess_index = self%get_previous_guess_index(self%next_index)

        ! Get linear combination of the previous guesses multiplied with
        ! the calculated coefficients and add the contribution to the existing new orbitals
        ! obtained from the Helmholtz scf cycle.
        ! NOTE: at this point the guess_count has not been updated, but is updated after
        ! the loop
        ! calculate the changes to the orbitals
        allocate(new_orbitals(self%orbital_count))
        new_eigen_values = latest_energies
        do iorbital = 1, self%orbital_count
            update_n => self%previous_updates(iorbital, previous_guess_index)
            e_update_n = self%previous_energy_updates(iorbital, previous_guess_index)
                    
            call orbital_update%init_copy(latest_guess(iorbital)%p, copy_content = .FALSE.)
            orbital_update = 0.0d0
            energy_update = 0.0d0
            do iguess = 1, self%guess_count

                if (previous_guess_index /= iguess) then

                    ! calculate the change in orbital energies between j:th iteration and the 
                    ! j-1:th iteration
                    ! and calculate the difference between the orbital i at the latest guess n and
                    ! the guess (n - 1) 

                    next_guess_index = self%get_next_guess_index(iguess)
                    update_i   => self%previous_updates(iorbital, iguess)
                    e_update_i =  self%previous_energy_updates(iorbital, iguess)        
                    
                    ! update the orbital energies to the scf
                    energy_update = energy_update + &
                        coefficients(iguess) * (                 &
                            ( self%previous_energies(iorbital, iguess)  &
                            - self%previous_energies(iorbital, previous_guess_index) ) &
                            + (e_update_i - e_update_n(iorbital)) ) 

                    print *, "energy  update", energy_update

                    call temp%init_copy(update_i, copy_content = .TRUE.)

                    ! get f(x^i) - f(x^n) 
                    call temp%subtract_in_place(update_n)

                    ! calculate the difference between the orbital i at guess i and
                    ! the second latest guess (n - 1), i.e., get x^i - x^n
                    call temp%add_in_place(self%previous_guesses(iorbital, iguess))
                    call temp%subtract_in_place(self%previous_guesses( iorbital, previous_guess_index ))
                    call temp%product_in_place_REAL64(coefficients(iguess))
                    
                    ! finally update the 
                    call orbital_update%add_in_place(temp)
                    call temp%destroy()
                end if
            end do
            call new_orbitals(iorbital)%init_copy(latest_guess(iorbital)%p, copy_content = .TRUE.)
            call new_orbitals(iorbital)%add_in_place(orbital_update)
            call orbital_update%destroy()
            normalization_factor = 1.0d0 / (new_orbitals(iorbital) .dot. new_orbitals(iorbital))
            call new_orbitals(iorbital)%product_in_place_REAL64(sqrt(normalization_factor))
            new_eigen_values(iorbital) = new_eigen_values(iorbital) + energy_update
            print *, "new eigen value", new_eigen_values(iorbital), &
                     "energy update", energy_update, &
                     "normalization factor", normalization_factor
        end do
        nullify(latest_guess)
        call self%store_guess()

        do iorbital = 1, self%orbital_count
            call self%scf_cycle%all_orbitals(iorbital)%p%copy_content(new_orbitals(iorbital))
            call new_orbitals(iorbital)%destroy()
        end do
        call self%scf_cycle%set_orbital_energies(new_eigen_values)
        deallocate(new_orbitals)

    end subroutine

    subroutine HelmholtzDIIS_store_guess(self)
        class(HelmholtzDIIS), intent(inout) :: self
        type(Function3DPointer), pointer    :: latest_guess(:)
        real(REAL64)                        :: latest_energies(self%orbital_count)
        integer                             :: iorbital

        latest_guess => self%scf_cycle%all_orbitals
        latest_energies = self%scf_cycle%get_orbital_energies()

        ! update the stored guess count
        if (self%guess_count < self%used_guess_count) then
            self%guess_count = self%guess_count + 1 
        end if 

        !print *, "Storing the guess to index", self%next_index
        ! destroy the function to avoid memory problems, and save the latest guess
        ! to previous guesses
        do iorbital = 1, self%orbital_count
            call self%previous_guesses(iorbital, self%next_index)%destroy()
            call self%previous_guesses(iorbital, self%next_index)%init_copy(latest_guess(iorbital)%p, &
                                                                            copy_content = .TRUE.)
            self%previous_energies(iorbital, self%next_index) = latest_energies(iorbital)
        end do
        nullify(latest_guess)
        !print *, "Increasing next index to", self%next_index 
    end subroutine
    
    subroutine HelmholtzDIIS_store_update(self)
        class(HelmholtzDIIS), intent(inout) :: self
        type(Function3DPointer), pointer    :: latest_guess(:)
        real(REAL64)                        :: latest_energies(self%orbital_count)
        integer                             :: iorbital

        latest_guess => self%scf_cycle%all_orbitals
        latest_energies = self%scf_cycle%get_orbital_energies()

        ! update the stored guess count
        if (self%guess_count < self%used_guess_count) then
            self%guess_count = self%guess_count + 1 
        end if 

        !print *, "Storing the guess to index", self%next_index
        ! destroy the function to avoid memory problems, and save the latest guess
        ! to previous guesses
        do iorbital = 1, self%orbital_count
            call self%previous_updates(iorbital, self%next_index)%destroy()
            call self%previous_updates(iorbital, self%next_index)%init_copy(latest_guess(iorbital)%p, &
                                                                            copy_content = .TRUE.)
            call self%previous_updates(iorbital, self%next_index)%subtract_in_place( &
               self%previous_guesses(iorbital, self%next_index))
            self%previous_energy_updates(iorbital, self%next_index) = latest_energies(iorbital) - &
                self%previous_energies(iorbital, self%next_index)
        end do

        ! get the index of the function that will be replaced the next time
        ! this function is called
        if (self%next_index == self%used_guess_count) then
            self%next_index = 1
        else
            self%next_index = self%next_index + 1
        end if
        nullify(latest_guess)
        !print *, "Increasing next index to", self%next_index 
    end subroutine

    ! Evaluate x^n = (\phi^n, E^n), f(x^n) = (\Delta \phi^n, \Delta E^n)
    ! and evaluate and store 
    !    A_ij =  < x^n - x^i | f(x^n) - f(x^j) > and
    !    b_i  =  < x^n - x^i | f(x^n) >, 
    ! where i and j are the numbers of the guesses.
    ! (A and b are stored to self%diis_matrix and self%diis_vector, respectively)
    subroutine HelmholtzDIIS_update_data(self)
        class(HelmholtzDIIS), intent(inout) :: self
        integer                             :: iguess, jguess, iorbital
        type(Function3DPointer), pointer    :: latest_guess(:)
        type(Function3D)                    :: update_j, update_n, temp, difference_in
        real(REAL64)                        :: latest_energies(self%orbital_count), &
                                               e_difference(self%orbital_count), &
                                               e_update_n(self%orbital_count), &
                                               e_update_j(self%orbital_count), error
        integer                             :: previous_guess_index, next_guess_index

        
        ! get the latest guess from the scf cycle and corresponding energies
        latest_guess => self%scf_cycle%all_orbitals
        latest_energies = self%scf_cycle%get_orbital_energies()

        
        ! calculate the change in orbital energies between current iteration
        ! and the previous iteration, i.e., \Delta E
        previous_guess_index = self%get_previous_guess_index(self%next_index)
        e_update_n = self%previous_energy_updates(:, previous_guess_index) 

        ! init the error
        if (self%guess_count == 0) then
            self%error = 1.0d0
        else
            self%error = 0.0d0
        end if
 
        ! go through all previous guesses
        do iguess = 1, self%guess_count
            ! calculate the change in orbital energies between previous iteration n
            ! and the ith iteration
            e_difference = self%previous_energies(:, previous_guess_index) - self%previous_energies(:, iguess) 

            ! calculate <e_n - e_i | de_n>
            self%diis_vector(iguess) = dot_product(e_difference, e_update_n)

            do jguess = 1, self%guess_count
                ! get the change in orbital energy at j:th iteration
                e_update_j = self%previous_energy_updates(:, jguess)

                ! calculate < e_n - e_i | de_n - de_j > 
                self%diis_matrix(iguess, jguess) = dot_product(e_difference, e_update_n - e_update_j)

            end do
            
            do iorbital = 1, self%orbital_count
                !print *, "iorbital", iorbital
                ! calculate the difference between the orbital i at the latest guess n and
                ! the guess (n - 1), i.e., \Delta phi_i
                call update_n%init_copy(latest_guess(iorbital)%p, copy_content = .TRUE.)
                call update_n%subtract_in_place(self%previous_guesses(iorbital, previous_guess_index))

                ! calculate <phi_n - phi_i | dphi_n>
                self%diis_vector(iguess) =   &
                    self%diis_vector(iguess) &
                     + (update_n .dot. self%previous_updates(iorbital, previous_guess_index))

                do jguess = 1, self%guess_count
                    ! check if the guess is the newest guess _stored_
                    next_guess_index = self%get_next_guess_index(jguess)
                    !print *, "next guess", next_guess_index
                    if (next_guess_index == -1) then
                        call update_j%init_copy(update_n, copy_content = .TRUE.)
                    else
                        call update_j%init_copy(self%previous_guesses(iorbital, next_guess_index), copy_content = .TRUE.)
                        call update_j%subtract_in_place(self%previous_guesses(iorbital, jguess))
                    end if

                    !print *, "have update j", next_guess_index
                    ! calculate the dot product between 'difference' and the difference between
                    ! changes  at step j and n, i.e., < phi_n - phi_i | dphi_n - dphi_j >
                    call temp%init_copy(update_n, copy_content = .TRUE.)
                    call temp%subtract_in_place(update_j)
                    call update_j%destroy()
                    self%diis_matrix(iguess, jguess) = self%diis_matrix(iguess, jguess) + &
                        (difference_in .dot. temp)
                    call temp%destroy()
                end do
                call update_n%destroy()
                call difference_in%destroy()
            end do
        end do
        nullify(latest_guess)
        print *, "Updated data"
    end subroutine 

    


     subroutine HelmholtzDIIS_destroy(self)
         class(HelmholtzDIIS)           :: self
         integer               :: iguess, ifunction
         if (allocated(self%previous_guesses)) then
             do iguess = 1, size(self%previous_guesses, 2)
                 do ifunction = 1, size(self%previous_guesses, 1)
                     call self%previous_guesses(ifunction, iguess)%destroy()
                 end do
             end do 
             deallocate(self%previous_guesses)
         end if
         if (allocated(self%diis_matrix)) deallocate(self%diis_matrix)
         if (allocated(self%diis_vector)) deallocate(self%diis_vector)
         if (allocated(self%previous_energies)) deallocate(self%previous_energies)

     end subroutine

!--------------------------------------------------------------------!
!        LCMO DIIS functions & subroutines
!--------------------------------------------------------------------!

    function LCMODIIS_init(scf_cycle, needed_guess_count, used_guess_count, &
             total_energy_convergence_threshold, eigenvalue_convergence_threshold, &
             initialization_threshold) result(new)
        class(LCMOSCFCycle), target   :: scf_cycle
        !> Number of guesses needed for evaluation
        integer, intent(in), optional :: needed_guess_count
        !> The maximum number of guesses used 
        integer, intent(in), optional :: used_guess_count
        !> The threshold at which the DIIS has been converged
        integer, intent(in), optional :: total_energy_convergence_threshold
        !> The threshold at which the DIIS has been converged
        integer, intent(in), optional :: eigenvalue_convergence_threshold
        !> The error threshold at which the DIIS process starts
        integer, intent(in), optional :: initialization_threshold
       
        type(LCMODIIS), target        :: new

        print *, "initing lcmodiis"
        if (present(needed_guess_count)) then
            new%needed_guess_count = needed_guess_count
        end if 

        if (present(used_guess_count)) then
            new%used_guess_count = used_guess_count
        end if 

        if (present(total_energy_convergence_threshold)) then
            new%total_energy_convergence_threshold = total_energy_convergence_threshold
        end if 
        
        if (present(eigenvalue_convergence_threshold)) then
            new%eigenvalue_convergence_threshold = eigenvalue_convergence_threshold
        end if 

        if (present(initialization_threshold)) then
            new%initialization_threshold = initialization_threshold
        end if 

        allocate(new%lcmo_scf_cycle, source = scf_cycle)
        new%scf_cycle => new%lcmo_scf_cycle
        new%orbital_count = new%scf_cycle%total_orbital_count
        
        allocate(new%hamiltonian_matrices(new%orbital_count, new%orbital_count, new%used_guess_count))
        allocate(new%error_matrices(new%orbital_count, new%orbital_count, new%used_guess_count))
        allocate(new%diis_matrix(new%used_guess_count, new%used_guess_count))
        
    end function

    function LCMODIIS_get_error_matrix(self) result(error_matrix)
        class(LCMODIIS),  intent(in)      :: self

        real(REAL64)                      :: error_matrix(self%orbital_count, &
                                                      self%orbital_count)

        

        ! multiplicity * XFDSX - XSDFX
        error_matrix = self%lcmo_scf_cycle%multiplicity * (              &
                xmatmul(self%lcmo_scf_cycle%orthogonalizing_matrix_a,      &
                xmatmul(self%lcmo_scf_cycle%hamiltonian_matrix_a,          &
                xmatmul(self%lcmo_scf_cycle%density_matrix_a,              &
                xmatmul(self%lcmo_scf_cycle%overlap_matrix_a,              &
                        self%lcmo_scf_cycle%orthogonalizing_matrix_a))))   &
            -   xmatmul(self%lcmo_scf_cycle%orthogonalizing_matrix_a,      &
                xmatmul(self%lcmo_scf_cycle%overlap_matrix_a,              &
                xmatmul(self%lcmo_scf_cycle%density_matrix_a,              &
                xmatmul(self%lcmo_scf_cycle%hamiltonian_matrix_a,          &
                        self%lcmo_scf_cycle%orthogonalizing_matrix_a)))))
    end function

     subroutine LCMODIIS_update(self, coefficients)
        class(LCMODIIS), intent(inout), target :: self
        real(REAL64), intent(in)               :: coefficients(:)

        real(REAL64)                     :: new_hamiltonian_matrix           &
                                                (self%orbital_count, self%orbital_count) 
        integer                          :: iguess

        do iguess = 1, self%guess_count
            new_hamiltonian_matrix = new_hamiltonian_matrix + &
                coefficients(iguess) * self%hamiltonian_matrices(:, :, iguess)
        end do
        self%lcmo_scf_cycle%hamiltonian_matrix = new_hamiltonian_matrix
    end subroutine   


    subroutine LCMODIIS_update_data(self) 
        class(LCMODIIS), intent(inout):: self
        real(REAL64)                  :: error_matrix &
            (self%orbital_count, self%orbital_count)
        integer                       :: iguess, previous_guess_index

        ! get error matrix
        error_matrix = self%get_error_matrix() ! self%scf_cycle%hamiltonian_matrix 
        !previous_guess_index = self%get_previous_guess_index(self%next_index)
        !if (previous_guess_index /= 0) then
        !    error_matrix = error_matrix - self%hamiltonian_matrices(:, : ,previous_guess_index)
        !end if
        

        ! update the maximum error
        self%error = maxval(abs(error_matrix))
        print *, "Maximum error", self%error

        ! update the stored guess count
        if (self%guess_count < self%used_guess_count) then
            self%guess_count = self%guess_count + 1 
        end if 
        
        self%error_matrices(:, :, self%next_index) = error_matrix(:, :)
        ! update the actual diis B matrix
        do iguess = 1, self%guess_count
             ! B_ij = Tr(e_i e_j)
             self%diis_matrix(self%next_index, iguess) = get_trace( &
                 xmatmul(self%error_matrices(:, :, self%next_index), &
                         self%error_matrices(:, :, iguess)))
             ! we are assuming that the trace of the multiplication is the 
             ! same regardless of the direction
             self%diis_matrix(iguess, self%next_index) = &
                     self%diis_matrix(self%next_index, iguess)
        end do

        ! store the hamiltonian matrix
        self%hamiltonian_matrices(:, :, self%next_index) = &
             self%scf_cycle%hamiltonian_matrix

        ! get the index of the function that will be replaced the next time
        ! this function is called
        if (self%next_index == self%used_guess_count) then
            self%next_index = 1
        else
            self%next_index = self%next_index + 1
        end if
         
    end subroutine 

    subroutine LCMODIIS_destroy(self)
         class(LCMODIIS)           :: self
         if (allocated(self%hamiltonian_matrices)) deallocate(self%hamiltonian_matrices)
         if (allocated(self%diis_matrix)) deallocate(self%diis_matrix)
         if (allocated(self%error_matrices)) then
             deallocate(self%error_matrices)
         end if
     end subroutine

!--------------------------------------------------------------------!
!        MISC functions
!--------------------------------------------------------------------!
    
    pure function get_trace(matrix) result(trace)
        real(REAL64), intent(in)     :: matrix(:, :)
        real(REAL64)                 :: trace
        integer                      :: i
        
        !print *, "getting trace", matrix
        trace = 0.0d0
        do i = 1, size(matrix, 1)
            trace = trace + matrix(i, i)
        end do
        
    end function 
    
    

 
end module
