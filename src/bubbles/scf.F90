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
!> @file scf.F90 The file defining the SCFOptimizer abstract class and 
!!  its most primitive implementation: 'PowerMethod'.
module SCF_class
    use Globals_m
    use Function3D_class
    use SCFCycle_class
    use Action_class
    use SCFEnergetics_class
    implicit none
    public :: SCFOptimizer, PowerMethod
   
    private

    type, abstract :: SCFOptimizer
        class(SCFCycle), pointer    :: scf_cycle => NULL()
        !> The action object containing the information of the orders from the used
        type(Action),      pointer    :: action_object
        !> The total energy error threshold at which the SCF Optimizer has converged
        real(REAL64)                  :: total_energy_convergence_threshold = 1.e-6
        !> The eigen value error threshold at which the SCF Optimizer has converged
        real(REAL64)                  :: eigenvalue_convergence_threshold = 1.e-6
        !> Error at the latest iteration
        real(REAL64)                  :: error
        !> Double precision number determining the weight with which the orbitals are updated
        !! between SCF-loops
        real(REAL64)                  :: update_weight
        !> Number of orbitals in the optimized structure
        integer                       :: orbital_count
        !> The current iteration order number
        integer                       :: iteration_count
        !> Maximum number of iterations
        integer                       :: max_iterations = 50
        !> Nuclear repulsion energy
        real(REAL64)                  :: nuc_rep_energy
        !> Eigen value history        
        type(SCFEnergetics)           :: scf_energetics
    contains
        procedure, public  :: is_converged        => SCFOptimizer_is_converged
        procedure, public  :: store_energetics    => SCFOptimizer_store_energetics
        procedure, public  :: load_scf_energetics => SCFOptimizer_load_scf_energetics
        procedure(destroy), public, deferred        :: destroy
        procedure(optimize), public, deferred       :: optimize
    end type

    abstract interface
       subroutine destroy(self)
           import SCFOptimizer
           class(SCFOptimizer)                      :: self
       end subroutine

       subroutine optimize(self)
           import SCFOptimizer, Function3D
           class(SCFOptimizer)                      :: self
           type(Function3D)                         :: new_orbitals(self%orbital_count)
       end subroutine
    end interface

    type, extends(SCFOptimizer) :: PowerMethod
        logical                  :: diagonalize = .FALSE.
        logical                  :: orthogonalize = .FALSE.
    contains
        procedure, public :: destroy => PowerMethod_destroy
        procedure, public :: optimize => PowerMethod_optimize
    end type

    interface PowerMethod
        module procedure PowerMethod_init
    end interface

contains

!-------------------------------------------------------!
!   SCFOptimizer functions                              !
!-------------------------------------------------------!

    !> Checks if the calculation is converged by checking agaist the criteria
    function SCFOptimizer_is_converged(self) result(converged)
        class(SCFOptimizer), intent(in)            :: self
        real(REAL64)                               :: delta, previous_energy
        logical                                    :: converged, total_energy_converged, &
                                                      eigen_values_converged
        integer                                    :: i
        
        if (self%iteration_count == 1) then
            previous_energy = 0.0d0
        else
            previous_energy = self%scf_energetics%total_energy(self%iteration_count-1)
        end if
        
        ! get the delta and check against the total energy convergence threshold
        delta = abs(self%scf_energetics%total_energy(self%iteration_count) - previous_energy)
        total_energy_converged = self%iteration_count > 1 .and. delta < self%total_energy_convergence_threshold
        
        ! if the total energy criterion is fulfilled, check the eigen value criteria
        if (self%iteration_count > 1) then
            do i = 1, size(self%scf_energetics%eigen_values, 1)
                if (  abs(  self%scf_energetics%eigen_values(i, self%iteration_count) &
                        - self%scf_energetics%eigen_values(i, self%iteration_count-1)) &
                    > self%eigenvalue_convergence_threshold) then
                    eigen_values_converged = .FALSE.
                    exit
                end if
            end do
        else
            eigen_values_converged = .FALSE.
        end if
        converged = total_energy_converged .and. eigen_values_converged

 
        print '("|", a21, " | ", a21, " | ", a20, " |")', "Total E Converged?", "Eig. Val. Converged?", "Converged?"      
        print '("|", l21, " | ", l21, " | ", l20, " |")', total_energy_converged, eigen_values_converged, converged
        print '("-----------------------------------------------------------------------")'
        
    end function

    subroutine SCFOptimizer_store_energetics(self)
        class(SCFOptimizer), intent(inout)         :: self
        
        self%scf_energetics%total_energy(self%iteration_count) = &
            self%scf_cycle%energy + self%scf_energetics%nuclear_repulsion_energy
        self%scf_energetics%eigen_values(:, self%iteration_count)  = self%scf_cycle%get_orbital_energies()
    end subroutine
    
    !> Copy the existing scf energetics to the parameter of the 
    !! SCFOptimizer-object using the correct parameters in the copy.
    !! Additionally se the correct iteration count.
    subroutine SCFOptimizer_load_scf_energetics(self, scf_energetics)
        class(SCFOptimizer), intent(inout)         :: self
        type(SCFEnergetics), intent(in)            :: scf_energetics
        
        self%scf_energetics = SCFEnergetics(scf_energetics, self%orbital_count, self%max_iterations)
        
        
        ! if we have iteration history, do not increase the iteraction_count
        ! but let the scf optimizer to handle its evaluation
        if (allocated(scf_energetics%eigen_values)) then
            self%iteration_count = size(scf_energetics%eigen_values, 2)
        else
            self%iteration_count = 1
        end if
    end subroutine
    
    subroutine SCFOptimizer_destroy(self)
        class(SCFOptimizer), intent(inout)         :: self
        
        call self%scf_energetics%destroy()
    end subroutine
    
!-------------------------------------------------------!
!   PowerMethod functions                               !
!-------------------------------------------------------!


    function PowerMethod_init(scf_cycle, max_iterations, total_energy_convergence_threshold, &
                              eigenvalue_convergence_threshold, &
                              diagonalize, orthogonalize, update_weight, action_object, &
                              scf_energetics) result(new)
        class(SCFCycle), intent(inout), target :: scf_cycle 
        !> Maximum number of iterations
        integer,         intent(in)            :: max_iterations
        !> The error threshold for total energy at which the SCF Optimizer has converged
        real(REAL64),    intent(in)            :: total_energy_convergence_threshold
        !> The error threshold for each eigen value at which the SCF Optimizer has converged
        real(REAL64),    intent(in)            :: eigenvalue_convergence_threshold
        !> Logical variable determining if the orbitals are diagonalized during the
        !! SCF-loop
        logical,         intent(in)            :: diagonalize
        !> Logical variable determining if the orbitals are orthogonalized during the
        !! SCF-loop
        logical,         intent(in)            :: orthogonalize
        !> double precision number determining the weight with which the orbitals are updated
        !! between SCF-loops
        real(REAL64),    intent(in)            :: update_weight
        type(Action),    intent(in),    target :: action_object
        !> A SCFEnergetics object containing the previous update history
        type(SCFEnergetics), intent(in)        :: scf_energetics
        type(PowerMethod)                      :: new

        new%action_object => action_object
        new%scf_cycle => scf_cycle
        new%total_energy_convergence_threshold = total_energy_convergence_threshold
        new%eigenvalue_convergence_threshold = eigenvalue_convergence_threshold
        new%max_iterations = max_iterations
        new%diagonalize = diagonalize
        new%orthogonalize = orthogonalize
        new%orbital_count = size(new%scf_cycle%orbitals)
        new%update_weight = update_weight
        call new%load_scf_energetics(scf_energetics)
    end function

    subroutine PowerMethod_optimize(self)
        class(PowerMethod)           :: self
        real(REAL64)                 :: current_energy, previous_energy, energy_change

        if (self%iteration_count == 1) then
            previous_energy = 0.0d0
        else
            previous_energy = self%scf_energetics%total_energy(self%iteration_count-1)
        end if
            
        do
            call bigben%split("PowerMethod loop")
            ! if we have energy for this iteration, do not reevaluate it
            if (abs(self%scf_energetics%total_energy(self%iteration_count)) < 1d-6) then
                ! evaluate the matrix elements and evaluate the energy
                call self%scf_cycle%calculate_hamiltonian_matrix(self%diagonalize)

                ! diagonalize and update via doing a linear combination of orbitals,
                ! if self%diagonalize is .TRUE.
                if (self%diagonalize) then
                    call self%scf_cycle%diagonalize()
                    call self%scf_cycle%update_orbitals_linear_combination()
                end if
                ! store the energetics to history
                call self%store_energetics()
                current_energy = self%scf_cycle%energy
            else
                current_energy = self%scf_energetics%total_energy(self%iteration_count)
                self%scf_cycle%energy = current_energy - self%scf_energetics%nuclear_repulsion_energy
            end if
            
            energy_change = current_energy - previous_energy
            self%error = abs(energy_change)
            previous_energy = current_energy

            write(*, '("-----------------------------------------------------------------------")')
            write(*, '("in power method")') ! remove lnw
            write(*, '("Iteration ",i4," completed")')     self%iteration_count
            write(*, '("Total energy:      ", f24.16,"")') self%scf_energetics%total_energy(self%iteration_count)
            write(*, '("Electronic energy: ", f24.16,"")')   self%scf_energetics%total_energy(self%iteration_count) &
                                                            - self%scf_energetics%nuclear_repulsion_energy 
            write(*, '("Nuclear repulsion: ", f24.16,"")') self%scf_energetics%nuclear_repulsion_energy 
            write(*, '("Energy change:     ", f24.16,"")')  energy_change
            write(*, '("-----------------------------------------------------------------------")')
            
! flush(6)
! call abort()

            if (self%is_converged()) then
                write(*, '("Power method converged after", i4, " iterations.")') self%iteration_count
                call bigben%stop_and_print()
                ! Store the result orbitals. Note: the subroutine contains 
                ! the check whether we are actually doing the storing.
                call self%scf_cycle%store_orbitals(self%action_object%output_folder, &
                                              self%action_object%store_result_functions)
                call self%scf_energetics%write_energetics(self%action_object%output_folder, &
                                                          "scf_energetics", self%iteration_count)
                exit
            else if (self%iteration_count >= self%max_iterations) then
                write(*, '("Power method exceeded the maximum number of iterations: ", i4)') self%max_iterations
                call bigben%stop_and_print()
                ! Store the result orbitals. Note: the subroutine contains 
                ! the check whether we are actually doing the storing.
                call self%scf_cycle%store_orbitals(self%action_object%output_folder, &
                                              self%action_object%store_result_functions)
                call self%scf_energetics%write_energetics(self%action_object%output_folder, &
                                                          "scf_energetics", self%iteration_count)
                exit
            end if
            !write(*, '("Updating orbitals using power method with weight: ", f5.3)'), self%update_weight
            call self%scf_cycle%update(self%update_weight)
            if (self%orthogonalize) &
                call self%scf_cycle%gs_orthogonalize_orbitals()
            
            self%iteration_count = self%iteration_count + 1
            call bigben%stop_and_print()
            
            ! check if we are doing intermediate orbital storage, and perform them if necessary
            if (self%action_object%intermediate_orbital_store_interval > 0) then
                
                if (mod(self%iteration_count, self%action_object%intermediate_orbital_store_interval) == 0) then
                    
                    call self%scf_energetics%write_energetics(self%action_object%output_folder, &
                                                              "scf_energetics", self%iteration_count)
                    write(*, *) "Performing intermediate orbital storage"
                    ! Store the result orbitals. Note: the subroutine contains 
                    ! the check whether we are actually doing the storing.
                    call self%scf_cycle%store_orbitals(self%action_object%output_folder, &
                                                  self%action_object%store_result_functions)
                end if
            end if

        end do
    end subroutine

    subroutine PowerMethod_destroy(self)
        class(PowerMethod)           :: self
        nullify(self%scf_cycle)
        call SCFOptimizer_destroy(self)
    end subroutine
end module
