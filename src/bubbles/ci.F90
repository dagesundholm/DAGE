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
module ConfigurationInteraction_class
    use Function3D_class
    use SCFCycle_class
    use xmatrix_m
    use Globals_m

    implicit none
    public ConfigurationInteraction_init, ConfigurationInteraction
 
    private

!--------------------------------------------------------------------!
!        ConfigurationInteraction abstract class definition          !
!--------------------------------------------------------------------! 

    type, abstract :: ConfigurationInteraction
        !> Pointer to the used SCFCycle
        class(SCFCycle), pointer       :: scf_cycle
        !> The types of excitations taken into account
        logical                        :: singles, doubles, triples, quartets
        !> The Configuration interaction -matrix
        real(REAL64), allocatable      :: ci_matrix(:, :)
   contains 
        procedure(form_ci_matrix),      private, deferred   :: form_ci_matrix
        procedure,                      public              :: optimize => &
                                                                   ConfigurationInteraction_optimize
        procedure,                      public              :: diagonalize_ci_matrix => &
                                                                   ConfigurationInteraction_diagonalize_ci_matrix
        procedure,                      public              :: destroy => &
                                                                   ConfigurationInteraction_destroy
        
   end type 

    abstract interface 
       subroutine form_ci_matrix(self)
           import ConfigurationInteraction
           class(ConfigurationInteraction), intent(inout) ::  self
       end subroutine
   end interface
!--------------------------------------------------------------------!
!        RestrictedConfigurationInteraction class definition         !
!--------------------------------------------------------------------! 

    type, extends(ConfigurationInteraction) :: RestrictedConfigurationInteraction
   contains 
        procedure, public                     :: form_ci_matrix        => &
                                                     RestrictedConfigurationInteraction_form_ci_matrix
        procedure, private                    :: calculate_number_of_states        => &
                                                     RestrictedConfigurationInteraction_calculate_number_of_states
   end type 

contains

!--------------------------------------------------------------------!
!        ConfigurationInteraction functions and subroutines          !
!--------------------------------------------------------------------! 

     subroutine ConfigurationInteraction_init(new, scf_cycle, &
                                              singles, doubles, triples, quartets)
        class(ConfigurationInteraction), allocatable :: new
        !> The input SCF Cycle
        class(SCFCycle), intent(inout), target    :: scf_cycle
        !> The types of excitations taken into account
        logical,         intent(in)               :: singles, doubles, triples, quartets

        if (scf_cycle%restricted) then
            allocate(RestrictedConfigurationInteraction :: new)
        end if

        new%scf_cycle => scf_cycle
        new%singles = singles
        new%doubles = doubles
        new%triples = triples
        new%quartets = quartets
    end subroutine
    
    subroutine ConfigurationInteraction_destroy(self)
        class(ConfigurationInteraction), intent(inout) ::  self
        
        if (allocated(self%ci_matrix)) deallocate(self%ci_matrix)
    end subroutine

    subroutine ConfigurationInteraction_optimize(self)
        class(ConfigurationInteraction), intent(inout) ::  self
        
        !call self%scf_cycle%calculate_hamiltonian_matrix(.TRUE.)
        !call self%scf_cycle%diagonalize()

        call self%form_ci_matrix()
        call self%diagonalize_ci_matrix()

    end subroutine

    subroutine ConfigurationInteraction_diagonalize_ci_matrix(self)
        class(ConfigurationInteraction), intent(in)  ::  self
        real(REAL64)                                 :: eigen_values(size(self%ci_matrix, 1)), &
                                                        coefficients(size(self%ci_matrix, 1), size(self%ci_matrix, 2))
        print *, "ci matrix", self%ci_matrix(:, 1)
        call matrix_eigensolver(self%ci_matrix, eigen_values, coefficients)
        print *, "eigen values", eigen_values
        print *, "coefficients", coefficients(:, 1)
    end subroutine

!--------------------------------------------------------------------!
!        RestrictedConfigurationInteraction functions and subroutines!
!--------------------------------------------------------------------!


    !> Calculates the number of determinants taken into account with the settings included in
    !! 'self'. The result calculated to 'number_of_states' and returned to caller.
    function RestrictedConfigurationInteraction_calculate_number_of_states(self) result(number_of_states)
        class(RestrictedConfigurationInteraction), intent(inout) ::  self
        integer                                                  :: number_of_states 
        integer                                                  :: novot_a, novot_b, novvt_a, novvt_b
        
        number_of_states = 1
        if (self%singles) then
            number_of_states = &
                number_of_states + self%scf_cycle%nvir_a * self%scf_cycle%nocc_a
            !if (self%scf_cycle%nocc_a /= self%scf_cycle%nocc_b) &
                number_of_states = &
                    number_of_states + self%scf_cycle%nvir_b * self%scf_cycle%nocc_b
        end if
        
        print *, "number of singles", number_of_states-1
        
        ! calculate the number of states
        if (self%doubles) then
            ! add the number of valid singlet excitations
            !if (self%scf_cycle%nocc_a /= self%scf_cycle%nocc_b) then
                number_of_states = &
                    number_of_states +   self%scf_cycle%nvir_a &
                                    * self%scf_cycle%nvir_b &
                                    * self%scf_cycle%nocc_a &
                                    * self%scf_cycle%nocc_b
            !else
            !    number_of_states = &
            !        number_of_states +    self%scf_cycle%nvir_a &
            !                            * self%scf_cycle%nvir_b &
            !                            * self%scf_cycle%nocc_a &
            !                            * (self%scf_cycle%nocc_a+1) / 2
            !end if

            ! calculate the numbers of valid occupied/virtual triplet combinations novot_a/novvt_a/novot_a/novvt_b
            novot_a = (self%scf_cycle%nocc_a - 1) * self%scf_cycle%nocc_a / 2
            novot_b = (self%scf_cycle%nocc_b - 1) * self%scf_cycle%nocc_b / 2
            novvt_a = (self%scf_cycle%nvir_a - 1) * self%scf_cycle%nvir_a / 2
            novvt_b = (self%scf_cycle%nvir_b - 1) * self%scf_cycle%nvir_b / 2

            ! finally add the number of valid triplet excitations
             number_of_states = &
                number_of_states + novot_a * novvt_a
            if (self%scf_cycle%nocc_a /= self%scf_cycle%nocc_b) then
                number_of_states = &
                    number_of_states + novot_b * novvt_b
            end if
        end if
    end function
   

    subroutine RestrictedConfigurationInteraction_form_ci_matrix(self)
        class(RestrictedConfigurationInteraction), intent(inout) ::  self
        integer                                                  :: number_of_states, istate, &
                                                                    novot_a, novot_b, novvt_a, novvt_b, i, j, k, l
        integer, allocatable                                     :: state_occupations(:, :), occupations(:)
        real(REAL64)                                             :: temp
        
        ! check the type of the scf cycle and do the needed actions to get all the required 
        ! matrix elements in the orbital form.
        select type(scf_cycle => self%scf_cycle)
            ! if the cycle is based on the lcmo-rhf,
            ! reformulate the one electron matrices and the two electron integrals to be in
            ! orbital-form as the basis function two electron integrals and one electron matrices
            ! are evaluated by the scf-cycle constructor.
            type is (LCMORHFCycle)
                call self%scf_cycle%calculate_orbital_one_electron_matrices &
                        (self%scf_cycle%coefficients_a, self%scf_cycle%kinetic_matrix, &
                        self%scf_cycle%overlap_matrix, self%scf_cycle%nuclear_electron_matrix)
                call self%scf_cycle%calculate_orbital_two_electron_integrals &
                        (self%scf_cycle%coefficients_a, self%scf_cycle%two_electron_integrals)
            ! if the cycle is based on the lcmo-rdft, calculate all the matrix elements in 
            ! basis function form. Then reformulate the one electron matrices and the two
            ! electron integrals to be in orbital-form.
            type is (LCMORDFTCycle)
                ! calculate 
                call self%scf_cycle%calculate_one_electron_matrices(.TRUE.)
                call self%scf_cycle%calculate_all_two_electron_integrals()
                call self%scf_cycle%calculate_orbital_one_electron_matrices &
                        (self%scf_cycle%coefficients_a, self%scf_cycle%kinetic_matrix, &
                        self%scf_cycle%overlap_matrix, self%scf_cycle%nuclear_electron_matrix)
                call self%scf_cycle%calculate_orbital_two_electron_integrals &
                        (self%scf_cycle%coefficients_a, self%scf_cycle%two_electron_integrals)
            ! calculate the matrix elements directly in orbital form for the rdft and rhf cycles
            type is (RHFCycle)
                call self%scf_cycle%calculate_one_electron_matrices(.TRUE.)
                call self%scf_cycle%calculate_all_two_electron_integrals()
            type is (RDFTCycle)
                call self%scf_cycle%calculate_one_electron_matrices(.TRUE.)
                call self%scf_cycle%calculate_all_two_electron_integrals()
        end select

        ! calculate the number of states
        number_of_states = self%calculate_number_of_states()
        print *, "NUMBER OF STATES", number_of_states
        
        allocate(self%ci_matrix(number_of_states, number_of_states), source = 0.0d0)
        allocate(state_occupations(self%scf_cycle%nocc_a+self%scf_cycle%nvir_a, number_of_states+1), source = 0)
        allocate(occupations(self%scf_cycle%nocc_a+self%scf_cycle%nvir_a), source = 0)
        
        print *, "SCF energy", self%scf_cycle%energy
        self%ci_matrix(1, 1) = self%scf_cycle%energy
        
        occupations(:min(self%scf_cycle%nocc_a, self%scf_cycle%nocc_b)) = 2
        occupations(min(self%scf_cycle%nocc_a, self%scf_cycle%nocc_b)+1 : &
                    self%scf_cycle%nocc_a) = 1
        
        forall (i = 1 : number_of_states)
            state_occupations(:, i) = occupations
        end forall

        istate = 2
        if (self%singles) then

            do i = 1, self%scf_cycle%nocc_a
                state_occupations(i, istate : istate + self%scf_cycle%nvir_a-1) = &
                    state_occupations(i, istate) / (-2)
                do j = self%scf_cycle%nocc_a+1, self%scf_cycle%nocc_a + self%scf_cycle%nvir_a
                    state_occupations(j, istate) = 1
                    istate = istate + 1
                end do
            end do
            ! if the number of occupied states is the same for both spins, then the
            ! single excitations are the same regardless of the spin of the excited
            ! electron
            !if (self%scf_cycle%nocc_a /= self%scf_cycle%nocc_b)  then
                do i = 1, self%scf_cycle%nocc_b
                    state_occupations(i, istate : istate + self%scf_cycle%nvir_b-1) = &
                        state_occupations(i, istate) / 2
                    do j = self%scf_cycle%nocc_b+1, self%scf_cycle%nocc_b + self%scf_cycle%nvir_b
                        state_occupations(j, istate) = -1
                        istate = istate + 1
                    end do
                end do
            !end if
        end if

        if (self%doubles) then
            ! go through all possible valid singlet excitations (a and b electron excited)
            !if (self%scf_cycle%nocc_a /= self%scf_cycle%nocc_b)  then
                do i = 1, self%scf_cycle%nocc_a
                    
                    state_occupations(i, istate : istate + self%scf_cycle%nocc_b*self%scf_cycle%nvir_a*self%scf_cycle%nvir_b-1) = &
                        state_occupations(i, istate) / (-2)

                    do j = 1, self%scf_cycle%nocc_b
                        state_occupations(j, istate : istate + self%scf_cycle%nvir_a*self%scf_cycle%nvir_b-1) = &
                            state_occupations(j, istate) / 2

                        do k = self%scf_cycle%nocc_a+1, self%scf_cycle%nocc_a + self%scf_cycle%nvir_a
                            state_occupations(k, istate : istate + self%scf_cycle%nvir_b-1) =  1 
                        
                            do l = self%scf_cycle%nocc_b+1, self%scf_cycle%nocc_b + self%scf_cycle%nvir_b
                                state_occupations(l, istate) = state_occupations(l, istate) * 3 - 1
                                istate = istate + 1
                            end do
                        end do
                    end do
                end do
            !else
            !    do i = 1, self%scf_cycle%nocc_a
            !        state_occupations(i, istate : istate +  &
            !           (self%scf_cycle%nocc_a-i+1)*self%scf_cycle%nvir_a*self%scf_cycle%nvir_b-1) = &
            !            state_occupations(i, istate) / (-2)

            !        do j = i, self%scf_cycle%nocc_a
            !            state_occupations(j, istate : istate + self%scf_cycle%nvir_a*self%scf_cycle%nvir_b-1) = &
            !                state_occupations(j, istate) / 2

            !            do k = self%scf_cycle%nocc_a+1, self%scf_cycle%nocc_a + self%scf_cycle%nvir_a
            !                state_occupations(k, istate : istate + self%scf_cycle%nvir_b-1) =  1 
                        
            !                do l = self%scf_cycle%nocc_b+1, self%scf_cycle%nocc_b + self%scf_cycle%nvir_b
            !                    state_occupations(l, istate) = state_occupations(l, istate) * 3 - 1
            !                    istate = istate + 1
            !                end do
            !            end do
            !        end do
            !    end do
            !end if

            ! go through all possible valid triplet excitations with spin a 
            do i = 1, self%scf_cycle%nocc_a
                state_occupations(i, istate : istate + (self%scf_cycle%nocc_a - i) * novvt_a -1) = occupations(i) / (-2)
                do j = i+1, self%scf_cycle%nocc_a
                    state_occupations(j, istate : istate + novvt_a -1) = state_occupations(j, istate) / (-2)
                    do k = self%scf_cycle%nocc_a+1, self%scf_cycle%nocc_a + self%scf_cycle%nvir_a
                        state_occupations(k, istate : istate + self%scf_cycle%nocc_a + self%scf_cycle%nvir_a - k -1) = 1
                        do l = k+1, self%scf_cycle%nocc_a + self%scf_cycle%nvir_a
                            state_occupations(l, istate) = 1
                            istate = istate + 1
                        end do
                    end do
                end do
            end do

            !if (self%scf_cycle%nocc_a /= self%scf_cycle%nocc_b)  then
                ! go through all possible valid triplet excitations with spin b 
                do i = 1, self%scf_cycle%nocc_b
                    state_occupations(i, istate : istate + (self%scf_cycle%nocc_b - i) * novvt_b -1) = occupations(i) / (2)
                    do j = i+1, self%scf_cycle%nocc_b
                        state_occupations(j, istate : istate + novvt_b -1) = state_occupations(j, istate) / (2)
                        do k = self%scf_cycle%nocc_b+1, self%scf_cycle%nocc_b + self%scf_cycle%nvir_b
                            state_occupations(k, istate : istate + self%scf_cycle%nocc_b + self%scf_cycle%nvir_b - k -1) = &
                                state_occupations(k, istate) * 3 -1
                            do l = k+1, self%scf_cycle%nocc_b + self%scf_cycle%nvir_b
                                state_occupations(l, istate) = state_occupations(l, istate) * 3 -1
                                istate = istate + 1
                            end do
                        end do
                    end do
                end do
            !end if
            !print *, "istate after doubles", istate
        end if
    
#ifdef HAVE_OMP
        !$OMP PARALLEL DO PRIVATE(i, j) 
#endif 
        do k = 1, number_of_states * (number_of_states + 1) / 2
            i = 1
            do while(k > i*(i+1) / 2)
                i = i +1
            end do
            j = k - i*(i-1) / 2
            
            if (i == 1 .or. j /= i) then
                self%ci_matrix(i, j) = self%scf_cycle%get_rhf_bracket_energy(state_occupations(:, i), state_occupations(:, j))
                self%ci_matrix(j, i) = self%ci_matrix(i, j)
            else
                self%ci_matrix(i, i) = 0.0d0
            end if
            
            !print *, "-- Coefficient", i, j, self%ci_matrix(i, j), &
            !    "vs", temp
            !if (i == j) then
            !    print *, "--- Energy of state", i, ":", state_occupations(:, i), ":", self%ci_matrix(i, j)
            !end if
        end do
#ifdef HAVE_OMP
        !$OMP END PARALLEL DO
#endif 
        !print *, "---Coefficients-1", self%ci_matrix(:, 6)
        deallocate(occupations, state_occupations)
        
    end subroutine
end module