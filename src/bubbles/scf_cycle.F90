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
!> @file scf.F90 Procedures for handling molecular orbitals and performing SCF
!! cycles.
module SCFCycle_class
    use Function3D_class
    use GBFMMCoulomb3D_class
    use GBFMMLaplacian3D_class
    use GBFMMParallelInfo_class
    use Function3D_types_m
    use Helmholtz3D_class
    use Laplacian3D_class
    use Globals_m
    use xmatrix_m
    use GaussQuad_class
    use Coulomb3D_class
    use GBFMMHelmholtz3D_class
    use MemoryLeakChecker_m
    use CoreEvaluator_class
    use Action_class
#ifdef HAVE_DFT
    use XC_class
#endif
#ifdef HAVE_CUDA
    use CUDA_m
#endif
    implicit none

    public  :: RHFCycle, URHFCycle,LCMORHFCycle, &
               LCMOSCFCycle, HelmholtzSCFCycle, RestrictedHelmholtzSCFCycle, &
               SCFCycle
    public  :: RHFCycle_init, RHFCycle_with_EEF_init,  &
               LCMORHFCycle_init, URHFCycle_init, ROHFCycle_init
#ifdef HAVE_DFT
    public  :: LCMORDFTCycle_init, RDFTCycle_init
    public  :: RDFTCycle, LCMORDFTCycle
#endif
    public  :: transform, get_number_of_resumable_orbitals, resume_orbitals, resume_orbital, &
               make_parallelization_info
    !public :: orthonormalize

    private
!--------------------------------------------------------------------!
!        SCF Cycle -interface definition                             !
!--------------------------------------------------------------------!
! A SCFCycle object is an object that describes one cycle in an SCF  !
! procedure, whether it is a Hartree-Fock, DFT or any                !
! other. It has two core subroutines, calculate_hamiltonian_matrix   !
! and update. The 'calculate_hamiltonian_matrix'                     !
! function executes the evaluates the matrices of the SCF cycle and  !
! 'update' sets the SCFCycle object ready for the next cycle. Other  !
! functionalities are meant to retrieve/calculate misc information   !
! related to the system.                                             !
!
! This object is meant to aid the implementation of SCF solvers and  !
! to give options in combining differents solvers with different     !
! methods (HF, DFT) by making an abstraction of a cycle.             !
!--------------------------------------------------------------------!
    type, abstract :: SCFCycle
        class(Coulomb3D),   pointer             :: coulomb_operator
        class(Laplacian3D), pointer             :: laplacian_operator
        type(CoreEvaluator), pointer            :: core_evaluator
        real(REAL64), public, allocatable       :: hamiltonian_matrix_a(:, :)
        real(REAL64), public, allocatable       :: hamiltonian_matrix_b(:, :)
        real(REAL64), public, pointer           :: hamiltonian_matrix(:, :)
        !> Pointer to the electon electron repulsion energy matrix used in calculation
        real(REAL64), pointer                   :: coulomb_matrix(:, :)
        real(REAL64), public, allocatable       :: coefficients_a(:, :)
        real(REAL64), public, allocatable       :: coefficients_b(:, :)
        real(REAL64), public, pointer           :: coefficients(:, :)
        real(REAL64), public, allocatable       :: eigen_values_a(:)
        real(REAL64), public, allocatable       :: eigen_values_b(:)
        real(REAL64), public, pointer           :: eigen_values(:)
        !> Overlap matrix containing the values of overlap integrals between orbitals with
        !! spin a
        real(REAL64), public, allocatable       :: overlap_matrix_a(:, :)
        !> Overlap matrix containing the values of overlap integrals between orbitals with
        !! spin b
        real(REAL64), public, allocatable       :: overlap_matrix_b(:, :)
        real(REAL64), public, pointer           :: overlap_matrix(:, :)
        real(REAL64), public                    :: energy
        type(Function3DPointer), allocatable    :: all_orbitals(:)
        type(Function3D), allocatable           :: orbitals_a(:)
        type(Function3D), allocatable           :: orbitals_b(:)
        type(Function3D), pointer               :: orbitals(:)
        !> A temporary function used in potential evaluation
        type(Function3D), allocatable           :: potential_input
        !> A temporary function to contain a coulomb potential, kinetic potential, and some other
        class(Function3D), allocatable          :: coulomb_potential, kin, potential2
        !> Temporary functions to store misc stuff
        type(Function3D), allocatable           :: temp
        integer                                 :: multiplicity
        !> Pointer to the nuclear electron matrix used in calculation
        real(REAL64), pointer                   :: nuclear_electron_matrix(:, :)
        !> Nuclear electron attraction energy matrix for spin a orbitals
        real(REAL64), allocatable               :: nuclear_electron_matrix_a(:, :)
        !> Nuclear electron attraction energy matrix for spin b orbitals
        real(REAL64), allocatable               :: nuclear_electron_matrix_b(:, :)
        !> Kinetic energy matrix for spin a orbitals 
        real(REAL64), allocatable               :: kinetic_matrix_a(:, :)
        !> Kinetic energy matrix for spin b orbitals 
        real(REAL64), allocatable               :: kinetic_matrix_b(:, :)
        !> Pointer to the kinetic energy matrix used in calculation
        real(REAL64), pointer                   :: kinetic_matrix(:, :)
        integer                                 :: total_orbital_count
        !> If GBFMM is used 
        logical                                 :: use_gbfmm = .FALSE.
        !> If the SCF-cycle is restricted 
        logical                                 :: restricted = .FALSE.
        !> The values of two electron integrals with spin a
        real(REAL64), allocatable               :: two_electron_integrals_a(:, :, :, :)
        !> The boolean values to keep track if the two electron integrals have been evaluated
        logical, allocatable                    :: two_electron_integral_evaluated_a(:, :, :, :)
        !> The values of two electron integrals for spin b
        real(REAL64), allocatable                :: two_electron_integrals_b(:, :, :, :)
        !> The boolean values to keep track if the two electron integrals have been evaluated for spin b
        logical, allocatable                     :: two_electron_integral_evaluated_b(:, :, :, :)
        real(REAL64), pointer                    :: two_electron_integrals(:, :, :, :)
        logical, pointer                         :: two_electron_integral_evaluated(:, :, :, :)
        !> The values of two electron integrals for spin a caused by density of spin b
        real(REAL64), allocatable                :: two_electron_integrals_ab(:, :, :, :)
        !> The values of two electron integrals for spin b caused by density of spin a
        real(REAL64), allocatable                :: two_electron_integrals_ba(:, :, :, :)
        !> Pointer to correct number of occupied orbitals value
        integer, pointer                         :: nocc
        !> Number of occupied orbitals for spin a and b
        integer                                  :: nocc_a, nocc_b
        !> Variable containing the number of virtual orbitals for spin a and b
        integer                                  :: nvir_a, nvir_b
        !> If coulomb potential is valid for the current electron density
        logical                                  :: coulomb_potential_is_valid
        !> If coulomb potentials are valid for the current electron density
        logical                                  :: orbital_potentials_are_valid
        !> Pointer to electron density function
        type(Function3D), pointer                :: electron_density
        !> Temporary functions to store misc stuff
        type(Function3D), allocatable            :: temp2, temp3
        !> Pointer to the exchange matrix used in calculation
        !! Note: Exchange matrix is needed even in DFT to allow exact exchange
        real(REAL64), pointer                    :: exchange_matrix(:, :)
#ifdef HAVE_DFT
        !> Exchange and correlation object (used only in DFT)
        type(XC)                                 :: exchange_correlation
        !> Exchange and correaltion energy (used only by DFT)
        real(REAL64)                             :: xc_energy
        !> Pointer to electron density function
        type(Function3D), pointer                :: xc_potential
        !> Pointer to Exchange and correlation matrix
        real(REAL64),     pointer                :: exchange_correlation_matrix(:, :)
#endif
    contains
        procedure(calculate_hamiltonian_matrix), public, deferred :: calculate_hamiltonian_matrix
        procedure(destroy), public, deferred                   :: destroy
        procedure(update), public, deferred                    :: update
        procedure, public                                      :: store_orbitals => &
                                                                      SCFCycle_store_orbitals
        procedure, public                                      :: diagonalize => &
                                                                      SCFCycle_diagonalize_worker
        procedure, public                                      :: diagonalize_worker => &
                                                                      SCFCycle_diagonalize_worker
        procedure, public                                      :: get_orbital_energies => &
                                                                      SCFCycle_get_orbital_energies
        procedure, public                                      :: set_orbital_energies => &
                                                                      SCFCycle_set_orbital_energies
        procedure, public                                      :: get_energy_order => &
                                                                      SCFCycle_get_energy_order
        procedure, public                                      :: gs_orthogonalize_orbitals => &
                                                                      SCFCycle_gs_orthogonalize_orbitals
        procedure, public                                      :: calculate_all_two_electron_integrals => &
                                                                      SCFCycle_calculate_all_two_electron_integrals
        procedure, public                                      :: validate_two_electron_integrals => &
                                                                      SCFCycle_validate_two_electron_integrals
        procedure, public                                      :: update_orbitals_linear_combination => &
                                                                      SCFCycle_update_orbitals_linear_combination_worker          
        procedure, public                                      :: update_orbitals_linear_combination_worker => &
                                                                      SCFCycle_update_orbitals_linear_combination_worker 
        procedure, private                                     :: set_two_electron_integral_value => &
                                                                      SCFCycle_set_two_electron_integral_value
        procedure, public                                      :: calculate_one_electron_matrices => &
                                                                      SCFCycle_calculate_one_electron_mats  
        procedure, public                                      :: calculate_orbital_one_electron_matrices => &
                                                                      SCFCycle_calculate_orbital_one_electron_matrices
        procedure, public                                      :: calculate_orbital_two_electron_integrals => &
                                                                      SCFCycle_calculate_orbital_two_electron_integrals
        procedure, public                                      :: get_rhf_bracket_energy => &
                                                                      SCFCycle_get_rhf_bracket_energy
        procedure                                              :: calculate_coulomb_matrix => &
                                                                      SCFCycle_calculate_coulomb_matrix
        procedure, public                                      :: get_coulomb_energy => &
                                                                      SCFCycle_get_coulomb_energy
        procedure, public                                      :: get_exchange_energy => &
                                                                      SCFCycle_get_exchange_energy
        procedure, public                                      :: get_total_coulomb_energy => &
                                                                      SCFCycle_get_total_coulomb_energy
        procedure, public                                      :: get_total_exchange_energy => &
                                                                      SCFCycle_get_total_exchange_energy
        procedure, private                                     :: print_orbital_energies => &
                                                                      SCFCycle_print_orbital_energies
#ifdef HAVE_DFT
        procedure                                              :: calculate_xc_matrix => &
                                                                      SCFCycle_calculate_xc_matrix
#endif
                               
    end type


    abstract interface
        subroutine calculate_hamiltonian_matrix(self, evaluate_all)
            import SCFCycle
            class(SCFCycle),   intent(inout), target :: self
            logical,           intent(in)            :: evaluate_all
        end subroutine

        subroutine destroy(self)
            import SCFCycle
            class(SCFCycle), intent(inout) :: self
        end subroutine

        subroutine update(self, scaling_factor)
            import SCFCycle, REAL64
            class(SCFCycle), intent(inout), target :: self
            real(REAL64),     intent(in)           :: scaling_factor
        end subroutine
    end interface

!--------------------------------------------------------------------!
!        Helmholtz SCF Cycle definition                              !
!--------------------------------------------------------------------!
! SCF Cycle object that uses a Helmholtz kernel to do orbital update !
!--------------------------------------------------------------------!

    
    type, abstract, extends(SCFCycle) :: HelmholtzSCFCycle
        class(Helmholtz3D), pointer       :: helmholtz_operator
        class(Function3D), allocatable    :: helmholtz_potential
        type(Function3D), allocatable     :: orbital_potentials(:)
        real(REAL64)                      :: coulomb_energy
    contains
        procedure, private:: subspace_optimize => &
                                 HelmholtzSCFCycle_subspace_optimize  
        procedure         :: calculate_overlap_matrix => &
                                 HelmholtzSCFCycle_calculate_overlap_matrix
        procedure         :: form_coulomb_matrix => &
                                 HelmholtzSCFCycle_form_coulomb_matrix
        procedure         :: form_diagonal_coulomb_matrix => &
                                 HelmholtzSCFCycle_form_diagonal_coulomb_matrix
        procedure         :: form_exchange_matrix => &
                                 HelmholtzSCFCycle_form_exchange_matrix
        procedure         :: form_diagonal_exchange_matrix => &
                                 HelmholtzSCFCycle_form_diagonal_exchange_matrix
        procedure         :: calculate_exchange_matrix => &
                                 HelmholtzSCFCycle_calculate_exchange_matrix
        procedure, public :: lowdin_orthogonalize_orbitals => &
                                 HelmholtzSCFCycle_lowdin_orthogonalize_orbitals
        procedure         :: calculate_two_electron_integrals => &
                                 HelmholtzSCFCycle_calculate_two_electron_integrals
        procedure, private:: update_orbitals_worker => &
                                 HelmholtzSCFCycle_update_orbitals_worker
        procedure         :: calculate_electron_density_worker => &
                                 HelmholtzSCFCycle_calculate_electron_density_worker
                                 
        procedure(calculate_orbital_potentials), public, deferred  :: calculate_orbital_potentials
    end type

    abstract interface

        subroutine calculate_orbital_potentials(self)
            import HelmholtzSCFCycle, Function3D
            class(HelmholtzSCFCycle), intent(inout), target :: self
        end subroutine
 
    end interface

!--------------------------------------------------------------------!
!        Restricted Helmholtz SCF Cycle definition                   !
!--------------------------------------------------------------------!
! Restricted SCF Cycle object that uses a Helmholtz kernel           !
! to do orbital update                                               !
!--------------------------------------------------------------------!

    type, abstract, extends(HelmholtzSCFCycle) :: RestrictedHelmholtzSCFCycle
        !> Electron-Electron repulsion energy matrix between all orbitals
        real(REAL64), allocatable         :: coulomb_matrix_a(:, :)
        !> Exchange energy matrix for all orbitals.
        !> Note: Exchange matrix is needed even in DFT to allow exact exchange
        real(REAL64), allocatable         :: exchange_matrix_a(:, :)
        !> Function containing the electron density
        type(Function3D), allocatable     :: electron_density_a
    contains
        procedure         :: update => &
                                 RestrictedHelmholtzSCFCycle_update_orbitals
        procedure         :: calculate_closed_shell_electron_density => &
                                 RestrictedHelmholtzSCFCycle_calculate_closed_shell_el_dens 
        procedure         :: calculate_open_shell_electron_density => &
                                 RestrictedHelmholtzSCFCycle_calculate_open_shell_el_dens  
    end type

!--------------------------------------------------------------------!
!        Un-Restricted Helmholtz SCF Cycle definition                !
!--------------------------------------------------------------------!
! Un-Restricted SCF Cycle object that uses a Helmholtz kernel        !
! to do orbital update                                               !
!--------------------------------------------------------------------!

    type, abstract, extends(HelmholtzSCFCycle) :: UnRestrictedHelmholtzSCFCycle
        !> Electron-Electron repulsion energy matrix between spin a orbitals
        real(REAL64), allocatable         :: coulomb_matrix_a(:, :)
        !> Electron-Electron repulsion energy matrix between spin b orbitals
        real(REAL64), allocatable         :: coulomb_matrix_b(:, :)
        !> Electron-Electron repulsion energy matrix to spin a orbitals caused by b orbitals
        real(REAL64), allocatable         :: coulomb_matrix_ab(:, :)
        !> Electron-Electron repulsion energy matrix to spin a orbitals caused by b orbitals
        real(REAL64), allocatable         :: coulomb_matrix_ba(:, :)
        !> Exchange energy matrix for spin a orbitals.
        !! Note: Exchange matrix is needed even in DFT to allow exact exchange
        real(REAL64), allocatable         :: exchange_matrix_a(:, :)
        !> Exchange energy matrix for spin b orbitals.
        !! Note: Exchange matrix is needed even in DFT to allow exact exchange
        real(REAL64), allocatable         :: exchange_matrix_b(:, :)
        !> The electron density for spin a
        type(Function3D), allocatable     :: electron_density_a
        !> The electron density for spin a
        type(Function3D), allocatable     :: electron_density_b
    contains
        procedure         :: calculate_ab_two_electron_integrals => &
                                 UnRestrictedHelmholtzSCFCycle_calculate_ab_2_el_integrals
        procedure         :: update => &
                                 UnRestrictedHelmholtzSCFCycle_update_orbitals
        procedure         :: calculate_spin_densities => &
                                 UnRestrictedHelmholtzSCFCycle_calculate_spin_densities
        procedure         :: gs_orthogonalize_orbitals => &
                                 UnRestrictedHelmholtzSCFCycle_gs_orthogonalize_orbitals 
        procedure         :: diagonalize => & 
                                 UnRestrictedHelmholtzSCFCycle_diagonalize
        procedure         :: update_orbitals_linear_combination => & 
                                 UnRestrictedHelmholtzSCFCycle_update_orbitals_lc
    end type

!--------------------------------------------------------------------!
!        LCMO Cycle definition                                       !
!--------------------------------------------------------------------!
! SCF Cycle interface defining objects that use (L)inear             !
! (C)ombination of (M)olecular (O)rbitals to do optimization.        !
!--------------------------------------------------------------------!


    
    type, abstract, extends(SCFCycle) :: LCMOSCFCycle
        real(REAL64), public, allocatable       :: orthogonalizing_matrix_a(:, :)
        real(REAL64), public, allocatable       :: orthogonalizing_matrix_b(:, :)
        real(REAL64), public, allocatable       :: density_matrix_a(:, :)
        real(REAL64), public, allocatable       :: density_matrix_b(:, :)
        real(REAL64), pointer                   :: density_matrix(:, :)
    contains
        procedure, public                       :: get_occupied_orbitals => &
                                                        LCMOSCFCycle_get_occupied_orbitals
        procedure, public                       :: calculate_electron_density => &
                                                        LCMOSCFCycle_calculate_electron_density
        procedure, public                       :: update => LCMOSCFCycle_update
    end type

!--------------------------------------------------------------------!
!        RHF Cycle definition                                        !
!--------------------------------------------------------------------!
! Restricted Hartree Fock Cycle that uses Helmholtz kernel to update !
! orbitals                                                           !
!--------------------------------------------------------------------!

    type, extends(RestrictedHelmholtzSCFCycle) :: RHFCycle
    contains
        procedure, public                 :: destroy => RHFCycle_destroy
        procedure, public                 :: calculate_hamiltonian_matrix => RHFCycle_calculate_hamiltonian_matrix
        procedure, public                 :: calculate_orbital_potentials => RHFCycle_calculate_orbital_potentials
    end type

!--------------------------------------------------------------------!
!        ROHF Cycle definition                                       !
!--------------------------------------------------------------------!
! Restricted Open-Shell Hartree Fock Cycle that uses Helmholtz kernel!
! to update orbitals                                                 !
!--------------------------------------------------------------------!

    type, extends(RestrictedHelmholtzSCFCycle) :: ROHFCycle
        !> The ROHF Parameters
        real(REAL64)                      :: f = 0.5d0, a=1.0d0, b=2.0d0
    contains
        procedure, public                 :: destroy => ROHFCycle_destroy
        procedure, public                 :: calculate_hamiltonian_matrix => ROHFCycle_calculate_hamiltonian_matrix
        procedure, public                 :: calculate_orbital_potentials => ROHFCycle_calculate_orbital_potentials
    end type
    
!--------------------------------------------------------------------!
!        RHF Cycle with external electric field definition           !
!--------------------------------------------------------------------!
! Restricted Hartree Fock Cycle that uses Helmholtz kernel to update !
! orbitals between run calls with external, spatially constant       !
! electric field                                                     !
!--------------------------------------------------------------------!
    type, extends(RHFCycle)   :: RHFCycle_with_EEF
        ! external electric field 
        real(REAL64)                   :: electric_field(3)
        ! potential of external electric field
        class(Function3D), allocatable :: external_electric_potential
        ! potential energy of nuclei from the external electric potential
        real(REAL64)                   :: nuclei_in_eef
    contains
        procedure, public              :: destroy => RHFCycle_with_EEF_destroy
        procedure, public              :: calculate_hamiltonian_matrix &
                                              =>  RHFCycle_with_EEF_calculate_hamiltonian_matrix
        procedure, public              :: calculate_orbital_potentials &
                                              => RHFCycle_with_EEF_calculate_orbital_potentials
    
 
    end type
! 
! interface RHFCycle_with_EEF
!     module procedure RHFCycle_with_EEF_init
! end interface
 

!--------------------------------------------------------------------!
!        URHF Cycle definition                                       !
!--------------------------------------------------------------------!
! UnRestricted Hartree Fock Cycle that uses Helmholtz kernel         !
! to update orbitals                                                 !
!--------------------------------------------------------------------!

    type, extends(UnRestrictedHelmholtzSCFCycle) :: URHFCycle
    contains
        procedure, public                 :: destroy => URHFCycle_destroy
        procedure, public                 :: calculate_hamiltonian_matrix => URHFCycle_calculate_hamiltonian_matrix
        procedure, public                 :: calculate_orbital_potentials => URHFCycle_calculate_orbital_potentials
    end type
 


!--------------------------------------------------------------------!
!        LCMO RHF Cycle Definition                                   !
!--------------------------------------------------------------------!
! Restricted Hartree Fock Cycle that uses LCMO to update  orbitals   !                                                  !
!--------------------------------------------------------------------!

    type, extends(LCMOSCFCycle) :: LCMORHFCycle
    contains
        procedure, public                 :: destroy => LCMORHFCycle_destroy
        procedure, public                 :: calculate_hamiltonian_matrix => LCMORHFCycle_calculate_hamiltonian_matrix
    end type
    
!--------------------------------------------------------------------!
!        LCMO RDFT Cycle Definition                                  !
!--------------------------------------------------------------------!
! Restricted DFT Cycle that uses LCMO to update  orbitals            ! 
!--------------------------------------------------------------------!

#ifdef HAVE_DFT
    type, extends(LCMOSCFCycle) :: LCMORDFTCycle
        !> The implementation of the electron-Electron repulsion energy matrix between all orbitals
        real(REAL64), allocatable         :: coulomb_matrix_a(:, :)
        !> The implementation of the xc matrix between all orbitals
        real(REAL64), allocatable         :: exchange_correlation_matrix_a(:, :)
        !> Function containing the XC  potential 
        type(Function3D)                  :: xc_potential_a
        !> Function containing the electron density 
        type(Function3D), allocatable     :: electron_density_a

    contains
        procedure, public                 :: destroy => LCMORDFTCycle_destroy
        procedure, public                 :: calculate_hamiltonian_matrix => LCMORDFTCycle_calculate_hamiltonian_matrix
    end type
#endif


!--------------------------------------------------------------------!
!        RDFT Cycle definition                                       !
!--------------------------------------------------------------------!
! Restricted Density Functional Theory Cycle that uses Helmholtz     !
! kernel to update orbitals                                          !
!--------------------------------------------------------------------!
#ifdef HAVE_DFT
    type, extends(RestrictedHelmholtzSCFCycle) :: RDFTCycle
        real(REAL64), allocatable         :: exchange_correlation_matrix_a(:, :)
        integer(INT32)                    :: xc_update_method

        type(Function3D)                  :: xc_potential_a

    contains
        procedure, public                 :: destroy => RDFTCycle_destroy
        procedure, public                 :: calculate_hamiltonian_matrix => RDFTCycle_calculate_hamiltonian_matrix
        procedure, public                 :: calculate_orbital_potentials => RDFTCycle_calculate_orbital_potentials
    end type
 
#endif


   
contains

!--------------------------------------------------------------------!
!      MISC FUNCTIONS                                                !
!--------------------------------------------------------------------!

    !> Convert a set of orbitals \f$\{\phi_i\}\rightarrow\{\phi'_i\}\f$ with
    !! the transformation matrix \f$\mathbf{U}\f$ as
    !! \f$\phi'_i=\sum_j \phi_j U_{ji} \f$ .
    function transform(phi, u) result(phi_new)
        real(REAL64),     intent(in) :: u(:,:)
        type(Function3D), intent(in) :: phi    (size(u,1))
        type(Function3D)             :: phi_new(size(u,2))

        integer  :: i,j
        integer  :: n_in
        integer  :: n_out

        n_in= size(u,1)
        n_out=size(u,2)

        do i=1,n_out
            phi_new(i)= phi(1)*u(1,i)
            do j=2,n_in
                phi_new(i)=phi_new(i) + phi(j) * u(j,i)
            end do
        end do
    end function

    !> Symmetric orthonormalization of orbitals.

    !> A set of non-orthonormal orbitals is transformed as 
    !! \f$\phi'_i=\sum_j \chi_j S^{-1/2}_{ji} \f$, where \f$\mathbf{S}\f$ is
    !! the overlap matrix (\f$S_{ij}=\big<\chi_i\big|\chi_j\big>\f$).
    function orthonormalize(phi_in) result(phi_out)
        type(Function3D), intent(in) :: phi_in (:)
        type(Function3D)             :: phi_out(size(phi_in))
        real(REAL64)                 :: overlap(size(phi_in), size(phi_in))
        real(REAL64)                 :: s2(size(phi_in), size(phi_in))


        overlap = get_overlap_matrix(phi_in)
        s2= get_orthogonalizing_matrix(overlap)
        phi_out=transform(phi_in, s2)
    end function

    !> A set of non-orthonormal orbitals is transformed as 
    !! \f$\phi'_i=\sum_j \chi_j S^{-1/2}_{ji} \f$, where \f$\mathbf{S}\f$ is
    !! the overlap matrix (\f$S_{ij}=\big<\chi_i\big|\chi_j\big>\f$).
    function get_orthogonalizing_matrix(overlap_matrix) result(orthogonalizing_matrix)
        real(REAL64), intent(in)     :: overlap_matrix(:, :)
        real(REAL64)                 :: orthogonalizing_matrix(size(overlap_matrix, 1), &
                                                               size(overlap_matrix, 2))
        real(REAL64)                 :: d(size(overlap_matrix, 1))
        real(REAL64)                 :: u(size(overlap_matrix, 1), size(overlap_matrix, 2))
        real(REAL64)                 :: s2(size(overlap_matrix, 1), size(overlap_matrix, 2))
        integer                      :: norb
        integer                      :: i


        ! diagonalize overlap matrix to get the coefficients for diagonalization 'u' and
        ! the diagonalized eigen values 'd'
        call matrix_eigensolver(overlap_matrix, d, u)
        orthogonalizing_matrix=0.d0
        ! get the ^(-1/2) of the eigen values
        forall(i=1:size(overlap_matrix, 1))
            orthogonalizing_matrix(i,i)=d(i)**(-.5d0)
        end forall
        ! do u . orthogonalizing_matrix . u^-1
        orthogonalizing_matrix = xmatmul(u, xmatmul(orthogonalizing_matrix, matrix_inverse(u)) )
        
    end function
    
    function get_overlap_matrix(phi_in) result(overlap)
        type(Function3D), intent(in) :: phi_in (:)
        real(REAL64)                 :: overlap(size(phi_in), size(phi_in)), &
                                        normalizing_factors(size(phi_in))
        
        integer                      :: norbitals
        integer                      :: i, j
        
        norbitals=size(phi_in)
        do j=1,norbitals
            normalizing_factors(j) = 1.0d0 / sqrt(phi_in(j) .dot. phi_in(j))
        end do

        do j=1,norbitals
            do i=1, j 
                overlap(i, j) = (phi_in(i) .dot. phi_in(j))
                overlap(i, j) =  overlap(i, j) &
                     * normalizing_factors(i) * normalizing_factors(j)
                overlap(j, i) = overlap(i, j) 
            end do
        end do
    end function

    pure function get_spin_density_matrix_from_occupations(coefficients, occupations) result(density_matrix)
        real(REAL64), intent(in)     :: coefficients(:, :)
        integer,      intent(in)     :: occupations(:)
        real(REAL64)                 :: density_matrix(size(coefficients, 1), &
                                                       size(coefficients, 2), 2)

        integer                      :: i, j, iorbital, first_spin, last_spin
        density_matrix = 0.0d0
        do iorbital=1, size(occupations)
            if (occupations(iorbital) == 2) then
                first_spin = 1
                last_spin = 2
            else if (occupations(iorbital) == -1) then
                first_spin = 2
                last_spin = 2
            else if (occupations(iorbital) == 1) then
                first_spin = 1
                last_spin = 1
            else
                first_spin = 1
                last_spin = 0
            end if
            do i=1,size(coefficients, 1)
                do j=1, size(coefficients, 1)
                    density_matrix(i, j, first_spin:last_spin) = density_matrix(i, j, first_spin:last_spin) + &
                         coefficients(i, iorbital) * coefficients(j, iorbital)
                 
                    
                end do
            end do
        end do
    end function

    pure function get_density_matrix_from_occupations(coefficients, occupations) result(density_matrix)
        real(REAL64), intent(in)     :: coefficients(:, :)
        integer,      intent(in)     :: occupations(:)
        real(REAL64)                 :: density_matrix(size(coefficients, 1), &
                                                       size(coefficients, 2))

        integer                      :: i, j, iorbital
        density_matrix = 0.0d0
        do iorbital=1, size(occupations)
            do i=1,size(coefficients, 1)
                do j=1, size(coefficients, 1)
                    density_matrix(i, j) = density_matrix(i, j) + &
                        abs(occupations(iorbital)) * coefficients(i, iorbital) * coefficients(j, iorbital)
                end do
            end do
        end do
    end function

    pure function get_density_matrix(coefficients, nocc) result(density_matrix)
        real(REAL64), intent(in)     :: coefficients(:, :)
        integer, intent(in)          :: nocc
        real(REAL64)                 :: density_matrix(size(coefficients, 1), &
                                                       size(coefficients, 2))
        integer                      :: occupations(size(coefficients, 1))

        occupations(:nocc) = 2
        occupations(nocc+1:) = 0
        density_matrix = get_density_matrix_from_occupations(coefficients, occupations)
    end function

    


    subroutine print_ut_matrix(name,matrix)
        character(*), intent(in) :: name
        real(REAL64), intent(in) :: matrix(:,:)

        integer :: i
        print*,"¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤ ",trim(name)," ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤"
        do i=1,size(matrix,1)
            write(*,'(a)',advance='no') repeat(" ",12*(i-1))
            print'(*(e12.4))',matrix(i,i:)
        end do
    end subroutine
    
    !> Calculates potential for a constant electric field
    !! as a Function3D object
    function constant_field_potential(self, field, origin_on_atom, origin_on_cube) &
                                          result(pot)
        class(SCFCycle), intent(in)    :: self
        real(REAL64)                   :: field(3)
        integer, optional              :: origin_on_atom
        real(REAL64), optional         :: origin_on_cube(3)
        !class(Function3D), allocatable :: pot
        type(Function3D) :: pot
        real(REAL64)                   :: origin(3)       
        integer                        :: atom_origin 
        real(REAL64), pointer          :: fval(:)
        integer                        :: i,j,k
        integer                        :: cubeshape(3)
        real(REAL64), pointer          :: x(:), y(:), z(:)
        real(REAL64), pointer          :: cubeval(:,:,:)
        ! origin moving
        real(REAL64)                   :: loc(3)

        ! make an empty copy of the first orbital of self
        !call pot%init_copy(self%orbitals(1), copy_content = .FALSE.)
        !pot = 0d0 *self%orbitals(1)
        call pot%init_copy(self%orbitals_a(1), copy_content = .TRUE.)
        call pot%product_in_place_REAL64( 0d0 )

        if(present(origin_on_atom)) then
            atom_origin = origin_on_atom
        else
            ! by default set the first atom to be origin of potential
            atom_origin = 1
        end if

        ! if origin on cube is given, potential is put on cube 
        if(present(origin_on_cube)) then
            cubeshape = shape(pot%get_cube())
            ! there is probably a smarter way to do this
            x => pot%grid%axis(X_)%get_coord()
            y => pot%grid%axis(Y_)%get_coord()
            z => pot%grid%axis(Z_)%get_coord()
            cubeval => pot%get_cube()
            do i=1, cubeshape(1)
                do j=1, cubeshape(2)
                    do k=1, cubeshape(3)
                        cubeval(i,j,k) = - field(1)*(x(i)-origin_on_cube(1) ) &
                                         - field(2)*(y(j)-origin_on_cube(2) ) &
                                         - field(3)*(z(k)-origin_on_cube(3) ) 
                    end do
                end do
            end do
            
            nullify(x,y,z,cubeval)

        ! otherwise potential is put to bubbles
        else
            ! set k in r^k to be 1
            call pot%bubbles%set_k(1)
            ! put -E_x * x to bubble
            fval => pot%bubbles%get_f(atom_origin, 1, 1)
            fval = -field(1) 

            ! put -E_z * z to bubble
            fval => pot%bubbles%get_f(atom_origin, 1, 0)
            fval = -field(3) 

            ! put -E_y *y to bubble
            fval => pot%bubbles%get_f(atom_origin, 1, -1)
            fval = -field(2) 

            nullify(fval)


           
            ! test: move field zero point to origin
            !cubeval => pot%get_cube()
            !loc = pot%bubbles%get_centers(atom_origin)
            !cubeval = -loc(1)*field(1) - loc(2)*field(2) - loc(3)*field(3)
            !cubeval = loc(1)*field(1) + loc(2)*field(2) + loc(3)*field(3)
            !nullify(cubeval)

        end if


    end function

    !> calculating dipole moment of a charge distribution
    !! moment is origin independent for neutral molecules
    function calculate_dipole_moment(self, electron_density, origin_on_atom, origin_on_cube) &
                                      result(moment)
        class(SCFCycle)                :: self
        type(Function3D)               :: electron_density
        integer, optional              :: origin_on_atom
        real(REAL64), optional         :: origin_on_cube(3)
        real(REAL64)                   :: origin(3)
        real(REAL64)                   :: moment(3)
        !class(Function3D), allocatable :: temp
        type(Function3D)               :: temp
        integer                        :: iatom
        real(REAL64), allocatable      :: centers(:,:)
        integer                        :: i        
        real(REAL64), allocatable      :: charges(:)

        moment = [0d0, 0d0, 0d0]

        ! around what point is the moment calculated
        if(present(origin_on_cube))then
            origin = origin_on_cube
        else
            if(present(origin_on_atom)) then
                origin = electron_density%bubbles%get_centers(origin_on_atom ) 
                iatom = origin_on_atom
            else
                iatom = 1
                origin = electron_density%bubbles%get_centers(iatom)
            end if
        end if
!print *, 'origin', origin

        ! electron contribution
call check_memory_cuda()
        if(present(origin_on_cube)) then
print *, 'cube moment calculation start'
            ! mux
            temp = constant_field_potential(self, [-1d0,0d0,0d0],origin_on_cube=origin)  
            moment(1) = moment(1) - ( temp .dot. electron_density )
call temp%destroy()
call check_memory_cuda()
            ! muy
            temp = constant_field_potential(self, [0d0,-1d0,0d0],origin_on_cube=origin)  
            moment(2) = moment(2) - ( temp .dot. electron_density )
call temp%destroy()
call check_memory_cuda()
            ! muz
print *, 'creating temp'
            temp = constant_field_potential(self, [0d0,0d0,-1d0],origin_on_cube=origin)  
call check_memory_cuda()
print *, 'temp created'
            !moment(3) = moment(3) - ( temp .dot. electron_density )
            moment(3) = moment(3) - (  electron_density .dot. temp )
call check_memory_cuda()
print *, 'dot product calculated'
call temp%destroy()
call check_memory_cuda()
!deallocate(temp)
call check_memory_cuda()
print *, 'temp deallocated'

print *, 'cube moment calculation stop'
        else
print *, 'bubbles moment calculation start'
            ! mux
            temp = constant_field_potential(self, [-1d0,0d0,0d0], iatom)  
            moment(1) = moment(1) - ( temp .dot. electron_density )
call check_memory_cuda()
            ! muy
            temp = constant_field_potential(self, [0d0,-1d0,0d0], iatom)  
            moment(2) = moment(2) - ( temp .dot. electron_density )
call check_memory_cuda()
            ! muz
            temp = constant_field_potential(self, [0d0,0d0,-1d0], iatom)  
            moment(3) = moment(3) - ( temp .dot. electron_density )
print *, 'bubbles moment calculation stop'
call temp%destroy()
call check_memory_cuda()
        end if
        !call temp%destroy()
        !deallocate(temp)
        !if(allocated(temp)) deallocate(temp)
call check_memory_cuda()

        ! nuclear contribution
        centers = electron_density%bubbles%get_centers()
!print *, 'centers before shift' , centers
        
        !do i=1, shape(centers,2)
        do i=1, size(centers)/3
            centers(:,i) = centers(:,i) - origin
        end do
!print *, 'centers after shift', centers
        charges = electron_density%bubbles%get_z()
        moment(1) = moment(1) + sum (centers(1,:)*charges)
        moment(2) = moment(2) + sum (centers(2,:)*charges)
        moment(3) = moment(3) + sum (centers(3,:)*charges)

        ! cleanup
        deallocate(centers, charges)

    end function

   
!--------------------------------------------------------------------!
!        SCF Cycle                                                   !
!--------------------------------------------------------------------! 

    subroutine SCFCycle_init_commons(self)
        class(SCFCycle), intent(inout) :: self
    end subroutine

    function SCFCycle_get_orbital_energies(self) result(orbital_energies)
        class(SCFCycle), intent(in) :: self
        real(REAL64)                :: orbital_energies(self%total_orbital_count)
        
        orbital_energies(1:size(self%orbitals_a)) = self%eigen_values_a

        if (size(self%orbitals_a) /= self%total_orbital_count) &
            orbital_energies(size(self%orbitals_a) + 1: size(self%orbitals_a) + size(self%orbitals_b)) = self%eigen_values_b
        
    end function

    subroutine SCFCycle_set_orbital_energies(self, orbital_energies)
        class(SCFCycle), intent(inout) :: self
        real(REAL64),    intent(in)    :: orbital_energies(self%total_orbital_count)
        
        self%eigen_values_a = orbital_energies(1:size(self%orbitals_a)) 
        if (size(self%orbitals_a) /= self%total_orbital_count) &
            self%eigen_values_b = orbital_energies(size(self%orbitals_a) + 1: size(self%orbitals_a) + size(self%orbitals_b))
        
    end subroutine
    

    !> Normalize the first 'number_of_orbitals' input orbitals 'orbitals'
    !! (or all if number_of_orbitals is not given).
    subroutine SCFCycle_normalize_orbitals(orbitals, number_of_orbitals)
        type(Function3D), intent(inout),  allocatable :: orbitals(:)
        integer,           intent(in),    optional    :: number_of_orbitals
        integer                                       :: i, number_of_orbitals_
        real(REAL64)                                  :: normalizing_factor
        
        if (present(number_of_orbitals)) then
            number_of_orbitals_ = number_of_orbitals
        else
            number_of_orbitals_ = size(orbitals)
        end if
        
        do i = 1, number_of_orbitals_
            normalizing_factor = 1.0d0 / sqrt(orbitals(i) .dot. orbitals(i))
            ! normalizing_factor = truncate_number(normalizing_factor, 4) ! used to be 4, lnw
            call orbitals(i)%product_in_place_REAL64(normalizing_factor)
        end do
    end subroutine
    
    
    !> Form a linear combination of 'input_orbitals' using 'coefficients' and store them
    !! to 'output_orbitals'. In order to perform the linear combination to a limited 
    !! amount of orbitals the 'number_of_orbitals' can be given to limit the operation 
    !! to only the first 'number_of_orbitals' orbitals.
    subroutine SCFCycle_get_linear_combination_of_orbitals(coefficients, input_orbitals, &
                                               output_orbitals, number_of_orbitals)
        real(REAL64),     intent(in)                :: coefficients(:, :)
        type(Function3D), intent(in)                :: input_orbitals(:)
        type(Function3D), intent(out),  allocatable :: output_orbitals(:)
        integer,          intent(in),   optional    :: number_of_orbitals
        integer                                     :: iorbital, jorbital, factor, number_of_orbitals_
        type(Function3D)                            :: temp
        
        if (present(number_of_orbitals)) then
            number_of_orbitals_ = number_of_orbitals
        else
            number_of_orbitals_ = size(input_orbitals)
        end if
        print *, "coefficients shape", shape(coefficients)
        print *, "number_of_orbitals", number_of_orbitals_, "nof input orbitals", size(input_orbitals)
        allocate(output_orbitals(number_of_orbitals_))
        
        do iorbital = 1, number_of_orbitals_
            factor = 1.0d0
            do jorbital = 1, number_of_orbitals_
                if (jorbital == 1) then
                    output_orbitals(iorbital) = (factor * coefficients(jorbital, iorbital)) * input_orbitals(jorbital)
                else
                    temp = factor * coefficients(jorbital, iorbital) * input_orbitals(jorbital)
                    call output_orbitals(iorbital)%add_in_place(temp)
                    call temp%destroy()
                end if
            end do
        end do
    end subroutine 

    !> Updates the orbitals (self%orbitals) by doing a linear combination
    !! of molecular orbitals. This requires that self%coefficients are set.
    subroutine SCFCycle_update_orbitals_linear_combination_worker(self, number_of_orbitals)
        class(SCFCycle), intent(inout), target  :: self
        integer,         intent(in), optional   :: number_of_orbitals
        type(Function3D), allocatable           :: orbitals(:)
        integer                                 :: iorbital, jorbital, factor, korbital, number_of_orbitals_

        if (present(number_of_orbitals)) then
            number_of_orbitals_ = number_of_orbitals
        else
            number_of_orbitals_ = size(self%orbitals)
        end if
       
        call bigben%split("Updating orbitals via LC")
        !call pinfo("Updating orbitals by doing a linear combination")
        
        ! get the linear combination of orbitals
        call SCFCycle_get_linear_combination_of_orbitals(self%coefficients, self%orbitals, &
                                                         orbitals, number_of_orbitals_)
        
        ! normalize the new orbitals
        call SCFCycle_normalize_orbitals(orbitals, number_of_orbitals_)

        ! replace the old orbitals and destroy the temp orbitals
        do iorbital = 1, number_of_orbitals_
            call self%orbitals(iorbital)%destroy()
            call self%orbitals(iorbital)%init_copy(orbitals(iorbital), copy_content = .TRUE.)
            call self%orbitals(iorbital)%communicate_cube_borders(reversed_order = .TRUE.)
            call self%orbitals(iorbital)%precalculate_taylor_series_bubbles()
            call orbitals(iorbital)%destroy()
        end do
        deallocate(orbitals)
        call bigben%stop()
        self%coulomb_potential_is_valid = .FALSE.

    end subroutine

    subroutine SCFCycle_get_energy_order(self, order)
        class(SCFCycle), intent(in)                :: self
        integer, allocatable, intent(inout)        :: order(:)
        integer, allocatable                       :: order_numbers(:)
        integer                                    :: order_number
        integer                                    :: iorbital, jorbital

        allocate(order_numbers(size(self%orbitals)))
        allocate(order(size(self%orbitals)))
        
        do iorbital = 1, size(self%orbitals)
            order_number = 1
            do jorbital = 1, size(self%orbitals)
                if (iorbital /= jorbital .and. &
                        (self%eigen_values(iorbital) > self%eigen_values(jorbital) .or. &
                            (self%eigen_values(iorbital) == self%eigen_values(jorbital) &
                             .and. iorbital > jorbital))) then
                    order_number = order_number + 1
                end if
            end do
            order_numbers(iorbital) = order_number
        end do
        ! determine the energy order of orbitals
        do iorbital = 1, size(self%orbitals)
            do jorbital = 1, size(self%orbitals)
                if (order_numbers(jorbital) == iorbital) then
                    order(iorbital) = jorbital 
                    exit
                end if
            end do
        end do
        deallocate(order_numbers)

    end subroutine 

    !> Performs Modified Gram-Schmidt Orthogonalization to orbitals of 'self'.
    !! Assumes that the input orbitals are already normalized.
    subroutine SCFCycle_gs_orthogonalize_orbitals(self)
        class(SCFCycle), intent(inout), target :: self
        integer                                :: iorbital, jorbital, j, i, k
        type(Function3D)                       :: orbital, temp
        type(Function3D), allocatable          :: new_orbitals(:)
        real(REAL64)                           :: normalization_factor, overlap
        integer, allocatable                   :: order(:)
        !call pinfo("Performing Gram-Schmidt orthogonalization")
        
        call self%get_energy_order(order)
        allocate(new_orbitals(size(self%orbitals)))
        do i = 1, size(self%orbitals)
            call new_orbitals(i)%init_copy(self%orbitals(order(i)), copy_content = .TRUE.)
            call self%orbitals(order(i))%destroy()
        end do

        do i = 1, size(self%orbitals)
            normalization_factor = 1.0d0 / sqrt(new_orbitals(i) .dot. new_orbitals(i))
            call self%orbitals(i)%init_copy(new_orbitals(i), copy_content = .TRUE.)
            call self%orbitals(i)%product_in_place_REAL64(normalization_factor)
            call self%orbitals(i)%precalculate_taylor_series_bubbles()
            call new_orbitals(i)%destroy()
            do j = i+1, size(self%orbitals)
                overlap = (new_orbitals(j) .dot. self%orbitals(i))
                call temp%init_copy(self%orbitals(i), copy_content = .TRUE.)
                call temp%product_in_place_REAL64(overlap)
                call new_orbitals(j)%subtract_in_place(temp)
                call temp%destroy()
            end do
        end do

        deallocate(new_orbitals)
        deallocate(order)
    end subroutine

    !> Create parallelization info by creating new 
    subroutine make_parallelization_info(parallelization_info, cubegrid, gbfmm, is_potential_input)
        !> The result parallelization info
        class(ParallelInfo), allocatable, intent(inout) :: parallelization_info
        !> Determines if the grid based fmm is on (affects slightly to cube grid
        !! spacings)
        type(Grid3D),         intent(in)                :: cubegrid
        logical,    optional, intent(in)                :: gbfmm
        !> Are we making a mould for potential input
        logical, optional, intent(in)                   :: is_potential_input

        logical                          :: potential_input, is_gbfmm

        if (present(gbfmm)) then
            is_gbfmm = gbfmm
        else
            is_gbfmm = .FALSE.
        end if

        if (present(is_potential_input)) then
            potential_input = is_potential_input
        else
            potential_input = .FALSE.
        end if


        if (is_gbfmm) then
            allocate(GBFMMParallelInfo :: parallelization_info)
            select type(parallelization_info)
                type is (GBFMMParallelInfo)
                    parallelization_info = GBFMMParallelInfo(cubegrid, is_potential_input= potential_input )
            end select
        else
            allocate(SerialInfo :: parallelization_info)
            select type(parallelization_info)
                type is (SerialInfo)
                    parallelization_info = SerialInfo(cubegrid)
            end select
        end if
          
    end subroutine

    subroutine resume_orbital(folder, orbital_order_number, restricted, spin_a, orbital)
        character(*),     intent(in)             :: folder
        !> The order number of the resumed orbital within the group of orbitals with 
        !! same spin
        integer,          intent(in)             :: orbital_order_number
        !> Whether the resumed orbitals have same spatial functions for
        !> a and b functions 
        logical,          intent(in)             :: restricted
        !> If the spin of the loaded orbital is a
        logical,          intent(in)             :: spin_a
        !> Function3D object in which the orbital is resumed to
        type(Function3D), intent(inout)          :: orbital
        character(len=17)                        :: filename

        
        ! get the name of orbital file in the output folder
        if (restricted) then
            write(filename, "('orbital_', I3.3, '.fun')") orbital_order_number
        else 
            if (spin_a) then
                write(filename, "('orbital_a_', I3.3, '.fun')") orbital_order_number
            else
                write(filename, "('orbital_b_', I3.3, '.fun')") orbital_order_number
            end if
        end if
        ! load the orbital from the file
        ! NOTE: there is a check in the called subroutine, which 
        ! check if this load is possible, however, one should
        ! still use the 'get_number_of_resumable_orbitals' beforehand
        ! to avoid any segmentation faults
        call orbital%load(folder, filename)
    end subroutine

    !> Resume 'resumed_count(1)' spin a orbitals and 'resumed_count(1)' spin b orbitals from file.
    !! and fit them to the mould given for the orbitals.
    !!
    !! This subroutine assumes that all the resumed orbitals have the same grids.
    subroutine resume_orbitals(folder, mould, restricted, orbitals_a, orbitals_b, resumed_limits)
        character(*),     intent(in)      :: folder
        !> The function used as a mould for the orbitals
        type(Function3D), intent(in)      :: mould
        !> Whether the resumed orbitals have same spatial functions for
        !> a and b functions 
        logical,          intent(in)      :: restricted
        !> The result functions for spin a orbitals
        type(Function3D), intent(inout), target :: orbitals_a(:)
        !> The result functions for spin b orbitals
        type(Function3D), intent(inout), target :: orbitals_b(:)
        !> The start and end indices of a-spin and b-spin orbitals resumed
        integer,          intent(in)      :: resumed_limits(2, 2)
        integer                           :: i, j
        type(Projector3D), allocatable    :: projector
        type(Function3D)                  :: temp
        class(Function3D), allocatable    :: projection_result
        type(Grid3D), target              :: cubegrid
        class(ParallelInfo), allocatable  :: parallel_info    
        type(Grid1D), target, allocatable :: bubble_grids(:)
        type(Grid1DPointer),  allocatable :: bubble_grid_pointers(:)
        
        

        if (resumed_limits(2, 1) - resumed_limits(1, 1) >= 0 .or. &
            resumed_limits(2, 2) - resumed_limits(1, 2) >= 0) then
            ! load the grid of the cube
            cubegrid = Grid3D(folder, "cubegrid.g3d")
#ifdef HAVE_CUDA
            call cubegrid%cuda_init()
#endif
            call make_parallelization_info(parallel_info, &
                                           cubegrid = cubegrid,&
                                           gbfmm = .FALSE., &
                                           is_potential_input = .FALSE.)
            call resume_Grid1Ds(bubble_grids, folder, "bubblegrids.g1d")
            allocate(bubble_grid_pointers(size(bubble_grids)))
            do i = 1, size(bubble_grids)
                bubble_grid_pointers(i)%p => bubble_grids(i)
            end do
        end if

        do i = resumed_limits(1, 1), resumed_limits(2, 1)
            ! load the orbital from the file
            temp = Function3D(parallel_info, type=F3D_TYPE_CUSP)
            temp%bubbles%global_grids = bubble_grid_pointers
            call resume_orbital(folder, i, restricted, .TRUE., temp)
            
            ! init the empty orbital from mould
            projection_result=Function3D(mould, type=F3D_TYPE_CUSP)

            ! if the projector is not allocated, allocate and init it
            if (.not. allocated(projector)) then
                projector = Projector3D(cubegrid, gridout=mould%grid, &
                                        bubble_grids_in=bubble_grid_pointers, &
                                        bubble_grids_out=mould%bubbles%global_grids)
#ifdef HAVE_CUDA
                call projector%cuda_init()
#endif
                
            end if

            ! project the resumed orbital to the mould
            call projector%operate_on(temp, projection_result)
            orbitals_a(i) = Function3D(projection_result, copy_content = .TRUE.)
            call temp%destroy()
            call projection_result%destroy()
            deallocate(projection_result)
        end do

        do i = resumed_limits(1, 2), resumed_limits(2, 2)
            ! load the orbital from the file
            temp = Function3D(parallel_info, type=F3D_TYPE_CUSP)
            temp%bubbles%global_grids = bubble_grid_pointers
            call resume_orbital(folder, i, restricted, .FALSE., temp)

            ! inite the empty orbital from mould
            projection_result=Function3D(mould, type=F3D_TYPE_CUSP)

            ! if the projector is not allocated, allocate and init it
            if (.not. allocated(projector)) then
                projector = Projector3D(cubegrid, gridout=mould%grid, &
                                        bubble_grids_in=bubble_grid_pointers, &
                                        bubble_grids_out=mould%bubbles%gr)
            end if
            
            ! project the resumed orbital to the mould
            call projector%operate_on(temp, projection_result)
            orbitals_b(i) = Function3D(projection_result, copy_content = .TRUE.)
            call temp%destroy()
            call projection_result%destroy()
            deallocate(projection_result)
        end do

        if (resumed_limits(2, 1) - resumed_limits(1, 1) >= 0 .or. &
            resumed_limits(2, 2) - resumed_limits(1, 2) >= 0) then
            
#ifdef HAVE_CUDA
            call parallel_info%destroy_cuda()
            call cubegrid%cuda_destroy()
#endif
            call parallel_info%destroy()
            deallocate(parallel_info)

            ! deallocate the projector
            if (allocated(projector)) then
                call projector%destroy()
                deallocate(projector)
            end if
            call cubegrid%destroy()

            do i = 1, size(bubble_grids)
                call bubble_grids(i)%destroy()
#ifdef HAVE_CUDA
                call bubble_grids(i)%cuda_destroy()
#endif
            end do
            deallocate(bubble_grids, bubble_grid_pointers)
        end if
        
            
    end subroutine

    !> Calculates the number of orbitals that can be resumed from file
    function get_number_of_resumable_orbitals(folder, restricted) result(resumable_count)
        character(*),                  intent(in)       :: folder
        logical,                       intent(in)       :: restricted
        integer                                         :: resumable_count(2)
        character(len=17)                               :: filename
        logical                                         :: file_exists_
        
        resumable_count = 0
        if (restricted) then
            do
                ! check if the orbital file exists in the output folder
                write(filename, "('orbital_', I3.3, '.fun')") resumable_count(1) +1
                inquire(file=trim(folder)//"/"//trim(filename), exist=file_exists_)
                ! if it does, add the count, if not end loop
                if (file_exists_) then
                    resumable_count(1) = resumable_count(1) + 1
                else
                    exit
                end if

            end do
        else
            do
                ! check if the orbital file exists in the output folder
                write(filename, "('orbital_a_', I3.3, '.fun')") resumable_count(1) +1
                inquire(file=trim(folder)//"/"//trim(filename), exist=file_exists_)
                ! if it does, add the count, if not end loop
                if (file_exists_) then
                    resumable_count(1) = resumable_count(1) + 1
                else
                    exit
                end if

            end do

            do
                ! check if the orbital file exists in the output folder
                write(filename, "('orbital_b_', I3.3, '.fun')") resumable_count(2) +1
                inquire(file=trim(folder)//"/"//trim(filename), exist=file_exists_)
                
                ! if it does, add the count, if not end loop
                if (file_exists_) then
                    resumable_count(2) = resumable_count(2) + 1
                else
                    exit
                end if

            end do
        end if

    end function
    
    


    !> Store the orbitals of the SCFCycle to files residing in 'folder'. 
    subroutine SCFCycle_store_orbitals(self, folder, mode)
        class(SCFCycle), intent(in), target  :: self
        !> The output folder
        character(*),    intent(in)          :: folder
        !> Storage mode, where 0: store nothing, 1: store all, 2: store only bubbles, 3: store only cubes
        integer,         intent(in)          :: mode
        integer                              :: i
        character(len=13)                    :: filename
        type(Function3DPointer), allocatable :: orbitals_a_ptrs(:), orbitals_b_ptrs(:)
        type(Function3D),        allocatable, target :: new_orbitals_a(:), new_orbitals_b(:)
       

        if (mode > DO_NOT_STORE_RESULT_FUNCTIONS) then
            
            
            ! check if we are using lcmoscfcycle, where the orbitals stored in 'self' are 
            ! not orbitals but basis functions. If this is the case, get the orbitals 
            ! by doing a linear combination.
            select type(self)
                class is (LCMOSCFCycle)
                    if (allocated(self%orbitals_a)) then
                        allocate(orbitals_a_ptrs(size(self%orbitals_a)))
                        
                        call SCFCycle_get_linear_combination_of_orbitals &
                                            (self%coefficients_a, self%orbitals_a, new_orbitals_a)
                        do i = 1, size(new_orbitals_a)
                            orbitals_a_ptrs(i)%p => new_orbitals_a(i)
                        end do
                    end if
                    
                    ! do the b-spin orbitals
                    if (allocated(self%orbitals_b) .and. .not. self%restricted) then
                        allocate(orbitals_b_ptrs(size(self%orbitals_b)))
                        call SCFCycle_get_linear_combination_of_orbitals &
                                            (self%coefficients_b, self%orbitals_b, new_orbitals_b)
                        do i = 1, size(new_orbitals_b)
                            orbitals_b_ptrs(i)%p => new_orbitals_b(i)
                        end do
                    end if
                class default
                    if (allocated(self%orbitals_a)) then
                        allocate(orbitals_a_ptrs(size(self%orbitals_a)))
                        do i = 1, size(self%orbitals_a)
                            orbitals_a_ptrs(i)%p => self%orbitals_a(i)
                        end do
                    end if
                    
                    if (allocated(self%orbitals_b) .and. .not. self%restricted) then
                        allocate(orbitals_b_ptrs(size(self%orbitals_b)))
                        do i = 1, size(self%orbitals_b)
                            orbitals_b_ptrs(i)%p => self%orbitals_b(i)
                        end do
                    end if
            end select
            
            if (self%restricted) then
                do i = 1, size(orbitals_a_ptrs)
                    write(filename, "('orbital_', I3.3)") i
                    call orbitals_a_ptrs(i)%p%dump(folder, filename, mode)
                end do
            else
                if (allocated(orbitals_a_ptrs)) then
                    do i = 1, size(orbitals_a_ptrs)
                        write(filename, "('orbital_a_', I3.3)") i
                        call orbitals_a_ptrs(i)%p%dump(folder, filename, mode)
                    end do
                end if

                if (allocated(orbitals_b_ptrs)) then
                    do i = 1, size(orbitals_b_ptrs)
                        write(filename, "('orbital_b_', I3.3)") i
                        call orbitals_b_ptrs(i)%p%dump(folder, filename, mode)
                    end do
                end if
            end if
            ! do the cleanup of the temporary variables
            if (allocated(new_orbitals_a)) then
                do i = 1, size(new_orbitals_a)
                    call new_orbitals_a(i)%destroy()
                end do
                deallocate(new_orbitals_a)
            end if
            
            if (allocated(new_orbitals_b)) then
                do i = 1, size(new_orbitals_b)
                    call new_orbitals_b(i)%destroy()
                end do
                deallocate(new_orbitals_b)
            end if
            
            
            if (allocated(orbitals_a_ptrs)) deallocate(orbitals_a_ptrs)
            if (allocated(orbitals_b_ptrs)) deallocate(orbitals_b_ptrs)
        end if
    end subroutine


    subroutine SCFCycle_diagonalize_worker(self)
        class(SCFCycle), intent(inout), target      :: self
        real(REAL64), allocatable                   :: orthogonalizing_matrix(:, :)

        !call pinfo("Diagonalizing Hamiltonian matrix")
        call bigben%split("Diagonalizing Hamiltonian matrix")
        call start_memory_follower()
        orthogonalizing_matrix = get_orthogonalizing_matrix(self%overlap_matrix)

        ! orthogonalize fock matrix
        self%hamiltonian_matrix = xmatmul(orthogonalizing_matrix, &
            xmatmul(self%hamiltonian_matrix, orthogonalizing_matrix))

        ! diagonalize orthonormal fock matrix
        call matrix_eigensolver(self%hamiltonian_matrix, self%eigen_values, self%coefficients)

        ! transform coefficients to original non-orthogonal basis
        self%coefficients = xmatmul(orthogonalizing_matrix, self%coefficients)

        deallocate(orthogonalizing_matrix)

        call bigben%stop()
    end subroutine


    !> calculate coulomb matrix from pre-calculated electron density
    subroutine SCFCycle_calculate_coulomb_matrix(self, evaluate_all)
        class(SCFCycle), intent(inout), target           :: self
        logical, intent(in)                              :: evaluate_all
        class(Function3D), pointer                       :: coulomb_potential
  
        integer                                          :: i,j
        real(REAL64)                                     :: temp_e
write(*,*) 'begin SCFCycle_calculate_coulomb_matrix'

        if (size(self%orbitals) > 0) then
#ifdef HAVE_CUDA_PROFILING
            call start_nvtx_timing("Coulomb Matrix")
#endif
            self%coulomb_matrix = 0.0d0

            ! calculate matrices, start with coulomb (and nuclear), 
            ! then go to exchange matrix

            call bigben%split("Computing Coulomb matrix")
            !call pinfo("Computing Coulomb matrix")
#ifdef HAVE_CUDA
            call self%coulomb_operator%cuda_prepare()
#endif
            call start_memory_follower()
            ! get the potential caused by electron density
            if (.not. self%coulomb_potential_is_valid) then
                call self%coulomb_operator%operate_on(self%electron_density, self%coulomb_potential)
                self%coulomb_potential_is_valid = .TRUE.
            end if
            coulomb_potential => self%coulomb_potential
            
            ! go through all functions self%orbitals
            do j=1, size(self%orbitals)
                call bigben%split("Coulomb and Nuclear Loop")
                ! and all i <= j
                call coulomb_potential%multiply_sub(self%orbitals(j), self%temp2)
                
                call self%temp2%precalculate_taylor_series_bubbles()
                if (.not. evaluate_all) then
                    self%coulomb_matrix(j, j) = self%orbitals(j) .dot. self%temp2
                else
                    do i=1, j
                        ! evaluate the coulomb matrix value i, j
                        self%coulomb_matrix(i, j) = self%orbitals(i) .dot. self%temp2
                        self%coulomb_matrix(j, i) = self%coulomb_matrix(i, j)
                    end do
                end if
                
                call bigben%stop()
            end do
            nullify(coulomb_potential)
            call bigben%stop()
            call stop_memory_follower()
#ifdef HAVE_CUDA_PROFILING
            call stop_nvtx_timing()
#endif
        end if
write(*,*) 'end SCFCycle_calculate_coulomb_matrix'
    end subroutine
    
    
#ifdef HAVE_DFT
    !> Calculates XC-matrix from pre-evaluated XC-potential stored in 
    !! 'self%xc_potential'
    subroutine SCFCycle_calculate_xc_matrix(self, evaluate_all)
        class(SCFCycle), intent(inout), target           :: self
        !> If the off diagonal elements of the XC-matrix are evaluated
        logical, intent(in)                              :: evaluate_all
        type(Function3D), allocatable                    :: temp
        integer                                          :: i, j

write(*,*) 'begin SCFCycle_calculate_xc_matrix'
        self%exchange_correlation_matrix = 0.0d0
        
        ! Go through all functions self%orbitals and calculate the coulomb
        ! energy
        do i=1, size(self%orbitals)
            
            call self%orbitals(i)%precalculate_taylor_series_bubbles(ignore_bubbles = .FALSE.)
            
            if (evaluate_all) then
                call self%orbitals(i)%multiply_sub(self%xc_potential, temp)
        
                call temp%precalculate_taylor_series_bubbles()
                
                do j=1,i
                    self%exchange_correlation_matrix(i, j) = temp .dot. self%orbitals(j)
                    self%exchange_correlation_matrix(j, i) = self%exchange_correlation_matrix(i, j)
                end do
            else
                call self%orbitals(i)%multiply_sub(self%orbitals(i), temp)
                call temp%precalculate_taylor_series_bubbles()
            
                self%exchange_correlation_matrix(i, i) = temp .dot. self%xc_potential 
                
            end if
            call temp%destroy()
            deallocate(temp)
        end do
write(*,*) 'end SCFCycle_calculate_xc_matrix'
flush(6)
    end subroutine
#endif


    subroutine SCFCycle_calculate_all_two_electron_integrals(self)
        class(SCFCycle), intent(inout), target          :: self
        class(Function3D), pointer                      :: pot
        real(REAL64)                                    :: temp_value
        integer                                         :: i,j,k,l, orbital_count 

        if (size(self%orbitals) > 0) then
#ifdef HAVE_CUDA_PROFILING
            call start_nvtx_timing("Exchange Matrix")
#endif
            orbital_count = size(self%orbitals)
            self%two_electron_integrals = 0.0d0
            self%two_electron_integral_evaluated = .FALSE.
            call start_memory_follower()

            !call pinfo("Computing two electron integrals")
            call bigben%split("Computing two electron integrals")
            ! go through all occupied orbitals
            do i=1, orbital_count
                ! go through all orbitals
                call bigben%split("Two electron integrals - Loop")
                do j=1, i
                    !print *, "i", i, "j", j
                    ! multiply orbital 'k' with orbital 'j': |kj> and store the result to self%potential_input
                    call self%orbitals(i)%multiply_sub(self%orbitals(j), self%temp)
                    call self%potential_input%copy_content(self%temp)
                    call self%potential_input%communicate_cube(reversed_order = .TRUE.)

                    ! calculate potential 1/r |kj>
                    call self%coulomb_operator%operate_on(self%potential_input, self%potential2)
                    pot => self%potential2
                    
                    do k=1, orbital_count
                        if (any([(.not. self%two_electron_integral_evaluated(i, j, k, l), l = 1, k)])) then
                            call pot%multiply_sub(self%orbitals(k), self%temp)
                            call self%temp%precalculate_taylor_series_bubbles()
                            do l = 1, k
                                if (.not. self%two_electron_integral_evaluated(i, j, k, l)) then
                                    temp_value = self%orbitals(l) .dot. self%temp
                                    call self%set_two_electron_integral_value(i, j, k, l, temp_value)   
                                end if
                            end do
                        end if
                    end do

                    nullify(pot)
                    
                end do
                call bigben%stop()
            end do
            call bigben%stop()
            call stop_memory_follower()
#ifdef HAVE_CUDA_PROFILING
            call stop_nvtx_timing()
#endif
        end if
        
    end subroutine

    !> Validates that there are no mistakes in the evaluation of two electron integrals
    !! by checking the equality of terms that should be equal.
    subroutine SCFCycle_validate_two_electron_integrals(self)
        class(SCFCycle), intent(inout)                  :: self
        integer                                         :: i, j, k, l
        do k = 1, size(self%two_electron_integrals, 4)
            do l = 1, size(self%two_electron_integrals, 3)
                do j = 1, size(self%two_electron_integrals, 2)
                    do i = 1, size(self%two_electron_integrals, 1)
                        if (k /= l .and. &
                            abs(self%two_electron_integrals(i, j, k, l) - &
                                self%two_electron_integrals(i, j, l, k)) > 1e-10) then
                            print "('(', i1, ' ', i1, ' | ', i1, ' ', i1, ') different from &
                                       &(', i1, ' ', i1, ' | ', i1, ' ', i1, '):', f24.12, 'vs', f24.12)", &
                                       i, j, k, l, i, j, l, k, self%two_electron_integrals(i, j, k, l), &
                                       self%two_electron_integrals(i, j, l, k)
                        end if
                        
                        if (i /= j .and. &
                            abs(self%two_electron_integrals(i, j, k, l) - &
                                self%two_electron_integrals(j, i, k, l)) > 1e-10) then
                            print "('(', i1, ' ', i1, ' | ', i1, ' ', i1, ') different from &
                                    &(', i1, ' ', i1, ' | ', i1, ' ', i1, '):', f24.12, 'vs', f24.12)", &
                                       i, j, k, l, j, i, k, l, self%two_electron_integrals(i, j, k, l), &
                                       self%two_electron_integrals(j, i, k, l)
                        end if
                        
                        
                        if (i /= j .and. k /= l .and. &
                            abs(self%two_electron_integrals(i, j, k, l) -&
                                self%two_electron_integrals(j, i, l, k)) > 1e-10) then
                            print "('(', i1, ' ', i1, ' | ', i1, ' ', i1, ') different from &
                                    &(', i1, ' ', i1, ' | ', i1, ' ', i1, '):', f24.12, 'vs', f24.12)", &
                                       i, j, k, l, j, i, l, k, self%two_electron_integrals(i, j, k, l), &
                                       self%two_electron_integrals(j, i, l, k)
                        end if
                        
                        if (k /= i .and. l /= j .and. &
                            abs(self%two_electron_integrals(i, j, k, l) - &
                                self%two_electron_integrals(k, l, i, j)) > 1e-10) then
                            print "('(', i1, ' ', i1, ' | ', i1, ' ', i1, ') different from &
                                    &(', i1, ' ', i1, ' | ', i1, ' ', i1, '):', f24.12, 'vs', f24.12)", &
                                       i, j, k, l, k, l, i, j, self%two_electron_integrals(i, j, k, l), &
                                       self%two_electron_integrals(k, l, i, j)
                        end if
                        
                        if (k /= i .and. l /= j .and. i /= j .and. &
                            abs(self%two_electron_integrals(i, j, k, l) - &
                                self%two_electron_integrals(k, l, j, i)) > 1e-10) then
                            print "('(', i1, ' ', i1, ' | ', i1, ' ', i1, ') different from &
                                    &(', i1, ' ', i1, ' | ', i1, ' ', i1, '):', f24.12, 'vs', f24.12)", &
                                       i, j, k, l, k, l, j, i, self%two_electron_integrals(i, j, k, l), &
                                       self%two_electron_integrals(k, l, j, i)
                        end if
                        
                        if (k /= i .and. l /= j .and. k /= l .and. &
                            abs(self%two_electron_integrals(i, j, k, l) - &
                                self%two_electron_integrals(l, k, i, j)) > 1e-10) then
                            print "('(', i1, ' ', i1, ' | ', i1, ' ', i1, ') different from &
                                    &(', i1, ' ', i1, ' | ', i1, ' ', i1, '):', f24.12, 'vs', f24.12)", &
                                       i, j, k, l, l, k, i, j, self%two_electron_integrals(i, j, k, l), &
                                       self%two_electron_integrals(l, k, i, j)
                        end if
                        
                        if (k /= i .and. l /= j .and. i /= j .and. k /= l .and. &
                            abs(self%two_electron_integrals(i, j, k, l) - &
                                self%two_electron_integrals(l, k, j, i)) > 1e-10) then
                            print "('(', i1, ' ', i1, ' | ', i1, ' ', i1, ') different from &
                                    &(', i1, ' ', i1, ' | ', i1, ' ', i1, '):', f24.12, 'vs', f24.12)", &
                                       i, j, k, l, l, k, j, i, self%two_electron_integrals(i, j, k, l), &
                                       self%two_electron_integrals(l, k, j, i)
                        end if
                        
                    end do
                end do
            end do
        end do
    end subroutine

    
    subroutine SCFCycle_set_two_electron_integral_value(self, i, j, k, l, value) 
        class(SCFCycle), intent(inout)                  :: self
        integer                                         :: i, j, k, l
        real(REAL64)                                    :: value

        self%two_electron_integrals(i, j, k, l) = value
        self%two_electron_integrals(i, j, l, k) = value
        self%two_electron_integrals(j, i, k, l) = value
        self%two_electron_integrals(j, i, l, k) = value
        self%two_electron_integrals(k, l, i, j) = value
        self%two_electron_integrals(k, l, j, i) = value
        self%two_electron_integrals(l, k, i, j) = value
        self%two_electron_integrals(l, k, j, i) = value

        self%two_electron_integral_evaluated(i, j, k, l) = .TRUE.
        self%two_electron_integral_evaluated(i, j, l, k) = .TRUE.
        self%two_electron_integral_evaluated(j, i, k, l) = .TRUE.
        self%two_electron_integral_evaluated(j, i, l, k) = .TRUE.
        self%two_electron_integral_evaluated(k, l, i, j) = .TRUE.
        self%two_electron_integral_evaluated(k, l, j, i) = .TRUE.
        self%two_electron_integral_evaluated(l, k, i, j) = .TRUE.
        self%two_electron_integral_evaluated(l, k, j, i) = .TRUE.
    end subroutine

    !> Get the energy of the bracket where the first determinant has occupation numbers:
    !! 'occupations_1' and the second determinant has occupation numbers 'occupations_2'
    !! To use this function, the two_electron integrals has to be in orbital-form, instead
    !! of the basis set form.
    function SCFCycle_get_rhf_bracket_energy(self, occupations_1, occupations_2) result(energy)
        class(SCFCycle), intent(inout), target :: self
        integer,         intent(in)            :: occupations_1(size(self%orbitals))
        integer,         intent(in)            :: occupations_2(size(self%orbitals))
        real(REAL64)                           :: spin_density_matrix(size(self%orbitals), size(self%orbitals), 2), &
                                                  total_density_matrix(size(self%orbitals), size(self%orbitals)), &
                                                  difference_density_matrix(size(self%orbitals), size(self%orbitals))
        integer                                :: occupation_difference(size(self%orbitals)), total_difference, i, j, k, l, &
                                                  index1, index2, index3, index4, spin1, spin2, spin3, spin4, difference, &
                                                  spin_factor, iorbital, indeces(4), spins(4)
        integer                                :: occupations_3(size(self%orbitals)), spin_index
        integer, target                        :: spin_occupations(size(self%orbitals), 2)
        integer, pointer                       :: spin_occupations_ptr(:)
        real(REAL64)                           :: energy

        occupation_difference = occupations_2 - occupations_1

        total_difference = 0
        ! evaluate the total number of excitations * 2
        do i = 1, size(self%orbitals)
            difference = occupation_difference(i)
            if (abs(difference) < 3) then
                if (occupations_2(i) == -1 * occupations_1(i) .and. abs(occupations_2(i)) == 1) then
                    total_difference = total_difference + 6
                else
                    total_difference = total_difference + abs(difference)
                end if
            else
                total_difference = total_difference + 1
            end if
        end do

        ! no excitation
        if (total_difference == 0) then
            energy = 0.0d0
            
            ! get the kinetic energy
            do i = 1, size(self%orbitals)
                energy = energy +     abs(occupations_1(i)) &
                                   * (self%kinetic_matrix_a(i, i) + self%nuclear_electron_matrix_a(i, i))
            end do
            
            energy = energy + self%get_total_coulomb_energy(occupations_1)
            energy = energy + self%get_total_exchange_energy(occupations_1)

        ! single excitation
        else if (total_difference == 2) then
            indeces = -1
            j = 1
            do i = 1, size(self%orbitals)
                if (occupations_1(i) /= occupations_2(i)) then
                    indeces(j) = i
                    j = j + 1
                    if (occupations_1(i) == 2 .and. occupations_2(i) == 1) then
                        ! the affected electron is of spin -1: beta
                        spin_index = 2
                    else if (occupations_1(i) == 2 .and. occupations_2(i) == -1) then
                        ! the affected electron is of spin 1: alpha
                        spin_index = 1
                    else if (occupations_1(i) == -1 .and. occupations_2(i) == 0) then
                        ! the affected electron is of spin -1: beta
                        spin_index = 2
                    else if (occupations_1(i) == -1 .and. occupations_2(i) == 2) then
                        ! the affected electron is of spin 1: alpha
                        spin_index = 1
                    else if (occupations_1(i) == 1 .and. occupations_2(i) == 0) then
                        ! the affected electron is of spin 1: alpha
                        spin_index = 1
                    else if (occupations_1(i) == 0 .and. occupations_2(i) == -1) then
                        ! the affected electron is of spin -1: beta
                        spin_index = 2
                    else if (occupations_1(i) == 0 .and. occupations_2(i) == 1) then
                        ! the affected electron is of spin 1: alpha
                        spin_index = 1
                    end if
                end if
            end do
            
            
            ! get the spin occupations
            spin_occupations(:, 1) = (occupations_1(:) + 1) / 2
            spin_occupations(:, 2) = abs(occupations_1(:) - spin_occupations(:, 1))
            spin_occupations_ptr => spin_occupations(:, spin_index)
            
            
                

            energy =   self%kinetic_matrix_a(indeces(1), indeces(2)) &
                     + self%nuclear_electron_matrix_a(indeces(1), indeces(2))

            energy = energy + self%get_coulomb_energy(indeces(1), indeces(2), occupations_1) &
                            - self%get_exchange_energy(indeces(1), indeces(2), spin_occupations_ptr)
                            
        ! double excitation
        else if (total_difference == 4) then
            indeces = -1
            spins = 0
            j = 1
            k = 3
            do i = 1, size(self%orbitals)
                if (occupation_difference(i) /= 0) then
                    if (occupations_1(i) == 2) then
                        if (occupations_2(i) == 1) then
                            indeces(j) = i
                            spins(j)   = -1
                            j = j + 1
                        else if (occupations_2(i) == -1) then
                            indeces(j) = i
                            spins(j)   = 1 
                            j = j + 1
                        else if (occupations_2(i) == 0) then
                            indeces(j) = i
                            indeces(j+1) = i
                            spins(j) = 1
                            spins(j+1) = -1
                            j = j + 2
                        end if
                    else if (occupations_1(i) == 1) then
                        if (occupations_2(i) == 2) then
                            indeces(k) = i
                            spins(k)   = -1
                            k = k + 1
                        else if (occupations_2(i) == 0) then
                            indeces(j) = i
                            spins(j)   = 1 
                            j = j + 1
                        else if (occupations_2(i) == -1) then
                            indeces(j) = i
                            spins(j) = 1
                            indeces(k) = i
                            spins(k) = -1
                            j = j + 1
                            k = k + 1
                        end if
                    else if (occupations_1(i) == -1) then
                        if (occupations_2(i) == 2) then
                            indeces(k) = i
                            spins(k)   = 1
                            k = k + 1
                        else if (occupations_2(i) == 0) then
                            indeces(j) = i
                            spins(j)   = -1 
                            j = j + 1
                        else if (occupations_2(i) == 1) then
                            indeces(j) = i
                            spins(j) = -1
                            indeces(k) = i
                            spins(k) = 1
                            j = j + 1
                            k = k + 1
                        end if
                    else if (occupations_1(i) == 0) then
                        if (occupations_2(i) == 1) then
                            indeces(k) = i
                            spins(k)   = 1
                            k = k + 1
                        else if (occupations_2(i) == -1) then
                            indeces(k) = i
                            spins(k)   = -1 
                            k = k + 1
                        else if (occupations_2(i) == 2) then
                            indeces(k) = i
                            indeces(k+1) = i
                            spins(k) = 1
                            spins(k+1) = -1
                            k = k + 2
                        end if
                    end if
                end if
            end do
            ! if the hole indeces are the same, this means that the excitation must have the
            ! same symmetry
            if (spins(1) == spins(2)  .and. spins(2) == spins(3) .and. spins(3) == spins(4)) then
                energy =   self%two_electron_integrals(indeces(1), indeces(3), indeces(2), indeces(4)) &
                         - self%two_electron_integrals(indeces(1), indeces(4), indeces(2), indeces(3))
                         
            else if (spins(1) == spins(3) .and. spins(2) == spins(4) .and. spins(1) /= spins(2)) then
                energy =   self%two_electron_integrals(indeces(1), indeces(3), indeces(2), indeces(4)) 
                
            else if (spins(1) == spins(4) .and. spins(2) == spins(3) .and. spins(1) /= spins(2)) then
                energy =   self%two_electron_integrals(indeces(1), indeces(4), indeces(2), indeces(3))
                
            else
                energy = 0.0d0
            end if
        else
            
            energy = 0.0d0
        end if

        
    end function
    
    subroutine SCFCycle_destroy(self)
        class(SCFCycle), intent(inout)         :: self
        integer                                :: i

        nullify(self%laplacian_operator)
        nullify(self%coulomb_operator)
        nullify(self%core_evaluator)

        if(allocated(self%kinetic_matrix_a)) deallocate(self%kinetic_matrix_a)
        if(allocated(self%nuclear_electron_matrix_a)) deallocate(self%nuclear_electron_matrix_a)
        if(allocated(self%overlap_matrix_a)) deallocate(self%overlap_matrix_a)
        if(allocated(self%hamiltonian_matrix_a)) deallocate(self%hamiltonian_matrix_a)
        if(allocated(self%coefficients_a)) deallocate(self%coefficients_a)
        if(allocated(self%eigen_values_a)) deallocate(self%eigen_values_a)
        if(allocated(self%two_electron_integrals_a)) deallocate(self%two_electron_integrals_a)
        if(allocated(self%two_electron_integral_evaluated_a)) deallocate(self%two_electron_integral_evaluated_a)



        if (allocated(self%orbitals_a)) then
            do i = 1, size(self%orbitals_a)
                call self%orbitals_a(i)%destroy()
            end do
            deallocate(self%orbitals_a)
        end if 
        
        if(allocated(self%kinetic_matrix_b)) deallocate(self%kinetic_matrix_b)
        if(allocated(self%overlap_matrix_b)) deallocate(self%overlap_matrix_b)
        if(allocated(self%nuclear_electron_matrix_b)) deallocate(self%nuclear_electron_matrix_b)
        if(allocated(self%hamiltonian_matrix_b)) deallocate(self%hamiltonian_matrix_b)
        if(allocated(self%coefficients_b)) deallocate(self%coefficients_b)
        if(allocated(self%eigen_values_b)) deallocate(self%eigen_values_b)
        if(allocated(self%two_electron_integrals_b)) deallocate(self%two_electron_integrals_b)
        if(allocated(self%two_electron_integrals_ab)) deallocate(self%two_electron_integrals_ab)
        if(allocated(self%two_electron_integrals_ba)) deallocate(self%two_electron_integrals_ba)
        if(allocated(self%two_electron_integral_evaluated_b)) deallocate(self%two_electron_integral_evaluated_b)




        if (allocated(self%orbitals_b)) then
            do i = 1, size(self%orbitals_b)
                call self%orbitals_b(i)%destroy()
            end do
            deallocate(self%orbitals_b)
        end if 
    
        
        if (allocated(self%all_orbitals)) deallocate(self%all_orbitals)
        if (allocated(self%potential_input)) then
            call self%potential_input%destroy()
            deallocate(self%potential_input)
        end if
        
        if (allocated(self%coulomb_potential)) then
            call self%coulomb_potential%destroy()
            deallocate(self%coulomb_potential)
        end if 
        
        if (allocated(self%kin)) then
            call self%kin%destroy()
            deallocate(self%kin)
        end if 
        
        if (allocated(self%potential2)) then
            call self%potential2%destroy()
            deallocate(self%potential2)
        end if
        
        if (allocated(self%temp)) then
            call self%temp%destroy()
            deallocate(self%temp)
        end if
        
        if (allocated(self%temp2)) then
            call self%temp2%destroy()
            deallocate(self%temp2)
        end if
        
        if (allocated(self%temp3)) then
            call self%temp3%destroy()
            deallocate(self%temp3)
        end if
    end subroutine


    !> Calculates the matrices involving only one function on
    !! each side, namely the overlap matrix, kinetic matrix and the
    !! nuclear electron attraction matrix. The result matrices are 
    !! calculated for 'self%orbitals' and stored in 'self'.
    subroutine SCFCycle_calculate_one_electron_mats(self, evaluate_all)
        class(SCFCycle),          intent(inout) :: self
        !> Determines if also the off-diagonal elements of the matrix are 
        !! evaluated
        logical,                  intent(in)    :: evaluate_all
        type(Function3D), pointer               :: nuclear_potential
        integer                                 :: i,j, orbital_count
        real(REAL64)                            :: temp
        type(Function3D)                        :: temp_function
write(*,*) 'begin SCFCycle_calculate_one_electron_mats'
        
        if (size(self%orbitals) > 0) then
            orbital_count = size(self%orbitals)

            ! get the nuclear potential Z/r bubbles summed to a Function3D bubbles
            nuclear_potential => self%coulomb_operator%get_nuclear_potential()

#ifdef HAVE_CUDA_PROFILING
            call start_nvtx_timing("Kinetic and Overlap Matrices")
#endif
            call start_memory_follower()
            !call pinfo("Computing kinetic energy matrix")
            call bigben%split("Calculating kinetic matrix")
            do j=1, orbital_count
                call bigben%split("Operate laplacian")
                call temp_function%init_copy(self%orbitals(j), copy_content = .TRUE.)
                call temp_function%inject_extra_bubbles(4)
                call self%laplacian_operator%operate_on(temp_function, self%kin) 
                call bigben%stop()
                !self%kin%taylor_order = 1
                call self%kin%product_in_place_REAL64((-0.5d0))
                !call self%core_evaluator%evaluate_contaminants(self%kin)
                call self%kin%precalculate_taylor_series_bubbles() !taylor_order = 1)
                self%kin%taylor_series_bubbles = 0.0d0
                call self%orbitals(j)%multiply_sub(nuclear_potential, self%temp)
                call self%temp%precalculate_taylor_series_bubbles()
                if (evaluate_all) then
                    do i=1,j
                        self%kinetic_matrix(i,j) = self%orbitals(i) .dot. self%kin
                        ! self%kinetic_matrix(i,j) = truncate_number(self%kinetic_matrix(i,j), 8) ! used to be 8, lnw
                        self%kinetic_matrix(j,i) = self%kinetic_matrix(i, j)
                        self%overlap_matrix(i,j) = self%orbitals(i) .dot. self%orbitals(j)
                        self%overlap_matrix(j,i) = self%overlap_matrix(i, j)
                        self%nuclear_electron_matrix(i,j) = self%orbitals(i) .dot. self%temp
                        self%nuclear_electron_matrix(j,i) = self%nuclear_electron_matrix(i,j)
                    end do
                else
                    self%kinetic_matrix(j,j) = temp_function .dot. self%kin
                    ! self%kinetic_matrix(j,j) = truncate_number(self%kinetic_matrix(j,j), 8) ! used to be 8, lnw
                    call temp_function%destroy()
                    self%overlap_matrix(j,j) = self%orbitals(j) .dot. self%orbitals(j)
                    self%nuclear_electron_matrix(j,j) = self%orbitals(j) .dot. self%temp
                end if
            end do
            call self%kin%destroy()
            deallocate(self%kin)
            nullify(nuclear_potential)
            call bigben%stop()
            call stop_memory_follower()
#ifdef HAVE_CUDA_PROFILING
            call stop_nvtx_timing()
#endif
        end if
write(*,*) 'end SCFCycle_calculate_one_electron_mats'
    end subroutine
    
    
    !> Apply the orbital coefficients to the basis function one electron matrices to get 
    !! the orbital one electron matrices. The results are stored to the input matrices.
    subroutine SCFCycle_calculate_orbital_one_electron_matrices(self, coefficients, kinetic_matrix, &
                                                                overlap_matrix, nuclear_electron_matrix)
        class(SCFCycle),  intent(inout)    :: self
        !> The orbital coefficients
        real(REAL64),     intent(in)       :: coefficients(:, :)
        !> In: The kinetic matrix for basis functions.
        !! Out: The kinetic matrix for orbitals.
        real(REAL64),     intent(inout)    :: kinetic_matrix(:, :)
        !> In:  The overlap matrix for basis functions
        !! Out: The overlap matrix for orbitals
        real(REAL64),     intent(inout)    :: overlap_matrix(:, :)
        !> In: The nuclear electron attraction matrix for basis functions
        !! Out: The nuclear electron attraction matrix for orbitals
        real(REAL64),     intent(inout)    :: nuclear_electron_matrix(:, :)
        integer                            :: iorbital, jorbital, i, j
        real(REAL64)                       :: new_coefficients(size(coefficients, 1), size(coefficients, 2)), &
                                              new_kinetic_matrix(size(kinetic_matrix, 1), size(kinetic_matrix, 2)), &
                                              new_overlap_matrix(size(overlap_matrix, 1), size(overlap_matrix, 2)), &
                                              new_nuclear_electron_matrix(size(nuclear_electron_matrix, 1), &
                                                                          size(nuclear_electron_matrix, 2))
        
        new_kinetic_matrix = 0.0d0
        new_overlap_matrix = 0.0d0
        new_nuclear_electron_matrix = 0.0d0
        
        do iorbital = 1, size(coefficients, 2)
            do jorbital = 1, iorbital
                do i = 1, size(coefficients, 1)
                    do j = 1, size(coefficients, 1)
                        ! get the new kinetic energies
                        new_kinetic_matrix(jorbital, iorbital) = new_kinetic_matrix(jorbital, iorbital) &
                            + coefficients(i, iorbital) * coefficients(j, jorbital) * kinetic_matrix(i, j)
                        
                        ! get the new overlaps
                        new_overlap_matrix(jorbital, iorbital) = new_overlap_matrix(jorbital, iorbital) &
                            + coefficients(i, iorbital) * coefficients(j, jorbital) * overlap_matrix(i, j)
                        
                        
                        ! get the new nuclear electron attraction energies
                        new_nuclear_electron_matrix(jorbital, iorbital) = new_nuclear_electron_matrix(jorbital, iorbital) &
                            + coefficients(i, iorbital) * coefficients(j, jorbital) * nuclear_electron_matrix(i, j)
                    end do
                    
                end do
                
                new_kinetic_matrix(iorbital, jorbital) = new_kinetic_matrix(jorbital, iorbital)
                new_overlap_matrix(iorbital, jorbital) = new_overlap_matrix(jorbital, iorbital)
                new_nuclear_electron_matrix(iorbital, jorbital) = new_nuclear_electron_matrix(jorbital, iorbital)
            end do
        end do
        kinetic_matrix(:, :) = new_kinetic_matrix(:, :)
        overlap_matrix(:, :) = new_overlap_matrix(:, :)
        nuclear_electron_matrix(:, :) = new_nuclear_electron_matrix(:, :)
    end subroutine
    
    !> Apply the orbital coefficients to the basis function one electron matrices to get 
    !! the orbital one electron matrices. The results are stored to the input matrices.
    subroutine SCFCycle_calculate_orbital_two_electron_integrals(self, coefficients, two_electron_integrals)
        class(SCFCycle),  intent(inout)    :: self
        !> The orbital coefficients
        real(REAL64),     intent(in)       :: coefficients(:, :)
        !> In: The two electron integrals in basis function format.
        !! Out: The two electron integrals in orbital format. 
        real(REAL64),     intent(inout)    :: two_electron_integrals(:, :, :, :)
        integer                            :: iorbital, jorbital, korbital, lorbital, i, j, k, l
        real(REAL64)                       :: new_two_electron_integrals(size(two_electron_integrals, 1), &
                                                                         size(two_electron_integrals, 2), &
                                                                         size(two_electron_integrals, 3), &
                                                                         size(two_electron_integrals, 4))
        real(REAL64)                       :: temp_two_electron_integrals(size(two_electron_integrals, 1), &
                                                                          size(two_electron_integrals, 2), &
                                                                          size(two_electron_integrals, 3), &
                                                                          size(two_electron_integrals, 4))
        
        new_two_electron_integrals = 0.0d0
        
        do lorbital = 1, size(coefficients, 2)
            do l = 1, size(coefficients, 1)
                new_two_electron_integrals(:, :, :, lorbital) = &
                        new_two_electron_integrals(:, :, :, lorbital) &
                      + coefficients(l, lorbital) * two_electron_integrals(:, :, :, l)
            end do
        end do
        
        
        temp_two_electron_integrals(:, :, :, :) = new_two_electron_integrals(:, :, :, :)
        new_two_electron_integrals(:, :, :, :) = 0.0d0
        
        do korbital = 1, size(coefficients, 2)
            do k = 1, size(coefficients, 1)
                new_two_electron_integrals(:, :, korbital, :) = &
                    new_two_electron_integrals(:, :, korbital, :) &
                  + coefficients(k, korbital) * temp_two_electron_integrals(:, :, k, :)
            end do
        end do
        
        temp_two_electron_integrals(:, :, :, :) = new_two_electron_integrals(:, :, :, :)
        new_two_electron_integrals(:, :, :, :) = 0.0d0
        
        do jorbital = 1, size(coefficients, 2)
           do j = 1, size(coefficients, 1)
                new_two_electron_integrals(:, jorbital, :, :) = &
                    new_two_electron_integrals(:, jorbital, :, :) &
                  + coefficients(j, jorbital) * temp_two_electron_integrals(:, j, :, :)
           end do
        end do
        
        temp_two_electron_integrals(:, :, :, :) = new_two_electron_integrals(:, :, :, :)
        new_two_electron_integrals(:, :, :, :) = 0.0d0
                    
        do iorbital = 1, size(coefficients, 2)
            do i = 1, size(coefficients, 1)
                new_two_electron_integrals(iorbital, :, :, :) = &
                         new_two_electron_integrals(iorbital, :, :, :) &
                    +    coefficients(i, iorbital) &
                       * temp_two_electron_integrals(i, :, :, :)
            end do
        end do
        two_electron_integrals(:, :, :, :) = new_two_electron_integrals(:, :, :, :)
    end subroutine
    
    !> Get the coulomb energy for a bracket element of a calculation for 
    !! a configuration with occupations specified in 'occupations'. In order to use this function
    !! the two electron integrals must be evaluated for orthonormal orbitals, not for basis functions.
    function SCFCycle_get_coulomb_energy(self, iorbital, jorbital, occupations) result(coulomb_energy)
        class(SCFCycle), intent(in) :: self
        integer,         intent(in) :: iorbital
        integer,         intent(in) :: jorbital
        integer,         intent(in) :: occupations(:)
        integer                     :: k
        real(REAL64)                :: coulomb_energy
        
        coulomb_energy = 0.0d0
        do k = 1, size(occupations)
            !print '(i1, " ", i1, "|" i1, " ", i1, ":", f24.16)', iorbital, jorbital, k, k, &
            !         self%two_electron_integrals(iorbital, jorbital, k, k)
            coulomb_energy = coulomb_energy + abs(occupations(k)) *  self%two_electron_integrals(iorbital, jorbital, k, k)
        end do
    
    end function
    
    !> Get the total coulomb energy for a configuration with occupations specified in 'occupations'.
    !! In order to use this function the two electron integrals must be evaluated for orthonormal orbitals,
    !! not for basis functions.
    function SCFCycle_get_total_coulomb_energy(self, occupations) result(coulomb_energy)
        class(SCFCycle), intent(in) :: self
        integer,         intent(in) :: occupations(:)
        integer                     :: i
        real(REAL64)                :: coulomb_energy
        
        coulomb_energy = 0.0d0
        do i = 1, size(occupations)
            coulomb_energy = coulomb_energy + abs(occupations(i)) * self%get_coulomb_energy(i, i, occupations)
        end do
        coulomb_energy = 0.5d0 * coulomb_energy
    
    end function
    
    !> Get the exchange energy for a bracket element of a calculation for 
    !! a configuration with occupations specified in 'occupations'. In order to use this function
    !! the two electron integrals must be evaluated for orthonormal orbitals, not for basis functions.
    function SCFCycle_get_exchange_energy(self, iorbital, jorbital, occupations) result(exchange_energy)
        class(SCFCycle), intent(in) :: self
        integer,         intent(in) :: iorbital
        integer,         intent(in) :: jorbital
        integer,         intent(in) :: occupations(:)
        integer                     :: k
        real(REAL64)                :: exchange_energy
        
        exchange_energy = 0.0d0
        do k = 1, size(occupations)
            !print '(i1, " ", i1, "|" i1, " ", i1, ":", f24.16)', iorbital, k, k, iorbital, &
            !          self%two_electron_integrals(iorbital, k, k, jorbital)
            exchange_energy =    exchange_energy &
                              +  abs(occupations(k)) *  self%two_electron_integrals(iorbital, k, k, jorbital)
        end do
    
    end function
    
    !> Get the total exchange energy for a configuration with occupations specified in 'occupations'.
    !! In order to use this function the two electron integrals must be evaluated for orthonormal orbitals,
    !! not for basis functions.
    function SCFCycle_get_total_exchange_energy(self, occupations) result(exchange_energy)
        class(SCFCycle), intent(in) :: self
        integer,         intent(in) :: occupations(:)
        integer                     :: spin_occupations(size(occupations), 2)
        integer                     :: i
        real(REAL64)                :: exchange_energy
        
        exchange_energy = 0.0d0
        spin_occupations(:, 1) = (occupations(:) + 1) / 2
        spin_occupations(:, 2) = abs(occupations(:) - spin_occupations(:, 1))
        do i = 1, size(occupations)
            exchange_energy =   exchange_energy &
                              + spin_occupations(i, 1) * self%get_exchange_energy(i, i, spin_occupations(:, 1)) &
                              + spin_occupations(i, 2) * self%get_exchange_energy(i, i, spin_occupations(:, 2))
        end do
        exchange_energy = -0.5d0 * exchange_energy
    
    end function

    !> Print the current orbital energies and possibly the components 
    !! affecting to the energy.
    subroutine SCFCycle_print_orbital_energies(self, print_components)
        class(SCFCycle), intent(in) :: self
        !> If the components affecting to the orbital energies are printed
        logical,         intent(in) :: print_components
        integer                     :: i

        
write(*,*) 'kin,pot,eval', self%kinetic_matrix(1, 1), self%nuclear_electron_matrix(1,1), self%eigen_values(1)

        if (print_components) then
            print '("-----------------------------------------------------------------&
                      &-----------------------------")'
            print  '("| ", a10, " | ", a13, " | ", a13, " | ", a13, " | ", a13, " | ", a13, " |")', &
                    "Orbital #", "Kinetic", "Nucl-Elec", "Elec-Elec.", "Exch. (& Corr.)", "Total"
            print '("-----------------------------------------------------------------&
                      &-----------------------------")'
            do i = 1, size(self%coulomb_matrix, 1)
                print '("| ", i10, " | ", ES13.6, " | ", ES13.6, " | ", ES13.6, " | ", ES13.6, " | ", ES13.6, " |")', &
                      i, self%kinetic_matrix(i, i), self%nuclear_electron_matrix(i,i), self%coulomb_matrix(i, i), &
                      self%exchange_matrix(i, i), self%eigen_values(i)
            end do
            print '("-----------------------------------------------------------------&
                      &-----------------------------")'
        else
            print '("-------------------------------------------------------------")'
            do i = 1, size(self%coulomb_matrix, 1)
                if (mod(i, 3) == 0) then
                    write (*, '("| ", i3, ": ", ES12.5, " |")') i, self%eigen_values(i)
                else
                    write (*, '("| ", i3, ": ", ES12.5, " ")', advance = 'no')  i, self%eigen_values(i)
                end if
            end do

            ! complete the last row with blank fields
            if (mod(size(self%coulomb_matrix, 1), 3) /= 0) then
                do i = mod(size(self%coulomb_matrix, 1), 3)+1, 3
                    if (i == 3) then
                        write (*, '("| ", a17, " |")') " "
                    else
                        write (*, '("| ", a17, " ")', advance = 'no') " "
                    end if
                end do
            end if
            print '("-------------------------------------------------------------")'
        end if
    end subroutine

!--------------------------------------------------------------------!
!        Helmholtz SCF Cycle  function implementations               !
!--------------------------------------------------------------------! 

    !> Calculate electron density from  orbitals stored in 'orbitals' and
    !! using the occupations given in input parameter 'occupations'. The result 
    !! electron density is stored in 'electron_density'.
    subroutine HelmholtzSCFCycle_calculate_electron_density_worker(self, orbitals, occupations, electron_density)
        class(HelmholtzSCFCycle),           intent(inout) :: self
        !> The orbitals from which the electron density is calculated
        type(Function3D),                   intent(inout) :: orbitals(:)
        !> The occupation numbers of the orbitals
        real(REAL64),                       intent(in)    :: occupations(:)
        !> The result electron density
        type(Function3D),                   intent(inout) :: electron_density
        integer                                           :: iorbital

        if (allocated(self%temp)) then
            call self%temp%destroy()
            deallocate(self%temp)
        end if                
        if (allocated(self%temp2)) then
            call self%temp2%destroy()
            deallocate(self%temp2)
        end if 

        do iorbital = 1, size(orbitals)
            if (abs(occupations(iorbital)) >  1d-9) then
                call orbitals(iorbital)%precalculate_taylor_series_bubbles()
                call orbitals(iorbital)%multiply_sub(orbitals(iorbital), self%temp)
                call self%temp%product_in_place_REAL64(abs(occupations(iorbital)))
                
                if (.not. allocated(self%temp2)) then
                    allocate(Function3D :: self%temp2)
                    call self%temp2%init_copy(self%temp, copy_content = .TRUE.)
                else 
                    call self%temp2%add_in_place(self%temp)
                end if
            end if
        end do
        call electron_density%destroy()
        call electron_density%init_copy(self%temp2, copy_content = .TRUE.)
        electron_density%type = F3D_TYPE_CUSP
#ifdef HAVE_CUDA    
        ! make sure that we are all finished before communicating the electron density between
        ! computational nodes
        call CUDASync_all()
#endif
        if (allocated(self%temp)) then
            call self%temp%destroy()
            deallocate(self%temp)
        end if                
        if (allocated(self%temp2)) then
            call self%temp2%destroy()
            deallocate(self%temp2)
        end if 
        call electron_density%communicate_cube(reversed_order = .TRUE.)
        
write(*,*) 'end HelmholtzSCFCycle_calculate_electron_density_worker'
    end subroutine 

    
    !> Performs L'o'wdin Orthogonalization to orbitals of 'self'.
    !! Assumes that the orbitals are already normalized.
    subroutine HelmholtzSCFCycle_lowdin_orthogonalize_orbitals(self)
        class(HelmholtzSCFCycle), intent(inout) :: self

        call self%calculate_overlap_matrix()
        self%coefficients = get_orthogonalizing_matrix(self%overlap_matrix(:self%nocc, :self%nocc))
        call self%update_orbitals_linear_combination_worker(self%nocc)
    end subroutine
    

    !> Calculates the overlap matrix of the self%orbitals
    subroutine HelmholtzSCFCycle_calculate_overlap_matrix(self)
        class(HelmholtzSCFCycle), intent(inout) :: self
        integer                                 :: i,j
        
        if (size(self%orbitals) > 0) then
#ifdef HAVE_CUDA_PROFILING
            call start_nvtx_timing("Overlap Matrix")
#endif
            call start_memory_follower()
            !call pinfo("Computing overlap matrix")
            call bigben%split("Calculating overlap matrix")
            do j=1, size(self%orbitals)
                do i=1,j
                    self%overlap_matrix(i,j) = self%orbitals(i) .dot. self%orbitals(j)
                    self%overlap_matrix(j,i) = self%overlap_matrix(i, j)
                end do
            end do
            call bigben%stop()
            call stop_memory_follower()
#ifdef HAVE_CUDA_PROFILING
            call stop_nvtx_timing()
#endif
        end if
    end subroutine

    !> Calculates the two electron integrals needed in evaluation of coulomb and
    !! exchange matrices and stored the results to 'self%two_electron_integrals'.
    subroutine HelmholtzSCFCycle_calculate_two_electron_integrals(self, evaluate_all)
        class(HelmholtzSCFCycle), intent(inout), target :: self
        !> Mode of the evaluation if not evaluate all, then evaluate only the ones needed
        !! in occupied orbital energy evaluation
        logical,                  intent(in)            :: evaluate_all
        class(Function3D), pointer                      :: pot
        real(REAL64)                                    :: temp_value
        integer                                         :: i,j,k,l, orbital_count 

        if (size(self%orbitals) > 0) then
#ifdef HAVE_CUDA_PROFILING
            call start_nvtx_timing("Exchange Matrix")
#endif

            call start_memory_follower()
            !call pinfo("Computing two electron integrals")
            call bigben%split("Computing two electron integrals")
            if (evaluate_all) then
                ! go through all orbitals
                do k=1, size(self%orbitals)
                    call bigben%split("Two electron integrals - Loop")
                    
                    ! multiply orbital 'k' with orbital 'k': |kk> and store the result to self%potential_input
                    call self%orbitals(k)%multiply_sub(self%orbitals(k), self%temp)
                    call self%potential_input%copy_content(self%temp)
                    call self%potential_input%communicate_cube(reversed_order = .TRUE.)
                    
                    ! calculate potential 1/r |kk>
                    call self%coulomb_operator%operate_on(self%potential_input, self%potential2)
                    pot => self%potential2
                    
                    do j=1, size(self%orbitals)
                        call pot%multiply_sub(self%orbitals(j), self%temp)
                        do i = 1, j
                            temp_value = self%orbitals(i) .dot. self%temp
                            call self%set_two_electron_integral_value(i, j, k, k, temp_value) 
                        end do
                    end do
                    nullify(pot)
                    
                    do j=1, k-1
                        ! multiply orbital 'k' with orbital 'j': |kj> and store the result to self%potential_input
                        call self%orbitals(k)%multiply_sub(self%orbitals(j), self%temp)
                        call self%potential_input%copy_content(self%temp)
                        call self%potential_input%communicate_cube(reversed_order = .TRUE.)

                        ! calculate potential 1/r |kj>
                        call self%coulomb_operator%operate_on(self%potential_input, self%potential2)
                        pot => self%potential2
                        

                        ! multiply potential caused by |kj>, 'pot', with orbital 'k' and store the result to self%temp
                        call pot%multiply_sub(self%orbitals(k), self%temp)
                        call self%temp%precalculate_taylor_series_bubbles()
                        
                        do i=1, size(self%orbitals)
                            temp_value = self%orbitals(i) .dot. self%temp
                            call self%set_two_electron_integral_value(i, k, k, j, &
                                    temp_value) 
                        end do

                        nullify(pot)
                        
                    end do
                    call bigben%stop()
                end do
            else
                do i = 1, self%nocc
                    ! calculate |ii>
                    call self%orbitals(i)%multiply_sub(self%orbitals(i), self%temp)
                    call self%potential_input%copy_content(self%temp)
                    call self%potential_input%communicate_cube(reversed_order = .TRUE.)
                    
                    ! calculate potential 1/r |ii>
                    call self%coulomb_operator%operate_on(self%potential_input, self%potential2)
                    pot => self%potential2
                    
                    temp_value = self%temp .dot. pot 
                    call self%set_two_electron_integral_value(i, i, i, i, temp_value) 
                    
                    do j = 1, i-1
                        ! calculate |jj>
                        call self%orbitals(j)%multiply_sub(self%orbitals(j), self%temp)
                        temp_value = self%temp .dot. pot 
                        call self%set_two_electron_integral_value(j, j, i, i, temp_value) 
                    end do
                    
                    do j = 1, i-1
                        ! calculate |ij>
                        call self%orbitals(i)%multiply_sub(self%orbitals(j), self%temp)
                        call self%potential_input%copy_content(self%temp)
                        call self%potential_input%communicate_cube(reversed_order = .TRUE.)
                        
                        ! calculate potential 1/r |ij>
                        call self%coulomb_operator%operate_on(self%potential_input, self%potential2)
                        pot => self%potential2
                        
                        ! and finally [ij|ji]
                        temp_value = self%temp .dot. pot
                        call self%set_two_electron_integral_value(i, j, j, i, &
                                    temp_value) 
                    end do
                end do
                
                do i = self%nocc+1, size(self%orbitals)
                    ! calculate |ii>
                    call self%orbitals(i)%multiply_sub(self%orbitals(i), self%temp)
                    call self%potential_input%copy_content(self%temp)
                    call self%potential_input%communicate_cube(reversed_order = .TRUE.)
                    
                    ! calculate potential 1/r |ii>
                    call self%coulomb_operator%operate_on(self%potential_input, self%potential2)
                    pot => self%potential2
                    
                    temp_value = self%temp .dot. pot 
                    call self%set_two_electron_integral_value(i, i, i, i, temp_value) 
                    
                    do j = 1, self%nocc-1
                        ! calculate |jj>
                        call self%orbitals(j)%multiply_sub(self%orbitals(j), self%temp)
                        
                        ! and finally [ii|jj]
                        temp_value = self%temp .dot. pot
                        call self%set_two_electron_integral_value(j, j, i, i, temp_value) 
                    end do
                    
                    do j = 1, self%nocc-1
                        ! calculate |ij>
                        call self%orbitals(i)%multiply_sub(self%orbitals(j), self%temp)
                        call self%potential_input%copy_content(self%temp)
                        call self%potential_input%communicate_cube(reversed_order = .TRUE.)
                        
                        ! calculate potential 1/r |ij>
                        call self%coulomb_operator%operate_on(self%potential_input, self%potential2)
                        pot => self%potential2
                        
                        temp_value = self%temp .dot. pot
                        call self%set_two_electron_integral_value(i, j, j, i, &
                                    temp_value) 
                    end do
                end do
            end if
            
            call bigben%stop()
            call stop_memory_follower()
#ifdef HAVE_CUDA_PROFILING
            call stop_nvtx_timing()
#endif
        end if
        
    end subroutine

    
    

    !> Form coulomb matrix using pre-calculated two-electron integrals
    !! and store it to self%coulomb_matrix. 
    !! Before this routine is called, 'self%calculate_two_electron_integrals'
    !! has to be executed.
    subroutine HelmholtzSCFCycle_form_coulomb_matrix(self, occupations)
        class(HelmholtzSCFCycle), intent(inout), target  :: self
        integer,                  intent(in), optional   :: occupations(:)
  
        integer                                          :: i,j, k
        integer                                          :: occupations_(size(self%orbitals))
        
        if (present(occupations)) then
            occupations_ = occupations
        else
            occupations_ = 0
            occupations_(:self%nocc) = 1
        end if

        if (size(self%orbitals) > 0) then
            self%coulomb_matrix = 0.0d0
            ! go through all functions self%orbitals
            do j=1, size(self%orbitals)
                ! and all i <= j
                do i=1, j
                    ! evaluate the coulomb matrix value i, j
                    self%coulomb_matrix(i, j) = self%get_coulomb_energy(i, j, occupations_)
                    self%coulomb_matrix(j, i) = self%coulomb_matrix(i, j)
                end do
            end do
        end if
    end subroutine
    
    !> Fill the diagonal coulomb matrix elements using pre-calculated two-electron integrals
    !! and store it to self%coulomb_matrix.
    !! Before this routine is called, 'self%calculate_two_electron_integrals'
    !! has to be executed.
    subroutine HelmholtzSCFCycle_form_diagonal_coulomb_matrix(self)
        class(HelmholtzSCFCycle), intent(inout), target  :: self
  
        integer                                          :: i
        integer                                          :: occupations_(size(self%orbitals))

        if (size(self%orbitals) > 0) then
            self%coulomb_matrix = 0.0d0
            occupations_ = 0
            occupations_(:self%nocc) = 1
            
            ! go through all functions self%orbitals
            do i=1, self%nocc
                self%coulomb_matrix(i, i) = self%get_coulomb_energy(i, i, occupations_)
            end do
            
            ! handle the virtual orbitals
            occupations_(self%nocc) = 0
            do i=self%nocc+1, size(self%orbitals)
                occupations_(self%nocc:) = 0
                occupations_(i) = 1
                self%coulomb_matrix(i, i) = self%get_coulomb_energy(i, i, occupations_)
            end do
        end if
    end subroutine

    !> Form exchange matrix using pre-calculated two-electron integrals and 
    !! store it to self%exchange_matrix.
    !! Before this routine is called, 'self%calculate_two_electron_integrals'
    !! has to be executed.
    subroutine HelmholtzSCFCycle_form_exchange_matrix(self, occupations)
        class(HelmholtzSCFCycle), intent(inout), target     :: self
        integer                                             :: i,j,k
        integer,                  intent(in),    optional   :: occupations(:)
        integer                                             :: occupations_(size(self%orbitals))
        
        if (present(occupations)) then
            occupations_ = occupations
        else
            occupations_ = 0
            occupations_(:self%nocc) = 1
        end if

        if (size(self%orbitals) > 0) then
#ifdef HAVE_CUDA_PROFILING
            call start_nvtx_timing("Exchange Matrix")
#endif

            call start_memory_follower()
            self%exchange_matrix = 0.0d0
            !call pinfo("Forming exchange matrix")
            call bigben%split("Forming exchange matrix")
            ! go through all orbitals
            do j=1, size(self%orbitals)
                do i=1,j
                    ! go through all occupied orbitals
                    self%exchange_matrix(i,j)= self%get_exchange_energy(i, j, occupations_)
                    self%exchange_matrix(j,i) = self%exchange_matrix(i, j)
                end do
            end do
            call bigben%stop()
            call stop_memory_follower()
#ifdef HAVE_CUDA_PROFILING
            call stop_nvtx_timing()
#endif
        end if
    end subroutine
    
    !> Fill the diagonal exchange matrix elements using pre-calculated two-electron integrals
    !! and store it to self%exchange_matrix.
    !! Before this routine is called, 'self%calculate_two_electron_integrals'
    !! has to be executed.
    subroutine HelmholtzSCFCycle_form_diagonal_exchange_matrix(self)
        class(HelmholtzSCFCycle), intent(inout), target  :: self
  
        integer                                          :: i
        integer                                          :: occupations_(size(self%orbitals))

        if (size(self%orbitals) > 0) then
            self%exchange_matrix = 0.0d0
            occupations_ = 0
            occupations_(:self%nocc) = 1
            
            ! go through all functions self%orbitals
            do i=1, self%nocc
                self%exchange_matrix(i, i) = self%get_exchange_energy(i, i, occupations_)
            end do
            
            ! handle the virtual orbitals
            occupations_(self%nocc) = 0
            do i=self%nocc+1, size(self%orbitals)
                occupations_(self%nocc:) = 0
                occupations_(i) = 1
                self%exchange_matrix(i, i) = self%get_exchange_energy(i, i, occupations_)
            end do
        end if
    end subroutine

    
    !> Calculate exchange matrix from scratch for the 'self%orbitals'
    !! and store the result to 'self%exchange_matrix'.
    subroutine HelmholtzSCFCycle_calculate_exchange_matrix(self)
        class(HelmholtzSCFCycle), intent(inout), target :: self
        class(Function3D), pointer                      :: pot
        real(REAL64)                                    :: temp_value
        integer                                         :: i,j,k

        if (size(self%orbitals) > 0) then
#ifdef HAVE_CUDA_PROFILING
            call start_nvtx_timing("Exchange Matrix")
#endif

            call start_memory_follower()
            self%exchange_matrix = 0.0d0
            !call pinfo("Computing exchange matrix")
            call bigben%split("Computing exchange matrix")
            ! go through all occupied orbitals
            do k=1,self%nocc
                ! go through all orbitals
                call bigben%split("Exchange Matrix One Loop")
                do j=1, k
                    print *, "k", k, "j", j
                    ! multiply orbital 'k' with orbital 'j': |kj> and store the result to self%potential_input
                    call self%orbitals(k)%multiply_sub(self%orbitals(j), self%temp)
                    call self%potential_input%copy_content(self%temp)
                    call self%potential_input%communicate_cube(reversed_order = .TRUE.)

                    ! calculate potential 1/r |kj>
                    call self%coulomb_operator%operate_on(self%potential_input, self%potential2)
                    pot => self%potential2
                    

                    ! multiply potential caused by |kj>, 'pot', with orbital 'k' and store the result to self%temp
                    call pot%multiply_sub(self%orbitals(k), self%temp)
                    call self%temp%precalculate_taylor_series_bubbles()
                
                    do i=1,j
                        if (.not. self%two_electron_integral_evaluated(i, k, k, j)) then
                            temp_value = self%orbitals(i) .dot. self%temp
                            call self%set_two_electron_integral_value(i, k, k, j, &
                                temp_value)   
                        end if
                        self%exchange_matrix(i,j)= self%exchange_matrix(i,j) + &
                            self%two_electron_integrals(i, k, k, j)
                        self%exchange_matrix(j,i) = self%exchange_matrix(i, j)
                    end do

                
                    if (k > j) then
                        ! multiply potential caused by |kj>, 'pot', with orbital 'j' and store the result to temp
                        call pot%multiply_sub(self%orbitals(j), self%temp)
                
                        call self%temp%precalculate_taylor_series_bubbles()
                        do i=1,k

                            if (.not. self%two_electron_integral_evaluated(i, j, j, k)) then
                                temp_value = self%orbitals(i) .dot. self%temp
                                call self%set_two_electron_integral_value(i, j, j, k, &
                                    temp_value)   
                            end if
                            self%exchange_matrix(i,k)= self%exchange_matrix(i,k) + &
                                self%two_electron_integrals(i, j, j, k)
                            self%exchange_matrix(k,i) = self%exchange_matrix(i, k)

                        end do
                    end if

                    nullify(pot)
                    
                end do
                call bigben%stop()
            end do
            call bigben%stop()
            call bigben%stop()
            call stop_memory_follower()
#ifdef HAVE_CUDA_PROFILING
            call stop_nvtx_timing()
#endif
        end if
    end subroutine

    


    

    ! more complex orbital update method
    ! fock matrix is formed with phi_old and phi_trial and then diagonalized
    subroutine HelmholtzSCFCycle_subspace_optimize(self, &
        new_orbital_energy, new_orbital, old_orbital_energy, old_orbital, &
        trial_orbital_energy, trial_orbital, old_orbital_potential, &
        trial_orbital_potential)

        class(HelmholtzSCFCycle)                     :: self
        real(REAL64),                  intent(out)   :: new_orbital_energy
        type(Function3D), allocatable, intent(out)   :: new_orbital
        real(REAL64),                  intent(in)    :: old_orbital_energy
        type(Function3D),              intent(in)    :: old_orbital
        real(REAL64),                  intent(in)    :: trial_orbital_energy
        type(Function3D),              intent(in)    :: trial_orbital
        type(Function3D),              intent(in)    :: old_orbital_potential
        type(Function3D),              intent(in)    :: trial_orbital_potential
        class(Function3D), allocatable               :: poly_temp
        integer                                      :: loc, i,j
        real(REAL64)                                 :: old_energy, kinetic_energy
        real(REAL64)                                 :: norm_squared
        real(REAL64), allocatable                    :: hamiltonian_matrix(:,:), overlap_matrix(:,:), &
                                                 orbital_coefficients(:,:), eigen_values(:)

        allocate(hamiltonian_matrix(2, 2))
        allocate(overlap_matrix(2, 2))
        allocate(orbital_coefficients(2, 2))
        allocate(eigen_values(2))

        ! forming the overlap matrix between phi_old and phi_trial
        write(*,*) 'calculating overlap'

        overlap_matrix(1,1) = old_orbital .dot. old_orbital
        overlap_matrix(1,2) = old_orbital .dot. trial_orbital
        overlap_matrix(2,1) = overlap_matrix(1,2)
        overlap_matrix(2,2) = trial_orbital .dot. trial_orbital

        print *, "overlap_matrix", overlap_matrix
        ! Initialize the matrix for orbital coefficients for the updated orbital
        orbital_coefficients = 0d0
        eigen_values = 0.0d0

        ! form the small hamiltonian atrix
        hamiltonian_matrix(1, 1) = old_orbital_energy
        poly_temp = self%laplacian_operator .apply. old_orbital
        kinetic_energy = -0.5d0 * (trial_orbital .dot. &
                        (self%laplacian_operator .apply. old_orbital))
        print *, "kinetic energy, 2, 1", kinetic_energy
        hamiltonian_matrix(2, 1) =  kinetic_energy &
               + (trial_orbital .dot. old_orbital_potential)
        hamiltonian_matrix(1, 2) = hamiltonian_matrix(2, 1)
        call poly_temp%destroy()
        deallocate(poly_temp)
        
        kinetic_energy = -0.5d0 * (old_orbital .dot. &
                        (self%laplacian_operator .apply. trial_orbital))
        print *, "kinetic energy, 2, 1", kinetic_energy
        hamiltonian_matrix(1, 2) = kinetic_energy + &
               (old_orbital .dot. trial_orbital_potential)
        !kinetic_energy = -0.5d0 * (trial_orbital .dot. &
        !                (self%laplacian_operator .apply. trial_orbital))
        hamiltonian_matrix(2, 2) = trial_orbital_energy !kinetic_energy & 
        !   +  (trial_orbital .dot. trial_orbital_potential)
        !call poly_temp%destroy()
        !deallocate(poly_temp)

        print *, "hamiltonian_matrix", hamiltonian_matrix

        ! check if hamiltonian matrix or overlap matrix contain nans, if yes
        ! set the corresponding items to 0
        do j=1,2
            do i=1,2
                if(isNaN(hamiltonian_matrix(i,j))) hamiltonian_matrix(i,j) = 0d0
                if(isNaN(overlap_matrix(i,j))) overlap_matrix(i,j) = 0d0
            end do
        end do

        write(*,*) 'diagonalizing'
    
        ! diagonalize the subspace
        call matrix_generalized_eigensolver(hamiltonian_matrix, overlap_matrix, &
                                            orbital_coefficients, eigen_values)
        loc = minloc(abs(eigen_values-old_orbital_energy), rank(eigen_values))

        ! check if orbital coefficients contain nans, if yes set the nan to 0
        do j=1,2
            do i=1,2
                if (isNaN(orbital_coefficients(i,j))) orbital_coefficients(i,j) =0d0
            end do
        end do

        write(*,*) 'eigen values', eigen_values
        write(*,*) 'orbital coefficients', orbital_coefficients
        if (sum(orbital_coefficients(:, loc)) < 0.0d0) then
            orbital_coefficients(:, loc) = orbital_coefficients(:, loc) * (-1.0d0)
        end if

        ! do linear combination of the two input orbitals
        new_orbital = ( orbital_coefficients(1,loc) * old_orbital ) + &
                      ( orbital_coefficients(2,loc) * trial_orbital ) 
        norm_squared = new_orbital .dot. new_orbital

        ! if the normalization factor is sensible, normalize the new orbital
        if (norm_squared > 0d0) then
            ! normalize the new orbital
            new_orbital = new_orbital / sqrt(norm_squared)

            ! set the result orbital
            new_orbital_energy = eigen_values(loc)

            ! check for positive energy
            if( eigen_values(loc) > 0d0 ) then
                new_orbital = old_orbital
                new_orbital_energy = old_orbital_energy
            end if

        else
            new_orbital = old_orbital
            new_orbital_energy = old_orbital_energy
        end if
        deallocate(overlap_matrix, hamiltonian_matrix, eigen_values, orbital_coefficients)

    end subroutine

    !> Worker routine used to update the orbitals using the bound state
    !! helmholtz method. This routine is called by the extending
    !! classes. 
    !!
    !! Note: The orbital potentials must be calculated before calling this routine
    subroutine HelmholtzSCFCycle_update_orbitals_worker(self, scaling_factor)
        class(HelmholtzSCFCycle), intent(inout), target :: self
        !> The factor used to scale the update.
        real(REAL64),     intent(in)            :: scaling_factor
        integer                                 :: iorbital, jorbital, i, j
        type(Function3D), allocatable           :: update_potentials(:), &
                                                   update_potential, orbital_update, &
                                                   orbital, new_orbital, orbital_potential, &
                                                   old_orbitals(:), new_orbital_potential, &
                                                   scaled_update
        type(Function3D)                        :: temp
        class(Function3D), pointer              :: helmholtz_potential
        real(REAL64), allocatable               :: normalizing_factors(:)
        real(REAL64)                            :: energy_update, new_orbital_energy, &
                                                   old_orbital_energy, normalizing_factor, &
                                                   temp_val, original_eigen_value
        integer                                 :: k, id, l, m
        integer, allocatable                    :: order(:)

        call self%get_energy_order(order)
        allocate(normalizing_factors(size(self%orbitals)))
        
        call memoryfollower_print_status()

#ifdef HAVE_CUDA
        ! the following two calls are empty unless GBFMM is used

        ! the usage of the coulomb operator is done for this cycle, thus
        ! we can relieve the memory load caused by it
        call self%coulomb_operator%cuda_unprepare()

        ! the usage of the helmholtz operator is starting from here, thus
        ! we are initalizing some components
        call self%helmholtz_operator%cuda_prepare()
#endif
        do i = 1, size(self%orbitals)

            call memoryfollower_print_status()
            original_eigen_value = self%eigen_values(order(i))
            !call self%orbital_potentials(order(i))%multiply_sub(self%orbitals(order(i)), self%temp)
            ! use the orbital potential as the input for the helmholtz operator
            call self%potential_input%copy_content(self%orbital_potentials(order(i)))

            ! modify the potential with a slight shift to have the input energy to
            ! be negative
            if (self%eigen_values(order(i)) > 0d0) then
                call temp%init_copy(self%orbitals(order(i)), copy_content = .TRUE.)
                call temp%product_in_place_REAL64(self%eigen_values(order(i)) + 0.1d0)
                call self%potential_input%subtract_in_place(temp)
                self%eigen_values(order(i)) = - 0.1d0
                call temp%destroy()
            end if
            
            ! communicate the needed data with other processes (when there is only one process, 
            ! there is no communication)
            call self%potential_input%communicate_cube(reversed_order = .TRUE.)
            call self%helmholtz_operator%set_energy(self%eigen_values(order(i)))
                
            call self%helmholtz_operator%operate_on(self%potential_input, &
                                                    self%helmholtz_potential)


            helmholtz_potential => self%helmholtz_potential
            !self%eigen_values(order(i)) = self%eigen_values(order(i)) - 0.1d0
            call helmholtz_potential%product_in_place_REAL64((-1.0d0 / (2.0d0 * PI)))
            call helmholtz_potential%communicate_cube_borders(reversed_order = .TRUE.)
            ! temp_val = self%helmholtz_potential%integrate() 

            ! normalizing_factor = sqrt(helmholtz_potential .dot. helmholtz_potential)
            ! normalizing_factor = anint(1.0d14 * normalizing_factor) / 1.0d14
            ! print *, "orbital update normalizing factor 0", normalizing_factor
            
            allocate(Function3D :: orbital_update)
            call orbital_update%init_copy(helmholtz_potential, copy_content = .TRUE.)
            call orbital_update%dispose_extra_bubbles(13)
            call orbital_update%inject_extra_bubbles(self%orbitals(order(i))%bubbles%get_lmax())
            call orbital_update%subtract_in_place(self%orbitals(order(i)))
            

            call orbital_update%communicate_cube_borders(reversed_order = .TRUE.)
            call orbital_update%precalculate_taylor_series_bubbles()

            ! get the norm of the update
            ! normalizing_factor = sqrt(orbital_update .dot. orbital_update)
            ! normalizing_factor = anint(1.0d14 * normalizing_factor) / 1.0d14
            ! print *, "orbital update normalizing factor", normalizing_factor, 'scaling_factor', scaling_factor
 
            ! scale the update with the input 'scaling_factor'
            call orbital_update%product_in_place_REAL64(scaling_factor)

            ! form the new orbital by adding the update to the old orbital
            call self%orbitals(order(i))%add_in_place(orbital_update)
            call self%orbitals(order(i))%precalculate_taylor_series_bubbles()
            
            ! normalize the new orbital
            normalizing_factors(order(i)) = 1.0d0 / sqrt(self%orbitals(order(i)) .dot. self%orbitals(order(i)))
            !print *, "norm square1", normalizing_factors(order(i))**2
            call self%orbitals(order(i))%product_in_place_REAL64(normalizing_factors(order(i)))
            call orbital_update%destroy()
            deallocate(orbital_update)
 
            ! set new orbital                        
            call self%orbitals(order(i))%precalculate_taylor_series_bubbles()
            
            ! undo the change to orbital energy made due to level shifting of orbital
            self%eigen_values(order(i)) = original_eigen_value

            call memoryfollower_print_status()
        end do
 
#ifdef HAVE_CUDA
        ! the usage of the helmholtz operator is done for this cycle, thus
        ! we can relieve the memory load caused by it
        call self%helmholtz_operator%cuda_unprepare()
#endif
        
        call memoryfollower_print_status()
        do i = 1, size(self%orbitals)  
            call self%orbital_potentials(i)%destroy()
        end do 
        self%coulomb_potential_is_valid = .FALSE.
        self%orbital_potentials_are_valid = .FALSE.
        deallocate(order)
        deallocate(self%orbital_potentials)
        deallocate(normalizing_factors)
        
        call memoryfollower_print_status()
    end subroutine
    

    !> Destroy and deallocate the things defined in 
    !! HelmholtzSCFCycle
    subroutine HelmholtzSCFCycle_destroy(self)
        class(HelmholtzSCFCycle), intent(inout) :: self
        integer                                 :: i
        
        call SCFCycle_destroy(self)
        
        if (allocated(self%orbital_potentials)) then
            do i = 1, size(self%orbital_potentials)  
                call self%orbital_potentials(i)%destroy()
            end do 
            deallocate(self%orbital_potentials)
        end if
        
        if (allocated(self%helmholtz_potential)) then
            call self%helmholtz_potential%destroy()
            deallocate(self%helmholtz_potential)
        end if
        
        nullify(self%helmholtz_operator)
        
    end subroutine


!--------------------------------------------------------------------!
!        Restricted Helmholtz SCF Cycle  function implementations    !
!--------------------------------------------------------------------! 

    subroutine RestrictedHelmholtzSCFCycle_init(self, orbitals, electron_density, &
                           nocc_a, nocc_b, virtual_orbital_count, &
                           laplacian_operator, coulomb_operator, helmholtz_operator, core_evaluator) 
        !> The initialized object
        class(RestrictedHelmholtzSCFCycle), intent(inout), target :: self
        !> The input orbitals
        type(Function3D), intent(in)                  :: orbitals(:)
        !> Input electron density holder
        type(Function3D), intent(in)                  :: electron_density
        !> Number of occupied orbitals with a spin
        integer, intent(in)                           :: nocc_a
        !> Number of occupied orbitals with b spin
        integer, intent(in)                           :: nocc_b
        !> Number of virtual orbitals
        integer, intent(in)                           :: virtual_orbital_count
        !> Used laplacian operator
        class(Laplacian3D), intent(inout), target     :: laplacian_operator
        !> Used coulomb operator
        class(Coulomb3D), intent(inout),   target     :: coulomb_operator
        !> Used helmholtz operator
        class(Helmholtz3D), intent(inout), target     :: helmholtz_operator
        !> Evaluator to evaluate the cubes more accurately near core
        type(CoreEvaluator), intent(inout), target    :: core_evaluator

        integer                                       :: k, iorbital, orbital_count
        
        
        orbital_count = min(nocc_a + virtual_orbital_count, size(orbitals))
        self%total_orbital_count = orbital_count

        if (nocc_b > nocc_a) then
            print "('ERROR: The number of b occupied orbitals (', i3,') exceeds the number of occupied a orbitals&
                    & (', i3,'). This is not allowed.')", nocc_b, nocc_a
            stop
        end if
        
        allocate(self%orbitals_a(orbital_count))
        allocate(self%all_orbitals(orbital_count))
        do iorbital = 1, orbital_count
            call self%orbitals_a(iorbital)%init_copy(orbitals(iorbital),  copy_content = .TRUE.)
            self%all_orbitals(iorbital)%p => self%orbitals_a(iorbital)
        end do
        
        allocate(Function3D :: self%electron_density_a)
        call self%electron_density_a%init_copy(electron_density, copy_content = .TRUE.)
        allocate(Function3D :: self%potential_input)
        call self%potential_input%init_copy(electron_density, lmax = 12)
        
        self%laplacian_operator => laplacian_operator
        self%coulomb_operator   => coulomb_operator
        self%helmholtz_operator => helmholtz_operator

        
        self%nocc_a = nocc_a
        self%nocc_b = nocc_b
        self%nvir_a = virtual_orbital_count
        self%nvir_b = virtual_orbital_count
        allocate(self%kinetic_matrix_a    (size(self%orbitals_a),size(self%orbitals_a)))
        allocate(self%nuclear_electron_matrix_a (size(self%orbitals_a),size(self%orbitals_a)))
        allocate(self%coulomb_matrix_a    (size(self%orbitals_a),size(self%orbitals_a)))
        allocate(self%overlap_matrix_a    (size(self%orbitals_a),size(self%orbitals_a)))
        allocate(self%exchange_matrix_a   (size(self%orbitals_a),size(self%orbitals_a)))
        allocate(self%hamiltonian_matrix_a(size(self%orbitals_a),size(self%orbitals_a)))   
        allocate(self%coefficients_a      (size(self%orbitals_a),size(self%orbitals_a)))  
        allocate(self%eigen_values_a      (size(self%orbitals_a)), source = 0.0d0)
        allocate(self%two_electron_integrals_a(orbital_count,orbital_count, &
                                            orbital_count,orbital_count))
        allocate(self%two_electron_integral_evaluated_a(orbital_count,orbital_count, &
                                            orbital_count,orbital_count), source = .FALSE.)  

        self%core_evaluator => core_evaluator

        ! set the pointers to correct arrays
        self%electron_density => self%electron_density_a
        self%kinetic_matrix => self%kinetic_matrix_a
        self%nuclear_electron_matrix => self%nuclear_electron_matrix_a
        self%coulomb_matrix => self%coulomb_matrix_a
        self%exchange_matrix => self%exchange_matrix_a
        self%overlap_matrix => self%overlap_matrix_a
        self%hamiltonian_matrix => self%hamiltonian_matrix_a
        self%coefficients => self%coefficients_a
        self%eigen_values => self%eigen_values_a
        self%nocc => self%nocc_a
        self%orbitals => self%orbitals_a
        self%two_electron_integrals => self%two_electron_integrals_a
        self%two_electron_integral_evaluated => self%two_electron_integral_evaluated_a
        
        ! do orthogonalization and set some flags
        call self%gs_orthogonalize_orbitals()
        self%restricted = .TRUE.
        self%coulomb_potential_is_valid = .FALSE.
        self%orbital_potentials_are_valid = .FALSE.
    end subroutine

    

    !> Calculate the closed shell electron density from the 'self%orbitals'.
    subroutine RestrictedHelmholtzSCFCycle_calculate_closed_shell_el_dens(self)
        class(RestrictedHelmholtzSCFCycle), intent(inout) :: self
        real(REAL64)                                      :: occupations(size(self%orbitals))
        integer                                           :: nclosed
write(*,*) 'begin RestrictedHelmholtzSCFCycle_calculate_closed_shell_el_dens'
        
        call bigben%split("Computing closed shell electron density")
        nclosed = min(self%nocc_a, self%nocc_b)
        occupations(:nclosed) = 2
        occupations(nclosed+1:) = 0
        call self%calculate_electron_density_worker(self%orbitals, occupations, self%electron_density)
        call bigben%stop()

write(*,*) 'end RestrictedHelmholtzSCFCycle_calculate_closed_shell_el_dens'
    end subroutine


    !> Calculate the closed shell electron density from the 'self%orbitals'.
    subroutine RestrictedHelmholtzSCFCycle_calculate_open_shell_el_dens(self)
        class(RestrictedHelmholtzSCFCycle), intent(inout) :: self
        real(REAL64)                                      :: occupations(size(self%orbitals))
        integer                                           :: nclosed, nocc
write(*,*) 'begin RestrictedHelmholtzSCFCycle_calculate_open_shell_el_dens'
        
        call bigben%split("Computing open shell electron density")
        nclosed = min(self%nocc_a, self%nocc_b)
        nocc    = max(self%nocc_a, self%nocc_b)
        occupations(:nclosed) = 0
        occupations(nclosed+1:nocc) = 1
        occupations(nocc+1:) = 0
        call self%calculate_electron_density_worker(self%orbitals, occupations, self%electron_density)
        call bigben%stop()
        
write(*,*) 'end RestrictedHelmholtzSCFCycle_calculate_open_shell_el_dens'
    end subroutine


    subroutine RestrictedHelmholtzSCFCycle_update_orbitals(self, scaling_factor)
        class(RestrictedHelmholtzSCFCycle), &
                          intent(inout), target :: self
        real(REAL64),     intent(in)            :: scaling_factor
write(*,*) 'begin RestrictedHelmholtzSCFCycle_update_orbitals'

        ! get orbital potentials, if they are not valid
        if (.not. self%orbital_potentials_are_valid) then
            call self%calculate_orbital_potentials()
        end if

        ! call the worker routine that does the hard work
        call self%update_orbitals_worker(scaling_factor)
        
write(*,*) 'end RestrictedHelmholtzSCFCycle_update_orbitals'
    end subroutine
    
    !> Destroy and deallocate the things defined in 
    !! RestrictedHelmholtzSCFCycle
    subroutine RestrictedHelmholtzSCFCycle_destroy(self)
        class(RestrictedHelmholtzSCFCycle), intent(inout) :: self
        
        call HelmholtzSCFCycle_destroy(self)

        if(allocated(self%electron_density_a)) then
            call self%electron_density_a%destroy()
            deallocate(self%electron_density_a)
        end if
        
        if(allocated(self%coulomb_matrix_a)) deallocate(self%coulomb_matrix_a)
        if(allocated(self%exchange_matrix_a)) deallocate(self%exchange_matrix_a)
        
    end subroutine

!--------------------------------------------------------------------!
!        UnRestricted Helmholtz SCF Cycle  function implementations  !
!--------------------------------------------------------------------! 

    !> Calculate the spin electron densities for unrestricted calculation
    !! from 'self%orbitals_a' and 'self%orbitals_b'. The results are stored 
    !! to 'self%electron_density_a' and 'self%electron_density_b'.
    subroutine UnRestrictedHelmholtzSCFCycle_calculate_spin_densities(self)
        class(UnRestrictedHelmholtzSCFCycle), intent(inout) :: self
        integer                                             :: iorbital
        real(REAL64)                                        :: temp
        real(REAL64)                                        :: occupations_a(size(self%orbitals_a)), &
                                                               occupations_b(size(self%orbitals_b))

        call bigben%split("Computing spin densities")
        ! calculate the a-spin electron density
        if (self%nocc_a > 0) then
            occupations_a(:self%nocc_a) = 1
            occupations_a(self%nocc_a+1:) = 0
            call self%calculate_electron_density_worker(self%orbitals_a, occupations_a, self%electron_density_a)
        end if

        ! calculate the b-spin electron density
        if (self%nocc_b > 0) then
            occupations_b(:self%nocc_b) = 1
            occupations_b(self%nocc_b+1:) = 0
            call self%calculate_electron_density_worker(self%orbitals_b, occupations_b, self%electron_density_b)
        end if
        call bigben%stop()
    end subroutine

    
    subroutine UnRestrictedHelmholtzSCFCycle_calculate_ab_2_el_integrals(self, evaluate_all)
        class(UnRestrictedHelmholtzSCFCycle), intent(inout), target :: self
        logical,                  intent(in)                        :: evaluate_all
        class(Function3D), pointer                                  :: pot
        real(REAL64)                                                :: temp_value
        integer                                                     :: i,j,k

        if (size(self%orbitals_a) > 0 .and. size(self%orbitals_b) > 0) then
#ifdef HAVE_CUDA_PROFILING
            call start_nvtx_timing("Exchange Matrix")
#endif

            call start_memory_follower()
            self%two_electron_integrals_ab = 0.0d0
            !call pinfo("Computing cross spin two-electron matrix")
            call bigben%split("Computing cross spin two-electron matrix")
            ! go through occupied orbitals with spin b
            do i=1, size(self%orbitals_b)
                ! multiply orbital 'k' with orbital 'j': |kj> and store the result to self%potential_input
                call self%orbitals_b(k)%multiply_sub(self%orbitals_b(j), self%temp)
                call self%potential_input%copy_content(self%temp)
                call self%potential_input%communicate_cube(reversed_order = .TRUE.)

                ! calculate potential 1/r |kj>
                call self%coulomb_operator%operate_on(self%potential_input, self%potential2)
                pot => self%potential2

                ! go through all orbitals
                call bigben%split("Cross two electron integrals - Loop")
                do j=1, size(self%orbitals_a)
                    ! multiply potential caused by |kj>, 'pot', with orbital 'k' and store the result to self%temp
                    call pot%multiply_sub(self%orbitals_a(j), self%temp)
                    call self%temp%precalculate_taylor_series_bubbles()
                    
                    temp_value = self%orbitals_a(j) .dot. self%temp
                    self%two_electron_integrals_ab(j, j, i, i) = temp_value
                    self%two_electron_integrals_ba(i, i, j, j) = temp_value  
                    if (evaluate_all) then
                        do k=1,j
                            if (k /= j) then
                                temp_value = self%orbitals_a(k) .dot. self%temp
                                self%two_electron_integrals_ab(j, k, i, i) = temp_value  
                                self%two_electron_integrals_ab(k, j, i, i) = temp_value
                            end if
                        end do
                    end if
                end do
                nullify(pot)
                call bigben%stop()
            end do

            do i=1, size(self%orbitals_a)
                ! multiply orbital 'k' with orbital 'j': |kj> and store the result to self%potential_input
                call self%orbitals_a(i)%multiply_sub(self%orbitals_a(i), self%temp)
                call self%potential_input%copy_content(self%temp)
                call self%potential_input%communicate_cube(reversed_order = .TRUE.)

                ! calculate potential 1/r |kj>
                call self%coulomb_operator%operate_on(self%potential_input, self%potential2)
                pot => self%potential2

                ! go through all orbitals
                call bigben%split("Cross two electron integrals - Loop")
                do j=1, size(self%orbitals_b)
                    ! multiply potential caused by |kj>, 'pot', with orbital 'k' and store the result to self%temp
                    call pot%multiply_sub(self%orbitals_b(j), self%temp)
                    call self%temp%precalculate_taylor_series_bubbles()
                    if (evaluate_all) then
                        do k=1,j
                            if (k /= j) then
                                temp_value = self%orbitals_b(k) .dot. self%temp
                                self%two_electron_integrals_ba(j, k, i, i) = temp_value  
                                self%two_electron_integrals_ba(k, j, i, i) = temp_value 
                            end if
                        end do
                    end if

                    
                end do
                nullify(pot)
                call bigben%stop()
            end do
            call bigben%stop()
            call bigben%stop()
            call stop_memory_follower()
#ifdef HAVE_CUDA_PROFILING
            call stop_nvtx_timing()
#endif
        end if
        
    end subroutine

    subroutine UnRestrictedHelmholtzSCFCycle_update_orbitals(self, scaling_factor)
        class(UnRestrictedHelmholtzSCFCycle), &
                          intent(inout), target :: self
        real(REAL64),     intent(in)            :: scaling_factor
        integer                                 :: iorbital, jorbital, i, j
        type(Function3D), allocatable           :: update_potentials(:), &
                                                   update_potential, orbital_update, &
                                                   orbital, new_orbital, orbital_potential, &
                                                   old_orbitals(:), new_orbital_potential, &
                                                   scaled_update
        type(Function3D)                        :: temp
        class(Function3D), pointer              :: helmholtz_potential
        real(REAL64), allocatable               :: normalizing_factors(:)
        real(REAL64)                            :: energy_update, new_orbital_energy, &
                                                   old_orbital_energy, normalizing_factor, &
                                                   temp_val
        integer                                 :: k
write(*,*) 'begin UnRestrictedHelmholtzSCFCycle_update_orbitals'
        
        call self%calculate_spin_densities()
        call self%electron_density_a%add_in_place(self%electron_density_b)
        self%electron_density => self%electron_density_a

        ! get orbital potentials for orbitals with spin a
        if (size(self%orbitals_a) > 0) then
            self%orbitals => self%orbitals_a
            self%eigen_values => self%eigen_values_a
            self%coefficients => self%coefficients_a
            self%overlap_matrix => self%overlap_matrix_a
            self%nocc => self%nocc_a
            self%kinetic_matrix => self%kinetic_matrix_a
            call self%calculate_orbital_potentials()

            ! call the worker routine
            call self%update_orbitals_worker(scaling_factor)
        end if

        ! get orbital potentials for orbitals with spin b
        if (size(self%orbitals_b) > 0) then
            self%nocc => self%nocc_b
            self%orbitals => self%orbitals_b        
            self%coefficients => self%coefficients_b
            self%overlap_matrix => self%overlap_matrix_b
            self%eigen_values => self%eigen_values_b
            self%kinetic_matrix => self%kinetic_matrix_b
            call self%calculate_orbital_potentials()

            ! call the worker routine
            call self%update_orbitals_worker(scaling_factor)
        end if

write(*,*) 'end UnRestrictedHelmholtzSCFCycle_update_orbitals'
    end subroutine


    !> Diagonalizes orbitals
    subroutine UnRestrictedHelmholtzSCFCycle_diagonalize(self)
        class(UnRestrictedHelmholtzSCFCycle), intent(inout), target :: self

        self%coefficients => self%coefficients_a
        self%hamiltonian_matrix => self%hamiltonian_matrix_a
        self%overlap_matrix => self%overlap_matrix_a
        self%eigen_values => self%eigen_values_a
        self%orbitals => self%orbitals_a
        call self%diagonalize_worker()

        self%coefficients => self%coefficients_b
        self%hamiltonian_matrix => self%hamiltonian_matrix_b
        self%overlap_matrix => self%overlap_matrix_b
        self%eigen_values => self%eigen_values_b
        self%orbitals => self%orbitals_b
        call self%diagonalize_worker()

    end subroutine


    !> Performs update by making a linear combination of molecular orbitals by
    !! using the coefficient store in 'self%coefficients_a' and 'self%coefficients_b'.
    !! The input and output orbitals are stored in 'self%orbitals_a' and 'self%orbitals_b'. 
    subroutine UnRestrictedHelmholtzSCFCycle_update_orbitals_lc(self, number_of_orbitals)
        class(UnRestrictedHelmholtzSCFCycle), target, intent(inout) :: self
        !> Number of orbitals updated via linear combination: NOTE: this parameter is ignored,
        !! but is needed to fit the parent class method definition
        integer, optional,                            intent(in)    :: number_of_orbitals  
        self%coefficients => self%coefficients_a
        self%orbitals => self%orbitals_a
        call self%update_orbitals_linear_combination_worker()

        self%coefficients => self%coefficients_b
        self%orbitals => self%orbitals_b
        call self%update_orbitals_linear_combination_worker()
    end subroutine

    !> Performs Modified Gram-Schmidt Orthogonalization to orbitals of 'self'.
    !! Assumes that the orbitals are already normalized.
    subroutine UnRestrictedHelmholtzSCFCycle_gs_orthogonalize_orbitals(self)
        class(UnRestrictedHelmholtzSCFCycle), &
                          intent(inout), target:: self

        self%orbitals => self%orbitals_a
        self%eigen_values => self%eigen_values_a
        call SCFCycle_gs_orthogonalize_orbitals(self)

        self%orbitals => self%orbitals_b
        self%eigen_values => self%eigen_values_b
        call SCFCycle_gs_orthogonalize_orbitals(self)
        
    end subroutine
    
    !> Destroy and deallocate the things defined in 
    !! UnRestrictedHelmholtzSCFCycle
    subroutine UnRestrictedHelmholtzSCFCycle_destroy(self)
        class(UnRestrictedHelmholtzSCFCycle), intent(inout) :: self
        
        call HelmholtzSCFCycle_destroy(self)

        if(allocated(self%electron_density_a)) then
            call self%electron_density_a%destroy()
            deallocate(self%electron_density_a)
        end if
        
        if(allocated(self%electron_density_b)) then
            call self%electron_density_b%destroy()
            deallocate(self%electron_density_b)
        end if
        
        if(allocated(self%coulomb_matrix_a)) deallocate(self%coulomb_matrix_a)
        if(allocated(self%coulomb_matrix_b)) deallocate(self%coulomb_matrix_b)
        if(allocated(self%coulomb_matrix_ab)) deallocate(self%coulomb_matrix_ab)
        if(allocated(self%coulomb_matrix_ba)) deallocate(self%coulomb_matrix_ba)
        if(allocated(self%exchange_matrix_a)) deallocate(self%exchange_matrix_a)
        if(allocated(self%exchange_matrix_b)) deallocate(self%exchange_matrix_b)
        
    end subroutine




!--------------------------------------------------------------------!
!        RHF Cycle
!--------------------------------------------------------------------!

    !> Initialize the Restricted Hartree Fock Cycle.
    subroutine RHFCycle_init(orbitals, electron_density, &
                           nocc_a, nocc_b, virtual_orbital_count, &
                           laplacian_operator, coulomb_operator, helmholtz_operator, &
                           core_evaluator, result_object)
        !> Input orbitals
        type(Function3D), intent(in)                  :: orbitals(:)
        !> Input electron density holder
        type(Function3D), intent(in)                  :: electron_density
        !> Number of occupied orbitals with a spin
        integer, intent(in)                           :: nocc_a
        !> Number of occupied orbitals with b spin
        integer, intent(in)                           :: nocc_b
        !> Number of virtual orbitals
        integer, intent(in)                           :: virtual_orbital_count
        !> Laplacian operator
        class(Laplacian3D), intent(inout), target     :: laplacian_operator
        class(Coulomb3D), intent(inout),   target     :: coulomb_operator
        class(Helmholtz3D), intent(inout), target     :: helmholtz_operator
        !> Evaluator to evaluate the cubes more accurately near core
        type(CoreEvaluator), intent(inout), target    :: core_evaluator
        ! the result object
        class(SCFCycle), allocatable, target          :: result_object
        type(RHFCycle), pointer                       :: new
        integer                                       :: iorbital, orbital_count

        allocate(RHFCycle :: result_object)

        ! set the non-polymorphic pointer to point to new
        select type(result_object)
            type is (RHFCycle)
                new => result_object
        end select 
        
        call RestrictedHelmholtzSCFCycle_init(new, orbitals, electron_density, &
                           nocc_a, nocc_b, virtual_orbital_count, &
                           laplacian_operator, coulomb_operator, &
                           helmholtz_operator, core_evaluator) 

    end subroutine

    

    subroutine RHFCycle_destroy(self)
        class(RHFCycle), intent(inout)         :: self

        call RestrictedHelmholtzSCFCycle_destroy(self)
    end subroutine

    

  
    
    !> NOTE: the result potentials are not communicated in the end!
    !!       Thus, the user has to take care of that when using the
    !!       potentials
    subroutine RHFCycle_calculate_orbital_potentials(self)
        class(RHFCycle), intent(inout), target :: self
        class(Function3D), pointer             :: pot, coulomb_potential
        type(Function3D), pointer              :: nuclear_potential
        integer                                :: i,j,k, orbital_count
        integer                                :: domain(2), cube_ranges(2, 3), &
                                                  box_cell_index_limits(2, 3), ibox
        type(Integrator)                       :: integr
        type(Grid3D)                           :: grid
        real(REAL64)                           :: tmp, total, energy
      
        
#ifdef HAVE_CUDA_PROFILING
        call start_nvtx_timing("Orbital Potentials")
#endif
        !call pinfo("Computing RHF Orbital Potential")
        call bigben%split("RHF Orbital Potential Calculation")
        orbital_count=size(self%orbitals)

        allocate(self%orbital_potentials(size(self%orbitals)))
        ! calculate matrices, start with coulomb (and nuclear), 
        ! then go to exchange matrix
        ! get the -Z/r (could be as well taken from any orbital)
        nuclear_potential => self%coulomb_operator%get_nuclear_potential() 
        if (.not. self%coulomb_potential_is_valid) then
            call self%coulomb_operator%operate_on(self%electron_density, self%coulomb_potential)
            self%coulomb_potential_is_valid = .TRUE.
        end if
            
        coulomb_potential => self%coulomb_potential
        

        ! handle the occupied orbitals
        do i=1, self%nocc

            ! calculate coulomb operator application to orbitals(i)
            call coulomb_potential%multiply_sub(self%orbitals(i), self%temp)
            call nuclear_potential%multiply_sub(self%orbitals(i), self%temp2)
            call self%orbital_potentials(i)%init_copy(self%temp, copy_content = .TRUE.)
            call self%orbital_potentials(i)%add_in_place(self%temp2)
            
            ! calculate exchange operator application
            do j = 1, i
                call self%orbitals(i)%multiply_sub(self%orbitals(j), self%temp)
                call self%potential_input%copy_content(self%temp)
                call self%potential_input%communicate_cube(reversed_order = .TRUE.)
                call self%coulomb_operator%operate_on(self%potential_input, self%potential2)
                pot => self%potential2
                    
                call pot%multiply_sub(self%orbitals(j), self%temp)
                call self%orbital_potentials(i)%subtract_in_place(self%temp)
                    
                    
                if (i /= j) then
                    call pot%multiply_sub(self%orbitals(i), self%temp)
                    call self%orbital_potentials(j)%subtract_in_place(self%temp)
                end if

                    
                nullify(pot)
            end do
            
            
        end do
        
        ! We are handling the non-occupied orbitals as doubly excited orbitals in a
        ! way where the excitation occurs from the HOMO. Thus, first we 
        ! evaluate the new coulomb potential by reducing the effect of homo electrons from it.
        
        if (size(self%orbitals) > self%nocc) then
            call self%orbitals(self%nocc)%multiply_sub(self%orbitals(self%nocc), self%temp)
            call self%temp%product_in_place_REAL64(2.0d0)
            call self%potential_input%copy_content(self%temp)
            call self%potential_input%communicate_cube(reversed_order = .TRUE.)
            call self%coulomb_operator%operate_on(self%potential_input, self%potential2)
            call coulomb_potential%subtract_in_place(self%potential2)
            self%coulomb_potential_is_valid = .FALSE.
        end if
        
        ! go through all the virtual orbitals and evaluate the orbital potentials
        do i = self%nocc+1, orbital_count
            ! calculate coulomb operator application to orbitals(i)
            call coulomb_potential%multiply_sub(self%orbitals(i), self%temp)
            call nuclear_potential%multiply_sub(self%orbitals(i), self%temp2)
            call self%orbital_potentials(i)%init_copy(self%temp, copy_content = .TRUE.)
            call self%orbital_potentials(i)%add_in_place(self%temp2)
            
            ! add the coulomb contribution and subtract the exchange contribution
            ! in total: add contribution from one electron 
            call self%orbitals(i)%multiply_sub(self%orbitals(i), self%temp)
            call self%potential_input%copy_content(self%temp)
            call self%potential_input%communicate_cube(reversed_order = .TRUE.)
            call self%coulomb_operator%operate_on(self%potential_input, self%potential2)
            pot => self%potential2
            call pot%multiply_sub(self%orbitals(i), self%temp)
            call self%orbital_potentials(i)%add_in_place(self%temp)
            nullify(pot)
            
            ! calculate exchange operator application (again we ignore the HOMO)
            do j = 1, self%nocc-1
                call self%orbitals(i)%multiply_sub(self%orbitals(j), self%temp)
                call self%potential_input%copy_content(self%temp)
                call self%potential_input%communicate_cube(reversed_order = .TRUE.)
                call self%coulomb_operator%operate_on(self%potential_input, self%potential2)
                pot => self%potential2
                    
                call pot%multiply_sub(self%orbitals(j), self%temp)
                call self%orbital_potentials(i)%subtract_in_place(self%temp)
                    
                nullify(pot)
            end do
        end do
        nullify(coulomb_potential)
        nullify(nuclear_potential)
        call bigben%stop()
#ifdef HAVE_CUDA_PROFILING
        call stop_nvtx_timing()
#endif

        ! mark the orbital potentials valid
        self%orbital_potentials_are_valid = .TRUE.
    end subroutine

    

    subroutine RHFCycle_calculate_hamiltonian_matrix(self, evaluate_all)
        class(RHFCycle), intent(inout), target :: self
        !> if the complete hamiltonian matrix is evaluated, or only the ones
        !! needed to evaluate the energy
        logical,         intent(in)            :: evaluate_all
        integer                                :: i,j,k
        real(REAL64)                           :: num_el, temp_value
        real(REAL64), allocatable              :: orthogonalizing_matrix(:, :)
        logical                                :: do_update
write(*,*) 'begin RHFCycle_calculate_hamiltonian_matrix'
        
        call bigben%split("Creating Fock matrix")

        self%two_electron_integral_evaluated = .FALSE.

        ! calculate the kinetic energy matrix, nuclear-electron attraction matrix and the overlap matrix
        call self%calculate_one_electron_matrices(evaluate_all)
        
        
        ! calculate electron density
        call self%calculate_closed_shell_electron_density()


        ! calculate two electron integrals, do not evaluate all integrals, if we are not diagonalizing,
        ! and thus don't need the off-diagonal values
        call self%calculate_two_electron_integrals(evaluate_all)

        ! fill the needed elements of the coulomb and exchange matrix.
        ! in case of 'evaluate_all' the full coulomb and exchange matrices are 
        ! formed. Otherwise only the diagonal elements.
        if (evaluate_all) then
            call self%form_coulomb_matrix()
            call self%form_exchange_matrix()
        else
            call self%form_diagonal_coulomb_matrix()
            call self%form_diagonal_exchange_matrix()
        end if

        self%coulomb_matrix = 2.0d0 * self%coulomb_matrix
        self%exchange_matrix = -1.0d0 * self%exchange_matrix
        

        ! form the fock matrix
        self%hamiltonian_matrix=self%kinetic_matrix + self%nuclear_electron_matrix &
                                + self%coulomb_matrix + self%exchange_matrix

        ! calculate the total energy
        self%energy=2.d0*sum([ (self%kinetic_matrix(i,i) +                             &
                                self%nuclear_electron_matrix(i,i), i=1,self%nocc_b) ]) + &
                         sum([ (self%kinetic_matrix(i,i) +                             &
                                self%nuclear_electron_matrix(i,i), i=self%nocc_b+1,self%nocc_a) ]) + &
                         sum([ (self%coulomb_matrix(i,i) +                             &
                                self%exchange_matrix(i,i), i=1,self%nocc) ])
        
        
        self%eigen_values_a = [(self%hamiltonian_matrix(i, i), i = 1, size(self%hamiltonian_matrix, 1))]
        !call self%print_orbital_energies(.FALSE.)
        call self%print_orbital_energies(.TRUE.)

        ! stop fock matrix creation timing
        call bigben%stop()
write(*,*) 'end RHFCycle_calculate_hamiltonian_matrix'
    end subroutine

!--------------------------------------------------------------------!
!        ROHF Cycle
!--------------------------------------------------------------------!

    !> Initialize the Restricted Open-Shell Hartree Fock Cycle.
    subroutine ROHFCycle_init(orbitals, electron_density, &
                           nocc_a, nocc_b, virtual_orbital_count, &
                           laplacian_operator, coulomb_operator, helmholtz_operator, &
                           core_evaluator, result_object, &
                           a, b, f)
        !> Input orbitals
        type(Function3D), intent(in)                  :: orbitals(:)
        !> Input electron density holder
        type(Function3D), intent(in)                  :: electron_density
        !> Number of occupied orbitals with a spin
        integer, intent(in)                           :: nocc_a
        !> Number of occupied orbitals with b spin
        integer, intent(in)                           :: nocc_b
        !> Number of virtual orbitals
        integer, intent(in)                           :: virtual_orbital_count
        !> Laplacian operator
        class(Laplacian3D), intent(inout), target     :: laplacian_operator
        !> Coulomb operator
        class(Coulomb3D), intent(inout),   target     :: coulomb_operator
        !> Bound-state Helmholtz operator
        class(Helmholtz3D), intent(inout), target     :: helmholtz_operator
        !> ROHF Parameters a, b, and f
        real(REAL64),       intent(in)                :: a, b, f
        !> Evaluator to evaluate the cubes more accurately near core
        type(CoreEvaluator), intent(inout), target    :: core_evaluator
        ! the result object
        class(SCFCycle), allocatable, target          :: result_object
        type(ROHFCycle), pointer                      :: new
        integer                                       :: iorbital, orbital_count

        allocate(ROHFCycle :: result_object)

        ! set the non-polymorphic pointer to point to new
        select type(result_object)
            type is (ROHFCycle)
                new => result_object
        end select 
        
        call RestrictedHelmholtzSCFCycle_init(new, orbitals, electron_density, &
                           nocc_a, nocc_b, virtual_orbital_count, &
                           laplacian_operator, coulomb_operator, &
                           helmholtz_operator, core_evaluator) 

        new%a = a
        new%b = b
        new%f = f

    end subroutine

    

    subroutine ROHFCycle_destroy(self)
        class(ROHFCycle), intent(inout)         :: self

        call RestrictedHelmholtzSCFCycle_destroy(self)
    end subroutine

    

    subroutine ROHFCycle_calculate_orbital_potentials(self)
        class(ROHFCycle), intent(inout), target :: self
        class(Function3D), pointer              :: pot, coulomb_potential
        type(Function3D), pointer               :: nuclear_potential
        integer                                 :: i,j,k, orbital_count
        integer                                 :: domain(2), cube_ranges(2, 3), &
                                                   box_cell_index_limits(2, 3), ibox, nocc, nclosed
        type(Integrator)                        :: integr
        type(Grid3D)                            :: grid
        real(REAL64)                            :: tmp, total, energy, alpha, beta
      
write(*,*) 'begin ROHFCycle_calculate_orbital_potentials'
        
#ifdef HAVE_CUDA_PROFILING
        call start_nvtx_timing("Orbital Potentials")
#endif
        !call pinfo("Computing RHF Orbital Potential")
        call bigben%split("ROHF Orbital Potential Calculation")
        orbital_count=size(self%orbitals)
        nclosed = min(self%nocc_a, self%nocc_b)
        nocc    = max(self%nocc_a, self%nocc_b) 

        alpha = (1.0d0-self%a) / (1.0d0-self%f)
        beta  = (1.0d0-self%b) / (1.0d0-self%f)

        allocate(self%orbital_potentials(size(self%orbitals)))
        ! calculate matrices, start with coulomb (and nuclear), 
        ! then go to exchange matrix
        ! get the -Z/r (could be as well taken from any orbital)
        nuclear_potential => self%coulomb_operator%get_nuclear_potential() 


        ! handle the nuclear-electron repulsion for all orbitals
        do i = 1, size(self%orbitals)
            call nuclear_potential%multiply_sub(self%orbitals(i), self%temp)
            call self%orbital_potentials(i)%init_copy(self%temp, copy_content = .TRUE.)
        end do


        ! handle the closed orbitals coulomb and exchange potentials
        call self%calculate_closed_shell_electron_density()
        call self%coulomb_operator%operate_on(self%electron_density, self%coulomb_potential)
        coulomb_potential => self%coulomb_potential
        
        ! handle the closed orbitals coulomb potential
        do i=1, size(self%orbitals)
            ! calculate coulomb operator application to orbitals(i)
            call coulomb_potential%multiply_sub(self%orbitals(i), self%temp)
            call self%orbital_potentials(i)%add_in_place(self%temp)

        end do

        
        ! handle then open shell coulomb potentials
        if (nclosed /= nocc) then
            call self%calculate_open_shell_electron_density()
            call self%coulomb_operator%operate_on(self%electron_density, self%coulomb_potential)
            coulomb_potential => self%coulomb_potential

            ! handle the open shell coulomb potential effect on closed shell orbitals
            do i=1, nclosed
                call coulomb_potential%multiply_sub(self%orbitals(i), self%temp)
                call self%temp%product_in_place_REAL64(self%f * 2.0d0)
                call self%orbital_potentials(i)%add_in_place(self%temp)
            end do

            ! handle the open shell coulomb potential effect on open shell orbitals
            do i=nclosed+1, size(self%orbitals)
                call coulomb_potential%multiply_sub(self%orbitals(i), self%temp)
                call self%temp%product_in_place_REAL64(self%f * self%a * 2.0d0)
                call self%orbital_potentials(i)%add_in_place(self%temp)
            end do
        end if

        ! calculate extra two electron integral values needed in coulomb terms
        ! in open shell calculations. Calculate all terms (ij|kk), where i is a 
        ! closed shell orbital and j and k are open shell orbitals.
        do k = nclosed+1, nocc
            call self%orbitals(k)%multiply_sub(self%orbitals(k), self%temp)
            call self%potential_input%copy_content(self%temp)
            call self%potential_input%communicate_cube(reversed_order = .TRUE.)
            call self%coulomb_operator%operate_on(self%potential_input, self%potential2)
            pot => self%potential2

            do j = nclosed+1, nocc
                call pot%multiply_sub(self%orbitals(j), self%temp)
                do i = 1, nclosed
                     tmp = self%temp .dot. self%orbitals(i)
                     call self%set_two_electron_integral_value(k, i, j, j, tmp)
                end do
            end do
        end do

        
        ! calculate exchange operator application
        do i=1, size(self%orbitals)

            do j = 1, i
                call self%orbitals(i)%multiply_sub(self%orbitals(j), self%temp)
                call self%potential_input%copy_content(self%temp)
                call self%potential_input%communicate_cube(reversed_order = .TRUE.)
                call self%coulomb_operator%operate_on(self%potential_input, self%potential2)
                pot => self%potential2
                    
                call pot%multiply_sub(self%orbitals(j), self%temp)
    
                ! if i and j are open shell orbitals, calculate the extra two electron integrals
                ! (kj|ji).
                if (i > nclosed .and. i <= nocc .and. j > nclosed .and. j <= nocc) then
                    do k = 1, nclosed
                        tmp = self%temp .dot. self%orbitals(k)
                        call self%set_two_electron_integral_value(k, j, j, i, tmp)
                    end do
                end if

                if (j > nclosed .and. j <= nocc .and. i <= nclosed) then
                    call self%temp%product_in_place_REAL64(self%f)
                end if

                if (j > nclosed .and. i > nclosed) &
                    call self%temp%product_in_place_REAL64(self%b * self%f)
                call self%orbital_potentials(i)%subtract_in_place(self%temp)
                    
                    
                if (i /= j) then
                    call pot%multiply_sub(self%orbitals(i), self%temp)

                    ! if i and j are open shell orbitals, calculate the extra two electron integrals
                    ! (ki|ij).
                    if (i > nclosed .and. i <= nocc .and. j > nclosed .and. j <= nocc) then
                        do k = 1, nclosed
                            tmp = self%temp .dot. self%orbitals(k)
                            call self%set_two_electron_integral_value(k, j, j, i, tmp)
                        end do
                    end if

                    if (i > nclosed .and. j <= nclosed) &
                        call self%temp%product_in_place_REAL64(self%f)
                    if (i > nclosed .and. j > nclosed) &
                        call self%temp%product_in_place_REAL64(self%b * self%f)
                    call self%orbital_potentials(j)%subtract_in_place(self%temp)
                end if

                    
                nullify(pot)
            end do
        end do

        ! calculate the coupling terms L and M for the closed shell 
        ! orbitals and add to corresponding potentials.
        do i = 1, nclosed
            do j = nclosed+1, nocc
                tmp = 0.0d0
                do k = nclosed+1, nocc
                    tmp = tmp + self%two_electron_integrals(j, i, k, k)
                end do
                call self%orbital_potentials(i)%add_in_place(self%orbitals(j), 2.0d0*alpha*self%f*tmp)
                
                tmp = 0.0d0
                do k = nclosed+1, nocc
                    tmp = tmp + self%two_electron_integrals(j, k, k, i)
                end do
                call self%orbital_potentials(i)%subtract_in_place(self%orbitals(j), beta*self%f*tmp)
            end do
        end do

        ! calculate the coupling terms L and M for the open shell 
        ! orbitals and add them to corresponding potentials.
        do i = nclosed+1, nocc
            do j = 1, nclosed
                tmp = 0.0d0
                do k = nclosed+1, nocc
                    tmp = tmp + self%two_electron_integrals(j, i, k, k)
                end do
                call self%orbital_potentials(i)%add_in_place(self%orbitals(j), 2.0d0*alpha*tmp)
                
                tmp = 0.0d0
                do k = nclosed+1, nocc
                    tmp = tmp + self%two_electron_integrals(j, k, k, i)
                end do
                call self%orbital_potentials(i)%subtract_in_place(self%orbitals(j), beta*tmp)
            end do
        end do

        nullify(coulomb_potential)
        nullify(nuclear_potential)
        call bigben%stop()
#ifdef HAVE_CUDA_PROFILING
        call stop_nvtx_timing()
#endif

        ! mark the orbital potentials valid
        self%orbital_potentials_are_valid = .TRUE.
    end subroutine

    

    subroutine ROHFCycle_calculate_hamiltonian_matrix(self, evaluate_all)
        class(ROHFCycle), intent(inout), target :: self
        !> if the complete hamiltonian matrix is evaluated, or only the ones
        !! needed to evaluate the energy
        logical,         intent(in)             :: evaluate_all
        integer                                 :: i,j,k, nclosed, nocc
        real(REAL64)                            :: num_el, temp_value
        real(REAL64), allocatable               :: orthogonalizing_matrix(:, :)
        logical                                 :: do_update

write(*,*) 'You are trying to run a restricted open shell HF calculation, but',&
'this is not implemented.  If you do want to implement it, look for example at',&
'the electron density which is wrong atm.'
flush(6)
call abort()

        call bigben%split("Creating Fock matrix")
        
        nclosed = min(self%nocc_a, self%nocc_b)
        nocc    = max(self%nocc_a, self%nocc_b)
        self%two_electron_integral_evaluated = .FALSE.

        ! calculate the kinetic energy matrix, nuclear-electron attraction matrix and the overlap matrix
        call self%calculate_one_electron_matrices(evaluate_all)
        
        ! calculate electron density
        call self%calculate_closed_shell_electron_density()

        ! calculate two electron integrals, do not evaluate all integrals, if we are not diagonalizing,
        ! and thus don't need the off-diagonal values
        call self%calculate_two_electron_integrals(evaluate_all)

        ! fill the needed elements of the coulomb and exchange matrix.
        ! in case of 'evaluate_all' the full coulomb and exchange matrices are 
        ! formed. Otherwise only the diagonal elements.
        if (evaluate_all) then
            print *, "ERROR: diagonalizing is not possible currently for the ROHF."
            stop
        else

            call self%calculate_orbital_potentials()

            do i = 1, size(self%orbitals)
                self%eigen_values_a(i) =   self%kinetic_matrix(i, i)  &
                                         + (self%orbitals(i) .dot. self%orbital_potentials(i)) 
            end do

            self%energy = 0.0d0
            do i = 1, nclosed

                self%energy = self%energy +                                             &
                                  2.0d0 * self%kinetic_matrix(i,i)                      &
                                + 2.0d0 * self%nuclear_electron_matrix(i,i)            
                do j = 1, nclosed
                    self%energy = self%energy + 2.0d0 * self%two_electron_integrals(i, i, j, j) &
                                              -         self%two_electron_integrals(i, j, j, i)
                end do
        
                do j = nclosed+1, nocc
                    self%energy = self%energy +     self%f *  &
                                                 (    4.0d0 * self%two_electron_integrals(i, i, j, j) &
                                                   -  2.0d0 * self%two_electron_integrals(i, j, j, i))
                end do
            end do

            do i = nclosed+1, nocc
                
                self%energy = self%energy + self%f * (                                  &
                                  2.0d0 * self%kinetic_matrix(i,i)                      &
                                + 2.0d0 * self%nuclear_electron_matrix(i,i))
        
                
                do j = nclosed+1, nocc
                    self%energy = self%energy +     self%f * self%f *  &
                                                 (    2.0d0 * self%a * self%two_electron_integrals(i, i, j, j) &
                                                   -          self%b * self%two_electron_integrals(i, j, j, i))
                end do
            end do
        end if

        call self%print_orbital_energies(.FALSE.)

        ! stop fock matrix creation timing
        call bigben%stop()
    end subroutine
    
!--------------------------------------------------------------------!
!        RHF Cycle with external homogeneous electric field          !
!--------------------------------------------------------------------!
 
    subroutine RHFCycle_with_EEF_init(orbitals, electron_density, nocc_a, nocc_b, virtual_orbital_count, &
                           laplacian_operator, coulomb_operator, helmholtz_operator, &
                           core_evaluator, field, result_object)
        !> Input orbitals
        type(Function3D), intent(in)                  :: orbitals(:)
        !> Input electron density holder
        type(Function3D), intent(in)                  :: electron_density
        !> Number of occupied orbitals with a spin
        integer, intent(in)                           :: nocc_a
        !> Number of occupied orbitals with b spin
        integer, intent(in)                           :: nocc_b
        !> Number of virtual orbitals
        integer, intent(in)                           :: virtual_orbital_count
        !> Laplacian operator
        class(Laplacian3D), intent(inout), target     :: laplacian_operator
        class(Coulomb3D), intent(inout),   target     :: coulomb_operator
        class(Helmholtz3D), intent(inout), target     :: helmholtz_operator
        !> Evaluator to evaluate the cubes more accurately near core
        type(CoreEvaluator), intent(inout), target    :: core_evaluator
        !> external electric field
        real(REAL64),       intent(in)                :: field(3)
        ! the result object
        class(SCFCycle), allocatable, target          :: result_object
        type(RHFCycle_with_EEF), pointer              :: new
        integer                                       :: iorbital, orbital_count
        ! these are related to temporary energy fix
        integer                                       :: i,j, natom
        real(REAL64), allocatable                     :: centers(:,:), charges(:)

        allocate(RHFCycle_with_EEF :: result_object)

        ! set the non-polymorphic pointer to point to new
        select type(result_object)
            type is (RHFCycle_with_EEF)
                new => result_object
        end select 
        
       call RestrictedHelmholtzSCFCycle_init(new, orbitals, electron_density, &
                           nocc_a, nocc_b, virtual_orbital_count, &
                           laplacian_operator, coulomb_operator, &
                           helmholtz_operator, core_evaluator) 
        
        ! external field stuff 
        new%electric_field = field
        new%external_electric_potential  = constant_field_potential(new, field)
        !new%external_electric_potential  = &
        !    constant_field_potential(new, field, origin_on_cube=[0d0, 0d0,0d0] )
        

        new%nuclei_in_eef = sum(new%external_electric_potential%evaluate( &
                                    new%external_electric_potential%bubbles%get_centers() ) * &
                                    new%external_electric_potential%bubbles%get_z()   )
        print *, 'nuclear potential energy', new%nuclei_in_eef


    end subroutine

    !! calculating orbital potentials
    subroutine RHFCycle_with_EEF_calculate_orbital_potentials(self)
        class(RHFCycle_with_EEF), intent(inout), target:: self
        integer                                        :: i
        type(Function3D), allocatable                  :: temp

        ! calculate orbital potentials without external potential
        call RHFCycle_calculate_orbital_potentials(self)

        ! add external field contribution
        do i=1, size(self%orbitals)
            call self%external_electric_potential%multiply_sub(self%orbitals(i), temp)
            call self%orbital_potentials(i)%add_in_place(temp)
            call temp%destroy()
            deallocate(temp)
        end do


    end subroutine

    subroutine RHFCycle_with_EEF_calculate_hamiltonian_matrix(self, evaluate_all)
        class(RHFCycle_with_EEF), intent(inout), target :: self
        logical,                  intent(in)            :: evaluate_all
        integer                                         :: i,j,k
        real(REAL64)                                    :: num_el, temp_value
        real(REAL64), allocatable                       :: orthogonalizing_matrix(:, :)
        logical                                         :: do_update
        ! electric dipole moment
        real(REAL64)                                    :: moment(3)
        type(Function3D), allocatable                   :: temp

        call start_memory_follower()
        call bigben%split("RHF with external electric field SCF cycle")
        print *, 'external electric field', self%electric_field

        self%two_electron_integral_evaluated = .FALSE.
        
        call bigben%split("Creating Fock matrix")
        
        ! calculate the kinetic energy matrix and the overlap matrix
        call self%calculate_one_electron_matrices(evaluate_all)
        print *, "Kinetic matrix - diagonal", [(self%kinetic_matrix(i,i), i=1,self%nocc) ]
        print *, "Overlap matrix - diagonal", [(self%overlap_matrix(i, i), i=1, self%nocc)]

        ! recalculate orbital potentials, if the potentials are valid
        if ( .not. self%orbital_potentials_are_valid) then
            call self%calculate_closed_shell_electron_density()
            call self%calculate_orbital_potentials()
        end if

        ! fill Fock matrix
        do i=1,size(self%orbitals)
            !call self%orbitals(i)%multiply_sub(self%orbital_potentials(i), temp)
            if (evaluate_all) then
                do j=1,i
                    self%hamiltonian_matrix(i,j) = self%orbital_potentials(i) .dot. self%orbitals(j)
                    self%hamiltonian_matrix(i,j) = self%hamiltonian_matrix(i,j) + self%kinetic_matrix(i,j)
                    self%hamiltonian_matrix(j,i) = self%hamiltonian_matrix(i,j)
                end do
            else
                self%hamiltonian_matrix(i, i) = self%orbital_potentials(i) .dot. self%orbitals(i)
                self%hamiltonian_matrix(i, i) = self%kinetic_matrix(i, i)
            end if
            !call temp%destroy()
            !deallocate(temp)
        end do
        

        ! estimation of total electronic energy
        print *, 'old electronic energy', self%energy
        
        ! calculate total energy
        self%energy=0d0
        do i=1, self%nocc
            self%energy = self%energy + self%hamiltonian_matrix(i,i)
print *, 'i orbital energy', self%hamiltonian_matrix(i,i)
            self%energy = self%energy + self%kinetic_matrix(i,i)
print *, 'i kin', self%kinetic_matrix(i,i)
            self%energy = self%energy + self%nuclear_electron_matrix(i,i)
print *, 'i vne', self%nuclear_electron_matrix(i,i)
        end do        
        print *, 'new electronic energy', self%energy
        self%energy = self%energy + self%nuclei_in_eef
        print *, 'energy including nuclei in external electric field', self%energy

        ! calculate dipole moment
        call check_memory_cuda()
        call bigben%split("calculating electric dipole moment")
            moment =  calculate_dipole_moment(self, self%electron_density)  
            print *, 'electric dipole moment from bubbles', moment
        print *, 'dipole moment magnitude', sqrt(sum(moment**2)), 'a.u. , ', sqrt(sum(moment**2))* 2.541746, 'debye'
        call check_memory_cuda()
            moment =  calculate_dipole_moment(self, self%electron_density, origin_on_cube=[0d0,0d0,0d0])  
            print *, 'electric dipole moment from cube', moment 
        print *, 'dipole moment magnitude', sqrt(sum(moment**2)), 'a.u. , ', sqrt(sum(moment**2))* 2.541746, 'debye'
        call check_memory_cuda()
        call bigben%stop_and_print()
        call check_memory_cuda()


        ! stop scf cycle timing
        call bigben%stop()
        call stop_memory_follower()
    end subroutine

    subroutine RHFCycle_with_EEF_destroy(self)
        class(RHFCycle_with_EEF), intent(inout)   :: self

        if (allocated(self%external_electric_potential)) then
            call self%external_electric_potential%destroy()
            deallocate(self%external_electric_potential)
        end if
        
        call RHFCycle_destroy(self)
    end subroutine


!--------------------------------------------------------------------!
!        RDFT Cycle
!--------------------------------------------------------------------!
#ifdef HAVE_DFT
    subroutine RDFTCycle_init(orbitals, electron_density, &
                              nocc_a, nocc_b, virtual_orbital_count, &
                              exchange_functional, correlation_functional, &
                              xc_update_method, xc_lmax, laplacian_operator, &
                              coulomb_operator, helmholtz_operator, core_evaluator, &
                              result_object, orbitals_density_evaluation, &
                              finite_diff_order) 
        type(Function3D), intent(in)                  :: orbitals(:)
        !> Input electron density holder
        type(Function3D), intent(in)                  :: electron_density
        !> Number of occupied orbitals with a spin
        integer, intent(in)                           :: nocc_a
        !> Number of occupied orbitals with b spin
        integer, intent(in)                           :: nocc_b
        !> Number of virtual orbitals
        integer, intent(in)                           :: virtual_orbital_count
        !> The type of the Exchange functional
        integer(INT32), intent(in)                    :: exchange_functional
        !> The type of the Correlation functional
        integer(INT32), intent(in)                    :: correlation_functional

        integer(int32), intent(in)                    :: xc_update_method
        integer(int32), intent(in)                    :: xc_lmax
        !> Laplacian operator
        class(Laplacian3D), intent(inout), target     :: laplacian_operator
        class(Coulomb3D), intent(inout),   target     :: coulomb_operator
        class(Helmholtz3D), intent(inout), target     :: helmholtz_operator
        type(CoreEvaluator), intent(inout), target    :: core_evaluator
        ! the result object
        class(SCFCycle), allocatable, target, intent(inout) :: result_object
        !> If electron density and its gradient are evaluated using the more complex
        !! but accurate method in the exchange and correalation. 
        logical,        intent(in)                    :: orbitals_density_evaluation
        integer(INT32), intent(in)                    :: finite_diff_order

        ! the result object
        integer                                       :: k, iorbital
        
        class(RDFTCycle), pointer                      :: new
        class(RestrictedHelmholtzSCFCycle), pointer    :: scf_cycle
        
        allocate(RDFTCycle :: result_object)
        
        ! set the non-polymorphic pointer to point to new
        select type(result_object)
            type is (RDFTCycle)
                new => result_object
                scf_cycle => result_object
        end select 
        
        call RestrictedHelmholtzSCFCycle_init(scf_cycle, orbitals, electron_density, &
                           nocc_a, nocc_b, virtual_orbital_count, &
                           laplacian_operator, coulomb_operator, helmholtz_operator, core_evaluator) 

        allocate(new%exchange_correlation_matrix_a (size(new%orbitals_a),size(new%orbitals_a)))

        ! initialize the Exchange & Correlation calculator
        new%exchange_correlation = XC(new%electron_density_a, exchange_functional, correlation_functional, &
                                      xc_lmax, laplacian_operator, core_evaluator, orbitals_density_evaluation, &
                                      finite_diff_order)

        ! set the pointers to correct arrays
        new%exchange_correlation_matrix => new%exchange_correlation_matrix_a
        new%exchange_matrix => new%exchange_correlation_matrix_a

        ! store the update method
        new%xc_update_method = xc_update_method
    end subroutine


    subroutine RDFTCycle_destroy(self)
        class(RDFTCycle), intent(inout)         :: self
        integer                                 :: i
        
        if(allocated(self%exchange_correlation_matrix_a)) deallocate(self%exchange_correlation_matrix_a)
        call self%xc_potential_a%destroy()
        call self%exchange_correlation%destroy()
        call RestrictedHelmholtzSCFCycle_destroy(self)
        
    end subroutine


    !> Calculates the application of the RDFT potential to each of the orbitals
    ! v_{eff} * \psi_i
    subroutine RDFTCycle_calculate_orbital_potentials(self)
        class(RDFTCycle), intent(inout), target:: self
        class(Function3D), pointer             :: coulomb_potential
        type(Function3D), pointer              :: nuclear_potential
        type(Function3D)                       :: xc_energy_density, temp
        type(Bubbles)                          :: temp_bubbles
        type(Function3D), allocatable          :: temp2
        integer                                :: i,j,k, orbital_count
        real(REAL64)                           :: test
write(*,*) 'begin RDFTCycle_calculate_orbital_potentials'
      
        call memoryfollower_print_status()        
        !call pinfo("Computing RDFT Orbital Potential")
        call bigben%split("RDFT Orbital Potential Calculation")

        allocate(self%orbital_potentials(size(self%orbitals)))
        !call self%electron_density%bubbles%extrapolate_origo(2, lmin = 1)
        
        if (.not. self%coulomb_potential_is_valid) then
            call self%xc_potential_a%destroy()
            
            ! evaluate the exchange correlation
            call self%exchange_correlation%eval( &
                                                self%electron_density,  &
                                                self%xc_potential_a, xc_energy_density, self%xc_energy, &
                                                self%orbitals(:self%nocc))
            self%xc_potential => self%xc_potential_a
            
            call self%coulomb_operator%operate_on(self%electron_density, self%coulomb_potential)
            call self%coulomb_potential%precalculate_taylor_series_bubbles()
        end if

        orbital_count=size(self%orbitals)
        
        nuclear_potential => self%coulomb_operator%get_nuclear_potential()
        coulomb_potential => self%coulomb_potential
        
        call temp%init_copy(self%xc_potential, copy_content = .TRUE.)
        call temp%add_in_place(coulomb_potential)
        call temp%add_in_place(nuclear_potential)
        
        do i=1, orbital_count
            call temp%multiply_sub(self%orbitals(i), temp2)
            call self%orbital_potentials(i)%init_copy(temp2, copy_content = .TRUE.)
            call temp2%destroy()
            deallocate(temp2)
        end do
       
        ! mark the orbital potentials valid
        self%orbital_potentials_are_valid = .TRUE.
       
        call temp%destroy()
        call xc_energy_density%destroy()
        nullify(coulomb_potential)
        nullify(nuclear_potential)
        call bigben%stop() 
        call memoryfollower_print_status()
    end subroutine


    subroutine RDFTCycle_calculate_hamiltonian_matrix(self, evaluate_all)
        class(RDFTCycle), intent(inout), target:: self
        logical,          intent(in)           :: evaluate_all
        type(Function3D)                       :: xc_energy_density
        type(Function3D), allocatable          :: temp, temp2
        integer                                :: i,j,k
        real(REAL64)                           :: xc_energy, temp_e
!        real(REAL64), allocatable              :: previous_kinetic_matrix(:, :), previous_coulomb_matrix(:, :), &
!                                                  previous_nuclear_electron_matrix(:, :), previous_xc_matrix(:, :), &
!                                                  previous_hamiltonian_matrix(:, :)
        logical                                :: do_update

write(*,*) 'begin RDFTCycle_calculate_hamiltonian_matrix'
        call memoryfollower_print_status()
        call bigben%split("RDFT SCF cycle")
        ! previous_kinetic_matrix = self%kinetic_matrix
        ! previous_coulomb_matrix = self%coulomb_matrix
        ! previous_nuclear_electron_matrix = self%nuclear_electron_matrix
        ! previous_xc_matrix = self%exchange_correlation_matrix_a
        ! previous_hamiltonian_matrix = self%hamiltonian_matrix

        call self%calculate_one_electron_matrices(evaluate_all)
write(*,*) 'kin: ', self%kinetic_matrix
write(*,*) 'pot: ', self%nuclear_electron_matrix
        if ( .not. self%coulomb_potential_is_valid) then
            ! calculate electron density
            call self%calculate_closed_shell_electron_density()
            call self%exchange_correlation%eval( &
                                                self%electron_density, &
                                                self%xc_potential_a, xc_energy_density, &
                                                self%xc_energy, self%orbitals(:self%nocc))
            self%xc_potential => self%xc_potential_a
        end if
        !call self%calculate_electron_density(non_overlapping = .FALSE.)
        ! calculate the coulomb and nuclear matrices
        call self%calculate_coulomb_matrix(evaluate_all)
write(*,*) 'J: ', self%coulomb_matrix
        call self%calculate_xc_matrix(evaluate_all)
write(*,*) 'xc: ', self%exchange_correlation_matrix_a
        !if (self%xc_update_method == 1) call self%exchange_correlation%eval_g(self%electron_density)
        !if (self%xc_update_method == 2) call self%exchange_correlation%eval_taylor(self%electron_density)


        call xc_energy_density%destroy()
        call memoryfollower_print_status()

        ! form the hamiltonian matrix
        self%hamiltonian_matrix=self%kinetic_matrix + self%nuclear_electron_matrix &
                                + self%coulomb_matrix + self%exchange_correlation_matrix

        self%eigen_values_a = [(self%hamiltonian_matrix(i, i), i = 1, size(self%orbitals))]
        ! calculate the total energy
        self%energy=2.d0*sum([ (self%kinetic_matrix(i,i) +  &
                                 self%nuclear_electron_matrix(i,i), i=1,self%nocc) ]) +&
                          sum([ (self%coulomb_matrix(i,i), i=1,self%nocc) ]) +& 
                          self%xc_energy

        call self%print_orbital_energies(.TRUE.)
        
        
        call memoryfollower_print_status()
        ! deallocate(previous_kinetic_matrix, previous_coulomb_matrix, &
        !            previous_nuclear_electron_matrix, previous_xc_matrix, &
        !                                           previous_hamiltonian_matrix)

        ! stop scf cycle timing
        call bigben%stop()
write(*,*) 'end RDFTCycle_calculate_hamiltonian_matrix'

     end subroutine RDFTCycle_calculate_hamiltonian_matrix
    
#endif

!--------------------------------------------------------------------!
!        URHF Cycle
!--------------------------------------------------------------------!

    subroutine URHFCycle_init(orbitals_a, orbitals_b, electron_density_a, &
                              electron_density_b, nocc_a, nocc_b, virtual_orbital_count_a, &
                              virtual_orbital_count_b, laplacian_operator, &
                              coulomb_operator, helmholtz_operator, result_object)
        !> Input orbitals with spin a
        type(Function3D), intent(in)                  :: orbitals_a(:)
        !> Input orbitals with spin b
        type(Function3D), intent(in)                  :: orbitals_b(:)
        !> Input electron density of orbitals with spin a
        type(Function3D), intent(in)                  :: electron_density_a
        !> Input electron density of orbitals with spin a
        type(Function3D), intent(in)                  :: electron_density_b
        !> Number of occupied orbitals with spin a
        integer, intent(in)                           :: nocc_a
        !> Number of occupied orbitals with spin b
        integer, intent(in)                           :: nocc_b
        !> Number of virtual orbitals with spin a
        integer, intent(in)                           :: virtual_orbital_count_a
        !> Number of virtual orbitals with spin b
        integer, intent(in)                           :: virtual_orbital_count_b
        !> Laplacian operator
        class(Laplacian3D), intent(inout), target     :: laplacian_operator
        !> Coulomb operator
        class(Coulomb3D), intent(inout),   target     :: coulomb_operator
        !> Helmholtz operator
        class(Helmholtz3D), intent(inout), target     :: helmholtz_operator

        ! the result object
        class(SCFCycle), allocatable, target          :: result_object
        type(URHFCycle), pointer                      :: new
        integer                                       :: iorbital, orbital_count_a, orbital_count_b

        allocate(URHFCycle :: result_object)

        ! set the non-polymorphic pointer to point to new
        select type(result_object)
            type is (URHFCycle)
                new => result_object
        end select 
        orbital_count_a = min(nocc_a + virtual_orbital_count_a, size(orbitals_a))
        orbital_count_b = min(nocc_b + virtual_orbital_count_b, size(orbitals_b))


        allocate(new%orbitals_a(orbital_count_a))
        allocate(new%orbitals_b(orbital_count_b))
        allocate(new%all_orbitals(orbital_count_a + orbital_count_b))
        new%total_orbital_count = orbital_count_a + orbital_count_b
        !print *, "orbitals size", size(orbitals), "dot", orbitals(1) .dot. orbitals(1)
        do iorbital = 1, orbital_count_a
            call new%orbitals_a(iorbital)%init_copy(orbitals_a(iorbital), copy_content = .TRUE.)
            new%all_orbitals(iorbital)%p => new%orbitals_a(iorbital)
        end do
        do iorbital = 1, orbital_count_b
            call new%orbitals_b(iorbital)%init_copy(orbitals_b(iorbital), copy_content = .TRUE.)
            new%all_orbitals(iorbital+orbital_count_a)%p => new%orbitals_b(iorbital)
        end do
        allocate(Function3D :: new%electron_density_a)
        allocate(Function3D :: new%electron_density_b)
        call new%electron_density_a%init_copy(electron_density_a, copy_content = .TRUE.)
        call new%electron_density_b%init_copy(electron_density_b, copy_content = .TRUE.)
        allocate(Function3D :: new%potential_input)
        call new%potential_input%init_copy(electron_density_a)
        allocate(Function3D :: new%temp3)
        call new%temp3%init_copy(orbitals_a(1), copy_content = .TRUE.)
        new%temp3 = 0.0d0
        
        new%laplacian_operator => laplacian_operator
        new%coulomb_operator => coulomb_operator
        new%helmholtz_operator => helmholtz_operator
        
        new%nocc_a = nocc_a
        new%nvir_a = virtual_orbital_count_a
        allocate(new%kinetic_matrix_a  (orbital_count_a,orbital_count_a))
        allocate(new%nuclear_electron_matrix_a (orbital_count_a,orbital_count_a))
        allocate(new%coulomb_matrix_a    (orbital_count_a,orbital_count_a))
        allocate(new%overlap_matrix_a    (orbital_count_a,orbital_count_a))
        allocate(new%exchange_matrix_a   (orbital_count_a,orbital_count_a))
        allocate(new%hamiltonian_matrix_a(orbital_count_a,orbital_count_a))   
        allocate(new%coefficients_a      (orbital_count_a,orbital_count_a))  
        allocate(new%eigen_values_a      (orbital_count_a), source = 0.0d0)  
        allocate(new%two_electron_integrals_a(orbital_count_a,orbital_count_a, &
                                            orbital_count_a,orbital_count_a))
        allocate(new%two_electron_integral_evaluated_a(orbital_count_a,orbital_count_a, &
                                            orbital_count_a,orbital_count_a), source = .FALSE.)

        allocate(new%coulomb_matrix_ab    (orbital_count_a,orbital_count_a))

        new%nocc_b = nocc_b
        new%nvir_b = virtual_orbital_count_b
        allocate(new%kinetic_matrix_b  (orbital_count_b,orbital_count_b))
        allocate(new%nuclear_electron_matrix_b (orbital_count_b,orbital_count_b))
        allocate(new%coulomb_matrix_b    (orbital_count_b,orbital_count_b))
        allocate(new%overlap_matrix_b    (orbital_count_b,orbital_count_b))
        allocate(new%exchange_matrix_b   (orbital_count_b,orbital_count_b))
        allocate(new%hamiltonian_matrix_b(orbital_count_b,orbital_count_b))   
        allocate(new%coefficients_b      (orbital_count_b,orbital_count_b))  
        allocate(new%eigen_values_b      (orbital_count_b), source = 0.0d0)  
        allocate(new%two_electron_integrals_b(orbital_count_b,orbital_count_b, &
                                            orbital_count_b,orbital_count_b))
        allocate(new%two_electron_integral_evaluated_b(orbital_count_b,orbital_count_b, &
                                            orbital_count_b,orbital_count_b), source = .FALSE.)

        
        allocate(new%coulomb_matrix_ba    (orbital_count_b,orbital_count_b))
        call new%gs_orthogonalize_orbitals()
        new%coulomb_potential_is_valid = .FALSE.

    end subroutine

    
    subroutine URHFCycle_destroy(self)
        class(URHFCycle), intent(inout)         :: self
        integer                                :: i

        call UnRestrictedHelmholtzSCFCycle_destroy(self)
    end subroutine

    
    !> NOTE: the result potentials are not communicated in the end!
    !!       Thus, the user has to take care of that when using the
    !!       potentials
    subroutine URHFCycle_calculate_orbital_potentials(self)
        class(URHFCycle), intent(inout), target :: self
        class(Function3D), pointer              :: pot, coulomb_potential
        type(Function3D), pointer               :: nuclear_potential
        integer                                 :: i,j,k, orbital_count
        integer                                 :: domain(2), cube_ranges(2, 3), &
                                                   box_cell_index_limits(2, 3), ibox
        type(Integrator)                        :: integr
        type(Grid3D)                            :: grid
        real(REAL64)                            :: tmp, total, energy
      
        
#ifdef HAVE_CUDA_PROFILING
        call start_nvtx_timing("Orbital Potentials")
#endif
        !call pinfo("Computing URHF Orbital Potential")
        call bigben%split("URHF Orbital Potential Calculation")

        allocate(self%orbital_potentials(size(self%orbitals)))
        ! calculate matrices, start with coulomb (and nuclear), 
        ! then go to exchange matrix
        ! get the -Z/r (could be as well taken from any orbital)
        nuclear_potential => self%coulomb_operator%get_nuclear_potential() 
        
        if (.not. self%coulomb_potential_is_valid) then
            call self%coulomb_operator%operate_on(self%electron_density, self%coulomb_potential)
        end if
        coulomb_potential => self%coulomb_potential
        orbital_count = size(self%orbitals)
#ifdef HAVE_CUDA       
        call CUDASync_all()
#endif

        do i=1, orbital_count
            ! calculate coulomb operator application to orbitals(i)
            ! note in potential multiplication potential must be first. Otherwise, weird shit happens
            call coulomb_potential%multiply_sub(self%orbitals(i), self%temp)
            call nuclear_potential%multiply_sub(self%orbitals(i), self%temp2)
#ifdef HAVE_CUDA       
            call CUDASync_all()
#endif
            call self%orbital_potentials(i)%init_copy(self%temp, copy_content = .TRUE.)
            call self%orbital_potentials(i)%add_in_place(self%temp2)

            !print *, "Energy from pot after coulomb", self%orbitals(i).dot.self%orbital_potentials(i)

            ! calculate exchange operator application
            do j = 1, i
                if (i <= self%nocc .or. j <= self%nocc) then
                    call self%orbitals(i)%multiply_sub(self%orbitals(j), self%temp)
                    call self%potential_input%copy_content(self%temp)
                    call self%potential_input%communicate_cube(reversed_order = .TRUE.)
                    call self%coulomb_operator%operate_on(self%potential_input, self%potential2)
                    pot => self%potential2
                
                    if (j <= self%nocc) then
                        call pot%multiply_sub(self%orbitals(j), self%temp)
                        call self%orbital_potentials(i)%subtract_in_place(self%temp)
                    end if

                    if (i /= j .and. i <= self%nocc) then
                        call pot%multiply_sub(self%orbitals(i), self%temp)
                        call self%orbital_potentials(j)%subtract_in_place(self%temp)                           
                    end if
                    nullify(pot)
                end if
            end do

        end do

        do i=1, orbital_count            
        !    self%eigen_values(i) = self%orbitals(i) .dot. self%orbital_potentials(i)
        !    self%eigen_values(i) = self%eigen_values(i) + self%kinetic_matrix(i, i)
        !    print *, "energy after potential", i,  self%eigen_values(i)
        !     call self%orbital_potentials(i)%destroy()
        end do
        !deallocate(self%orbital_potentials)
        !> NOTE: we do not perform communication here!!!
        nullify(coulomb_potential)
        nullify(nuclear_potential)
        call bigben%stop()
#ifdef HAVE_CUDA_PROFILING
        call stop_nvtx_timing()
#endif
        
        ! mark the orbital potentials valid
        self%orbital_potentials_are_valid = .TRUE.
    end subroutine


    subroutine URHFCycle_calculate_hamiltonian_matrix(self, evaluate_all)
        class(URHFCycle), intent(inout), target :: self
        !> if the complete hamiltonian matrix is evaluated, or only the ones
        !! needed to evaluate the energy
        logical,         intent(in)             :: evaluate_all
        integer                                 :: i,j,k
        real(REAL64)                            :: num_el, temp_value, kinetic_energy
        real(REAL64), allocatable               :: orthogonalizing_matrix(:, :)
        logical                                 :: do_update

        call bigben%split("URHF SCF cycle")

        self%two_electron_integral_evaluated_a = .FALSE.
        self%two_electron_integral_evaluated_b = .FALSE.
        self%two_electron_integrals_a = 0.0d0
        self%two_electron_integrals_b = 0.0d0

        call self%calculate_spin_densities()
        call self%electron_density_a%add_in_place(self%electron_density_b)
        self%electron_density => self%electron_density_a

        call bigben%split("Creating Fock matrix a")
        ! calculate the kinetic energy matrix and the overlap matrix for the orbitals with spin a
        self%nocc => self%nocc_a
        self%orbitals => self%orbitals_a
        self%kinetic_matrix => self%kinetic_matrix_a
        self%overlap_matrix => self%overlap_matrix_a
        self%nuclear_electron_matrix => self%nuclear_electron_matrix_a
        call self%calculate_one_electron_matrices(evaluate_all)

        ! calculate the coulomb and exchange matrices for the orbitals with spin a
        self%coulomb_matrix => self%coulomb_matrix_a
        self%exchange_matrix => self%exchange_matrix_a
        self%two_electron_integral_evaluated => self%two_electron_integral_evaluated_a
        self%two_electron_integrals => self%two_electron_integrals_a
        call self%calculate_two_electron_integrals(evaluate_all)
        call self%calculate_coulomb_matrix(evaluate_all)
        call self%form_exchange_matrix()

        ! form the fock matrix for spin a
        self%hamiltonian_matrix_a = self%kinetic_matrix_a + self%nuclear_electron_matrix_a &
                                   + self%coulomb_matrix_a - self%exchange_matrix_a 
        
        self%eigen_values_a = [(self%hamiltonian_matrix_a(i, i), i = 1, size(self%orbitals_a))]
        self%eigen_values => self%eigen_values_a
        call bigben%stop()

        call self%print_orbital_energies(.TRUE.)


        call bigben%split("Creating Fock matrix b")
        ! calculate the kinetic energy matrix and the overlap matrix for the orbitals with spin b
        self%nocc => self%nocc_b
        self%orbitals => self%orbitals_b
        self%kinetic_matrix => self%kinetic_matrix_b
        self%overlap_matrix => self%overlap_matrix_b
        self%nuclear_electron_matrix => self%nuclear_electron_matrix_b
        call self%calculate_one_electron_matrices(evaluate_all)

        ! calculate the coulomb and exchange matrices for the orbitals with spin b
        self%coulomb_matrix => self%coulomb_matrix_b       
        self%exchange_matrix => self%exchange_matrix_b
        self%two_electron_integral_evaluated => self%two_electron_integral_evaluated_b
        self%two_electron_integrals => self%two_electron_integrals_b
        call self%calculate_two_electron_integrals(evaluate_all)
        call self%calculate_coulomb_matrix(evaluate_all)
        call self%form_exchange_matrix()

        ! form the fock matrix for spin b
        self%hamiltonian_matrix_b = self%kinetic_matrix_b + self%nuclear_electron_matrix_b &
                                   + self%coulomb_matrix_b - self%exchange_matrix_b
        self%eigen_values_b = [(self%hamiltonian_matrix_b(i, i), i = 1, size(self%orbitals_b))]
        
        self%eigen_values => self%eigen_values_b
        call self%print_orbital_energies(.TRUE.)
    
        call bigben%stop()

        
        ! calculate the total energy
        self%energy= 0.5d0 * sum([ (self%kinetic_matrix_a(i,i) +  &
                                    self%nuclear_electron_matrix_a(i,i) + &
                                    self%hamiltonian_matrix_a(i, i), i=1,self%nocc_a) ]) + &
                     0.5d0 * sum([ (self%kinetic_matrix_b(i,i) +  &
                                    self%nuclear_electron_matrix_b(i,i) + &
                                    self%hamiltonian_matrix_b(i, i), i=1,self%nocc_b) ]) 

        ! stop scf cycle timing
        call bigben%stop_and_print()
    end subroutine



!--------------------------------------------------------------------!
!        LCMO Cycle
!--------------------------------------------------------------------!
        

    subroutine LCMOSCFCycle_update(self, scaling_factor)
        class(LCMOSCFCycle), intent(inout), target:: self
        real(REAL64),     intent(in)              :: scaling_factor
        integer                                   :: i, j, k, l, m, n, orbital_count

        orbital_count = size(self%orbitals)
        !call pinfo("Running LCMO update")
        ! orthogonalize fock matrix
        self%hamiltonian_matrix = xmatmul(self%orthogonalizing_matrix_a, &
            xmatmul(self%hamiltonian_matrix, self%orthogonalizing_matrix_a))

        ! diagonalize orthonormal fock matrix
        call matrix_eigensolver(self%hamiltonian_matrix, self%eigen_values_a, self%coefficients_a)
        ! transform coefficients to original non-orthogonal basis
        self%coefficients_a = xmatmul(self%orthogonalizing_matrix_a, self%coefficients_a)
        ! get the density matrix
        self%density_matrix_a =   (1.0d0-scaling_factor) * self%density_matrix_a &
                                + scaling_factor* get_density_matrix(self%coefficients_a, self%nocc_a)

        
        call self%print_orbital_energies(.FALSE.)
        
    end subroutine
    
    !> Get the occupied orbitals stored to array of functions
    subroutine LCMOSCFCycle_get_occupied_orbitals(self, orbitals)
        class(LCMOSCFCycle), intent(in)                   :: self
        type(Function3D),    intent(inout),  allocatable  :: orbitals(:)
        type(Function3D)                                  :: temp
        integer                                           :: i, iorbital

        allocate(orbitals(self%nocc))
        call bigben%split("Computing LCMO occupied orbitals")
        print *, "NOCC", self%nocc
        do iorbital = 1, self%nocc
            do i = 1, size(self%orbitals)
                if (i == 1) then
                    call orbitals(iorbital)%init_copy(self%orbitals(i), copy_content = .TRUE.)
                    call orbitals(iorbital)%product_in_place_REAL64(self%coefficients(i, iorbital))
                else
                    call temp%init_copy(self%orbitals(i), copy_content = .TRUE.)
                    call temp%product_in_place_REAL64(self%coefficients(i, iorbital))
                    call orbitals(iorbital)%add_in_place(temp)
                    call temp%destroy()
                end if
            end do
        end do
        call bigben%stop()
    end subroutine

    
    !> Calculate the electron density from the basis set stored in 'orbitals'
    !! and the density matrix stored in 'density_matrix'
    subroutine LCMOSCFCycle_calculate_electron_density(self, non_overlapping)
        class(LCMOSCFCycle), intent(inout)                :: self
        logical, optional,                  intent(in)    :: non_overlapping
        integer                                           :: iorbital, jorbital
        real(REAL64)                                      :: temp

        if (allocated(self%temp)) then
            call self%temp%destroy()
            deallocate(self%temp)
        end if                
        if (allocated(self%temp2)) then
            call self%temp2%destroy()
            deallocate(self%temp2)
        end if 
        call bigben%split("Computing LCMO electron density")
        do iorbital = 1, size(self%orbitals)
            do jorbital = 1, iorbital
                if (jorbital == 1 .and. iorbital == 1) then
                    call self%orbitals(jorbital)%multiply_sub(self%orbitals(iorbital), self%temp2)
                    call self%temp2%product_in_place_REAL64(self%density_matrix(iorbital, jorbital))
                else
                    call self%orbitals(jorbital)%multiply_sub(self%orbitals(iorbital), self%temp)
                    if (jorbital /= iorbital) then
                        call self%temp%product_in_place_REAL64(  self%density_matrix(jorbital, iorbital) &
                                                            + self%density_matrix(iorbital, jorbital))
                    else
                        call self%temp%product_in_place_REAL64(self%density_matrix(iorbital, jorbital))
                    end if
                    call self%temp2%add_in_place(self%temp)
                end if
            end do
        end do
        
        call self%electron_density%destroy()
        call self%electron_density%init_copy(self%temp2, copy_content = .TRUE.)
        self%electron_density%type = F3D_TYPE_CUSP
        if (allocated(self%temp)) then
            call self%temp%destroy()
            deallocate(self%temp)
        end if                
        if (allocated(self%temp2)) then
            call self%temp2%destroy()
            deallocate(self%temp2)
        end if 
        call self%electron_density%communicate_cube(reversed_order = .TRUE.)
        call bigben%stop()
    end subroutine
    
    subroutine LCMOSCFCycle_destroy(self)
        class(LCMOSCFCycle), intent(inout), target:: self

       
        if (allocated(self%orthogonalizing_matrix_a)) deallocate(self%orthogonalizing_matrix_a)
        if (allocated(self%orthogonalizing_matrix_b)) deallocate(self%orthogonalizing_matrix_b)
        if (allocated(self%density_matrix_a))         deallocate(self%density_matrix_a)
        if (allocated(self%density_matrix_b))         deallocate(self%density_matrix_b)
        nullify(self%density_matrix)
    
        call SCFCycle_destroy(self)
    end subroutine

!--------------------------------------------------------------------!
!        LCMO RHF Cycle
!--------------------------------------------------------------------!


    subroutine LCMORHFCycle_init(orbitals, electron_density, nocc_a, nocc_b, virtual_orbital_count, &
                           laplacian_operator, coulomb_operator, result_object, &
                           orbital_coefficients)
        !> Input orbitals
        type(Function3D), intent(in)                  :: orbitals(:)
        !> Input electron density holder
        type(Function3D), intent(in)                  :: electron_density
        !> Number of occupied orbitals with a spin
        integer, intent(in)                           :: nocc_a
        !> Number of occupied orbitals with b spin
        integer, intent(in)                           :: nocc_b
        !> Number of virtual orbitals
        integer, intent(in)                           :: virtual_orbital_count
        !> Laplacian operator
        class(Laplacian3D), intent(inout), target     :: laplacian_operator
        class(Coulomb3D), intent(inout),   target     :: coulomb_operator
        ! the result object
        class(SCFCycle), allocatable, target          :: result_object
        type(LCMORHFCycle), pointer                   :: new
        integer                                       :: iorbital, orbital_count, i
        ! Initial molecular orbital coefficients
        real(REAL64),   intent(in)                    :: orbital_coefficients(:, :)

        allocate(LCMORHFCycle :: result_object)
        orbital_count = min(nocc_a + virtual_orbital_count, size(orbitals))
        result_object%total_orbital_count = orbital_count

        if (nocc_b > nocc_a) then
            print "('ERROR: The number of b occupied orbitals (', i3,') exceeds the number of occupied a orbitals&
                    & (', i3,'). This is not allowed.')", nocc_b, nocc_a
            stop
        end if

        ! set the non-polymorphic pointer to point to new
        select type(result_object)
            type is (LCMORHFCycle)
                new => result_object
        end select 
       
        ! Note: orbitals_a is the place where all the orbitals are, 'all_orbitals'
        !       is the pointers to all the orbitals
        allocate(new%orbitals_a(orbital_count))
        allocate(new%all_orbitals(orbital_count))
        !print *, "orbitals size", orbital_count, "dot", orbitals(1) .dot. orbitals(1)
        do iorbital = 1, orbital_count
            call new%orbitals_a(iorbital)%init_copy(orbitals(iorbital), copy_content = .TRUE.)
            new%all_orbitals(iorbital)%p => new%orbitals_a(iorbital)
        end do
        allocate(Function3D :: new%potential_input)
        call new%potential_input%init_copy(electron_density)
        
        new%laplacian_operator => laplacian_operator
        new%coulomb_operator => coulomb_operator
        
        new%nocc_a = nocc_a
        new%nocc_b = nocc_b
        new%nvir_a = virtual_orbital_count
        new%nvir_b = virtual_orbital_count
        allocate(new%kinetic_matrix_a  (orbital_count,orbital_count))
        allocate(new%nuclear_electron_matrix_a (orbital_count,orbital_count))
        allocate(new%overlap_matrix_a    (orbital_count,orbital_count))
        allocate(new%hamiltonian_matrix_a(orbital_count,orbital_count))   
        allocate(new%coefficients_a      (orbital_count,orbital_count))  
        allocate(new%eigen_values_a      (orbital_count), source = 0.0d0)  
        allocate(new%two_electron_integrals_a(orbital_count,orbital_count, &
                                            orbital_count,orbital_count))
        allocate(new%two_electron_integral_evaluated_a(orbital_count,orbital_count, &
                                            orbital_count,orbital_count), source = .FALSE.)

        ! set the pointers to correct arrays
        new%kinetic_matrix => new%kinetic_matrix_a
        new%nuclear_electron_matrix => new%nuclear_electron_matrix_a
        new%overlap_matrix => new%overlap_matrix_a
        new%hamiltonian_matrix => new%hamiltonian_matrix_a
        new%coefficients => new%coefficients_a
        new%eigen_values => new%eigen_values_a
        new%two_electron_integrals => new%two_electron_integrals_a
        new%two_electron_integral_evaluated => new%two_electron_integral_evaluated_a
        new%nocc => new%nocc_a
        new%orbitals => new%orbitals_a
        new%restricted = .TRUE.
        !call new%gs_orthogonalize_orbitals()
        ! calculate the two electron integrals
        call new%calculate_all_two_electron_integrals()

        ! calculate the kinetic matrix, overlap matrix and nuclear-electron repulsion matrix
        call new%calculate_one_electron_matrices(.TRUE.)

        ! calculate the initial density matrix, hamiltonian matrix and orthogonalizing matrix
        new%orthogonalizing_matrix_a = get_orthogonalizing_matrix(new%overlap_matrix_a)
        new%coefficients_a = orbital_coefficients(:orbital_count, :orbital_count)
        new%density_matrix_a = get_density_matrix(new%coefficients_a, new%nocc_a)
        new%density_matrix => new%density_matrix_a

    end subroutine

    subroutine LCMORHFCycle_calculate_hamiltonian_matrix(self, evaluate_all)
        class(LCMORHFCycle), intent(inout), target   :: self
        logical,             intent(in)              :: evaluate_all
        integer                                      :: i, j, k, l, norbitals, occupations(size(self%orbitals))

        norbitals = size(self%orbitals) 
        !call pinfo("Updating LCMORHF")
        ! form the fock matrix
        do i = 1, norbitals
            do j = 1, i
                self%hamiltonian_matrix_a(i, j)=self%kinetic_matrix_a(i, j) + self%nuclear_electron_matrix_a(i, j)
                do k = 1, norbitals
                    do l = 1, norbitals
                        self%hamiltonian_matrix_a(i, j) = self%hamiltonian_matrix_a(i, j) + &
                              self%density_matrix_a(k, l) * (self%two_electron_integrals_a(i, j, l, k) &
                                           - 0.5d0 * self%two_electron_integrals_a(i, k, l, j))
                        
                    end do
                end do
                self%hamiltonian_matrix_a(j, i) = self%hamiltonian_matrix_a(i, j)
            end do
        end do

        ! calculate the total energy
        self%energy = 0.0d0
        do i = 1, size(self%density_matrix_a, 1)
            do j = 1, size(self%density_matrix_a, 2)
                self%energy = self%energy &
                +  self%density_matrix_a(i, j) * (self%hamiltonian_matrix(i, j) &
                      + self%kinetic_matrix(i, j) + self%nuclear_electron_matrix(i, j))
            end do
        end do
        self%energy = 0.5d0 * self%energy
    
        
        
    end subroutine

    subroutine LCMORHFCycle_destroy(self)
        class(LCMORHFCycle), intent(inout)       :: self
        integer                                  :: i

        call LCMOSCFCycle_destroy(self)
    end subroutine


!--------------------------------------------------------------------!
!        LCMO RDT Cycle
!--------------------------------------------------------------------!
#ifdef HAVE_DFT

    subroutine LCMORDFTCycle_init(orbitals, electron_density, nocc_a, nocc_b, virtual_orbital_count, &
                           laplacian_operator, coulomb_operator, core_evaluator, &
                           exchange_functional, correlation_functional, &
                           xc_update_method, xc_lmax, result_object, &
                           orbital_coefficients, orbitals_density_evaluation, &
                           finite_diff_order)
        !> Input orbitals
        type(Function3D), intent(in)                  :: orbitals(:)
        !> Input electron density holder
        type(Function3D), intent(in)                  :: electron_density
        !> Number of occupied orbitals with a spin
        integer, intent(in)                           :: nocc_a
        !> Number of occupied orbitals with b spin
        integer, intent(in)                           :: nocc_b
        !> Number of virtual orbitals
        integer, intent(in)                           :: virtual_orbital_count
        !> Laplacian operator
        class(Laplacian3D), intent(inout), target     :: laplacian_operator
        class(Coulomb3D), intent(inout),   target     :: coulomb_operator
        type(CoreEvaluator), intent(inout), target    :: core_evaluator
        !> The type of the Exchange functional
        integer(INT32), intent(in)                    :: exchange_functional
        !> The type of the Correlation functional
        integer(INT32), intent(in)                    :: correlation_functional

        integer(int32), intent(in)                    :: xc_update_method
        integer(int32), intent(in)                    :: xc_lmax
        ! the result object
        class(SCFCycle), allocatable, target          :: result_object
        ! Initial molecular orbital coefficients
        real(REAL64),   intent(in)                    :: orbital_coefficients(:, :)
        !> If electron density and its gradient are evaluated using the more complex
        !! but accurate method in the exchange and correalation. 
        logical,        intent(in)                    :: orbitals_density_evaluation
        integer(INT32), intent(in)                    :: finite_diff_order
        type(LCMORDFTCycle), pointer                  :: new
        integer                                       :: iorbital, orbital_count, i

        allocate(LCMORDFTCycle :: result_object)
        orbital_count = size(orbitals)
        result_object%total_orbital_count = orbital_count

        if (nocc_b > nocc_a) then
            print "('ERROR: The number of b occupied orbitals (', i3,') exceeds the number of occupied a orbitals&
                    & (', i3,'). This is not allowed.')", nocc_b, nocc_a
            stop
        end if

        ! set the non-polymorphic pointer to point to new
        select type(result_object)
            type is (LCMORDFTCycle)
                new => result_object
        end select 
       
        ! Note: orbitals_a is the place where all the orbitals are, 'all_orbitals'
        !       is the pointers to all the orbitals
        allocate(new%orbitals_a(orbital_count))
        allocate(new%all_orbitals(orbital_count))
        !print *, "orbitals size", orbital_count, "dot", orbitals(1) .dot. orbitals(1)
        do iorbital = 1, orbital_count
            call new%orbitals_a(iorbital)%init_copy(orbitals(iorbital), copy_content = .TRUE.)
            new%all_orbitals(iorbital)%p => new%orbitals_a(iorbital)
        end do
        allocate(Function3D :: new%potential_input)
        call new%potential_input%init_copy(electron_density)
        allocate(Function3D :: new%electron_density_a)
        call new%electron_density_a%init_copy(electron_density, copy_content = .TRUE.)
        new%electron_density => new%electron_density_a
        new%laplacian_operator => laplacian_operator
        new%coulomb_operator => coulomb_operator
        
        new%exchange_correlation = XC(new%electron_density_a, exchange_functional, correlation_functional, &
                                      xc_lmax, laplacian_operator, core_evaluator, orbitals_density_evaluation, &
                                      finite_diff_order)
        
        
        new%nocc_a = nocc_a
        new%nocc_b = nocc_b
        new%nvir_a = virtual_orbital_count
        new%nvir_b = virtual_orbital_count
        allocate(new%kinetic_matrix_a  (orbital_count,orbital_count))
        allocate(new%nuclear_electron_matrix_a (orbital_count,orbital_count))
        allocate(new%overlap_matrix_a    (orbital_count,orbital_count))
        allocate(new%hamiltonian_matrix_a(orbital_count,orbital_count))   
        allocate(new%coefficients_a      (orbital_count,orbital_count))  
        allocate(new%eigen_values_a      (orbital_count), source = 0.0d0)  
        allocate(new%two_electron_integrals_a(orbital_count,orbital_count, &
                                            orbital_count,orbital_count))
        allocate(new%two_electron_integral_evaluated_a(orbital_count,orbital_count, &
                                            orbital_count,orbital_count), source = .FALSE.)
        allocate(new%exchange_correlation_matrix_a(orbital_count,orbital_count))   
        allocate(new%coulomb_matrix_a(orbital_count,orbital_count))   

        ! set the pointers to correct arrays
        new%kinetic_matrix => new%kinetic_matrix_a
        new%nuclear_electron_matrix => new%nuclear_electron_matrix_a
        new%overlap_matrix => new%overlap_matrix_a
        new%hamiltonian_matrix => new%hamiltonian_matrix_a
        new%coulomb_matrix => new%coulomb_matrix_a
        new%exchange_correlation_matrix => new%exchange_correlation_matrix_a
        new%coefficients => new%coefficients_a
        new%eigen_values => new%eigen_values_a
        new%two_electron_integrals => new%two_electron_integrals_a
        new%two_electron_integral_evaluated => new%two_electron_integral_evaluated_a
        new%nocc => new%nocc_a
        new%orbitals => new%orbitals_a
        new%restricted = .TRUE.

        
        
        new%coefficients_a = orbital_coefficients(:orbital_count, :orbital_count)
        new%density_matrix_a = get_density_matrix(new%coefficients_a, new%nocc_a)
        new%density_matrix => new%density_matrix_a


    end subroutine
    

    subroutine LCMORDFTCycle_calculate_hamiltonian_matrix(self, evaluate_all)
        class(LCMORDFTCycle), intent(inout), target  :: self
        logical,             intent(in)              :: evaluate_all
        integer                                      :: i, j, k, l, norbitals
        real(REAL64)                                 :: temp
        type(Function3D)                             :: xc_energy_density
        real(REAL64)                                 :: coulomb_energy, kinetic_energy, &
                                                         ne_energy
        type(Function3D), allocatable                :: occupied_orbitals(:)
        
        ! get the one electron matrices and the initial guess on the first iteration
        if (.not. allocated(self%orthogonalizing_matrix_a)) then
            ! calculate the kinetic matrix, overlap matrix and nuclear-electron repulsion matrix
            call self%calculate_one_electron_matrices(.TRUE.)

            ! calculate the initial density matrix, hamiltonian matrix and orthogonalizing matrix
            self%orthogonalizing_matrix_a = get_orthogonalizing_matrix(self%overlap_matrix_a)
        end if

        self%coulomb_potential_is_valid = .FALSE.
        call self%calculate_electron_density(.FALSE.)
        
        call self%calculate_coulomb_matrix(.TRUE.)
        !call self%get_occupied_orbitals(occupied_orbitals)
        call self%exchange_correlation%eval(self%electron_density, &
                                            self%xc_potential_a, xc_energy_density, &
                                            self%xc_energy) !, occupied_orbitals = orbitals(self%nocc))
        call xc_energy_density%destroy()
        self%xc_potential => self%xc_potential_a
        !do i = 1, size(occupied_orbitals)
        !    call occupied_orbitals(i)%destroy()
        !end do
        !deallocate(occupied_orbitals)
        call self%calculate_xc_matrix(.TRUE.)
        
        norbitals = size(self%orbitals) 
        !call pinfo("Updating LCMODFT")
        
        ! form the kohn-sham matrix
        self%hamiltonian_matrix = self%kinetic_matrix + self%nuclear_electron_matrix &
                                    + self%coulomb_matrix + self%exchange_correlation_matrix
                                    

        ! calculate the total energy
        self%energy = 0.0d0
        kinetic_energy = 0.0d0
        ne_energy = 0.0d0
        coulomb_energy = 0.0d0
        do i = 1, size(self%density_matrix, 1)
            do j = 1, size(self%density_matrix, 2)
                self%energy = self%energy &
                +      self%density_matrix_a(i, j) &
                   *  (          self%kinetic_matrix(i, j) &
                       +         self%nuclear_electron_matrix(i, j) &
                       + 0.5d0 * self%coulomb_matrix(i, j))
                ne_energy =   ne_energy &
                            + self%density_matrix_a(i, j) * self%nuclear_electron_matrix(i, j)
                !kinetic_energy = kinetic_energy &
                !            + self%density_matrix_a(i, j) * self%kinetic_matrix(i, j)
                !coulomb_energy = coulomb_energy &
                !            + 0.5d0 * self%coulomb_matrix(i, j) * self%density_matrix_a(i, j)
                       
            end do
        end do
        !print *, "xc_energy", self%xc_energy
        !print *, "kinetic energy", kinetic_energy
        !print *, "coulomb_energy", coulomb_energy
        !print *, "ne energy", ne_energy
        self%energy = self%energy + self%xc_energy
           
        
    end subroutine

    subroutine LCMORDFTCycle_destroy(self)
        class(LCMORDFTCycle), intent(inout)       :: self
        integer                                  :: i

        if(allocated(self%exchange_correlation_matrix_a)) deallocate(self%exchange_correlation_matrix_a)
        if(allocated(self%coulomb_matrix_a))              deallocate(self%coulomb_matrix_a)
    
        !> Function containing the electron density 
        if (allocated(self%electron_density_a)) then
            call self%electron_density_a%destroy()
            deallocate(self%electron_density_a)
        end if
        
        call self%xc_potential_a%destroy()
        
        if (allocated(self%orbitals_a)) then
            do i = 1, size(self%orbitals_a)
                call self%orbitals_a(i)%destroy()
            end do
            deallocate(self%orbitals_a)
        end if 
        
        call self%exchange_correlation%destroy()
        call LCMOSCFCycle_destroy(self)
    end subroutine
#endif
    
end module
