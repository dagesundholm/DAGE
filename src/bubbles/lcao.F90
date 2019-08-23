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
!> @file lcao.F90
!! Construction of molecular orbitals using the Linear Combination of
!! Atomic Orbitals approximation. Clases for AO bases and molecular
!! structures.
module LCAO_m
    use Globals_m
    use Function3D_class
    use Function3D_types_m, only: F3D_TYPE_CUSP
    use ParallelInfo_class
    use GBFMMParallelInfo_class
    use Grid_class
    use Bubbles_class
    use SCFCycle_class
    use Evaluators_class
    use MemoryLeakChecker_m

    implicit none

    public :: Basis

    public :: Structure

    public :: mocoeffs_read
    public :: assignment(=)

        
    type :: Basis
        !> Determines if the basis set is of slater type (=1) or of gaussian type  (=2) or of bubbles type (=3)
        integer              :: basis_set_type
        !> Determines the total number of different a, n, l -quantum number triplets in this basis set
        integer              :: number_of_subshells
        !> Determines the number of different a, n - quantum number pairs in this basis set
        integer              :: number_of_shells
        !> Determines the number of coefficients/exponents per subshell/shell
        integer              :: size
        !> Determines the quantum number 'a' of atoms each of the shells
        integer, allocatable :: atom_type(:)
        !> Determines the number of subshells for each shell
        integer, allocatable :: local_number_of_subshells(:)
        !> Quantum numbers l for each of the subshells
        integer, allocatable :: l(:)
        !> Coefficients for the basis set functions
        real(REAL64), allocatable :: coefficients(:, :)
        !> Exponents
        real(REAL64), allocatable :: exponentials(:, :)
        !> If the basis is restricted of nature
        logical              :: restricted
        !> Determines the number of different folders in this basis set
        integer              :: number_of_folders
        !> Determines the quantum number 'a' of atoms each of the folders
        integer, allocatable :: folder_atom_type(:)
        !> Determines the number of subshells for each shell
        character(256), allocatable :: folder_path(:)
    contains
        procedure            :: make_lcao_mos => Basis_make_lcao_mos
        procedure            :: make_basis_functions     => Basis_make_basis_functions
        procedure            :: make_basis_functions_gto => Basis_make_basis_functions_gto
        procedure            :: evaluate_radial_function => Basis_evaluate_radial_function 
        procedure, private   :: init_basis_functions         => Basis_init_basis_functions
        procedure, private   :: get_total_number_of_basis_functions &
                                                         => Basis_get_total_number_of_basis_functions
        procedure, private   :: init_orbitals                => Basis_init_orbitals
        procedure, private   :: make_lcao_mos_bubbles => Basis_make_lcao_mos_bubbles
        procedure, private   :: make_lcao_mos_gto     => Basis_make_lcao_mos_gto
    end type


    !> Nuclear coordinates and atom type indices
    type :: Structure
        !> Atom type id
        integer,      allocatable :: atom_type(:)
        !> Atomic number
        real(REAL64), allocatable :: nuclear_charge(:)
        !> Sum of all nuclear and electronic charges
        real(REAL64)              :: system_charge
        !> Indeces of the ignored basis functions for each atom
        integer, allocatable      :: ignored_basis_functions(:, :)
        !> The number of basis functions taken into account
        integer, allocatable      :: number_of_basis_functions(:)
        !> Nuclear positions
        real(REAL64), allocatable :: coordinates(:,:)
        !> Molecular orbital coefficients 
        real(REAL64), allocatable :: orbital_coefficients(:, :)
        !> Molecular orbital spins: 0: a, 1: b
        integer,      allocatable :: orbital_spin(:)
        !> Multiplicity of the electronic state
        integer                   :: multiplicity
        !> Number of virtual orbitals
        integer                   :: number_of_virtual_orbitals
        !> Name of the structure
        character(256)            :: name
        !> Name of the basis set used
        character(256)            :: basis_set_name
        integer                   :: basis_set_id
        integer                   :: number_of_atoms
        integer                   :: number_of_orbitals
        real(REAL64)              :: external_electric_field(3)
    contains
        procedure :: destroy                   => Structure_destroy
        procedure :: get_natoms                => Structure_get_natoms
        procedure :: make_f3d_mould            => Structure_make_f3d_mould
        procedure :: make_cubegrid             => Structure_make_cubegrid
        procedure :: make_bubblegrids          => Structure_make_bubblegrids
        procedure :: get_different_atom_types  => Structure_get_different_atom_types
        procedure :: get_number_of_atom_types  => Structure_get_number_of_atom_types
        procedure :: get_atom_type_order_number => Structure_get_atom_type_order_number
        procedure :: get_number_of_occupied_orbitals => Structure_get_number_of_occupied_orbitals
        procedure :: get_orbital_coefficients  => Structure_get_orbital_coefficients
        procedure :: get_max_number_of_basis_functions => Structure_get_max_number_of_basis_functions
        procedure :: get_number_of_ignored_basis_functions &
                                               => Structure_get_number_of_ignored_basis_functions
        procedure, private :: write_atoms      => Structure_write_atoms
        procedure, private :: write_orbital_coefficients => Structure_write_orbital_coefficients
        procedure, private :: write_all_coefficients     => Structure_write_all_coefficients
        procedure          :: write_structure  => Structure_write_structure
    end type


    interface Structure
        module procedure :: Structure_init_explicit
        module procedure :: Structure_init_read
    end interface
contains
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%    Basis                                                               %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function Basis_evaluate_radial_function(self, ishell, isubshell, r) result(radial_function)
        class(Basis), intent(in)  :: self
        integer,      intent(in)  :: ishell
        integer,      intent(in)  :: isubshell
        real(REAL64), intent(in)  :: r(:)
        real(REAL64)              :: radial_function(size(r)), norm_const

        integer                   :: l, i

        l = self%l(isubshell)
        radial_function = 0.0d0
        do i = 1, size(self%exponentials, 1)
            norm_const = gto_norm_const( self%exponentials(i, ishell), l)
            radial_function = radial_function + norm_const  *  &
                     self%coefficients(i, isubshell) * exp(-self%exponentials(i, ishell)*r**2 )
        end do
        if (l > 0) radial_function = radial_function * r**l
        
    end function


!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%   Structure                                                                  %
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function Structure_init_explicit(atom_type, nuclear_charge, coordinates) result(new)
        integer,      intent(in) :: atom_type(:)
        real(REAL64), intent(in) :: nuclear_charge(size(atom_type))
        real(REAL64), intent(in) :: coordinates(3,size(atom_type))
        type(Structure)           :: new
  
        new%atom_type      = atom_type
        new%nuclear_charge = nuclear_charge
        new%coordinates    = coordinates
    end function

    !> Initialize Structure from file.

    !> The format of the file is:
    !!~~~~~~~~~~~~~~~~~~~~~~~~
    !![NUMBER OF ATOMS]
    !![ATOMTYPE OF ATOM 1] [Z OF ATOM 1] [X,Y,Z COORDINATES (IN A.U.) OF ATOM 1]
    !![ATOMTYPE OF ATOM 2] [Z OF ATOM 2] [X,Y,Z COORDINATES (IN A.U.) OF ATOM 2]
    !!...
    !![ATOMTYPE OF ATOM N] [Z OF ATOM N] [X,Y,Z COORDINATES (IN A.U.) OF ATOM N]
    !!~~~~~~~~~~~~~~~~~~~~~~~~
    function Structure_init_read(coords_fd) result(new)
        !> Descriptor of the file containing the molecular structure info.

        !> The file must be already open with default `access="sequential"`
        !! and `form="formatted"`.
        integer, intent(in)      :: coords_fd

        type(Structure)           :: new

        integer                  :: natoms
        integer                  :: iatom

        read(coords_fd,*) natoms
        allocate(new%atom_type(natoms))
        allocate(new%nuclear_charge(natoms))
        allocate(new%coordinates(3, natoms))
        do iatom=1,natoms
            read(coords_fd,*) new%atom_type(iatom), new%nuclear_charge(iatom), new%coordinates(:,iatom)
        end do
    end function

    !> Return the number of atoms in the structure.
    pure function Structure_get_natoms(self) result(natoms)
        class(Structure), intent(in) :: self
        integer                     :: natoms

        natoms=size(self%atom_type)
    end function

    !> Return the number of atom types in the structure
    pure function Structure_get_number_of_atom_types(self) result(ntypes)
        class(Structure), intent(in) :: self
        integer                      :: ntypes, i, j
        logical                      :: found
    
        ntypes = 0
        do i = 1, self%get_natoms()
            found = .FALSE.
            do j = 1, i-1
                if (self%atom_type(j) == self%atom_type(i)) then
                    found = .TRUE.
                end if
            end do
            if (.NOT. found) then
                ntypes = ntypes + 1
            end if
        end do
    end function

    !> Return the number of atom types in the structure
    pure function Structure_get_different_atom_types(self) result(types)
        class(Structure), intent(in) :: self
        integer                      :: ntypes, i, j
        integer                      :: types(self%get_number_of_atom_types())
        logical                      :: found
    
        ntypes = 0
        do i = 1, self%get_natoms()
            found = .FALSE.
            do j = 1, i-1
                if (self%atom_type(j) == self%atom_type(i)) then
                    found = .TRUE.
                end if
            end do
            if (.NOT. found) then
                ntypes = ntypes + 1
                types(ntypes) = self%atom_type(i)
            end if
        end do
    end function


    !> Return the orbital coefficients for orbitals with specified spin (0: a, 1: b)
    function Structure_get_orbital_coefficients(self, basis_object, spin) result(orbital_coefficients)
        class(Structure), intent(in) :: self
        type(Basis),      intent(in) :: basis_object
        integer,          intent(in) :: spin
        real(REAL64), allocatable    :: orbital_coefficients(:, :)
        integer                      :: i, j, number_of_orbitals, number_of_basis_functions

        ! get the number of orbitals with spin = 'spin'
        if (spin == 1) then
            number_of_orbitals = sum(self%orbital_spin)
        else
            number_of_orbitals = self%number_of_orbitals - sum(self%orbital_spin)
        end if
        
        ! get the total number of basis functions in the calculation
        number_of_basis_functions = basis_object%get_total_number_of_basis_functions(self)

        if(number_of_orbitals > number_of_basis_functions) then
          write(*,*) 'You have specified ', number_of_orbitals, ' MOs, but there are only ', &
                     number_of_basis_functions, 'basis functions. Exiting ...'
          flush(6)
          call abort()
        endif

        allocate(orbital_coefficients(number_of_basis_functions, number_of_basis_functions), &
                 source = 0.0d0)
        ! go through all orbitals and store the ones with spin == 'spin' 
        j = 1
        do i = 1, self%number_of_orbitals
            if (self%orbital_spin(i) == spin) then
                orbital_coefficients(:, j) = &
                    self%orbital_coefficients(:number_of_basis_functions, i)
                j = j + 1
            end if
        end do

        ! create default coefficients for the orbitals that do not have given coefficients
        do i = j, number_of_basis_functions
            orbital_coefficients(:, i) = 0.0d0
            orbital_coefficients(i, i) = 1.0d0
        end do
    end function


    pure function Structure_get_atom_type_order_number(self, atom_number) result(atom_type)
        class(Structure), intent(in) :: self
        integer, intent(in)          :: atom_number
        integer                      :: types(self%get_number_of_atom_types())
        integer                      :: atom_type, i

        types = self%get_different_atom_types()
        do i = 1, size(types)
            if (atom_number == types(i)) then
                atom_type = i
                exit
            end if
        end do
    end function

    pure function Structure_get_max_number_of_basis_functions(self, atom_type) result(res)
        class(Structure), intent(in) :: self
        integer, intent(in)          :: atom_type
        integer                      :: res, i

        res = -1
        do i = 1, self%get_natoms()
            if (self%atom_type(i) == atom_type) then
                res = max(res, self%number_of_basis_functions(i))
            end if
        end do
    end function

    subroutine Structure_write_structure(self, folder, base_filename, coefficients_a, coefficients_b)
        class(Structure),  intent(in)           :: self
        !> Folder for the file          
        character(len=*),  intent(in)           :: folder
        !> Basename for the file
        character(len=*),  intent(in)           :: base_filename
        !> The stored orbital coefficients with a-spin, if not given
        !! the orbital coefficients stored in 'self' will be stored
        real(REAL64),      intent(in), optional :: coefficients_a(:, :)
        !> The stored orbital coefficients with b-spin, if not given
        !! the orbital coefficients stored in 'self' will be stored
        real(REAL64),      intent(in), optional :: coefficients_b(:, :)
        character(len=:), allocatable           :: filename

        filename = trim(base_filename)//'.xml'
        call pdebug("Dumping structure to `"//trim(folder)//'/'//trim(filename)//"'.", 1)
        open(unit=WRITE_FD, file=trim(folder)//'/'//trim(filename), access='stream', form='formatted')
        write(WRITE_FD, '("<?xml version=""1.0""?>")')
        write(WRITE_FD, '("<structure multiplicity=""", i0, """")') self%multiplicity
        write(WRITE_FD, *) '          name="'//trim(self%name)//'"'
        write(WRITE_FD, *) '          basis_set_name="'//trim(self%basis_set_name)//'"'
        write(WRITE_FD, '("           number_of_virtual_orbitals=""", i0, """ >")') self%number_of_virtual_orbitals
        call self%write_atoms(WRITE_FD)
        
        ! write the orbital coefficients
        if (present(coefficients_a) .or. present(coefficients_b)) then
            if (present(coefficients_a)) call self%write_orbital_coefficients(coefficients_a, 0, WRITE_FD)
            if (present(coefficients_b)) call self%write_orbital_coefficients(coefficients_b, 1, WRITE_FD)
        else
            call self%write_all_coefficients(WRITE_FD)
        end if
        write(WRITE_FD, *)   '</structure>'
        close(unit=WRITE_FD)
        deallocate(filename)
    end subroutine

    subroutine Structure_write_atoms(self, file_descriptor)
        class(Structure),  intent(in)           :: self
        !> The descriptor of the file we are writing to
        integer,           intent(in)           :: file_descriptor
        integer                                 :: i, j
        
        do i = 1, size(self%atom_type)
            write(file_descriptor, '("    <atom atom_type=""",i0,""" charge=""",f7.4,""" ")', advance='no') &
                self%atom_type(i), self%nuclear_charge(i)
                
            ! write the ignored basis functions
            do j = 1, size(self%ignored_basis_functions, 1)
                if (self%ignored_basis_functions(j, i) == 0) then
                    if (j /= 1) then
                        write(file_descriptor, '(""" ")', advance='no') 
                    end if
                    exit
                else
                    if (j == 1) then
                        write(file_descriptor, '("ignored_basis_functions=""",i0)', advance='no') &
                            self%ignored_basis_functions(j, i)
                    else
                        write(file_descriptor, '(", ", i0)', advance = 'no') &
                            self%ignored_basis_functions(j, i)
                    end if
                end if
            end do
            write(file_descriptor, '("coordinates=""",f13.8,", ",f13.8,", ",f13.8,""" />")') &
                self%coordinates(1, i), self%coordinates(2, i), self%coordinates(3, i)
            
        end do
    end subroutine
    
    !> Writes orbital coefficients stored in 'self' as xml to the stream
    !! with specifier 'file_descriptor'.
    subroutine Structure_write_all_coefficients(self, file_descriptor)
        class(Structure),  intent(in)            :: self
        !> The descriptor of the file we are writing to
        integer,           intent(in)            :: file_descriptor
        integer                                  :: i, j
        
        do i = 1, size(self%orbital_coefficients, 2)
            write(file_descriptor, &
                '("    <molecular_orbital orbital_spin=""",i1,""" orbital_coefficients=""")', &
                advance = 'no') &
                self%orbital_spin(i)
            do j = 1, size(self%orbital_coefficients, 1)-1
                write(file_descriptor, '(E15.8, ", ")', advance='no') self%orbital_coefficients(j, i)
            end do
            write(file_descriptor, '(E15.8, """ />")') &
                self%orbital_coefficients(size(self%orbital_coefficients, 1), i)
        end do
    end subroutine
    

    !> Writes orbital coefficients in 'coefficients' as xml to the stream
    !! with specifier 'file_descriptor'. The spin of the written coefficients is given in 'orbital_spin'.
    subroutine Structure_write_orbital_coefficients(self, coefficients, orbital_spin, file_descriptor)
        class(Structure),  intent(in)            :: self
        !> The stored orbital coefficients
        real(REAL64),      intent(in)           :: coefficients(:, :)
        !> The spin of the stored orbital coefficients
        integer,           intent(in)           :: orbital_spin 
        !> The descriptor of the file we are writing to
        integer,           intent(in)           :: file_descriptor 

        integer                                 :: i, j
        real(REAL64)                            :: temp


        do i = 1, size(coefficients, 2)
            write(file_descriptor, &
                '("    <molecular_orbital orbital_spin=""",i1,""" orbital_coefficients=""")', &
                advance = 'no') &
                orbital_spin
            do j = 1, size(coefficients, 1)-1
                write(file_descriptor, '(E15.8, ", ")', advance='no') coefficients(j, i)
            end do
            write(file_descriptor, '(E15.8, """ />")') coefficients(size(coefficients, 1), i)
        end do

    end subroutine
    
    pure subroutine Structure_destroy(self)
        class(Structure), intent(inout) :: self
        if (allocated(self%atom_type))      deallocate(self%atom_type)
        if (allocated(self%nuclear_charge)) deallocate(self%nuclear_charge)
        if (allocated(self%coordinates))    deallocate(self%coordinates)
    end subroutine


    function Structure_make_cubegrid(self, step, radius, nlip, grid_type) result(cubegrid)
        class(Structure),     intent(in)                :: self
        !> Step of the cube grid in \f$\mathrm{a}_0\f$.
        !! Recommended value: 0.1 a0.
        real(REAL64),      intent(in)                   :: step
        !> Radial extent of the atoms for constructing the cube grid in a0.
        !! Recommended value: 8 a0.
        real(REAL64),      intent(in)                   :: radius
        !> Number of Lagrange interpolation polynomials used
        integer,           intent(in)                   :: nlip
        integer,           intent(in)                   :: grid_type
        type(Grid3D)                                    :: cubegrid
        integer                                         :: iatom

        ! Grid3D_init_spheres
        cubegrid=Grid3D(centers   = self%coordinates,&
                        radii     = [( radius, iatom=1,self%get_natoms() )],&
                        step      = step,&
                        nlip      = nlip, &
                        grid_type = grid_type, &
                        gbfmm     = .TRUE. )
    end function

    subroutine Structure_make_bubblegrids(self, grids, n0, cutoff, nlip, grid_type)
        class(Structure),          intent(in)    :: self
        !> The array of grids that is allocated and initialized in this subroutine
        type(Grid1D), allocatable, intent(inout) :: grids(:)
        !> Number of cells for the radial functions of the bubbles for Z=1.
        !! Recommended value: 400.
        integer,                   intent(in)    :: n0
        !> Radial extent of the bubbles in a0.
        !! Recommended value: 20 a0.
        real(REAL64),              intent(in)    :: cutoff
        !> Number of Lagrange interpolation polynomials used
        integer,                   intent(in)    :: nlip
        integer,                   intent(in)    :: grid_type
        integer                                  :: iatom

        allocate(grids(self%get_natoms()))
        do iatom = 1, self%get_natoms()
            grids(iatom) = Grid1D(self%nuclear_charge(iatom), n0, nlip, cutoff, grid_type)
        end do
    end subroutine

    

    !> Construct a dummy Function3D instance adequate to represent orbitals
    !! etc. for this structure.
    function Structure_make_f3d_mould(self, step, lmax, bubble_grids, parallelization_info, &
                                      taylor_series_order, bubbles_center_offset) result(mould)
        class(Structure),     intent(in) :: self
        !> Step of the cube grid in \f$\mathrm{a}_0\f$.
        !! Recommended value: 0.1 a0
        real(REAL64),         intent(in) :: step
        !> Maximum angular momentum number of the bubbles.
        !! Recommended value: 2.
        integer,              intent(in) :: lmax
        !> The bubble grids
        type(Grid1D), target, intent(in) :: bubble_grids(:)
        
        !> The used parallelization info
        class(ParallelInfo), intent(in) :: parallelization_info
        integer, optional, intent(in) :: taylor_series_order
        !> The center adjustment of the bubbles relative to the cube
        !! as cube grid points
        real(REAL64),         intent(in) :: bubbles_center_offset(:)
        type(Function3D)              :: mould
        real(REAL64),     allocatable :: centers(:, :)
        type(Grid1D)                  :: bubbles_grids(self%get_natoms())
        type(Bubbles)                 :: bubs
        type(Grid1DPointer)           :: bubble_grid_pointers(self%get_natoms())
        real(REAL64)                  :: center_modulos(3, self%get_natoms()), adjust(3)
        type(Grid3D), pointer         :: global_grid
        real(REAL64), pointer         :: cell_scales(:)

        integer                       :: iatom, i
       
        global_grid => parallelization_info%get_global_grid()
        do iatom = 1, self%get_natoms()
            bubble_grid_pointers(iatom)%p => bubble_grids(iatom)
            center_modulos(:, iatom) = &
                global_grid%coordinates_to_grid_point_coordinates(self%coordinates(:, iatom))
            center_modulos(:, iatom) = modulo(center_modulos(:, iatom), 1.0d0)
        end do
        do i = X_, Z_
            !print *, i, 1.0d0 - maxval(center_modulos(i, :)),  minval(center_modulos(i, :))
            adjust(i) = &
                max((1.0d0 - maxval(center_modulos(i, :)) - minval(center_modulos(i, :))) / 2.0d0, &
                             maxval(center_modulos(i, :)  - minval(center_modulos(i, :))) / 2.0d0)
            !if (abs(maxval(center_modulos(i, :)) - minval(center_modulos(i, :))) < 0.10d0) then
            !    adjust(i) = adjust(i) - 0.15d0
            !end if
        end do
        !adjust(X_) = 2.5d0
        !adjust(Y_) = 2.5d0!1.5d0
        !adjust(Z_) = -2.20d0!1.2d0 ! adjust(Z_) + 2.0d0
        adjust = bubbles_center_offset
        !adjust = 0.0d0

        centers = self%coordinates
        cell_scales => global_grid%axis(X_)%get_cell_scales()
        centers(X_, :) = centers(X_, :) + adjust(X_) * cell_scales(1)
        cell_scales => global_grid%axis(Y_)%get_cell_scales()
        centers(Y_, :) = centers(Y_, :) + adjust(Y_) * cell_scales(1)
        cell_scales => global_grid%axis(Z_)%get_cell_scales()
        centers(Z_, :) = centers(Z_, :) + adjust(Z_) * cell_scales(1)
     
        
        bubs=Bubbles(&
            lmax    = lmax,&
            centers = centers, &
            global_centers = centers, &
            grids   = bubble_grid_pointers, &
            global_grids = bubble_grid_pointers, &
            z       = self%nuclear_charge, &
            global_z= self%nuclear_charge )
    
        deallocate(centers)
        mould=Function3D( parallelization_info, bubs, type=F3D_TYPE_CUSP, &
                          taylor_series_order = taylor_series_order)
        call bubs%destroy()
    end function


    ! returns a pair, where the first entry is the number of orbitals with at
    ! least one electron, and the second entry is the number of orbitals with
    ! two electrons
    function Structure_get_number_of_occupied_orbitals(self) result(nocc)
        class(Structure),   intent(in) :: self
        integer                        :: nocc(2)
        integer                        :: modulus, multiplicity, n_electrons
        
        n_electrons = nint(sum(self%nuclear_charge)) - self%system_charge
        modulus = mod(n_electrons, 2)
        nocc(1) = n_electrons / 2 + modulus ! at least signly occupied
        nocc(2) = n_electrons / 2 ! double occupied

        ! if no input multiplicity is given, select the singlet or doublet
        ! multiplicity taking into account the number of extra electrons
        if (self%multiplicity == 0) then
            if (modulus == 0) then
                multiplicity = 1
            else
                multiplicity = 2
            end if
        else
            multiplicity = self%multiplicity
        end if

        if (modulus == mod(multiplicity, 2)) then
            print *, "--- ERROR: The input multiplicity is not possible with for the number of &
                     &electrons in the input system!"
            stop
        end if

        if (multiplicity >= 2) then
            nocc(1) = nocc(1) + (multiplicity-1) / 2
            nocc(2) = nocc(2) - (multiplicity-1) / 2
        end if
    end function 


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! % MO constructor                                                            %
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    !> Creates the molecular orbitals as Linear combinations of atomic orbitals. Determines
    !! the basis function types etc by comparing parameters of the basis set 'self'.
    subroutine Basis_make_lcao_mos(self, molec, mould, mos_a, mos_b, &
                                   first_made_orbital, restricted)
        class(Basis),                  intent(in)    :: self
        type(Structure),               intent(in)    :: molec
        type(Function3D),              intent(in)    :: mould
        type(Function3D),              intent(inout) :: mos_a(:)
        type(Function3D),              intent(inout) :: mos_b(:)
        integer,                       intent(in)    :: first_made_orbital(2)
        !> If the calculation for which we are getting the orbitals is restricted
        logical,                       intent(in)    :: restricted

        call self%init_orbitals(mould, first_made_orbital(1), mos_a)
        if (.not. restricted) &
            call self%init_orbitals(mould, first_made_orbital(2), mos_b)

        ! check the type of the basis set and select corresponding subroutine to be called
        if (self%basis_set_type == 1 .or. self%basis_set_type == 2) then
            if (size(mos_a) > 0 .and. first_made_orbital(1) <= size(mos_a)) &
                call self%make_lcao_mos_gto(molec, mos_a, first_made_orbital(1), 0)
                
            if (.not. restricted) then
                if (size(mos_b) > 0 .and. first_made_orbital(2) <= size(mos_b)) then
                    call self%make_lcao_mos_gto(molec, mos_b, first_made_orbital(2), 1)
                end if
            end if
        else if (self%basis_set_type == 3) then
            if (      (size(mos_a) > 0 .and. first_made_orbital(1) <= size(mos_a)) &
                 .or. (size(mos_b) > 0 .and. first_made_orbital(2) <= size(mos_b))) &
                call self%make_lcao_mos_bubbles(molec, mos_a, mos_b, first_made_orbital, &
                                                restricted)
        end if
        
        
    end subroutine
    
    !> Reads the basis functions as Function3D objects.
    subroutine Basis_make_basis_functions(self, molec, mould, basis_functions)
        class(Basis),                  intent(in)    :: self
        type(Structure),               intent(in)    :: molec
        type(Function3D),              intent(in)    :: mould
        type(Function3D), allocatable, intent(inout) :: basis_functions(:)


        ! check the type of the basis set and select corresponding subroutine to be called
        if (self%basis_set_type == 1 .or. self%basis_set_type == 2) then
            call self%make_basis_functions_gto(molec, mould, basis_functions)
        else if (self%basis_set_type == 3) then
            
            !call self%init_orbitals(mould, 1, basis_functions)
        end if
        
        
    end subroutine
    
    

    !> Init blank orbitals to 'mos'  from the 'mould'.
    subroutine Basis_init_orbitals(self, mould, first_made_orbital, mos)
        class(Basis),                  intent(in)    :: self
        type(Function3D),              intent(in)    :: mould
        integer,                       intent(in)    :: first_made_orbital
        type(Function3D),              intent(inout) :: mos(:)
        integer                                      :: imo
        character(20)                                :: label
        
        do imo=first_made_orbital, size(mos)
            write(label, '("MO#",i0)') imo
            mos(imo)=Function3D(mould, type=F3D_TYPE_CUSP, label=label)
            mos(imo)=0.d0
        end do
    end subroutine

    !> Init Bubbles basis functions to Function3D objects by reading the basis set folders
    !! from disk. 
    !!
    !! This subroutine fills the number of basis functions per atom type to 
    !! 'number_of_basis_functions'. The actual basis functions are stored to the 'basis_functions'.
    !! All result arrays are allocated by the subroutine.
    subroutine Basis_init_basis_functions(self, molec, basis_functions, &
                                          atom_type_grids, number_of_basis_functions)
        class(Basis),                             intent(in)    :: self
        type(Structure),                          intent(in)    :: molec
        type(Function3D),    allocatable,         intent(inout) :: basis_functions(:)
        type(Grid1D),        allocatable, target, intent(inout) :: atom_type_grids(:)
        integer,             allocatable,         intent(inout) :: number_of_basis_functions(:, :)
        type(Function3D)                                        :: stored_functions
        integer                                                 :: i, j, ifunction
        integer,             allocatable                        :: atom_types(:), folder_indices(:)
        type(Grid1D),        allocatable                        :: temp_grids(:)
        type(Grid1DPointer)                                     :: atom_type_grid_pointers(1)

        ! get the different atom types present in the Structure 'molec'
        atom_types = molec%get_different_atom_types()
        allocate(folder_indices(size(atom_types)), source = -1)
        allocate(number_of_basis_functions(2, size(atom_types)), source = 0)


        ! detemine the indices of folders containing the needed basis functions
        ! and get the numbers of basis functions per atom type and in total
        do i = 1, size(atom_types)
            do j = 1, size(self%folder_atom_type)
                if (self%folder_atom_type(j) == atom_types(i)) then
                    folder_indices(i) = j
                    number_of_basis_functions(:, i) = &
                        get_number_of_resumable_orbitals(self%folder_path(j), self%restricted)

                    ! verify that there are basis functions in the folder specified, if not,
                    ! raise error
                    if (sum(number_of_basis_functions(:, i)) == 0) then
                        print "('ERROR: Incomplete Bubbles basis! The specified bubbles basis folder &
                                &does not contain basis functions &
                                &for atom type ', i3, '.')", atom_types(i)
                        stop
                    end if

                    exit
                end if
            end do
            ! if there are no folders for the atom type, raise an error
            if (folder_indices(i) == -1) then
                print "('ERROR: Incomplete Bubbles basis! The specified bubbles does &
                       &not contain basis folder&
                       &for atom type ', i3, ' used in input structure.')", atom_types(i)
                stop
            end if
        end do

        ! init the basis functions result array
        allocate(basis_functions(sum(number_of_basis_functions)))
        allocate(atom_type_grids(size(atom_types)))
        ifunction = 1
        ! and resume 
        do i = 1, size(atom_types)
            ! resume the atom type related grids
            call resume_Grid1Ds(temp_grids, self%folder_path(folder_indices(i)), "bubblegrids.g1d")
            atom_type_grids(i) = temp_grids(1)
            call temp_grids(1)%destroy()
            deallocate(temp_grids)
            atom_type_grid_pointers(1)%p => atom_type_grids(i)
            

            ! resume the basis functions with spin a
            do j = 1, number_of_basis_functions(1, i)
                basis_functions(ifunction)%bubbles%global_grids = atom_type_grid_pointers
                call resume_orbital(self%folder_path(folder_indices(i)), j, &
                         self%restricted, .TRUE., basis_functions(ifunction))
                if (basis_functions(ifunction)%bubbles%get_k() /= 0) then
                    call basis_functions(ifunction)%bubbles%set_k(0)
                    call basis_functions(ifunction)%bubbles%extrapolate_origo(3)
                end if
                ifunction = ifunction + 1
            end do

            ! resume the basis functions with spin b
            do j = 1, number_of_basis_functions(2, i)
                basis_functions(ifunction)%bubbles%global_grids = atom_type_grid_pointers
                call resume_orbital(self%folder_path(folder_indices(i)), j, &
                         self%restricted, .FALSE., basis_functions(ifunction))
                if (basis_functions(ifunction)%bubbles%get_k() /= 0) then
                    call basis_functions(ifunction)%bubbles%set_k(0)
                    call basis_functions(ifunction)%bubbles%extrapolate_origo(3)
                end if
                ifunction = ifunction + 1
            end do
        end do
        deallocate(atom_types, folder_indices)
    end subroutine 

    
    !> Create molecular orbitals (MOs) by making a linear combination of atomic orbitals (LCAO)
    !! from basis consisting of Bubbles.
    !!
    !! Modifies the 'mos_a' and 'mos_b' arrays. Upon entry, these must be allocated.
    !! NOTE: ignoring basis functions does not work with this currently
    subroutine Basis_make_lcao_mos_bubbles(self, molec, mos_a, mos_b, first_made_orbital, &
                                           restricted)
        class(Basis),                  intent(in)    :: self
        !> The structure for which the MOs are created, this is needed because the 
        !! orbital coefficients lie in this object
        type(Structure),               intent(in)    :: molec
        !> The result spin a MOs
        type(Function3D),              intent(inout) :: mos_a(:)
        !> The result spin b MOs
        type(Function3D),              intent(inout) :: mos_b(:)
        !> The first indices of orbitals created using this method
        !! for spin a: 'first_made_orbital(1)' and for spin b 'first_made_orbital(2)'
        integer,                       intent(in)    :: first_made_orbital(2)
        !> If the calculation for which we are getting the orbitals is restricted
        logical,                       intent(in)    :: restricted
        type(Function3D), allocatable                :: basis_functions(:)
        integer,          allocatable                :: number_of_basis_functions(:, :), atom_types(:), ibubs(:)
        integer                                      :: itype, imo, iatom, ibasis, i, j, icoeff, atom_type, ibub, nbub, &
                                                        atom_type_order_number, index1, index2, max_number_of_basis_functions
        real(REAL64),     allocatable                :: interpolated_basis_function(:, :), mocoeffs_a(:, :), &
                                                        mocoeffs_b(:, :)
        ! an array to keep track if the orbital has ibub present
        logical,          allocatable                :: contains_ibub_a(:, :)  
        logical,          allocatable                :: contains_ibub_b(:, :)  
        type(SimpleInterpolator1D)                   :: interpol
        type(Bubbles)                                :: temp
        type(Grid1D),     allocatable                :: atom_type_grids(:)
        logical                                      :: ignore

        ! get the molecular orbital coefficients having spin: 0:a and 1:b
        mocoeffs_a = molec%get_orbital_coefficients(self, 0)
        mocoeffs_b = molec%get_orbital_coefficients(self, 1)

        if (size(mocoeffs_a, 2) < size(mos_a)) then
            print "('INPUT ERROR: Not enough molecular orbitals specified&
                    & for with spin value: ', i1 , '. Number of orbitals&
                    & required is ', i3, ' and specified ', i3, '.')", &
                     0, size(mos_a), size(mocoeffs_a, 2) 
            stop
        end if

        if (.not. restricted .and. size(mocoeffs_b, 2) < size(mos_b)) then
            print "('INPUT ERROR: Not enough molecular orbitals specified&
                    & for with spin value: ', i1 , '. Number of orbitals&
                    & required is ', i3, ' and specified ', i3, '.')", &
                     1, size(mos_b), size(mocoeffs_b, 2) 
            stop
        end if

        ! get the different atom types present in the Structure 'molec'
        atom_types = molec%get_different_atom_types()

        ! init the basis functions
        call self%init_basis_functions(molec, basis_functions, atom_type_grids, number_of_basis_functions)


        ! allocate the array to keep track of the ibubs present in each of the mos
        if (size(mos_a) > 0) allocate(contains_ibub_a(mos_a(1)%bubbles%get_nbub_global(), size(mos_a)), source = .FALSE.)
        if (size(mos_b) > 0 .and. .not. restricted) &
            allocate(contains_ibub_b(mos_b(1)%bubbles%get_nbub_global(), size(mos_b)), source = .FALSE.)

        ibasis = 1
        do itype = 1, size(atom_types)
            ! detemine the maximum number of basis functions used for atom of this type
            max_number_of_basis_functions = molec%get_max_number_of_basis_functions(atom_types(itype))
            if (max_number_of_basis_functions == -1) &
                max_number_of_basis_functions = sum(number_of_basis_functions(:, itype))

            do i = 1, max_number_of_basis_functions
                ! init the interpolator that evaluates the values of the basis
                ! function, note: there can be only one grid in the basis function bubbles
                interpol = SimpleInterpolator1D(basis_functions(ibasis)%bubbles%gr(1)%p)

                icoeff = i
                ! go through all bubbles with the same atom type as the basis function
                do iatom = 1, mos_a(1)%bubbles%get_nbub_global()
                    atom_type = nint(mos_a(1)%bubbles%get_z(iatom))
                    atom_type_order_number = molec%get_atom_type_order_number(atom_type)
                    if (  atom_type == atom_types(itype) .and. &
                         (molec%number_of_basis_functions(iatom) == -1 .or. &
                          molec%number_of_basis_functions(iatom) >= i)) then
                          
                        ! check if this function is ignored
                        ignore = .FALSE.
                        do j = 1, size(molec%ignored_basis_functions, 1)
                            if (molec%ignored_basis_functions(j, iatom) == 0) then
                                exit
                            else if (molec%ignored_basis_functions(j, iatom) == i) then
                                ignore = .TRUE.
                                exit
                            end if
                        end do
                        
                        if (.not. ignore) then
                        
                            ! evaluate the basis function values at the grid points
                            ! note: the basis function should only have one bubble
                            interpolated_basis_function = &
                                interpol%eval(basis_functions(ibasis)%bubbles%bf(1)%p, &
                                            mos_a(1)%bubbles%gr(iatom)%p%get_coord())
                            
                            ! multiply the interpolated basis function with the orbital coefficient and
                            ! add to the result orbitals a and b
                            do imo = first_made_orbital(1), size(mos_a)
                                index1 = min(size(interpolated_basis_function, 1), size(mos_a(imo)%bubbles%bf(iatom)%p, 1))
                                index2 = min(size(interpolated_basis_function, 2), size(mos_a(imo)%bubbles%bf(iatom)%p, 2))
                                mos_a(imo)%bubbles%bf(iatom)%p(:index1, :index2) = &
                                    mos_a(imo)%bubbles%bf(iatom)%p(:index1, :index2) &
                                    + mocoeffs_a(icoeff, imo) * interpolated_basis_function(:index1, :index2)
                                contains_ibub_a(iatom, imo) = contains_ibub_a(iatom, imo) .or. abs(mocoeffs_a(icoeff, imo)) > 1e-8 
                            end do

                            if (.not. restricted) then
                                do imo = first_made_orbital(2), size(mos_b)
                                    index1 = min(size(interpolated_basis_function, 1), size(mos_b(imo)%bubbles%bf(iatom)%p, 1))
                                    index2 = min(size(interpolated_basis_function, 2), size(mos_b(imo)%bubbles%bf(iatom)%p, 2))
                                    mos_b(imo)%bubbles%bf(iatom)%p(:index1, :index2) = &
                                        mos_b(imo)%bubbles%bf(iatom)%p(:index1, :index2) &
                                        + mocoeffs_b(icoeff, imo) &
                                          * interpolated_basis_function(:index1, :index2)
                                    contains_ibub_b(iatom, imo) = contains_ibub_b(iatom, imo) &
                                                                  .or. abs(mocoeffs_b(icoeff, imo)) > 1e-8 
                                end do
                            end if

                            deallocate(interpolated_basis_function)
                        end if
                    end if

                    if (molec%number_of_basis_functions(iatom) == -1) then
                        icoeff = icoeff + sum(number_of_basis_functions(:, atom_type_order_number))
                    else
                        icoeff = icoeff + min(molec%number_of_basis_functions(iatom), &
                                              sum(number_of_basis_functions(:, atom_type_order_number)))
                    end if
                    icoeff = icoeff - molec%get_number_of_ignored_basis_functions(iatom)
                end do

                ! destroy the interpolator
                call interpol%destroy()

                ! add the basis function counter
                ibasis = ibasis + 1
            end do
        end do
        ! filter the bubbles that are zero from the ibubs of the final orbitals
        allocate(ibubs(mos_a(1)%bubbles%get_nbub_global()))
        do imo = first_made_orbital(1), size(mos_a)
            nbub = 0
            ! store the ibubs of the mo to ibubs 
            do ibub = 1, mos_a(1)%bubbles%get_nbub_global()
                if (contains_ibub_a(ibub, imo)) then
                    nbub = nbub + 1
                    ibubs(nbub) = ibub
                end if
            end do
            temp = Bubbles(mos_a(imo)%bubbles, copy_content = .TRUE.)
            call mos_a(imo)%bubbles%destroy()
            mos_a(imo)%bubbles = temp%get_sub_bubbles(ibubs(:nbub), copy_content = .TRUE.)
            call temp%destroy()
        end do

        if (.not. restricted) then
            do imo = first_made_orbital(2), size(mos_b)
                nbub = 0
                ! store the ibubs of the mo to ibubs 
                do ibub = 1, mos_a(1)%bubbles%get_nbub_global()
                    if (contains_ibub_b(ibub, imo)) then
                        nbub = nbub + 1
                        ibubs(nbub) = ibub
                    end if
                end do
                
                temp = Bubbles(mos_b(imo)%bubbles, copy_content = .TRUE.)
                call mos_b(imo)%bubbles%destroy()
                mos_b(imo)%bubbles = temp%get_sub_bubbles(ibubs(:nbub), copy_content = .TRUE.)
                call temp%destroy()
            end do
        end if
        deallocate(ibubs)
        ! destroy the basis functions 
        do ibasis = 1, size(basis_functions)
            call basis_functions(ibasis)%destroy()
        end do
        ! destroy the grids related to atom types / basis functions
        do i = 1, size(atom_type_grids)
            call atom_type_grids(i)%destroy()
        end do
        deallocate(basis_functions, number_of_basis_functions, atom_types, mocoeffs_a, mocoeffs_b)
        if (allocated(contains_ibub_a)) deallocate(contains_ibub_a)
        if (allocated(contains_ibub_b)) deallocate(contains_ibub_b)
    end subroutine
    

    subroutine Basis_make_lcao_mos_gto(self, molec, mos, first_made_orbital, spin)
        class(Basis),                  intent(in)    :: self
        type(Structure),               intent(in)    :: molec
        type(Function3D),              intent(inout) :: mos(:)
        integer,                       intent(in)    :: first_made_orbital
        integer,                       intent(in)    :: spin

        character(20)     :: label

        integer  :: iatom
        integer  :: l
        integer  :: m
        integer  :: ishell, isubshell, subshell_offset
        integer  :: ifun, ifun_atom
        integer  :: imo, jfun, j
        integer  :: nmo

        type(Grid1D), pointer         :: gr
        real(REAL64), pointer         :: r(:)
        real(REAL64), pointer         :: f_lm(:)
        real(REAL64), allocatable     :: radial_function(:), mocoeffs(:, :)
        logical                       :: ignore

        ! get the molecular orbital coefficients having spin: 'spin'
        mocoeffs = molec%get_orbital_coefficients(self, spin)

        if (size(mocoeffs, 2) < size(mos)) then
            print "('INPUT ERROR: Not enough molecular orbitals specified&
                    & with spin value: ', i1 , '. Number of orbitals&
                    & required is ', i3, ' and specified is ', i3, '.')", &
                     spin, size(mos), size(mocoeffs, 2)
            stop
        end if
        nmo = size(mos)

        ifun=0
        do iatom=1, molec%get_natoms()
            gr=>mos(1)%bubbles%get_grid(iatom)
            r=>gr%get_coord()
            subshell_offset = 0
            ifun_atom = 0
            do ishell = 1, self%number_of_shells
                if (self%atom_type(ishell) == molec%atom_type(iatom)) then
                    do isubshell = 1, self%local_number_of_subshells(ishell)
                        l = self%l(isubshell+subshell_offset)
                        radial_function = self%evaluate_radial_function(ishell, isubshell+subshell_offset, r)
                        do m = -l, l
                            ifun_atom = ifun_atom + 1
                            ! check if this function is ignored
                            ignore = .FALSE.
                            do j = 1, size(molec%ignored_basis_functions, 1)
                                if (molec%ignored_basis_functions(j, iatom) == 0) then
                                    exit
                                else if (molec%ignored_basis_functions(j, iatom) == ifun_atom) then
                                    ignore = .TRUE.
                                    exit
                                end if
                            end do
                            
                            if (.not. ignore) then
                                ifun = ifun + 1
                                do imo = first_made_orbital, nmo
                                    f_lm=>mos(imo)%bubbles%get_f(iatom, l, m)
                                    f_lm = f_lm + mocoeffs(ifun, imo) * radial_function
                                    nullify(f_lm)
                                end do
                            end if
                        end do
                        deallocate(radial_function)
                    end do
                end if
                subshell_offset = subshell_offset + self%local_number_of_subshells(ishell)
            end do
            nullify(gr)
            nullify(r)
        end do
        deallocate(mocoeffs)
    end subroutine
    
    !> Calculate the number of basis functions ignored for the atom 'iatom'
    function Structure_get_number_of_ignored_basis_functions(self, iatom) &
        result(number_of_ignored_basis_functions)
        class(Structure),              intent(in)    :: self
        integer,                       intent(in)    :: iatom
        integer                                      :: j
        integer                                      :: number_of_ignored_basis_functions

        ! calculate the total number of basis functions
        number_of_ignored_basis_functions = 0
        do j = 1, size(self%ignored_basis_functions, 1)
            if (self%ignored_basis_functions(j, iatom) == 0) then
                exit
            else
                number_of_ignored_basis_functions = number_of_ignored_basis_functions +1
            end if
        
        end do
    end function
    
    !> Calculate the total number of basis functions of the system contained
    !! in 'molec' using the basis 'self'.
    !!
    !! Handles erroneous input in 'ignored_basis_functions' of 'molec'.
    function Basis_get_total_number_of_basis_functions(self, molec) &
        result(number_of_basis_functions)
        class(Basis),                  intent(in)    :: self
        type(Structure),               intent(in)    :: molec
        integer                                      :: iatom
        integer                                      :: l
        integer                                      :: m
        integer                                      :: ishell, isubshell, subshell_offset
        integer                                      :: ifun
        integer                                      :: j
        integer                                      :: number_of_basis_functions, & 
                                                        number_of_basis_functions_for_atom

        ! calculate the total number of basis functions
        number_of_basis_functions = 0
        do iatom = 1, molec%get_natoms()
            number_of_basis_functions_for_atom = 0
            subshell_offset = 0
            do ishell = 1, self%number_of_shells
                if (self%atom_type(ishell) == molec%atom_type(iatom)) then
                    do isubshell = 1, self%local_number_of_subshells(ishell)
                        l = self%l(isubshell+subshell_offset)
                        number_of_basis_functions = number_of_basis_functions + 2*l+1
                        number_of_basis_functions_for_atom = &
                            number_of_basis_functions_for_atom + 2*l+1
                    end do
                end if
                subshell_offset = subshell_offset + self%local_number_of_subshells(ishell)
            end do
            
            do j = 1, size(molec%ignored_basis_functions, 1)
                if (molec%ignored_basis_functions(j, iatom) == 0) then
                    exit
                else if (molec%ignored_basis_functions(j, iatom) < 0 .or. &
                         molec%ignored_basis_functions(j, iatom) > number_of_basis_functions_for_atom) then
                    print '("INPUT ERROR: Invalid ignored basis function id: ", i0)', &
                        molec%ignored_basis_functions(j, iatom)
                else
                    number_of_basis_functions = number_of_basis_functions -1
                end if
            end do
        end do
    end function

    !> Get the basis functions of a GTO basis and get them as Function3D objects. 
    !! The resulting objects are stored and returned in 'basis_functions'. 
    subroutine Basis_make_basis_functions_gto(self, molec, mould, basis_functions)
        class(Basis),                  intent(in)    :: self
        type(Structure),               intent(in)    :: molec
        type(Function3D),              intent(in)    :: mould
        type(Function3D), allocatable, intent(out)   :: basis_functions(:)

        character(20)                                :: label

        integer                                      :: iatom
        integer                                      :: l
        integer                                      :: m
        integer                                      :: ishell, isubshell, subshell_offset
        integer                                      :: ifun
        integer                                      :: imo, jfun, j, ifun_atom
        integer                                      :: number_of_basis_functions

        type(Grid1D), pointer                        :: gr
        real(REAL64), pointer                        :: r(:)
        real(REAL64), pointer                        :: f_lm(:)
        real(REAL64), allocatable                    :: radial_function(:)
        logical                                      :: ignore

        ! calculate the total number of basis functions
        number_of_basis_functions = self%get_total_number_of_basis_functions(molec)
        
        allocate(basis_functions(number_of_basis_functions))
        ifun=0
        do iatom=1, molec%get_natoms()
            
            gr=>mould%bubbles%get_grid(iatom)
            r=>gr%get_coord()
            subshell_offset = 0
            ifun_atom = 0
            do ishell = 1, self%number_of_shells
                if (self%atom_type(ishell) == molec%atom_type(iatom)) then
                    do isubshell = 1, self%local_number_of_subshells(ishell)
                        l = self%l(isubshell+subshell_offset)
                        radial_function = self%evaluate_radial_function(ishell, isubshell+subshell_offset, r)
                        do m = -l, l
                            ifun_atom = ifun_atom +1
                            
                            ! check if this function is ignored
                            ignore = .FALSE.
                            do j = 1, size(molec%ignored_basis_functions, 1)
                                if (molec%ignored_basis_functions(j, iatom) == 0) then
                                    exit
                                else if (molec%ignored_basis_functions(j, iatom) == ifun_atom) then
                                    ignore = .TRUE.
                                    exit
                                end if
                            end do
                            
                            if (.not. ignore) then
                                ifun = ifun + 1
                                
                                ! init the orbital with bubbles containing only the bubble corresponding
                                ! to the current atom (as other bubbls do not contain any data that is not 0)
                                write(label, '("AO#",i0)') ifun
                                basis_functions(ifun)=Function3D(mould, type=F3D_TYPE_CUSP, label=label)
                                basis_functions(ifun)=0.0d0
                                !call basis_functions(ifun)%bubbles%destroy()
                                !basis_functions(ifun)%bubbles = mould%bubbles%get_sub_bubbles(ibubs=[iatom])
                                
                                ! copy the radial function to the basis function
                                f_lm=>basis_functions(ifun)%bubbles%get_f(iatom, l, m)
                                f_lm = radial_function
                            end if
                        end do
                        deallocate(radial_function)
                    end do
                end if
                subshell_offset = subshell_offset + self%local_number_of_subshells(ishell)
            end do
            nullify(gr)
            nullify(r)
        end do
        
        
    end subroutine 

    !> Read MO coefficients from file.

    !> The format consists of the number of total and occupied orbitals in the
    !! first line, and all the coefficients of each MO in each of the following
    !! lines.
    subroutine mocoeffs_read(mocoeffs_fd, mocoeffs, nocc)
        !> Descriptor of the file containing the MO coefficients.

        !> The file must be already open with default `access="sequential"`
        !! and `form="formatted"`.
        integer,                   intent(in)  :: mocoeffs_fd
        integer,                   intent(out) :: nocc
        real(REAL64), allocatable, intent(out) :: mocoeffs(:,:)

        integer      :: ntot
        integer      :: imo

        read(mocoeffs_fd,*) ntot,nocc

        allocate(mocoeffs(ntot, ntot))
        do imo=1,ntot
            read(mocoeffs_fd,*) mocoeffs(:,imo)
        end do
    end subroutine

    pure function gto_norm_const(alpha, l) result(const)
        real(REAL64), intent(in) :: alpha
        integer,      intent(in) :: l
        real(REAL64)             :: const

        const=2.d0**(.75d0+l) * alpha**(.75d0+.5d0*l) * &
              PI**(-.75d0) * dfact(2*l-1)**(-.5d0)
    end function

    !> Double factorial (\f$7!=7\cdot 5\cdot 3\cdot 1\f$)
    pure function dfact(i)
        integer,       intent(in) :: i
        integer(INT64)            :: dfact
        integer                   :: j
        dfact=1
        do j=i,1,-2
            dfact=dfact*j
        end do
    end function

end module

