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

! Auxiliary module for reading xyz files
module read_xyz_file_m
    use ISO_FORTRAN_ENV
    use globals_m
    implicit none
    public :: read_xyz_file
    private
contains
    subroutine read_xyz_file(pos, fname)
        real(REAL64), allocatable, intent(out) :: pos(:,:)
        character(len=*) :: fname

        real(REAL64) :: xyz(3)
        character(len=2) :: element
        integer :: natom, iatom

        open(unit=44, file=fname, action='read')
        read(unit=44, fmt=*) natom
        read(unit=44, fmt=*)
        allocate(pos(0:3, natom), source=0.0d0)

        
        do iatom=1, natom
            read(unit=44, fmt=*) element, xyz
            pos(0, iatom) = elem2charge(element)
            pos(1:,iatom) = xyz
        enddo

        ! The coordinates in xyz files are in angstroms by definition
        pos(1:, :) = A2AU*pos(1:, :)

        close(44)
    end subroutine
   

    function elem2charge(element) result(q)
        character(len=2), intent(in) :: element
        real(REAL64) :: q
        select case (element)
        case ('h')
            q = 1
        case ('he')
            q = 2
        case ('li')
            q = 3
        case ('be')
            q = 4
        case ('b')
            q = 5
        case ('c')
            q = 6
        case ('n')
            q = 7
        case ('o')
            q = 8
        case ('f')
            q = 9
        case ('ne')
            q = 10
        case ('mg')
            q = 12
        case default
            q = 0
        end select
    end function
    
   
end module 

program gbfmm
    use ISO_FORTRAN_ENV
#ifdef HAVE_MPI
    use mpi
#endif
    use LCAO_m
    use read_xyz_file_m
    use GaussGenerator_class
    use SlaterGenerator_class
    use Generator_class
    use Function3D_class
    use Coulomb3D_class
    use GBFMMCoulomb3D_class
    use GBFMMFunction3D_class
    use Globals_m

    use Grid_class 
    implicit none

    integer                             :: argc
    character(len=256)                  :: arg
    character(len=256)                  :: system
    integer                             :: i, test_type

    real(REAL64), parameter             :: alphagau = 1.0d0
    real(REAL64), allocatable           :: pos(:,:)
    integer                             :: Nelectron

    ! GBFMM variables
    real(REAL64)                        :: Jpq(2), Jpq_comp(2)
    real(REAL64)                        :: Jpq_ex, stepmax
    
    
    type(SlaterGenerator)                :: rhogen

    

    ! MPI variables
    integer                             :: ierr, lmax, maxlevel
    integer, parameter                  :: MASTER = 0
    
#ifdef HAVE_MPI
    call mpi_init(ierr)
    call mpi_comm_rank(MPI_COMM_WORLD, iproc, ierr)
    call mpi_comm_size(MPI_COMM_WORLD, nproc, ierr)
#endif

    !if (nproc == 1) then
    !    print *, 'GBFMM is meant to run with MPI. Run with at least two nodes!'
    !    call mpi_finalize(ierr)
    !    stop
    !endif

    verbo_g = -1
    argc = command_argument_count()
    if (argc /= 5) then
        call get_command_argument(0, arg)
        write(ERROR_UNIT, "('Usage: ', A,' INPUT_FILE', ' STEP', ' LMAX', ' MAXLEVEL', ' TEST_TYPE')")  trim(arg)
        stop
    endif

    call get_command_argument(1, arg)
    system = trim(arg)

    call read_xyz_file(pos, trim(arg))
    Nelectron = int(sum(pos(0,:)))

    call get_command_argument(2, arg)
    read(arg, fmt=*) stepmax

    call get_command_argument(3, arg)
    read(arg, fmt=*) lmax

    call get_command_argument(4, arg)
    read(arg, fmt=*) maxlevel

    call get_command_argument(5, arg)
    read(arg, fmt=*) test_type


    
    ! Header
    if (iproc == MASTER) then
        call pprint("GB-FMM -- Grid-based Fast Multipole Method")
        call pprint("Revision: @BUBBLES_REVISION_NUMBER@")
        print *, 'system ', system
        print *, 'number of electrons', Nelectron
        print *, 'step', stepmax
        print *, 'lmax', lmax
        print *, 'maxlevel', maxlevel
    endif
    
    bigben = Chrono("GBFMM Test")
    select case(test_type) 
        case (2)
            Jpq = new_test_gaussian_bubbles(pos, lmax, maxlevel, stepmax)
            Jpq_comp = new_test_gaussian_bubbles_basic(pos, lmax, maxlevel, stepmax)
        case (1)
            Jpq = new_test_gaussian(pos, lmax, maxlevel, stepmax)
            Jpq_comp = new_test_gaussian_basic(pos, lmax, maxlevel, stepmax)
        case DEFAULT
            Jpq = new_test(pos, lmax, maxlevel, stepmax)
    end select
    call bigben%stop()
   
    if (iproc == MASTER) then
        Jpq_ex = 0.0
        !Jpq_ex = rhogen%selfint(rhogen%gen_cubegrid())
        write(OUTPUT_UNIT, *) 'FMM', Jpq(1)
        write(OUTPUT_UNIT, *) 'Dage', Jpq_comp(1)
        write(OUTPUT_UNIT, *) 'Exact', Jpq(2)
        write(OUTPUT_UNIT, *) 'Absolute Error', abs(Jpq(1)-Jpq(2))
        write(OUTPUT_UNIT, *) 'Abs. Relative Error', abs(Jpq(1)/Jpq(2) - 1.0d0)

        ! The estimate is calculated based on the fact that the Coulomb matrix is
        ! symmetric and that there are N/2 occupied MOs
        ! The timings are stored in the GBFMM module

        call bigben%print(maxdepth=1)
        !write(OUTPUT_UNIT, '("Potential/s ")') 
        !do i=1, ubound(tpot, 1)
        !    print *, 'Node', i, 'Time/s', tpot(i)
        !enddo
        !write(OUTPUT_UNIT, *) repeat('-', 52)
        !write(OUTPUT_UNIT, '("FMM contraction/s ", G0.4)') tcontract(0)
        !do i=1, ubound(tcontract, 1)
        !    print *, 'Node', i, 'Contraction/s', tcontract(i)
        !enddo

        !write (*,*) 'timeEnergy: ', maxval(tcontract)
        !write (*,*) 'itmePotential: ', maxval(tpot)
        !write (*,*) 'timeRun time: ', maxval(tpot) + maxval(tcontract)

        !write(OUTPUT_UNIT, *) repeat('-', 52)
        !write(OUTPUT_UNIT, '("Estimate for constructing the occupied MO block of the Coulomb matrix:")') 
        !write(OUTPUT_UNIT, '(G0.4, " s")') maxval(tpot) + maxval(tcontract)*(Nelectron + 2)*Nelectron/8
        !write(OUTPUT_UNIT, *) repeat('-', 52)
        print *, 'Done!'
    endif

    ! deallocate memory
    deallocate(pos)
    call rhogen%destroy()
    
#ifdef HAVE_MPI
    call mpi_finalize(ierr)
#endif
    stop
contains
    function new_test(pos, lmax, maxlevel, stepmax) result(Jpq)
        integer,     intent(in)             :: lmax, maxlevel
        real(REAL64),intent(in)             :: stepmax
        real(REAL64),allocatable, intent(in):: pos(:,:)
        real(REAL64)                        :: Jpq(2)
        
        ! potential and operator variables
        class(Function3D), allocatable      :: pot
        class(Coulomb3D), allocatable       :: V2
        
        ! generator variable
        type(SlaterGenerator)                :: rhogen
        
        ! The bounding box
        type(Grid3D)                        :: bounding_box
        real(REAL64)                        :: ranges(2,3)
        real(REAL64)                        :: rootlen(3)

        integer(int32), parameter :: nlip = 7
        integer(int32), parameter :: grid_type = 2
        
        class(Function3D), allocatable      :: rho
        
        ! initialize gaussian generator
        rhogen     = charges_and_positions_to_slater_generator(pos)

        bounding_box = rhogen%gen_cubegrid()
        ranges = bounding_box%get_range()
    
        rootlen = ranges(2,:) - ranges(1,:)
        ! initialize charge density
        allocate(rho, source = function3d_boxed_density(rhogen, maxlevel, ranges, stepmax, nlip, grid_type))
        ! initialize coulomb operator
        allocate(V2, source = GBFMMCoulomb3D(rho%grid, rho%bubbles, maxlevel, lmax))
        ! apply coulomb operator
        allocate(pot, source = V2 .apply. rho) 
        
        Jpq(1) = rho .dot. pot
        Jpq(2) = rhogen%selfint(bounding_box)
        call bounding_box%destroy()
        call rhogen%destroy()
        call rho%destroy()
    end function
    
    function new_test_gaussian(pos, lmax, maxlevel, stepmax) result(Jpq)
        integer,     intent(in)             :: lmax, maxlevel
        real(REAL64),intent(in)             :: stepmax
        real(REAL64),allocatable, intent(in):: pos(:,:)
        real(REAL64)                        :: Jpq(2)
        
        ! potential and operator variables
        class(Function3D), allocatable      :: pot
        class(Coulomb3D), allocatable       :: V2
        
        ! generator variable
        type(GaussGenerator)                :: rhogen
        
        ! The bounding box
        type(Grid3D)                        :: bounding_box
        real(REAL64)                        :: ranges(2,3)
        real(REAL64)                        :: rootlen(3)

        integer(int32), parameter :: nlip = 7
        integer(int32), parameter :: grid_type = 2
        
        
        class(Function3D), allocatable      :: rho
        
        ! initialize gaussian generator
        rhogen     = charges_and_positions_to_gaussian_generator(pos)
        
        bounding_box = rhogen%gen_cubegrid()
        ranges = bounding_box%get_range()
    
        rootlen = ranges(2,:) - ranges(1,:)
        ! initialize charge density
        allocate(rho, source = function3d_boxed_density(rhogen, maxlevel, ranges, stepmax, nlip, grid_type))
        ! initialize coulomb operator
        allocate(V2, source = GBFMMCoulomb3D(rho%grid, rho%bubbles, maxlevel, lmax))
        ! apply coulomb operator
        allocate(pot, source = V2 .apply. rho) 
        
        Jpq(1) = rho .dot. pot
        Jpq(2) = rhogen%selfint(bounding_box)
        call bounding_box%destroy()
        call rhogen%destroy()
        call rho%destroy()
    end function
    
    function new_test_gaussian_basic(pos, lmax, maxlevel, stepmax) result(Jpq)
        integer,     intent(in)             :: lmax, maxlevel
        real(REAL64),intent(in)             :: stepmax
        real(REAL64),allocatable, intent(in):: pos(:,:)
        real(REAL64)                        :: Jpq(2)
        
        ! potential and operator variables
        class(Function3D), allocatable      :: pot
        class(Coulomb3D), allocatable       :: V2
        
        ! generator variable
        type(GaussGenerator)                :: rhogen
        
        ! The bounding box
        type(Grid3D)                        :: bounding_box
        real(REAL64)                        :: ranges(2,3)
        real(REAL64)                        :: rootlen(3)

        integer(int32), parameter :: nlip = 7
        integer(int32), parameter :: grid_type = 2
        
        
        class(Function3D), allocatable      :: rho
        
        ! initialize gaussian generator
        rhogen     = charges_and_positions_to_gaussian_generator(pos)
        
        bounding_box = rhogen%gen_cubegrid()
        ranges = bounding_box%get_range()
    
        rootlen = ranges(2,:) - ranges(1,:)
        ! initialize charge density
        allocate(rho, source = function3d_boxed_density(rhogen, maxlevel, ranges, stepmax, nlip, grid_type))
        ! initialize coulomb operator
        allocate(V2, source = Coulomb3D(rho%grid, rho%bubbles))
        ! apply coulomb operator
        allocate(pot, source = V2 .apply. rho) 
        
        Jpq(1) = rho .dot. pot
        Jpq(2) = rhogen%selfint(bounding_box)
        call bounding_box%destroy()
        call rhogen%destroy()
        call rho%destroy()
    end function

    function new_test_gaussian_bubbles(pos, lmax, maxlevel, stepmax) result(Jpq)
        integer,     intent(in)             :: lmax, maxlevel
        real(REAL64),intent(in)             :: stepmax
        real(REAL64),allocatable, intent(in):: pos(:,:)
        real(REAL64)                        :: Jpq(2)
        
        ! potential and operator variables
        class(Function3D), allocatable      :: pot
        class(Coulomb3D), allocatable       :: V2
        
        ! generator variable
        type(GaussBubblesGenerator)         :: rhogen
        
        ! The bounding box
        type(Grid3D)                        :: bounding_box
        real(REAL64)                        :: ranges(2,3)
        real(REAL64)                        :: rootlen(3)

        integer(int32), parameter :: nlip = 7
        integer(int32), parameter :: grid_type = 2
        
        
        class(Function3D), allocatable      :: rho
        
        ! initialize gaussian generator
        rhogen     = charges_and_positions_to_gauss_bubbles_generator(pos)
    
        bounding_box = rhogen%gen_cubegrid()
        ranges = bounding_box%get_range()
    
        rootlen = ranges(2,:) - ranges(1,:)
        ! initialize charge density
        allocate(rho, source = function3d_boxed_density(rhogen, maxlevel, ranges, stepmax, nlip, grid_type))
        ! initialize coulomb operator
        allocate(V2, source = GBFMMCoulomb3D(rho%grid, rho%bubbles, maxlevel, lmax))
        ! apply coulomb operator
        allocate(pot, source = V2 .apply. rho)  
        
        Jpq(1) = rho .dot. pot
        print *, "Here bubbles", Jpq(1)
        Jpq(2) = rhogen%selfint(bounding_box)
        call bounding_box%destroy()
        call rhogen%destroy()
        call rho%destroy()
    end function
    
    function new_test_gaussian_bubbles_basic(pos, lmax, maxlevel, stepmax) result(Jpq)
        integer,     intent(in)             :: lmax, maxlevel
        real(REAL64),intent(in)             :: stepmax
        real(REAL64),allocatable, intent(in):: pos(:,:)
        real(REAL64)                        :: Jpq(2)
        
        ! potential and operator variables
        class(Function3D), allocatable      :: pot
        class(Coulomb3D), allocatable       :: V2
        
        ! generator variable
        type(GaussBubblesGenerator)         :: rhogen
        
        ! The bounding box
        type(Grid3D)                        :: bounding_box
        real(REAL64)                        :: ranges(2,3)
        real(REAL64)                        :: rootlen(3)

        integer(int32), parameter :: nlip = 7
        integer(int32), parameter :: grid_type = 2
        
        
        class(Function3D), allocatable      :: rho
        
        ! initialize gaussian generator
        rhogen     = charges_and_positions_to_gauss_bubbles_generator(pos)
        
        bounding_box = rhogen%gen_cubegrid()
        ranges = bounding_box%get_range()
    
        rootlen = ranges(2,:) - ranges(1,:)
        ! initialize charge density
        allocate(rho, source = function3d_boxed_density(rhogen, maxlevel, ranges, stepmax, nlip, grid_type))
        ! initialize coulomb operator
        allocate(V2, source = Coulomb3D(rho%grid, rho%bubbles))
        ! apply coulomb operator
        allocate(pot, source = V2 .apply. rho) 
        
        Jpq(1) = rho .dot. pot
        print *, "Here basic bubbles", Jpq(1)
        Jpq(2) = rhogen%selfint(bounding_box)
        call bounding_box%destroy()
        call rhogen%destroy()
        call rho%destroy()
    end function

    
    
    function charges_and_positions_to_gaussian_generator(pos) result(generator)
        real(REAL64),allocatable, intent(in):: pos(:,:)
        
        type(GaussGenerator)               :: generator
        
        generator     = GaussGenerator(q=pos(0,:),&
                             center=pos(1:3,:),&
                             alpha=[(alphagau, i=1, size(pos, 2))])
        
                
    end function
    
    function charges_and_positions_to_slater_generator(pos) result(generator)
        real(REAL64),allocatable, intent(in):: pos(:,:)
        integer, allocatable                :: charges(:), ls(:), ms(:), &
                                               centers(:, :), ibubs(:)
        integer                             :: iatom, lmax, s, l, &
                                               m, shell_number, i, &
                                               nelectron, charge, nshell
        integer, parameter                  :: NMAX = 10
        type(SlaterGenerator)               :: generator
        allocate(ls (NMAX*size(pos)) )
        allocate(ms (NMAX*size(pos)) )
        allocate(ibubs (NMAX*size(pos)) )
        nshell = 0
        do iatom = 1, size(pos, 2)
            nelectron = 0
            charge = pos(0, iatom)
            shell_number = get_shell_number(charge)
            lmax = get_lmax(shell_number)
            do l = 0, lmax
                do s = 1, 1
                    do m = -l, l
                        nelectron = nelectron + 2
                        nshell = nshell + 1
                        ibubs(nshell) = iatom
                        ls(nshell)    = l
                        ms(nshell)    = m
                        print *, iatom, l, m
                        if (nelectron >= charge) exit
                    end do
                    if (nelectron >= charge) exit
                end do
                if (nelectron >= charge) exit
            end do
        end do
        
        generator = SlaterGenerator(z      = pos(0, :),&
                           centers= pos(1 :, :), & 
                           ibub   = ibubs(:nshell), &
                           l      = ls(:nshell), &
                           m      = ms(:nshell), &
                           expos  = [(1.d0, i=1, size(pos, 2))], &
                           coeffs = [(1.d0, i=1, size(pos, 2))])
        deallocate(ls)
        deallocate(ms)
        deallocate(ibubs)
        
                
    end function
    
    function charges_and_positions_to_gauss_bubbles_generator(pos) result(generator)
        real(REAL64),allocatable, intent(in):: pos(:,:)
        integer, allocatable                :: charges(:), ls(:), ms(:), &
                                               centers(:, :), ibubs(:)
        integer                             :: iatom, lmax, s, l, &
                                               m, shell_number, i, &
                                               nelectron, charge, nshell, shell
        integer, parameter                  :: NMAX = 10
        type(GaussBubblesGenerator)         :: generator
        allocate(ls (NMAX*size(pos)) )
        allocate(ms (NMAX*size(pos)) )
        allocate(ibubs (NMAX*size(pos)) )
        nshell = 0
        do iatom = 1, size(pos, 2)
            nelectron = 0
            charge = pos(0, iatom)
            shell_number = get_shell_number(charge)
            
            
            do shell = 1, shell_number
                lmax = get_lmax(shell)
                do l = 0, lmax
                    do s = 1, 1
                        do m = -l, l
                            nelectron = nelectron + 2
                            nshell = nshell + 1
                            ibubs(nshell) = iatom
                            ls(nshell)    = l
                            ms(nshell)    = m
                            if (nelectron >= charge) exit
                        end do
                        if (nelectron >= charge) exit
                    end do
                    
                    if (nelectron >= charge) exit
                end do
                if (nelectron >= charge) exit
            end do
        end do
        
        generator = GaussBubblesGenerator(q      = pos(0, :),&
                           center = pos(1 :, :), & 
                           ibub   = ibubs(:nshell), &
                           l      = ls(:nshell), &
                           m      = ms(:nshell), &
                           alpha  = [(1.d0, i=1, size(pos, 2))], &
                           cutoff = 5.d0)
        deallocate(ls)
        deallocate(ms)
        deallocate(ibubs)
        
                
    end function
    
    pure function get_shell_number(charge) result(shell_number)
        integer, intent(in)      :: charge
        integer                  :: shell_number
        
        select case (charge)
            case (1 : 2)
                shell_number = 1
            case (3 : 10)
                shell_number = 2
            case (11 : 18)
                shell_number = 3
            case (19 : 36)
                shell_number = 4
            case (37 : 54)
                shell_number = 5
            case (55 : 86)
                shell_number = 6
            case (87 : 118)
                shell_number = 7
        end select
    end function
    
    pure function get_lmax(shell_number) result(lmax)
        integer, intent(in) :: shell_number
        integer             :: lmax
        
        select case (shell_number)
            case (1)
                lmax = 0
            case (2)
                lmax = 1
            case (3)
                lmax = 2
            case (4)
                lmax = 3
            case (5)
                lmax = 4
            case (6)
                lmax = 5
            case (7)
                lmax = 6
        end select
    end function
end program

