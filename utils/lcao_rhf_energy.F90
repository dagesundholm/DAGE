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
!> @file lcao_rhf_energy.F90
!! Compute the total energy of a LCAO RHF wavefunction.

! Usage: lcao_rhf_energy input_file [-s|--step [CUBE GRID STEP]] [-r|--radius [BUBBLE RADIUS]] [-d|--dump]
program lcao_rhf_energy
    use ISO_FORTRAN_ENV
#ifdef HAVE_MPI
    use mpi
#endif
    use LCAO_m
    use DIIS_class
    use Function3D_class
    use Function3D_types_m, only: F3D_TYPE_CUSP
    use Globals_m
    use SCFCycle_class
    use Bubbles_class
    use Potential_class, only: nuclear_repulsion
    use GaussQuad_class, only: GaussQuad
    use Core_class
    implicit none

    
    integer                       :: nargs
    character(100)                :: arg

    character(100)                :: input_file
    real(REAL64)                  :: step     =0.2d0
    real(REAL64)                  :: radius   =8.d0
    logical                       :: fdump    =.FALSE.
    logical                       :: ftime    =.FALSE.
    logical                       :: gbfmm    =.FALSE.
    integer                       :: result_fd=OUTPUT_UNIT
    character(100)                :: output_name="", program_name = ""

    integer                       :: timing_fd=97
    character(100)                :: output_filename
    character(100)                :: timing_filename

    type(Function3D), allocatable :: mos(:), orbital_potentials(:)
    type(Function3D)              :: dens, temp, temp2, mould

    real(REAL64)                  :: num_el
    real(REAL64)                  :: energy
    real(REAL64)                  :: nuc_rep_energy

    class(AOBasis), allocatable   :: basis_set
    type(Structure)               :: molec
    type(RHFCycle)                :: rhf_cycle
    type(LCMORHFCycle)            :: lcmo_rhf_cycle
    type(LCMODIIS)                :: lcmo_diis
    type(HelmholtzDIIS)           :: helmholtz_diis
    type(Bubbles)                 :: test_bubbles
    class(Helmholtz3D), allocatable :: helmholtz_operator
    class(Laplacian3D), allocatable :: laplacian_operator
    class(Coulomb3D),   allocatable :: coulomb_operator
    type(GaussQuad)               :: quadrature

    integer                       :: ntot
    integer                       :: nocc
    real(REAL64), allocatable     :: mocoeffs(:,:)
    type(Core)                    :: core_object


    integer                       :: i,j,ierr

    character, parameter :: LB=new_line(" ")

#ifdef HAVE_MPI
    call mpi_init(ierr)
    ! get number of processors and store it to nproc
    call mpi_comm_size(MPI_COMM_WORLD, nproc, ierr)
    ! get rank of the processor (index in range 0-nproc) and
    ! store it to iproc
    call mpi_comm_rank(MPI_COMM_WORLD, iproc, ierr)
#else 
    iproc = 0
    nproc = 1
#endif
   program_name = "lcao_rhf_energy"
   core_object = read_command_line_input(program_name)

! %%%%%%%%%%%%%% Print header %%%%%%%%%%%

    !if(iproc == 0) then
    !    write(result_fd,'(&
    !        &"========================================","'//LB//'",&
    !        &"========== lcao_rhf_energy.x ===========","'//LB//'",&
    !        &"========================================","'//LB//'",&
    !        &"  Input file:  ",a                       ,"'//LB//'",&
    !        &"        Step:  ",f12.3                   ,"'//LB//'",&
    !        &"      Radius:  ",f12.3                   ,"'//LB//'",&
    !        &"========================================")') &
    !        trim(input_file), step, radius
    !end if

! %%%%%%%%%%%%%% Start main timer %%%%%%%%%%%%
    bigben=Chrono("LCAO RHF")

! %%%%%%%%%%%%%% Construct MOs %%%%%%%%%%%
    mos= core_object%get_input_orbitals()

! %%%%%%%%%%%%%%%%%% Dump MOs %%%%%%%%%%%%%%%%%%%%%
    if(.FALSE.) then
        call bigben%split("Write MOs")
        do i=1,nocc
            call mos(i)%dump()
        end do
        call bigben%stop()
    end if

! %%%%%%%%%%%%%% Print grid information %%%%%%%%%%%
    if(iproc == 0) then
        write(result_fd,*)
        write(result_fd,'(&
            &"========================================","'//LB//'",&
            &"===== Function3D grid information =o====","'//LB//'",&
            &a)') mos(1)%info()
    end if

! %%%%%%%%% Report number of electrons %%%%%%%%%%%%
    !call bigben%split("Integrate density")
    dens=Function3D(mos(1),type=F3D_TYPE_CUSP)
    dens=0.d0
    do i=1,size(mos)
        temp = mos(i) / sqrt(mos(i).dot.mos(i))
        call mos(i)%destroy()
        mos(i) = temp
        call temp%destroy()
        temp = mos(i) * mos(i)
        temp2 = 2.0d0 * temp
        call temp%destroy()
        temp=dens + temp2
        dens = temp
        call temp%destroy()
        call temp2%destroy()
        allocate(mos(i)%taylor_series_bubbles, source = mos(i)%get_taylor_series_bubbles())
    end do
    !print *, "Number of electrons", dens%integrate()
    !call dens%set_label("density")
    if(fdump) call dens%dump()
    num_el=dens%integrate()

    !print *, "number of electrons", num_el
    call dens%destroy()
    !call bigben%stop()

! %%%%%%%%%%% Nuclear repulsion energy %%%%%%%%%%%
    nuc_rep_energy=nuclear_repulsion(mos(1)%bubbles%get_z(), &
                                     mos(1)%bubbles%get_centers())

! %%%%%%%%%%%%%%%%% SCF cycle %%%%%%%%%%%%%%%%%%%%

    !print *, "number of orbitals", size(mos)
    !print *, "orbitals dottie", mos(1) .dot. mos(1)
    !quadrature = GaussQuad(NLIN=20, NLOG=16)
    !if (core_object%settings(1)%gbfmm) then
    !    allocate(laplacian_operator, source = )
    !rhf_cycle = RHFCycle(mos, size(mos),  &
    !    quadrature = GaussQuad(NLIN=20, NLOG=16), use_gbfmm = core_object%settings(1)%gbfmm)
    !helmholtz_diis = HelmholtzDIIS(rhf_cycle)
    !helmholtz_diis%nuc_rep_energy = nuc_rep_energy
    !do i = 1, 30
    !    call bigben%split("SCF loop")
    !    call rhf_cycle%run()
    !    write(result_fd, '("-----------------------------------------------------------------------")')
    !    write(result_fd, '("Iteration ",i4," completed")'), i
    !    write(result_fd, '("Total energy:      ", f24.16,"")'), rhf_cycle%energy + nuc_rep_energy
    !    write(result_fd, '("Electronic energy: ", f24.16,"")'), rhf_cycle%energy
    !    write(result_fd, '("Nuclear repulsion: ", f24.16,"")'), nuc_rep_energy
    !    write(result_fd, '("Energy change:     ", f24.16,"")'),  rhf_cycle%energy - energy
    !    write(result_fd, '("-----------------------------------------------------------------------")')
    !    call bigben%stop_and_print()
    !    energy = rhf_cycle%energy
    !end do
    !lcmo_rhf_cycle = LCMORHFCycle(rhf_cycle)
    !mos = helmholtz_diis%optimize()
    !call helmholtz_diis%destroy()
    !call rhf_cycle%destroy()
    
    !lcmo_diis = LCMODIIS(lcmo_rhf_cycle)
    !mos =  lcmo_diis%optimize()
    


! %%%%%%%%%%%%%%%%%%%% Report %%%%%%%%%%%%%%%%%%%% 
    if(iproc == -1) then
        write(result_fd,*)
        write(result_fd,'(&
            &"========================================","'//LB//'",&
            &"================ RESULTS ===============","'//LB//'",&
            &"========================================","'//LB//'",&
            &" Num. of electrons: ",f20.16             ,"'//LB//'",&
            &"       TOTAL RHF ENERGY"                 ,"'//LB//'",&
            &" Electronic:    ",f24.16                 ,"'//LB//'",&
            &" Nuc. repulsion:",f24.16                 ,"'//LB//'",&
            &" Total:         ",f24.16                 ,"'//LB//'",&
            &"========================================")') &
            num_el, rhf_cycle%energy, nuc_rep_energy, rhf_cycle%energy+nuc_rep_energy
    end if
    call bigben%stop()

!%%%%%%%%% REPORT TIME %%%%%%%%%%%%%
    ! Brief time report
    if(iproc == -1) then
        call pprint(repeat("~",70))
        call pprint("            -=-=-=-=-=-=-=- Time breakdown -=-=-=-=-=-=-=-")
        call bigben%print(maxdepth=2)
        call pprint(repeat("~",70))
        ! If timings are requested, produce a full timing report
        if(ftime) then
            ! Either to a file, if -o option given
            if(trim(output_name)/="") then
                timing_filename=trim(output_name)//".time"
                open(timing_fd, file=timing_filename)
                call pinfo(&
                    "Timing report will be written to "//trim(timing_filename))
        ! Otherwise to standard output
            else
                timing_fd=OUTPUT_UNIT
            end if
            call bigben%dump(timing_fd)
            close(timing_fd)
        end if
    end if
    call bigben%destroy()

    do i=1,size(mos)
        call mos(i)%destroy()
    end do
    deallocate(mos)
    
#ifdef HAVE_MPI
    call mpi_finalize(ierr)
#endif

end program
