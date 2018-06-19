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
program main

    ! Functions
    use Grid_class
    use Function3D_class

    ! Generators
    use Generator_class
    use GaussGenerator_class
    use SlaterGenerator_class

    ! Density -> Potential
    use GaussQuad_class
    use Coulomb3D_class

    ! Numerical and analytic interaction energies
    use Analytical_m
    use Potential_class

    ! Utils
    use Globals_m
    use Timer_m
    use MPI_m

    ! Configuration file library
    use datadump_m ! Deprecated
    use getkw_class

    implicit none

    ! TODO: Clean up
    real(REAL32)                            :: etime
    real(REAL32)                            :: delta_t
    real(REAL32)                            :: times(2)
    integer(INT32)                          :: rank
    integer(INT32)                          :: ncpu
    integer(INT32)                          :: rank_t,&
                                               rank_d,&
                                               rank_p,&
                                               nslices_d,&
                                               nslices_p
    character(BUFLEN)                       :: dens_orig,&
                                               fname,&
                                               potential_name

    type(Function3D)                        :: rho,&
                                               apot
    class(Function3D), allocatable          :: potential

    integer(INT32)                          :: nlin,&
                                               nlog
    real(REAL64)                            :: tlog,&
                                               tend
    real(REAL64), pointer                   :: trange(:)
    type(GaussQuad)                         :: quadrature_params
    type(Coulomb3D)                         :: genpot
    real(REAL64)                            :: coul=0.d0,&
                                               coul_apot=0.d0,&
                                               coul_exact=0.d0,&
                                               nuc=0.d0,&
                                               nuc_exact=0.d0
    logical                                 :: test_p=.FALSE.,&
                                               testbub_p=.FALSE.,&
                                               new_grid_p,&
                                               analytical_pot_p,&
                                               selfint_an_p,&
                                               writetestdens_p

    character(LINELEN), dimension(:), pointer :: dns

    ! Timers
    real                                    :: t_start,&
                                               t_end,&
                                               t0,&
                                               t1,&
                                               dt_tot,&
                                               dt_init_dens,&
                                               dt_init_coulop,&
                                               dt_genpot,&
                                               dt_selfint,&
                                               dt_nuc

    ! End clean up

    class(Generator), allocatable           :: rhogen
    class(Generator), allocatable           :: potgen

    ! Opening input file
    open(INPUT_FD,file='GENPOT')

    call new_getkw(input, INPUT_FD)
    call initialize()

    t_start=x_time()

    !(MPI) Get number of nodes and mpi rank
    ncpu=xmpi_ncpu()
    rank=get_mpi_rank()

    t0=x_time()

! %%%%%%%%%%%%%%%%%%%%% DENSITY INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%
    ! Get origin of the density: grd file /plt file/test gaussian/etc
    dens_orig=''
    call getkw(input,'density.origin', dens_orig)
    call getkw(input,'bubbles', bubbles_p)
    call getkw(input,'nuclear_pot', nuclear_p)

    ! Establish flags for tests/input from file
    ! test_p: density initialized from parameter (BOTH Gaussians & Slaters)
    if(trim(dens_orig)=='test' .or. trim(dens_orig)=='bubbles_test') &
        test_p=.TRUE.
    ! testbub_p: Slaters
    if(trim(dens_orig)=='bubbles_test') &
        testbub_p=.TRUE.

    ! %%% Generate input rho from parameters (test)?
    if(test_p) then
        ! %%% orig = "bubbles_test" => Generate from Slaters
        if(testbub_p) then
            call getkw_ptr(input, 'density.slater', dns)
            allocate(rhogen, source=SlaterGenerator(dns))
        ! %%% orig = "test" => Generate from Gaussians
        else
            call getkw_ptr(input, 'density.gaussian', dns)
            allocate(rhogen, source=GaussGenerator(dns))
        end if
        ! Ignore the default grid generation and use user given parameters
        rho=rhogen%gen(grid=grid_from_kw('density'))
    ! Otherwise read from file
    else
        rho=Function3D(dens_orig)
    end if
    call rho%set_label('density')

    write(ppbuf,'(a,f12.7)') "Total charge: ",rho%integrate()
    call pprint(ppbuf)

    call getkw(input, 'density.writetestdens', writetestdens_p)
    if(master_p.and.writetestdens_p) then
        call rho%write('dens.cub')
        call rho%dump()
    end if

    t1=x_time()
    dt_init_dens=t1-t0

    ! %%%%%%%%%%%%%%%%%%%%% ANALYTICAL POTENTIAL %%%%%%%%%%%%%%%%%%%%%%%%%%%
    ! Initialize analytical potential if applicable
    call getkw(input, 'potential.analytical', analytical_pot_p)

    analytical_pot_p=test_p.and.analytical_pot_p

    if(analytical_pot_p) then
        call pinfo("Calculating analytical potential. This can be slow&
        & with PBCs.")
        select type (rhogen)
        type is (GaussGenerator)
            allocate(potgen, source=GaussPotGenerator(rhogen))
        type is (SlaterGenerator)
            allocate(potgen, source=SlaterPotGenerator(rhogen))
        end select
        apot=potgen%gen(rho%grid)
        call apot%set_label("analytical potential")
        call potential%set_label("xpotential")
    end if

    ! (MPI) Get the number of slicing parameters (number of slices and ranks,
    ! i.e., which slice will this node take.
    call get_nslices_and_ranks(ncpu,nslices_d,nslices_p,rank_t,rank_d,rank_p)

    ! Check if the potential grid is the same as the density grid
    call getkw(input, 'potential.new_grid', new_grid_p)

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! Create the potential grid:
    ! * copy,
    ! * initializing a new one explicitly, or
    ! * TODO: slice the density grid
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if(new_grid_p) then
        ! Explicit new grid given in input file
        call pinfo('Creating explicit new grid for the potential...')
        allocate(potential, source = Function3D(grid_from_kw('potential')))
    else if(rank_d/=rank_p.or.nslices_p/=nslices_d) then
        ! The global grids are the same, but NOT the local ones
        ! because we have different slices: read density grid again and slice
        call pinfo('Re-slicing the density grid for the potential...')
        call pinfo('Uh, not working yet!')
    else
        ! The local grids are exactly the same
        call pinfo('Reusing the density grid for the numerical potential...')
        allocate(potential, source = Function3D(rho))
    end if


    call potential%set_label('potential')

   ! !(MPI) Slice grids
   ! call set_grid_slice(grid,SL_COORD,nslices_d,rank_d)
   ! call slice_grid(grid,SL_COORD) !Slices grid in the X dimension
   ! if(associated(grid_pot, grid2)) then
   !     call set_grid_slice(grid2,SL_COORD,nslices_p,rank_p)
   !     call slice_grid(grid_pot,SL_COORD) !Slices grid in the X dimension
   ! end if

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! Electric potential
    ! The crucial part of the program!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    call getkw(input, 'integral.nlin', nlin)
    call getkw(input, 'integral.nlog', nlog)
    call getkw(input, 'integral.tlog', tlog)
    call getkw(input, 'integral.tend', tend)
    quadrature_params=GaussQuad(nlin,nlog,tlog,tend)

    call getkw(input,'spectrum',spectrum_p)
    if(spectrum_p) then
        call getkw(input, 'integral.trange', trange)
!        call calc_spectrum(density,potential,trange)
        call perror("SPECTRUM HAS BEEN DISABLED!!")
        stop
    else
        ! %%%%%%%%%%%%%%%%%%%%% INITIALIZE COULOMB OPERATOR %%%%%%%%%%%%%%%%%%
        t0=x_time()
        genpot=Coulomb3D(rho%grid, rho%bubbles, potential%grid, quadrature_params)
        t1=x_time()
        dt_init_coulop=t1-t0

        call quadrature_params%destroy()
        ! %%%%%%%%%%%%%%%% COMPUTE ELECTROSTATIC POTENTIAL %%%%%%%%%%%%%%%%%%%
        t0=x_time()
        allocate(potential, source = genpot. apply . rho)
        t1=x_time()
        dt_genpot=t1-t0
    end if

    if(bubbles_p .and. debug_g>0) then
        call potential%write('pot_nonsph.cub')
    end if

    call pprint('*** done with potential ***')
    delta_t=etime(times)
    write(ppbuf, '(a,f9.2,a)') '   wall time:', delta_t, 'sec'
    call pprint(ppbuf)
    write(ppbuf, '(a,f9.2,a)') '        user:', times(1), 'sec'
    call pprint(ppbuf)
    write(ppbuf, '(a,f9.2,a)') '         sys:', times(2), 'sec'
    call pprint(ppbuf)
    call pprint('*****************************')

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! Potential energy density and energies
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if (nslices_p>1) call setup_new_comm(io_comm,io_node_p)

    if (io_node_p) then

        ! Write the potential to disk
        fname=''
        call getkw(input, 'potential.file', fname)
        if (trim(fname) /= '' ) then
            call potential%write(fname)
            if (analytical_pot_p) then
                call apot%write('xpot.'//filext(fname))
            end if
        end if

        ! %%%%%%%%%%%%%%%% SELF-INTERACTION ENERGY %%%%%%%%%%%%%%%%%%%
        call getkw(input, 'selfint', selfint_p)
        if(selfint_p) then
            t0=x_time()
            coul=0.5d0*selfint_cheap(potential, rho)
            t1=x_time()
            dt_selfint=t1-t0
            !coul2=0.5d0*selfint(potential, density)
            if (analytical_pot_p) then
                call pinfo('Contracting analytical potential with density.')
                coul_apot=0.5d0*selfint_cheap(apot, rho)
                !coul2_apot=0.5d0*selfint(apot, density)
            endif
        end if

        ! %%%%%%%%%%%% NUCLEAR-ELECTRON INTERACTION ENERGY %%%%%%%%%%%%%%%%%%%
        if (nuclear_p) then
            t0=x_time()
            nuc = nucint(potential)
            t1=x_time()
            dt_nuc=t1-t0
        endif

        ! Analytical interaction energies
        call getkw(input, 'selfint_an', selfint_an_p)
        if (selfint_an_p) then

            call pinfo("Calculating analytical self-interaction energy...")
            select type (rhogen)
            type is (GaussGenerator)
                coul_exact=0.5d0*rhogen%selfint(rho%grid)
            type is (SlaterGenerator)
                coul_exact=0.5d0*rhogen%selfint(rho%grid)
                if (nuclear_p) then
                    nuc_exact = rhogen%nucint(rho%grid)
                endif
            end select
        endif

        call rhogen%destroy()
        call potgen%destroy()

        ! Output section
        call nl
        call nl

        if (selfint_p) then

            call pprint(repeat('-', 78))
            write (*,"(16X,A20,A20,A20)") "Self-Interaction/Ha",&
                                          "Abs. error/Ha",&
                                          "Rel. error"
            if (selfint_an_p) then
                write (*,"(A16,3E20.12)") "Numerical: ", coul,&
                        abs(coul-coul_exact),&
                        (coul-coul_exact)/coul_exact
            else
                write (*,"(A16,E20.12)") "Numerical: ", coul
            endif

            if (analytical_pot_p) then
                if (selfint_an_p) then
                    write (*,"(A16,3E20.12)") "Analytic pot.: ",&
                                            coul_apot,&
                                            abs((coul_apot-coul_exact)),&
                                            (coul_apot-coul_exact)/coul_exact
                else
                    write (*,"(A16,E20.12)") "Analytic pot.: ", coul_apot
                endif
            endif

            if (selfint_an_p) then
                write (*, "(A16,E20.12)") "Exact: ", coul_exact
            endif

            if (nuclear_p .and. bubbles_p) then

                call pprint(repeat('-', 78))
                write (*,"(16X,3A20)") "Nuclear interaction/Ha",&
                                              "Abs. error/Ha",&
                                              "Rel. error"
                if (selfint_an_p) then
                    write (*,"(A16,3E20.12)") "Numerical: ",&
                                            nuc,&
                                            abs((nuc-nuc_exact)),&
                                            (nuc-nuc_exact)/nuc_exact
                else
                    write (*,"(A16,E20.12)") "Numerical.: ", nuc
                endif

                if (selfint_an_p) then
                    write (*, "(A16,E20.12)") "Exact: ", nuc_exact
                endif

                call pprint(repeat('-', 78))
                write (*,"(16X,3A20)") "Total/Ha",&
                                              "Abs. error/Ha",&
                                              "Rel. error"

                write (*,"(A16,3E20.12)") "Numerical: ", nuc + coul, &
                                abs((nuc+coul-nuc_exact-coul_exact)),&
                                (nuc+coul-nuc_exact-coul_exact)/(nuc_exact+coul_exact)

                if (selfint_an_p) then
                    write (*, "(A16,E20.12)") "Exact: ", coul_exact + nuc_exact
                endif

            endif


        endif

        call pprint(repeat('-', 78))


    end if ! IO node

   ! delete temp files
   ! if (.not.direct_p) then
   !     call gen_mpi_fname(DDEN_FN, ppbuf)
   !     call unlink(trim(ppbuf))
   !     call gen_mpi_fname(DPOT_FN, ppbuf)
   !     call unlink(trim(ppbuf))
   ! end if

    t_end=x_time()
    dt_tot=t_end-t_start

    call getkw(input,'potential.save', potential_name)

    if (trim(potential_name).NE.'') then
        call potential%dump(trim(potential_name))
    end if

    call potential%destroy()
    call rho%destroy()
    call apot%destroy()
    call finalize()
    call del_getkw(input)

    if(master_p) then
        write(ppbuf, "(A, F10.5)")  'Total execution time (mpi_time):', t_end-t_start
        call pinfo(ppbuf)
    end if

    call report_time('TOTAL', dt_tot)
    call report_time('Density initialization',   dt_init_dens,   dt_tot)
    call report_time('Coulomb operator',         dt_init_coulop, dt_tot)
    call report_time('Electrostatic potential',  dt_genpot     , dt_tot)
    call report_time('Self-repulsion energy',    dt_selfint    , dt_tot)
    call report_time('Nuclear interaction',      dt_nuc        , dt_tot)

    if(master_p) then
        call pprint(repeat('-', 78))
        call stockas_klocka()
        call pprint(repeat('-', 78))
        call nl
        call program_footer()
    end if

    close(INPUT_FD)

contains

    subroutine initialize()

        integer(INT32) :: hostnm, rank, ierr
        character(BUFLEN) :: title, fdate, sys

        ! Get 'mpirun' keyword and set up parallel environment (start_mpi() function)
        ! if this is a parallel run.
        call getkw(input, 'mpirun', mpirun_p)
        if (mpirun_p) then
            rank=start_mpi()
        else
            master_p=.true.
            rank=0
        end if

        ! Get 'debug' keyword and set up debugging framework
        call getkw(input, 'debug', debug_g)
        call set_debug_level(debug_g)
        if (debug_g > 0) then
            call genfile('debug_file', fname)
            open(DEBUG_FD, file=trim(fname))
            call set_pplun(DEBUG_FD, 'debug')
        end if

        ! Get verbosity level
        call getkw(input, 'verbosity', verbo_g)

        ierr=hostnm(sys)
        ! Master process gets 'title' keyword and prints header. Random seed is initialized
        if (master_p) then
            call pprint(fdate())
            call program_header

            call getkw(input, 'title', title)
            call pprint(' TITLE: '// trim(title))
            call nl
        ! Slave processes get 'outfile' keyword.
        else
            fname=''
            call getkw(input, 'outfile', fname)
            if (trim(fname) /= '') then
                call genfile('outfile', fname)
                open(OUTPUT_FD, file=trim(fname))
                call set_pplun(OUTPUT_FD)
            else
                call set_pplun(DEVNULL)
            end if
        end if
        write(ppbuf, '(a,i3)') 'Debug level is ', debug_g
        call pinfo(ppbuf)

        if (mpirun_p) then
            write(ppbuf, '(3a,i3)') &
                'Initialized MPI on ', trim(sys),' rank ', rank
            call pdebug(ppbuf,2)
        end if

    end subroutine

    subroutine finalize()
        if (debug_g > 0) close(DEBUG_FD)
        if (.not.master_p) then
            call getkw(input, 'outfile', fname)
            if (trim(fname) /= '') close(OUTPUT_FD)
        end if
        call stop_mpi()
    end subroutine

    subroutine program_header
        integer(INT32) :: i,j,sz
        integer(INT32), dimension(3) :: iti
        integer(INT32), dimension(:), allocatable :: seed

        character(*), dimension(13), parameter :: head=(/ &
'    **************************************************************************', &
'    ***                                                                    ***', &
'    ***                            DAGE                                    ***', &
'    ***                        Dage Sundholm                               ***', &
'    ***                        Jonas Juselius                              ***', &
'    ***                        Sergio Losilla                              ***', &
'    ***                        Elias Toivanen                              ***', &
'    ***                            Dou Du                                  ***', &
'    ***                                                                    ***', &
'    ***      This software is copyright (c) 2005-2012 by Dage Sundholm     ***', &
'    ***                    University of Helsinki.                         ***', &
'    ***                                                                    ***', &
'    **************************************************************************' /)


        call random_seed(size=sz)
        allocate(seed(sz))
        call random_seed(get=seed)
        call itime(iti)
        j=sum(iti)
        do i=1,sz
            seed(i)=seed(i)*j
        end do
        call random_seed(put=seed)
        deallocate(seed)

        do i=1,size(head)
            call pprint(head(i))
        end do

        call pprint("    Revision: @BUBBLES_REVISION_NUMBER@")
        call nl
    end subroutine

    subroutine program_footer
        real(REAL64) :: rnd
        character(*), dimension(4), parameter :: raboof=(/ &
            'DAGE - Direct Approach to Gravitation and Electrostatics', &
            'DIPP - Direct Integration of the Poisson Problem        ', &
            'DAGE - DAGE is A Goat Eater                             ', &
            'genpot - The advanced Hello World program               '/)

        call random_number(rnd)
        call nl
        call pprint(raboof(nint(rnd*2.d0)+1))
        call nl
    end subroutine

    subroutine get_nslices_and_ranks(ncpu,nslices_d,nslices_p,rank_t,rank_d,rank_p)
        integer(INT32),intent(in) :: ncpu
        integer(INT32),intent(out) :: nslices_d,nslices_p,rank_t,rank_d,rank_p
        integer(INT32) :: ntpoints, nslices_tot, tmp
    !   Calculate slices
        call getkw(input, 'integral.nlin', ntpoints)
        call getkw(input, 'integral.nlog', tmp)
        ntpoints=ntpoints+tmp

        !Calculate number total number of slices
        if(ncpu>ntpoints) then
            if(mod(ncpu,ntpoints)/=0) stop 'The number of nodes should be an exact multiple&
            & of the number of t-points! Aborting...'
            nslices_tot=ncpu/ntpoints
        else
          nslices_tot=1
        end if

        ! Calculate nslices_d and nslices_p
        ! If potential.nslices is given in input, then use that, otherwise,
        ! ~equipartitioning

        call getkw(input, 'potential.nslices', nslices_p)
        if(nslices_p>0) then
            if (mod(nslices_tot,nslices_p)/=0) then
                call pinfo('Number of slices for the potential grid not reasonable. Aborting...')
                stop
            end if
            nslices_d=nslices_tot/nslices_p
        else
            nslices_d=int(sqrt(real(nslices_tot)))
            nslices_p=nslices_tot/nslices_d
        end if

        !Get ranks
        !They are given in the following way:
        !Example: 3 density slices, 2 potential slices (that is 60*2*3=360 processes)
        ! mpi rank: 0 1 2 3 4 5		 6 7 8 9 10 11	 12 ... 59	 60 61 62 ... 354 355 356 357 358 359
        !   t rank: 1 1 1 1 1 1		 2 2 2 2  2  2	  3 ... 10	 11 11 11 ...  60  60  60  60  60  60
        !   d rank: 1 2 3 1 2 3		 1 2 3 1  2  3	  1 ...  3	  1  2  3 ...   1   2   3   1   2   3
        !   p rank: 1 1 1 2 2 2		 1 1 1 2  2  2	  1 ...  2	  1  1  1 ...   1   1   1   2   2   2
        !           A	  A
        ! I/O nodes_|_____|

        rank_d=mod(mod(rank,nslices_tot),nslices_d)+1
        rank_p=mod(rank,nslices_tot)/nslices_d+1
        rank_t=rank/nslices_tot+1
        if(rank_t==1.and.rank_d==1) io_node_p=.true.

    !nslices_p=2
    !rank_p=rank_t
    !    if((rank_t)<3) io_node_p=.true.

    !if(master_p) print*,'rank_t,rank_d,rank_p,nslices_tot,nslices_d,nslices_p'
    !    print*,rank_t,rank_d,rank_p,nslices_tot,nslices_d,nslices_p
        end subroutine

    function grid_from_kw(func_kw) result(grid)
        character(len=*), intent(in) :: func_kw
        type(Grid3D) :: grid

        integer(INT32), dimension(3) :: gdims, ncell
        real(REAL64), dimension(2) :: xrange, yrange, zrange
        real(REAL64), dimension(3) :: tight
        integer(INT32) :: nlip
        character(BUFLEN) :: pbc_string
        real(REAL64) :: ranges(2,3)

        call getkw(input, 'lip_points', nlip)
        call getkw(input, trim(func_kw)//'.ncells', ncell)
        call getkw(input, trim(func_kw)//'.xrange', xrange)
        call getkw(input, trim(func_kw)//'.yrange', yrange)
        call getkw(input, trim(func_kw)//'.zrange', zrange)

        gdims=ncell*(nlip-1)+1

        ! Here the MPI corrections (slicing) should come
        tight(1)=(xrange(2)-xrange(1))/dble(ncell(1)*(nlip-1))
        tight(2)=(yrange(2)-yrange(1))/dble(ncell(2)*(nlip-1))
        tight(3)=(zrange(2)-zrange(1))/dble(ncell(3)*(nlip-1))

        call getkw(input, trim(func_kw)//'.pbc', pbc_string)

        ranges(:,1) = xrange
        ranges(:,2) = yrange
        ranges(:,3) = zrange

        grid = Grid3D(ranges, gdims, nlip=nlip, pbc_string=pbc_string)

        return
    end function

end program

