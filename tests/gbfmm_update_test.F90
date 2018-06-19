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
program potential_test_new
    use ISO_FORTRAN_ENV
    use Function3D_class
    use SCFCycle_class
    use Core_class
    use GaussQuad_class
#ifdef HAVE_MPI
    use mpi
#endif
    
    implicit none
  
    type(Core)                    :: core_object
    type(Function3D), allocatable :: mos(:), orbitals(:), gbfmm_orbitals(:), difference
    type(RHFCycle)                :: rhf_cycle, gbfmm_rhf_cycle
    character(100)                :: program_name = "gbfmm_update_test"
    integer                       :: i, j, ierr

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

    core_object = read_command_line_input(program_name)

    core_object%settings(1)%gbfmm = .TRUE.
    mos= core_object%get_input_orbitals()
   
    write(core_object%settings(1)%result_fd,*)
    write(core_object%settings(1)%result_fd,'(&
        &"========================================","'//new_line(" ")//'",&
        &"===== Function3D grid information =o====","'//new_line(" ")//'",&
        &a)') mos(1)%info()

    !gbfmm_rhf_cycle  = RHFCycle(mos, size(mos),  &
    !    quadrature = GaussQuad(NLIN=20, NLOG=16), use_gbfmm = core_object%settings(1)%gbfmm)

    !call gbfmm_rhf_cycle%run(update_orbitals = .TRUE.)
    !gbfmm_orbitals = gbfmm_rhf_cycle%orbitals
    
    !core_object%settings(1)%gbfmm = .FALSE.
   
    !rhf_cycle = RHFCycle(mos, size(mos),  &
    !    quadrature = GaussQuad(NLIN=20, NLOG=16), use_gbfmm = core_object%settings(1)%gbfmm)
    !call rhf_cycle%run(update_orbitals = .TRUE.)
    !orbitals = rhf_cycle%orbitals
    

    do i = 1, size(orbitals)
        difference = orbitals(i) - gbfmm_orbitals(i) 
        print *, "Maxval orbital", maxval(orbitals(i)%cube), maxloc(orbitals(i)%cube)
        print *, "Minval orbital", minval(orbitals(i)%cube), minloc(orbitals(i)%cube)
        print *, "Maxval gbfmm orbital", maxval(gbfmm_orbitals(i)%cube), maxloc(gbfmm_orbitals(i)%cube)
        print *, "Minval gbfmm orbital", minval(gbfmm_orbitals(i)%cube), minloc(gbfmm_orbitals(i)%cube)
        print *, "Orbital cube difference maxval: ", maxval(difference%cube), &
                                                       maxloc(difference%cube) 
        print *, "Orbital cube difference minval: ", minval(difference%cube), &
                                                       minloc(difference%cube) 
        print *, "Orbital cube difference relative maxval: ", maxval(difference%cube/orbitals(i)%cube), &
                                                       maxloc(difference%cube/orbitals(i)%cube) 
        print *, "Orbital cube difference relative minval: ", minval(difference%cube/orbitals(i)%cube), &
                                                       minloc(difference%cube/orbitals(i)%cube) 
        do j = 1, difference%bubbles%get_nbub()
            print *, "Bubble", j, "Orbital differenc maxval", &
                     maxval(difference%bubbles%bf(j)%p), maxloc(difference%bubbles%bf(j)%p)
            print *, "Bubble", j, "Orbital differenc maxval", &
                     minval(difference%bubbles%bf(j)%p), minloc(difference%bubbles%bf(j)%p)
        end do
    end do
    do i=1,size(mos)
        call mos(i)%destroy()
        call orbitals(i)%destroy()
        call gbfmm_orbitals(i)%destroy()
    end do
    deallocate(orbitals)
    deallocate(gbfmm_orbitals)
    deallocate(mos)
    call rhf_cycle%destroy()
    call gbfmm_rhf_cycle%destroy()
    
end program