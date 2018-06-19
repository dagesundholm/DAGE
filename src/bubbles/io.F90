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
!> @file io.F90
!! Frontend to various IO text formats

!> Unified frontend for reading and writing numerical data
!! 
!! This module provides routines for reading and writing `libdage` objects
!! via intermediate `FileFormat` objects. The basic idea of an `FileFormat`
!! object is to act as a package sent from higher-level parts of the code
!! to this module where the data contained in a `FileFormat` instance is
!! passed to actual IO routines. The name of the file that is read from or
!! written to must also be specified, and the extension of the filename is
!! used to determine the appropriate file format.
!!
! ### Example
!  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.f90}
!  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
module io_m
    use globals_m
    use file_format_class
    use cubeplot_m
    use gopenmol_m
    implicit none

    ! Broadcast
    public :: FileFormat
    public :: FileFormatInit

    ! IO routines
    public :: read_file
    public :: write_file

    private

contains

    !> Reads data from a formatted text file.
    !!
    !! @returns `.true.` if reading succeeded.
    function read_file(filename, f) result(ok)
            character(len=*), intent(in)   :: filename
            type(FileFormat), intent(inout), allocatable :: f

            logical                        :: ok

            character(len=len(filename)) :: fext

            inquire(file=trim(filename), exist=ok)

            if(.not.ok) then
                call perror("File `"//trim(filename)//"' could not be found!")
                return
            end if

            ! Doing a re-read
            if (allocated(f) .and. f%is_writable()) then
                call f%destroy()
                deallocate(f)
            endif

            fext = filext(filename)
            select case(fext)
            case('cub')
                allocate(f)
                ok = read_cubeplot(filename, f)
            case('plt')
                allocate(f)
                ok = read_gopenmol(filename, f)
            case default
                ok = .false.
                call perror("File type with &
                        &extension"//trim(fext)//" is not supported!")
                return
            end select
    end function

    !> Writes data to a formatted text file.
    !!
    !! @returns `.true.` if writing succeeded.
    function write_file(filename, f) result(ok)
            character(len=*), intent(in) :: filename
            type(FileFormat), intent(inout) :: f

            logical                        :: ok

            character(len=:), allocatable  :: fext

            ok = .false.

            if (f%is_writable()) then
                fext = filext(filename)
                select case(fext)
                case('cub')
                    call write_cubeplot(filename, f)
                case('plt')
                    call write_gopenmol(filename, f)
                case default
                    ok = .false.
                    call perror("File type with extension"//trim(fext)//" is not supported!")
                    return
                end select
                ok = .true.
            else
                ok = .false.
            endif

    end function
end module
