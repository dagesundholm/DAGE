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
!> @file file_format.F90
!! Base class for all IO file formats

!> Formatted text files
!!
!! The IO routines available in `libdage` have been designed with
!! rank-three arrays (*cubes*) in mind. Any three-dimensional function that
!! is stored in a file can be imported to `libdage` with an instance of
!! a `FileFormat` class.
!!
!! A given IO routine shall read numeric data to the `cube(:,:,:)` attribute of
!! `FileFormat` and store additional information about the shape and
!! min/max values of the data to the `gdims(3)` and `ranges(2,3)`
!! attributes. 
!!
!! The IO routines shall write an associated `cube(:,:,:)` array to a file
!! with IO mode and format pertaining to the file format in question. Any
!! header data must be deducible from the `cube`, `ranges` and `gdims`
!! attributes.
!!
!! The IO routines shall be used in the [IO module](@ref io_m) where
!! `libdage` objects pass `FileFormat` class instances for reading and
!! writing. In other words, no `libdage` objects should call a specific IO
!! routine directly but always resort to the generic routines provided by
!! the IO module.
module file_format_class
    use globals_m
    implicit none

    type, public :: FileFormat
        integer                         :: gdims(3)
        real(REAL64)                    :: ranges(2, 3)
        real(REAL64), pointer           :: cube(:,:,:) => NULL()

        logical, private                :: io_status = .false.
    contains
        procedure                       :: destroy => fileformat_destroy
        procedure                       :: is_writable => fileformat_is_writable 
    end type

    !TODO type interface
    interface FileFormatInit
        module procedure fileformat_init
    end interface

    public :: FileFormatInit

    private 
contains

    !> File format constructor
    function fileformat_init(cube, gdims, ranges) result(f)
        !character(BUFLEN)               :: label
        real(REAL64), intent(in), pointer :: cube(:,:,:)
        integer,intent(in)              :: gdims(3)
        real(REAL64), intent(in)        :: ranges(2, 3)

        type(FileFormat)                :: f

        if (associated(cube)) then
            f = FileFormat(gdims=gdims, ranges=ranges, cube=cube)
            f%io_status = .true.
        else
            f%io_status = .false.
        endif

    end function

    !> Checks if the FileFormat instance is ready for writing
    function fileformat_is_writable(self) result(ok)
            class(FileFormat) :: self
            logical           :: ok
            ok = self%io_status
    end function

    !> Destroys a FileFormat object.
    subroutine fileformat_destroy(self)
            class(FileFormat) :: self
            deallocate(self%cube)
    end subroutine

end module 
