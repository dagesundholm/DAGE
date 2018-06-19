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
!> @file cartiter.F90
!! Iterators over Cartesian indices
!> Iterators over Cartesian indices
module CartIter_class
    implicit none

    public :: CartIter
    public :: CartShellIter

    private

    !> Iterator over Cartesian indices.
    !!
    !! \c CartIter is an iterator that produces all multi-indeces
    !! \f$\boldsymbol{\kappa}=\{\kappa_1,\kappa_2,...,\kappa_{N_{dim}}\}\f$
    !! with number of dimensions \f$N_{dim}\f$ and maximum modulus \f$D\f$,
    !! such that \f$|\boldsymbol{\kappa}|=\sum_{i} \kappa_i \le D\f$.
    !! Multi-indeces are produced in lexicographical order, *i.e.*, a
    !! multi-index comes before another one if all first \f$k\f$ entries are
    !! smaller.
    !!
    !! ### Example ###
    !!
    !! For \f$N_{dim}=3\f$ and \f$D=2\f$, the
    !! following multi-indeces are produced, in this order:
    !! (0,0,0), (0,0,1), (0,0,2), (0,1,0), (0,1,1), (0,2,0), (1,0,0),
    !! (1,0,1), (1,1,0), (2,0,0)
    !!
    !! The code used to generate them:
    !! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.f90}
    !! program cartiter_test
    !!     use CartIter_class
    !!     implicit none
    !!     type(CartIter) :: iter
    !!     integer, parameter :: NDIM=3
    !!     integer :: kappa(NDIM)
    !!
    !!     iter=CartIter(ndim=NDIM,d=2)
    !!     do while(iter%next(kappa))
    !!         print*,kappa
    !!     end do
    !!     call iter%destroy()
    !!
    !! end program
    !! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    !! Output:
    !! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    !!     0           0           0
    !!     0           0           1
    !!     0           0           2
    !!     0           1           0
    !!     0           1           1
    !!     0           2           0
    !!     1           0           0
    !!     1           0           1
    !!     1           1           0
    !!     2           0           0
    !! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    !! \sa [CartShellIter] (@ref cartshelliter): Restrict multi-indeces to a given
    !! \f$|\boldsymbol{\kappa}|\f$ (one single shell)

    type :: CartIter
        private
        integer :: ndim
        integer :: maximum_modulus
        integer,allocatable :: kappa(:)
    contains
        !> Produce the next multi-index.
        procedure :: next => cartiter_next
        !> Destructor
        procedure :: destroy => cartiter_destroy
        !> Get the total number of multi-indeces that will be produced
        procedure :: get_num_iter => cartiter_get_num_iter
        !> Get the total number of multi-indeces that will be produced
        procedure :: get_possible_values => CartIter_get_possible_values
    end type

    !> Iterator over shells of Cartesian indeces
    !!
    !! \c CartIShellter is an iterator that produces all multi-indeces
    !! \f$\boldsymbol{\kappa}=\{\kappa_1,\kappa_2,...,\kappa_{N_{dim}}\}\f$
    !! with number of dimensions \f$N_{dim}\f$ and modulus \f$D\f$,
    !! such that \f$|\boldsymbol{\kappa}|=\sum_{i} \kappa_i = D\f$.
    !! Multi-indeces are produced in lexicographical order, *i.e.*,
    !! a multi-index comes before another one if all first \f$k\f$ entries are
    !! smaller.
    !!
    !! This iterator is identical to CartIter, but restricts the results
    !! to a given shell, such that \f$|\boldsymbol{\kappa}| = D\f$ instead of
    !! \f$|\boldsymbol{\kappa}| \le D\f$
    !!
    !! ### Example ###
    !!
    !! For \f$N_{dim}=3\f$ and \f$D=2\f$, the
    !! following multi-indeces are produced, in this order:
    !! (0,0,0), (0,0,1), (0,0,2), (0,1,0), (0,1,1), (0,2,0), (1,0,0),
    !! (1,0,1), (1,1,0), (2,0,0)
    !!
    !! The code used to generate them:
    !! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.f90}
    !! program cartiter_test
    !!     use CartIter_class
    !!     implicit none
    !!     type(CartShellIter) :: iter
    !!     integer, parameter :: NDIM=3
    !!     integer :: kappa(NDIM)
    !!
    !!     iter=CartShellIter(ndim=NDIM,d=2)
    !!     do while(iter%next(kappa))
    !!         print*,kappa
    !!     end do
    !!     call iter%destroy()
    !!
    !! end program
    !! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    !! Output:
    !! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    !!     0           0           2
    !!     0           1           1
    !!     0           2           0
    !!     1           0           1
    !!     1           1           0
    !!     2           0           0
    !! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    !! \sa [CartIter](@ref cartiter): Produce all multi-indeces up to a given
    !! \f$|\boldsymbol{\kappa}|\f$ (*i.e.*, include all shells)
    type, extends(CartIter) :: CartShellIter
        private
    contains
        procedure :: next         => cartshelliter_next
        procedure :: get_num_iter => cartshelliter_get_num_iter
    end type

    interface CartIter
        module procedure CartIter_init
    end interface

    interface CartShellIter
        module procedure CartShellIter_init
    end interface
contains

    ! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ! % CartIter                            %
    ! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    !> Constructor
    pure function CartIter_init(ndim, maximum_modulus) result(new)
        !> Number of dimensions
        integer,           intent(in) :: ndim
        !> Maximum modulus
        integer,           intent(in) :: maximum_modulus
        type(CartIter)            :: new

        new%ndim=ndim
        new%maximum_modulus=maximum_modulus
        allocate(new%kappa(new%ndim))

        ! This initialization is a trick to get [0,0,0] in the first iteration
        ! and flipdim=1
        new%kappa = maximum_modulus
        new%kappa(1)=-1

    end function

    !> Destructor
    pure subroutine CartIter_destroy(self)
        class(CartIter), intent(inout) :: self

        deallocate(self%kappa)
    end subroutine

    !> Produce the next multi-index. Return \c .FALSE. if no more iterations
    !! should be carried out
    pure subroutine CartIter_next(self, kappa, continue, flip)
        class(CartIter),   intent(inout) :: self
        !> Next multi-index
        integer,           intent(out)   :: kappa(self%ndim)
        !> Tells whether the iteration should be continued
        logical,           intent(out)   :: continue
        !> Dimension that changed in the last iteration
        integer, optional, intent(out)   :: flip
        
        
          
        integer :: idim


        
        continue=.TRUE.
        do idim=self%ndim,1,-1
            if(self%kappa(idim)<self%maximum_modulus .and. sum(self%kappa)<self%maximum_modulus) then
                self%kappa(idim)=self%kappa(idim)+1
                kappa=self%kappa
                if(present(flip)) flip=idim
                return
            else
                self%kappa(idim)=0
            end if
        end do
        continue=.FALSE.
    end subroutine

    pure function CartIter_get_num_iter(self) result(num_iter)
        class(CartIter), intent(in) :: self
        integer                     :: num_iter

        integer                     :: idim

        ! The number of iterations is given by
        ! \$f\frac{1}{N_{dim}!}\prod_{i=1}^{N_{dim}}d_{max}+i\f$
        num_iter=1
        do idim=1,self%ndim
            num_iter=num_iter*(self%maximum_modulus+idim)/idim
        end do
    end function

    subroutine CartIter_get_possible_values(self, values, changed_dimensions)
        class(CartIter), intent(inout) :: self
        integer, intent(out)           :: changed_dimensions(:), &
                                          values(:, :)
        integer                        :: i
        logical                        :: continue

        call self%next(values(:, 1), continue, changed_dimensions(1))
        i = 2
        do i = 2, self%get_num_iter()
            call self%next(values(:, i), continue, changed_dimensions(i))
        end do
    end subroutine

    ! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ! % CartShellIter                       %
    ! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    !>Constructor
    pure function CartShellIter_init(ndim,d) result(new)
        !> Number of dimensions
        integer,           intent(in) :: ndim
        !> Maximum modulus
        integer,           intent(in) :: d
        type(CartShellIter)           :: new

        new%CartIter=CartIter(ndim,d)
    end function

    !> Produce the next multi-index. Return \c .FALSE. if no more iterations
    !! should be carried out
    pure subroutine CartShellIter_next(self,kappa, continue, flip) 
        class(CartShellIter), intent(inout) :: self
        !> Next multi-index
        integer,              intent(out)   :: kappa(self%ndim)
        !> Tells whether the iteration should be continued
        logical, intent(out)                :: continue
        !> Dimension that changed in the last iteration
        integer, optional,    intent(out)   :: flip

        call self%CartIter%next(kappa, continue, flip)
        do while(sum(kappa)/=self%maximum_modulus)
            call self%CartIter%next(kappa,continue, flip)
        end do
        return
    end subroutine

    pure function CartShellIter_get_num_iter(self) result(num_iter)
        class(CartShellIter), intent(in) :: self
        integer                          :: num_iter

        ! The formula is the same, as if ndim was one unit smaller
        num_iter=self%CartIter%get_num_iter() * self%ndim/(self%maximum_modulus+self%ndim)
    end function

end module

