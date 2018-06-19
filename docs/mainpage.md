
Documentation of DAGE                         {#mainpage}
========================

Introduction                                     
------------
  
  DAGE is a Fortran/C++ (CUDA) code for electron structure calculations.
  
  Instead of solving Poisson's equation 

  \f[ \nabla^2 V = -4\pi\rho, \f]

  the potential is obtained by direct numerical integration from the
  Coulomb-law expression 

  \f[ V = \int_{\Omega} \frac{\rho(\vec{r}')}{\|\vec{r}-\vec{r}'\|}\;d^3r'. \f]

  The electrostatic potential energy *U* can then be calculated in
  suitable units from 

  \f[ U = \int_{\Omega} \rho(\vec{r})V(\vec{r})\;d^3r. \f]

Building 
--------

  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  $ mkdir build
  $ cd build
  $ cmake $PATH_TO_SOURCE
  $ ccmake $PATH_TO_SOURCE
  $ make bubbles # Compiles the library
  $ make doc # Produces this documentation with doxygen (>=1.8.1)
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
Using
-----

  DAGE can be used with two different approaches.
  1. Using XML-input files and running the calculation with command
     dage.x <input_file.xml>.
  2. Using Python-API.

Development
-----------

  - [Goals](@ref bubbles-goals)
  - [bubbles coding style](@ref bubbles-coding-style)
  - [Notes about writing documentation](@ref bubbles-writing-documentation)
    

References
----------

1. D. Sundholm, *The Journal of chemical physics*, 2005, **122**, 194107.
2. J. Jusélius and D. Sundholm, *The Journal of chemical physics*, 2007, **126**, 094101.
3. S. Losilla, D. Sundholm and J. Jusélius, *The Journal of Chemical Physics*, 2010, **132**, 24102.
4. S. Losilla and D. Sundholm, *The Journal of Chemical Physics*, 2012, **136**, 214104.
5. Parkkinen, P., Losilla S. A., Solala E., Toivanen E. A., Xu W., and D. Sundholm, *The Journal of Chemical Theory and Computation*, 2017,
6. Solala, E., Losilla S. A., Sundholm D., Xu W., and Parkkinen, P, *The Journal of Chemical Physics*, 2017, 

<!-- Additional pages --> 

@page Development Development

Development                              {#development}
===========

Goals                                    {#bubbles-goals}
-----

#### Fortran 2003 features ###

Our objects shall be constructed with the following syntax:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.f90}
! Declaration
type(OurObject) :: f 
...
! (Generic) construction
f = OurObject(args) 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our destructors shall be typed with the `final` keyword:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.f90}
type :: ourobject
    ...
    contains
    procedure :: my_method
    ...
    ! Calls `nullify()` and `deallocate()` on all relevant attributes
    ! of an object.
    final :: destructor 
end type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is to a) avoid boilerplate routines  (`init_zuul()` and suchlike)
and b) fight memory leaks. It shall be safe to deallocate an object
and be sure its contents (pointers, other types...) are deallocated as
well. 

Generic constructors are available starting from gfortran \>= 4.7. The
`final` keyword is currently not implemented in any official gfortran
release (but it's a part of F2003 standard nonetheless).

Coding style                             {#bubbles-coding-style}
------------------------

### Naming conventions ###

The purpose of using consistent naming scheme across bubbles is to increase
code readability and lessen the mental effort of understanding a given
code snippet. **NB**: Fortran is a case-insensitive language by nature
which means that consistency cannot be unfortunately enforced in a
portable manner. Also, Doxygen lists all Fortran code in lowercase so
the listings in our documetation may not reflect our naming conventions
exactly.

However, these rules adopted from [Style Guide for Python
code](http://www.python.org/dev/peps/pep-0008/) should be adhered to:

- Variables are always in lowercase.

- Parameters (constants) in UPPERCASE.

- Procedures shall always be written in lowercase with underscores
  substituting for spaces, e.g. `calc_spectrum().

- The names of types (classes) shall be
  typed in CamelCase, e.g. `CoulombOperator`.

- Type-bound procedures are the all lowercase, with their type name prepended
  (e.g. `MyClass%method()` is declared as `MyClass_method`).

  * The variable `self` shall only appear in procedures that are
    type-bound. They are always associated with the keyword `class`. Do
    not use `this` or `cls` or anything else to refer to the class
    instance.

  * NOTE: Doxygen doesn't get along with Fortran so well. In some cases, such as
    when declaring type-bound methods, the class name must be typed in
    lowercase, so that Doxygen links it to the correct subprogram.

- Constructors should be called `MyClass_init`. In case several constructors are
  available, a string should be appended after an underscore (e.g.
  `MyClass_init_from_ints`, `MyClass_init_from_real64`, `MyClass_init_default`).

  * The result variable should always be called new.

- The name of the module implementing `MyClass` shall be
  `MyClass_class`, e.g. `SlaterGenerator_class`. If multiple classes are
  implemented in the same module, the module name should reflect how the
  classes are related to each other, e.g. `Evaluators_class` could
  implement both `Integrator` and `Interpolator`.

- Modules that don't publicize types should instead by appended `_m` (for
  module).

### Module structure ###

- Everything should be `private`, except when explicitly declared as `public`.

- Between `implicit none` and `private` there should be only `public ::`
  statements. `public` should appear *nowhere else in the code*.

- Between `private` and `contains`, declare module variables, types and
  interfaces.

- Between `contains` and `end module`, functions and subroutine.


### Formatting ###

- Make sure your code does not exceed 80 characters per line.

- Use a 4-space indent.
  * Loops.
  * if-blocks.
  * Type components.
  * Interface generic procedures.
  * `select case` blocks.

- Align the code to make it more readable. Similar blocks of code should be
  aligned.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
allocate(new%cube(new%grid%axis(X_)%get_shape(),&
                  new%grid%axis(Y_)%get_shape(),&
                  new%grid%axis(Z_)%get_shape()))
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  * This is particularly important for variable declarations. Example:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
type(A),                    intent(in)  :: a
type(B),          optional, intent(in)  :: b
character(len=*), optional, intent(in)  :: c
integer,          optional, intent(out) :: d
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### Miscellaneous  ###
- When a procedure has optional arguments, use the keyword
  of the optional argument in order to make absolutely sure that
  arguments are correctly matched, e.g.  `my_routine(x, y, optarg=z)`
- Explicitly declare the intents (`in`, `out` or `inout`) of variables.
- Functions should be declared `pure` whenever possible.
- Type components which are themselves derived types with all components private
  are allowed to be public.

- TODO

### Code example ###
~~~~~~~~~~~~~~~~~~~~~
module MyClass_class
    use AnotherClass_class
    use SomeModule_m
    implicit none

    public :: MyClass

    private

    type :: MyClass
        private
        integer              :: a
        integer, allocatable :: b(:)
        real,    allocatable :: c(:,:)
        real,    pointer     :: d(:,:)
    contains
        ! Notice that myclass_method is lowercase for Doxygen to be able to
        ! produce the correct links.
        procedure :: method => myclass_method
    end type

    interface MyClass
        module procedure :: MyClass_init_from_int
        module procedure :: MyClass_init_from_real
    end interface
contains
    function MyClass_init_from_int(n) result(new)
        integer,      intent(in) :: n
        type(MyClass)            :: new
        ...
    end function

    function MyClass_init_from_real(r) result(new)
        real,         intent(in) :: r
        type(MyClass)            :: new
        ...
    end function

    subroutine MyClass_method(self)
        class(MyClass), intent(in) :: self
        ...
    end subroutine
end module
~~~~~~~~~~~~~~~~~~~~~
Writing documentation                 {#bubbles-writing-documentation}
---------------------

- bubbles uses
  [doxygen](http://www.stack.nl/~dimitri/doxygen/index.html) (\>= 1.8.1) to
  facilitate documentation.
- The primary goal is to document the public API (Application Programming
  Interface). This means that each module and all of its
  public functions/subroutines should have a docstring preceding the
  actual code.
- The secondary goal is to document bubbles internals, but this
  information should be hidden from end users.
- As bubbles API is experimental at this stage, extra attention should
  be paid to ensure that the documentation is up to date with actual
  code with minimal maintenance. In other words, do **not** overspecify!
- The best way to learn how to document your code is to look around in
  the source for already documented entities.
- However, here is code snippet that should give you an idea what is
  needed to document a subroutine.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.f90}
!\> Brief description.
!!
!! Notes about my_subroutine()... We can even use
!! LaTeX:
!!
!! \@f[ \nabla f = \partial_i f \hat{e}_i  \@f]
!!
subroutine my_subroutine(x, y)
    !\> Description of x
    integer, intent(in) :: x
    !\> Description of y
    integer, intent(out) :: y
    ...
end subroutine 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


<!-- vim: set ft=markdown tw=72 sw=2 smartindent: --> 
