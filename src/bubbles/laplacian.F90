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
module Laplacian3D_class
    use Function3D_class 
    use Function3D_types_m
    use Grid_class 
    use globals_m
    use GaussQuad_class
    use Bubbles_class
    use timer_m
    use xmatrix_m
    use LIPBasis_class
    use mpi_m
    use Evaluators_class
    use ParallelInfo_class
    use expo_m
    use harmonic_class
#ifdef HAVE_OMP
    use omp_lib
#endif
    
    public :: Laplacian3D
    public :: assignment(=)
    
    !> Computes the Laplacian of Function3D in the output grid.
    type, extends(Operator3D) :: Laplacian3D
    contains
        procedure :: transform_bubbles     => laplacian3d_laplacian_bubbles
        procedure :: laplacian_bubbles_sub => Laplacian3D_laplacian_bubbles_sub
    end type

    interface assignment(=)
        module procedure :: Laplacian3D_assign
    end interface
    
    interface Laplacian3D
        module procedure :: Laplacian3D_init
    end interface
contains

    !> Constructor for Laplacian operator (\f$ \nabla^2 f\f$)
    !! \sa bubbles_laplacian
    function Laplacian3D_init(input_parallel_info, output_parallel_info, suboperator) result(new)
        !> parallelization info of the input function
        class(ParallelInfo), intent(in), target   :: input_parallel_info
        !> parallelization info of the output function
        class(ParallelInfo), intent(in), target   :: output_parallel_info
        !> If the resulting operator is a part of a parent operator
        logical,            intent(in), optional :: suboperator
        
        
        type(Laplacian3D)                           :: new


        call bigben%split("Laplacian operator initialization")

        ! if present, use the parallelization info given, in other cases
        ! the serialinfo
        new%result_parallelization_info => output_parallel_info
        new%gridin  => input_parallel_info%get_grid()
        new%gridout => output_parallel_info%get_grid()
        new%w       = [1.d0, 1.d0, 1.d0]
        new%result_type = F3D_TYPE_CUSP

        new%f=alloc_transformation_matrices(new%gridin, new%gridout, 3)

        new%f(X_)%p(:,:,1) = interp_matrix(new%gridin, new%gridout, X_, 2)
        new%f(X_)%p(:,:,2) = interp_matrix(new%gridin, new%gridout, X_, 0)
        new%f(X_)%p(:,:,3) = interp_matrix(new%gridin, new%gridout, X_, 0)

        new%f(Y_)%p(:,:,1) = interp_matrix(new%gridin, new%gridout, Y_, 0, 't')
        new%f(Y_)%p(:,:,2) = interp_matrix(new%gridin, new%gridout, Y_, 2, 't')
        new%f(Y_)%p(:,:,3) = interp_matrix(new%gridin, new%gridout, Y_, 0, 't')

        new%f(Z_)%p(:,:,1) = interp_matrix(new%gridin, new%gridout, Z_, 0, 't')
        new%f(Z_)%p(:,:,2) = interp_matrix(new%gridin, new%gridout, Z_, 0, 't')
        new%f(Z_)%p(:,:,3) = interp_matrix(new%gridin, new%gridout, Z_, 2, 't')

        new%suboperator = .FALSE.
        if (present(suboperator)) new%suboperator = suboperator
            
        
#ifdef HAVE_CUDA
        new%cuda_inited = .FALSE.
#endif

        call bigben%stop()
    end function

    
    function Laplacian3D_laplacian_bubbles(self, bubsin) result(new)
        class(Laplacian3D), intent(in)       :: self
        type(Bubbles), intent(in)            :: bubsin
        type(Bubbles), target                :: new

        new = Bubbles(bubsin,k=-2)
        call self%laplacian_bubbles_sub(bubsin, new)
    end function

    

    !> Compute the Laplacian of a set of bubbles.

    !> For \f$ \sum_{Alm}{g^A_{lm}(r_A)  Y^A_{lm} (\theta_A, \phi_A)}=
    !! \nabla^2\sum_{Alm}{f^A_{lm}(r_A)  Y^A_{lm} (\theta_A, \phi_A)} \f$,
    !! the expression for the output radial functions is given by
    !! \f[
    !!    g_{Alm}(r_A) = r_A^{-2} \left[
    !!    \frac{\mathrm{d}}{\mathrm{d}r_A} r_A^2
    !!    \frac{\mathrm{d}}{\mathrm{d}r_A} f_{Alm}(r_A)
    !!    -l(l+1) f_{Alm}(r_A) \right]
    !! \f]
    !! The factor \f$r_A^{-2} \f$ is considered by setting the output `k` to -2.
    subroutine Laplacian3D_laplacian_bubbles_sub(self, bubsin, new)
        class(Laplacian3D), intent(in)       :: self
        type(Bubbles), intent(in)            :: bubsin
        type(Bubbles), intent(inout), target :: new

        type(Interpolator1D)                 :: interpol
        real(REAL64), pointer                :: r(:), f_all_out(:, :), f_all_in(:, :)
        real(REAL64), allocatable            :: tmp(:, :, :), rpow(:)
        real(REAL64), allocatable, target    :: f_in(:, :)
        integer                              :: ibub, l, m, domain(2), first_bubble, last_bubble, &
                                                id, first_idx, last_idx, n, &
                                                first_l, first_m, last_l, last_m, l_m(2), num_threads, thread_num
        if(.not. (bubsin%get_nbub()>0)) return
        !if (bubsin%get_k()/=0) then
        !    write(ppbuf,'("bubbles_laplacian(): Not implemented for k/=0 (got k=",g0,")")'), bubsin%get_k()
        !    call perror(ppbuf)
        !end if
        ! get the bubble domain indices
        domain = self%result_parallelization_info%get_bubble_domain_indices &
                     (bubsin%get_nbub(), bubsin%get_lmax())
        first_bubble = (domain(1)-1) / (bubsin%get_lmax() + 1)**2 + 1
        last_bubble = (domain(2)-1) / (bubsin%get_lmax() + 1)**2 + 1  
        new = 0.0d0
        do ibub=first_bubble, last_bubble
            ! get the first idx, i.e., the last l, m pair calculated by this processor
            first_idx = 1
            if (ibub == first_bubble) then
                first_idx = domain(1) - (ibub-1) * (bubsin%get_lmax() + 1)**2
            end if
            l_m = lm(first_idx)
            first_l = l_m(1)
            
            ! get the last idx, i.e., the last l, m pair calculated by this processor
            last_idx = (bubsin%get_lmax() + 1)**2
            if (ibub == last_bubble) then
                last_idx = domain(2) - (ibub-1) * (bubsin%get_lmax() + 1)**2
            end if
            l_m = lm(last_idx)
            last_l = l_m(1)
            id = first_idx

            ! set the result pointer and the input pointer
            f_all_in  => bubsin%get_f(ibub)
            f_all_out => new%get_f(ibub)

            ! multiply the input function with r^k
            if (bubsin%get_k() /= 0) then
                rpow = bubsin%rpow(ibub, bubsin%get_k())
                allocate(f_in(size(f_all_in, 1), size(f_all_in, 2)))
#ifdef HAVE_OMP
                !$OMP PARALLEL DO
#endif 
                do ilm = 1, last_idx-first_idx+1
                    f_in(:, ilm) = f_all_in(:, ilm) * rpow1
                end do
#ifdef HAVE_OMP
                !$OMP END PARALLEL DO
#endif
                f_all_in => f_in
                deallocate(rpow)
            end if
            r=>new%gr(ibub)%p%get_coord()
            interpol=Interpolator1D(bubsin%gr(ibub)%p, 1, ignore_first = .TRUE.)
            allocate(tmp(size(r), 2, last_idx-first_idx+1))

            ! get the derivates for each l,m value at each grid point
            tmp(:, :, :) = interpol%eval(f_all_in(:, first_idx:last_idx), r)

            ! multiply the derivatives with r^2
#ifdef HAVE_OMP
            !$OMP PARALLEL DO
#endif 
            do ilm = 1, last_idx-first_idx+1
                f_all_out(:, ilm) = tmp(:, 2, ilm) * 2 * r
            end do
#ifdef HAVE_OMP
            !$OMP END PARALLEL DO
#endif
            

            ! get the derivative of r^2 * f'(r)
            tmp(:, :, :) = interpol%eval(tmp(:, 2, 1:last_idx-first_idx+1), r)
#ifdef HAVE_OMP
            !$OMP PARALLEL DO 
#endif 
            do l = first_l, last_l
                forall (n = max(l*l+1, first_idx) : min((l+1) * (l+1), last_idx))
                    f_all_out(:, n) = &
                          f_all_out(:, n) &
                        + r * r * tmp(:, 2, n-first_idx+1) & 
                        - l*(l+1) * f_all_in(:, n)
                end forall
            end do
#ifdef HAVE_OMP
            !$OMP END PARALLEL DO
#endif
            
            if (allocated(f_in)) deallocate(f_in)

            ! multiply the output function with r^k
            if (new%get_k() /= -2) then
                rpow = new%rpow(ibub, -2-new%get_k())
#ifdef HAVE_OMP
                !$OMP PARALLEL DO
#endif 
                do ilm = 1, last_idx-first_idx+1
                    f_all_out(:, ilm) = f_all_out(:, ilm) * rpow
                end do
#ifdef HAVE_OMP
                !$OMP END PARALLEL DO
#endif
                deallocate(rpow)
            end if

            ! the old version
            !do l=0,min(bubsin%get_lmax(), new%get_lmax() )
            !    do m=-l,l
            !        f_in=>bubsin%get_f(ibub,l,m)
            !        f_out=>new%get_f(ibub,l,m)
            !        ! -d/dr(r**2 * d/dr f(r) )
            !        tmp=interpol%eval(f_in,r_w)
            !        tmp=interpol%eval(r**2*tmp(:,2),r_w)
            !        f_out=tmp(:,2)
            !         f_out(2:)=f_out(2:)!*r(2:)**(-2)
            !        if(l>0) f_out=f_out-l*(l+1)*f_in
            !    end do
            !end do
            nullify(f_all_in, f_all_out)
            call interpol%destroy()
            nullify(r)
            deallocate(tmp)

        end do
        
    end subroutine
    
    subroutine Laplacian3D_assign(operator1, operator2)
        class(Laplacian3D), intent(inout), allocatable :: operator1
        class(Laplacian3D), intent(in)                 :: operator2

        allocate(operator1, source = operator2)
    end subroutine
end module