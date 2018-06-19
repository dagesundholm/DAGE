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
module Projector3D_class
    use Operators_class 
    use Grid_class 
    use globals_m
    use GaussQuad_class
    use Bubbles_class
    use timer_m
    use xmatrix_m
    use LIPBasis_class
    use mpi_m
    use Evaluators_class
    use expo_m

    public :: Projector3D
    public :: cube_project

    !> Projection operator

    !> Projects the Function3D into the output grid. The Bubbles are copied.
    type, extends(Operator3D) :: Projector3D
    contains
        procedure :: transform_bubbles => Projector3D_copy_bubbles
        procedure :: transform_cube    => Projector3D_transform_cube
    end type

    interface Projector3D
        module procedure :: Projector3D_init
    end interface

    integer, parameter :: IN_ =1
    integer, parameter :: OUT_=2

contains

    !> Bubbles transformation operation for `Projector3D`: return a copy
    !! of bubsin.
    function Projector3D_copy_bubbles(self, bubsin) result(new)
        class(Projector3D), intent(in) :: self
        type(Bubbles), intent(in)     :: bubsin
        type(Bubbles)                 :: new

        new=bubsin
        return
    end function

    !> Constructor for projector operator
    function Projector3D_init(gridin, gridout) result(new)
        type(Projector3D)                           :: new
        !> Grid of input functions
        type(Grid3D), intent(in), target           :: gridin
        !> Grid on which the output functions are constructed. If not given,
        !! construct on the input grid
        type(Grid3D), intent(in), target, optional :: gridout

        type(Grid3D), pointer              :: gridout_p

        if(present(gridout)) then
            gridout_p=>gridout
        else
            gridout_p=>gridin
        end if

        new%gridin  = gridin
        new%gridout = gridout_p
        new%w       = [1.d0]

        new%f=alloc_transformation_matrices(new%gridin, new%gridout, 1)

        new%f(X_)%p(:,:,1) = interp_matrix(new%gridin, new%gridout, X_)
        new%f(Y_)%p(:,:,1) = interp_matrix(new%gridin, new%gridout, Y_, trsp='t')
        new%f(Z_)%p(:,:,1) = interp_matrix(new%gridin, new%gridout, Z_, trsp='t')

    end function

    ! Wrapper(s) to operator calls

    !> Projects the cube of a function to the grid of another function.
    !!
    !! The `res` function is identical to source except for its cube. In
    !! case, `source` and `mould` share the same grid, the cube of `res`
    !! is a fresh copy of source.
    function cube_project(cubein, gridin, gridout) result(cubeout)
        !> Input cube
        real(REAL64), intent(in)   :: cubein(:,:,:)
        !> Grid of `cubein`
        class(Grid3D), intent(in)  :: gridin
        !> Grid onto which `cubein` is to be projected
        class(Grid3D), intent(in)  :: gridout

        real(REAL64), allocatable  :: cubeout(:,:,:)

        type(Projector3D)           :: projector

        call pdebug("function_project()", 1)

        if(have_same_grid(gridin, gridout)) then
            ! Source and mould share the same grid
            ! Copy the cube of source
            cubeout=cubein
        else
            projector = Projector3D(gridin, gridout)
            cubeout= projector%transform_cube( cubein )
            call projector%destroy()
        end if
    end function


    !> Transform cube (matmul tandem) and bubbles
    recursive function Projector3D_transform_cube(self, cubein) result(cubeout)
        class(Operator3D), intent(in) :: self
        !> Input cube
        real(REAL64), intent(in)      :: cubein(:,:,:)

        real(REAL64), allocatable     :: cubeout(:,:,:)

        integer                       :: iz, iy, ip

#ifdef HAVE_CUDA
        type(CUDAcube_t) :: CUcubein, CUcubeout, CUcubetemp,&
                            CUfx, CUfy, CUfz, CUtmp1, CUtmp2
#else
        real(REAL64), allocatable :: tmp1(:,:)
        real(REAL64), allocatable :: tmp_u(:,:,:)
#endif
        integer            :: dims(X_:Z_, IN_:OUT_) ! (3, 2) array

        character(22) :: label

        ! dims: number of grid points per each axis 
        dims=self%get_dims()
        ! allocate result matrix
        allocate(cubeout( dims(X_, OUT_), dims(Y_, OUT_), dims(Z_, OUT_) ) )

        write(label, '("Operator cube (R=",i0,")")') size(self%w)
        call bigben%split(label)
#ifdef HAVE_CUDA
        call pdebug("Using CUDA", 1)
        call bigben%split("CuBLAS initialization")
        call CUBLAS_init()
        call bigben%stop()

        call bigben%split("CUDA alloc. and upload")
        call CUDAcube_init(CUcubein,cubein)
        call CUDAcube_init(CUcubeout,cubeout)
        call CUDAcube_init(CUfx,self%f(X_)%p)
        call CUDAcube_init(CUfy,self%f(Y_)%p)
        call CUDAcube_init(CUfz,self%f(Z_)%p)

        call CUDAcube_upload(CUcubein)
        call CUDAcube_upload(CUfx)
        call CUDAcube_upload(CUfy)
        call CUDAcube_upload(CUfz)

        call CUDAcube_init_remote(CUtmp1,&
                 [dims(X_,OUT_),dims(Y_,IN_),dims(Z_,IN_)])
        call CUDAcube_init_remote(CUtmp2,&
                 [dims(X_,OUT_),dims(Y_,OUT_),dims(Z_,IN_)])
        call bigben%stop()

#else
        allocate(tmp1(    dims(X_, OUT_), dims(Y_, IN_) ) )
        allocate(tmp_u(   dims(X_, OUT_), dims(Y_, OUT_), dims(Z_, IN_) ) )
#endif
        if (verbo_g>0) call progress_bar(0,size(self%w))

        cubeout=0.d0

        call bigben%split("t-points")
        ! go through all t-points
        iploop: do ip=1,size(self%w)
#ifdef HAVE_CUDA
            do iz=1,dims(Z_,IN_)
                call CUBLAS_dgemm(iz,&
                        CUDAcube_slice(CUfx,Z_,ip),&
                        CUDAcube_slice(CUcubein,Z_,iz),&
                        CUDAcube_slice(CUtmp1,Z_,iz),&
                        1.d0, 0.d0)
            end do
            call CUDA_sync()
            do iz=1,dims(Z_,IN_)
                call CUBLAS_dgemm(iz,&
                        CUDAcube_slice(CUtmp1,Z_,iz),&
                        CUDAcube_slice(CUfy,  Z_,ip),&
                        CUDAcube_slice(CUtmp2,Z_,iz),1.d0,0.d0)
            end do
            call CUDA_sync()
            do iy=1,dims(Y_,OUT_)
                call CUBLAS_dgemm(iy,&
                        CUDAcube_slice(CUtmp2,Y_,iy),&
                        CUDAcube_slice(CUfz,Z_,ip),&
                        CUDAcube_slice(CUcubeout,Y_,iy),&
                        self%w(ip), 1.d0)
            end do
            call CUDAsync()
#else
            ! go through all z slices in the        
            do iz=1,dims(Z_,IN_)
                ! multiply tranformation matrices 2d-slice with the corresponding input cube
                tmp1=xmatmul(self%f(X_)%p(:,:,ip),cubein(:,:,iz))
                tmp_u(:,:,iz)=xmatmul(tmp1,self%f(Y_)%p(:,:,ip))
            end do
            do iy=1,dims(Y_,OUT_)
                cubeout(:,iy,:) = cubeout(:,iy,:) + &
                        self%w(ip)*xmatmul(tmp_u(:,iy,:),self%f(Z_)%p(:,:,ip))
            end do
#endif
            if (verbo_g>0) call progress_bar(ip)
        end do iploop
        call bigben%stop()

#ifdef HAVE_CUDA
! Download cube
        call bigben%split("CUDA download")
        call CUDAcube_download(CUcubeout)
        call bigben%stop()
#endif

        ! This could be done using CUDA. Worth it?
        if (self%coda /= 0.d0) then
            call bigben%split("Coda")
            cubeout=cubeout + &
                self%coda * cube_project(cubein, self%gridin, self%gridout)
            call bigben%stop()
        end if

#ifdef HAVE_CUDA
        ! Finalize CUDA objects
        call bigben%split("CUDA destroy")
        call CUDAcube_destroy(CUcubein)
        call CUDAcube_destroy(CUcubeout)
        call CUDAcube_destroy(CUfx)
        call CUDAcube_destroy(CUfy)
        call CUDAcube_destroy(CUfz)
        call CUDAcube_destroy(CUtmp1)
        call CUDAcube_destroy(CUtmp2)
        call CUBLAS_destroy()
        call bigben%stop()
#else
        deallocate(tmp1)
        deallocate(tmp_u)
#endif
        call bigben%stop()

    end function
end module