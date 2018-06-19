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
!> @file potential.F90
!! Routines for calculating electrostatic interaction energies

!> Calculates electrostatic interaction energies numerically
!!
module potential_class
    use globals_m
    use Function3D_class
    use bubbles_class
    use harmonic_class
    use grid_class
    use expo_m
    use GaussQuad_class
    use mpi_m
    !use datadump_m
    use xmatrix_m
    use timer_m
    use ParallelInfo_class
#ifdef HAVE_BLAS95
    use blas95
#endif
    
    implicit none
    private

    real(REAL64) :: dtime
    public :: selfint_cheap
    public :: nucint
    public :: nuclear_repulsion
!    public :: calc_spectrum

contains

    !> Compute interaction energy
    !! \f$ E=\int_\Omega \rho(\mathbf{r})V(\mathbf{r})\mathrm{d}^3r\f$.

    !> This is the newer method to compute integrals over products of
    !! functions. The product of the two functions is decomposed into bubbles
    !! and cube, and the different parts are integrated:
    !!
    !! \f[ \int_\Omega \rho(\mathbf{r})V(\mathbf{r})\mathrm{d}^3r=
    !!  \int_\Omega\left[( \rho V )^\Delta(\mathbf{r}) +
    !! \sum_{Alm}{( \rho V )^A_{lm}(r_A)
    !!           Y^A_{lm} (\theta_A,\phi_A)}\right]\mathrm{d}^3r=
    !!     \sum_{ijk}(\rho V)^\Delta_{ijk}
    !! {\textstyle \left[ \int_{-\infty}^{\infty} \chi_i(x)\mathrm{d}x\right]
    !!             \left[ \int_{-\infty}^{\infty} \chi_j(y)\mathrm{d}y\right]
    !!             \left[ \int_{-\infty}^{\infty} \chi_k(z)\mathrm{d}z\right]}+
    !! \sum_{A}{\int_0^\infty( \rho V )^A_{00}(r_A)
    !!           4\pi r_A^2\mathrm{d}r_A}         \f]
    !!
    !! For functions with no bubbles, this is slightly less accurate
    !! \c selfint, but cheaper, and can be used in combination with bubbles.
    function selfint_cheap(pot, dens) result(res)
        type(Function3D)    :: pot, dens
        type(Function3D)    :: rhov
        real(REAL64)            :: res

        call bigben%split("Selfint_cheap")

        rhov = dens * pot

        call rhov%set_label('energy density')

        if(debug_g>0) then
            call rhov%write('rhov_'//pot%get_label()//'_nonsph.cub')!,write_bubbles=.FALSE.)
        end if

        call bigben%split("Integrating energy density")

        res = rhov%integrate()

        call bigben%stop()

        if(debug_g>0) then
            call rhov%write('rhov_'//pot%get_label()//'.cub')!,write_bubbles=.FALSE.)
            if (rhov%bubbles%get_nbub()>0) then
                call rhov%bubbles%print('rhov_'//pot%get_label())
            end if
        end if
        call rhov%destroy()
        call bigben%stop()
        return
    end function

    !> Compute the interaction energy of a set of point charges with an
    !! electrostatic potential.
    !! \f$ E=\sum_A -Z_A V(\mathbf{R}_A)\f$.
    function nucint(pot)
        real(REAL64) :: nucint
        type(Function3D),target :: pot

        nucint=- sum( pot%bubbles%get_z() * &
                      pot%evaluate( pot%bubbles%get_centers() ) )

        return
    end function

    !> @todo Documentation

    !> Compute the nuclear interaction energy using the (slower and less
    !! accurate) algorithm in XXXXXXX
    function nucint_weird(dens) result(res)
        type(Function3D), intent(in), target :: dens
        real(REAL64) :: res

        type(Function3D) :: rhov
        type(Bubbles) :: alpha, beta, rhovbubs

        type(Bubbles), pointer :: densbubs
        type(YBundle) :: yb
        type(Grid3D),pointer :: grid

        integer(INT32) :: gdims(X_:Z_)
        real(REAL64),pointer :: crd(:,:),z(:)
        real(REAL64),pointer :: xgrid(:),ygrid(:),zgrid(:)
        real(REAL64),pointer :: denscube(:,:,:), rhovcube(:,:,:)

        real(REAL64),allocatable :: r(:,:),u(:),d(:),a(:),b(:),s(:,:),tmp(:)
        real(REAL64),pointer :: dd,rr
        real(REAL64),pointer :: bf(:),df(:),af(:),rf(:),rg(:)

        integer(INT32) :: ix,iy,iz,nbub,inuc,jnuc,dm,nf,lmax
        real(REAL64) :: grcrd(X_:Z_), v

        res=0.d0

        if (dens%bubbles%get_nbub()==0) then
            call pwarn("Nuclear interaction only possible with bubbles")
            return
        end if

        denscube => dens%get_cube()
        densbubs => dens%bubbles
        grid     => dens%grid

        crd=> densbubs%get_centers()
        z=>   densbubs%get_z()

        gdims = grid%get_shape()
        xgrid => grid%axis(X_)%get_coord()
        ygrid => grid%axis(Y_)%get_coord()
        zgrid => grid%axis(Z_)%get_coord()

        lmax=densbubs%get_lmax()
        nbub=dens%bubbles%get_nbub()
        nf=(lmax+1)**2
        
        ! alpha is the contaminant bubbles of density bubbles weighted with erfc(r-1/r)
        alpha=alpha_bubbles(densbubs, dens%get_cube_contaminants(tmax=lmax),&
                            lmax=lmax)

        yb=YBundle(alpha%get_lmax())

        beta =beta_bubbles (densbubs, lmax=lmax)
        if (debug_g>=1) then
            call alpha%print('alpha')
            call beta%print('beta')
        end if

        !  the s-bubbles
        rhovbubs=Bubbles(densbubs, copy_content=.FALSE.,lmax=0,k=-2)
        do inuc=1,nbub
            af=>alpha%get_f(   inuc,0,0)
            bf=>beta%get_f(    inuc,0,0)
            df=>densbubs%get_f(inuc,0,0)
            rf=>rhovbubs%get_f(inuc,0,0)
            rg => rhovbubs%gr(inuc)%p%get_coord()
            rf=rg*(bf*rg-z(inuc)*(df+af))
        end do
        res=res+rhovbubs%integrate()
        nullify(rf)
        call rhovbubs%destroy()

        !call init_function_copy(rhov, dens, 'nuclear potential',copy_bubbles=.FALSE.)
        rhov = Function3D(SerialInfo(dens%grid), Bubbles(dens%bubbles) )
        rhovcube => rhov%get_cube()
        allocate(r(4,nbub))
        allocate(u(nbub))
        allocate(d(nbub))
        allocate(a(nbub))
        allocate(b(nbub))
        allocate(s(nf,nbub))
        allocate(tmp(nf))

        if (verbo_g>=1) call pinfo('Calculating residual nuclear interaction energy density...')
        if (verbo_g>=1) call progress_bar(0,gdims(Z_))
        ! Calculate the remaining part of the cube
        do iz=1,gdims(Z_)
            grcrd(Z_)=zgrid(iz)
            do iy=1,gdims(Y_)
                grcrd(Y_)=ygrid(iy)
                do ix=1,gdims(X_)
                    grcrd(X_)=xgrid(ix)
                    ! Calculate relative coordinates
                    do dm=X_,Z_
                        r(dm,:)=grcrd(dm)-crd(dm,:)
                    end do
                    ! Calculate distances
                    r(4,:)=sqrt(sum(r(:3,:)*r(:3,:),dim=1))
                    ! Calculate nucpots
                    where (r(4,:)>1.e-12)
                        u=-z/r(4,:)
                    elsewhere
                        u=0.d0
                    end where
                    ! Calculate d,a,b,s
                    do inuc=1,nbub
!                            s(:,inuc:inuc)=Ybundle_eval(yb,r(:,inuc:inuc))
                        tmp=reshape(yb%eval(r(:3,inuc:inuc), r(4, inuc:inuc)),[nf])
                        s(:,inuc)=tmp
                    end do
                    d(:)=sum(densbubs%eval_radial(r(4,:))*s,dim=1)
                    a(:)=sum(   alpha%eval_radial(r(4,:))*s,dim=1)
                    b(:)=sum(    beta%eval_radial(r(4,:))*s,dim=1)
                    dd=>denscube(ix,iy,iz)
                    rr=>rhovcube(ix,iy,iz)
                    ! extracted
                    rr=0.d0
                    do inuc=1,nbub
                        v=dd-a(inuc)
                        do jnuc=1,inuc-1
                            v=v+d(jnuc)
                        end do
                        do jnuc=inuc+1,nbub
                            v=v+d(jnuc)
                        end do
                        rr=rr  +  u(inuc)*v  -  b(inuc)
                    end do
                     ! injected
!                       rr=dd*sum(u) - sum(u*(d+a)+b)
                end do
            end do
            if (verbo_g>=1) call progress_bar(iz,gdims(Z_))
        end do
        res=res+rhov%integrate()
        !if (debug_g>=1) call write_function(rhov,"rhovnuc_nonsph.cub",write_bubbles=.FALSE.)

        ! And now clean up the whole mess!
        call alpha%destroy()
        call beta%destroy()
        call rhov%destroy()
        deallocate(r)
        deallocate(u)
        deallocate(d)
        deallocate(a)
        deallocate(b)
        deallocate(s)
        call yb%destroy()
        return

    end function

    !> Compute the contribution of each t-point to the self-interaction energy.
    !! \f$ W(t)=\int_\Omega' \int_\Omega \rho(\mathbf{r})
    !! e^{-t^2|\mathbf{r}-\mathbf{r}'|^2}\rho(\mathbf{r}')
    !! ~\mathrm{d}^3r~\mathrm{d}^3r'\f$.
!    subroutine calc_spectrum(dens, pot, trange)
!        type(Function3D),intent(in) :: dens
!        type(Function3D),intent(in) :: pot
!        real(REAL64)                    :: trange(:)
!        type(Function3D) :: rhov
!        type(GaussQuad) :: gau
!        type(Operator3D) :: genpot_w ! Working genpot
!        integer(INT32) :: i
!
!        real(REAL64), pointer :: tpoints(:)
!
!        gau=GaussQuadInitSpectrum(trange)
!        genpot_w = Coulomb3D(dens%grid, pot%grid, gau)
!        tpoints=> gau%get_tpoints()
!
!        do i=1, gau%get_npoints()
!            call genpot_w%apply_pick(dens, pot, i)
!            rhov = function_product(dens,pot)
!            print*, tpoints(i), rhov%integrate()
!        end do
!
!        call rhov%destroy()
!
!        return
!    end subroutine

    function nuclear_repulsion(z, coords) result(vnuc)
        real(REAL64), intent(in) :: z(:)
        real(REAL64), intent(in) :: coords(3,size(z))
        real(REAL64)             :: vnuc

        integer                  :: i, j, n

        call bigben%split("Computing nuclear repulsion energy")

        n=size(z)
        vnuc=sum([(sum([(z(i)*z(j)/norm2(coords(:,i)-coords(:,j)), &
                                                                j=i+1,n)] ), &
                                                                i=1,  n)] )
        call bigben%stop()
    end function
end module
