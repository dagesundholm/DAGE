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
!> @file exponentials.F90
!! Matrices for the Coulomb operator

!> Matrices for the CoulombOperator constructor
!! @todo Documentation
module expo_m
    use Globals_m
    use Grid_class
    use LIPBasis_class
    use Xmatrix_m
    use PBC_class
    use MPI_m
    use Timer_m
    implicit none

    public :: interp_matrix
    public :: make_Coulomb3D_matrices, point_charge_exponentials
    private
    contains

    function point_charge_exponentials(input_grid, axis, centers, tpoints, &
                                                               convergence_threshold) result(expmat)
        !> the grid given as input
        type(Grid3D), intent(in), target   :: input_grid
        !> axis of the target (X = 1, Y = 2, or Z = 3)
        integer(INT32), intent(in)         :: axis
        !> centers of the point charges 
        real(REAL64), intent(in)           :: centers(:)
        !> The t-point values in Singer integral identity
        real(REAL64), intent(in)           :: tpoints(:)
        !> Relative convergence threshold, if not give the
        !! threshold will be set to 1*10^-12
        real(REAL64), intent(in), optional :: convergence_threshold
        

        type(LIPBasis), pointer :: lip
        type(PBC), pointer :: pbc_ptr
        
        real(REAL64), allocatable, target :: expmat(:, :, :)
        
        real(REAL64)                         :: x2,beginning,theend, cell_coordinate, t
        real(REAL64)                         :: shft,pot_boxmid, box_length, displacement
        real(REAL64), allocatable            :: temp_expmat(:),contribution(:,:)
        integer(INT32)                       :: nlip,j

        ! Iterator index variables
        integer(INT32)                      :: idisp, ipoint, icell, itpoint
        ! The final varibale containing the convergence threshold
        real(REAL64)                        :: cvg=1.D-12 

        real(REAL64), pointer               :: expmat_slice(:, :)
        
        logical                             :: initialize,duplicate,thiswaslast
        real(REAL64), dimension(:), pointer :: cell_scales
        type(REAL64_2D), allocatable        :: lip_coeffs(:)

        ! gr1d_in is the density, gr1d_out is  the potential
        lip     => input_grid%get_lip()
        pbc_ptr => input_grid%get_pbc()


        allocate(expmat(input_grid%axis(axis)%get_shape(), size(centers), size(tpoints)))


        if(present(convergence_threshold)) cvg=convergence_threshold

        ! reset them here
        expmat=0.d0


        ! the integration is done in the q-space (q= x/h+shift)
        ! integration from (1-n)/2..(n-1)/2
        nlip = input_grid%get_nlip() 
        allocate(temp_expmat(nlip))
        allocate(contribution(nlip, 2))
        theend=lip%get_last()
        beginning=lip%get_first()

        box_length = input_grid%axis(axis)%get_qmax()-input_grid%axis(axis)%get_qmin()

        ! for all points in the x2 direction
        ! THIS SHOULD BE THE LOCAL NDIM

        cell_scales => input_grid%axis(axis)%get_cell_scales()
        lip_coeffs=lip%coeffs(0)
        ! Loop over all t-points
        do itpoint=1, size(tpoints) 
            t = tpoints(itpoint)
            ! Loop over all point charges
            do icell=1, input_grid%axis(axis)%get_ncell()
                
                ! get the exponent (t*h) measured in cells (each one being h in length)
                shft=t*cell_scales(icell)

                ! get the scaled box length (ie, as it is stored in grid%lip)
                displacement = box_length/cell_scales(icell)

                ! get the slice of the result matrix corrensponding to this t-point and cell
                expmat_slice => expmat((icell-1)*(nlip-1)+1 : (icell-1)*(nlip-1)+nlip, :, itpoint)

                do ipoint=1, size(centers)
                    ! get the coordinate of the point ipoint in the relative cell 'icell' coordinates,
                    ! where the center of the cell is the zero point.
                    cell_coordinate = input_grid%axis(axis)%x2cell(centers(ipoint), icell)

                    j=1 ! Counter of the number of images taken into account
                    initialize=.TRUE.
                    temp_expmat=0.d0
                    contribution=0.d0
                    do
                        ! Get new displacement
                        call pbc_ptr%get_1d_disp(axis, initialize, idisp, duplicate, thiswaslast)
                    
                        ! Calculate integral from 'beginning' to 'theend'
                        !! \math \int_{a}^{b} x^{i} e^{-t^{2}(x-p)^{2}}\;dx \math
                        !! integrals(a,b,t,p,n)
                        contribution(:, 2) = integrals(beginning, theend, shft, cell_coordinate - idisp*displacement, nlip)
                        ! Calculate also contribution of the image, if needed
                        if(duplicate) contribution(:, 2) = contribution(:,2)+&
                                     integrals(beginning,theend,shft,cell_coordinate + idisp*displacement, nlip)

                        ! add the contribution to the temporary exponential matrix
                        temp_expmat=temp_expmat+contribution(:,2)

                        ! Exit if this was the last idisp to consider
                        if(thiswaslast) exit

                        ! Otherwise check convergence
                        ! contrib(:,2) contains the largest contributions so far
                        ! contrib(:,2) was the current contribution

                        ! If contrib(2,:) brings any larger integral, put it in
                        where(abs(contribution(:, 2)) > abs(contribution(:, 1))) contribution(:, 1) = contribution(:, 2)

                        ! If the last contributions were all tiny
                        if(all(cvg*abs(contribution(:, 1)) >= abs(contribution(:, 2)))) exit
                        if(all(contribution(:, 2) == 0.d0) .and. j>1) exit

                        j=j+1
                    end do
                    ! Multiply the integrals times the polynomial coefficients
                    temp_expmat = xmatmul(lip_coeffs(1)%p(:,:),temp_expmat)*cell_scales(icell)
                    expmat_slice(:, ipoint) = expmat_slice(:, ipoint) + temp_expmat
                end do
            end do
        end do
        deallocate(temp_expmat)
        deallocate(contribution)
        deallocate(lip_coeffs(1)%p)
        deallocate(lip_coeffs)
        return

    end function

    function exponentials(dens_grid, pot_grid, crd, tpoints, j_max, trsp, conv_in) &
                                                                    result(expmat)
        !> density_grid and potential_grid
        type(Grid3D), intent(in), target   :: dens_grid, pot_grid
        !> axis of the target (X = 1, Y = 2, or Z = 3)
        integer(INT32), intent(in)         :: crd
        real(REAL64), intent(in)           :: tpoints(:)
        integer(INT32)                     :: j_max
        character, intent(in), optional    :: trsp
        real(REAL64), intent(in), optional :: conv_in
        

        type(LIPBasis), pointer :: lip
        type(PBC), pointer :: pbc_ptr
        
        real(REAL64), allocatable, target :: expmat(:,:, :)
        

        type(Grid1D), pointer :: gr1d_in, gr1d_out
        real(REAL64) :: x2,beginning,theend,pst0, t
        real(REAL64) :: shft,pot_boxmid,den_boxlen,disp
        real(REAL64), allocatable :: temp_expmat(:),contrib(:,:)
        integer(INT32) :: ix2,ncell, nlip,j, itpoint
        integer(INT32) :: idisp, first_gridpoint, last_gridpoint
        real(REAL64) :: qmax,qmin, cell_length, tic, toc, mul_time

        real(REAL64) :: cvg=1.D-12 ! Relative error threshold

        real(REAL64) :: total_cell_size
        real(REAL64), pointer :: expmat_slice(:, :)
        
        logical :: initialize,duplicate,thiswaslast
        real(REAL64),dimension(:),pointer :: grid, cell_scales
        type(REAL64_2D),allocatable :: lip_coeffs(:)

        ! gr1d_in is the density, gr1d_out is  the potential
        lip     => dens_grid%get_lip()
        pbc_ptr => dens_grid%get_pbc()

        ! choose coordinate x,y,z
        gr1d_in  => dens_grid%axis(crd)
        gr1d_out => pot_grid%axis(crd)

        ! allocate the result matrix, trsp : transpose
        if(.not.present(trsp)) then
            allocate(expmat(gr1d_out%get_shape(), gr1d_in%get_shape(), size(tpoints)))
        else
            allocate(expmat(gr1d_in%get_shape(), gr1d_out%get_shape(), size(tpoints)))
        end if

        total_cell_size = gr1d_in%get_delta() 

        if(present(conv_in)) cvg=conv_in

        ! reset them here
        expmat=0.d0

        !j_max=0

        ! the integration is done in the q-space (q= x/h+shift)
        ! integration from (1-n)/2..(n-1)/2
        nlip=dens_grid%get_nlip() 
        allocate(temp_expmat(nlip))
        allocate(contrib(nlip, 2))
        theend=lip%get_last()
        beginning=lip%get_first()

        qmin=gr1d_out%get_qmin()
        qmax=gr1d_out%get_qmax()
        den_boxlen=qmax-qmin

        ! for all points in the x2 direction
        ! THIS SHOULD BE THE LOCAL NDIM

        grid => gr1d_out%get_coord()
        cell_scales => gr1d_in%get_cell_scales()
        mul_time = 0
        lip_coeffs=lip%coeffs(0)
        do itpoint = 1, size(tpoints) 
            t = tpoints(itpoint)
            do ncell=1,gr1d_in%get_ncell()
                cell_length = cell_scales(ncell)
                shft=t*cell_length
                disp=den_boxlen/cell_length
                first_gridpoint = (ncell-1)*(nlip-1)+1
                last_gridpoint = (ncell-1)*(nlip-1)+nlip
            
                if (present(trsp)) then
                    expmat_slice => expmat(first_gridpoint:last_gridpoint, :, itpoint)
                else 
                    expmat_slice => expmat(:,first_gridpoint:last_gridpoint, itpoint)
                end if
                do ix2=1,gr1d_out%get_shape()
                    x2=grid(ix2)
            
            ! integrate every basis function with exponential weights
            
                
    ! the integration is done in the basis function coordinates
    ! pst0  = -(np-1)/2+(p-gr1d_in)/h is the distance from the center of the box to the center of the gaussian measured in cells (each one being h in length)
    ! shft =  shft= t*h is the exponent measured in cells (each one being h in length)
    ! disp is the box length in unit cells. For an equidistant grid, it is simply gr1d_in%ndim

!                pst0=beginning+(x2-cellb-disp0)/gr1d_in%cellh(ncell)
                    pst0=gr1d_in%x2cell(x2,ncell)
                

                    j=1 ! Counter of the number of images taken into account
                    initialize=.true.
                    temp_expmat=0.d0
                    contrib=0.d0
                
                    do
                        ! Get new displacement
                        call pbc_ptr%get_1d_disp(crd,initialize,idisp,duplicate,thiswaslast)
                    
                        ! Calculate integral from 'beginning' to 'theend'
                        !! \math \int_{a}^{b} x^{i} e^{-t^{2}(x-p)^{2}}\;dx \math
                        !! integrals(a,b,t,p,n)
                        contrib(:, 2)=integrals(beginning,theend,shft,pst0-idisp*disp,nlip)
                     
                        ! Calculate also contribution of the image, if needed
                        if(duplicate) contrib(:, 2)=contrib(:,2)+&
                                     integrals(beginning,theend,shft,pst0+idisp*disp,nlip)
                        temp_expmat=temp_expmat+contrib(:,2)
                        ! Exit if this was the last idisp to consider
                        if(thiswaslast) exit
                        ! Otherwise check convergence
                        ! contrib(:,2) contains the largest contributions so far
                        ! contrib(:,2) was the current contribution

                        ! If contrib(2,:) brings any larger integral, put it there
                        where(abs(contrib(:, 2))>abs(contrib(:, 1))) contrib(:, 1)=contrib(:, 2)
                        ! If the last contributions were all tiny
                        if(all(cvg*abs(contrib(:, 1))>=abs(contrib(:, 2)))) exit
                        if(all(contrib(:, 2)==0.d0).and.j>1) exit
!                        if(j>0) then
!                            if(j>1000) then
!                                print*,contrib(:,2)
!                                print*,contrib(:,2)
!                                read(*,*)
!                            end if
!                            if(all(1.d8*contrib(:,2)<contrib(:,1))) then
!                                exit
!                            else if(all(contrib(:,2)>contrib(:,2))) then
!                                contrib(:,1)=contrib(:,2)
!                            else if(all(contrib==0.d0)) then
!                                exit
!                            end if
!                        end if
                        j=j+1
                    end do
                    !j_max=max(j_max,j)
                    ! Multiply the integrals times the polynomial coefficients
                    temp_expmat=xmatmul(lip_coeffs(1)%p(:,:),temp_expmat)*&
                                                         cell_length
                    if (present(trsp)) then
                        expmat_slice(:, ix2)= expmat_slice(:, ix2) + temp_expmat
                    else
                        expmat_slice(ix2, :)= expmat_slice(ix2, :) + temp_expmat
                    end if 
                end do
            end do
        end do
        deallocate(temp_expmat)
        deallocate(contrib)
        deallocate(lip_coeffs(1)%p)
        deallocate(lip_coeffs)
        return

    end function exponentials

    function border_exponentials(dens_grid, pot_grid, crd, tpoints, j_max, trsp, conv_in) &
                                                                    result(expmat)
        !> density_grid and potential_grid
        type(Grid3D), intent(in), target   :: dens_grid, pot_grid
        !> axis of the target (X = 1, Y = 2, or Z = 3)
        integer(INT32), intent(in)         :: crd
        real(REAL64), intent(in)           :: tpoints(:)
        integer(INT32)                     :: j_max
        character, intent(in), optional    :: trsp
        real(REAL64), intent(in), optional :: conv_in

        !> the result matrix
        real(REAL64), allocatable, target :: expmat(:,:, :)
        

        type(LIPBasis), pointer :: lip
        type(PBC), pointer :: pbc_ptr        

        type(Grid1D), pointer :: gr1d_in, gr1d_out
        real(REAL64) :: x2,beginning,theend,pst0, t
        real(REAL64) :: shft,pot_boxmid,den_boxlen,disp
        real(REAL64), allocatable :: temp_expmat(:),contrib(:,:)
        integer(INT32) :: ix2,ncell, nlip,j, itpoint, evaluated_cells(2), evaluated_grid_points(2)
        integer(INT32) :: idisp, first_gridpoint, last_gridpoint, icell, i_grid_point
        real(REAL64) :: qmax,qmin, cell_length, tic, toc, mul_time

        real(REAL64) :: cvg=1.D-12 ! Relative error threshold

        real(REAL64) :: total_cell_size
        real(REAL64), pointer :: expmat_slice(:, :)
        
        logical :: initialize,duplicate,thiswaslast
        real(REAL64),dimension(:),pointer :: grid, cell_scales
        type(REAL64_2D),allocatable :: lip_coeffs(:)

        ! gr1d_in is the density, gr1d_out is  the potential
        lip     => dens_grid%get_lip()
        pbc_ptr => dens_grid%get_pbc()

        ! choose coordinate x,y,z
        gr1d_in  => dens_grid%axis(crd)
        gr1d_out => pot_grid%axis(crd)

        ! allocate the result matrix, trsp : transpose
        if(.not.present(trsp)) then
            allocate(expmat(gr1d_out%get_shape(), gr1d_in%get_shape(), size(tpoints)))
        else
            allocate(expmat(gr1d_in%get_shape(), gr1d_out%get_shape(), size(tpoints)))
        end if

        total_cell_size = gr1d_in%get_delta() 

        if(present(conv_in)) cvg=conv_in

        ! evaluate only the first and the last cell
        evaluated_cells = [1, gr1d_in%get_ncell()]
        evaluated_grid_points = [1, gr1d_out%get_shape()]

        expmat = 0.0d0

        ! the integration is done in the q-space (q= x/h+shift)
        ! integration from (1-n)/2..(n-1)/2
        nlip=dens_grid%get_nlip() 
        allocate(temp_expmat(nlip))
        allocate(contrib(nlip, 2))
        theend=lip%get_last()
        beginning=lip%get_first()

        qmin=gr1d_out%get_qmin()
        qmax=gr1d_out%get_qmax()
        den_boxlen=qmax-qmin
        ! for all points in the x2 direction
        ! THIS SHOULD BE THE LOCAL NDIM

        grid => gr1d_out%get_coord()
        cell_scales => gr1d_in%get_cell_scales()
        mul_time = 0
        lip_coeffs=lip%coeffs(0)
        do itpoint = 1, size(tpoints) 
            t = tpoints(itpoint)
            do icell=1,2
                ncell = evaluated_cells(icell)
                cell_length = cell_scales(ncell)
                shft=t*cell_length
                disp=den_boxlen/cell_length
                first_gridpoint = (ncell-1)*(nlip-1)+1
                last_gridpoint = (ncell-1)*(nlip-1)+nlip
                if (present(trsp)) then
                    expmat_slice => expmat(first_gridpoint:last_gridpoint, :, itpoint)
                else 
                    expmat_slice => expmat(:,first_gridpoint:last_gridpoint, itpoint)
                end if
                do ix2=1,gr1d_out%get_shape()
                    !ix2 = evaluated_grid_points(i_grid_point)
                    x2=grid(ix2)
            
            ! integrate every basis function with exponential weights
            
                
    ! the integration is done in the basis function coordinates
    ! pst0  = -(np-1)/2+(p-gr1d_in)/h is the distance from the center of the box to the center of the gaussian measured in cells (each one being h in length)
    ! shft =  shft= t*h is the exponent measured in cells (each one being h in length)
    ! disp is the box length in unit cells. For an equidistant grid, it is simply gr1d_in%ndim

!                pst0=beginning+(x2-cellb-disp0)/gr1d_in%cellh(ncell)
                    pst0=gr1d_in%x2cell(x2,ncell)
                

                    j=1 ! Counter of the number of images taken into account
                    initialize=.true.
                    temp_expmat=0.d0
                    contrib=0.d0
                
                    do
                        ! Get new displacement
                        call pbc_ptr%get_1d_disp(crd,initialize,idisp,duplicate,thiswaslast)
                    
                        ! Calculate integral
                        contrib(:, 2)=integrals(beginning,theend,shft,pst0-idisp*disp,nlip)
                     
                        ! Calculate also contribution of the image, if needed
                        if(duplicate) contrib(:, 2)=contrib(:,2)+&
                                     integrals(beginning,theend,shft,pst0+idisp*disp,nlip)
                        temp_expmat=temp_expmat+contrib(:,2)
                        ! Exit if this was the last idisp to consider
                        if(thiswaslast) exit
                        ! Otherwise check convergence
                        ! contrib(:,2) contains the largest contributions so far
                        ! contrib(:,2) was the current contribution

                        ! If contrib(2,:) brings any larger integral, put it there
                        where(abs(contrib(:, 2))>abs(contrib(:, 1))) contrib(:, 1)=contrib(:, 2)
                        ! If the last contributions were all tiny
                        if(all(cvg*abs(contrib(:, 1))>=abs(contrib(:, 2)))) exit
                        if(all(contrib(:, 2)==0.d0).and.j>1) exit
!                        if(j>0) then
!                            if(j>1000) then
!                                print*,contrib(:,2)
!                                print*,contrib(:,2)
!                                read(*,*)
!                            end if
!                            if(all(1.d8*contrib(:,2)<contrib(:,1))) then
!                                exit
!                            else if(all(contrib(:,2)>contrib(:,2))) then
!                                contrib(:,1)=contrib(:,2)
!                            else if(all(contrib==0.d0)) then
!                                exit
!                            end if
!                        end if
                        j=j+1
                    end do
                    !j_max=max(j_max,j)
                    ! Multiply the integrals times the polynomial coefficients
                    temp_expmat=xmatmul(lip_coeffs(1)%p(:,:),temp_expmat)*&
                                                         cell_length
                    if (present(trsp)) then
                        expmat_slice(:, ix2)= expmat_slice(:, ix2) + temp_expmat
                    else
                        expmat_slice(ix2, :)= expmat_slice(ix2, :) + temp_expmat
                    end if 
                end do
                
            end do
             
            do ncell=2, gr1d_in%get_ncell() -1 
                cell_length = cell_scales(ncell)
                shft=t*cell_length
                disp=den_boxlen/cell_length
                first_gridpoint = (ncell-1)*(nlip-1)+1
                last_gridpoint = (ncell-1)*(nlip-1)+nlip
                if (present(trsp)) then
                    expmat_slice => expmat(first_gridpoint:last_gridpoint, :, itpoint)
                else 
                    expmat_slice => expmat(:,first_gridpoint:last_gridpoint, itpoint)
                end if
                do i_grid_point=1,2
                    ix2 = evaluated_grid_points(i_grid_point)
                    x2=grid(ix2)
            
            ! integrate every basis function with exponential weights
            
                
    ! the integration is done in the basis function coordinates
    ! pst0  = -(np-1)/2+(p-gr1d_in)/h is the distance from the center of the box to the center of the gaussian measured in cells (each one being h in length)
    ! shft =  shft= t*h is the exponent measured in cells (each one being h in length)
    ! disp is the box length in unit cells. For an equidistant grid, it is simply gr1d_in%ndim

!                pst0=beginning+(x2-cellb-disp0)/gr1d_in%cellh(ncell)
                    pst0=gr1d_in%x2cell(x2,ncell)
                

                    j=1 ! Counter of the number of images taken into account
                    initialize=.true.
                    temp_expmat=0.d0
                    contrib=0.d0
                
                    do
                        ! Get new displacement
                        call pbc_ptr%get_1d_disp(crd,initialize,idisp,duplicate,thiswaslast)
                    
                        ! Calculate integral
                        contrib(:, 2)=integrals(beginning,theend,shft,pst0-idisp*disp,nlip)
                     
                        ! Calculate also contribution of the image, if needed
                        if(duplicate) contrib(:, 2)=contrib(:,2)+&
                                     integrals(beginning,theend,shft,pst0+idisp*disp,nlip)
                        temp_expmat=temp_expmat+contrib(:,2)
                        ! Exit if this was the last idisp to consider
                        if(thiswaslast) exit
                        ! Otherwise check convergence
                        ! contrib(:,2) contains the largest contributions so far
                        ! contrib(:,2) was the current contribution

                        ! If contrib(2,:) brings any larger integral, put it there
                        where(abs(contrib(:, 2))>abs(contrib(:, 1))) contrib(:, 1)=contrib(:, 2)
                        ! If the last contributions were all tiny
                        if(all(cvg*abs(contrib(:, 1))>=abs(contrib(:, 2)))) exit
                        if(all(contrib(:, 2)==0.d0).and.j>1) exit
!                        if(j>0) then
!                            if(j>1000) then
!                                print*,contrib(:,2)
!                                print*,contrib(:,2)
!                                read(*,*)
!                            end if
!                            if(all(1.d8*contrib(:,2)<contrib(:,1))) then
!                                exit
!                            else if(all(contrib(:,2)>contrib(:,2))) then
!                                contrib(:,1)=contrib(:,2)
!                            else if(all(contrib==0.d0)) then
!                                exit
!                            end if
!                        end if
                        j=j+1
                    end do
                    !j_max=max(j_max,j)
                    ! Multiply the integrals times the polynomial coefficients
                    temp_expmat=xmatmul(lip_coeffs(1)%p(:,:),temp_expmat)*&
                                                         cell_length
                    if (present(trsp)) then
                        expmat_slice(:, ix2)= expmat_slice(:, ix2) + temp_expmat
                    else
                        expmat_slice(ix2, :)= expmat_slice(ix2, :) + temp_expmat
                    end if 
                end do
                
            end do
        end do
        deallocate(temp_expmat)
        deallocate(contrib)
        deallocate(lip_coeffs(1)%p)
        deallocate(lip_coeffs)
        return

    end function

    function make_Coulomb3D_matrices(gridin, gridout, tp, jmax, threshold, only_border) &
                                                                    result(f)
        type(Grid3D), intent(in)           :: gridin
        type(Grid3D), intent(in)           :: gridout
        real(REAL64), intent(in)           :: tp(:)
        integer, intent(out)               :: jmax(3)
        real(REAL64), intent(in), optional :: threshold
        logical,      intent(in), optional :: only_border
        real(REAL64)                       :: tic, toc
        type(REAL64_3D)                    :: f(3)

        integer :: ip

        call bigben%split("Constructing Coulomb operator matrices")
        f=alloc_transformation_matrices(gridin, gridout, size(tp))
        if (verbo_g==1) then
            call progress_bar(0,size(tp))
        else if(verbo_g>1) then
            call pprint("           |          | Index of the outermost cell")
            call pprint("   t-point | Time (s) | needed to converge the sum")
            call pprint("           |          |      x     y     z  ")
            call pprint("------------------------------------------------")
        end if

        call cpu_time(toc)
        if (present(only_border) .and. only_border) then
            f(X_)%p=border_exponentials(gridin,gridout,X_,tp, jmax(X_))
            f(Y_)%p=border_exponentials(gridin,gridout,Y_,tp, jmax(Y_),'t')
            f(Z_)%p=border_exponentials(gridin,gridout,Z_,tp, jmax(Z_),'t')
        else
            f(X_)%p=exponentials(gridin,gridout,X_,tp, jmax(X_))
            f(Y_)%p=exponentials(gridin,gridout,Y_,tp, jmax(Y_),'t')
            f(Z_)%p=exponentials(gridin,gridout,Z_,tp, jmax(Z_),'t')
        end if
        if (verbo_g>=1) call progress_bar(ip)
        
        call bigben%stop()
    end function

    !> Returns a vector of integrals.
    !!
    !! Elements in `values` are given by 
    !! 
    !! \math \int_{a}^{b} x^{i} e^{-t^{2}(x-p)^{2}}\;dx \math
    pure function integrals(a,b,t,p,n) result(values)
        integer(INT32), intent(in) :: n
        real(REAL64), intent(in) :: a,b,t,p
        real(REAL64)             :: values(n)
        integer(INT32) :: i
!        if(ipow.lt.7) then
            if(t.lt.0.1d0) then
                do i=0,n-1
                    values(n-i)=gauss_expo_n(a,b,t,p,i) 
                end do
            else
!                values=recursint(a,b,t,p,n) 
                do i=0,n-1
                    values(n-i)=expo_an(a,b,t,p,i) 
                end do
            end if
!        else
!            value=expo_7(a,b,t,p)
!        end if
    end function

    pure function expo_an(a,b,t,p,n) result(value)
        ! Interface to the analytical exponentials
        real(REAL64), intent(in)   :: a,b,t,p
        real(REAL64)               :: value
        integer(INT32), intent(in) :: n
        select case(n)
            case(0)
                value=expo_0(a,b,t,p) 
            case(1)
                value=expo_1(a,b,t,p) 
            case(2)
                value=expo_2(a,b,t,p) 
            case(3)
                value=expo_3(a,b,t,p) 
            case(4)
                value=expo_4(a,b,t,p) 
            case(5)
                value=expo_5(a,b,t,p) 
            case(6)
                value=expo_6(a,b,t,p) 
        end select
        return
    end function

    !     
    ! Calculate the integral x^0*exp(-t*t*(x-p)^2) dx between a and b
    !                                                 where a < b
    pure function expo_0(a,b,t,p) result(value)
        real(REAL64), intent(in) :: a,b,t,p  

        ! Debug data: t=2, p=1 a=0 b=infinity yield:
        ! 0.25*sqrt(pii)+0.25*erf(2)*sqrt(pii)
        ! 30.1.2004

        real(REAL64) :: sqpii,derf,instability_threshold,value

        real(REAL64), parameter  :: pii = 3.141592653589793D0

        instability_threshold=1.0d-06

        if(t.eq.0.d0) then
            value=(b-a)
        else if(t.lt.instability_threshold) then
            value=(b-a)
        else 
            sqpii=dsqrt(pii)
            value=sqpii*(derf(b*t-t*p)-derf(a*t-t*p))/t/2.d0
        end if 
    end function expo_0

    !     
    ! Calculate the integral x^1*exp(-t*t*(x-p)^2) dx between a and b
    !                                                 where a < b
    pure function expo_1(a,b,t,p) result(value)
        real(REAL64), intent(in) :: a,b,t,p  

        ! Debug data: t=2, p=1 a=0 b=infinity yield:
        ! 0.25*sqrt(pii)+0.125*exp(-4)+0.25*erf(2)*sqrt(pii)
        ! 30.1.2004

        real(REAL64) :: sqpii,derf,value

        real(REAL64) :: v1,v2,v3,q,r,instability_threshold

        real(REAL64), parameter  :: pii = 3.141592653589793D0

        instability_threshold=1.0d-06

        if(t.eq.0.d0) then
            value=(b*b-a*a)/2.d0
        else if(t.lt.instability_threshold) then
            value=(b*b-a*a)/2.d0
        !      write(6,*) 'Instability warning in expo_1, t = and p = ',t,p
        else 
            sqpii=dsqrt(pii)

            v1=p*sqpii*derf(t*(b-p))

            if(t.gt.1.d-03) then
                v2=(dexp(-t**2*(a-p)**2)-dexp(-t**2*(b-p)**2))/t
            else
            ! this is not necessary !!
            !  s=t*(b-p)
            !  v=0.3275911d0
            !  u=1.d0/(1.d0+v)
            !  a1=0.254829592d0
            !  a2=-0.284496736d0
            !  a3=1.421413741d0
            !  a4=-1.453152027d0
            !  a5=1.061405429d0
            !erf(s)=1-(a1*u+a2*u*u+a3*u*u*u+a4*u*u*u*u+a5*u*u*u*u*u)* &
            !(1-s*s+s*s*s*s/2.d0)
                q=-t**2*(a-p)**2
                r=-t**2*(b-p)**2
                v2=(q+q*q/2.d0+q*q*q/6.d0+q*q*q*q/24.d0 &
                -r-r*r/2.d0-r*r*r/6.d0-r*r*r*r/24.d0)/t
            end if
            v3=-p*sqpii*derf(a*t-t*p)
            value=(v1+v2+v3)/t/2.d0

            ! value=(-dexp(-t**2*(b**2+p**2))*dexp(2*t**2*p*b)+ &
            !         dexp(-t**2*(b**2+p**2))*p*sqpii*derf(b*t-t*p)*t* &
            !         dexp(t**2*(b**2+p**2))+dexp(-t**2*(a**2+p**2))* &
            !         dexp(2*t**2*p*a)-dexp(-t**2*(a**2+p**2))*p*sqpii* &
            !         derf(a*t-t*p)*t*dexp(t**2*(a**2+p**2)))/t**2/2.d0
        end if 

    end function expo_1

    pure function expo_2(a,b,t,p) result(value)
        !     
        ! Calculate the integral x^2*exp(-t*t*(x-p)^2) dx between a and b
        !                                             where a < b

        ! Debug data: t=2, p=1 a=0 b=infinity yield:
        ! 9/32*sqrt(pii)+0.125*exp(-4)+9/32*erf(2)*sqrt(pii)
        ! 30.1.2004

        implicit none
        real(REAL64), intent(in) :: a,b,t,p
        real(REAL64) :: sqpii,derf,value  
        real(REAL64) :: v1,v2,v3,v4,v5
        real(REAL64) :: q,r,instability_threshold              

        real(REAL64), parameter  :: pii = 3.141592653589793D0

        instability_threshold=0.5d-04

        if(t.eq.0.d0) then
            value=(b**3-a**3)/3.d0
        else if(t.lt.instability_threshold) then
            value=(b**3-a**3)/3.d0
            ! write(6,*) 'Instability warning in expo_2, t = and p = ',t,p
        else 
            sqpii=dsqrt(pii)
            v1=p**2*sqpii*derf(b*t-t*p)/2.d0
            if(t.gt.1.d-03) then
                v2=(dexp(-t*t*(a-p)**2)*a-dexp(-t*t*(b-p)**2)*b)
                v3=(dexp(-t*t*(a-p)**2)-dexp(-t*t*(b-p)**2))*p/t/2.d0
            else
                q=-t*t*(a-p)**2
                r=-t*t*(b-p)**2
                v2=(q+q*q/2.d0+q*q*q/6.d0+q*q*q*q/24.d0)*a &
                -(r+r*r/2.d0+r*r*r/6.d0+r*r*r*r/24.d0)*b
                v2=(v2+a-b)
                v3=(q+q*q/2.d0+q*q*q/6.d0+q*q*q*q/24.d0) &
                -(r+r*r/2.d0+r*r*r/6.d0+r*r*r*r/24.d0)
                v3=v3*p/t/2.d0
            end if

        ! cancellation of significant numbers
        ! v2 and v5 should be combined to increase the accuracy for small t
        ! how this is done is not yet known, DS 2.2.2004, this is almost ok
            v5=sqpii*(derf(t*(b-p))-derf(t*(a-p)))/t/2.d0
            v2=(v2+v5)/t/2.d0
            v5=0.d0

            v4=-p**2*sqpii*derf(a*t-t*p)/2.d0

            value=(v1+v2+v3+v4+v5)/t

        !  value= (-2.d0*dexp(-t**2*(b**2+p**2))*b*dexp(2*t**2*p*b)*t- & 
        !   2.d0*dexp(-t**2*(b**2+p**2))*p*dexp(2*t**2*p*b)*t+     &
        !   2.d0*dexp(-t**2*(b**2+p**2))*t**2*p**2*sqpii*   &
        !   derf(b*t-t*p)*dexp(t**2*(b**2+p**2))+dexp(-t**2*(b**2+p**2))* &
        !   sqpii*derf(b*t-t*p)*dexp(t**2*(b**2+p**2))+ &
        !   2.d0*dexp(-t**2*(a**2+p**2))*a*dexp(2*t**2*p*a)*t+ &
        !   2.d0*dexp(-t**2*(a**2+p**2))*p*dexp(2*t**2*p*a)*t- &
        !   2.d0*dexp(-t**2*(a**2+p**2))*t**2*p**2*sqpii* & 
        !   derf(a*t-t*p)*dexp(t**2*(a**2+p**2))-dexp(-t**2*(a**2+p**2))* &
        !   sqpii*derf(a*t-t*p)*dexp(t**2*(a**2+p**2)))/t**3/4.d0
        end if

    end function expo_2

    pure function expo_3(a,b,t,p) result(value)
        !     
        ! Calculate the integral x^3*exp(-t*t*(x-p)^2) dx between a and b
        !                                             where a < b

        ! Debug data: t=2, p=1 a=0 b=infinity yield:
        ! 11/32*sqrt(pii)+5/32*exp(-4)+11/32*erf(2)*sqrt(pii)
        ! 30.1.2004

        implicit none
        real(REAL64), intent(in) :: a,b,t,p  
        real(REAL64) :: sqpii,derf,value
        real(REAL64) :: v1,v2,v3,v4,v5,v6
        real(REAL64) :: q,r,instability_threshold

        real(REAL64), parameter  :: pii = 3.141592653589793D0

        instability_threshold=1.0d-04

        if(t.eq.0.d0) then
            value=(b**4-a**4)/4.d0
        else if(t.lt.instability_threshold) then
            value=(b**4-a**4)/4.d0
        !      write(6,*) 'Instability warning in expo_3, t = and p = ',t,p
        else
            sqpii=dsqrt(pii)

            if(t.gt.1.d-03) then
                v1 =  2.d0*(dexp(-t**2*(a-p)**2)*a**2 &
                -dexp(-t**2*(b-p)**2)*b**2)/t**2  
                v2 =  2.d0*p*(a*dexp(-t*t*(a-p)**2)-b*dexp(-t*t*(b-p)**2))*t
                v3 =  2.d0*(dexp(-t**2*(a-p)**2)*p**2 &
                -dexp(-t**2*(b-p)**2)*p**2)/t**2 
                v5 =  2.d0*(dexp(-t**2*(a-p)**2)-dexp(-t**2*(b-p)**2))/t
            else
                q=-t*t*(a-p)**2
                r=-t*t*(b-p)**2
                v1 = (1.d0+q+q*q/2.d0+q*q*q/6.d0+q*q*q*q/24.d0)*a*a &
                -(1.d0+r+r*r/2.d0+r*r*r/6.d0+r*r*r*r/24.d0)*b*b
                v1=2.d0*v1/t/t
                v2 = (q+q*q/2.d0+q*q*q/6.d0+q*q*q*q/24.d0)*a &
                -(r+r*r/2.d0+r*r*r/6.d0+r*r*r*r/24.d0)*b
                v2=v2+a-b
                v2=2.d0*v2*p*t
                v3 =  (q+q*q/2.d0+q*q*q/6.d0+q*q*q*q/24.d0) &
                -(r+r*r/2.d0+r*r*r/6.d0+r*r*r*r/24.d0)
                v3=2.d0*v3*p*p/t/t
                v5 =  (q+q*q/2.d0+q*q*q/6.d0+q*q*q*q/24.d0) &
                -(r+r*r/2.d0+r*r*r/6.d0+r*r*r*r/24.d0)
                v5=2.d0*v5/t
            end if
            v4 =  2.d0*p**3*sqpii*(derf(b*t-t*p)-derf(a*t-t*p))/t 
            v6 =  3.d0*p*sqpii*(derf(b*t-t*p)-derf(a*t-t*p))  

    ! improved precision can be obtained by combining in an earlier stage
    ! the sum of them should be calculated as a series expansion for small t
            v6 =  (v6+v2+v5)/t/t/t
            v2=  0.d0 
            v5=  0.d0 
            value=0.25d0*(v1+v2+v3+v4+v5+v6)
    ! write(6,*) v1,v2,v3,v4
    ! write(6,*) v5,v6,value 
    ! write(6,*) value 

    ! s1 = 0.25d0      
    ! s4 = -2.d0*dexp(-t**2*(b**2+p**2))*b**2*dexp(2*t**2*p*b)*t**2- & 
    !       2.d0*dexp(-t**2*(b**2+p**2))*p*b*dexp(2*t**2*p*b)*t**2-  &
    !       2.d0*dexp(-t**2*(b**2+p**2))*p**2*dexp(2*t**2*p*b)*t**2+ &
    !       2.d0*dexp(-t**2*(b**2+p**2))*t**3*p**3*sqpii*derf(b*t-t*p)* &
    !       dexp(t**2*(b**2+p**2))+3.d0*dexp(-t**2*(b**2+p**2))*p*sqpii* & 
    !       derf(b*t-t*p)*t*dexp(t**2*(b**2+p**2))- &
    !       2.d0*dexp(-t**2*(b**2+p**2))*dexp(2*t**2*p*b)
    ! s3 = s4+2d0*dexp(-t**2*(a**2+p**2))*a**2*dexp(2*t**2*p*a)*t**2+ &
    !      2.d0*dexp(-t**2*(a**2+p**2))*p*a*dexp(2*t**2*p*a)*t**2+ &
    !      2.d0*dexp(-t**2*(a**2+p**2))*p**2*dexp(2*t**2*p*a)*t**2- &
    !      2.d0*dexp(-t**2*(a**2+p**2))*t**3*p**3*sqpii*derf(a*t-t*p)* &
    !      dexp(t**2*(a**2+p**2))-3.d0*dexp(-t**2*(a**2+p**2))*p*sqpii* &
    !      derf(a*t-t*p)*t*dexp(t**2*(a**2+p**2))+2.d0* &
    !      dexp(-t**2*(a**2+p**2))*dexp(2*t**2*p*a)
    ! s4 = 1/t**4
    ! s2 = s3*s4
    ! value = s1*s2
        end if

    end function expo_3

    pure function expo_4(a,b,t,p) result(value)
        !     
        ! Calculate the integral x^4*exp(-t*t*(x-p)^2) dx between a and b
        !                                             where a < b

        ! Debug data: t=2, p=1 a=0 b=infinity yield:
        ! 115/256*sqrt(pii)+13/64*exp(-4)+115/256*erf(2)*sqrt(pii)
        ! 30.1.2004

        implicit none
        real(REAL64), intent(in) :: a,b,t,p 
        real(REAL64) :: sqpii,derf,value 
        real(REAL64) :: v1,v2,v3,v4,v5,v6,v7,v8,v9
        real(REAL64) :: q,r,instability_threshold

        real(REAL64), parameter  :: pii = 3.141592653589793D0

        instability_threshold=1.0d-03

        if(t.eq.0.d0) then
            value=(b**5-a**5)/5.d0
        else if(t.lt.instability_threshold) then
            value=(b**5-a**5)/5.d0
            !  write(6,*) 'Instability warning in expo_4, t = and p = ',t,p
        else
            sqpii=dsqrt(pii)

            ! for small t values the integrals explode, the precision can be
            ! improved by removing cancellations of large numbers
            ! or alternatively an extrapolation approach
            if(t.gt.1.d-03) then
                v1 =  4.d0*(dexp(-t**2*(a-p)**2)*a**3 & 
                -dexp(-t**2*(b-p)**2)*b**3) 

                v2 =  4.d0*(dexp(-t**2*(a-p)**2)*p*a**2  & 
                -dexp(-t**2*(b-p)**2)*p*b**2)   

                v3 =  4.d0*(dexp(-t**2*(a-p)**2)*p**2*a  &   
                -dexp(-t**2*(b-p)**2)*p**2*b)   

                v4 =  4.d0*(dexp(-t**2*(a-p)**2)*p**3    &
                -dexp(-t**2*(b-p)**2)*p**3)   
                v7 =  10.d0*p*(dexp(-t**2*(a-p)**2)-dexp(-t**2*(b-p)**2))/t/t
                v8 =  6.d0*(dexp(-t**2*(a-p)**2)*a-dexp(-t**2*(b-p)**2)*b)
            else
                q=-t*t*(a-p)**2
                r=-t*t*(b-p)**2

                v1 = (1.d0+q+q*q/2.d0+q*q*q/6.d0+q*q*q*q/24.d0)*a**3 &
                -(1.d0+r+r*r/2.d0+r*r*r/6.d0+r*r*r*r/24.d0)*b**3
                v1 = 4.d0*v1

                v2 = (1.d0+q+q*q/2.d0+q*q*q/6.d0+q*q*q*q/24.d0)*p*a*a &
                -(1.d0+r+r*r/2.d0+r*r*r/6.d0+r*r*r*r/24.d0)*p*b*b
                v2 = 4.d0*v2

                v3 = (1.d0+q+q*q/2.d0+q*q*q/6.d0+q*q*q*q/24.d0)*p*p*a &
                -(1.d0+r+r*r/2.d0+r*r*r/6.d0+r*r*r*r/24.d0)*p*p*b
                v3 = 4.d0*v3

                v4 = (q+q*q/2.d0+q*q*q/6.d0+q*q*q*q/24.d0) &
                -(r+r*r/2.d0+r*r*r/6.d0+r*r*r*r/24.d0)
                v4 = 4.d0*v4*p**3

                v7 = (q+q*q/2.d0+q*q*q/6.d0+q*q*q*q/24.d0) &
                -(r+r*r/2.d0+r*r*r/6.d0+r*r*r*r/24.d0)
                v7 = 10.d0*v7*p/t/t

                v8 = (1.d0+q+q*q/2.d0+q*q*q/6.d0+q*q*q*q/24.d0)*a &
                -(1.d0+r+r*r/2.d0+r*r*r/6.d0+r*r*r*r/24.d0)*b
                v8 = 6.d0*v8
            end if

            v5 = -4.d0*t*p**4*sqpii*(derf(a*t-t*p)-derf(b*t-t*p))
            v6 = -12.d0/t*p**2*sqpii*(derf(a*t-t*p)-derf(b*t-t*p))
            v9 = -3.d0*sqpii*(derf(a*t-t*p)-derf(b*t-t*p))/t  

            v2=v2/t**2
            v5=(v4+v5)/t**2
            v7=(v3+v6+v7)/t**2
            v8=((v8+v9)/t/t+v1)/t**2
            v1=0.d0
            v3=0.d0
            v4=0.d0    
            v6=0.d0
            v9=0.d0
            value=0.125d0*(v1+v2+v3+v4+v5+v6+v7+v8+v9)
    ! write(6,*) v1,v2,v3
    ! write(6,*) v4,v5,v6
    ! write(6,*) v7,v8,v9

    ! s1 = 0.125d0      
    ! s4 = -4.d0*dexp(-t**2*(b**2+p**2))*b**3*dexp(2*t**2*p*b)*t**3- &
    !       4.d0*dexp(-t**2*(b**2+p**2))*p*b**2*dexp(2*t**2*p*b)*t**3- &
    !       4.d0*dexp(-t**2*(b**2+p**2))*p**2*b*dexp(2*t**2*p*b)*t**3- &
    !       4.d0*dexp(-t**2*(b**2+p**2))*p**3*dexp(2*t**2*p*b)*t**3+ &
    !       4.d0*dexp(-t**2*(b**2+p**2))*t**4*p**4*sqpii*derf(b*t-t*p)* &
    !       dexp(t**2*(b**2+p**2))+12.d0*dexp(-t**2*(b**2+p**2))*t**2*p**2* &
    !       sqpii*derf(b*t-t*p)*dexp(t**2*(b**2+p**2))- &
    !       10.d0*dexp(-t**2*(b**2+p**2))*p*dexp(2*t**2*p*b)*t- &
    !       6.d0*dexp(-t**2*(b**2+p**2))*b*dexp(2*t**2*p*b)*t+ &
    !       3.d0*dexp(-t**2*(b**2+p**2))*sqpii*derf(b*t-t*p)* &
    !       dexp(t**2*(b**2+p**2))
    ! s3 = s4+4.d0*dexp(-t**2*(a**2+p**2))*a**3*dexp(2*t**2*p*a)*t**3+ &
    !      4.d0*dexp(-t**2*(a**2+p**2))*p*a**2*dexp(2*t**2*p*a)*t**3+ &
    !      4.d0*dexp(-t**2*(a**2+p**2))*p**2*a*dexp(2*t**2*p*a)*t**3+ &
    !      4.d0*dexp(-t**2*(a**2+p**2))*p**3*dexp(2*t**2*p*a)*t**3- &
    !      4.d0*dexp(-t**2*(a**2+p**2))*t**4*p**4*sqpii*derf(a*t-t*p)* &
    !      dexp(t**2*(a**2+p**2))-12.d0*dexp(-t**2*(a**2+p**2))*t**2*p**2* &
    !      sqpii*derf(a*t-t*p)*dexp(t**2*(a**2+p**2))+ &
    !      10.d0*dexp(-t**2*(a**2+p**2))*p*dexp(2*t**2*p*a)*t+ &
    !      6.d0*dexp(-t**2*(a**2+p**2))*a*dexp(2*t**2*p*a)*t- &
    !      3.d0*dexp(-t**2*(a**2+p**2))*sqpii*derf(a*t-t*p)* &
    !      dexp(t**2*(a**2+p**2))
    ! s4 = 1/t**5
    ! s2 = s3*s4
    ! value = s1*s2

        end if

    end function expo_4

    pure function expo_5(a,b,t,p) result(value)
        !     
        ! Calculate the integral x^5*exp(-t*t*(x-p)^2) dx between a and b
        !                                             where a < b

        ! Debug data: t=2, p=1 a=0 b=infinity yield:
        ! 159/256*sqrt(pii)+9/32*exp(-4)+159/256*erf(2)*sqrt(pii)
        ! 30.1.2004

        implicit none
        real(REAL64), intent(in)  :: a,b,t,p  
        real(REAL64) :: sqpii,derf,value
        real(REAL64) :: q,r,instability_threshold    
        real(REAL64) :: v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12

        real(REAL64), parameter  :: pii = 3.141592653589793D0

        ! the error cancellation is not avoided, but it behaves better
        ! 2.2.2004

        instability_threshold=2.5d-03

        if(t.eq.0.d0) then
            value=(b**6-a**6)/6.d0
        else if(t.lt.instability_threshold) then
            value=(b**6-a**6)/6.d0
        ! write(6,*) 'Instability warning in expo_5, t = and p = ',t,p
        else
            sqpii=dsqrt(pii)

            if(t.gt.1.d-02) then

                v1 =  4.d0*(dexp(-t**2*(a-p)**2)*a**4    &   
                -dexp(-t**2*(b-p)**2)*b**4)

                v2 =  4.d0*(dexp(-t**2*(a-p)**2)*p*a**3  &   
                -dexp(-t**2*(b-p)**2)*p*b**3)

                v3 =  4.d0*(dexp(-t**2*(a-p)**2)*p**2*a**2  &  
                -dexp(-t**2*(b-p)**2)*p**2*b**2)

                v4 =  4.d0*(dexp(-t**2*(a-p)**2)*p**3*a   &
                -dexp(-t**2*(b-p)**2)*p**3*b)

                v5 =  4.d0*(dexp(-t**2*(a-p)**2)*p**4  &
                -dexp(-t**2*(b-p)**2)*p**4)

                v8 = 18.d0*(dexp(-t**2*(a-p)**2)-dexp(-t**2*(b-p)**2))*p**2/t**2

                v9 = 14.d0*(dexp(-t**2*(a-p)**2)*p*a  &
                -dexp(-t**2*(b-p)**2)*p*b)/t**2

                v11= 8.d0*(dexp(-t**2*(a-p)**2)*a**2  &
                -dexp(-t**2*(b-p)**2)*b**2)/t**2

                v12= 8.d0*(dexp(-t**2*(a-p)**2)-dexp(-t**2*(b-p)**2))/t**4

            else
                q=-t*t*(a-p)**2
                r=-t*t*(b-p)**2

                v1 = (1.d0+q+q*q/2.d0+q*q*q/6.d0+q*q*q*q/24.d0)*a**4 &
                -(1.d0+r+r*r/2.d0+r*r*r/6.d0+r*r*r*r/24.d0)*b**4
                v1 = 4.d0*v1

                v2 = (1.d0+q+q*q/2.d0+q*q*q/6.d0+q*q*q*q/24.d0)*p*a**3 &
                -(1.d0+r+r*r/2.d0+r*r*r/6.d0+r*r*r*r/24.d0)*p*b**3
                v2 = 4.d0*v2

                v3 = (1.d0+q+q*q/2.d0+q*q*q/6.d0+q*q*q*q/24.d0)*p**2*a**2 &
                -(1.d0+r+r*r/2.d0+r*r*r/6.d0+r*r*r*r/24.d0)*p**2*b**2
                v3 = 4.d0*v3

                v4 = (1.d0+q+q*q/2.d0+q*q*q/6.d0+q*q*q*q/24.d0)*p**3*a &
                -(1.d0+r+r*r/2.d0+r*r*r/6.d0+r*r*r*r/24.d0)*p**3*b
                v4 = 4.d0*v4

                v5 = (q+q*q/2.d0+q*q*q/6.d0+q*q*q*q/24.d0)*p**4 &
                -(r+r*r/2.d0+r*r*r/6.d0+r*r*r*r/24.d0)*p**4
                v5 = 4.d0*v5

                v8 = ((q+q*q/2.d0+q*q*q/6.d0+q*q*q*q/24.d0) &
                -(r+r*r/2.d0+r*r*r/6.d0+r*r*r*r/24.d0))*p**2/t**2
                v8 = 18.d0*v8

                v9 = ((1.d0+q+q*q/2.d0+q*q*q/6.d0+q*q*q*q/24.d0)*a*p &
                -(1.d0+r+r*r/2.d0+r*r*r/6.d0+r*r*r*r/24.d0)*b*p)/t**2
                v9 = 14.d0*v9

                v11 = ((1.d0+q+q*q/2.d0+q*q*q/6.d0+q*q*q*q/24.d0)*a*a &
                -(1.d0+r+r*r/2.d0+r*r*r/6.d0+r*r*r*r/24.d0)*b*b)/t**2
                v11 = 8.d0*v11

                v12 = ((q+q*q/2.d0+q*q*q/6.d0+q*q*q*q/24.d0) &
                -(r+r*r/2.d0+r*r*r/6.d0+r*r*r*r/24.d0))/t**4
                v12 = 8.d0*v12

            end if

            v6 = -4.d0*t*p**5*sqpii*(derf(a*t-t*p)-derf(b*t-t*p))
            v7 =-20.d0/t*p**3*sqpii*(derf(a*t-t*p)-derf(b*t-t*p))
            v10 = -15.d0*p*sqpii*(derf(a*t-t*p)-derf(b*t-t*p))/t**3

            ! write(6,*) v1,v2,v3
            ! write(6,*) v4,v5,v6
            ! write(6,*) v7,v8,v9
            ! write(6,*) v10,v11,v12

            value = v1+v2+v3+v4+v5+v6
            value = value + v7+v8+v9+v10+v11+v12

            value = value/t**2

            value = 0.125d0*value

    ! s1 = 0.125d0  
    ! s5 = -4.d0*dexp(-t**2*(b**2+p**2))*b**4*dexp(2*t**2*p*b)*t**4-  &
    !       4.d0*dexp(-t**2*(b**2+p**2))*p*b**3*dexp(2*t**2*p*b)*t**4-  &
    !       4.d0*dexp(-t**2*(b**2+p**2))*p**2*b**2*dexp(2*t**2*p*b)*t**4- &
    !       4.d0*dexp(-t**2*(b**2+p**2))*p**3*b*dexp(2*t**2*p*b)*t**4- &
    !       4.d0*dexp(-t**2*(b**2+p**2))*p**4*dexp(2*t**2*p*b)*t**4+ &
    !       4.d0*dexp(-t**2*(b**2+p**2))*t**5*p**5*sqpii*derf(b*t-t*p)* &
    !       dexp(t**2*(b**2+p**2)) 
    ! s4 = s5+20.d0*dexp(-t**2*(b**2+p**2))*t**3*p**3*sqpii*derf(b*t-t*p)* &
    !      dexp(t**2*(b**2+p**2))-18.d0*dexp(-t**2*(b**2+p**2))*p**2* &
    !      dexp(2*t**2*p*b)*t**2-14.d0*dexp(-t**2*(b**2+p**2))*p*b* &
    !      dexp(2*t**2*p*b)*t**2+15.d0*dexp(-t**2*(b**2+p**2))*p*sqpii* &
    !      derf(b*t-t*p)*t*dexp(t**2*(b**2+p**2))-8.d0* &
    !      dexp(-t**2*(b**2+p**2))*b**2*dexp(2*t**2*p*b)*t**2- &
    !      8.d0*dexp(-t**2*(b**2+p**2))*dexp(2*t**2*p*b)
    ! s5 = s4+4.d0*dexp(-t**2*(a**2+p**2))*a**4*dexp(2*t**2*p*a)*t**4+ &
    !      4.d0*dexp(-t**2*(a**2+p**2))*p*a**3*dexp(2*t**2*p*a)*t**4+ & 
    !      4.d0*dexp(-t**2*(a**2+p**2))*p**2*a**2*dexp(2*t**2*p*a)*t**4+ &
    !      4.d0*dexp(-t**2*(a**2+p**2))*p**3*a*dexp(2*t**2*p*a)*t**4+ &
    !      4.d0*dexp(-t**2*(a**2+p**2))*p**4*dexp(2*t**2*p*a)*t**4 
    ! s3 = s5-4.d0*dexp(-t**2*(a**2+p**2))*t**5*p**5*sqpii*derf(a*t-t*p)* &
    !      dexp(t**2*(a**2+p**2))-20.d0*dexp(-t**2*(a**2+p**2))*t**3*p**3* &
    !      sqpii*derf(a*t-t*p)*dexp(t**2*(a**2+p**2))+ &
    !      18.d0*dexp(-t**2*(a**2+p**2))*p**2*dexp(2*t**2*p*a)*t**2+ &
    !      14.d0*dexp(-t**2*(a**2+p**2))*p*a*dexp(2*t**2*p*a)*t**2- &
    !      15.d0*dexp(-t**2*(a**2+p**2))*p*sqpii*derf(a*t-t*p)*t* &
    !      dexp(t**2*(a**2+p**2))+8.d0*dexp(-t**2*(a**2+p**2))*a**2* &
    !      dexp(2*t**2*p*a)*t**2+8.d0*dexp(-t**2*(a**2+p**2))* &
    !      dexp(2*t**2*p*a)
    ! s4 = 1/t**6
    ! s2 = s3*s4
    ! value = s1*s2
        end if

    end function expo_5

    pure function expo_6(a,b,t,p) result(value) 
        !     
        ! Calculate the integral x^6*exp(-t*t*(x-p)^2) dx between a and b
        !                                             where a < b

        ! Debug data: t=2, p=1 a=0 b=infinity yield:
        ! 1847/2048*sqrt(pii)+209/512*exp(-4)+1847/2048*erf(2)*sqrt(pii)
        ! 30.1.2004

        implicit none
        real(REAL64), intent(in) :: a,b,t,p 
        real(REAL64) :: sqpii,derf,value 
        real(REAL64) :: q,r,instability_threshold 
        real(REAL64) :: v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16

        real(REAL64), parameter  :: pii = 3.141592653589793D0

        ! for small t values the integrals explode, the precision can be
        ! improved by removing cancellations of large numbers
        ! or alternatively an extrapolation approach

        instability_threshold=5.0d-03

        if(t.eq.0.d0) then
            value=(b**7-a**7)/7.d0

            ! It is better to remove the instabilities than to let them in
        else if(t.lt.instability_threshold) then
            value=(b**7-a**7)/7.d0
            ! write(6,*) 'Instability warning in expo_6, t = and p = ',t,p
        else
            sqpii=dsqrt(pii)

            v1=30.d0*(dexp(-t**2*(a-p)**2)*a-dexp(-t**2*(b-p)**2)*b)
            v2=15.d0*sqpii*(derf(b*t-t*p)-derf(a*t-t*p))
            v1=(v1+v2/t)/t/t/t
            v2=0.d0

            v3=90.d0*p**2*sqpii*(derf(b*t-t*p)-derf(a*t-t*p))/t/t

            v4=56.d0*(dexp(-t**2*(a-p)**2)-dexp(-t**2*(b-p)**2))*p**3/t 

            v5=66.d0*(dexp(-t**2*(a-p)**2)-dexp(-t**2*(b-p)**2))*p/t/t/t   

            v6=8.d0*(dexp(-t**2*(a-p)**2)-dexp(-t**2*(b-p)**2))*p**5*t

            v7=20.d0*(dexp(-t**2*(a-p)**2)*a**3-dexp(-t**2*(b-p)**2)*b**3)/t

            v8=36.d0*(dexp(-t**2*(a-p)**2)*a**2-dexp(-t**2*(b-p)**2)*b**2)*p/t

            v9=48.d0*(dexp(-t**2*(a-p)**2)*a-dexp(-t**2*(b-p)**2)*b)*p**2/t

            v10=8.d0*p**6*sqpii*(derf(b*t-t*p)-derf(a*t-t*p))*t**2

            v11=8.d0*(dexp(-t**2*(a-p)**2)*a**4-dexp(-t**2*(b-p)**2)*b**4)*p*t

            v12=8.d0*(dexp(-t**2*(a-p)**2)*a**3-dexp(-t**2*(b-p)**2)*b**3)*p**2*t

            v13=8.d0*(dexp(-t**2*(a-p)**2)*a**5-dexp(-t**2*(b-p)**2)*b**5)*t   

            v14=8.d0*(dexp(-t**2*(a-p)**2)*a-dexp(-t**2*(b-p)**2)*b)*p**4*t

            v15=8.d0*(dexp(-t**2*(a-p)**2)*a**2-dexp(-t**2*(b-p)**2)*b**2)*p**3*t

            v16=60.d0*p**4*sqpii*(derf(b*t-t*p)-derf(a*t-t*p))

            value=v1+v2+v3+v4+v5+v6+v7+v8
            value=value+v9+v10+v11+v12+v13+v14+v15+v16

            ! write(6,*) v1,v2,v3
            ! write(6,*) v4,v5,v6
            ! write(6,*) v7,v8,v9
            ! write(6,*) v10,v11,v12
            ! write(6,*) v13,v14,v15
            ! write(6,*) v16,value, ' final'
            value = value/t**3
            value = 0.0625d0*value

    ! s1 = 0.0625d0
    ! s5 = 30.d0*dexp(-t**2*(a**2+p**2))*a*dexp(2*t**2*p*a)*t- &
    !      15.d0*dexp(-t**2*(a**2+p**2))*sqpii*derf(a*t-t*p)* &
    !      dexp(t**2*(a**2+p**2))-8.d0*dexp(-t**2*(b**2+p**2))*p**2*b**3* &
    !      dexp(2*t**2*p*b)*t**5-90.d0*dexp(-t**2*(a**2+p**2))*t**2*p**2* &
    !      sqpii*derf(a*t-t*p)*dexp(t**2*(a**2+p**2))+ &
    !      56.d0*dexp(-t**2*(a**2+p**2))*p**3*dexp(2*t**2*p*a)*t**3+ &
    !      66.d0*dexp(-t**2*(a**2+p**2))*p*dexp(2*t**2*p*a)*t+ &
    !      8.d0*dexp(-t**2*(a**2+p**2))*a**5*dexp(2*t**2*p*a)*t**5+ &
    !      8.d0*dexp(-t**2*(a**2+p**2))*p**5*dexp(2*t**2*p*a)*t**5
    ! s4 = s5-30.d0*dexp(-t**2*(b**2+p**2))*b*dexp(2*t**2*p*b)*t+ &
    !      15.d0*dexp(-t**2*(b**2+p**2))*sqpii*derf(b*t-t*p)* &
    !      dexp(t**2*(b**2+p**2))+20.d0*dexp(-t**2*(a**2+p**2))*a**3* &
    !      dexp(2*t**2*p*a)*t**3-8.d0*dexp(-t**2*(b**2+p**2))*p**5* &
    !      dexp(2*t**2*p*b)*t**5-36.d0*dexp(-t**2*(b**2+p**2))*p*b**2* &
    !      dexp(2*t**2*p*b)*t**3-48.d0*dexp(-t**2*(b**2+p**2))*p**2*b* &
    !      dexp(2*t**2*p*b)*t**3-8.d0*dexp(-t**2*(b**2+p**2))*p**4*b* &
    !      dexp(2*t**2*p*b)*t**5+8.d0*dexp(-t**2*(b**2+p**2))*t**6*p**6* &
    !      sqpii*derf(b*t-t*p)*dexp(t**2*(b**2+p**2))
    ! s5 = s4-8.d0*dexp(-t**2*(b**2+p**2))*p**3*b**2*dexp(2*t**2*p*b)*t**5- &
    !      56.d0*dexp(-t**2*(b**2+p**2))*p**3*dexp(2*t**2*p*b)*t**3- &
    !      66d0*dexp(-t**2*(b**2+p**2))*p*dexp(2*t**2*p*b)*t+ &
    !      8.d0*dexp(-t**2*(a**2+p**2))*p*a**4*dexp(2*t**2*p*a)*t**5+ &
    !      8.d0*dexp(-t**2*(a**2+p**2))*p**2*a**3*dexp(2*t**2*p*a)*t**5- &
    !      8.d0*dexp(-t**2*(b**2+p**2))*b**5*dexp(2*t**2*p*b)*t**5-8* &
    !      dexp(-t**2*(b**2+p**2))*p*b**4*dexp(2*t**2*p*b)*t**5
    ! s6 = s5+60.d0*dexp(-t**2*(b**2+p**2))*t**4*p**4*sqpii*derf(b*t-t*p)* &
    !      dexp(t**2*(b**2+p**2))-20.d0*dexp(-t**2*(b**2+p**2))*b**3* &
    !      dexp(2*t**2*p*b)*t**3+8.d0*dexp(-t**2*(a**2+p**2))*p**4*a* &
    !      dexp(2*t**2*p*a)*t**5+8.d0*dexp(-t**2*(a**2+p**2))*p**3*a**2* &
    !      dexp(2*t**2*p*a)*t**5 
    ! s3 = s6+90.d0*dexp(-t**2*(b**2+p**2))*t**2*p**2*sqpii*derf(b*t-t*p)* &
    !      dexp(t**2*(b**2+p**2))-8.d0*dexp(-t**2*(a**2+p**2))*t**6*p**6* &
    !      sqpii*derf(a*t-t*p)*dexp(t**2*(a**2+p**2))+ &
    !      48.d0*dexp(-t**2*(a**2+p**2))*p**2*a*dexp(2*t**2*p*a)*t**3- &
    !      60.d0*dexp(-t**2*(a**2+p**2))*t**4*p**4*sqpii*derf(a*t-t*p)* &
    !      dexp(t**2*(a**2+p**2))+36.d0*dexp(-t**2*(a**2+p**2))*p*a**2* &
    !      dexp(2*t**2*p*a)*t**3
    ! s4 = 1/t**7
    ! s2 = s3*s4
    ! value = s1*s2
        end if

    end function expo_6

    pure function expo_7(a,b,t,p) result(value)
        !     
        ! Calculate the integral x^7*exp(-t*t*(x-p)^2) dx between a and b
        !                                             where a < b
        ! 30.1.2004

        implicit none
        real(REAL64), intent(in) :: a,b,t,p 
        real(REAL64) :: sqpii,derf,value 

        real(REAL64), parameter  :: pii = 3.141592653589793D0

        sqpii=dsqrt(pii)

        !write(6,*) ' the exponential matrices have not been coded'
        !write(6,*) ' for cases where n>6 in x^n*exp(-t^2*(x-p)^2)'
        !write(6,*) ' Therefore I stop now in expo_7'
        !write(6,*) ' Thus the program needs some coding if you'
        !write(6,*) ' really want to do this calculation'
        !write(6,*) ' Stopped in: expo_7'
        !stop

        value=1.23456789d10
    end function expo_7

    pure function ds_derf(x) result(value)
        !
        ! Error function from Numerical Recipes now in double precision.
        ! derf(x) = 1.d0 - derfc(x)
        !

        real(REAL64), intent(in) :: x
        real(REAL64) t, z, value, dumerfc

        z = dabs(x)
        t = 1.0d0 / ( 1.0d0 + 0.5d0 * z )

        dumerfc =       t * dexp(-z * z - 1.26551223d0 + t *    &
        ( 1.00002368d0 + t * ( 0.37409196d0 + t *       &
        ( 0.09678418d0 + t * (-0.18628806d0 + t *       &
        ( 0.27886807d0 + t * (-1.13520398d0 + t *       &
        ( 1.48851587d0 + t * (-0.82215223d0 + t * 0.17087277d0 )))))))))

        if ( x.lt.0.0d0 ) dumerfc = 2.0d0 - dumerfc

        value= 1.0d0 - dumerfc

    end function ds_derf

    pure function gauss_expo_0(a,b,t,p) result(value)
        !     
        ! Calculate the integral x^0*exp(-t*t*(x-p)^2) dx between a and b
        !                                                 where a < b
        ! using Gauss integration
        !
        ! Debug data: t=2, p=1 a=0 b=infinity yield:
        ! 0.25*sqrt(pii)+0.25*erf(2)*sqrt(pii)
        ! 30.1.2004
        ! Calculate the integral exp(-t*t*(x-p)^2) dx between a and b
        !                                             where a < b

        implicit none
        real(REAL64), intent(in) :: a,b,t,p  
        real(REAL64) :: two,a0,a1,pr,qr,tw,tp,summa,value
        real(REAL64), dimension(12) :: wk,tk
        integer(INT32) :: ngauss,i,j,k

        real(REAL64) :: x_limit_a,x_limit_b
        real(REAL64), dimension(4) :: limit   
        integer(INT32) :: interval 

        ngauss=12
        two=2.d0

        if(t.eq.0.d0) then
            summa=(b-a) 
        else 

            ! x value for half width
            x_limit_b=dsqrt(dlog(two))/t+p
            x_limit_a=-dsqrt(dlog(two))/t+p

            ! Three main cases can be identified
            if(p.gt.a.and.p.lt.b) then
                if(x_limit_a.gt.a.and.x_limit_b.lt.b) then
                    interval=3
                    limit(1)=a
                    limit(2)=x_limit_a
                    limit(3)=x_limit_b
                    limit(4)=b
                    !         a-> x_limit_a -> x_limit_b -> b 
                else if(x_limit_a.le.a.and.x_limit_b.lt.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=x_limit_b 
                    limit(3)=b
                    !          a -> x_limit_b -> b
                else if(x_limit_a.gt.a.and.x_limit_b.ge.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=x_limit_a 
                    limit(3)=b
                    !          a-> x_limit_a -> b
                else if(x_limit_a.le.a.and.x_limit_b.ge.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=p 
                    limit(3)=b
                    !          a-> p -> b
                end if
            else if(p.le.a) then
                if(x_limit_b.gt.a.and.x_limit_b.lt.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=x_limit_b
                    limit(3)=b
                    !         a -> x_limit_b -> b
                else 
                    interval=2
                    limit(1)=a
                    limit(2)=(a+b)/2.d0
                    limit(3)=b
                    !         a-> (a+b)/2 -> b
                end if
            else if(p.ge.b) then
                if(x_limit_a.gt.a.and.x_limit_a.lt.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=x_limit_a
                    limit(3)=b
                    !         a-> xlimit_a -> b
                else 
                    interval=2
                    limit(1)=a
                    limit(2)=(a+b)/2.d0
                    limit(3)=b
                    !          a -> (a+b)/2 -> b
                end if
            end if

            wk(1) =0.047175336386512D00
            wk(2) =0.106939325995318D00
            wk(3) =0.160078328543346D00
            wk(4) =0.203167426723066D00
            wk(5) =0.233492536538355D00
            wk(6) =0.249147045813403D00
            wk(7) =0.249147045813403D00
            wk(8) =0.233492536538355D00
            wk(9) =0.203167426723066D00
            wk(10)=0.160078328543346D00
            wk(11)=0.106939325995318D00
            wk(12)=0.047175336386512D00

            tk(1) =-0.981560634246719D00
            tk(2) =-0.904117256370475D00
            tk(3) =-0.769902674194305D00
            tk(4) =-0.587317954286617D00
            tk(5) =-0.367831498998180D00
            tk(6) =-0.125233408511469D00
            tk(7) = 0.125233408511469D00
            tk(8) = 0.367831498998180D00
            tk(9) = 0.587317954286617D00
            tk(10)= 0.769902674194305D00
            tk(11)= 0.904117256370475D00
            tk(12)= 0.981560634246719D00

            ! Gauss integration
            summa=0.d0
            do i=1,interval 
                a0=limit(i)
                a1=limit(i+1)
                pr=1.d0-a1*2.d0/(a1-a0)
                qr=2.d0/(a1-a0)
                do j=1,ngauss
                    k=(i-1)*ngauss+j
                    tp=(tk(j)-pr)/qr
                    tw=wk(j)/qr
                    summa=summa+dexp(-t*t*(tp-p)*(tp-p))*tw
                end do
            end do
        end if

        value=summa

    end function gauss_expo_0

    pure function gauss_expo_1(a,b,t,p) result(value)
        !     
        ! Calculate the integral x^1*exp(-t*t*(x-p)^2) dx between a and b
        !                                                 where a < b
        ! using Gauss integration

        ! Debug data: t=2, p=1 a=0 b=infinity yield:
        ! 0.25*sqrt(pii)+0.125*exp(-4)+0.25*erf(2)*sqrt(pii)
        ! 30.1.2004

        implicit none
        real(REAL64), intent(in) :: a,b,t,p 
        real(REAL64) :: two,a0,a1,pr,qr,tw,tp,summa,value 
        real(REAL64), dimension(12) :: wk,tk
        integer(INT32) :: ngauss,i,j,k

        real(REAL64) :: x_limit_a,x_limit_b
        real(REAL64), dimension(4) :: limit   
        integer(INT32) :: interval 

        ngauss=12
        two=2.d0

        if(t.eq.0.d0) then
            summa=(b*b-a*a)/two 
        else 

            ! x value for half width
            x_limit_b=dsqrt(dlog(two))/t+p
            x_limit_a=-dsqrt(dlog(two))/t+p

            ! Three main cases can be identified
            if(p.gt.a.and.p.lt.b) then
                if(x_limit_a.gt.a.and.x_limit_b.lt.b) then
                    interval=3
                    limit(1)=a
                    limit(2)=x_limit_a
                    limit(3)=x_limit_b
                    limit(4)=b
                    !         a-> x_limit_a -> x_limit_b -> b 
                else if(x_limit_a.le.a.and.x_limit_b.lt.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=x_limit_b 
                    limit(3)=b
                    !          a -> x_limit_b -> b
                else if(x_limit_a.gt.a.and.x_limit_b.ge.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=x_limit_a 
                    limit(3)=b
                    !          a-> x_limit_a -> b
                else if(x_limit_a.le.a.and.x_limit_b.ge.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=p 
                    limit(3)=b
                    !          a-> p -> b
                end if
            else if(p.le.a) then
                if(x_limit_b.gt.a.and.x_limit_b.lt.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=x_limit_b
                    limit(3)=b
                    !         a -> x_limit_b -> b
                else 
                    interval=2
                    limit(1)=a
                    limit(2)=(a+b)/2.d0
                    limit(3)=b
                    !         a-> (a+b)/2 -> b
                end if
            else if(p.ge.b) then
                if(x_limit_a.gt.a.and.x_limit_a.lt.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=x_limit_a
                    limit(3)=b
                    !         a-> xlimit_a -> b
                else 
                    interval=2
                    limit(1)=a
                    limit(2)=(a+b)/2.d0
                    limit(3)=b
                    !          a -> (a+b)/2 -> b
                end if
            end if

            wk(1) =0.047175336386512D00
            wk(2) =0.106939325995318D00
            wk(3) =0.160078328543346D00
            wk(4) =0.203167426723066D00
            wk(5) =0.233492536538355D00
            wk(6) =0.249147045813403D00
            wk(7) =0.249147045813403D00
            wk(8) =0.233492536538355D00
            wk(9) =0.203167426723066D00
            wk(10)=0.160078328543346D00
            wk(11)=0.106939325995318D00
            wk(12)=0.047175336386512D00

            tk(1) =-0.981560634246719D00
            tk(2) =-0.904117256370475D00
            tk(3) =-0.769902674194305D00
            tk(4) =-0.587317954286617D00
            tk(5) =-0.367831498998180D00
            tk(6) =-0.125233408511469D00
            tk(7) = 0.125233408511469D00
            tk(8) = 0.367831498998180D00
            tk(9) = 0.587317954286617D00
            tk(10)= 0.769902674194305D00
            tk(11)= 0.904117256370475D00
            tk(12)= 0.981560634246719D00
            ! Gauss integration
            summa=0.d0
            do i=1,interval 
                a0=limit(i)
                a1=limit(i+1)
                pr=1.d0-a1*2.d0/(a1-a0)
                qr=2.d0/(a1-a0)
                do j=1,ngauss
                    k=(i-1)*ngauss+j
                    tp=(tk(j)-pr)/qr
                    tw=wk(j)/qr
                    summa=summa+tp*dexp(-t*t*(tp-p)*(tp-p))*tw
                end do
            end do
        end if

        value=summa

    end function gauss_expo_1



    pure function gauss_expo_2(a,b,t,p) result(value)
        !     
        ! Calculate the integral x^2*exp(-t*t*(x-p)^2) dx between a and b
        !                                             where a < b
        ! using Gauss integration
        !
        ! Debug data: t=2, p=1 a=0 b=infinity yield:
        ! 9/32*sqrt(pii)+0.125*exp(-4)+9/32*erf(2)*sqrt(pii)
        ! 30.1.2004

        implicit none
        real(REAL64), intent(in) :: a,b,t,p  
        real(REAL64) :: two,a0,a1,pr,qr,tw,tp,summa,value
        real(REAL64), dimension(12) :: wk,tk
        integer(INT32) :: ngauss,i,j,k

        real(REAL64) :: x_limit_a,x_limit_b
        real(REAL64), dimension(4) :: limit   
        integer(INT32) :: interval 

        ngauss=12
        two=2.d0

        if(t.eq.0.d0) then
            summa=(b*b*b-a*a*a)/3.d0 
        else 

            ! x value for half width
            x_limit_b=dsqrt(dlog(two))/t+p
            x_limit_a=-dsqrt(dlog(two))/t+p

            ! Three main cases can be identified
            if(p.gt.a.and.p.lt.b) then
                if(x_limit_a.gt.a.and.x_limit_b.lt.b) then
                    interval=3
                    limit(1)=a
                    limit(2)=x_limit_a
                    limit(3)=x_limit_b
                    limit(4)=b
                    !         a-> x_limit_a -> x_limit_b -> b 
                else if(x_limit_a.le.a.and.x_limit_b.lt.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=x_limit_b 
                    limit(3)=b
                    !          a -> x_limit_b -> b
                else if(x_limit_a.gt.a.and.x_limit_b.ge.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=x_limit_a 
                    limit(3)=b
                    !          a-> x_limit_a -> b
                else if(x_limit_a.le.a.and.x_limit_b.ge.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=p 
                    limit(3)=b
                    !          a-> p -> b
                end if
            else if(p.le.a) then
                if(x_limit_b.gt.a.and.x_limit_b.lt.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=x_limit_b
                    limit(3)=b
                    !         a -> x_limit_b -> b
                else 
                    interval=2
                    limit(1)=a
                    limit(2)=(a+b)/2.d0
                    limit(3)=b
                    !         a-> (a+b)/2 -> b
                end if
            else if(p.ge.b) then
                if(x_limit_a.gt.a.and.x_limit_a.lt.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=x_limit_a
                    limit(3)=b
                    !         a-> xlimit_a -> b
                else 
                    interval=2
                    limit(1)=a
                    limit(2)=(a+b)/2.d0
                    limit(3)=b
                    !          a -> (a+b)/2 -> b
                end if
            end if

            wk(1) =0.047175336386512D00
            wk(2) =0.106939325995318D00
            wk(3) =0.160078328543346D00
            wk(4) =0.203167426723066D00
            wk(5) =0.233492536538355D00
            wk(6) =0.249147045813403D00
            wk(7) =0.249147045813403D00
            wk(8) =0.233492536538355D00
            wk(9) =0.203167426723066D00
            wk(10)=0.160078328543346D00
            wk(11)=0.106939325995318D00
            wk(12)=0.047175336386512D00

            tk(1) =-0.981560634246719D00
            tk(2) =-0.904117256370475D00
            tk(3) =-0.769902674194305D00
            tk(4) =-0.587317954286617D00
            tk(5) =-0.367831498998180D00
            tk(6) =-0.125233408511469D00
            tk(7) = 0.125233408511469D00
            tk(8) = 0.367831498998180D00
            tk(9) = 0.587317954286617D00
            tk(10)= 0.769902674194305D00
            tk(11)= 0.904117256370475D00
            tk(12)= 0.981560634246719D00
            ! Gauss integration
            summa=0.d0
            do i=1,interval 
                a0=limit(i)
                a1=limit(i+1)
                pr=1.d0-a1*2.d0/(a1-a0)
                qr=2.d0/(a1-a0)
                do j=1,ngauss
                    k=(i-1)*ngauss+j
                    tp=(tk(j)-pr)/qr
                    tw=wk(j)/qr
                    summa=summa+tp*tp*dexp(-t*t*(tp-p)*(tp-p))*tw
                end do
            end do
        end if

        value=summa

    end function gauss_expo_2

    pure function gauss_expo_3(a,b,t,p) result(value)
        !     
        ! Calculate the integral x^3*exp(-t*t*(x-p)^2) dx between a and b
        !                                             where a < b
        ! using Gauss integration
        !
        ! Debug data: t=2, p=1 a=0 b=infinity yield:
        ! 11/32*sqrt(pii)+5/32*exp(-4)+11/32*erf(2)*sqrt(pii)
        ! 30.1.2004

        implicit none
        real(REAL64), intent(in) :: a,b,t,p  
        real(REAL64) :: two,a0,a1,pr,qr,tw,tp,summa,value
        real(REAL64), dimension(12) :: wk,tk
        integer(INT32) :: ngauss,i,j,k

        real(REAL64) :: x_limit_a,x_limit_b
        real(REAL64), dimension(4) :: limit   
        integer(INT32) :: interval 

        ngauss=12
        two=2.d0

        if(t.eq.0.d0) then
            summa=(b**4-a**4)/4.d0 
        else 

            ! x value for half width
            x_limit_b=dsqrt(dlog(two))/t+p
            x_limit_a=-dsqrt(dlog(two))/t+p

            ! Three main cases can be identified
            if(p.gt.a.and.p.lt.b) then
                if(x_limit_a.gt.a.and.x_limit_b.lt.b) then
                    interval=3
                    limit(1)=a
                    limit(2)=x_limit_a
                    limit(3)=x_limit_b
                    limit(4)=b
                    !         a-> x_limit_a -> x_limit_b -> b 
                else if(x_limit_a.le.a.and.x_limit_b.lt.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=x_limit_b 
                    limit(3)=b
                    !          a -> x_limit_b -> b
                else if(x_limit_a.gt.a.and.x_limit_b.ge.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=x_limit_a 
                    limit(3)=b
                    !          a-> x_limit_a -> b
                else if(x_limit_a.le.a.and.x_limit_b.ge.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=p 
                    limit(3)=b
                    !          a-> p -> b
                end if
            else if(p.le.a) then
                if(x_limit_b.gt.a.and.x_limit_b.lt.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=x_limit_b
                    limit(3)=b
                    !         a -> x_limit_b -> b
                else 
                    interval=2
                    limit(1)=a
                    limit(2)=(a+b)/2.d0
                    limit(3)=b
                    !         a-> (a+b)/2 -> b
                end if
            else if(p.ge.b) then
                if(x_limit_a.gt.a.and.x_limit_a.lt.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=x_limit_a
                    limit(3)=b
                    !         a-> xlimit_a -> b
                else 
                    interval=2
                    limit(1)=a
                    limit(2)=(a+b)/2.d0
                    limit(3)=b
                    !          a -> (a+b)/2 -> b
                end if
            end if

            wk(1) =0.047175336386512D00
            wk(2) =0.106939325995318D00
            wk(3) =0.160078328543346D00
            wk(4) =0.203167426723066D00
            wk(5) =0.233492536538355D00
            wk(6) =0.249147045813403D00
            wk(7) =0.249147045813403D00
            wk(8) =0.233492536538355D00
            wk(9) =0.203167426723066D00
            wk(10)=0.160078328543346D00
            wk(11)=0.106939325995318D00
            wk(12)=0.047175336386512D00

            tk(1) =-0.981560634246719D00
            tk(2) =-0.904117256370475D00
            tk(3) =-0.769902674194305D00
            tk(4) =-0.587317954286617D00
            tk(5) =-0.367831498998180D00
            tk(6) =-0.125233408511469D00
            tk(7) = 0.125233408511469D00
            tk(8) = 0.367831498998180D00
            tk(9) = 0.587317954286617D00
            tk(10)= 0.769902674194305D00
            tk(11)= 0.904117256370475D00
            tk(12)= 0.981560634246719D00
            ! Gauss integration
            summa=0.d0
            do i=1,interval 
                a0=limit(i)
                a1=limit(i+1)
                pr=1.d0-a1*2.d0/(a1-a0)
                qr=2.d0/(a1-a0)
                do j=1,ngauss
                    k=(i-1)*ngauss+j
                    tp=(tk(j)-pr)/qr
                    tw=wk(j)/qr
                    summa=summa+tp**3*dexp(-t*t*(tp-p)*(tp-p))*tw
                end do
            end do
        end if

        value=summa

    end function gauss_expo_3

    pure function gauss_expo_4(a,b,t,p) result(value)
        !     
        ! Calculate the integral x^4*exp(-t*t*(x-p)^2) dx between a and b
        !                                             where a < b
        ! using Gauss integration
        !
        ! Debug data: t=2, p=1 a=0 b=infinity yield:
        ! 115/256*sqrt(pii)+13/64*exp(-4)+115/256*erf(2)*sqrt(pii)
        ! 30.1.2004

        implicit none
        real(REAL64), intent(in) :: a,b,t,p  
        real(REAL64) :: two,a0,a1,pr,qr,tw,tp,summa,value
        real(REAL64), dimension(12) :: wk,tk
        integer(INT32) :: ngauss,i,j,k

        real(REAL64) :: x_limit_a,x_limit_b
        real(REAL64), dimension(4) :: limit   
        integer(INT32) :: interval 

        ngauss=12
        two=2.d0

        if(t.eq.0.d0) then
            summa=(b**5-a**5)/5.d0 
        else 

            ! x value for half width
            x_limit_b=dsqrt(dlog(two))/t+p
            x_limit_a=-dsqrt(dlog(two))/t+p

            ! Three main cases can be identified
            if(p.gt.a.and.p.lt.b) then
                if(x_limit_a.gt.a.and.x_limit_b.lt.b) then
                    interval=3
                    limit(1)=a
                    limit(2)=x_limit_a
                    limit(3)=x_limit_b
                    limit(4)=b
                    !         a-> x_limit_a -> x_limit_b -> b 
                else if(x_limit_a.le.a.and.x_limit_b.lt.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=x_limit_b 
                    limit(3)=b
                    !          a -> x_limit_b -> b
                else if(x_limit_a.gt.a.and.x_limit_b.ge.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=x_limit_a 
                    limit(3)=b
                    !          a-> x_limit_a -> b
                else if(x_limit_a.le.a.and.x_limit_b.ge.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=p 
                    limit(3)=b
                    !          a-> p -> b
                end if
            else if(p.le.a) then
                if(x_limit_b.gt.a.and.x_limit_b.lt.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=x_limit_b
                    limit(3)=b
                    !         a -> x_limit_b -> b
                else 
                    interval=2
                    limit(1)=a
                    limit(2)=(a+b)/2.d0
                    limit(3)=b
                    !         a-> (a+b)/2 -> b
                end if
            else if(p.ge.b) then
                if(x_limit_a.gt.a.and.x_limit_a.lt.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=x_limit_a
                    limit(3)=b
                    !         a-> xlimit_a -> b
                else 
                    interval=2
                    limit(1)=a
                    limit(2)=(a+b)/2.d0
                    limit(3)=b
                    !          a -> (a+b)/2 -> b
                end if
            end if

            wk(1) =0.047175336386512D00
            wk(2) =0.106939325995318D00
            wk(3) =0.160078328543346D00
            wk(4) =0.203167426723066D00
            wk(5) =0.233492536538355D00
            wk(6) =0.249147045813403D00
            wk(7) =0.249147045813403D00
            wk(8) =0.233492536538355D00
            wk(9) =0.203167426723066D00
            wk(10)=0.160078328543346D00
            wk(11)=0.106939325995318D00
            wk(12)=0.047175336386512D00

            tk(1) =-0.981560634246719D00
            tk(2) =-0.904117256370475D00
            tk(3) =-0.769902674194305D00
            tk(4) =-0.587317954286617D00
            tk(5) =-0.367831498998180D00
            tk(6) =-0.125233408511469D00
            tk(7) = 0.125233408511469D00
            tk(8) = 0.367831498998180D00
            tk(9) = 0.587317954286617D00
            tk(10)= 0.769902674194305D00
            tk(11)= 0.904117256370475D00
            tk(12)= 0.981560634246719D00
            ! Gauss integration
            summa=0.d0
            do i=1,interval 
                a0=limit(i)
                a1=limit(i+1)
                pr=1.d0-a1*2.d0/(a1-a0)
                qr=2.d0/(a1-a0)
                do j=1,ngauss
                    k=(i-1)*ngauss+j
                    tp=(tk(j)-pr)/qr
                    tw=wk(j)/qr
                    summa=summa+tp**4*dexp(-t*t*(tp-p)*(tp-p))*tw
                end do
            end do
        end if

        value=summa

    end function gauss_expo_4

    pure function gauss_expo_5(a,b,t,p) result(value)
        !     
        ! Calculate the integral x^5*exp(-t*t*(x-p)^2) dx between a and b
        !                                             where a < b
        ! using Gauss integration
        !
        ! Debug data: t=2, p=1 a=0 b=infinity yield:
        ! 159/256*sqrt(pii)+9/32*exp(-4)+159/256*erf(2)*sqrt(pii)
        ! 30.1.2004

        implicit none
        real(REAL64), intent(in) :: a,b,t,p  
        real(REAL64) :: two,a0,a1,pr,qr,tw,tp,summa,value
        real(REAL64), dimension(12) :: wk,tk
        integer(INT32) :: ngauss,i,j,k

        real(REAL64) :: x_limit_a,x_limit_b
        real(REAL64), dimension(4) :: limit   
        integer(INT32) :: interval 

        ngauss=12
        two=2.d0

        if(t.eq.0.d0) then
            summa=(b**6-a**6)/6.d0 
        else 

            ! x value for half width
            x_limit_b=dsqrt(dlog(two))/t+p
            x_limit_a=-dsqrt(dlog(two))/t+p

            ! Three main cases can be identified
            if(p.gt.a.and.p.lt.b) then
                if(x_limit_a.gt.a.and.x_limit_b.lt.b) then
                    interval=3
                    limit(1)=a
                    limit(2)=x_limit_a
                    limit(3)=x_limit_b
                    limit(4)=b
                    !         a-> x_limit_a -> x_limit_b -> b 
                else if(x_limit_a.le.a.and.x_limit_b.lt.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=x_limit_b 
                    limit(3)=b
                    !          a -> x_limit_b -> b
                else if(x_limit_a.gt.a.and.x_limit_b.ge.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=x_limit_a 
                    limit(3)=b
                    !          a-> x_limit_a -> b
                else if(x_limit_a.le.a.and.x_limit_b.ge.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=p 
                    limit(3)=b
                    !          a-> p -> b
                end if
            else if(p.le.a) then
                if(x_limit_b.gt.a.and.x_limit_b.lt.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=x_limit_b
                    limit(3)=b
                    !         a -> x_limit_b -> b
                else 
                    interval=2
                    limit(1)=a
                    limit(2)=(a+b)/2.d0
                    limit(3)=b
                    !         a-> (a+b)/2 -> b
                end if
            else if(p.ge.b) then
                if(x_limit_a.gt.a.and.x_limit_a.lt.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=x_limit_a
                    limit(3)=b
                    !         a-> xlimit_a -> b
                else 
                    interval=2
                    limit(1)=a
                    limit(2)=(a+b)/2.d0
                    limit(3)=b
                    !          a -> (a+b)/2 -> b
                end if
            end if

            wk(1) =0.047175336386512D00
            wk(2) =0.106939325995318D00
            wk(3) =0.160078328543346D00
            wk(4) =0.203167426723066D00
            wk(5) =0.233492536538355D00
            wk(6) =0.249147045813403D00
            wk(7) =0.249147045813403D00
            wk(8) =0.233492536538355D00
            wk(9) =0.203167426723066D00
            wk(10)=0.160078328543346D00
            wk(11)=0.106939325995318D00
            wk(12)=0.047175336386512D00

            tk(1) =-0.981560634246719D00
            tk(2) =-0.904117256370475D00
            tk(3) =-0.769902674194305D00
            tk(4) =-0.587317954286617D00
            tk(5) =-0.367831498998180D00
            tk(6) =-0.125233408511469D00
            tk(7) = 0.125233408511469D00
            tk(8) = 0.367831498998180D00
            tk(9) = 0.587317954286617D00
            tk(10)= 0.769902674194305D00
            tk(11)= 0.904117256370475D00
            tk(12)= 0.981560634246719D00
            ! Gauss integration
            summa=0.d0
            do i=1,interval 
                a0=limit(i)
                a1=limit(i+1)
                pr=1.d0-a1*2.d0/(a1-a0)
                qr=2.d0/(a1-a0)
                do j=1,ngauss
                    k=(i-1)*ngauss+j
                    tp=(tk(j)-pr)/qr
                    tw=wk(j)/qr
                    summa=summa+tp**5*dexp(-t*t*(tp-p)*(tp-p))*tw
                end do
            end do
        end if

        value=summa

    end function gauss_expo_5

    pure function gauss_expo_6(a,b,t,p) result(value)
        !     
        ! Calculate the integral x^6*exp(-t*t*(x-p)^2) dx between a and b
        !                                             where a < b
        ! using Gauss integration
        !
        ! Debug data: t=2, p=1 a=0 b=infinity yield:
        ! 1847/2048*sqrt(pii)+209/512*exp(-4)+1847/2048*erf(2)*sqrt(pii)
        ! 30.1.2004

        implicit none
        real(REAL64), intent(in) :: a,b,t,p
        real(REAL64) :: two,a0,a1,pr,qr,tw,tp,summa,value  
        real(REAL64), dimension(12) :: wk,tk
        integer(INT32) :: ngauss,i,j,k

        real(REAL64) :: x_limit_a,x_limit_b
        real(REAL64), dimension(4) :: limit   
        integer(INT32) :: interval 

        ngauss=12
        two=2.d0

        if(t.eq.0.d0) then
            summa=(b**7-a**7)/7.d0 
        else 

            ! x value for half width
            x_limit_b=dsqrt(dlog(two))/t+p
            x_limit_a=-dsqrt(dlog(two))/t+p

            ! Three main cases can be identified
            if(p.gt.a.and.p.lt.b) then
                if(x_limit_a.gt.a.and.x_limit_b.lt.b) then
                    interval=3
                    limit(1)=a
                    limit(2)=x_limit_a
                    limit(3)=x_limit_b
                    limit(4)=b
                    !         a-> x_limit_a -> x_limit_b -> b 
                else if(x_limit_a.le.a.and.x_limit_b.lt.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=x_limit_b 
                    limit(3)=b
                    !          a -> x_limit_b -> b
                else if(x_limit_a.gt.a.and.x_limit_b.ge.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=x_limit_a 
                    limit(3)=b
                    !          a-> x_limit_a -> b
                else if(x_limit_a.le.a.and.x_limit_b.ge.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=p 
                    limit(3)=b
                    !          a-> p -> b
                end if
            else if(p.le.a) then
                if(x_limit_b.gt.a.and.x_limit_b.lt.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=x_limit_b
                    limit(3)=b
                    !         a -> x_limit_b -> b
                else 
                    interval=2
                    limit(1)=a
                    limit(2)=(a+b)/2.d0
                    limit(3)=b
                    !         a-> (a+b)/2 -> b
                end if
            else if(p.ge.b) then
                if(x_limit_a.gt.a.and.x_limit_a.lt.b) then
                    interval=2
                    limit(1)=a
                    limit(2)=x_limit_a
                    limit(3)=b
                    !         a-> xlimit_a -> b
                else 
                    interval=2
                    limit(1)=a
                    limit(2)=(a+b)/2.d0
                    limit(3)=b
                    !          a -> (a+b)/2 -> b
                end if
            end if

            wk(1) =0.047175336386512D00
            wk(2) =0.106939325995318D00
            wk(3) =0.160078328543346D00
            wk(4) =0.203167426723066D00
            wk(5) =0.233492536538355D00
            wk(6) =0.249147045813403D00
            wk(7) =0.249147045813403D00
            wk(8) =0.233492536538355D00
            wk(9) =0.203167426723066D00
            wk(10)=0.160078328543346D00
            wk(11)=0.106939325995318D00
            wk(12)=0.047175336386512D00

            tk(1) =-0.981560634246719D00
            tk(2) =-0.904117256370475D00
            tk(3) =-0.769902674194305D00
            tk(4) =-0.587317954286617D00
            tk(5) =-0.367831498998180D00
            tk(6) =-0.125233408511469D00
            tk(7) = 0.125233408511469D00
            tk(8) = 0.367831498998180D00
            tk(9) = 0.587317954286617D00
            tk(10)= 0.769902674194305D00
            tk(11)= 0.904117256370475D00
            tk(12)= 0.981560634246719D00
            ! Gauss integration
            summa=0.d0
            do i=1,interval 
                a0=limit(i)
                a1=limit(i+1)
                pr=1.d0-a1*2.d0/(a1-a0)
                qr=2.d0/(a1-a0)
                do j=1,ngauss
                    k=(i-1)*ngauss+j
                    tp=(tk(j)-pr)/qr
                    tw=wk(j)/qr
                    summa=summa+tp**6*dexp(-t*t*(tp-p)*(tp-p))*tw
                end do
            end do
        end if

        value=summa

    end function gauss_expo_6

    pure function gauss_expo_n(a,b,t,p,n) result(value)
        !     
        ! Calculate the integral x^2*exp(-t*t*(x-p)^2) dx between a and b
        !                                             where a < b
        ! using Gauss integration

        implicit none
        integer(INT32), intent(in) :: n 
        real(REAL64), intent(in) :: a,b,t,p 
        real(REAL64) :: two,a0,a1,pr,qr,tw,tp,summa, value 
        real(REAL64), dimension(12) :: wk,tk
        integer(INT32) :: ngauss,i,j,k, interval

        real(REAL64), dimension(5) :: limit   
        

        ngauss=12
        two=2.d0

        if(t.eq.0.d0) then
            if(n.eq.0) then
                summa=b-a
            else
                summa=(b**(n+1)-a**(n+1))/(n+1)
            end if
        else 

            ! x value for half width
            !      x_limit_b=dsqrt(dlog(two))/t+p
            !      x_limit_a=-dsqrt(dlog(two))/t+p

            ! Three main cases can be identified
            !      if(p.gt.a.and.p.lt.b) then
            !        if(x_limit_a.gt.a.and.x_limit_b.lt.b) then
            !          interval=3
            !          limit(1)=a
            !          limit(2)=x_limit_a
            !          limit(3)=x_limit_b
            !          limit(4)=b
            !         a-> x_limit_a -> x_limit_b -> b 
            !        else if(x_limit_a.le.a.and.x_limit_b.lt.b) then
            !           interval=2
            !           limit(1)=a
            !           limit(2)=x_limit_b 
            !           limit(3)=b
            !          a -> x_limit_b -> b
            !        else if(x_limit_a.gt.a.and.x_limit_b.ge.b) then
            !           interval=2
            !           limit(1)=a
            !           limit(2)=x_limit_a 
            !           limit(3)=b
            !          a-> x_limit_a -> b
            !        else if(x_limit_a.le.a.and.x_limit_b.ge.b) then
            !           interval=2
            !           limit(1)=a
            !           limit(2)=p 
            !           limit(3)=b
            !          a-> p -> b
            !        end if
            !      else if(p.le.a) then
            !        if(x_limit_b.gt.a.and.x_limit_b.lt.b) then
            !          interval=2
            !          limit(1)=a
            !          limit(2)=x_limit_b
            !          limit(3)=b
            !         a -> x_limit_b -> b
            !        else 
            !          interval=2
            !          limit(1)=a
            !          limit(2)=(a+b)/2.d0
            !          limit(3)=b
            !         a-> (a+b)/2 -> b
            !        end if
            !      else if(p.ge.b) then
            !        if(x_limit_a.gt.a.and.x_limit_a.lt.b) then
            !          interval=2
            !          limit(1)=a
            !          limit(2)=x_limit_a
            !          limit(3)=b
            !         a-> xlimit_a -> b
            !        else 
            !          interval=2
            !          limit(1)=a
            !          limit(2)=(a+b)/2.d0
            !          limit(3)=b
            !          a -> (a+b)/2 -> b
            !        end if
            !      end if

            interval=4
            do i=1,interval+1
                limit(i)=((interval+1-i)*a+(i-1)*b)/interval
            end do

            wk(1) =0.047175336386512D00
            wk(2) =0.106939325995318D00
            wk(3) =0.160078328543346D00
            wk(4) =0.203167426723066D00
            wk(5) =0.233492536538355D00
            wk(6) =0.249147045813403D00
            wk(7) =0.249147045813403D00
            wk(8) =0.233492536538355D00
            wk(9) =0.203167426723066D00
            wk(10)=0.160078328543346D00
            wk(11)=0.106939325995318D00
            wk(12)=0.047175336386512D00

            tk(1) =-0.981560634246719D00
            tk(2) =-0.904117256370475D00
            tk(3) =-0.769902674194305D00
            tk(4) =-0.587317954286617D00
            tk(5) =-0.367831498998180D00
            tk(6) =-0.125233408511469D00
            tk(7) = 0.125233408511469D00
            tk(8) = 0.367831498998180D00
            tk(9) = 0.587317954286617D00
            tk(10)= 0.769902674194305D00
            tk(11)= 0.904117256370475D00
            tk(12)= 0.981560634246719D00

            ! Gauss integration
            summa=0.d0
            do i=1,interval 
                a0=limit(i)
                a1=limit(i+1)
                pr=1.d0-a1*2.d0/(a1-a0)
                qr=2.d0/(a1-a0)
                do j=1,ngauss
                    k=(i-1)*ngauss+j
                    tp=(tk(j)-pr)/qr
                    tw=wk(j)/qr
                    if(n.gt.0) then
                        summa=summa+tp**n*dexp(-t*t*(tp-p)*(tp-p))*tw
                    else
                        summa=summa+dexp(-t*t*(tp-p)*(tp-p))*tw
                    end if
                end do
            end do
        end if

        value=summa

    end function gauss_expo_n

    pure function recursint(a,b,t,d,n) result(ints)
        ! Calculates the integral
        ! \int_{a}^{b} x^{i} e^{-t^{2}(x-d)^{2}}dx
        ! using a recursive formula
        real(REAL64), intent(in) :: a,b,t,d
        integer(INT32), intent(in) :: n

        integer(INT32) :: i
        
        real(REAL64) :: tf,ap,bp,ea,eb, ints(n)
        real(REAL64), parameter :: fac=0.88622692545275801364D0 !sqrt(pi)/2
        real(REAL64) :: erf

        ap=t*(a-d);bp=t*(b-d)
        ea=exp(-ap*ap);eb=exp(-bp*bp)

!        bin_coeffs=init_bin_coeffs(n,d*t)

        ints(1)=fac/t*(erf(bp)-erf(ap))
        tf=0.5d0/t/t
        ints(2)=d*ints(1)+tf*(ea-eb)
        do i=3,n
            ints(i)=d*ints(i-1)+tf*((i-2)*ints(i-2)+a**(i-2)*ea-b**(i-2)*eb)
        end do

    end function

    !> Make interpolation matrix

    !> Return a matrix \f$\mathbf{M}^{(k)}\f$ with elements
    !! \f$M^{(k)}_{ij}=\left(\frac{\mathrm{d}}{\mathrm{d}x}\right)^k\chi_i(x)\Big|_{x=x_j}\f$
    !!
    !> @todo Remove this from here
    function interp_matrix(grid_in, grid_out, crd, dv, trsp) result(matrix)
        type(Grid3D) :: grid_in, grid_out
        integer, intent(in) :: crd
        integer, optional :: dv
        character, optional :: trsp

        real(REAL64), allocatable :: matrix(:,:)

        type(LIPBasis) :: lip
        type(PBC),pointer :: pbc_ptr
        integer   :: der
        logical :: do_trsp

        real(REAL64),allocatable :: temp_mat(:),x0
        integer :: ix2,icell, nlip,j
        integer :: idisp
        real(REAL64),pointer :: points_out(:), cellh_in(:)
        type(REAL64_2D),allocatable :: lip_coeffs(:)
        real(REAL64) :: qmin_in, total_size_in

        real(REAL64),pointer :: celld_in(:)

        integer :: i

        do_trsp=( present(trsp) .and. (trsp=='t'.or.trsp=='T'))

        ! a is the density, a2 is  the potential

        der=0
        if (present(dv)) then
            der = dv
        else
            der = 0
        end if

        if(do_trsp) then
            allocate(matrix(grid_in %axis(crd)%get_shape(), &
                            grid_out%axis(crd)%get_shape()))
        else
            allocate(matrix(grid_out%axis(crd)%get_shape(), &
                            grid_in %axis(crd)%get_shape()))
        end if
                               
        ! get lip and pbc
        lip=grid_in%get_lip()

        nlip = grid_in%get_nlip()
        allocate(temp_mat(nlip))
        temp_mat=0.d0
        pbc_ptr=>grid_in%get_pbc()

        ! reset them here
        matrix = 0.d0

        ! We do the interpolation point by point
        points_out => grid_out%axis(crd)%get_coord()
        qmin_in       =  grid_in%axis(crd)%get_qmin()
        total_size_in =  grid_in%axis(crd)%get_delta()
        cellh_in      => grid_in%axis(crd)%get_cell_scales()

        celld_in      => grid_in%axis(crd)%get_cell_starts()

        lip_coeffs=lip%coeffs(der)

        do ix2=1, size(points_out)
            ! Find in which cell of the density grid (grid_in) we are
            ! If we have pbc's, we have to take them into account
            idisp=floor((points_out(ix2)-qmin_in)/total_size_in)
            if (ix2 == size(points_out)) then
                
                ! Find the density cell
                icell=grid_in%axis(crd)%get_icell(points_out(ix2)-0.0000001d0)
                ! Transform into cell units
                x0=grid_in%axis(crd)%x2cell(points_out(ix2)-0.0000001d0)
                ! Interpolate polynomials at that point
                temp_mat(:)=eval_polys( lip_coeffs(der+1)%p, x0)
                ! Divide by h**-der
                if(der>0) &
                    temp_mat = temp_mat * cellh_in(icell)**(-der)
                
            else if(pbc_ptr%contains(idisp,crd)) then
                ! Find the density cell
                icell=grid_in%axis(crd)%get_icell(points_out(ix2))
                ! Transform into cell units
                x0=grid_in%axis(crd)%x2cell(points_out(ix2))
                ! Interpolate polynomials at that point
                temp_mat(:)=eval_polys( lip_coeffs(der+1)%p, x0)
                ! Divide by h**-der
                if(der>0) &
                    temp_mat = temp_mat * cellh_in(icell)**(-der)
            else
                print *, "outside: ix2", ix2
                ! Otherwise, we are OUT of the defined density, therefore the
                ! density does not contribute at this point
                cycle
            end if

            ! Now we simply put the values into the right place
            if (do_trsp) then
                matrix((icell-1)*(nlip-1)+1:icell*(nlip-1)+1,ix2)=&
                matrix((icell-1)*(nlip-1)+1:icell*(nlip-1)+1,ix2)+temp_mat
            else
                matrix(ix2,(icell-1)*(nlip-1)+1:icell*(nlip-1)+1)=&
                matrix(ix2,(icell-1)*(nlip-1)+1:icell*(nlip-1)+1)+temp_mat
            end if
        end do

        deallocate(temp_mat)

        do i=1,der+1
            deallocate(lip_coeffs(i)%p)
        end do

        deallocate(lip_coeffs)

        return
    end function

end module
