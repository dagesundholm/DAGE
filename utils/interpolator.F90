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
program interpolator

    ! Reads a previously generated potential, and a set of vectors (1, 2 or 3)
    ! (same as $pointval in turbomole) defining a grid.
    ! It gives back the potential values at those grid points in standard input
    ! x y z f(x,y,z)
    ! If -rotate is given, in the output the axes are rotated so that x is parallel
    ! to the first vector, y is orthogonal to the first vector, in the planed formed
    ! by the first and second vectors, and z is orthogonal to x and y, pointing to
    ! the same side as the third vector. In this mode, only as many coordinates
    ! as given vectors are printed out. This allows an easier plotting.

      use globals_m
      use LIPBasis_class
      use grid_class
      use Function3D_class
      implicit none

      type(Function3D), target :: func
      type(Grid3D), pointer :: grid
      real(REAL64), pointer :: cube(:,:,:)

      type vector_t
          real(REAL64) :: dir(3),ran(2)
          integer(INT32) :: np
      end type

      type(vector_t) :: vector(3)
      real(REAL64), allocatable :: points(:,:), f(:)

      character(len=250) :: line(4),buffer,func_orig,out_fname, fmtstring

      integer(INT32) :: i,j,nvector=0,prevdim,nptot=1,gdims(3),nlip
      logical :: rotate_p=.TRUE.
      logical :: add_bubbles_p=.TRUE.
      logical :: add_cube_p=.TRUE.
      real(REAL64) :: lsq,orig(3)=0.d0
      real(REAL64) :: box(2,3)

      ! Variables used for axis rotation
      real(REAL64) :: revectors(3,3),recoords(3)

      verbo_g=1

      ! Opening input file and initializing stuff
      if(iargc()<2) then
          print*,&
             'interpolator.x <infile> <outfile> [--no-rotate] [--no-bubbles]'
          stop
      else
          call getarg(1,func_orig)
          call getarg(2,out_fname)
          call get_command(buffer)
          if (index(buffer,"--no-rotate")>0) rotate_p=.FALSE.
          if (index(buffer,"--no-bubbles")>0) add_bubbles_p=.FALSE.
          if (index(buffer,"--no-cube")>0) add_cube_p=.FALSE.
      end if

      func = Function3D(func_orig)

      grid => func%grid
      cube => func%get_cube()
      ! For the moment nlip=7...
      nlip=7
  
      ! Calculating box size
      ! CAREFUL: ONLY EQUAL BOXES
  !    call getkw(input, 'density.ncells', gdims)
      gdims(:)=grid%get_shape()
      box(:,:)=grid%get_range()
  
      print*,'Box read from file ',func_orig
      print*,'Points    :',gdims(1),'x',gdims(2),'x',gdims(3)
      print*,'Range (au):',box(:,1),'x'
      print*,'           ',box(:,2),'x'
      print*,'           ',box(:,3)
      print*
      ! Read grid vectors into strings first
      print*,'Input vectors in the following format'
      print*,'Direction (real)    Range (real)  Number of points (integer)'
      print*,' x    y    z         l1     l2               n'
  
      i=1
      do
          read(*,'(a)') line(i)
          if(line(i)(1:4)=='orig') then
              read(line(i)(5:),*) orig
              cycle
          end if
          if(trim(line(i))=='') exit
          nvector=nvector+1
          i=i+1
      end do
      if(nvector==0) then
          stop 'No vectors?'
      end if
  
      ! Read and normalize from strings
      do i=1,nvector
          read(line(i),*) vector(i)%dir,vector(i)%ran(1),vector(i)%ran(2),vector(i)%np
          lsq=sum(vector(i)%dir*vector(i)%dir)
          vector(i)%dir=vector(i)%dir*(vector(i)%ran(2)-vector(i)%ran(1))/(vector(i)%np-1)/sqrt(lsq)
      end do
  
      ! Set unused vectors to 0
      do i=nvector+1,3
          vector(i)%dir=0.d0
          vector(i)%ran(1)=0.d0
          vector(i)%ran(2)=0.d0
          vector(i)%np=0
      end do
      
      ! Calculate total number of points and allocate
      do i=1,nvector
          nptot=nptot*vector(i)%np
      end do
      allocate(points(3,nptot))
      allocate(f(nptot))
  
      ! Calculate first point
      points(:,1)=orig
  
      do i=1,nvector
          points(:,1)=points(:,1)+&
              (vector(i)%np-1)/(vector(i)%ran(2)/vector(i)%ran(1)-1.d0)*vector(i)%dir
      end do
  
      j=1
      prevdim=1
  
      ! Copy point, first in 1d, then that line in 2d to get a plane, then that plane into 3d
      do i=1,nvector
          do while(j<prevdim*vector(i)%np)
              j=j+1
              points(:,j)=points(:,j-prevdim)+vector(i)%dir
          end do
          prevdim=prevdim*vector(i)%np
      end do
  
      if(add_cube_p) then
          f=func%evaluate(points,add_bubbles_p)
      else
          f=func%bubbles%eval(points,[0,0,0])
      end if
  
      if(rotate_p) then
          ! We have to rotate the axes.
          ! The new X axis is parallel to the first vector (normalize it)
          revectors(:,1)=vector(1)%dir/sqrt(sum(vector(1)%dir*vector(1)%dir))
          if (nvector>1) then
              ! The second vector is orthogonal to the first one
              revectors(:,2)=vector(2)%dir&
                            -sum(vector(2)%dir*revectors(:,1))*revectors(:,1)
              ! And renormalize
              revectors(:,2)=revectors(:,2)/sqrt(sum(revectors(:,2)*revectors(:,2)))
              if (nvector>3) then
                  ! The third vector is orthogonal to the first and second ones
                  ! v=a x b (cross product)
                  revectors(1,3)=revectors(2,1)*revectors(3,2)-revectors(3,1)*revectors(2,2)
                  revectors(2,3)=revectors(3,1)*revectors(1,2)-revectors(1,1)*revectors(3,2)
                  revectors(3,3)=revectors(1,1)*revectors(2,2)-revectors(2,1)*revectors(1,2)
                  ! And reorient respect to the last vector
                  if (sum(vector(2)%dir*revectors(:,3))<0.d0) revectors(:,3)=-revectors(:,3)
              end if
          end if
      end if

      call pinfo("Writing interpolation results in "//trim(out_fname))
      open(1,file=out_fname)
      write(1,'("#",a)') 'Data interpolated from '//trim(func_orig)
      write(1,'("# origin : ",3f10.6)') orig
      write(1,'("#",a)') repeat("_",60)
      write(1,'("#",a)') ' Vector | Number of points |  Displacement vector (a.u.)'
      do i=1,nvector
          write(1,'("#",3x,i1,4x,"|",6x,i6,6x,"|"3f10.6)')&
                 i,vector(i)%np,vector(i)%dir
      end do
      write(1,'("#",a)') repeat("_",60)
      if(rotate_p) then
          select case(nvector)
              case(1)
                  ! Print line
                  write(1,'("#",a)') '    u (a.u.)           f'
                  fmtstring='(2e16.8)'
              case(2)
                  ! Print plane
                  write(1,'("#",a)') '    u (a.u.)        v (a.u.)           f'
                  fmtstring='(3e16.8)'
              case(3)
                  ! Print cube
                  write(1,'("#",a)') '    u (a.u.)        v (a.u.)        w (a.u.)           f'
                  fmtstring='(4e16.8)'
          end select
          do i=1,nptot
              forall (j=1:nvector) recoords(j)=sum((points(:,i)-orig)*revectors(:,j))
              write(1,fmtstring) recoords(1:nvector),f(i)
              if(mod(i,vector(1)%np)==0) write(1,*) ''
          end do
      else
          ! If no rotation of the axes is required, simply print coordinates and values
          write(1,'("#",a)') '    x (a.u.)        y (a.u.)        z (a.u.)           f'
          fmtstring='(4e16.8)'
          do i=1,nptot
              write(1,fmtstring) points(:,i),f(i)
              if(mod(i,vector(1)%np)==0) write(1,*) ''
          end do
      end if
end program

