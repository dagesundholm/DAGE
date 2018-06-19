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
program combine
    use Globals_m
    use bubbles_class
    use Function3D_class
    implicit none

    type(Function3D) :: func   !, func_test
    character(len=200) :: cube_name,bubblib_name,output_name

    if(iargc()<3) then
        print *, 'Usage combine.x <input cube file>  <input bubblib file> &
                  <output file>'
        stop
    else
        call getarg(1,cube_name)
        call getarg(2,bubblib_name)
        call getarg(3,output_name)
        
        if(filext(trim(cube_name)).ne.'cub') then
            print*, 'the first input file must be the type of cub'
            stop
        end if

        if(filext(trim(bubblib_name)).ne.'dat') then
            print*, 'the second bubblib input file must be in the form of &
                     type dat'
            stop
        end if

        call func%combine(trim(bubblib_name),trim(cube_name))
        call func%dump(trim(output_name))

    end if
    return
    
end program combine
