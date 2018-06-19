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
module MemoryLeakChecker_m
    implicit none
    public check_memory_status_cuda
    public start_memory_follower, stop_memory_follower

    private

contains
    subroutine check_memory_status_cuda()
#ifdef HAVE_CUDA
        call check_memory_cuda()
#endif
    end subroutine

    subroutine start_memory_follower()
#ifdef HAVE_CUDA
#ifdef DEBUG_MEMORY_LEAKS
        call memoryfollower_init()
#endif
#endif
    end subroutine

    subroutine stop_memory_follower()
#ifdef HAVE_CUDA
#ifdef DEBUG_MEMORY_LEAKS
        call memoryfollower_destroy()
#endif
#endif
    end subroutine

end module
