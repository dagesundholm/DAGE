

module gpu_info
    use globals_m

#ifdef HAVE_CUDA
    public gpu_print_info_short
    public gpu_print_info_long

    interface
        subroutine gpu_print_info_short() bind(C)
            use ISO_C_BINDING
        end subroutine
    end interface

    interface
        subroutine gpu_print_info_long() bind(C)
            use ISO_C_BINDING
        end subroutine
    end interface

#endif


end module
