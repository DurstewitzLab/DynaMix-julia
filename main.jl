using MKL, LinearAlgebra
BLAS.set_num_threads(2)

using DSR
ENV["GKSwstype"] = "nul"
main_routine(parse_commandline())