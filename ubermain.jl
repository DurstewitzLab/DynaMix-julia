using Distributed
using ArgParse

@everywhere using LinearAlgebra; BLAS.set_num_threads(1)

function parse_ubermain()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--procs", "-p"
        help = "Number of parallel processes/workers to spawn."
        arg_type = Int
        default = 1

        "--runs", "-r"
        help = "Number of runs per experiment setting."
        arg_type = Int
        default = 5
    end
    return parse_args(s)
end

# parse number of procs, number of runs
ub_args = parse_ubermain()

# start workers in DSR env
addprocs(
    ub_args["procs"];
    exeflags = `--threads=$(Threads.nthreads()) --project=$(Base.active_project())`,
)

# make pkgs available in all processes
@everywhere using DSR
@everywhere ENV["GKSwstype"] = "nul"

"""
    ubermain(n_runs)

Start multiple parallel trainings, with optional grid search and
multiple runs per experiment.
"""
function ubermain(n_runs::Int, args::DSR.ArgVec)
    # load defaults with correct data types
    defaults = parse_args([], argtable())

    # prepare tasks
    tasks = prepare_tasks(defaults, args, n_runs)
    println(length(tasks))

    # run tasks
    pmap(main_routine, tasks)
end

# list arguments here
args = DSR.ArgVec([
    Argument("experiment", "DynaMix_training"),
    Argument("name", "DynaMix_default"),
])

# run experiments
ubermain(ub_args["runs"], args)