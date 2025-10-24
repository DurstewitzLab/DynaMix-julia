using JSON
using BSON: @save
using Flux: cpu
using BSON: load

const RES = "Results"

"""
    create_folder_structure(exp::String, run::Int)

Creates basic saving structure for a single run/experiment.
"""
function create_folder_structure(exp::String, name::String, run::Int)::String
    # create folder
    path_to_run = joinpath(RES, exp, name, format_run_ID(run))
    mkpath(joinpath(path_to_run, "checkpoints"))
    mkpath(joinpath(path_to_run, "plots"))
    return path_to_run
end

function format_run_ID(run::Int)::String
    # only allow three digit numbers
    @assert run < 1000
    return string(run, pad = 3)
end

store_hypers(dict::Dict, path::String) =
    open(joinpath(path, "args.json"), "w") do f
        JSON.print(f, dict, 4)
    end

function convert_to_Float32(dict::Dict)
    for (key, val) in dict
        dict[key] = val isa AbstractFloat ? Float32(val) : val
    end
    return dict
end

load_defaults() = load_json_f32(joinpath(@__DIR__, "..", "..", "settings", "defaults.json"))

load_json_f32(path) = convert_to_Float32(JSON.parsefile(path))

save_model(model, path::String) = @save path model = cpu(model)

load_model(path::String) = load(path, @__MODULE__)[:model]

check_for_NaNs(θ) = any(!isfinite(sum(p)) for p ∈ θ)

decide_on_measure(scaling::Real, bins::Int, N::Int) =
    N < 7 ? (bins, "(BIN)") : (scaling, "(GMM)")

num_params(m) = sum(length, Flux.params(m))

offdiagonal(X::AbstractMatrix) = X - Diagonal(X)
@inbounds offdiagonal!(X::AbstractMatrix) = X[diagind(X)] .= zero(eltype(X))

uniform(a, b) = rand(eltype(a)) * (b - a) + a
uniform(size, a, b) = rand(eltype(a), size) .* (b - a) .+ a

randn_like(X::AbstractArray{T, N}) where {T, N} = randn!(similar(X))

add_gaussian_noise!(X::AbstractArray{T, N}, noise_level::T) where {T, N} =
    X .+= noise_level .* randn_like(X)
add_gaussian_noise(X::AbstractArray{T, N}, noise_level::T) where {T, N} =
    X .+ noise_level .* randn_like(X)