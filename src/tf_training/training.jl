using Flux
using BSON: @save
using DataStructures
using JLD
using Statistics
using NaNStatistics

using ..ReconstructionMeasures
using ..Model
using ..ObservationModels
using ..Utilities

"""
    loss(tfrec, X̃, C̃)

Performs a forward pass using the teacher forced recursion wrapper `tfrec` and
computes and return the loss w.r.t. data `X̃`. Optionally external context `C̃`
can be provided.
"""

function loss(
    tfrec::AbstractTFRecur,
    X̃::AbstractArray{T, 3},
    Ẑ::AbstractArray{T, 3},
    C̃::AbstractArray{T, 2},  # 2D array: context_size × batch_size
) where {T}
    Z = tfrec(X̃, Ẑ, C̃)
    X̂ = tfrec.O(Z)
    return @views Flux.mse(X̂, X̃[:, :, 2:end])
end

"""
    regularization_loss(tfrec, λₗ, λₒ)

Computes the regularization loss for the model `tfrec` with latent model regularization parameter `λₗ`
and observation model regularization parameter `λₒ`.
"""
function regularization_loss(
    tfrec::AbstractTFRecur,
    λₗ::Float32,
    λₒ::Float32,
)
    # latent model regularization
    Lᵣ = 0.0f0

    if λₗ > 0
        Lᵣ += regularize(tfrec.model, λₗ)
    end

    # observation model regularization
    Lᵣ += (λₒ > 0) ? regularize(tfrec.O, λₒ) : 0
    return Lᵣ
end

function regularize(m::AbstractDynaMix, λ::Float32; c = 0.01)
    return λ * mean(exp.(-abs.(m.gating_network.sigma) ./ c))
end


"""
    train_!(m, O, 𝒟, opt, args, save_path)

Train a model `m` with observation model `O` on dataset `𝒟` using the optimizer `opt`.
"""

function train_!(
    m::AbstractDynaMix,
    O::ObservationModel,
    𝒟::AbstractDataset,
    opt::Flux.Optimise.Optimiser,
    args::AbstractDict,
    save_path::String,
)
    # hypers
    E = args["epochs"]::Int
    M = args["latent_dim"]::Int
    Sₑ = args["batches_per_epoch"]::Int
    S = args["batch_size"]::Int
    τ = args["teacher_forcing_interval"]::Int
    σ_noise = args["gaussian_noise_level"]::Float32
    T̃ = args["sequence_length"]::Int
    σ²_scaling = args["D_stsp_scaling"]::Float32
    bins = args["D_stsp_bins"]::Int
    σ_smoothing = args["PSE_smoothing"]::Float32
    PE_n = args["PE_n"]::Int
    isi = args["image_saving_interval"]::Int
    ssi = args["scalar_saving_interval"]::Int
    exp = args["experiment"]::String
    name = args["name"]::String
    run = args["run"]::Int
    α = args["gtf_alpha"]::Float32
    α_method = args["gtf_alpha_method"]::String
    γ = args["gtf_alpha_decay"]::Float32
    λₒ = args["obs_model_regularization"]::Float32
    λₗ = args["lat_model_regularization"]::Float32
    partial_forcing = args["partial_forcing"]::Bool
    k = args["alpha_update_interval"]::Int
    use_gtf = args["use_gtf"]::Bool

    # data shape
    T, N, n = size(𝒟.X)

    #metrics
    metrics = []

    # zero-shot testing hyperparameters
    if 𝒟.Test_data !== nothing
        test_context_length = 2000
        test_prediction_length = size(𝒟.Test_data,1) - test_context_length
        test_systems = size(𝒟.Test_data,3)
    end

    # progress tracking
    prog = Progress(joinpath(exp, name), run, 20, E, 0.8)

    # decide on D_stsp scaling
    scal, stsp_name = decide_on_measure(σ²_scaling, bins, size(𝒟.Test_data,2))

    # initialize stateful model wrapper
    z₀ = similar(𝒟.X, M, S)
    if use_gtf
        tfrec = GTFRecur(m, O, z₀, α)
        println(
            "Using GTF with initial α = $α and annealing method: $α_method (γ = $γ, k = $k)",
        )
        println("Partial forcing set to: $partial_forcing (N = $N, M = $M)")
    else
        tfrec = TFRecur(m, O, z₀, τ)
        println("Using sparse TF with τ = $τ")
        println("Partial forcing set to: $partial_forcing (N = $N, M = $M)")
    end

    # model parameters
    θ = Flux.params(tfrec)

    # initial α
    α_est = α

    # losses
    ∑L, Lₜᵣ, Lᵣ = 0.0f0, 0.0f0, 0.0f0

    for e = 1:E
        # process a couple of batches
        t₁ = time_ns()
        for sₑ = 1:Sₑ
            # sample a batch
            X̃, C̃_exts = sample_batch(𝒟, T̃, S)
            C̃_exts = (C̃_exts,)

            # add noise noise if noise level > 0
            σ_noise > zero(σ_noise) ? add_gaussian_noise!(X̃, σ_noise) : nothing

            # precompute forcing signals
            Ẑ = estimate_forcing_signals(tfrec, X̃)

            # α estimation & annealing
            if sₑ % k == 0 && use_gtf
                α_est = compute_α(tfrec, @view(Ẑ[:, :, 2:end]), α_method)
                if α_est > tfrec.α
                    tfrec.α = α_est
                else
                    tfrec.α = γ * tfrec.α + (1 - γ) * α_est
                end
            end

            # partial forcing
            @views Ẑ_subset = partial_forcing ? Ẑ[1:N, :, 2:end] : Ẑ[:, :, 2:end]

            # forward and backward pass
            (∑L, Lₜᵣ, Lᵣ), grads = Flux.withgradient(θ) do
                Lₜᵣ = loss(tfrec, X̃, Ẑ_subset, C̃_exts...)
                Lᵣ = regularization_loss(tfrec, λₗ, λₒ)
                return Lₜᵣ + Lᵣ, Lₜᵣ, Lᵣ
            end

            # optimiser step
            Flux.Optimise.update!(opt, θ, grads)

            # check for NaNs in parameters (exploding gradients)
            if check_for_NaNs(θ)
                save_model(
                    [tfrec.model, tfrec.O],
                    joinpath(save_path, "checkpoints", "model_$e.bson"),
                )
                save(joinpath("Results", args["experiment"], args["name"], Utilities.format_run_ID(args["run"]), "metrics.jld"), "metrics", metrics)
                @warn "NaN(s) in parameters detected! \
                    This is likely due to exploding gradients. Aborting training..."
                return nothing
            end
        end
        t₂ = time_ns()
        Δt = (t₂ - t₁) / 1e9
        update!(prog, Δt, e)

        # Monitoring metrics
        if e % ssi == 0

            # Calculate zero-shot metrics if test data is provided
            if 𝒟.Test_data !== nothing
                D_stsp_all = []; D_H_all = []; pe_all = []
                for i in 1:test_systems
                    X_gen = DynaMix_forecasting_pipeline(tfrec.model, tfrec.O, 𝒟.Test_data[1:test_context_length, :, i], 
                            test_prediction_length, preprocessing_method="zero_embedding",standardize=true)
                    
                    push!(D_stsp_all, state_space_divergence(Float32.(𝒟.Test_data[test_context_length+1:end, :, i]), Float32.(X_gen), scal)) # gets calculated separately for each trajectory
                    push!(D_H_all, power_spectrum_error(Float32.(𝒟.Test_data[test_context_length+1:end, :, i]), Float32.(X_gen), σ_smoothing)) # same here
                    push!(pe_all, MASE(Float32.(𝒟.Test_data[test_context_length+1:end, :, i]), Float32.(X_gen),PE_n))
                end
                D_stsp = nanmedian(Float32.(D_stsp_all)); D_H = nanmedian(Float32.(D_H_all)); pe = nanmedian(Float32.(pe_all))
            else
                D_stsp = 0; D_H = 0; pe = 0
            end

            # progress printing
            scalars = OrderedDict("∑L" => ∑L)
            Lᵣ > 0.0f0 ? scalars["Lₜᵣ"] = Lₜᵣ : nothing
            Lᵣ > 0.0f0 ? scalars["Lᵣ"] = Lᵣ : nothing
            scalars["Dₛₜₛₚ $stsp_name"] = D_stsp
            scalars["Dₕ"] = D_H
            scalars["PE($PE_n)"] = pe
            typeof(tfrec) <: GTFRecur ? scalars["α"] = round(tfrec.α, digits = 3) : nothing
            push!(metrics, getindex.(Ref(scalars), ["∑L","Dₛₜₛₚ $stsp_name", "Dₕ","PE($PE_n)"]))

            print_progress(prog, Δt, scalars)

            # Save model
            save_model(
                [tfrec.model, tfrec.O],
                joinpath(save_path, "checkpoints", "model_$e.bson"),
            )

            # Plot trajectory if test data is provided
            if e % isi == 0 && 𝒟.Test_data !== nothing
                @views plot_reconstruction(
                    Float32.(DynaMix_forecasting_pipeline(tfrec.model, tfrec.O, 
                        𝒟.Test_data[1:test_context_length, :, 1], test_prediction_length,
                        preprocessing_method="zero_embedding",standardize=true)),
                    Float32.(𝒟.Test_data[test_context_length+1:end, :, 1]),
                    joinpath(save_path, "plots", "generated_$e.png"),
                )
            end
        end
    end

    #Save training metrics
    save(joinpath("Results", args["experiment"], args["name"], Utilities.format_run_ID(args["run"]), "metrics.jld"), "metrics", metrics)

    return nothing
end
