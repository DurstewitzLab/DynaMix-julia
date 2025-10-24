using Flux

using ..Model
using ..Utilities
using ..ObservationModels

abstract type AbstractTFRecur end
Flux.trainable(tfrec::AbstractTFRecur) = (tfrec.model, tfrec.O)

(tfrec::AbstractTFRecur)(
    X::AbstractArray{T, 3},
    Ẑ::AbstractArray{T, 3},
    C::AbstractArray{T, 2},  # 2D array: context_size × batch_size
) where {T} = forward(tfrec, X, Ẑ, C)

# compute forcing signals using observation model
estimate_forcing_signals(tfrec::AbstractTFRecur, X::AbstractArray{T, 3}) where {T} =
    apply_inverse(tfrec.O, X)



"""
Wrapper of recurrent model used for teacher forcing during training.
Teacher forcing is a training technique where the model's predictions are
periodically replaced with ground truth values to stabilize training.
This wrapper manages a recurrent model and handles the periodic forcing
based on a specified interval `τ`.
"""

mutable struct TFRecur{ℳ <: AbstractDynaMix, M <: AbstractMatrix, 𝒪 <: ObservationModel} <:
               AbstractTFRecur
    # stateful model
    model::ℳ
    # observation model
    O::𝒪
    # state of the model
    z::M
    # forcing interval
    const τ::Int
end
Flux.@functor TFRecur

# sequenceforward pass for basic teacher forcing
function forward(
    tfrec::TFRecur,
    X::AbstractArray{T, 3},  # data: M × batch_size × T
    Ẑ::AbstractArray{T, 3},  # forcing signals: M × batch_size × T
    C::AbstractArray{T, 2},  # context: context_size × batch_size
) where {T}
    T_ = size(X, 3)

    # Initialize latent state
    tfrec.z = init_state(tfrec.O, @view(X[:, :, 1]))

    # Process sequence
    Z = [tfrec(ẑ, C, t) for (ẑ, t) ∈ zip(eachslice(Ẑ, dims=3), 2:T_)]

    # Reshape to 3d array and return
    return reshape(reduce(hcat, Z), size(tfrec.z)..., :)
end

# single step forward pass for basic teacher forcing
function (tfrec::TFRecur)(x::AbstractMatrix, c::AbstractMatrix, t::Int)
    # determine if it is time to force the model
    z = tfrec.z

    # perform one step using the model, update model state
    z = tfrec.model(z, c)

    # force
    z̃ = (t - 1) % tfrec.τ == 0 ? force(z, x) : z
    tfrec.z = z̃
    return z
end



"""
Wrapper of recurrent model used for generalized teacher forcing during training.
Generalized teacher forcing is a training technique where the model's predictions
are adapted with ground truth values to stabilize training.
This wrapper manages a recurrent model and handles the forcing
based on a specified parameter `α`.
"""

mutable struct GTFRecur{
    ℳ <: AbstractDynaMix,
    𝒪 <: ObservationModel,
    M <: AbstractMatrix,
    T <: AbstractFloat,
} <: AbstractTFRecur
    model::ℳ
    # ObservationModel
    O::𝒪
    # state of the model
    z::M
    # forcing α
    α::T
end
Flux.@functor GTFRecur

# sequence forward pass for generalized teacher forcing
function forward(
    tfrec::GTFRecur,
    X::AbstractArray{T, 3},
    Ẑ::AbstractArray{T, 3},
    C::AbstractArray{T, 2},  # 2D array: context_size × batch_size
) where {T}
    # Initialize latent state
    tfrec.z = init_state(tfrec.O, @view(X[:, :, 1]))
    
    # Process sequence
    Z = [tfrec(ẑ, C) for ẑ ∈ eachslice(Ẑ, dims=3)]

    # Reshape to 3d array and return
    return reshape(reduce(hcat, Z), size(tfrec.z)..., :)
end

# single step forward pass for generalized teacher forcing
function (tfrec::GTFRecur)(ẑ::AbstractMatrix, c::AbstractMatrix)
    z = tfrec.z
    D, M = size(ẑ, 1), size(z, 1)
    z = tfrec.model(z, c)
    # gtf
    z̃ = force(@view(z[1:D, :]), ẑ, tfrec.α)
    z̃ = (D == M) ? z̃ : force(z, z̃)

    tfrec.z = z̃
    return z
end