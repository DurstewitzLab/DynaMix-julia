using Flux: @functor
using Statistics: mean
using Flux: softmax, Conv
using Flux
using Zygote
using LinearAlgebra

using ..ObservationModels: ObservationModel, init_state


# abstract type
abstract type AbstractDynaMix end

(m::AbstractDynaMix)(z::AbstractVecOrMat, context::AbstractVecOrMat) = step(m, z, context)


"""
Gating Network for DynaMix. 
Processes context and current state to generate mixing weights for experts.
"""

mutable struct GatingNetwork{C <: Conv, V1 <: AbstractVector, M1 <: AbstractMatrix, M2 <: AbstractMatrix, M3 <: AbstractMatrix, V2 <: AbstractVector, V3 <: AbstractVector}
    conv::C  
    softmax_temp1::V1
    D::M1
    mlp_layer1::M2
    mlp_layer2::M3
    softmax_temp2::V2
    sigma::V3
end

@functor GatingNetwork (conv, softmax_temp1, D, mlp_layer1, mlp_layer2, softmax_temp2, sigma)

# constructor for GatingNetwork
function GatingNetwork(N::Int, M::Int, Experts::Int)
    conv = Conv((2,), N => N, identity)
    softmax_temp1 = Float32.([0.1])    
    D = Float32.(zeros(N,M))
    D[:,1:N] = Float32.(Matrix(I,N,N))
    mlp_layer1 = gaussian_init(Experts, M+N)
    mlp_layer2 = gaussian_init(Experts, Experts)
    softmax_temp2 = Float32.([0.1])
    sigma = Float32.(ones(N).*0.05)
    
    return GatingNetwork(
        conv, softmax_temp1, D, mlp_layer1, mlp_layer2, softmax_temp2, sigma
    )
end

# forward pass for gating network
function (m::GatingNetwork)(context::AbstractMatrix, z::AbstractMatrix)
    batch_size = size(z, 2)
    N = size(m.D, 1)
    seq_length = div(size(context, 1), N)
    M = size(z, 1)

    # Compute attention weights
    z_current = Zygote.ignore() do
        copy(z)
    end
    z_current = (m.D * z_current) .+ m.sigma .* randn(Float32,(N,batch_size))

    context_reshaped = reshape(context, seq_length, N, batch_size)
    distances = dropdims(sum(abs.(context_reshaped[1:end-1, :, :] .- reshape(z_current, 1, N, batch_size)), dims=2), dims=2)
    attention_weights = softmax(-distances ./ abs(m.softmax_temp1[1]); dims=1)
    
    # Process context with convolution
    encoded = m.conv(context_reshaped)
    
    # Build weighted embedding
    embedding = dropdims(batched_mul(permutedims(encoded,(2,1,3)),reshape(attention_weights,(:,1,batch_size))), dims=2)

    # Predict expert weights
    w_exp = softmax((-m.mlp_layer2 * relu.(m.mlp_layer1 * vcat(embedding, z))) ./ abs(m.softmax_temp2[1]))
    
    return w_exp
end


"""
DynaMix base model.
Combines multiple experts through a weighting obtained from a gating network operating on the context and current state.

The model consists of:
- AL-RNN Experts (A_MoE, W_MoE, h_MoE)
- A gating network (gating_network)
- Hyperparameters (N, Experts, P)
"""

mutable struct DynaMix{V1,V2,V3,C} <: AbstractDynaMix
    A_MoE::V1
    W_MoE::V2
    h_MoE::V3
    gating_network::C
    N::Int
    Experts::Int
    P::Int
end

@functor DynaMix (A_MoE, W_MoE, h_MoE, gating_network)

# constructor for DynaMix model
function DynaMix(M::Int, N::Int, Experts::Int, P::Int, Kext::Int)
    A_MoE = [initialize_A_W_h(M)[1] for _ in 1:Experts]
    W_MoE = [Float32.(randn(M, M) .* 0.01) for _ in 1:Experts]
    h_MoE = [Float32.(zeros(M)) for _ in 1:Experts]

    gating_network = GatingNetwork(N, M, Experts)
    
    return DynaMix(
        A_MoE, W_MoE, h_MoE, gating_network, N, Experts, P
    )
end


# one step forward pass for DynaMix model
function step(m::DynaMix, z::AbstractVecOrMat, context::AbstractVecOrMat)
    w_exp = m.gating_network(context, z)
    
    exp = [A .* z .+ W * vcat(z[1:end-m.P,:], relu.(z)[end-(m.P-1):end,:]) .+ h
           for (A,W,h) in zip(m.A_MoE,m.W_MoE,m.h_MoE)]
    results = [exp[i] .* w_exp[i, :]' for i in 1:m.Experts]
    
    return sum(results)
end

