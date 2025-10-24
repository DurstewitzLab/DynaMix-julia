using MultivariateStats
using Flux
using FFTW
using DelayEmbeddings
using Statistics
using StatsBase
using MultivariateStats
using Optim

using ..Model
using ..ObservationModels
using ..ReconstructionMeasures: smooth_dims!

"""
Context embedding functions, one can choose between the following:
- Zero embedding
- Delay embedding using PECUZAL algorithm
- Random delay embedding
- Autocorrelation based delay embedding
- Positional embedding
"""

# random delay embedding function
function delay_embedding_random(data, model_dim; upper_tau=10, lower_tau=3, apply_smoothing=false)
    N_data = size(data,2)
    needed_dims = model_dim - N_data

    if needed_dims <= 0
        return data
    end
    
    if apply_smoothing
        smoothed_data = copy(data)
        smooth_dims!(smoothed_data, 2.0)
        processed_data = smoothed_data
    else
        processed_data = data
    end

    taus = rand(lower_tau:upper_tau, needed_dims)
    ts = processed_data[:, 1]

    result = copy(processed_data[maximum(taus)+1:end, :])
    for i in 1:needed_dims
        result = hcat(result, ts[maximum(taus)-taus[i]+1:end-taus[i]])
    end
    
    return result
end


# Estimate tau using autocorrelation
function estimate_TDM_tau(data, threshold=0.368)

    t, n = size(data)
    tau_vals = zeros(Int, n)
    
    for dim in 1:n
        acf = autocor(data[:, dim], 0:round(Int, t/2))
        tau = findfirst(x -> x < threshold, acf[2:end])
        tau_vals[dim] = isnothing(tau) ? 1 : tau
    end
    
    return maximum(tau_vals)
end

# Standard delay embedding function
function delay_embedding(data, model_dim; τ=nothing, apply_smoothing=false)
    N_data = size(data, 2)
    needed_dims = model_dim - N_data
    
    if needed_dims <= 0
        return data
    end
    
    if apply_smoothing
        smoothed_data = copy(data)
        smooth_dims!(smoothed_data, 2.0)
        processed_data = smoothed_data
    else
        processed_data = data
    end
    
    if isnothing(τ)
        τ = estimate_TDM_tau(processed_data)
    end
    
    # Create embedding with optimal tau
    ts = processed_data[:, end]
    start_idx = needed_dims * τ + 1
    
    # Handle case where start_idx is too large
    if start_idx > size(processed_data, 1)
        τ = max(1, div(size(processed_data, 1), needed_dims + 1))
        start_idx = needed_dims * τ + 1
    end
    
    # Create shortened data and embedding
    shortened_data = processed_data[start_idx:end, :]
    result = copy(shortened_data)
    
    # Add delayed versions
    for i in 1:needed_dims
        result = hcat(result, ts[start_idx-i*τ:end-i*τ])
    end
    
    return result
end


# Zero embedding
function zero_embedding(data, model_dim)
    N_data = size(data,2)
    needed_dims = model_dim - N_data

    if needed_dims > 0
        data = hcat(data, Float32.(zeros(size(data,1), needed_dims)))
    end

    return data
end

# Delay embedding using PECUZAL algorithm
function delay_embedding_PECUZAL(data, model_dim)
    N_data = size(data,2)
    needed_dims = model_dim - N_data

    if needed_dims > 0
        reconstruction, τ_vals, _, _, _ = pecuzal_embedding(StateSpaceSet(reverse(Float64.(data[:,end].+randn(size(data[:,end])).*0.001))))
        if length(τ_vals) < needed_dims + 1
            for i in 1:(needed_dims + 1 - length(τ_vals))
                reconstruction = hcat(reconstruction, zeros(size(reconstruction,1))) # Pad with zeros if not enough delay embeddings
            end
        end
        reconstruction = reverse(Float32.(Matrix(copy(reconstruction))),dims=1)
        reconstruction = hcat(data[end-size(reconstruction,1)+1:end,:], reconstruction[:,2:needed_dims+1])
    else
        reconstruction = data
    end

    return reconstruction
end

# Estimation of time constant tau for positional embedding 
function estimate_pos_tau(data, max_lag=round(size(data, 1)-1), min_lag=round(size(data, 1)/10))
    t, n = size(data)
    tau_vals = zeros(Int, n)
    
    for dim in 1:n
        ts = data[:, dim]
        acf = autocor(ts, 0:max_lag)
        acf = acf[2:end]  # Skip lag 0
        
        peaks = findall(i -> i > min_lag && i < length(acf) && 
                           acf[i-1] < acf[i] && acf[i] > acf[i+1], 1:length(acf))
        
        if !isempty(peaks)
            peak_values = acf[peaks]
            max_peak_idx = argmax(peak_values)
            tau_vals[dim] = peaks[max_peak_idx]
        else
            tau_vals[dim] = argmax(acf)
        end
    end
    
    return maximum(tau_vals)
end

# Positional embedding function
function pos_embedding(data, model_dim)
    t = size(data,1)
    N_data = size(data,2)
    needed_dims = model_dim - N_data
    needed_dims != 1 ? shifts = range(0,pi/2,needed_dims) : shifts = 0
    tau = estimate_pos_tau(data)

    for shift in shifts
        data = hcat(data, sin.(2*pi/tau .*(1:t) .+ shift))
    end

    return Float32.(data)
end

# Data preprocessing function wrapper
function data_preprocessing(data, model_dim, preprocessing_method="pos_embedding")
    method = Symbol(preprocessing_method)
    data_embedded = @eval $method($data, $model_dim)

    data_embedded_re = reshape(data_embedded,(:,1)) # Flatten for model input

    return data_embedded_re, data_embedded
end

# Estimate initial condition
function estimate_initial_condition(initial_x, context_embedded)
    # Get dimensions
    T, N = size(context_embedded)
    N_partial = length(initial_x)
    @assert N_partial <= N "Initial condition dimension must be <= embedding dimension"

    # Find the timestep with closest match to initial condition in first N_partial dimensions
    distances = zeros(T)
    for t in 1:T
        distances[t] = sum((context_embedded[t,1:N_partial] .- initial_x).^2)
    end
    closest_t = argmin(distances)

    # Return full state vector by combining initial condition with closest matching state
    return vcat(initial_x, context_embedded[closest_t, (N_partial+1):N])
end


"""
Forecasting pipeline for DynaMix foundation model. Requires the following inputs:
- model: DynaMix foundation model
- O: Observation model
- context: Context data
- T: Forecast horizon
- preprocessing_method
- standardize: standardize data? True/False
- initial_x: Optional initial condition for the model, else last context value is used
"""

function DynaMix_forecasting_pipeline(model, O, context, T; preprocessing_method="pos_embedding", standardize=true, initial_x=nothing)
    M = size(model.W_MoE[1])[1]
    Z_gen = zeros(T, M)

    #standardize data
    if standardize
        context_mean = mean(context,dims=1)
        context_std = std(context,dims=1)
        context = Float32.((context .- context_mean) ./ context_std)
        if !isnothing(initial_x)
            initial_x = Float32.((initial_x .- context_mean[1,:]) ./ context_std[1,:])
        end
    end

    # Data embedding + initial condition
    model_context, context_embedded = data_preprocessing(context, model.N, preprocessing_method)
    if isnothing(initial_x)
        initial = reshape(model_context,Int(size(model_context)[1]/model.N),model.N)[end,:] # take last context value as initial value to the model
    else
        if size(initial_x,1) < model.N
            initial = estimate_initial_condition(initial_x, context_embedded)   
        else
            initial = initial_x
        end
    end

    #Forecasting
    zₜ = Float32.(init_state(O,initial))
    for t in 1:T
        zₜ = model(Float32.(reshape(zₜ,(M,1))), model_context)
        Z_gen[t, :] = zₜ
    end

    #Rescaling data due to standardization
    if standardize
        Z_gen[:,1:size(context,2)] = Z_gen[:,1:size(context,2)] .* context_std .+ context_mean
        context_embedded[:,1:size(context,2)] = context_embedded[:,1:size(context,2)] .* context_std .+ context_mean
    end

    return Z_gen[:,1:size(context,2)]
end
