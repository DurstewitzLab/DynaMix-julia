using Flux

function prediction_error(
    X::AbstractMatrix{T},
    X̃::AbstractMatrix{T},
    steps::Int = 10 
) where {T}
    return mean(sum(abs.(X[steps, :] - X̃[steps, :])))
end

function prediction_error(
    X::AbstractArray{T, 3},
    X̃::AbstractArray{T, 3},
    steps::Int = 10
) where {T}
    return mean(prediction_error.(eachslice(X, dims = 3), eachslice(X̃, dims = 3), steps), dims = 3)
end



"""
    MASE(X::AbstractMatrix{T}, X̃::AbstractMatrix{T}, steps::Int = 10, naive_lag::Int = 1, average::Bool = true) where {T}

Multivariate Mean Absolute Scaled Error (MASE).

# Arguments
- `X`: Matrix of shape (T, D), actual values
- `X̃`: Matrix of shape (T, D), predicted values
- `steps`: Number of time steps to consider in MAE computation
- `naive_lag`: Lag for the naive forecast (default = 1)
- `average`: If true, return mean MASE over all variables; else return per-variable MASE

# Returns
- MASE score (scalar if average=true, else vector of shape (D,))
"""
function MASE(
    X::AbstractMatrix{T}, 
    X̃::AbstractMatrix{T}, 
    steps::Int = 10, 
    naive_lag::Int = 1, 
    average::Bool = true
) where {T}
    # Check dimensions
    size(X) == size(X̃) || error("Shapes of X and X̃ must match")
    
    # Truncate steps if too long
    steps = min(steps, size(X, 1))
    
    # MAE between prediction and ground truth (per variable)
    mae = mean(abs.(X[1:steps, :] - X̃[1:steps, :]), dims=1)  # shape: (1, D)
    
    # Naive forecast MAE: |X[t] - X[t - naive_lag]|
    naive_mae = mean(abs.(X[(naive_lag+1):end, :] - X[1:(end-naive_lag), :]), dims=1)  # shape: (1, D)
    
    # Avoid division by zero
    epsilon = 1e-8
    mase = mae ./ (naive_mae .+ epsilon)
    
    return average ? mean(mase) : vec(mase)
end

# 3D array version for batched computation
function MASE(
    X::AbstractArray{T, 3}, 
    X̃::AbstractArray{T, 3}, 
    steps::Int = 10,
    naive_lag::Int = 1,
    average::Bool = true
) where {T}
    return mean(MASE.(eachslice(X, dims=3), eachslice(X̃, dims=3), steps, naive_lag, average))
end
