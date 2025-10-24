using NPZ

abstract type AbstractDataset end

struct Dataset{T, N, A <: AbstractArray{T, N}, B <: AbstractMatrix{T}} <: AbstractDataset
    X::A  # 3D array: (time, features, series)
    C::B  # 2D array: (context_size, series)
    Test_data::Union{A, Nothing}
    name::String
end

function Dataset(
    data_path::String,
    context_path::String,
    test_data_path::String,
    name::String;
    dtype = Float32,
)
    X = npzread(data_path) .|> dtype
    C = npzread(context_path) .|> dtype
    Test_data = npzread(test_data_path) .|> dtype
    # Ensure X is 3D: (time, features, series)
    if ndims(X) == 2
        X = reshape(X, size(X)..., 1)
    end
    # Ensure C is 2D: (context_size, series)
    if ndims(C) == 1
        C = reshape(C, size(C)..., 1)
    end
    # Ensure Test_data is 3D: (time, features, series)
    if ndims(Test_data) == 2
        Test_data = reshape(Test_data, size(Test_data)..., 1)
    elseif ndims(Test_data) !== 3
        Test_data = nothing
    end
    
    return Dataset(X, C, Test_data, name)
end

Dataset(
    data_path::String,
    context_path::String,
    test_data_path::String;
    dtype = Float32,
) = Dataset(data_path, context_path, test_data_path, ""; dtype = dtype)

@inbounds """
    sample_sequence(dataset, sequence_length)

Sample a sequence of length `T̃` from a time series X.
Returns X sequence and corresponding context vector C.
"""
function sample_sequence(D::Dataset{T_, 3, A, B}, T̃::Int, j::Int) where {T_, A, B}
    T = size(D.X, 1)
    i = rand(1:T-T̃-1)
    return D.X[i:i+T̃, :, j], D.C[:, j]
end

"""
    sample_batch(dataset, seq_len, batch_size)

Sample a batch of sequences of batch size `batch_size` from time series X
(with replacement!). The same context is used for all time steps in a sequence.
"""
function sample_batch(D::Dataset{T_, 3, A, B}, T̃::Int, batch_size::Int) where {T_, A, B}
    N = size(D.X, 2)      # Number of features in X
    K = size(D.C, 1)      # Context size dimension
    n = size(D.X, 3)      # Number of time series
    
    # Create arrays for batch
    Xs = similar(D.X, N, batch_size, T̃ + 1)
    Cs = similar(D.C, K, batch_size)  # 2D array: context_size × batch_size
    
    for i = 1:batch_size
        j = rand(1:n)  # Randomly select a time series
        X̃, C̃ = sample_sequence(D, T̃, j)
        Xs[:, i, :] .= X̃'
        Cs[:, i] .= C̃
    end
    
    return Xs, Cs
end
