module Model

using Flux, LinearAlgebra

using ..Utilities

export AbstractDynaMix, 
    DynaMix,
    DynaMix_forecasting_pipeline,
    data_preprocessing,
    pos_embedding,
    delay_embedding,
    delay_embedding_random,
    delay_embedding_PECUZAL,
    uniform_init,
    gaussian_init,
    initialize_A_W_h

include("initialization.jl")
include("DynaMix.jl")
include("model_utilities.jl")

end