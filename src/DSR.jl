module DSR

using Reexport

include("measures/ReconstructionMeasures.jl")
@reexport using .ReconstructionMeasures

include("utilities/Utilities.jl")
@reexport using .Utilities

include("model/ObservationModels.jl")
@reexport using .ObservationModels

include("model/Model.jl")
@reexport using .Model

include("tf_training/TFTraining.jl")
@reexport using .TFTraining

# meta stuff
include("parsing.jl")
export parse_commandline,
    initialize_model,
    initialize_optimizer,
    argtable,
    initialize_observation_model

include("multitasking.jl")
export Argument, prepare_tasks, main_routine

end
