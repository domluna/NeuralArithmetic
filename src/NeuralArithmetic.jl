__precompile__(false)

module NeuralArithmetic

using Flux
using Flux: @treelike, glorot_uniform, tanh, sigmoid

export NAC, NALU, RNAC, RNALU

include("nac.jl")
include("nalu.jl")
include("recurrent.jl")

end # module
