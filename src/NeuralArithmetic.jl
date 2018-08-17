__precompile__(false)

module NeuralArithmetic

using Flux
using Flux: @treelike, glorot_uniform, tanh, sigmoid

export NAC, NALU

include("nac.jl")
include("nalu.jl")

end # module
