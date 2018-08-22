module NeuralArithmetic

using Flux
using Flux: @treelike, glorot_uniform, tanh
using NNlib: σ_stable


export NAC, NALU, RNAC, RNALU

include("nac.jl")
include("nalu.jl")
include("recurrent.jl")

end # module
