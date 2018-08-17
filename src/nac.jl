"""
    NAC(in::Int, out::Int, init=glorot_uniform)

Creates a Neural Accumulator (NAC), which contrains the output values close to -1, 0, and 1.

    W = tanh.(Ŵ) .* sigmoid.(M̂)
    a = W * x

The input `x` is a:
    * vector of length `in`. 
    * matrix of size (`in`, `batch_size`).

https://arxiv.org/abs/1808.00508
"""
struct NAC{T}
    Ŵ::T
    M̂::T
end


function NAC(in::Int, out::Int, init=glorot_uniform)
    return NAC(param(init(out, in)), param(init(out, in)))
end

@treelike NAC

function (n::NAC)(x)
    W = tanh.(n.Ŵ) .* sigmoid.(n.M̂)
    W * x
end

Base.show(io::IO, l::NAC) = print(io, "NAC(", size(l.Ŵ, 2), ", ", size(l.Ŵ, 1), ")")
