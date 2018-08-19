"""
    NAC(in::Integer, out::Integer; initW = glorot_uniform, initb = zeros)

Creates a Neural Accumulator (NAC), which contrains the output values close to -1, 0, and 1.

    W = tanh.(Ŵ) .* sigmoid.(M̂)
    a = W * x .+ b

The input `x` is a:
    * vector of length `in`. 
    * matrix of size (`in`, `batch_size`).

https://arxiv.org/abs/1808.00508
"""
struct NAC{T,S}
    Ŵ::T
    M̂::T
    b::S
end


function NAC(in::Integer, out::Integer; initW = glorot_uniform, initb = zeros)
    return NAC(param(initW(out, in)), param(initW(out, in)), param(initb(out)))
end

@treelike NAC

function (n::NAC)(x)
    Ŵ, M̂, b = n.Ŵ, n.M̂, n.b
    W = tanh.(Ŵ) .* sigmoid.(M̂)
    W * x .+ b
end

Base.show(io::IO, l::NAC) = print(io, "NAC(", size(l.Ŵ, 2), ", ", size(l.Ŵ, 1), ")")
