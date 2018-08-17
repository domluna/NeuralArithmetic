"""
Neural Arithmetic Logic Unit (NALU). [1]

    W = tanh.(Ŵ) .* sigmoid.(M̂)
    a = W * x
    g = sigmoid.(n.G * x)
    m = exp.(W * (log.(abs.(x) .+ 1e-7)))
    g .* a + (1 .- g) .* m

The input `x` is a:
    * vector of length `in`. 
    * matrix of size (`in`, `batch_size`).
    
[1]: https://arxiv.org/abs/1808.00508
"""
struct NALU{T}
    Ŵ::T
    M̂::T
    G::T
end


function NALU(in::Int, out::Int, init=glorot_uniform)
    return NALU(param(init(out, in)), param(init(out, in)), param(init(out, in)))
end

@treelike NALU

function (n::NALU)(x)
    W = tanh.(n.Ŵ) .* sigmoid.(n.M̂)
    a = W * x
    g = sigmoid.(n.G * x)
    m = exp.(W * (log.(abs.(x) .+ 1e-7)))
    g .* a + (1 .- g) .* m
end

Base.show(io::IO, l::NALU) = print(io, "NALU(", size(l.G, 2), ", ", size(l.G, 1), ")")
