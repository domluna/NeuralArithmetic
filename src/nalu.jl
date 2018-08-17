"""
Neural Arithmetic Logic Unit (NALU). [1]

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
    g = sigmoid.(n.G * x)
    a = W * x
    m = exp.(W * (log.(abs.(x) .+ 1e-10)))
    g .* a + (1 .- g) .* m
end

Base.show(io::IO, l::NALU) = print(io, "NALU(", size(l.G, 2), ", ", size(l.G, 1), ")")
