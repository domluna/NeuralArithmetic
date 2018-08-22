"""
    NALU(in::Integer, out::Integer; initW = glorot_uniform, initb = zeros, ϵ=1e-10)

Neural Arithmetic Logic Unit (NALU). [1]

    W = tanh.(Ŵ) .* sigmoid.(M̂)
    a = W * x .+ b1
    g = sigmoid.(G * x .+ b2)
    m = exp.(W * (log.(abs.(x) .+ ϵ)))
    g .* a + (1 .- g) .* m

The input `x` is a:
    * vector of length `in`. 
    * matrix of size (`in`, `batch_size`).
    
[1]: https://arxiv.org/abs/1808.00508
"""
struct NALU{N,T,S,E}
    nac::N
    G::T
    b::S
    ϵ::E
end


function NALU(in::Integer, out::Integer; initW = glorot_uniform, initb = zeros, ϵ=1e-10)
    return NALU(NAC(in, out), param(initW(out, in)), param(initb(out)), ϵ)
end

@treelike NALU

function (n::NALU)(x)
    nac, G, b, ϵ = n.nac, n.G, n.b, n.ϵ
    a = nac(x)
    g = σ_stable.(G * x .+ b)
    m = exp.(nac((log.(abs.(x) .+ ϵ))))
    g .* a + (1 .- g) .* m
end

Base.show(io::IO, l::NALU) = print(io, "NALU(", size(l.G, 2), ", ", size(l.G, 1), ")")
