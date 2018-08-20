mutable struct NACCell{T,V}
    nac::NAC{T}
    h::V
end

NACCell(in::Integer, out::Integer; init = glorot_uniform) = NACCell(NAC(in, out), param(init(out)))

Flux.hidden(c::NACCell) = c.h

function (c::NACCell)(h, x)
    nac = c.nac
    h = nac(x) .+ h
    return h, h
end

@treelike NACCell

RNAC(a...; ka...) = Flux.Recur(NACCell(a...; ka...))


mutable struct NALUCell{T,V}
    nalu::NALU{T}
    h::V
end

NALUCell(in::Integer, out::Integer; init = glorot_uniform) = NALUCell(NALU(in, out), param(init(out)))

Flux.hidden(c::NALUCell) = c.h

function (c::NALUCell)(h, x)
    nalu = c.nalu
    h = nalu(x) .+ h
    return h, h
end

@treelike NALUCell

RNALU(a...; ka...) = Flux.Recur(NALUCell(a...; ka...))
