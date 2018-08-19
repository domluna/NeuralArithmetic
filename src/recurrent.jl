mutable struct NACCell{T,V}
    n::NAC{T}
    h::V
end

@treelike NACCell

mutable struct NALUCell{T,V}
    n::NALU{T}
    h::V
end

@treelike NALUCell
