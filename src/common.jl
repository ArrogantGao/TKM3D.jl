struct TKMVals{P, G, PT, GT}
    pot::P
    grad::G
    pottarg::PT
    gradtarg::GT
    ier::Int
end

function as_3xn(points::AbstractMatrix{<:Real})
    size(points, 1) == 3 || throw(ArgumentError("points must have shape (3, N)"))
    return Matrix{float(eltype(points))}(points)
end

function as_nx3(points::AbstractMatrix{<:Real})
    size(points, 2) == 3 || throw(ArgumentError("points must have shape (N, 3)"))
    return Matrix{float(eltype(points))}(points)
end

function combined_box_geometry_3xn(sources::AbstractMatrix{T}, targets::AbstractMatrix{T}) where {T <: Real}
    mins = vec(min.(minimum(sources; dims = 2), minimum(targets; dims = 2)))
    maxs = vec(max.(maximum(sources; dims = 2), maximum(targets; dims = 2)))
    lengths = maxs .- mins
    center = (mins .+ maxs) ./ 2
    return lengths, center
end

function combined_box_geometry(sources::AbstractMatrix{T}, targets::AbstractMatrix{T}) where {T <: Real}
    mins = vec(min.(minimum(sources; dims = 1), minimum(targets; dims = 1)))
    maxs = vec(max.(maximum(sources; dims = 1), maximum(targets; dims = 1)))
    lengths = maxs .- mins
    center = (mins .+ maxs) ./ 2
    return lengths, center
end

function centered_mode_axis(Δk::T, k_max::T) where {T <: Real}
    Δk > zero(T) || throw(ArgumentError("mode spacing must be positive"))
    k_max >= zero(T) || throw(ArgumentError("k_max must be nonnegative"))

    mmax = ceil(Int, k_max / Δk)
    nmodes = 2 * mmax + 1
    axis = Vector{T}(undef, nmodes)
    shift = nmodes ÷ 2
    for i in 1:nmodes
        axis[i] = Δk * (i - 1 - shift)
    end
    return axis
end

@inline function truncated_laplace3d_hat(k::T, L::T) where {T <: Real}
    if iszero(k)
        return L^2 / 2
    end
    return 2 * (sin(L * k / 2) / k)^2
end
