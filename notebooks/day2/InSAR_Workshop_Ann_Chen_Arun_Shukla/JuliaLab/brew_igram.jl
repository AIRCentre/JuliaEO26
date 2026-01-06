function brew_igram()

    # Setup
    vec = ones(Float32, 255)
    vec[85:85+85] .= 1 .- 0.6/85 .* (0:85)
    vec[85+85:end] .= 0.4 .+ 0.6/85 .* (0:85)
    circshift(v::AbstractVector, n::Int) = v[mod.(collect(1:length(v)) .- 1 .- n, length(v)) .+ 1]

    # Populate
    g = circshift(vec, 1)
    b = circshift(g, 85)
    r = circshift(b, 85)
    cmap = hcat(g, b, r)

    return cmap
end
