using Enzyme, GLMakie, ProgressBars, Statistics, HDF5

function V(x)
    return (x[1]^2 - 1)^2 / 4
end
# -∇V = x (1 - x^2)    
scatter(-2:0.01:2, [V([x]) for x in -2:0.01:2])
∇V(x) = -(Enzyme.gradient(Enzyme.Reverse, V, x)[1])
∇V([2.0])
##
ϵ = sqrt(2) # 0.5
Nₜ = 1000
Nₑ = 10000
Δt = 0.1 
x₀ = randn(Nₑ)
for t in ProgressBar(1:Nₜ)
    𝒩 = randn(Nₑ)
    for ω in 1:Nₑ
        x = x₀[ω]
        # Runge-Kutta 4 stages
        k1 = ∇V([x])[1]
        k2 = ∇V([x + Δt/2 * k1])[1]
        k3 = ∇V([x + Δt/2 * k2])[1]
        k4 = ∇V([x + Δt * k3])[1]
        # RK4 update
        x₀[ω] = x + Δt/6 * (k1 + 2*k2 + 2*k3 + k4)
        # Add stochastic noise
        x₀[ω] += ϵ * √Δt * 𝒩[ω]
    end
end

fig = Figure()
ax = Axis(fig[1, 1])
hist!(ax, x₀[:], bins = 100, normalization = :pdf)
display(fig)

# Save data to HDF5 file
h5write("potential_data_1D.hdf5", "data", x₀)

##
function V(x)
    return (x[1]^2 - 1)^2 / 4 + (x[2]^2 - 1)^2/4 + x[1] * x[2] /3
end
∇V(x) = -(Enzyme.gradient(Enzyme.Reverse, V, x)[1])
∇V([2.0, 2.0])

ϵ = sqrt(2) # 0.5
Nₜ = 1000
Nₑ = 10000
Δt = 0.1
x₀ = randn(2, Nₑ)
for t in ProgressBar(1:Nₜ)
    𝒩 = randn(2, Nₑ)
    for ω in 1:Nₑ
        x = x₀[:, ω]
        # Runge-Kutta 4 stages
        k1 = ∇V(x)
        k2 = ∇V(x + Δt / 2 * k1)
        k3 = ∇V(x + Δt / 2 * k2)
        k4 = ∇V(x + Δt * k3)
        # RK4 update
        x₀[:, ω] .= x + Δt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        # Add stochastic noise
        x₀[:, ω] .+= ϵ * √Δt * 𝒩[:, ω]
    end
end

h5write("potential_data_2D.hdf5", "data", x₀)

fig = Figure()
ax = Axis(fig[1, 1])
hist!(ax, x₀[1, :], bins=100, normalization=:pdf)
ax = Axis(fig[1, 2])
hist!(ax, x₀[2, :], bins=100, normalization=:pdf)
display(fig)

# Plot vector field with improved visualization
x_range = range(-3, 3, length=50)
y_range = range(-3, 3, length=50)
x_grid = [x for x in x_range, y in y_range]
y_grid = [y for x in x_range, y in y_range]

# Compute gradient vectors and potential at each grid point
u_grid = zeros(length(x_range), length(y_range))
v_grid = zeros(length(x_range), length(y_range))
V_grid = zeros(length(x_range), length(y_range))
magnitude_grid = zeros(length(x_range), length(y_range))
for (i, x) in enumerate(x_range)
    for (j, y) in enumerate(y_range)
        grad = ∇V([x, y])
        u_grid[i, j] = grad[1]
        v_grid[i, j] = grad[2]
        V_grid[i, j] = V([x, y])
        magnitude_grid[i, j] = sqrt(grad[1]^2 + grad[2]^2)
    end
end

# Normalize vectors for better visualization (scale by grid spacing)
dx = x_range[2] - x_range[1]
dy = y_range[2] - y_range[1]
scale_factor = min(dx, dy) * 0.8  # Increased to 0.8 to make arrows longer for tip visibility
u_scaled = u_grid .* scale_factor ./ (magnitude_grid .+ 1e-10)
v_scaled = v_grid .* scale_factor ./ (magnitude_grid .+ 1e-10)

fig = Figure(size=(900, 900))
ax = Axis(fig[1, 1], title="Vector Field ∇V", xlabel="x₁", ylabel="x₂", aspect=DataAspect())

# Plot potential as background heatmap
heatmap!(ax, x_range, y_range, V_grid, colormap=:viridis, alpha=0.6)

# Plot vector field using arrows - subsample for cleaner look
subsample = 1  # Use every arrow
x_arrows = x_grid[1:subsample:end, 1:subsample:end]
y_arrows = y_grid[1:subsample:end, 1:subsample:end]
u_arrows = u_scaled[1:subsample:end, 1:subsample:end]
v_arrows = v_scaled[1:subsample:end, 1:subsample:end]

# Create points and directions for arrows2d!
points = [Point2f(x_arrows[i, j], y_arrows[i, j]) for i in 1:size(x_arrows, 1), j in 1:size(x_arrows, 2)]
directions = [Point2f(u_arrows[i, j], v_arrows[i, j]) for i in 1:size(u_arrows, 1), j in 1:size(u_arrows, 2)]

# Use arrows2d! - tip size in data coordinates (larger values)
arrows2d!(ax, vec(points), vec(directions), 
          tipcolor=:white,
          shaftcolor=:white,
          shaftwidth=2.5,
          lengthscale=1.0,
          tipwidth=10.0,      # Much larger tip width in data coordinates
          tiplength=15.0,    # Much larger tip length in data coordinates
          normalize=false)  # Don't normalize, use directions as-is

display(fig)