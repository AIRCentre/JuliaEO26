using GLMakie, LinearAlgebra, ProgressBars, Random, Statistics
using Lux, Optimisers, Zygote, HDF5

Random.seed!(1234)

# Activation function
swish(x) = x / (1 + exp(-x))

# ─────────────────────────────────────────────────────────────────────────────
# Score Functions (2D)
# ─────────────────────────────────────────────────────────────────────────────

# Exact score function from the potential V(x) = (x₁²-1)²/4 + (x₂²-1)²/4 + x₁x₂/3
# For p(x) ∝ exp(-V(x)), the score is ∇log p(x) = -∇V(x)
function exact_score(x)
    x1, x2 = x[1], x[2]
    # ∇V = [x₁³ - x₁ + x₂/3, x₂³ - x₂ + x₁/3]
    # Score = -∇V = [x₁ - x₁³ - x₂/3, x₂ - x₂³ - x₁/3]
    return [x1 - x1^3 - x2/3, x2 - x2^3 - x1/3]
end

function gaussian_mixture_score(x, data, sigma)
    # x: 2-element vector
    # data: 2×M matrix
    m = size(data, 2)
    score_value = zeros(2)
    denominator = 0.0
    for i in 1:m
        Δ = data[:, i] .- x
        U = exp(-(0.5 / sigma^2) * dot(Δ, Δ))
        score_value .+= U .* Δ
        denominator += U
    end
    return score_value ./ (denominator * sigma^2)
end

# ─────────────────────────────────────────────────────────────────────────────
# Custom Lux Layer: OneLayerNetworkWithLinearByPass (2D version)
# Architecture: output = (W2 * swish(W1 * x + b1) + b2) + (W3 * x + b3)
# ─────────────────────────────────────────────────────────────────────────────

struct OneLayerWithBypass{F} <: Lux.AbstractLuxLayer
    in_dims::Int
    hidden_dims::Int
    out_dims::Int
    activation::F
end

function OneLayerWithBypass(in_dims::Int, hidden_dims::Int, out_dims::Int; activation=swish)
    return OneLayerWithBypass{typeof(activation)}(in_dims, hidden_dims, out_dims, activation)
end

function Lux.initialparameters(rng::AbstractRNG, layer::OneLayerWithBypass)
    # Xavier-like initialization
    W1 = randn(rng, Float64, layer.hidden_dims, layer.in_dims) / sqrt(layer.in_dims)
    b1 = randn(rng, Float64, layer.hidden_dims) / sqrt(layer.in_dims)
    W2 = randn(rng, Float64, layer.out_dims, layer.hidden_dims) / sqrt(layer.hidden_dims)
    b2 = randn(rng, Float64, layer.out_dims) / sqrt(layer.hidden_dims)
    W3 = randn(rng, Float64, layer.out_dims, layer.in_dims) / sqrt(layer.in_dims)
    b3 = randn(rng, Float64, layer.out_dims) / sqrt(layer.in_dims)
    return (; W1, b1, W2, b2, W3, b3)
end

Lux.initialstates(::AbstractRNG, ::OneLayerWithBypass) = (;)

function (layer::OneLayerWithBypass)(x, ps, st)
    # Nonlinear path
    y1 = ps.W1 * x .+ ps.b1
    y2 = ps.W2 * layer.activation.(y1) .+ ps.b2
    # Linear bypass
    y3 = ps.W3 * x .+ ps.b3
    return y2 .+ y3, st
end

# ─────────────────────────────────────────────────────────────────────────────
# Loss Functions (2D)
# ─────────────────────────────────────────────────────────────────────────────

function gaussian_mixture_loss_function(model, ps, st, data, σ, zs)
    # data: 2×batchsize matrix
    # zs: 2×batchsize matrix of noise
    batchsize = size(data, 2)
    lossval = 0.0
    for i in 1:batchsize
        x = data[:, i] .+ σ .* zs[:, i]
        ŷ, _ = model(x, ps, st)
        y = gaussian_mixture_score(x, data, σ)
        lossval += sum((y .- ŷ).^2) / batchsize
    end
    return lossval
end

function denoising_loss_function(model, ps, st, data, sigma, noises)
    # data: 2×batchsize matrix
    # noises: 2×batchsize matrix of noise
    batchsize = size(data, 2)
    lossval = 0.0
    for i in 1:batchsize
        x = data[:, i]
        z = noises[:, i]
        x̃ = x .+ sigma .* z
        ŷ, _ = model(x̃, ps, st)
        lossval += sum((ŷ .+ z ./ sigma).^2) / batchsize
    end
    return lossval
end

# Wrapper for gradient computation (only differentiate w.r.t. ps)
function loss_wrapper(ps, model, st, data, sigma, noises)
    return denoising_loss_function(model, ps, st, data, sigma, noises)
end

# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

function chunk_list(list, n)
    return [list[i:min(i+n-1, length(list))] for i in 1:n:length(list)]
end

# ─────────────────────────────────────────────────────────────────────────────
# Load Data and Setup
# ─────────────────────────────────────────────────────────────────────────────

# Load 2D data from HDF5 file
data = h5read("potential_data_2D.hdf5", "data")  # 2×M matrix
M = size(data, 2)
println("Loaded $(M) samples of 2D data")

sigma = σ = 0.1

# Visualize data distribution
fig = Figure(size=(800, 400))
ax1 = Axis(fig[1, 1], title="x₁ Distribution")
hist!(ax1, data[1, :], bins=100, normalization=:pdf)
ax2 = Axis(fig[1, 2], title="x₂ Distribution")
hist!(ax2, data[2, :], bins=100, normalization=:pdf)
display(fig)

# Scatter plot of 2D data
fig2 = Figure()
ax = Axis(fig2[1, 1], title="2D Data Distribution", xlabel="x₁", ylabel="x₂")
scatter!(ax, data[1, :], data[2, :], markersize=2, alpha=0.3)
display(fig2)

# ─────────────────────────────────────────────────────────────────────────────
# Define Network with Lux
# ─────────────────────────────────────────────────────────────────────────────

Nθ = 2        # Input dimension (2D)
Nθᴴ = 32      # Hidden dimension (larger for 2D)
Nout = 2      # Output dimension (2D score)

# Create model
rng = Random.MersenneTwister(1234)
model = OneLayerWithBypass(Nθ, Nθᴴ, Nout; activation=swish)

# Initialize parameters and state
ps, st = Lux.setup(rng, model)

# Store initial parameters for smoothed averaging
ps_smoothed = deepcopy(ps)

# Helper functions for score evaluation
gmscore(x) = gaussian_mixture_score(x, data, sigma)
exactscore(x) = exact_score(x)

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

# Setup optimizer
opt = Optimisers.Adam(0.001)
opt_state = Optimisers.setup(opt, ps)

batchsize = 100
loss_list = Float64[]
test_loss_list = Float64[]
epochs = 1000

for i in ProgressBar(1:epochs)
    global opt_state, ps, ps_smoothed
    shuffled_list = chunk_list(shuffle(1:2:M), batchsize)
    shuffled_test_list = chunk_list(shuffle(2:2:M), batchsize)
    loss_value = 0.0
    N = length(shuffled_list)
    
    # Batched Gradient Descent and Loss Evaluation
    for permuted_list in shuffled_list
        θbatch = data[:, permuted_list]  # 2×batchsize matrix
        zs = randn(2, length(permuted_list))  # 2×batchsize noise
        
        # Compute gradient using Zygote
        grads = Zygote.gradient(p -> loss_wrapper(p, model, st, θbatch, sigma, zs), ps)[1]
        
        # Update parameters
        opt_state, ps = Optimisers.update(opt_state, ps, grads)
        
        loss_value += denoising_loss_function(model, ps, st, θbatch, sigma, zs) / N
    end
    push!(loss_list, loss_value)
    
    # Test Loss
    loss_value = 0.0
    N = length(shuffled_test_list)
    for permuted_list in shuffled_test_list
        θbatch = data[:, permuted_list]
        zs = randn(2, length(permuted_list))
        loss_value += denoising_loss_function(model, ps, st, θbatch, sigma, zs) / N
    end
    push!(test_loss_list, loss_value)
    
    # Exponential Moving Average of Parameters
    m = 0.9
    ps_smoothed = Lux.fmap((p_smooth, p) -> m * p_smooth + (1 - m) * p, ps_smoothed, ps)
end

@info "Training complete"

# ─────────────────────────────────────────────────────────────────────────────
# Plot Training Loss
# ─────────────────────────────────────────────────────────────────────────────

loss_fig = Figure()
ax = Axis(loss_fig[1, 1]; title="Log10 Loss", xlabel="Epoch", ylabel="Loss")
scatter!(ax, log10.(loss_list); color=:blue, label="Training Loss")
scatter!(ax, log10.(test_loss_list); color=:red, label="Test Loss")
axislegend(ax, position=:rt)
display(loss_fig)

# ─────────────────────────────────────────────────────────────────────────────
# Evaluate Final Losses
# ─────────────────────────────────────────────────────────────────────────────

function denoising_loss_with_fn(score_fn, data, σ, noises)
    # data: 2×batchsize matrix
    # noises: 2×batchsize matrix
    batchsize = size(data, 2)
    lossval = 0.0
    for i in 1:batchsize
        x = data[:, i]
        z = noises[:, i]
        x̃ = x .+ σ .* z
        ŷ = score_fn(x̃)
        lossval += sum((ŷ .+ z ./ σ).^2) / batchsize
    end
    return lossval
end

network_score_fn(x) = model(x, ps, st)[1]
skip = 100
l1 = mean([denoising_loss_with_fn(network_score_fn, data[:, 1:skip:end], σ, randn(2, M)) for _ in 1:100])
l2 = mean([denoising_loss_with_fn(gmscore, data[:, 1:skip:end], σ, randn(2, M)) for _ in ProgressBar(1:100)])
l3 = mean([denoising_loss_with_fn(exactscore, data[:, 1:skip:end], σ, randn(2, M)) for _ in 1:100])

println("Network Loss: ", l1)
println("Gaussian Mixture Loss: ", l2)
println("Exact Score Loss: ", l3)

# ─────────────────────────────────────────────────────────────────────────────
# Visualize Learned Score Field
# ─────────────────────────────────────────────────────────────────────────────

# Create grid for visualization
x_range = range(-2, 2, length=30)
y_range = range(-2, 2, length=30)

# Compute network score at each grid point
network_scores_u = zeros(length(x_range), length(y_range))
network_scores_v = zeros(length(x_range), length(y_range))
gm_scores_u = zeros(length(x_range), length(y_range))
gm_scores_v = zeros(length(x_range), length(y_range))
exact_scores_u = zeros(length(x_range), length(y_range))
exact_scores_v = zeros(length(x_range), length(y_range))

for (i, x) in ProgressBar(enumerate(x_range))
    for (j, y) in enumerate(y_range)
        net_score = model([x, y], ps, st)[1]
        network_scores_u[i, j] = net_score[1]
        network_scores_v[i, j] = net_score[2]
        
        gm_score = gaussian_mixture_score([x, y], data, sigma)
        gm_scores_u[i, j] = gm_score[1]
        gm_scores_v[i, j] = gm_score[2]
        
        exact_score_val = exact_score([x, y])
        exact_scores_u[i, j] = exact_score_val[1]
        exact_scores_v[i, j] = exact_score_val[2]
    end
end

# Normalize for visualization
function normalize_vectors(u, v, scale=0.15)
    mag = sqrt.(u.^2 .+ v.^2)
    mag[mag .< 1e-10] .= 1e-10
    return u .* scale ./ mag, v .* scale ./ mag
end

net_u_scaled, net_v_scaled = normalize_vectors(network_scores_u, network_scores_v)
gm_u_scaled, gm_v_scaled = normalize_vectors(gm_scores_u, gm_scores_v)
exact_u_scaled, exact_v_scaled = normalize_vectors(exact_scores_u, exact_scores_v)

# Create grid points
x_grid = [x for x in x_range, y in y_range]
y_grid = [y for x in x_range, y in y_range]

# Plot comparison
fig = Figure(size=(1800, 600), figure_padding=(10, 10, 10, 10))

# Network score field
ax1 = Axis(fig[1, 1], title="Network Score Field (Loss: $(round(l1, digits=1)))", xlabel="x₁", ylabel="x₂", 
           aspect=DataAspect(), limits=(-2, 2, -2, 2))
scatter!(ax1, data[1, 1:100:end], data[2, 1:100:end], markersize=3, alpha=0.3, color=:gray)
arrows!(ax1, vec(x_grid), vec(y_grid), vec(net_u_scaled), vec(net_v_scaled), 
        color=:blue, linewidth=1.5, arrowsize=8)

# Gaussian mixture score field
ax2 = Axis(fig[1, 2], title="Gaussian Mixture Score Field (Loss: $(round(l2, digits=1)))", xlabel="x₁", ylabel="x₂", 
           aspect=DataAspect(), limits=(-2, 2, -2, 2))
scatter!(ax2, data[1, 1:100:end], data[2, 1:100:end], markersize=3, alpha=0.3, color=:gray)
arrows!(ax2, vec(x_grid), vec(y_grid), vec(gm_u_scaled), vec(gm_v_scaled), 
        color=:red, linewidth=1.5, arrowsize=8)

# Exact score field
ax3 = Axis(fig[1, 3], title="Exact Score Field (Loss: $(round(l3, digits=1)))", xlabel="x₁", ylabel="x₂", 
           aspect=DataAspect(), limits=(-2, 2, -2, 2))
scatter!(ax3, data[1, 1:100:end], data[2, 1:100:end], markersize=3, alpha=0.3, color=:gray)
arrows!(ax3, vec(x_grid), vec(y_grid), vec(exact_u_scaled), vec(exact_v_scaled), 
        color=:green, linewidth=1.5, arrowsize=8)

display(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Plot Score Magnitude Comparison
# ─────────────────────────────────────────────────────────────────────────────

network_mag = sqrt.(network_scores_u.^2 .+ network_scores_v.^2)
gm_mag = sqrt.(gm_scores_u.^2 .+ gm_scores_v.^2)
exact_mag = sqrt.(exact_scores_u.^2 .+ exact_scores_v.^2)

fig_mag = Figure(size=(1800, 600), figure_padding=(10, 10, 10, 10))
ax1 = Axis(fig_mag[1, 1], title="Network Score Magnitude (Loss: $(round(l1, digits=1)))", xlabel="x₁", ylabel="x₂", 
           aspect=DataAspect(), limits=(-2, 2, -2, 2))
hm1 = heatmap!(ax1, x_range, y_range, log.(network_mag), colormap=:viridis)
Colorbar(fig_mag[1, 2], hm1)

ax2 = Axis(fig_mag[1, 3], title="GM Score Magnitude (Loss: $(round(l2, digits=1)))", xlabel="x₁", ylabel="x₂", 
           aspect=DataAspect(), limits=(-2, 2, -2, 2))
hm2 = heatmap!(ax2, x_range, y_range, log.(gm_mag), colormap=:viridis)
Colorbar(fig_mag[1, 4], hm2)

ax3 = Axis(fig_mag[1, 5], title="Exact Score Magnitude (Loss: $(round(l3, digits=1)))", xlabel="x₁", ylabel="x₂", 
           aspect=DataAspect(), limits=(-2, 2, -2, 2))
hm3 = heatmap!(ax3, x_range, y_range, log.(exact_mag), colormap=:viridis)
Colorbar(fig_mag[1, 6], hm3)

display(fig_mag)

