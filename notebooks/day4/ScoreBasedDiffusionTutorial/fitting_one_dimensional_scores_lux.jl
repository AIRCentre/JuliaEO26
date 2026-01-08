using GLMakie, LinearAlgebra, ProgressBars, Random, Statistics
using Lux, Optimisers, Zygote

Random.seed!(1234)

# Activation function
swish(x) = x / (1 + exp(-x))

# ─────────────────────────────────────────────────────────────────────────────
# Score Functions
# ─────────────────────────────────────────────────────────────────────────────

function gaussian_mixture_score(x, data, sigma)
    m = length(data)
    score_value = 0.0
    denominator = 0.0
    for i in 1:m
        Δ = data[i] - x
        U = exp(-(0.5 / sigma^2) * Δ^2)
        score_value += U * Δ
        denominator += U
    end
    return score_value / (denominator * sigma^2)
end

# ─────────────────────────────────────────────────────────────────────────────
# Custom Lux Layer: OneLayerNetworkWithLinearByPass
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
# Loss Functions
# ─────────────────────────────────────────────────────────────────────────────

function gaussian_mixture_loss_function(model, ps, st, data, σ, zs)
    batchsize = length(data)
    lossval = 0.0
    for i in 1:batchsize
        x = [data[i] + σ * zs[i]]
        ŷ, _ = model(x, ps, st)
        y = gaussian_mixture_score(x[1], data, σ)
        lossval += (y - ŷ[1])^2 / batchsize
    end
    return lossval
end

function denoising_loss_function(model, ps, st, data, sigma, noises)
    batchsize = length(data)
    lossval = 0.0
    for i in 1:batchsize
        x = data[i]
        z = noises[i]
        x̃ = [x + sigma * z]
        ŷ, _ = model(x̃, ps, st)
        lossval += (ŷ[1] + z / sigma)^2 / batchsize
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
# Setup and Visualization
# ─────────────────────────────────────────────────────────────────────────────

sigma = σ = 0.1
data = randn(10000)
ys = range(-2, 2, length=100)
tmp = [gaussian_mixture_score(y, data, sigma) for y in ys]
exact_score = -ys

fig = Figure()
ax = Axis(fig[1, 1])
scatter!(ax, ys[:], tmp[:], color=:red, label="Gaussian Mixture Score")
scatter!(ax, ys[:], exact_score, color=:blue, label="Exact Score")
axislegend(ax, position=:rb)
display(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Define Network with Lux
# ─────────────────────────────────────────────────────────────────────────────

θr = reshape(data, (length(data), 1))
M = size(θr, 1)
Nθ = 1        # Input dimension
Nθᴴ = 4       # Hidden dimension

# Create model
rng = Random.MersenneTwister(1234)
model = OneLayerWithBypass(Nθ, Nθᴴ, 1; activation=swish)

# Initialize parameters and state
ps, st = Lux.setup(rng, model)

# Store initial parameters for smoothed averaging
ps_smoothed = deepcopy(ps)

# Helper functions for score evaluation
gmscore(x) = gaussian_mixture_score(x[1], data, sigma)
exactscore(x) = -x

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
        θbatch = [θr[x, 1] for x in permuted_list]
        zs = randn(length(θbatch))
        
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
        θbatch = [θr[x, 1] for x in permuted_list]
        zs = randn(length(θbatch))
        loss_value += denoising_loss_function(model, ps, st, θbatch, sigma, zs) / N
    end
    push!(test_loss_list, loss_value)
    
    # Exponential Moving Average of Parameters
    m = 0.9
    ps_smoothed = Lux.fmap((p_smooth, p) -> m * p_smooth + (1 - m) * p, ps_smoothed, ps)
end

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
    batchsize = length(data)
    lossval = 0.0
    for i in 1:batchsize
        x = data[i]
        z = noises[i]
        x̃ = [x + σ * z]
        ŷ = score_fn(x̃)
        ŷ_val = ŷ isa AbstractArray ? ŷ[1] : ŷ
        lossval += (ŷ_val + z / σ)^2 / batchsize
    end
    return lossval
end

network_score_fn(x) = model(x, ps, st)[1]
l1 = mean([denoising_loss_with_fn(network_score_fn, θr[:], σ, randn(M)) for _ in 1:100])
l2 = mean([denoising_loss_with_fn(gmscore, θr[:], σ, randn(M)) for _ in 1:100])
l3 = mean([denoising_loss_with_fn(exactscore, θr[:], σ, randn(M)) for _ in 1:100])

println("Network Loss: ", l1)
println("Gaussian Mixture Loss: ", l2)
println("Exact Loss: ", l3)

# ─────────────────────────────────────────────────────────────────────────────
# Plot Final Scores
# ─────────────────────────────────────────────────────────────────────────────

network_score = [model([y], ps, st)[1][1] for y in ys]
fig = Figure()
ax = Axis(fig[1, 1])
scatter!(ax, ys[:], tmp[:], color=:red, label="Gaussian Mixture Score")
scatter!(ax, ys[:], exact_score, color=:blue, label="Exact Score")
scatter!(ax, ys[:], network_score, color=:green, label="Network Score")
ylims!(ax, -2, 2)
axislegend(ax, position=:rt)
display(fig)

