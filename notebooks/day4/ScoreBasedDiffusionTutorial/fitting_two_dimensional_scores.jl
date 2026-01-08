using GLMakie, LinearAlgebra, ProgressBars, Random, Statistics, HDF5

Random.seed!(1234)

include("simple_networks.jl")

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
# Loss Functions (2D)
# ─────────────────────────────────────────────────────────────────────────────

function gaussian_mixture_loss_function(model, data, σ, zs)
    # data: 2×batchsize matrix
    # zs: 2×batchsize matrix of noise
    batchsize = size(data, 2)
    lossval = [0.0]
    for i in 1:batchsize
        x = data[:, i] .+ σ .* zs[:, i]
        ŷ = model(x)
        y = gaussian_mixture_score(x, data, σ)
        lossval[1] += sum((y .- ŷ).^2) / batchsize
    end
    return lossval[1]
end

function denoising_loss_function(model, data, sigma, noises)
    # data: 2×batchsize matrix
    # noises: 2×batchsize matrix of noise
    batchsize = size(data, 2)
    lossval = [0.0]
    for i in 1:batchsize
        x = data[:, i]
        z = noises[:, i]
        x̃ = x .+ sigma .* z
        ŷ = model(x̃)
        lossval[1] += sum((ŷ .+ z ./ sigma).^2) / batchsize
    end
    return lossval[1]
end

loss_function = denoising_loss_function

# ─────────────────────────────────────────────────────────────────────────────
# Load Data and Setup
# ─────────────────────────────────────────────────────────────────────────────

# Load 2D data from HDF5 file
data = h5read("potential_data_2D.hdf5", "data")  # 2×M matrix
M = size(data, 2)
println("Loaded $(M) samples of 2D data")

sigma = σ = 0.05

# Visualize data distribution
fig = Figure(size=(800, 400))
ax1 = Axis(fig[1, 1], title="x₁ Distribution")
hist!(ax1, data[1, :], bins=100, normalization=:pdf)
ax2 = Axis(fig[1, 2], title="x₂ Distribution")
hist!(ax2, data[2, :], bins=100, normalization=:pdf)
display(fig)

# Test score computation on a grid
ys = range(-2, 2, length=20)
test_point = [0.0, 0.0]
tmp = gaussian_mixture_score(test_point, data, sigma)
println("Score at origin: ", tmp)

# ─────────────────────────────────────────────────────────────────────────────
# Define Network
# ─────────────────────────────────────────────────────────────────────────────

Nθ = 2        # Input dimension (2D)
Nθᴴ = 50      # Hidden dimension
Nout = 2      # Output dimension (2D score)

network = OneLayerNetworkWithLinearByPass(Nθ, Nout, Nθᴴ)
dnetwork = deepcopy(network)
smoothed_network = deepcopy(network)

# Test loss computation
test_data = data[:, 1:10]
test_noise = randn(2, 10)
loss_function(network, test_data, σ, test_noise)

gmscore(x) = gaussian_mixture_score(x, data, sigma)
exactscore(x) = exact_score(x)

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

adam = Adam(network)
batchsize = 20
loss_list = Float64[]
test_loss_list = Float64[]
epochs = 200
network_parameters = copy(parameters(network))

for i in ProgressBar(1:epochs)
    shuffled_list = chunk_list(shuffle(1:2:M), batchsize)
    shuffled_test_list = chunk_list(shuffle(2:2:M), batchsize)
    loss_value = 0.0
    N = length(shuffled_list)
    
    # Batched Gradient Descent and Loss Evaluation
    for permuted_list in shuffled_list
        θbatch = data[:, permuted_list]  # 2×batchsize matrix
        zero!(dnetwork)
        zs = randn(2, length(permuted_list))
        autodiff(Enzyme.Reverse, loss_function, Active, DuplicatedNoNeed(network, dnetwork), Const(θbatch), Const(sigma), Const(zs))
        update!(adam, network, dnetwork)
        loss_value += loss_function(network, θbatch, sigma, zs) / N
    end
    push!(loss_list, loss_value)
    
    # Test Loss
    loss_value = 0.0
    N = length(shuffled_test_list)
    for permuted_list in shuffled_test_list
        θbatch = data[:, permuted_list]
        zs = randn(2, length(permuted_list))
        loss_value += loss_function(network, θbatch, sigma, zs) / N
    end
    push!(test_loss_list, loss_value)
    
    # Weighted Averaging of Network
    m = 0.9
    network_parameters .= m * network_parameters + (1 - m) * parameters(network)
    set_parameters!(smoothed_network, network_parameters)
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

network_score_fn(x) = predict(network, x)
skip = 50
number_of_samples = 10
l1 = mean([denoising_loss_with_fn(network_score_fn, data[:, 1:skip:end], σ, randn(2, M)) for _ in 1:number_of_samples])
l2 = mean([denoising_loss_with_fn(gmscore, data[:, 1:skip:end], σ, randn(2, M)) for _ in ProgressBar(1:number_of_samples)])
l3 = mean([denoising_loss_with_fn(exactscore, data[:, 1:skip:end], σ, randn(2, M)) for _ in 1:number_of_samples])

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
        net_score = predict(network, [x, y])
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
fig = Figure(size=(1800, 600))

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


# ─────────────────────────────────────────────────────────────────────────────
# Plot Score Magnitude Comparison
# ─────────────────────────────────────────────────────────────────────────────

network_mag = sqrt.(network_scores_u.^2 .+ network_scores_v.^2)
gm_mag = sqrt.(gm_scores_u.^2 .+ gm_scores_v.^2)
exact_mag = sqrt.(exact_scores_u.^2 .+ exact_scores_v.^2)


cmin, cmax = quantile(exact_mag, [0.01, 0.99])
ax1 = Axis(fig[2, 1], title="Network Score Magnitude", xlabel="x₁", ylabel="x₂", 
           aspect=DataAspect(), limits=(-2, 2, -2, 2))
hm1 = heatmap!(ax1, x_range, y_range, network_mag, colormap=:viridis, colorrange = (-cmin, cmax))

ax2 = Axis(fig[2, 2], title="GM Score Magnitude", xlabel="x₁", ylabel="x₂", 
           aspect=DataAspect(), limits=(-2, 2, -2, 2))
hm2 = heatmap!(ax2, x_range, y_range, gm_mag, colormap=:viridis, colorrange = (-cmin, cmax))

ax3 = Axis(fig[2, 3], title="Exact Score Magnitude", xlabel="x₁", ylabel="x₂", 
           aspect=DataAspect(), limits=(-2, 2, -2, 2))
hm3 = heatmap!(ax3, x_range, y_range, exact_mag, colormap=:viridis, colorrange = (-cmin, cmax))

display(fig)


## Drawing samples from the learned score field
∇V(x) = predict(network, x)

ϵ = sqrt(2) # 0.5
Nₜ = 1000
Nₑ = 1000
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

# compare 1D histograms of x₀ and data
binsize = 30
fig = Figure()
ax = Axis(fig[1, 1])
hist!(ax, x₀[1, :], bins=binsize, normalization=:pdf, color=(:blue, 0.5), label="Sampled Data")
hist!(ax, data[1, :], bins=binsize, normalization=:pdf, color=(:red, 0.5), label="Original Data")
axislegend(ax, position=:rt)
ax = Axis(fig[1, 2])
hist!(ax, x₀[2, :], bins=binsize, normalization=:pdf, color=(:blue, 0.5))
hist!(ax, data[2, :], bins=binsize, normalization=:pdf, color=(:red, 0.5))
display(fig)

# print statistics of x₀ and data
println("Mean of x₀: ", mean(x₀, dims = 2))
println("Mean of data: ", mean(data, dims = 2))
println("Covariance of x₀: ", cov(x₀, dims = 2))
println("Covariance of data: ", cov(data, dims = 2))