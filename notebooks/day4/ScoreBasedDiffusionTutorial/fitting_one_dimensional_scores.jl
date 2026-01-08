using GLMakie, LinearAlgebra, ProgressBars, Random, Statistics

Random.seed!(1234)

include("simple_networks.jl")

function gaussian_mixture_score(x, data, sigma)
    m = length(data)
    score_value = zeros(1)
    denominator = [0.0]
    for i in 1:m
        Δ = data[i] - x
        U = exp(-(0.5 / sigma^2) * Δ' * Δ)
        weightedU = U
        score_value .+= weightedU * Δ
        denominator .+= weightedU
    end
    return score_value / (denominator[1] * sigma^2)
end

function gaussian_mixture_loss_function(model, data, σ, zs)
    batchsize = length(data)
    lossval = [0.0]
    for i in 1:batchsize
        x = data[i] .+ σ * zs[i]
        ŷ = model(x)[1]
        y = gaussian_mixture_score(x, data, σ)[1]
        lossval[1] += (y - ŷ)^2 / batchsize
    end
    return lossval[1]
end

function denoising_loss_function(model, data, sigma, noises)
    batchsize = length(data)
    lossval = [0.0]
    for i in 1:batchsize
        x = data[i]
        z = noises[i]
        x̃ = x .+ sigma * z
        ŷ = model(x̃)[1]
        lossval[1] += (ŷ + z / sigma)^2 / batchsize
    end
    return lossval[1]
end

loss_function = denoising_loss_function #  gaussian_mixture_loss_function #               

sigma = σ = 0.1
data = randn(10000)
ys = range(-2, 2, length=100)
tmp = [gaussian_mixture_score(i, data, sigma)[1] for i in ys]
exact_score = -ys

fig = Figure()
ax = Axis(fig[1, 1])
scatter!(ax, ys[:], tmp[:], color=:red, label="Gaussian Mixture Score")
scatter!(ax, ys[:], exact_score, color=:blue, label="Exact Score")
axislegend(ax, position=:rb)
display(fig)


# Define Network
θr = reshape(data, (length(data), 1))
M = size(θr, 1)
Nθ = size(θr, 2)
Nθᴴ = 4
W1 = randn(Nθᴴ, Nθ)
b1 = randn(Nθᴴ)
W2 = randn(1, Nθᴴ)
b2 = randn(1)
W3 = randn(1, Nθ)
b3 = randn(1)

network = OneLayerNetworkWithLinearByPass(W1, b1, W2, b2, W3, b3)
dnetwork = deepcopy(network)
smoothed_network = deepcopy(network)
loss_function(network, θr[:], σ, randn(M))
gmscore(x) = gaussian_mixture_score(x, data, sigma)
exactscore(x) = -x

gaussian_mixture_loss_function(network, θr[:], σ, randn(M))

adam = Adam(network)
batchsize = 100
loss_list = Float64[]
test_loss_list = Float64[]
epochs = 1000
network_parameters = copy(parameters(network))
for i in ProgressBar(1:epochs)
    shuffled_list = chunk_list(shuffle(1:2:M), batchsize)
    shuffled_test_list = chunk_list(shuffle(2:2:M), batchsize)
    loss_value = 0.0
    N = length(shuffled_list)
    # Batched Gradient Descent and Loss Evaluation
    for permuted_list in shuffled_list
        θbatch = [θr[x, :][1] for x in permuted_list]
        zero!(dnetwork)
        zs = randn(length(θbatch))
        autodiff(Enzyme.Reverse, loss_function, Active, DuplicatedNoNeed(network, dnetwork), Const(θbatch), Const(sigma), Const(zs))
        update!(adam, network, dnetwork)
        loss_value += loss_function(network, θbatch, sigma, zs) / N
    end
    push!(loss_list, loss_value)
    # Test Loss
    loss_value = 0.0
    N = length(shuffled_test_list)
    for permuted_list in shuffled_test_list
        θbatch = [θr[x, :] for x in permuted_list]
        zs = randn(length(θbatch))
        loss_value += loss_function(network, θbatch, sigma, zs) / N
    end
    push!(test_loss_list, loss_value)
    # Weighted Averaging of Network
    m = 0.9
    network_parameters .= m * network_parameters + (1 - m) * parameters(network)
    set_parameters!(smoothed_network, network_parameters)
end

loss_fig = Figure()
ax = Axis(loss_fig[1, 1]; title="Log10 Loss", xlabel="Epoch", ylabel="Loss")
scatter!(ax, log10.(loss_list); color=:blue, label="Training Loss")
scatter!(ax, log10.(test_loss_list); color=:red, label="Test Loss")
axislegend(ax, position=:rt)
display(loss_fig)


l1 = mean([denoising_loss_function(network, θr[:], σ, randn(M)) for i in 1:100])
l2 = mean([denoising_loss_function(gmscore, θr[:], σ, randn(M)) for i in 1:100])
l3 = mean([denoising_loss_function(exactscore, θr[:], σ, randn(M)) for i in 1:100])

println("Network Loss: ", l1)
println("Gaussian Mixture Loss: ", l2)
println("Exact Loss: ", l3)


network_score = [predict(network, i)[1] for i in ys]
fig = Figure()
ax = Axis(fig[1, 1])
scatter!(ax, ys[:], tmp[:], color=:red, label="Gaussian Mixture Score")
scatter!(ax, ys[:], exact_score, color=:blue, label="Exact Score")
scatter!(ax, ys[:], network_score, color=:green, label="Network Score")
ylims!(ax, -2, 2)
axislegend(ax, position=:rt)
display(fig)