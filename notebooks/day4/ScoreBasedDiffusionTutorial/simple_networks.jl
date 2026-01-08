using Enzyme, Random

abstract type SimpleNetwork end

struct OneLayerNetwork{M, V} <: SimpleNetwork
    W1::M
    b1::V
    W2::M
    b2::V
end

struct OneLayerGradientNetwork{M, V} <: SimpleNetwork
    W1::M
    b1::V
    W2::M
    b2::V
    WG::M 
    bG::V
end

struct MultiLayerNetwork{M, T, V} <: SimpleNetwork
    W1::M
    b1::V
    WH::T
    bH::M
    W2::M
    b2::V
end

struct OneLayerNetworkWithLinearByPass{M,V} <: SimpleNetwork
    W1::M
    b1::V
    W2::M
    b2::V
    W3::M
    b3::V
end

struct SimpleUnet{M, V} <: SimpleNetwork
    Wu::M
    bu::V
    Wd::M
    bd::V
    Wu2::M
    bu2::V 
    Wd2::M
    bd2::V
end

function zero!(dnetwork::OneLayerNetworkWithLinearByPass)
    dnetwork.W1 .= 0.0
    dnetwork.b1 .= 0.0
    dnetwork.W2 .= 0.0
    dnetwork.b2 .= 0.0
    dnetwork.W3 .= 0.0
    dnetwork.b3 .= 0.0
    return nothing
end

function zero!(dnetwork::OneLayerGradientNetwork)
    dnetwork.W1 .= 0.0
    dnetwork.b1 .= 0.0
    dnetwork.W2 .= 0.0
    dnetwork.b2 .= 0.0
    dnetwork.WG .= 0.0
    dnetwork.bG .= 0.0
    return nothing
end

function zero!(dnetwork::OneLayerNetwork)
    dnetwork.W1 .= 0.0
    dnetwork.b1 .= 0.0
    dnetwork.W2 .= 0.0
    dnetwork.b2 .= 0.0
    return nothing
end

function zero!(dnetwork::MultiLayerNetwork)
    dnetwork.W1 .= 0.0
    dnetwork.b1 .= 0.0
    dnetwork.WH .= 0.0
    dnetwork.bH .= 0.0
    dnetwork.W2 .= 0.0
    dnetwork.b2 .= 0.0
    return nothing
end

function zero!(dnetwork::SimpleUnet)
    dnetwork.Wu .= 0.0
    dnetwork.bu .= 0.0
    dnetwork.Wd .= 0.0
    dnetwork.bd .= 0.0
    dnetwork.Wu2 .= 0.0
    dnetwork.bu2 .= 0.0
    dnetwork.Wd2 .= 0.0
    dnetwork.bd2 .= 0.0
    return nothing
end

function update!(network::SimpleNetwork, dnetwork::SimpleNetwork, η)
    network.W1 .-= η .* dnetwork.W1
    network.b1 .-= η .* dnetwork.b1
    network.W2 .-= η .* dnetwork.W2
    network.b2 .-= η .* dnetwork.b2
    return nothing
end

swish(x) = x / (1 + exp(-x))
activation_function(x) = swish(x) # swish(x) # 

function predict(network::OneLayerNetwork, x)
    return network.W2 * activation_function.(network.W1 * x .+ network.b1) .+ network.b2
end
# could directly write the potential function 
# denote the anti-derivative of the activation by h(x)
# potential = W2 * h(W1 * x + b1) + 0.5 * x' * (network.WG + network.WG') * x + network.bG' * x
function predict(network::OneLayerGradientNetwork, x)
    nonlinear_part = activation_function.(network.W1 * x .+ network.b1) # size d-batch, W1 is of size d by d_in
    # gaussian_part = (network.WG + network.WG') * x 
    gradient_part = network.W1' * (network.W2' .* nonlinear_part) 
    return gradient_part # + gaussian_part
end

function predict(network::OneLayerNetworkWithLinearByPass, x)
    y1 = network.W1 * x .+ network.b1
    y2 = network.W2 * activation_function.(y1) .+ network.b2
    y3 = network.W3 * x .+ network.b3
    return y3 .+ y2
end

function predict(network::MultiLayerNetwork, x)
    y = network.W1 * x .+ network.b1
    M = size(network.WH, 3)
    dt = sqrt(1/M) # sqrt(1/M)
    for i in 1:M
            y = (1-dt) * y + (network.WH[:, :, i] * activation_function.(y) .+ network.bH[:, i]) * dt
    end
    y = network.W2 * y .+ network.b2
    return y
end

function predict(network::SimpleUnet, x)
    y = network.Wu * x .+ network.bu
    yu = activation_function.(y)
    yd = network.Wd * yu .+ network.bd
    yd = activation_function.(yd)
    yu2 = network.Wu2 * yd .+ network.bu2
    yu2 = activation_function.(yu2)
    y = cat(yu, yu2, dims = 1)
    y = network.Wd2 * y .+ network.bd2
    return y
end

function OneLayerNetwork(inchannel::Int64, outchannel::Int64, hiddenchannel::Int64; xavier = true)
    if xavier
        W1 = randn(hiddenchannel, inchannel) / sqrt(inchannel)
        b1 = randn(hiddenchannel) / sqrt(inchannel)
        W2 = randn(outchannel, hiddenchannel) / sqrt(hiddenchannel)
        b2 = randn(outchannel) / sqrt(hiddenchannel)
    else
        # He initialization
        W1 = randn(hiddenchannel, inchannel) * sqrt(2 / hiddenchannel)
        b1 = randn(hiddenchannel) * sqrt(2 / hiddenchannel)
        W2 = randn(outchannel, hiddenchannel) * sqrt(2 / hiddenchannel)
        b2 = randn(outchannel) * sqrt(2 / hiddenchannel)
    end
    return OneLayerNetwork(W1, b1, W2, b2)
end

function OneLayerGradientNetwork(inchannel::Int64, hiddenchannel::Int64; xavier = true)
    if xavier
        W1 = randn(hiddenchannel, inchannel) / sqrt(inchannel)
        b1 = randn(hiddenchannel) / sqrt(inchannel)
        W2 = randn(1, hiddenchannel) / sqrt(hiddenchannel)
        b2 = randn(1) / sqrt(hiddenchannel)
        WG = randn(inchannel, inchannel) / sqrt(inchannel)
        bG = randn(inchannel) / sqrt(inchannel)
    else
        W1 = randn(hiddenchannel, inchannel) * sqrt(2 / hiddenchannel)
        b1 = randn(hiddenchannel) * sqrt(2 / hiddenchannel)
        W2 = randn(1, hiddenchannel) * sqrt(2 / hiddenchannel)
        b2 = randn(1) * sqrt(2 / hiddenchannel)
        WG = randn(inchannel, inchannel) * sqrt(2 / inchannel)
        bG = randn(inchannel) * sqrt(2 / inchannel)
    end
    return OneLayerGradientNetwork(W1, b1, W2, b2, WG, bG)
end

function OneLayerNetworkWithLinearByPass(inchannel::Int64, outchannel::Int64, hiddenchannel::Int64; xavier = true)
    # Xavier initialization
    if xavier
        W1 = randn(hiddenchannel, inchannel) / sqrt(inchannel)
        b1 = randn(hiddenchannel) / sqrt(inchannel)
        W2 = randn(outchannel, hiddenchannel) / sqrt(hiddenchannel)
        b2 = randn(outchannel) / sqrt(hiddenchannel)
        W3 = randn(outchannel, inchannel) / sqrt(inchannel)
        b3 = randn(outchannel) / sqrt(inchannel)
    else
        # He initialization
        W1 = randn(hiddenchannel, inchannel) * sqrt(2 / hiddenchannel)
        b1 = randn(hiddenchannel) * sqrt(2 / hiddenchannel)
        W2 = randn(outchannel, hiddenchannel) * sqrt(2 / hiddenchannel)
        b2 = randn(outchannel) * sqrt(2 / hiddenchannel)
        W3 = randn(outchannel, inchannel) * sqrt(2 / inchannel)
        b3 = randn(outchannel) * sqrt(2 / inchannel)
    end
    return OneLayerNetworkWithLinearByPass(W1, b1, W2, b2, W3, b3)
end

"""
MultiLayerNetwork(inchannel::Int64, outchannel::Int64, hiddenchannel::Int64, depth::Int64; xavier = true)
"""
function MultiLayerNetwork(inchannel::Int64, outchannel::Int64, hiddenchannel::Int64, depth::Int64; xavier = true)
    if xavier
        W1 = randn(hiddenchannel, inchannel) / sqrt(inchannel)
        b1 = randn(hiddenchannel) / sqrt(inchannel)
        WH = randn(hiddenchannel, hiddenchannel, depth) / sqrt(hiddenchannel)
        bH = randn(hiddenchannel, depth) / sqrt(hiddenchannel)
        W2 = randn(outchannel, hiddenchannel) / sqrt(hiddenchannel)
        b2 = randn(outchannel) / sqrt(hiddenchannel)
    else
        # He initialization
        W1 = randn(hiddenchannel, inchannel) * sqrt(2 / inchannel)
        b1 = randn(hiddenchannel) * sqrt(2 / inchannel)
        WH = randn(hiddenchannel, hiddenchannel, depth) * sqrt(2 / hiddenchannel)
        bH = randn(hiddenchannel, depth) * sqrt(2 / hiddenchannel)
        W2 = randn(outchannel, hiddenchannel) * sqrt(2 / hiddenchannel)
        b2 = randn(outchannel) * sqrt(2 / hiddenchannel)
    end
    return MultiLayerNetwork(W1, b1, WH, bH, W2, b2)
end

function SimpleUnet(inchannel::Int64, outchannel::Int64, hiddenchannel::Int64, hiddenchannel2::Int64, hiddenchannel3::Int64; xavier = true)
    if xavier
        Wu = randn(hiddenchannel, inchannel) / sqrt(inchannel)
        bu = randn(hiddenchannel) / sqrt(inchannel)
        Wd = randn(hiddenchannel2, hiddenchannel) / sqrt(hiddenchannel)
        bd = randn(hiddenchannel2) / sqrt(hiddenchannel2)
        Wu2 = randn(hiddenchannel3, hiddenchannel2) / sqrt(hiddenchannel2)
        bu2 = randn(hiddenchannel3) / sqrt(hiddenchannel2)
        Wd2 = randn(outchannel, hiddenchannel3 + hiddenchannel) / sqrt(hiddenchannel3 + hiddenchannel)
        bd2 = randn(outchannel) / sqrt(outchannel)
    end
    return SimpleUnet(Wu, bu, Wd, bd, Wu2, bu2, Wd2, bd2)
end

function predict(network::OneLayerNetwork, x, activation::Function)
    return abs.(network.W2 * activation.(network.W1 * x .+ network.b1) .+ network.b2)
end

function (network::SimpleNetwork)(x)
    return predict(network, x)
end

function predict_scalar(network::SimpleNetwork, x)
    return predict(network, x)[1]
end

function loss(network::SimpleNetwork, x, y)
    ŷ = similar(y)
    for i in eachindex(ŷ)
        ŷ[i] = predict(network, x[i])[1]
    end
    return mean((y .- ŷ) .^ 2)
end

function chunk_list(list, n)
    return [list[i:min(i+n-1, length(list))] for i in 1:n:length(list)]
end

struct Adam{S, T, I}
    struct_copies::S
    parameters::T 
    t::I
end

struct AdamW{S, T, I}
    struct_copies::S
    parameters::T 
    t::I
end

function parameters(network::SimpleNetwork)
    network_parameters = []
    for names in propertynames(network)
        push!(network_parameters, getproperty(network, names)[:])
    end
    param_lengths = [length(params) for params in network_parameters]
    parameter_list = zeros(sum(param_lengths))
    start = 1
    for i in 1:length(param_lengths)
        parameter_list[start:start+param_lengths[i]-1] .= network_parameters[i]
        start += param_lengths[i]
    end
    return parameter_list
end

function set_parameters!(network::SimpleNetwork, parameters_list)
    param_lengths = Int64[]
    for names in propertynames(network)
        push!(param_lengths, length(getproperty(network, names)[:]))
    end
    start = 1
    for (i, names) in enumerate(propertynames(network))
        getproperty(network, names)[:] .= parameters_list[start:start+param_lengths[i]-1]
        start = start + param_lengths[i]
    end
    return nothing
end

function Adam(network::SimpleNetwork; α=0.001, β₁=0.9, β₂=0.999, ϵ=1e-8)
    parameters_list = (; α, β₁, β₂, ϵ)
    network_parameters = parameters(network)
    t = [1.0]
    θ  = deepcopy(network_parameters) .* 0.0
    gₜ = deepcopy(network_parameters) .* 0.0
    m₀ = deepcopy(network_parameters) .* 0.0
    mₜ = deepcopy(network_parameters) .* 0.0
    v₀ = deepcopy(network_parameters) .* 0.0
    vₜ = deepcopy(network_parameters) .* 0.0
    v̂ₜ = deepcopy(network_parameters) .* 0.0
    m̂ₜ = deepcopy(network_parameters) .* 0.0
    struct_copies = (; θ, gₜ, m₀, mₜ, v₀, vₜ, v̂ₜ, m̂ₜ)
    return Adam(struct_copies, parameters_list,  t)
end

function AdamW(network::SimpleNetwork; α=0.001, β₁=0.9, β₂=0.999, ϵ=1e-8, λ=0.0001)
    parameters_list = (; α, β₁, β₂, ϵ, λ)
    network_parameters = parameters(network)
    t = [1.0]
    θ  = deepcopy(network_parameters) .* 0.0
    gₜ = deepcopy(network_parameters) .* 0.0
    m₀ = deepcopy(network_parameters) .* 0.0
    mₜ = deepcopy(network_parameters) .* 0.0
    v₀ = deepcopy(network_parameters) .* 0.0
    vₜ = deepcopy(network_parameters) .* 0.0
    v̂ₜ = deepcopy(network_parameters) .* 0.0
    m̂ₜ = deepcopy(network_parameters) .* 0.0
    struct_copies = (; θ, gₜ, m₀, mₜ, v₀, vₜ, v̂ₜ, m̂ₜ)
    return AdamW(struct_copies, parameters_list,  t)
end

function update!(adam::Adam, network::SimpleNetwork, dnetwork::SimpleNetwork)
    # unpack
    (; α, β₁, β₂, ϵ) = adam.parameters
    t = adam.t[1]
    (; θ, gₜ, m₀, mₜ, v₀, vₜ, v̂ₜ, m̂ₜ) = adam.struct_copies
    t = adam.t[1]
    # get gradient
    θ .= parameters(network)
    gₜ .= parameters(dnetwork)
    # update
    @. mₜ = β₁ * m₀ + (1 - β₁) * gₜ
    @. vₜ = β₂ * v₀ + (1 - β₂) * (gₜ .^2)
    @. m̂ₜ = mₜ / (1 - β₁^t)
    @. v̂ₜ = vₜ / (1 - β₂^t)
    @. θ = θ - α * m̂ₜ / (sqrt(v̂ₜ) + ϵ)
    # update parameters
    m₀ .= mₜ
    v₀ .= vₜ
    adam.t[1] += 1
    set_parameters!(network, θ)
    return nothing
end

function update!(adamw::AdamW, network::SimpleNetwork, dnetwork::SimpleNetwork)
    (; α, β₁, β₂, ϵ, λ) = adamw.parameters
    t = adamw.t[1]
    (; θ, gₜ, m₀, mₜ, v₀, vₜ, v̂ₜ, m̂ₜ) = adamw.struct_copies
    t = adamw.t[1]
    # get gradient
    θ .= parameters(network)
    gₜ .= parameters(dnetwork)
    # update
    @. mₜ = β₁ * m₀ + (1 - β₁) * gₜ
    @. vₜ = β₂ * v₀ + (1 - β₂) * (gₜ .^2)
    @. m̂ₜ = mₜ / (1 - β₁^t)
    @. v̂ₜ = vₜ / (1 - β₂^t)
    # AdamW: decouple weight decay from gradient-based updates
    @. θ = θ - α * (m̂ₜ / (sqrt(v̂ₜ) + ϵ) + λ * θ)
    # update parameters
    m₀ .= mₜ
    v₀ .= vₜ
    adamw.t[1] += 1
    set_parameters!(network, θ)
    return nothing
end