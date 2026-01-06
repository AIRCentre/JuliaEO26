function plot_amp(img,name)
    A = abs.(img)
    q1 = quantile(vec(A), 0.01)
    q2 = quantile(vec(A), 0.99)
    rng = (q2 - q1) / 150
    A .= (A .- q1) ./ rng .+ 50
    A .= A./255
    A .= clamp01.(A)
    img = colorview(Gray, A)
    save(name,img)
end