function plot_phase(img,name)

    # Plots the phase
    phi = Array{Float32}(undef,size(img))
    phi.= (angle.(img).+pi)./(2*pi)
    phi.= phi.*254 .+ 1
    cmap = brew_igram()
    idx = clamp.(round.(Int, phi), 1, size(cmap,1))
    colors = RGB.(cmap[idx, 1], cmap[idx, 2], cmap[idx, 3])
    save(name,colors)

end
