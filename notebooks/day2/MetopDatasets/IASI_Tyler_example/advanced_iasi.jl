using Tyler, GLMakie, MetopDatasets

iasi_file = "IASI_xxx_1C_M01_20240925202059Z_20240925220258Z_N_O_20240925211316Z.nat"
println("Total file size ", filesize(iasi_file) /10^9)

ds = MetopDataset(iasi_file, maskingvalue=NaN);

# read geolocation points and data shape
pts, pts_original_size = let
    lon_lat = ds["ggeosondloc"][:,:,:,:]

    # make sure longitude is from -180 to 180
    lon_lat[1, :,:,:][ lon_lat[1,:,:,:] .>180] .-= 360

    # convert the points to tuples
    lon_lat = tuple.(lon_lat[1, :,:,:],lon_lat[2, :,:,:])

    # store the original shape of the points
    pts_original_size = size(lon_lat)

    # Flatten the points and convert them to web_mercator (the coordinate system used by Tyler)
    MT = Tyler.MapTiles
    pts = [Point2f(MT.project(lon_lat[i], MT.wgs84, MT.web_mercator)) 
            for i in eachindex(lon_lat)]
    
    pts, pts_original_size
end;

# read cloud fraction
cloud_fraction = Float32.(ds["geumavhrr1bcldfrac"][:]);

# helper function to read the spectrum for a single point
function read_spectrum_pts(ds, index::CartesianIndex)
    # read spectrum and Wavenumber
    spectrum = ds["gs1cspect"][:,Tuple(index)...]
    wavenumber = ds["spectra_wavenumber"][:, Tuple(index)[end]]

    # covert to brightness temperature
    T_B = brightness_temperature.(spectrum, wavenumber)
    wavenumber_cm = wavenumber./100

    # join brightness temperature and Wavenumber to points
    spectrum_pts = Point2f.(tuple.(wavenumber_cm, T_B))
    return spectrum_pts
end

# read an initial spectrum
spectrum_pts = read_spectrum_pts(ds, CartesianIndex(1,1,1));
spectrum_pts2 = read_spectrum_pts(ds, CartesianIndex(2,1,1));

# create the inter active plot.
fig = let
    fig = Figure()
    
    # select background map and initial zoom
    provider = Tyler.TileProviders.Esri(:WorldImagery);
    extent = Tyler.Extent(X = (-10, 10), Y = (-10, 10));

    # create background map
    ax1 = Axis(fig[1,1])
    m = Tyler.Map(extent; provider, figure=fig, axis=ax1);
    wait(m);

    # Plot observation points with cloud cover
    objscatter = scatter!(ax1, pts, color = cloud_fraction, 
        colorrange = (0,100), colormap=:grays, markersize=15)
    # hack from https://github.com/MakieOrg/Tyler.jl/issues/109
    translate!(objscatter, 0, 0, 10) 

    # Plot a red cross on top of the selected point
    selected_point = Observable(pts[1:1])
    selected_scatter = scatter!(ax1, selected_point, 
        color = :green, markersize=15, marker =:xcross)

    selected_point2 = Observable(pts[2:2])
    selected_scatter2 = scatter!(ax1, selected_point2, 
        color = :blue, markersize=15, marker =:xcross)

    # hack from https://github.com/MakieOrg/Tyler.jl/issues/109
    translate!(selected_scatter, 0, 0, 11)
    translate!(selected_scatter2, 0, 0, 11)

    # Add colorbar
    Colorbar(fig[1, 2], limits = (0.0,100.0), colormap =:grays, label = "Cloud Fraction")
    hidedecorations!(ax1)

    # Create the second plot for the spectrum
    ax2 = Axis(fig[2, 1],
        title = "Observation spectrum",
        xlabel = "Wavenumber (cm-1)",
        ylabel = "Brightness Temperature (K)")

    # plot the spectrum
    spectrum_observable = Observable(spectrum_pts)
    lines!(ax2, spectrum_observable, color = :green)

    spectrum_observable2 = Observable(spectrum_pts2)
    lines!(ax2, spectrum_observable2, color = :blue)
    
    first_point = true

    # Add event handler to update the plot when the user click on a new observation 
    obs_func = on(events(m.axis).mousebutton) do event
        if event.button == Mouse.left && event.action == Mouse.press
            plt, i = pick(m.axis)
            if plt == objscatter # check if an observation was clicked
                # get the CartesianIndex
                cartesian_i = CartesianIndices(pts_original_size)[i]
                # load the selected spectrum and update the spectrum plot
                if first_point
                    spectrum_observable[] = read_spectrum_pts(ds, cartesian_i)
                    # update the green x
                    selected_point[] = pts[i:i]
                    first_point = false
                else
                    spectrum_observable2[] = read_spectrum_pts(ds, cartesian_i)
                    # update the blue x
                    selected_point2[] = pts[i:i]
                    first_point = true
                end

            end
        end
    end

    fig
end
