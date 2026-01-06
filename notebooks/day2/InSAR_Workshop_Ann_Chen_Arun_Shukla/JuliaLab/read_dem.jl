function read_dem(dem_name,dem_length,dem_width)
    #
    # Alexandre Lasalarie
    #
    # Description:
    #       Reads the DEM.

    # Variables Definition:
    #   
    #   Inputs:
    #       dem_name        =   full path (absolute or relative) to DEM file
    #       dem_length      =   length of DEM file (in pixels)
    #       dem_width       =   width of DEM file (in pixels)
    #
    #   Outputs:
    #       dem             =   digital elevation model (DEM) as a (dem_length x dem_width) real-valued array

    dem = Array{Int16}(undef,dem_width,dem_length)
    open(dem_name,"r") do io
        read!(io, dem)
    end
    dem = transpose(dem)
    return dem

end
