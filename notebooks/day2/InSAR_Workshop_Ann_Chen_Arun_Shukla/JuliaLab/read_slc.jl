function read_slc(slc_name,slc_length,slc_width)
    #
    # Alexandre Lasalarie
    #
    # Description:
    #       Reads a single SLC file.

    # Variables Definition:
    #   
    #   Inputs:
    #       slc_name        =   full path (absolute or relative) to SLC file
    #       slc_length      =   length of SLC file (in pixels)
    #       slc_width       =   width of SLC file (in pixels)
    #
    #   Outputs:
    #       slc             =   single-look complex signal (2D complex-valued array)

    slc = Array{ComplexF32}(undef,slc_length,slc_width)
    data = Array{Float32}(undef,2*slc_width,slc_length)
    open(slc_name,"r") do io
        read!(io,data)
    end
    data = transpose(data)
    slc .= complex.(data[:,1:2:end],data[:,2:2:end])
    return slc
end


