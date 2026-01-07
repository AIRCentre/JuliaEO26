
using NCDatasets, Suppressor, GLMakie

CHANNEL_NAMES_1KM = ("vis_04", "vis_05", "vis_06", "vis_08", "vis_09", "nir_13", "nir_16", "nir_22")
CHANNEL_NAMES_2KM = ( "ir_38", "wv_63", "wv_73", "ir_87", "ir_97", "ir_105", "ir_123", "ir_133")

const CHANNEL_NAMES = CHANNEL_NAMES_1KM[1:3]

function get_index_range(ds, channel = CHANNEL_NAMES[1])
    measured = ds.group["data"].group[channel].group["measured"]
    row_range = Int(measured["start_position_row"][]):Int(measured["end_position_row"][])
    column_range = Int(measured["start_position_column"][]):Int(measured["end_position_column"][])
    return column_range, row_range
end

function get_index_range(file_name::AbstractString, channel = CHANNEL_NAMES[1])
    column_range, row_range = NCDataset(file_name) do ds
        get_index_range(ds, channel)
    end
    return column_range, row_range
end


function get_radiance(ds, channel_name)
    channel_i = ds.group["data"].group[channel_name].group["measured"];
    vals = Float32.(cfvariable(channel_i, "effective_radiance", maskingvalue = NaN)[:,:])
    return vals
end


function read_image!(img, column_range_grid, row_range_grid, ds::NCDatasets.AbstractNCDataset, scales)
    scales = Float32.(scales)
    column_range, row_range = get_index_range(ds)
    if column_range == column_range_grid
        for i in eachindex(CHANNEL_NAMES)
            channel = CHANNEL_NAMES[i]
            val = get_radiance(ds, channel)  ./ Float32(scales[i])
            offset_row_range = row_range.-(row_range_grid[1]-1) 
            for row_val in eachindex(row_range)
                row_img = offset_row_range[row_val]
                img[column_range,row_img,i] .= val[:,row_val]
            end
        end
    end
    return nothing
end


function read_image!(img, column_range_grid, row_range_grid, file_name::AbstractString, scales)
    NCDataset(file_name) do ds
        read_image!(img, column_range_grid, row_range_grid, ds, scales)
    end
    return nothing
end


function read_image(file_names::AbstractVector, scales)
    local column_range_grid, row_range_grid

    @suppress begin
        column_range_first, row_range_first = get_index_range(file_names[1])
        column_range_last, row_range_last = get_index_range(file_names[end])

        column_range_grid = column_range_first
        row_range_grid = min(row_range_first[1],row_range_last[1]):max(row_range_first[end],row_range_last[end])
    end
    img = fill(NaN32, (length(column_range_grid), length(row_range_grid), length(CHANNEL_NAMES)))

    for f in file_names
        println("processing: ", basename(f))
        @suppress begin
            read_image!(img, column_range_grid, row_range_grid, f, scales)
        end
    end
    return column_range_grid, row_range_grid, img
end


function crop_img!(img)
    T = eltype(img)
    x0 = zero(T)
    x1 = one(T)

    replace!(x -> x < x0 ? x0 : x, img)
    replace!(x -> x1 < x ? x1 : x, img)

    replace!(x -> isnan(x) ?  x0 : x, img)

    return img
end


function img_2_rgb(img)
    b = @view img[:,:,1] 
    g = @view img[:,:,2] 
    r = @view img[:,:,3]
    return GLMakie.RGBf.(r,g,b)
end


data_folder = "FCI_data/FCI_20260104160248_nominal"

files = sort(readdir(data_folder,join=true))
filter!(x -> endswith(x,".nc") , files)

column_range_grid, row_range_grid, nice_img = let
    @time column_range_grid, row_range_grid ,img = read_image(files[3:39],(5.0, 5.0, 5.0));
    # scales = (B,G,R)
    crop_img!(img);
    nice_img = img_2_rgb(img)
    column_range_grid, row_range_grid, nice_img;
end;


@show Base.format_bytes(sizeof(nice_img))

let
    fig = Figure()
    ax = Axis(fig[1,1])
    image!(ax,(column_range_grid[1],column_range_grid[end]),(row_range_grid[1],row_range_grid[end]),nice_img)
    ax.aspect = DataAspect()
    fig
end
