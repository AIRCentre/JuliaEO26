# # A walk through the JuliaGeo ecosystem
# Here we'll primarily focus on getting data - both raster and vector data - and 
# showing the various ways they can interact.

vector_file = download("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries.geojson", "countries.geojson")

using GeoDataFrames
countries = GeoDataFrames.read(vector_file)

# Ok, this is a dataframe with a geometry column, and a bunch of other columns.
# You can do any dataframe like thing on this.  But let's focus on the geometry and 
# geospatial things for now.  Dataframe operations will soon described in the 
# [geocompjl book](https://jl.geocompx.org).

countries.geometry[1]

# ## Plotting vector data
# You can plot with this of course:
using CairoMakie
poly(countries.geometry; color = countries.POP_RANK)
# and plot onto a projection:
using GeoMakie
poly(countries.geometry; color = countries.POP_RANK, axis = (; type = GeoAxis))
poly(countries.geometry; color = countries.POP_RANK, axis = (; type = GlobeAxis))

# ## GeoInterface accessors
# We can use GeoInterface to get things from the geometry:
import GeoInterface as GI
geom = countries.geometry[1]
GI.trait(geom)
GI.extent(geom)
GI.crs(geom)

GI.ngeom(geom)
g1 = GI.getgeom(geom, 1)
g1g1 = GI.getgeom(g1, 1)
g1g1g1 = GI.getgeom(g1g1, 1)

# ## Vector operations
import GeometryOps as GO

GO.perimeter(GO.Planar(),geom)
GO.area(geom)

# Wait, that seems wrong....

GO.perimeter(GO.Spherical(), geom)
GO.area(GO.Spherical(), geom)
GO.area(GO.Geodesic(), geom)

# There, that's better!

# Let's get the polygon of Switzerland from our data.
ch_idx = findfirst(x -> x == "Switzerland", countries.NAME)
switzerland = countries.geometry[ch_idx]

# We can flip it around its centroid in the x-axis.
switzerland_centroid = GO.centroid(switzerland)
flipped = GO.transform(switzerland) do point
    x, y = point
    return (-(x - switzerland_centroid[1]) + switzerland_centroid[1], y)
end
# `GO.transform` basically decomposes its input down to points, and then applies
# the function you provided to each point.
poly(switzerland)
poly!(flipped)
Makie.current_figure()

# We can now take the intersection of the two polygons.
inter = GO.intersection(switzerland, flipped; target = GI.MultiPolygonTrait())
# and plot it:
poly!(inter)
Makie.current_figure()

# Finally you also have a bunch of geometric predicates.
# Here let's see if Switzerland intersects with the first country in our data,
# Indonesia.
GO.intersects(switzerland, countries.geometry[1])

# # Raster data

using Rasters
using RasterDataSources

# RasterDataSources provides a lot of rasters from a variety of sources.
# You can also load a file by `Raster(filename)`.
ras = Raster(WorldClim{Climate}, :prec; month = 7)
# A raster is a regular Julia array
ras[1, 1]
# But you can also index it "logically"
lat, lon = 38.66254411742267, -27.22776583718217
ras[X(Near(-110)), Y(Near(50))]

# We can also do zonal statistics, which is a common operation.
precips = Rasters.zonal(sum, ras; skipmissing = true, of = countries.geometry)
# and plot it:
using CairoMakie
plot(countries.geometry; color = collect(precips))
