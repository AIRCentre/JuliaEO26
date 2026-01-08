
# setup

```
import Pkg
Pkg.activate(".")
```

# get some polygons for plotting

```
using MeshArrays, Shapefile, DataDeps
pol=MeshArrays.Dataset("countries_shp1")
```

# SurfaceDrifter

```
using OceanRobots, CairoMakie
drifter=read(SurfaceDrifter(),1)
plot(drifter,pol=pol)
```

# ArgoFloat

```
using OceanRobots, CairoMakie
argo=read(ArgoFloat(),wmo=2900668)
#plot(argo,pol=pol)
plot(argo)
```

# Drifters

```
using Drifters, CairoMakie
P=Drifters.Gulf_of_Mexico_setup()
F=FlowFields(u=P.u,v=P.v,period=P.T)
I=Individuals(F,P.x0,P.y0);
[solve!(I,P.T .+P.dT*(n-1)) for n in 1:P.nt]
summary(I.🔴)
```

```
using MeshArrays, GeoJSON, DataDeps
pol=MeshArrays.Dataset("countries_geojson1")

#prefix="real "; gdf=Drifters.groupby(P.obs,:ID)
prefix="virtual "; gdf=Drifters.groupby(I.🔴,:ID)
options=(plot_type="jcon_drifters",prefix=prefix,xlims=(-98,-78),ylims=(18,31),pol=pol)
LoopC=DriftersDataset(  data=(gdf=gdf,), options=options )
plot(LoopC)
```

# ClimateModels

```
using ClimateModels
fun=ClimateModels.RandomWalker
ModelRun(ModelConfig(model=fun))
MC=run(ModelConfig(fun))
```

```
log(MC)
```

```
readdir(MC)
```

### Oceananigans

```
using Pluto
Pluto.activate_notebook_environment("examples/Oceananigans.jl")
include("examples/Oceananigans.jl")
tz_fig
```

# Continuous Plankton Recorder

```
using Pluto
Pluto.run()
```

### copy and paste the URL into Pluto

<https://raw.githubusercontent.com/JuliaOcean/OceanRobots.jl/master/examples/CPR_notebook.jl>

### Hit Run and Let it Go

