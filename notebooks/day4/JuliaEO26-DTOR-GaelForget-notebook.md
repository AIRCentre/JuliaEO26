
# Hands-On Examples

- Author : Gaël Forget 
- Date : 2026/01/09
- Object : Live Demo

<img width="150" height="100" alt="Image" src="https://github.com/user-attachments/assets/ab3e447a-6316-4126-8acf-21f7d0be5186" />

## Setup

```
import Pkg
Pkg.activate(".")
```

## Polygons for Plotting

```
using MeshArrays, Shapefile, DataDeps
pol=MeshArrays.Dataset("countries_shp1")
```

## SurfaceDrifter

```
using OceanRobots, CairoMakie
drifter=read(SurfaceDrifter(),1)
plot(drifter,pol=pol)
```

## ArgoFloat

```
using OceanRobots, CairoMakie
argo=read(ArgoFloat(),wmo=2900668)
#plot(argo,pol=pol)
plot(argo)
```

## Drifters

### Simulation

```
using Drifters, CairoMakie
P=Drifters.Gulf_of_Mexico_setup()
F=FlowFields(u=P.u,v=P.v,period=P.T)
I=Individuals(F,P.x0,P.y0);
[solve!(I,P.T .+P.dT*(n-1)) for n in 1:P.nt]
summary(I.🔴)
```

### Plotting

```
using MeshArrays, GeoJSON, DataDeps
pol=MeshArrays.Dataset("countries_geojson1")

#prefix="real "; gdf=Drifters.groupby(P.obs,:ID)
prefix="virtual "; gdf=Drifters.groupby(I.🔴,:ID)
options=(plot_type="jcon_drifters",prefix=prefix,xlims=(-98,-78),ylims=(18,31),pol=pol)
LoopC=DriftersDataset(  data=(gdf=gdf,), options=options )
plot(LoopC)
```

## ClimateModels

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
using Downloads
url=""https://github.com/gaelforget/ClimateModels.jl/raw/refs/heads/master/examples/Oceananigans.jl"
Downloads.download(url,"Oceananigans_Pluto.jl")

using Pluto
Pluto.activate_notebook_environment("Oceananigans_Pluto.jl")

import Pkg; Pkg.instantiate()
include("Oceananigans_Pluto.jl")
tz_fig
```

# Continuous Plankton Recorder

```
using Pluto
Pluto.run()
```

- copy and paste thi [URL](https://raw.githubusercontent.com/JuliaOcean/OceanRobots.jl/master/examples/CPR_notebook.jl) into Pluto prompt
- Click `Run`
- Click `Run Notebook` 

