cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

# User inputs
datapath = "data/coarse_grained_field(y,x,t).mat"
savepath = "data/"

xspace = :cheb #can be :cheb or :fourier
yspace = :cheb
tspace = :cheb
fldnames = ["vx", "vy", "rho"] # ["px", "py", "rho"]
spline_degree = 5
## Load packages - all automatic from here :)
include("src/data2coefficients_v3.jl")
#include("plottingstartup.jl") #plotting I use
using .Transforms
using MAT

# Read in data
println("Reading in data")
datadict = matread(datapath)

# Grids
xs = datadict["x"][:]
ys = datadict["y"][:]
ts = datadict["t"][:]

# Variables
vx = datadict[fldnames[1]]
vy = datadict[fldnames[2]]
rho = datadict[fldnames[3]]

# Set up the grids and smoothing parameters
println("Setting up grids")
if xspace === :cheb
    xdomain = (xs[1], xs[end])
    xspacetyp = ChebyshevSpace(2, xdomain)
elseif xspace === :fourier
    dx = xs[2] - xs[1]
    xdomain = (xs[1], xs[end]+dx)
    xspacetyp = FourierSpace(2, xdomain)
end

if yspace === :cheb
    ydomain = (ys[1], ys[end])
    yspacetyp = ChebyshevSpace(1, ydomain)
elseif yspace === :fourier
    dy = ys[2] - ys[1]
    ydomain = (ys[1], ys[end]+dy)
    yspacetyp = FourierSpace(1, ydomain)
end

if tspace === :cheb
    tdomain = (ts[1], ts[end])
    tspacetyp = ChebyshevSpace(3, tdomain)
elseif tspace === :fourier
    dt = ts[2] - ts[1]
    tdomain = (ts[1], ts[end]+dt)
    tspacetyp = FourierSpace(3, tdomain)
end

spaces = (yspacetyp, xspacetyp, tspacetyp)
inputgrid = (ys, xs, ts)

# Construct representations
println("Calculating representations")
rhorepresentation = Representation(rho, inputgrid, spaces, k = spline_degree)
vxrepresentation = Representation(vx, inputgrid, spaces, k = spline_degree)
vyrepresentation = Representation(vy, inputgrid, spaces, k = spline_degree)

# save
println("Saving representations")
saverepresentation(savepath*"_representation_$(xspace)_$(yspace)_$(tspace)_$(fldnames[1]).mat", vxrepresentation)
saverepresentation(savepath*"_representation_$(xspace)_$(yspace)_$(tspace)_$(fldnames[2]).mat", vyrepresentation)
saverepresentation(savepath*"_representation_$(xspace)_$(yspace)_$(tspace)_$(fldnames[3]).mat", rhorepresentation)
println("Done :)")
