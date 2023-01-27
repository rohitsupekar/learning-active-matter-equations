cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

# Inputs
datafolderpath = "data/"
savefolderpath = "data/"

xspace = :cheb #can be :cheb or :fourier
yspace = :cheb
tspace = :cheb
fldnames = ["vx", "vy", "rho"] # ["px", "py", "rho"]
spline_degree = 5

#coefficient thresholds
space_thr_list = [100]
time_thr_list = [50]

derivdesc_id = ["", "t", "x", "y"]
derivdesc = ((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0))

## Automatic from here
# Load relevant packages
using MAT
using Printf
# Include packages and decompression functions
include("src/data2coefficients_v3.jl")
using .Transforms

#matwrite saves a v7.3 .mat file which uses a HDF5 file system. This, when loaded in Python using h5py tranposes the matrix
#Hence, here, we first transpose the matrix before saving
function transposeMats(ls)
    #ls: list of arrays
    return [permutedims(m, ndims(m):-1:1) for m in ls]
end

# Load representations
println("Loading representations")
vxrepresentation = readrepresentation(datafolderpath*"_representation_$(xspace)_$(yspace)_$(tspace)_$(fldnames[1]).mat")
vyrepresentation = readrepresentation(datafolderpath*"_representation_$(xspace)_$(yspace)_$(tspace)_$(fldnames[2]).mat")
rhorepresentation = readrepresentation(datafolderpath*"_representation_$(xspace)_$(yspace)_$(tspace)_$(fldnames[3]).mat")
ys, xs, ts = rhorepresentation.inputgrid
# Set up the threshold tups
println("Setting up thresholds")
if xspace === yspace === :cheb
    # Account for potentially non-square data (rectangles not squares)
    sz = size(vxrepresentation.coefficients)
    ysz = sz[1]; xsz = sz[2];
    if ysz > xsz
        scale = ysz/xsz
        ythrs = scale*space_thr_list
        xthrs = space_thr_list
    else
        scale = xsz/ysz
        xthrs = scale*space_thr_list
        ythrs = space_thr_list
    end
    spacethrtups = map((ythr, xthr) -> (MaxThreshold(1, ythr), MaxThreshold(2, xthr)), ythrs, xthrs)
elseif xspace === yspace === :fourier
    spacethrtups = map(thr -> (IntegerRadialThreshold(1:2, thr), ), space_thr_list)
else
    error("What is going on here?")
end

if tspace === :cheb
    timethrtups = map(thr -> MaxThreshold(3, thr), time_thr_list)
else
    error("What is going on here?")
end

println("Beginning sweep ...")
global count = 0;

for (timeidx, time_thr) in enumerate(timethrtups)
    for (spaceidx, space_thr) in enumerate(spacethrtups)

        global count
        count = count + 1;
        @printf("%i/%i : space_thr = %i, time_thr = %i\n", count, length(timethrtups)*length(spacethrtups), space_thr_list[spaceidx], time_thr_list[timeidx])
        savepath = savefolderpath * "_reconstruction_$(xspace)_$(yspace)_$(tspace)_space$(space_thr_list[spaceidx])_time$(time_thr_list[timeidx]).mat"

        thrtup = (space_thr..., time_thr)
        thrrhorepresentation = threshold(rhorepresentation, thrtup)
        thrvxrepresentation = threshold(vxrepresentation, thrtup)
        thrvyrepresentation = threshold(vyrepresentation, thrtup)
        ##
        println("$(fldnames[3]) derivatives")
        rho_derivdict = map(x -> reconstruct(thrrhorepresentation, x, k = spline_degree), derivdesc)
        println("$(fldnames[1]) derivatives")
        vx_derivdict = map(x -> reconstruct(thrvxrepresentation, x, k = spline_degree), derivdesc)
        println("$(fldnames[2]) derivatives")
        vy_derivdict = map(x -> reconstruct(thrvyrepresentation, x, k = spline_degree), derivdesc)

        ##
        rho_derivdict_tranpose = transposeMats(rho_derivdict)
        vx_derivdict_tranpose = transposeMats(vx_derivdict)
        vy_derivdict_tranpose = transposeMats(vy_derivdict)

        #tranpose the grid vectors
        x, y, t = transposeMats([xs, ys, ts])

        ## save as .mat
        println("saving data..")
        savedict = Dict(Pair.("$(fldnames[3])_".*derivdesc_id, rho_derivdict_tranpose)..., Pair.("$(fldnames[1])_".*derivdesc_id, vx_derivdict_tranpose)..., Pair.("$(fldnames[2])_".*derivdesc_id, vy_derivdict_tranpose)..., "x"=>x, "y"=>y, "t"=>t)
        matwrite(savepath, savedict)
    end
end
