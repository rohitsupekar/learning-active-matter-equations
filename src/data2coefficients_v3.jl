module Transforms
using FFTW, MAT, Dierckx #Interpolations
export FourierSpace, ChebyshevSpace
export Representation, reconstruct, threshold
export MaxThreshold, IntegerRadialThreshold
export saverepresentation, readrepresentation

# Space types 
abstract type RepresentationSpace end

struct FourierSpace <: RepresentationSpace 
    dim::Int
    domain::Tuple{Float64, Float64}
end

struct ChebyshevSpace <: RepresentationSpace
    dim::Int
    domain::Tuple{Float64, Float64}
end

iscomplextransform(::FourierSpace) = true
iscomplextransform(::ChebyshevSpace) = false


## =========================================================================================================== ##
#   Grids - in this problem there are two grids the input grid that the data is given on and the transform      #
#   grid that the data needs to be on to perform the transform. Grids are stored as a Tuple{AbstractVector}     #
## =========================================================================================================== ##

# Gridding types -- scale to the domain f(-1) = a; f(1) = b; f(x) = cx + d => c = d-a and c = b-d => 
function get_transformgrid(S::FourierSpace, N)
    a, b = S.domain
    scaledgrid = LinRange(a, b, N+1)[1:N]
end

function get_transformgrid(S::ChebyshevSpace, N)
    a, b = S.domain
    scaledgrid = 0.5*(b - a)*[cos(k*π/(N-1)) for k = 0:N-1] .+ 0.5*(a + b)
end

get_transformgrid(S::NTuple{N, RepresentationSpace}, Ns::NTuple{N, Int64}) where N = get_transformgrid.(S, Ns)

# This is called twice -- once on the input and once on the output
"""
Function to convert between grid -- currently this uses Dierckx.jl for spline interpolation
"""
# utilize type inference to call the same function in later functions
tospline(sortedgrid::Tuple{AbstractVector, AbstractVector}, sorteddata::Matrix, k) = Spline2D(sortedgrid..., sorteddata, kx = k, ky = k)
tospline(sortedgrid::Tuple{AbstractVector}, sorteddata::Vector, k) = Spline1D(sortedgrid..., sorteddata, k = k)
evalspline(spline::Spline1D, outputgrid) = evaluate(spline, outputgrid...)
evalspline(spline::Spline2D, outputgrid) = evalgrid(spline, outputgrid...)

function gridtransformation(inputgrid, outputgrid, data; k = 1)
    #Look after non increasing grids e.g. Chebyshev
    sp = sortperm.(inputgrid)
    sortedgrid = getindex.(inputgrid, sp)

    # Build interpolations
    itp = tospline(sortedgrid, data[sp...], k)

    # Interpolate
    transformdata = evalspline(itp, outputgrid)
    return transformdata
end

function gridtransformation(inputgrid, outputgrid, data::Array{T, 3}; k = 1) where {T}
    Nd = length(outputgrid)
    sp = sortperm.(inputgrid)
    sortedgrid = getindex.(inputgrid, sp)
    spoutput = sortperm.(outputgrid)
    sortedoutputgrid = getindex.(outputgrid, spoutput)
    tmptransformdata = Array{Float64, Nd}(undef, length.(inputgrid)[1:2]..., length(outputgrid[3]))
    transformdata = Array{Float64, Nd}(undef, length.(outputgrid))

    for I = CartesianIndices(length.(inputgrid)[1:2])
        tmptransformdata[I, :] = evaluate(Spline1D(sortedgrid[3], data[I, sp[3]], k = k), sortedoutputgrid[3])
    end

    for i = 1:size(data, 3)
        itp = tospline(sortedgrid[1:2], tmptransformdata[sp[1:2]..., i], k)
        transformdata[:, :, i] .= evalspline(itp, sortedoutputgrid[1:2])
    end

    # Interpolate
    return transformdata[invperm.(spoutput)...]
end

## ========================
# Transforms
# We need to define the transfrom along any dimension from transform grid 
# All are done using FFTs and DCTs
## ========================

function chebtransform(data, dim)
    cfsmat = FFTW.r2r(data, FFTW.REDFT00, dim)
    cfsmat ./= (size(data, dim)-1)
    selectdim(cfsmat, dim, 1) ./= 2
    selectdim(cfsmat, dim, size(data, dim)) ./= 2
    return cfsmat
end

function chebtransform!(data, dim)
    FFTW.r2r!(data, FFTW.REDFT00, dim)
    data ./= (size(data, dim)-1)
    selectdim(data, dim, 1) ./= 2
    selectdim(data, dim, size(data, dim)) ./= 2
    return data
end

function ichebtransform!(data, dim)
    selectdim(data, dim, 1) .*= 2
    selectdim(data, dim, size(data, dim)) .*= 2
    FFTW.r2r!(data, FFTW.REDFT00, dim)
    data ./= 2
    return data
end
ichebtransform(data, dim) = ichebtransform!(copy(data), dim)

transform(S::FourierSpace, data) = fft(data, S.dim)
transform(S::ChebyshevSpace, data) = chebtransform(data, S.dim)

itransform(S::FourierSpace, data) = ifft(data, S.dim)
itransform(S::ChebyshevSpace, data) = ichebtransform(data, S.dim)

# Inplace transforms
transform!(S::FourierSpace, data) = fft!(data, S.dim)
transform!(S::ChebyshevSpace, data) = chebtransform!(data, S.dim)

itransform!(S::FourierSpace, data) = ifft!(data, S.dim)
itransform!(S::ChebyshevSpace, data) = ichebtransform!(data, S.dim)

##
function transform(S::NTuple{N, RepresentationSpace}, data::Array{T, N}) where {T, N}
    if any(iscomplextransform.(S))
        transformdata = Complex.(data)
    else
        transformdata = copy(data)
    end

    for i = 1:N 
        transform!(S[i], transformdata)
    end
    return transformdata
end

function itransform(S::NTuple{N, RepresentationSpace}, data::Array{T, N}) where {T, N}
    transformdata = copy(data)
    for i = 1:N 
        itransform!(S[i], transformdata)
    end

    return transformdata
end

## Main Representation type
abstract type AbstractRepresentation{T, S, N} end

# T is input data type
# S is the coefficient type eg Fourier Real -> Complex
# N is the dimension of the data
function checkrepresetnationdims(spaces, N)
    dims = [S.dim for S ∈ spaces]
    @assert sort(unique(dims)) == 1:N "Dimension of spaces not consistent with input data"
end

struct Representation{T, S, N} <: AbstractRepresentation{T, S, N}
    # Inputs
    spaces::NTuple{N, RepresentationSpace}
    inputgrid::NTuple{N, AbstractVector}
    transformgrid::NTuple{N, AbstractVector}

    # Coefficients
    coefficients::Array{S, N}

    function Representation{T, S, N}(spaces::NTuple{N, RepresentationSpace}, inputgrid::NTuple{N, AbstractVector}, transformgrid::NTuple{N, AbstractVector}, coefficients::Array{S, N}) where {T, S, N}
        checkrepresetnationdims(spaces, N)
        new{T, S, N}(spaces, inputgrid, transformgrid, coefficients)
    end
end

function Representation(data::Array{T, N}, inputgrid::NTuple{N, AbstractVector}, spaces::NTuple{N, RepresentationSpace}; k = 1) where {T, N}
    checkrepresetnationdims(spaces, N)
    Ns = size(data)
    transformgrid = get_transformgrid(spaces, Ns)
    transformdata = gridtransformation(inputgrid, transformgrid, data, k=k)
    coefficients = transform(spaces, transformdata)
    Representation{T, eltype(coefficients), N}(spaces, inputgrid, transformgrid, coefficients)
end

## Differentiation
function differentiate(S::FourierSpace, coeffs::Array{T, N}, n = 1) where {T, N}
    n < 0 && error("n must be a nonnegative integer")
    n == 0 && return copy(coeffs)

    dim = S.dim
    @assert dim <= N
    Nd = size(coeffs, dim)

    # Calculate weights
    fftdiffweights = (2π*im*fftfreq(Nd)*(Nd-1)*(1/(S.domain[2] - S.domain[1]))).^n
    if iseven(Nd) && isodd(n) 
        fftdiffweights[div(Nd, 2) + 1] = 0
    end
    fftdiffweights = reshape(fftdiffweights, ones(Int, dim-1)..., :)
    diffcoeffs = coeffs .* fftdiffweights
    return diffcoeffs 
end

function differentiate(S::ChebyshevSpace, coeffs::Array{T, N}, n = 1) where {T, N}
    # Set up
    n < 0 && error("n must be a nonnegative integer")
    n == 0 && return copy(coeffs)

    dim = S.dim
    @assert dim <= N
    copycoeffs = copy(coeffs)
    diffcoeffs = similar(coeffs)
    Nd = size(diffcoeffs, dim)

    # Differentiate n times 
    for j = 1:n - 1
        _recursion_chebyshev_diff!(diffcoeffs, copycoeffs, dim, Nd)
        copyto!(copycoeffs, diffcoeffs)
    end
    _recursion_chebyshev_diff!(diffcoeffs, copycoeffs, dim, Nd)
    # Scale to the domain
    diffcoeffs .*= (2/(S.domain[2] - S.domain[1]))^n 
    return diffcoeffs
end

function differentiate(R::AbstractRepresentation{T, S, N}, derivs::NTuple{N, Int64}) where {T, S, N}
    diffcoeffs = R.coefficients
    for i = 1:N
        diffcoeffs = differentiate(R.spaces[i], diffcoeffs, derivs[i])
    end
    return diffcoeffs
end

function _recursion_chebyshev_diff!(diffcoeffs, coeffs, dim, Nd)
    selectdim(diffcoeffs, dim, Nd) .= 0
    selectdim(diffcoeffs, dim, Nd - 1) .= 2(Nd - 1)*selectdim(coeffs, dim, Nd)
    selectdim(diffcoeffs, dim, Nd - 2) .= 2(Nd - 2)*selectdim(coeffs, dim, Nd-1)
    for i = Nd - 3:-1:1
        selectdim(diffcoeffs, dim, i) .= selectdim(diffcoeffs, dim, i + 2) .+ 2*i*selectdim(coeffs, dim, i + 1)
    end
    selectdim(diffcoeffs, dim, 1) ./= 2
end

function reconstruct(R::AbstractRepresentation{T,S, N};  k = 1, toinputgrid = true) where {T, S, N}
    transformdata = itransform(R.spaces, R.coefficients)
    T <: Real && (transformdata = real.(transformdata))
    if toinputgrid
        return gridtransformation(R.transformgrid, R.inputgrid, transformdata, k = k)
    else
        return transformdata
    end
end

function reconstruct(R::AbstractRepresentation{T, S, N}, derivs::NTuple{N, Int}; k = 1, toinputgrid = true) where {T, S, N}
    coefficients = differentiate(R, derivs)
    transformdata = itransform(R.spaces, coefficients)
    T <: Real && (transformdata = real.(transformdata))
    if toinputgrid
        return gridtransformation(R.transformgrid, R.inputgrid, transformdata, k = k)
    else
        return transformdata
    end
end

## Thresholding
abstract type AbstractThreshold end
issupported(rt::AT, sp::RS) where {AT <: AbstractThreshold, RS <: RepresentationSpace} = false

struct ThresholdedRepresentation{T, S, N, Nthr} <: AbstractRepresentation{T, S, N}
    parent::AbstractRepresentation{T, S, N}
    coefficients::Array{S, N}
    threshold::NTuple{Nthr, AbstractThreshold}
end

function Base.getproperty(R::ThresholdedRepresentation, item::Symbol)
    if item ∈ fieldnames(ThresholdedRepresentation)
        return getfield(R, item)
    elseif item == :unthresholded_coefficients
        return getfield(R.parent, :coefficients)
    else
        return getfield(R.parent, item)
    end
end

Base.propertynames(R::ThresholdedRepresentation) = (fieldnames(typeof(R.parent))..., :threshold, :unthresholded_coefficients)

struct IntegerRadialThreshold <: AbstractThreshold
    dims::AbstractVector{Int64}
    r2
end 
getthr(thr::IntegerRadialThreshold) = thr.r2
issupported(rt::IntegerRadialThreshold, sp::FourierSpace) = true

function getnval(ind, n)
    if ind <= div(n + 1, 2)
        return ind - 1
    else
        return ind - 1 - n
    end
end

function checkgoodind(thrtyp::IntegerRadialThreshold, i, val, sz)
    dims = thrtyp.dims
    thr = thrtyp.r2 

    r2 = 0
    for d ∈ dims
        r2 += getnval(i[d], sz[d])^2
    end

    return r2 < thr
end


struct MaxThreshold <: AbstractThreshold
    dims::Int
    n::Int
end
getthr(thr::MaxThreshold) = thr.n
issupported(rt::MaxThreshold, sp::ChebyshevSpace) = true
checkgoodind(thrtyp::MaxThreshold, i, val, sz) = i[thrtyp.dims] <= thrtyp.n 

struct NoThreshold <: AbstractThreshold 
    dims::Union{AbstractVector{Int}, Int}
end
issupported(rt::NoThreshold, sp::RS) where {RS <: RepresentationSpace}= true
checkgoodind(thrtyp::NoThreshold, i, val, sz) = true


function check_threshold(spaces, thrtypes::NTuple{Nthr, AbstractThreshold}) where Nthr
    alldims = [t.dims for t ∈ thrtypes]
    Ndims = length(spaces)
    if any(vcat(alldims...) .> Ndims)
        error("Dimensions for thresholds do not match")
    elseif !isempty(intersect(alldims...)) && length(thrtypes) > 1
        error("Dimensions assigned to multiple thresholds")
    end

    for thrtype ∈ thrtypes
        check_threshold(spaces, thrtype)
    end

    return nothing
end

function check_threshold(spaces, thrtype)
    dims = thrtype.dims
    sptype = typeof(spaces[dims[1]])

    for i = 2:length(dims)
        !isa(spaces[dims[i]], sptype) && error("All dimensions must have the same dimension")
    end

    !issupported(thrtype, spaces[dims[1]]) && error("Threhsold type not supported for given space")

    return nothing 
end

function threshold(R::Representation{T, S, N}, thrtypes::NTuple{Nthr, AbstractThreshold}) where {T, S, N, Nthr}
    # Setup
    check_threshold(R.spaces, thrtypes)
    coefficients = R.coefficients
    sz = size(coefficients)
    thrcoeffs = zeros(S, sz...)

    cutthrcoeffsinds = getoptloops(thrtypes, thrcoeffs) ## Don't loop over regions we already know are zero
    for i ∈ cutthrcoeffsinds
        goodcoeff = true
        for thrtyp ∈ thrtypes
            goodcoeff = goodcoeff && checkgoodind(thrtyp, i, thrcoeffs[i], sz)
            !goodcoeff && break 
        end

        goodcoeff && (thrcoeffs[i] = coefficients[i])
    end
    return ThresholdedRepresentation(R, thrcoeffs, thrtypes)
end

function getoptloops(thrtypes, thrcoeffs)
    sizetup = size(thrcoeffs)
    axestup = axes(thrcoeffs)
    coeff_inds = trues(sizetup...)

    for thrtyp ∈ thrtypes
        if isa(thrtyp, Transforms.MaxThreshold)
            coeff_inds[CartesianIndices(axestup[1:thrtyp.dims-1]), thrtyp.n+1:sizetup[thrtyp.dims], CartesianIndices(axestup[thrtyp.dims+1:end])] .= false
        elseif isa(thrtyp, Transforms.IntegerRadialThreshold)
            r = sqrt(thrtyp.r2)
            for d ∈ thrtyp.dims 
                if r < div(sizetup[d], 2)
                    coeff_inds[CartesianIndices(axestup[1:d-1]), ceil(Int, r)+2:sizetup[d] - ceil(Int, r), CartesianIndices(axestup[d+1:end])] .= false
                end
            end
        end
    end
    return findall(coeff_inds)
end

"""
Functions to save the representation files in .mat format 
"""
function saverepresentation(path, repr::Representation{T, S, N}) where {T, S, N}
    savedict = Dict("coefficients" => repr.coefficients, "spaces" => [string.(repr.spaces)...], "T" => string(T), "inputgrid" => [collect.(repr.inputgrid)...], "transformgrid" => [collect.(repr.transformgrid)...])
    matwrite(path, savedict)
end

function readrepresentation(path)
    savedict = matread(path)
    coefficients = savedict["coefficients"]
    S = eltype(coefficients)
    T = eval(Meta.parse(savedict["T"]))
    N = ndims(coefficients)
    spaces = Tuple(eval.(Meta.parse.(savedict["spaces"])))
    return Representation{T, S, N}(spaces, Tuple(savedict["inputgrid"]), Tuple(savedict["transformgrid"]), coefficients)
end
end #module