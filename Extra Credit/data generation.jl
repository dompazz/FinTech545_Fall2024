using BenchmarkTools
using Distributions
using Random
using StatsBase
using DataFrames
using Plots
using StatsPlots
using LinearAlgebra
using JuMP
using Ipopt
using Dates
using ForwardDiff
using FiniteDiff
using CSV
using LoopVectorization
# using GLM

include("../library/RiskStats.jl")
include("../library/simulate.jl")
include("../library/return_calculate.jl")
include("../library/gbsm.jl")

function s1(n)
    m = [0.2, 0.2, -0.1]./2
    corr = [1.0 0.5 0.3
            0.5 1.0 0.2
            0.3 0.2 1.0]
    std = [0.1, 0.1, 0.15]./2
    covar = diagm(std) * corr * diagm(std)
    covar = (covar + covar') / 2
    return rand(MvNormal(m, covar), n)'
end

function s2(n)
    m = [-0.2, -0.2, 0.25]./2
    corr = [1.0 0.2 0.0 
            0.2 1.0 0.2 
            0.0 0.2 1.0 ]
    std = [0.1, 0.1, 0.10]./2
    covar = diagm(std) * corr * diagm(std)
    covar = (covar + covar') / 2
    return rand(MvNormal(m, covar), n)'
end

function s3(n)
    m = [0.0, 0.0, 0.0]
    corr = [1.0 0.0 0.0
            0.0 1.0 0.0
            0.0 0.0 1.0]
    std = [0.1, 0.1, 0.1].*1.5
    covar = diagm(std) * corr * diagm(std)
    covar = (covar + covar') / 2
    return rand(MvTDist(10,m, covar), n)'
end

# Random.seed!(123)
# x = vcat(s1(1500),s2(500),s3(8000))
# x = x ./ 8
function s4(n)
    m = [0.08, 0.08, 0.08]
    corr = [1.0 0.5 0.0
            0.5 1.0 0.5
            0.0 0.5 1.0]
    std = [0.1, 0.1, 0.1].*2
    covar = diagm(std) * corr * diagm(std)
    covar = (covar + covar') / 2
    return rand(MvTDist(10,m, covar), n)'
end

n = 10000
Random.seed!(123)
r = s4(n)

c1_c = gbsm(true,100,100,2,0.0475,0.0475,.22).value
p2_c = gbsm(false,100,100,2,0.0475,0.0475,.22).value
s3_c = 100
current = [c1_c,p2_c,s3_c]
p1 = exp.(r) .* 100
pnl = Array{Float64}(undef, n, 3)
pnl[:,1] = [gbsm(true,p1[i,1],100,1,0.0475,0.0475,.22).value for i in 1:n] .- c1_c
pnl[:,2] = [gbsm(false,p1[i,2],100,1,0.0475,0.0475,.22).value for i in 1:n] .- p2_c
pnl[:,3] = p1[:,3] .- s3_c

x = pnl ./ current' 

density(x[:,1], title="Density Plot", label="X1")
density!(x[:,2], label="X2")
density!(x[:,3], label="X3")

mean.(eachcol(x))
std.(eachcol(x))
skewness.(eachcol(x))
kurtosis.(eachcol(x))
extrema.(eachcol(x))
ES.(eachcol(x))
ES.(eachcol(-x))

CSV.write("data.csv", DataFrame(x,["A1","A2","A3"]))