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

#problem 1
RF = 0.0475
# Generate the constrained Max Share Ratio portfolio
p1Data = CSV.read("data.csv",DataFrame)
nms = names(p1Data)
p1Data = Matrix(p1Data)
n = size(p1Data)[2]
# Calculate the mean of the series
meanRet = mean.(eachcol(p1Data))
covar = cov(p1Data)

# Set up the optimization problem
m = Model(Ipopt.Optimizer)
function sr(w...)
    _w = collect(w)
    m = _w'*meanRet .- RF
    s = sqrt(_w'*covar*_w)
    return (m/s)
end
@variable(m, w[i=1:n] >=-1,start=1/n)
register(m,:sr,n,sr; autodiff = true)
@NLobjective(m,Max, sr(w...))
@constraint(m, sum(w)==1.0)
optimize!(m)
w_sr = value.(w)
println(w_sr)
# [0.151884814714277, -0.13305637640771478, 0.9811715616934378]


#Problem 2
function sr2(w...)
    _w = collect(w)
    r = p1Data * _w .- RF
    m = meanRet'*_w .- RF
    s = ES(r,alpha=0.025)
    return (m/s)
end
function g_sr2(g,w...)
    _w = collect(w)
    n = size(g, 1)
    sSR2 = sr2(_w...)
    for i in 1:n
        st = _w[i]
        _w[i] = st + 1e-6
        g[i] = (sr2(_w...) - sSR2) / 1e-6
        _w[i] = st
    end
end

#Approximate the solution
scope = DataFrame(:x=>-1:.1:1)
test = Matrix(filter(r-> r.x+r.y+r.z == 1.0, crossjoin(scope,rename(scope,:x=>:y),rename(scope,:x=>:z))))
sr2s = [sr2(test[i,:]...) for i in 1:size(test)[1]]
st = test[argmax(sr2s),:]

m = Model(Ipopt.Optimizer)
@variable(m, w[i=1:n] >=-1,start=st[i])
register(m,:sr,n,sr2, g_sr2)
@NLobjective(m,Max, sr(w...))
@constraint(m, sum(w)==1.0)
optimize!(m)
w_es = value.(w)
println(w_es)
# [0.45875612761827206, 0.03510777732773327, 0.5061360950539947]

#Problem 3
#why such a difference???
es_r = p1Data * w_es .- RF
sr_r = p1Data * w_sr .- RF

density(es_r, title="Density Plot", label="ES Based Risk Adjusted Return")
density!(sr_r, label="Variance Based Risk Adjusted Return")

println("ES Based Risk Adjusted Return")
println("Mean: ", mean(es_r))
println("Std: ", std(es_r))
println("Skewness: ", skewness(es_r))
println("Kurtosis: ", kurtosis(es_r))
println("ES: ", ES(es_r,alpha=0.025))

# ES Based Risk Adjusted Return
# Mean: 0.14003154147979827
# Std: 0.5659834430889673
# Skewness: 2.8722415428508206
# Kurtosis: 23.368595644837324
# ES: 0.5288077509516422

println("Variance Based Risk Adjusted Return")
println("Mean: ", mean(sr_r))
println("Std: ", std(sr_r))
println("Skewness: ", skewness(sr_r))
println("Kurtosis: ", kurtosis(sr_r))
println("ES: ", ES(sr_r,alpha=0.025))

# Variance Based Risk Adjusted Return
# Mean: 0.13454348051911924
# Std: 0.39096030302069296
# Skewness: 0.46408016508980365
# Kurtosis: 2.619132964327651
# ES: 0.8019317800867884

# The Variance based Sharpe Ratio does not consider higher moments of the distribution,
# and picks a portfolio that has a low skew and kurtosis and a higher downside risk.

# The ES based Risk Adjusted Return picks a portfolio that has a higher skew and kurtosis,
# at the expense of a higher volatility. While the means are similar, the higher volatility
# of the ES based portfolio means that the SR based optimization will not pick it, even though
# is has a higher positive skew and kurtosis, and lower downside risk.