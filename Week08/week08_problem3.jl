using BenchmarkTools
using Distributions
using Random
using StatsBase
using DataFrames
using Query
using Plots
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
include("../library/fitted_model.jl")

ff3 = CSV.read("F-F_Research_Data_Factors_daily.CSV", DataFrame)
mom = CSV.read("F-F_Momentum_Factor_daily.CSV",DataFrame)
returns = CSV.read("DailyReturn.csv",DataFrame)

# Join the FF3 data with the Momentum Data
ffData = innerjoin(ff3,mom,on=:Date)
rename!(ffData, names(ffData)[size(ffData,2)] => :Mom)
rename!(ffData, Symbol("Mkt-RF")=>:Mkt_RF)
ffData[!,names(ffData)[2:size(ffData,2)]] = Matrix(ffData[!,names(ffData)[2:size(ffData,2)]]) ./ 100
ffData[!,:Date] = Date.(string.(ffData.Date),dateformat"yyyymmdd")

returns[!,:Date] = Date.(returns.Date,dateformat"mm/dd/yyyy")

#join the FF3+1 to Stock data - filter to stocks we want
stocks = [:AAPL, :MSFT, Symbol("BRK-B"), :CSCO, :JNJ]
to_reg = innerjoin(returns[!,vcat(:Date, :SPY, stocks)], ffData, on=:Date)


xnames = [:Mkt_RF, :SMB, :HML, :Mom]

#OLS Regression for all Stocks
X = hcat(fill(1.0,size(to_reg,1)),Matrix(to_reg[!,xnames]))

Y = Matrix(to_reg[!,stocks])
Betas = (inv(X'*X)*X'*Y)'
resid = Y - X*Betas

Betas = Betas[:,2:size(xnames,1)+1]

max_dt = max(to_reg.Date...)
min_dt = max_dt - Year(10)
to_mean = ffData |>  @filter(_.Date >= min_dt && _.Date <= max_dt) |> DataFrame

#historic daily factor returns
exp_Factor_Return = mean.(eachcol(to_mean[!,xnames]))
expFactorReturns = DataFrame(:Factor=>xnames, :Er=>exp_Factor_Return)


#scale returns and covariance to geometric yearly numbers
stockMeans =log.(1 .+ Betas*exp_Factor_Return)*255 
covar = cov(log.(1.0 .+ Y))*255

# Function for Portfolio Volatility
function pvol(w...)
    x = collect(w)
    return(sqrt(x'*covar*x))
end

# Function for Component Standard Deviation
function pCSD(w...)
    x = collect(w)
    pVol = pvol(w...)
    csd = x.*(covar*x)./pVol
    return (csd)
end

# Sum Square Error of cSD
function sseCSD(w...)
    csd = pCSD(w...)
    mCSD = sum(csd)/n
    dCsd = csd .- mCSD
    se = dCsd .*dCsd
    return(1.0e5*sum(se)) # Add a large multiplier for better convergence
end

n = length(stocks)

m = Model(Ipopt.Optimizer)

# Weights with boundry at 0
@variable(m, w[i=1:n] >= 0,start=1/n)
register(m,:distSSE,n,sseCSD; autodiff = true)
@NLobjective(m,Min, distSSE(w...))
@constraint(m, sum(w)==1.0)
optimize!(m)

w = value.(w)

RPWeights = DataFrame(:Stock=>String.(stocks), :Weight => w, :cEr => stockMeans .* w, :CSD=>pCSD(w...))
println(RPWeights)


# RP on Simulated ES
#remove the mean
m = mean.(eachcol(Y))
Y = Y .- m'

#Fit T Models to the returns
n = size(Y,2)
m = size(Y,1)
models = Vector{FittedModel}(undef,n)
U = Array{Float64,2}(undef,m,n)
for i in 1:n
    models[i] = fit_general_t(Y[:,i])
    U[:,i] = models[i].u
end

nSim = 5000

# Gaussian Copula -- Technically we should do 255 days ahead...
corsp = corspearman(U)
_simU = cdf.(Normal(),rand(MvNormal(fill(0.0,n),corsp),nSim))'
simReturn = similar(_simU)

for i in 1:n
    simReturn[:,i] = models[i].eval(_simU[:,i])
end

# internal ES function
function _ES(w...)
    x = collect(w)
    r = simReturn*x 
    ES(r)
end

# Function for the component ES
function CES(w...)
    x = collect(w)
    n = size(x,1)
    ces = Vector{Any}(undef,n)
    es = _ES(x...)
    e = 1e-6
    for i in 1:n
        old = x[i]
        x[i] = x[i]+e
        ces[i] = old*(_ES(x...) - es)/e
        x[i] = old
    end
    ces
end

# SSE of the Component ES
function SSE_CES(w...)
    ces = CES(w...)
    ces = ces .- mean(ces)
    1e5*(ces'*ces)
end
    

#Optimize to find RP based on Expected Shortfall
n = length(stocks)

#update convergence criteria
m = Model(Ipopt.Optimizer)

# Weights with boundry at 0
@variable(m, w[i=1:n] >= 0,start=1/n)
register(m,:distSSE,n,SSE_CES; autodiff = true)
@NLobjective(m,Min, distSSE(w...))
@constraint(m, sum(w)==1.0)
optimize!(m)

w = value.(w)

ES_RPWeights = DataFrame(:Stock=>String.(stocks), :Weight => w, :cEr => stockMeans .* w, :CES=>CES(w...))
println(ES_RPWeights)
println(RPWeights)

for i in 1:length(stocks)
    print(stocks[i], "  --  ")
    println(models[i].errorModel)
end

