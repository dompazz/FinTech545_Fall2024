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
Betas = (inv(X'*X)*X'*Y)'[:,2:size(xnames,1)+1]

max_dt = max(to_reg.Date...)
min_dt = max_dt - Year(10)
to_mean = ffData |>  @filter(_.Date >= min_dt && _.Date <= max_dt) |> DataFrame

#historic daily factor returns
exp_Factor_Return = mean.(eachcol(to_mean[!,xnames]))
expFactorReturns = DataFrame(:Factor=>xnames, :Er=>exp_Factor_Return)


#scale returns and covariance to geometric yearly numbers
stockMeans =log.(1 .+ Betas*exp_Factor_Return)*255 
covar = cov(log.(1.0 .+ Y))*255

function sr(w...)
    _w = collect(w)
    m = _w'*stockMeans - .0025
    s = sqrt(_w'*covar*_w)
    return (m/s)
end

n = length(stocks)

m = Model(Ipopt.Optimizer)
# set_silent(m)
# Weights with boundry at 0
@variable(m, w[i=1:n] >= 0,start=1/n)
register(m,:sr,n,sr; autodiff = true)
@NLobjective(m,Max, sr(w...))
@constraint(m, sum(w)==1.0)
optimize!(m)

w = value.(w)

function RiskBudget(w)
    pSig = sqrt(w'*covar*w)
    CSD = (w .* covar*w) / pSig
    DataFrame( (CSD / pSig)', stocks)
end

riskBudget = RiskBudget(w)



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
RPw = copy(w)
riskBudget = DataFrame(:Stock=>stocks, :w=>w, :RiskBudget=>[RiskBudget(w)[1,:]...], :σ=>diag(covar).^.5)



covar_old = copy(covar)
_std = diagm(diag(covar_old)).^.5


covar = _std*fill(0.5,(5,5))*_std
# covar = _std*I(size(_std,1))*_std

n = length(stocks)

m = Model(Ipopt.Optimizer)

# Weights with boundry at 0
@variable(m, w[i=1:n] >= 0,start=1/n)
register(m,:distSSE,n,sseCSD; autodiff = true)
@NLobjective(m,Min, distSSE(w...))
@constraint(m, sum(w)==1.0)
optimize!(m)

w = value.(w)
riskBudget = DataFrame(:Stock=>stocks, :w=>w, :RiskBudget=>[RiskBudget(w)[1,:]...], :σ=>diag(covar).^.5)

inv_std = (1 ./ diag(_std))

riskBudget[!,:iVolWgt]=inv_std/sum(inv_std)
println(riskBudget)


covar = copy(covar_old)
rb = [1,2,1,1,0.5]


# Sum Square Error of cSD - updated for risk budgets
function sseCSD2(w...)
    csd = pCSD(w...) ./ rb
    mCSD = sum(csd)/n
    dCsd = csd .- mCSD
    se = dCsd .*dCsd
    return(1.0e5*sum(se)) # Add a large multiplier for better convergence
end
n = length(stocks)

m = Model(Ipopt.Optimizer)

# Weights with boundry at 0
@variable(m, w[i=1:n] >= 0,start=1/n)
register(m,:distSSE,n,sseCSD2; autodiff = true)
@NLobjective(m,Min, distSSE(w...))
@constraint(m, sum(w)==1.0)
optimize!(m)

w = value.(w)
riskBudget = DataFrame(:Stock=>stocks, :w=>w, :RiskBudget=>[RiskBudget(w)[1,:]...], :Rb=>rb)

# iStd = (1 ./ sqrt.(diag(covar)))
# correl = iStd .* covar .* iStd'
# CSV.write("c:/temp/corel.csv", DataFrame(correl, stocks))