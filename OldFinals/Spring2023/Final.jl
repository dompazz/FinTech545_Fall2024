using Distributions
using Random
using StatsBase
using DataFrames
using Roots
using LinearAlgebra
using JuMP
using Ipopt
using Dates
using CSV
using LoopVectorization
using Printf
using FiniteDiff
# using GLM

include("../library/RiskStats.jl")
include("../library/simulate.jl")
include("../library/fitted_model.jl")
include("../library/return_calculate.jl")
include("../library/gbsm.jl")
include("../library/missing_cov.jl")
include("../library/expost_factor.jl")


user = "Dominic Pazzula"

run(`python datageneration.py "$user" c:/temp`)

#problem 1
println("\n*******************PROBLEM 1*******************\n")
prices = CSV.read("c:/temp/Problem1.csv",DataFrame)
returns = return_calculate(prices,method="LOG",dateColumn="Date")

println(returns)

select!(returns,Not(:Date))

covar = missing_cov(Matrix(returns),skipMiss=false,fun=cov)

println(DataFrame(covar,:auto))

minEig = min(eigvals(covar)...)
if minEig < -1e-8
    println("NOT PSD -- $minEig")
    fixed = near_psd(covar)
    println("Fixed: ")
    println(DataFrame(fixed,:auto))
else
    println("PSD -- $minEig")
end

#Problem 2
println("\n*******************PROBLEM 2*******************\n")
optParams = CSV.read("c:/temp/Problem2.csv",DataFrame)

# gbsm(call::Bool,    underlying,            strike,             ttm,                 rf,             b,                                   ivol        ; includeGreeks=false)
callGBSM = gbsm(true,optParams.Underlying[1],optParams.Strike[1],optParams.TTM[1]/255,optParams.RF[1],optParams.RF[1]-optParams.DivRate[1],optParams.IV[1]; includeGreeks=true)
# callGBSM = gbsm(true,optParams.Underlying[1],optParams.Strike[1],optParams.TTM[1]/255,optParams.RF[1],optParams.DivRate[1],optParams.IV[1]; includeGreeks=true)

println("Price: $(callGBSM.value)")
println("Delta: $(callGBSM.delta)")
println("Gamma: $(callGBSM.gamma)")
println("Vega: $(callGBSM.vega)")
println("NOT Rho: $(callGBSM.rho)") # Do Finite Diff here...
fRho(rf) = gbsm(true,optParams.Underlying[1],optParams.Strike[1],optParams.TTM[1]/255,rf,rf-optParams.DivRate[1],optParams.IV[1]).value
rho = FiniteDiff.finite_difference_derivative(fRho,optParams.RF[1])
println("Actual Rho: $rho")
# I had the Rho wrong in here initialy, everyone got 1 point added to their final grade to compensate


iters = 1000
vars = Vector{Float64}(undef,iters)
ess = Vector{Float64}(undef,iters)
puts = Vector{Float64}(undef,iters)

putVal = gbsm(false,optParams.Underlying[1],optParams.Strike[1],optParams.TTM[1]/255,optParams.RF[1],optParams.RF[1]-optParams.DivRate[1],optParams.IV[1]).value

n = 1000
for k in 1:iters
    simP = (1 .+ rand(Normal(0.0,optParams.IV[1]/sqrt(255)),n)) .* optParams.Underlying[1]

    simPnL = (simP .- optParams.Underlying[1]) .- ([gbsm(true,simP[i],optParams.Strike[1],(optParams.TTM[1]-1)/255,optParams.RF[1],optParams.RF[1]-optParams.DivRate[1],optParams.IV[1]).value 
                for i in 1:n] .- callGBSM.value)
    # simPnL = (simP .- optParams.Underlying[1]) .- ([gbsm(true,simP[i],optParams.Strike[1],(optParams.TTM[1]-1)/255,optParams.RF[1],optParams.DivRate[1],optParams.IV[1]).value 
    #             for i in 1:n] .- callGBSM.value)
    vars[k] = VaR(simPnL)
    ess[k] = ES(simPnL)
    # _pts = [gbsm(false,simP[i],optParams.Strike[1],(optParams.TTM[1]-1)/255,optParams.RF[1],optParams.RF[1]-optParams.DivRate[1],optParams.IV[1]).value 
    #              for i in 1:n]
    # puts[k] = VaR(-(_pts .- putVal))
end

println("VaR: $(mean(vars)) [$(-VaR(vars)),$(-VaR(vars,alpha=0.95))]")
println("ES: $(mean(ess)) [$(-VaR(ess)),$(-VaR(ess,alpha=0.95))]")
# println("Mean Put VaR: $(mean(puts))")
    
#Problem 3

println("\n*******************PROBLEM 3*******************\n")
covar = Matrix(CSV.read("c:/temp/problem3_cov.csv",DataFrame))
er = CSV.read("c:/temp/problem3_ER.csv",DataFrame)

rf = er.RF[1]

er = vec(Matrix(select(er,Not(:RF))))

# covar = convert.(Float64,covar[:,2:4])

function sr(w...)
    _w = collect(w)
    m = _w'*er - rf
    s = sqrt(_w'*covar*_w)
    return (m/s)
end

n = length(er)

m = Model(Ipopt.Optimizer)
set_silent(m)
# Weights with boundry at 0
@variable(m, w[i=1:n] >= 0,start=1/n)
register(m,:sr,n,sr; autodiff = true)
@NLobjective(m,Max, sr(w...))
@constraint(m, sum(w)==1.0)
optimize!(m)
println(termination_status(m))

wSR = round.(value.(w),digits=4)

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

n = length(er)

m2 = Model(Ipopt.Optimizer)
set_silent(m2)
# Weights with boundry at 0
@variable(m2, w[i=1:n] >= 0,start=1/n)
register(m2,:distSSE,n,sseCSD; autodiff = true)
@NLobjective(m2,Min, distSSE(w...))
@constraint(m2, sum(w)==1.0)
optimize!(m2)
println(termination_status(m2))
wRP = round.(value.(w),digits=4)
println("Max SR Weights: $wSR")
println("Risk Parity Weights: $wRP")

#Problem 4

println("\n*******************PROBLEM 4*******************\n")
returns = CSV.read("c:/temp/problem4_returns.csv",DataFrame)
stWgt = vec(Matrix(CSV.read("c:/temp/problem4_startWeight.csv",DataFrame)))

select!(returns,Not(:Date))
attrn, weightUpd, fwghts = expost_factor(stWgt,returns,returns,I(3))
println(attrn)

#Problem 5

println("\n*******************PROBLEM 5*******************\n")
prices = CSV.read("C:/temp/problem5.csv",DataFrame)
returns = select(return_calculate(prices,dateColumn="Date"),Not(:Date))
lastP = last(prices,1)

vars = names(returns)

portfolio = DataFrame(:Asset=>["Price$i" for i in 1:4],
                      :P=>["1+2","1+2","3+4","3+4"],
                      :currentValue=>vec(Matrix(select(lastP,Not(:Date))))
 )


models = Dict{String,FittedModel}()
U = DataFrame()
for v in vars
    println(v)
    try
        models[v] = fit_general_t(returns[!,v])
    catch
        models[v] = fit_normal(returns[!,v])
    end
    U[!,v] = models[v].u
end

corr = corspearman(Matrix(U))

iters = 1000
assetVars = Array{Float64,2}(undef,(iters,4))
portVars = Array{Float64,2}(undef,(iters,2))
totVars = Array{Float64,2}(undef,(iters,1))

for iter in 1:iters
    nSim = 1000
    simU = DataFrame( cdf(Normal(),simulate_pca(corr,nSim,seed=iter)),vars)

    simR = DataFrame()
    for v in vars
        simR[!,v] = models[v].eval(simU[!,v])
    end

    iteration = [i for i in 1:nSim]
    values = crossjoin(portfolio, DataFrame(:iteration=>iteration))

    nv = size(values,1)
    pnl = Vector{Float64}(undef,nv)
    simulatedValue = copy(pnl)
    for i in 1:nv
        simulatedValue[i] = values.currentValue[i] * (1 + simR[values.iteration[i],values.Asset[i]])
        pnl[i] = simulatedValue[i] - values.currentValue[i]
    end

    values[!,:pnl] = pnl
    values[!,:simulatedValue] = simulatedValue

    assetRisk = aggRisk(values,[:Asset])
    assetVars[iter,:] = assetRisk.VaR95'

    portRisk = aggRisk(values,[:P])
    portVars[iter,:] = portRisk.VaR95'

    totRisk = aggRisk(values,Symbol[])
    totVars[iter] = totRisk.VaR95[1]
end

println("Asset 1: $(mean(assetVars[:,1])) [$(-VaR(assetVars[:,1])), $(-VaR(assetVars[:,1],alpha=.95))]")
println("Asset 2: $(mean(assetVars[:,2])) [$(-VaR(assetVars[:,2])), $(-VaR(assetVars[:,2],alpha=.95))]")
println("Asset 3: $(mean(assetVars[:,3])) [$(-VaR(assetVars[:,3])), $(-VaR(assetVars[:,3],alpha=.95))]")
println("Asset 4: $(mean(assetVars[:,4])) [$(-VaR(assetVars[:,4])), $(-VaR(assetVars[:,4],alpha=.95))]")

println("Portfolio 1+2: $(mean(portVars[:,1])) [$(-VaR(portVars[:,1])), $(-VaR(portVars[:,1],alpha=.95))]")
println("Portfolio 3+4: $(mean(portVars[:,2])) [$(-VaR(portVars[:,2])), $(-VaR(portVars[:,2],alpha=.95))]")

println("Total: $(mean(totVars[:,1])) [$(-VaR(totVars[:,1])), $(-VaR(totVars[:,1],alpha=.95))]")