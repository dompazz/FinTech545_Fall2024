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

include("../library/RiskStats.jl")
include("../library/simulate.jl")
include("../library/fitted_model.jl")
include("../library/return_calculate.jl")
include("../library/gbsm.jl")
include("../library/missing_cov.jl")
include("../library/expost_factor.jl")
include("../library/ewCov.jl")

#Problem 1
#Data Generation

# P0=42
# s = .02
# n=20
# Random.seed!(1)
# r = rand(Normal(0,s),n) .+ 1
# P = Vector{Float64}(undef,n)
# P[1] = P0*r[1]
# for i in 2:n
#     P[i] = P[i-1]*r[i]
# end

# CSV.write("problem1.csv",DataFrame(:XYZ=>P))
# P = nothing
# r = nothing
# n = nothing
# s = nothing


prices = CSV.read("problem1.csv",DataFrame)
prices[!,:Date] .= Date(2023,12,17)
returns = return_calculate(prices,dateColumn="Date",method="LOG")
println("Returns")
println(select(returns,:XYZ))
# 19×1 DataFrame
#  Row │ XYZ
#      │ Float64      
# ─────┼──────────────
#    1 │  0.00555267
#    2 │ -0.0119881
#    3 │  0.000932753
#    4 │  0.0214834
#    5 │ -0.0320391
#    6 │  0.00351262
#    7 │  0.0171595
#    8 │ -0.0574232
#    9 │ -0.0385748
#   10 │  0.00434811
#   11 │ -0.0132058
#   12 │  0.00533602
#   13 │  0.00014771
#   14 │  0.0209926
#   15 │ -0.0309598
#   16 │  0.0192539
#   17 │  0.0304596
#   18 │ -0.0117528
#   19 │  0.00931149

# r = rand(TDist(5),1000)
# returns = DataFrame(:XYZ=>r)

fitNormal = fit_normal(returns.XYZ)
nLL = sum(log.(pdf.(fitNormal.errorModel,returns.XYZ)))
println(fitNormal.errorModel, "  ll:$nLL AICc:$(4-2*nLL + (12/(19-2-1)))")
# Normal{Float64}(μ=-0.003023851474658957, σ=0.023245414231625357)  ll:45.01146854040041 AICc:-85.27293708080082

fitT = fit_general_t(returns.XYZ)
tLL = sum(log.(pdf.(fitT.errorModel,returns.XYZ)))
println(fitT.errorModel, "  ll:$tLL AICc:$(6-2*tLL + ((2*3^2 + 2*3)/(19-3-1)))")
# LocationScale{Float64, Continuous, TDist{Float64}}(
# μ: -0.003023166013984878
# σ: 0.022624589378910685
# ρ: TDist{Float64}(ν=24797.225388947903)
# )
#   ll:45.025074814764615 AICc:-82.45014962952924

#AICc is lower for the Normal Distribution.  Though both are effectively the same given the ν≈41,000. +1 point if student recognizes this

# Problem 2
# Underlying = 97.25
# Strike = 100
# TTM = 150
# RF = 0.0525
# DivRate = 0.0
# CallPrice = gbsm(true,Underlying,Strike,TTM/255,RF,RF,0.3)
# CSV.write("problem2.csv",DataFrame(:Underlying=>[Underlying], :Strike=>[Strike], :TTM=>[TTM], :RF=>[RF],:DivRate=>[DivRate], :CallPrice=>[round(100*CallPrice.value)/100]))
# Underlying = nothing
# Strike = nothing
# TTM = nothing
# RF = nothing
# DivRate = nothing
# CallPrice = nothing

p2Data = CSV.read("problem2.csv",DataFrame)
S = p2Data.Underlying[1]
X = p2Data.Strike[1]
TTM = p2Data.TTM[1]/255
rf = p2Data.RF[1]
cPrice = p2Data.CallPrice[1]

function _cp(iv)
    gbsm(true,S,X,TTM,rf,rf,iv).value - cPrice
end

iv = find_zero(_cp,0.01)
println("Implied Vol: $iv")
# Implied Vol: 0.2999965250180758

pPrice = cPrice + X*exp(-rf*TTM) - S
println("Put Price: $pPrice")
# Put Price: 8.748963573523639

pPrice = gbsm(false,S,X,TTM,rf,rf,iv,includeGreeks=true)
cPrice = gbsm(true,S,X,TTM,rf,rf,iv,includeGreeks=true)

println("Call Delta: $(cPrice.delta)  Call Vega: $(cPrice.vega)")
println("Put Delta: $(pPrice.delta)  Put Vega: $(pPrice.vega)")
# Call Delta: 0.5509530463057839  Call Vega: 29.51301683812504
# Put Delta: -0.4490469536942161  Put Vega: 29.51301683812504

sValue = (cPrice.value + pPrice.value)
pnl_exact = (gbsm(false,S,X,TTM,rf,rf,iv-.05).value + gbsm(true,S,X,TTM,rf,rf,iv-.05).value) - sValue
#Use Vega values to approximate the change
pnl_approx = (cPrice.vega + pPrice.vega) * -0.05
println("Exact: $pnl_exact  Approx: $pnl_approx")
# Exact: -2.9543278752519555  Approx: -2.9513016838125044

# Problem 3
pnl = (gbsm(false,S,X,TTM-100/255,rf,rf,iv).value + gbsm(true,S,X,TTM-100/255,rf,rf,iv).value) - sValue
println("Profit/Loss: $pnl")
# Profit/Loss: -7.274883302621589

Random.seed!(3)

vars = Vector{Float64}(undef,1000)
ess = Vector{Float64}(undef,1000)
# Calculating VaR/ES 1000 times for a distribution to compare for grading
for j in 1:1000
    simR = rand(fitNormal.errorModel,1000) .+ 1
    cprices = [gbsm(true,S*simR[i],X,TTM-1/255,rf,rf,iv).value for i in 1:1000]
    pprices = [gbsm(false,S*simR[i],X,TTM-1/255,rf,rf,iv).value for i in 1:1000]
    pnl = cprices + pprices .- sValue
    vars[j] = VaR(pnl)
    ess[j] = ES(pnl)
end
#Values with a 2sd interval
println("VaR: $(mean(vars)) [$(mean(vars)-2*std(vars)),$(mean(vars)+2*std(vars))]")
println("ES: $(mean(ess)) [$(mean(ess)-2*std(ess)),$(mean(ess)+2*std(ess))]")
# VaR: 0.19777620482144298 [0.19710376904326485,0.1984486405996211]
# ES: 0.19861120608181776 [0.19835325915834243,0.1988691530052931]

# Problem 4

# corr = convert.(Float64,Matrix(I(3)))
# corr[1,2] = .5
# corr[2,1] = .5
# corr[1,3] = -.1
# corr[3,1] = -.1
# sd = [.01, .02, .05]
# covar = diagm(sd)*corr*diagm(sd)

# mu = [.001, .001, .001]
# r = simulateNormal(100,covar,mean=mu,seed=5)
# CSV.write("problem4.csv",DataFrame(r,["A","B","C"]))

returns = CSV.read("problem4.csv",DataFrame)

covar = ewCovar(Matrix(returns),0.94)
# 3×3 Matrix{Float64}:
#   0.000104089  0.000111925  -6.43634e-5
#   0.000111925  0.00037909    0.000116709
#  -6.43634e-5   0.000116709   0.00324066

w = [0.3, 0.45, 0.25]
pvol = sqrt(w'*covar*w)
csd = (w .* ((covar*w)/pvol))/pvol
# 3-element Vector{Float64}:
#  0.05857128338149312
#  0.3129820167541605
#  0.6284466998643463

csd*pvol
# 3-element Vector{Float64}:
#  0.0010728298485864914
#  0.0057327828631932734
#  0.011511039863490176

function _sr(w...)
    _w = collect(w)
    m = _w'*mu - rf
    s = sqrt(_w'*covar*_w)
    return (m/s)
end

mu = mean.(eachcol(Matrix(returns)))
rf = (1.0525)^(1/365) - 1
sr = _sr(w...)
println("Sharpe Ratio: $sr")
# Sharpe Ratio: 0.1996449873892672


m = Model(Ipopt.Optimizer)
# set_silent(m)
# Weights with boundry at 0
@variable(m, w[i=1:3] >= 0,start=1/3)
register(m,:sr,3,_sr; autodiff = true)
@NLobjective(m,Max, sr(w...))
@constraint(m, sum(w)==1.0)
optimize!(m)

w = round.(value.(w),digits=4)
# 3-element Vector{Float64}:
#  -0.0
#   0.9648
#   0.0352

# Problem 5
# corr = convert.(Float64,Matrix(I(4)))
# corr[1,2] = .5
# corr[2,1] = .5
# corr[1,3] = -.1
# corr[3,1] = -.1
# sd = [.01, .02, .05, .03]

# d = Vector{UnivariateDistribution}(undef,4)
# for i in 1:4
#     d[i] = TDist(i*3+1)*sd[i]
# end

# uSim = cdf.(Normal(),simulateNormal(100,corr,seed=50))
# for i in 1:4
#     uSim[:,i] = quantile.(d[i],uSim[:,i]) .+ 1

#     for j in 2:100
#         uSim[j,i] *= uSim[j-1,i]
#     end
# end

# startP = [99.5, 42.2, 150.1, 69.0]
# sim = uSim .* startP'

# CSV.write("problem5.csv",DataFrame(sim,["A","B","C","D"]))


prices = CSV.read("problem5.csv",DataFrame)
prices[!,:Date] .= Dates.today()
returns = select(return_calculate(prices,dateColumn="Date"),Not(:Date))
lastP = last(prices,1)

vars = names(returns)

portfolio = DataFrame(:Asset=>vars,
                      :P=>["A+B","A+B","C+D","C+D"],
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

# Run it 1000 times to get confidence intervals
for iter in 1:iters
    nSim = 1000
    simU = DataFrame( cdf(Normal(),simulate_pca(corr,nSim,seed=iter*2)),vars)

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

    risk = aggRisk(values,[:P, :Asset])

    assetVars[iter,:] = filter(r->r.Asset != "Total", risk).ES95
    portVars[iter,:] = filter(r->r.Asset == "Total" && r.P != "Total", risk).ES95
    totVars[iter,:] = filter(r->r.Asset == "Total" && r.P == "Total", risk).ES95
end

println("Asset 1: $(mean(assetVars[:,1])) [$(-VaR(assetVars[:,1])), $(-VaR(assetVars[:,1],alpha=.95))]")
println("Asset 2: $(mean(assetVars[:,2])) [$(-VaR(assetVars[:,2])), $(-VaR(assetVars[:,2],alpha=.95))]")
println("Asset 3: $(mean(assetVars[:,3])) [$(-VaR(assetVars[:,3])), $(-VaR(assetVars[:,3],alpha=.95))]")
println("Asset 4: $(mean(assetVars[:,4])) [$(-VaR(assetVars[:,4])), $(-VaR(assetVars[:,4],alpha=.95))]")

println("Portfolio 1+2: $(mean(portVars[:,1])) [$(-VaR(portVars[:,1])), $(-VaR(portVars[:,1],alpha=.95))]")
println("Portfolio 3+4: $(mean(portVars[:,2])) [$(-VaR(portVars[:,2])), $(-VaR(portVars[:,2],alpha=.95))]")

println("Total: $(mean(totVars[:,1])) [$(-VaR(totVars[:,1])), $(-VaR(totVars[:,1],alpha=.95))]")

# Asset 1: 2.4206729866207923 [2.2601365779122453, 2.585347166080887]
# Asset 2: 2.18349939102899 [1.9945448393593623, 2.3877448090503086]
# Asset 3: 12.706246670839862 [11.557916530841435, 13.865995219740842]
# Asset 4: 4.575423182832173 [4.217202710016324, 4.957451708001945]
# Portfolio 1+2: 3.9142772249165514 [3.6332064666099666, 4.216140955915351]
# Portfolio 3+4: 12.738709250987 [11.569538514650063, 13.930906776147145]
# Total: 13.619749367624996 [12.49096872674308, 14.812303650213217]


# Problem 6
S = 100
X = 100
iv = .25
rf = 0.0525
putVal = gbsm(false,S,X,1,rf,rf,iv).value
Random.seed!(6)
r = rand(Normal(0.1,.25),100000) .+ 1
putSim = (max.(0,X .- r .* S) ./ putVal) .- 1
r .-= 1

returns = [r putSim]

covar = cov(returns)
# 2×2 Matrix{Float64}:
#   0.0627379  -0.295945
#  -0.295945    2.33532

mu = mean.(eachcol(returns))

msr = -99999
mw1 = 0.0
for w1 in -1:.0001:2
    sr = _sr(w1, 1-w1)
    if sr > msr
        msr = sr
        mw1 = w1
    end
end

println("Maximum SR: $msr")
println("W_stock: $mw1")
println("W_Opt: $(1-mw1)")
# Values will change based on simulated values, should be close
# Maximum SR: 0.19207392959839742
# W_stock: 1.089
# W_Opt: -0.08899999999999997


function _ses(w...)
    _w = collect(w)
    m = _w'*mu - rf
    s = ES(returns * _w,alpha=0.01)
    return (m/s)
end

msr = -99999
mw1 = 0.0
Threads.@threads for w1 in -1:.0001:2
    global msr,mw1
    sr = _ses(w1, 1-w1)
    if sr > msr
        msr = sr
        mw1 = w1
    end
end

println("Maximum SR with ES: $msr")
println("W_stock: $mw1")
println("W_Opt: $(1-mw1)")
# Values will change based on simulated values, should be close
# Maximum SR with ES: 0.37517651216233616
# W_stock: 0.9315
# W_Opt: 0.0685

# The put has a lot of positive skew with a negative expected value
# Shorting it gives you that expected value but the standard Deviation
# used in the Sharpe Ratio does not capture the long tail.