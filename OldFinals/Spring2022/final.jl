using BenchmarkTools
using Distributions
using Random
using StatsBase
using DataFrames
using Roots
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
using Printf
# using GLM

include("../library/RiskStats.jl")
include("../library/simulate.jl")
include("../library/fitted_model.jl")
include("../library/return_calculate.jl")
include("../library/gbsm.jl")
include("../library/missing_cov.jl")

#Question #2
# d = Normal(0,.01*sqrt(365))
# sim = rand(d,1000)
# simP = 100 * (1 .+ sim)
# pval = gbsm(false,100,100,1,.02,.02,.22).value
# payoff = -max.(0,100 .- simP)
# pnl = payoff .+ pval

# CSV.write("question2.csv",DataFrame(:data=>pnl))
pnl = CSV.read("question2.csv",DataFrame).data

println("Mean: $(mean(pnl))")
println("StDev: $(std(pnl))")
println("Variance: $(var(pnl))")
println("Skew: $(skewness(pnl))")
println("Kurt: $(kurtosis(pnl))")
println(" ")
println("VaR: $(VaR(pnl))")
println("ES: $(ES(pnl))")

# Mean: 0.6721682232440969
# StDev: 10.554533111422966
# Variance: 111.39816920012377
# Skew: -1.6289300391221222
# Kurt: 2.228214792921505
# Negative Skew and excess kurtosis make this a risky investment even if it has a positive expected return.
# This is seen with the large VaR and ES numbers.

#Question 3
c = [1 .5 .5; 0.5 1 0.5; 0.5 0.5 1]
v = [.1, .1, .1]
cv = diagm(v)*c*diagm(v)

#Generate some random numbers with missing values.
function generate_with_missing(r; pmiss=.25)
    n,m = size(r)
    x = Array{Union{Missing,Float64},2}(undef,n,m)

    for i in 1:n, j in 1:m
        if rand() >= pmiss
            x[i,j] = r[i,j]
        end
    end
    return x
end

# Random.seed!(3)
# r = rand(MvNormal(fill(0,3),cv),15)'
# x = generate_with_missing(r,pmiss=.4)
# CSV.write("question3.csv",DataFrame(x,[:var1, :var2, :var3]))

r = CSV.read("question3.csv",DataFrame)
x = Matrix(r)

pairwise = missing_cov(x,skipMiss=false,fun=cor)

eVal = eigvals(pairwise)

fixed = higham_nearestPSD(pairwise)

#  A.
# julia> pairwise
# 3×3 Matrix{Float64}:
#   1.0      -0.62016    0.82955
#  -0.62016   1.0       -0.989449
#   0.82955  -0.989449   1.0

#  B.
# julia> eVal
# 3-element Vector{Float64}:
#  -0.03222839607878862
#   0.3969001805241572
#   2.6353282155546314

#  C.
# julia> fixed
# 3×3 Matrix{Float64}:
#   1.0       -0.629592   0.816847
#  -0.629592   1.0       -0.962454
#   0.816847  -0.962454   1.0

# Question 4
# Calculate the covariance of X, Y1, and Y2.  Check that the matrix is PSD.  Simulate from that covariance matrix.
# No need to fit the structural model because everything is normally distributed and the conditional distributions 
# are the same 

# Question 5
# fit the models to get the parameters and U values for the e1 and e2 variables.  Find the U values for x
# Use the spearman correlation between the e1, e2, and X U values to fit the Gaussian copula.
# Simulate from the copula and transform back to e1, e2, and x
# use the fitted Alpha and Beta values to transform e1, e2, and X into Y1 and Y2.

# Question 6
pval = 7.18
rf = 0.02
b = 0.02
X = 100
S = 100
ttm = 1
f(iv) = gbsm(false,S,X,ttm,rf,b,iv).value - pval
implied_vol = find_zero(f,0.2)

# julia> implied_vol
# 0.2062422068928017

vars = Vector{Float64}(undef,1000)
ess = Vector{Float64}(undef,1000)

for j in 1:1000
    d = Normal(0.04/255,.2/sqrt(255))
    r = rand(d,1000)
    S = 100 * exp.(r)
    π =  [gbsm(false,S[i],X,254/255,rf,b,implied_vol).value for i in 1:1000]
    pnl = -1 * (π .- pval)

    vars[j] = VaR(pnl)
    ess[j] = ES(pnl)
end

println("VaR Range: $(mean(vars)) 95% confidence [$(quantile(vars,.025)), $(quantile(vars,.975))]")
println("ES Range : $(mean(ess)) 95% confidence [$(quantile(ess,.025)), $(quantile(ess,.975))]")

# VaR Range: 0.9543017430085824 95% confidence [0.8798010127757914, 1.039654382592531]
# ES Range : 1.2068392953197968 95% confidence [1.1152764224961653, 1.3070400985742867]

# 7
delta = gbsm(false,100,100,1,rf,b,implied_vol).delta
# julia> delta
# -0.420703351245022
# She is short the option, so her delta is 0.42
# She should buy or sell the negative of her position delta.  So she should
# sell 0.42 shares for each put sold.

# 8
corel = [1 .5 .5; .5 1 .5; .5 .5 1]
vols = [0.1, 0.2, 0.3]
er = [0.03, 0.06, .09]

covar = diagm(vols)*corel*diagm(vols)

function sr(w...)
    _w = collect(w)
    m = _w'*er
    s = sqrt(_w'*covar*_w)
    return (m/s)
end

n = 3

m = Model(Ipopt.Optimizer)
# set_silent(m)
# Weights with boundry at 0
@variable(m, w[i=1:n] ,start=1/n)
register(m,:sr,n,sr; autodiff = true)
@NLobjective(m,Max, sr(w...))
@constraint(m, sum(w)==1.0)
optimize!(m)

w = round.(value.(w),digits=4)
# julia> w
# 3-element Vector{Float64}:
#  0.5455
#  0.2727
#  0.1818

# 9
w = [1,1,1]/3
risk_contrib = w .* covar*w / (sqrt(w'*covar*w))
# 3-element Vector{Float64}:
#  0.023333333333333334
#  0.05333333333333334
#  0.09

# Correlations and Sharpe ratios are equal -> risk parity is the maximum sharpe ratio portfolio
# julia> w
# 3-element Vector{Float64}:
#  0.5455
#  0.2727
#  0.1818

# 10
w = [0.55, 0.27, 0.18]
nday=10
# r = rand(MvNormal(er/12,covar/12),nday)'
# CSV.write("question10.csv",DataFrame(r,[:A, :B, :C]))

stocks = [:A, :B, :C]
upReturns = CSV.read("question10.csv",DataFrame)

#calculate portfolio return and updated weights for each day
n = size(upReturns,1)
m = size(stocks,1)

pReturn = Vector{Float64}(undef,n)
weights = Array{Float64,2}(undef,n,length(w))
lastW = copy(w)
matReturns = Matrix(upReturns[!,stocks])

for i in 1:n
    # Save Current Weights in Matrix
    weights[i,:] = lastW

    # Update Weights by return
    lastW = lastW .* (1.0 .+ matReturns[i,:])
    
    # Portfolio return is the sum of the updated weights
    pR = sum(lastW)
    # Normalize the wieghts back so sum = 1
    lastW = lastW / pR
    # Store the return
    pReturn[i] = pR - 1
end

# Set the portfolio return in the Update Return DataFrame
upReturns[!,:Portfolio] = pReturn

# Calculate the total return
totalRet = exp(sum(log.(pReturn .+ 1)))-1
# Calculate the Carino K
k = log(totalRet + 1 ) / totalRet

# Carino k_t is the ratio scaled by 1/K 
carinoK = log.(1.0 .+ pReturn) ./ pReturn / k
# Calculate the return attribution
attrib = DataFrame(matReturns .* weights .* carinoK, stocks)

# Set up a Dataframe for output.
Attribution = DataFrame(:Value => ["TotalReturn", "Return Attribution"])
# Loop over the stocks
for s in vcat(stocks,:Portfolio)
    # Total Stock return over the period
    tr = exp(sum(log.(upReturns[!,s] .+ 1)))-1
    # Attribution Return (total portfolio return if we are updating the portfolio column)
    atr =  s != :Portfolio ?  sum(attrib[:,s]) : tr
    # Set the values
    Attribution[!,s] = [ tr,  atr ]
end

# julia> Attribution
# 2×5 DataFrame
#  Row │ Value               A          B           C          Portfolio 
#      │ String              Float64    Float64     Float64    Float64   
# ─────┼─────────────────────────────────────────────────────────────────
#    1 │ TotalReturn         0.0530015  -0.0891345  0.182844   0.0379964
#    2 │ Return Attribution  0.0301564  -0.0274756  0.0353156  0.0379964


#EC 1
# Random.seed!(10)
# d = Normal(.06,.01*sqrt(365))
# sim = rand(d,1000)
# simP = 100 * (1 .+ sim)
# pval = gbsm(false,100,100,1,.02,.02,.22).value
# payoff = max.(0,100 .- simP)
# put_ret = payoff / pval .- 1
# simR = simP/100 .- 1
# returns = DataFrame(:A=>simR, :B=>put_ret)

# CSV.write("ec1.csv",returns)

returns = CSV.read("ec1.csv",DataFrame)
mret = Matrix(returns)
covar = cov(Matrix(returns))
er = mean.(eachcol(returns))
rf = 0.02

function _sr(w...)
    _w = collect(w)
    m = _w'*er - rf
    s = sqrt(_w'*covar*_w)
    return (m/s)
end

#Brute Force Find the maximum
rng = -1:.0001:2
vals = fill(0.0,length(rng),3)

i=1
for a in rng
    s = _sr(a,1-a)
    vals[i,:] = [a, 1-a, s]
    i+=1
end
vals[argmax(vals[:,3]),[1,2]]

# 2-element Vector{Float64}:
#   2.0
#  -1.0

# Part B
function esr(w...)
    _w = collect(w)
    m = _w'*er -rf
    s = ES(mret*_w)
    return (1e6*m/s)
end

rng = -1:.0001:2
vals = fill(0.0,length(rng),3)

i=1
for a in rng
    s = esr(a,1-a)
    vals[i,:] = [a, 1-a, s]
    i+=1
end
vals[argmax(vals[:,3]),[1,2]]

# 2-element Vector{Float64}:
#  0.9284
#  0.0716

# the key to understanding the difference in optimized results is looking at the higher moments of asset B:
println(skewness(returns.B), " - ", kurtosis(returns.B))
#           2.2284282050577917 - 4.924134482821873
# compare that to the sharpe ratio of being fully short B
println(-sr(0,1))
# 0.3264848837220196
# and compare that to the sharpe ratio of being fully long a
println(sr(1,0))
# 0.2184928741410699

# Part C
# so while shorting B produces a positive expected return and high sharpe ratio than A the downside risk is much larger 
# ES takes the tail risk into account and chooses A over B.  Some B is choosen because of the negative correlation
# and it's ability to reduce the risk of being only long A.
println(ES(mret*[2.0,-1.0]))
println(ES(mret*[.9284,.0716]))
# 4.079742406893754 -- maximum sharpe ratio portfolio has a 408% expected shortfall!
# 0.07183484571264609 -- maximum risk/return using ES has a 7.2% expected shortfall!


# EC2
# Random.seed!(20)
# x = rand(Normal(0,0.02),100)
# e1 = rand(TDist(8)*.001,100)
# e2 = rand(TDist(10)*.002,100)
# Y1 = -0.001 .+ 1.2*x .+ e1
# Y2 = 0.001 .+ 0.8*x .+ e2

# CSV.write("ec2.csv",DataFrame(:x=>x, :y1=>Y1, :y2=>Y2))

stocks = CSV.read("ec2.csv",DataFrame)
nms = names(stocks)


fittedModels = Dict{String,FittedModel}()

fittedModels["y1"] = fit_regression_t(stocks.y1,stocks.x)
fittedModels["y2"] = fit_regression_t(stocks.y2,stocks.x)
fittedModels["x"] = fit_normal(stocks.x)

U = DataFrame()
for nm in nms
    U[!,nm] = fittedModels[nm].u
end

R = corspearman(Matrix(U))
evals = eigvals(R) # Passes

outResults = DataFrame()

for k in 1:1000
    NSim = 5000
    simU = DataFrame(
                #convert standard normals to U
                cdf(Normal(),
                    simulate_pca(R,NSim,seed=k)  #simulation the standard normals
                )   
                , nms
            )

    simulatedReturns = DataFrame()
    simulatedReturns[!,:x] = fittedModels["x"].eval(simU[!,:x])
    simulatedReturns[!,:y1] = fittedModels["y1"].eval(simulatedReturns.x, simU[!,:y1])
    simulatedReturns[!,:y2] = fittedModels["y2"].eval(simulatedReturns.x, simU[!,:y2])

    portfolio = DataFrame(:Stock=>["y1", "y2"], :Holding=>[100,100])
    currentPrice = DataFrame(:y1=>10.0, :y2=>50.0)[1,:]

    iteration = [i for i in 1:NSim]
    values = crossjoin(portfolio, DataFrame(:iteration=>iteration))

    nVals = size(values,1)
    currentValue = Vector{Float64}(undef,nVals)
    simulatedValue = Vector{Float64}(undef,nVals)
    pnl = Vector{Float64}(undef,nVals)
    for i in 1:nVals
        price = currentPrice[values.Stock[i]]
        currentValue[i] = values.Holding[i] * price
        simulatedValue[i] = values.Holding[i] * price*(1.0+simulatedReturns[values.iteration[i],values.Stock[i]])
        pnl[i] = simulatedValue[i] - currentValue[i]
    end
    values[!,:currentValue] = currentValue
    values[!,:simulatedValue] = simulatedValue
    values[!,:pnl] = pnl

    append!(outResults, aggRisk(values,Vector{Symbol}(undef,0)))
end

function printRange(x,nm,alpha=0.05)
    m = mean(x)
    up = quantile(x,1-alpha/2)
    dn = quantile(x,alpha/2)
    @sprintf("%s Mean: %0.2f -- %0.2f%% range[%0.2f, %0.2f]",nm, m,alpha*100, dn,up)
end

println(printRange(outResults.VaR95,"VaR"))
println(printRange(outResults.ES95,"ES"))
# VaR Mean: 158.53 -- 5.00% range[152.44, 164.95]
# ES Mean: 201.80 -- 5.00% range[194.47, 209.09]