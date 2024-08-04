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
using StateSpaceModels
# using GLM

include("../library/RiskStats.jl")
include("../library/simulate.jl")
include("../library/fitted_model.jl")
include("../library/return_calculate.jl")
include("../library/gbsm.jl")
include("../library/missing_cov.jl")

#Question #2
rf = 0.04
ttm = 30/365
ivol = 0.2
price = 100
strike = 0.9*price

val = gbsm(false,price,strike,ttm,rf,rf,ivol,includeGreeks=true)

println("Value: $(-val.value)")
println("Delta: $(-val.delta)")
println("Gamma: $(-val.gamma)")
println("Vega: $(-val.vega)")
println("Theta: $(-val.theta)")

# Value: -0.06081841007028865
# Delta: 0.02720649850677792
# Gamma: -0.01094018348983649
# Vega: -1.798386327096409
# Theta: 2.076777967537374

val2 = gbsm(false,price,strike,25/365,rf,rf,ivol,includeGreeks=true)
println("Value: $(-val2.value)")
println("Delta: $(-val2.delta)")
println("Gamma: $(-val2.gamma)")
println("Vega: $(-val2.vega)")
println("Theta: $(-val2.theta)")

# Value: -0.03527884450904728
# Delta: 0.018245062574478
# Gamma: -0.00855545669157836
# Vega: -1.1719803687093642
# Theta: 1.6366999342373985


# Question 3
# fwdPrices = round.(price * (1 .+ rand(Normal(0.005,0.18*sqrt(5/365)),5000)),digits=2)
# CSV.write("question3.csv",DataFrame(:fwdPrices=>fwdPrices))

fwdPrices = CSV.read("question3.csv",DataFrame).fwdPrices
pnl = val.value .- [ v.value for v in gbsm.(false,fwdPrices,strike,25/365,rf,rf,ivol)]

density(pnl)
scatter(fwdPrices,pnl)

println("Mean: $(mean(pnl))")
println("StDev: $(std(pnl))")
println("Variance: $(var(pnl))")
println("Skew: $(skewness(pnl))")
println("Kurt: $(kurtosis(pnl))")
println(" ")
println("VaR: $(VaR(pnl))")
println("ES: $(ES(pnl))")

# Mean: 0.014612270184060094
# StDev: 0.0562832911011184
# Variance: 0.0031678088571732337
# Skew: -3.6496331041648875
# Kurt: 21.642387486679375

# VaR: 0.09044738065107305
# ES: 0.16941224788432385
# Negative Skew and excess kurtosis make this a risky investment even if it has a positive expected return.
# This is seen with the large VaR and ES numbers.

#4
# Decide on an ex-Ante risk model.  
# Calculate the weight of each stock in the portfolio.
# Calculate the gradient of risk wrt stocks.
# Multiply each partial derivative with that stock's weight.
# The result is the ex-ante contribution to risk.

#5
#AR3
#y_t = 1.0 + 0.5*y_t-3 + e, e ~ N(0,0.1)
# n = 1000
# burn_in = 50
# y = Vector{Float64}(undef,n)

# yt_last = [1.0,1.0,1.0]
# d = Normal(0,0.1)
# e = rand(d,n+burn_in)

# for i in 1:(n+burn_in)
#     global yt_last
#     y_t = 1.0 + 0.5*yt_last[3] + e[i]
#     yt_last[3] = yt_last[2]
#     yt_last[2] = yt_last[1]
#     yt_last[1] = y_t
#     if i > burn_in
#         y[i-burn_in] = y_t
#     end
# end
# ar1 = SARIMA(y,order=(3,0,0),include_mean=true)

# StateSpaceModels.fit!(ar1)
# print_results(ar1)
# CSV.write("question5.csv",DataFrame(:data=>y))

y = CSV.read("question5.csv",DataFrame).data

function plot_ts(y;imgName="series", length=10,title=nothing)
    n = size(y,1)
    l = [i for i in 1:length]
    acf = autocor(y,l)
    p_acf = pacf(y, l)

    df = DataFrame(:t=>l, :acf=>acf, :pacf=>p_acf)
   
    theme(:dark)

    if title === nothing
        p0 = Plots.plot([i for i in 1:n], y, legend=false)
    else
        p0 = Plots.plot([i for i in 1:n], y, legend=false,title=title)
    end

    p1 = Plots.plot(df.t, df.acf, title="AutoCorrelation", seriestype=:bar, legend=false)
    p2 = Plots.plot(df.t, df.pacf, title="Partial AutoCorrelation", seriestype=:bar, legend=false)
    p = plot(p0, plot(p1,p2,layout=(1,2)),layout=(2,1))

    Plots.savefig(p,imgName)
    p
end
plot_ts(y;imgName="question5.png", length=10,title=nothing)
# Curve ball.  This is an AR(3) process where the AR(1) and AR(2) betas are 0.  We 
# can see from the PACF graph the only significant value is at lag=3.  The ACF has significant
# values at 3, 6, and 9.  These fall off in the typical pattern of an AR process.  

#6 
prices = CSV.read("question6.csv",DataFrame)
prices[!,:Date] = Date.(prices.Date,dateformat"mm/dd/yyyy")

returns = return_calculate(prices,dateColumn="Date")

stocks = [:AAPL, :AMZN]

currentPrices = last(prices,1)
currentValue = 100*Matrix(currentPrices[!,stocks])
totValue = sum(currentValue)
currentW = currentValue /totValue


covar = cov(Matrix(returns[!,stocks]))
σ = sqrt(currentW*covar*currentW')[1]

VaR95 = -quantile(Normal(),0.05)*σ
VaR99 = -quantile(Normal(),0.01)*σ
println("VaR 95%: $VaR95 : \$$(VaR95*totValue))")
println("VaR 99%: $VaR99 : \$$(VaR99*totValue))")
# VaR 95%: 0.03914663800861933 : $947.1137404072172)
# VaR 99%: 0.05536583718758151 : $1339.5210372334284)

#7
# fit the models to get the parameters and U values for the e1 and e2 variables.  Find the U values for SPY
# Use the spearman correlation between the e1, e2, and SPY U values to fit the Gaussian copula.
# Simulate from the copula and transform back to e1, e2, and SPY
# use the fitted Alpha and Beta values to transform e1, e2, and X into AMZN and AAPL.
# calculate the PnL for each simulations and calculate VaR

#8
# Standard Deviation and Expected Shortfall are coherent risk measures while VaR is not.
# Standard Deviation assumes a Symmetric distribution of returns where VaR and ES do not
# VaR and ES do not rely on a symmetric distribution of returns and take into account higher moments
# If a normal distribution is assumed, then SD, VaR, and ES are equivilent with VaR and ES 
#     being a multiple of SD 
# Most portfolio metrics assume normallity and as such use SD
# Breaking the normallity assumption for portfolio construction it is best to use ES as it is
# coherent and convex, making the optimization easier to solve.

#9
corel = [1 .7 .4
         .7 1 .6
          .4 .6 1]
vols = [0.1, 0.2, 0.3]
er = [0.05, 0.07, .09]
rf = 0.04

covar = diagm(vols)*corel*diagm(vols)

function sr(w...)
    _w = collect(w)
    m = _w'*er - rf
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

wop = round.(value.(w),digits=4)
# 3-element Vector{Float64}:
#  -0.0676
#   0.5405
#   0.527

#10
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

m = Model(Ipopt.Optimizer)

# Weights with boundry at 0
@variable(m, w[i=1:n] >= 0,start=1/n)
register(m,:distSSE,n,sseCSD; autodiff = true)
@NLobjective(m,Min, distSSE(w...))
@constraint(m, sum(w)==1.0)
optimize!(m)
wrp = round.(value.(w),digits=4)
# RP Weights
# 3-element Vector{Float64}:
#  0.5542
#  0.2545
#  0.1913

println("ER Optimal: $(wop'*er)")
println("SD Optimal: $(sqrt(wop'*covar*wop))")
println("SR Optimal: $((wop'*er - rf) / sqrt(wop'*covar*wop))")
println(" ")
println("ER RP: $(wrp'*er)")
println("SD RP: $(sqrt(wrp'*covar*wrp))")
println("SR Optimal: $((wrp'*er - rf) / sqrt(wrp'*covar*wrp))")
# ER Optimal: 0.081885
# SD Optimal: 0.23528192960786426
# SR Optimal: 0.1780204713120476

# ER RP: 0.062742
# SD RP: 0.13767662089113025
# SR Optimal: 0.16518418198238294
# The weights are drastically different between the portfolios and the SD for the optimal portfolio
# is very high.  This is because the high risk free rate pushes the tangency portfolio further output
# the risk curve.  
# the drastically different wieghts do not cause a large difference in SR, however.  

#EC1:
# rf = 0.003
# fcor = [1 .2
#         .2 1]
# fsd = [.1, .2]
# fer = [.05, .07]/12
# fcov = diagm(fsd)*fcor*diagm(fsd)/12

# rcorel = [1 .3 .3
#          .3 1 .2
#           .3 .2 1]
# rsd = [.02, 0.02, 0.02]
# rcov = diagm(rsd)*rcorel*diagm(rsd)/12

# beta = [.75 .5
#         .85 .2
#         .55 0]
# nhist = 100
# fsim = simulate_pca(fcov,nhist,mean=fer,seed=12345)
# rsim = simulate_pca(rcov,nhist,seed=54321)
# ssim = fsim*beta' .+ rsim .+ rf
# CSV.write("ec1_history.csv",DataFrame(hcat(ssim,fsim),[:S1, :S2, :S3, :F1, :F2]))

# covall = cov(hcat(ssim,fsim))
# erall = [(beta*fer .+ rf)..., fer...]
# nFwd = 10
# fwdsim = simulate_pca(covall,nFwd,mean=erall,seed=13579)
# CSV.write("ec1_fwd.csv",DataFrame(fwdsim,[:S1, :S2, :S3, :F1, :F2]))


hist = CSV.read("ec1_history.csv",DataFrame)

stocks = [:S1, :S2, :S3]
factors = [:F1, :F2]

rf = 0.003

X = hcat(fill(1.0,size(hist,1)),Matrix(hist[!,factors]))
Y = Matrix(hist[!,stocks]).- rf
Betas = (inv(X'*X)*X'*Y)'
# julia> Betas
# 3×3 adjoint(::Matrix{Float64}) with eltype Float64:
#   0.000228228  0.730244   0.487929
#  -0.000591213  0.885972   0.203751
#   4.04352e-5   0.555206  -0.00576451

Betas = Betas[:,2:3]

fwdRet = CSV.read("ec1_fwd.csv",DataFrame)
fwdRet[!,stocks] = fwdRet[!,stocks] .- rf

w = [.3, .45, .25]

attrn = expost_factor(w,fwdRet[!,stocks],fwdRet[!,factors],Betas)
# 3×5 DataFrame
#  Row │ Value               F1         F2           Alpha       Portfolio 
#      │ String              Float64    Float64      Float64     Float64   
# ─────┼───────────────────────────────────────────────────────────────────
#    1 │ TotalReturn         0.0393903  -0.120548    0.00944419  0.0116767
#    2 │ Return Attribution  0.030095   -0.0277782   0.00935993  0.0116767
#    3 │ Vol Attribution     0.0170657   0.00639294  0.00195266  0.0254113

# EC2
Random.seed!(20)
x = rand(Normal(0,0.02),100)
en = cdf.(Normal(),rand(MultivariateNormal([0,0],[1.0 .9;.9 1.0]),100)')
e1 = quantile.(TDist(8)*.0015,en[:,1])
e2 = quantile.(TDist(12)*.0022,en[:,2])
Y1 = -0.001 .+ 1.2*x .+ e1
Y2 = 0.001 .+ 0.8*x .+ e2

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

    append!(outResults, aggRisk(values,Symbol[]))
end

function printRange(x,nm,alpha=0.05)
    m = mean(x)
    up = quantile(x,1-alpha/2)
    dn = quantile(x,alpha/2)
    @sprintf("%s Mean: %0.2f -- %0.2f%% range[%0.2f, %0.2f]",nm, m,alpha*100, dn,up)
end

println(printRange(outResults.VaR95,"VaR"))
println(printRange(outResults.ES95,"ES"))
# VaR Mean: 177.99 -- 5.00% range[171.94, 183.65]
# ES Mean: 220.73 -- 5.00% range[214.10, 227.69]