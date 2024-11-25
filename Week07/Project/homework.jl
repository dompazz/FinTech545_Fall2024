using BenchmarkTools
using Distributions
using Random
using StatsBase
using Roots
using QuadGK
using DataFrames
using Plots
using LinearAlgebra
using JuMP
using Ipopt
using Dates
using ForwardDiff
using FiniteDiff
using CSV
using LoopVectorization

include("../../library/return_calculate.jl")
include("../../library/gbsm.jl")
include("../../library/bt_american.jl")
include("../../library/RiskStats.jl")
include("../../library/simulate.jl")
include("../../library/fitted_model.jl")


s = 151.03
x = 165
ttm = (Date(2022,4,15)-Date(2022,3,13)).value/365
rf = 0.0425
b = 0.0053

#Calculate the GBSM Values.  Return Struct has all values
bsm_call = gbsm(true,s,x,ttm,rf,rf-b,.2,includeGreeks=true)
bsm_put = gbsm(false,s,x,ttm,rf,rf-b,.2,includeGreeks=true)

outTable = DataFrame(
    :Valuation => ["GBSM","GBSM"],
    :Type => ["Call", "Put"],
    :Method => ["Closed Form","Closed Form"],
    :Delta => [bsm_call.delta, bsm_put.delta],
    :Gamma => [bsm_call.gamma, bsm_put.gamma],
    :Vega => [bsm_call.vega, bsm_put.vega],
    :Theta => [bsm_call.theta, bsm_put.theta],
    :Rho => [missing, missing],
    :CarryRho => [bsm_call.cRho, bsm_put.cRho]
)

#Differential Library call the calculate the gradient
_x = [s,x,ttm,rf,rf-b,.2]
f(_x) = gbsm(true,_x...).value
call_grad = ForwardDiff.gradient(f,_x)

f(_x) = gbsm(false,_x...).value
put_grad = ForwardDiff.gradient(f,_x)

#Derivative of Delta = Gamma
f(_x) = gbsm(true,_x...;includeGreeks=true).delta
call_gamma = ForwardDiff.gradient(f,_x)[1]
f(_x) = gbsm(false,_x...;includeGreeks=true).delta
put_gamma = ForwardDiff.gradient(f,_x)[1]

outTable = vcat(outTable,
    DataFrame(
        :Valuation => ["GBSM","GBSM"],
        :Type => ["Call", "Put"],
        :Method => ["Numeric","Numeric"],
        :Delta => [call_grad[1], put_grad[1]],
        :Gamma => [call_gamma, put_gamma],
        :Vega => [call_grad[6], put_grad[6]],
        :Theta => [-call_grad[3], -put_grad[3]],
        :Rho => [call_grad[4], put_grad[4]],
        :CarryRho => [call_grad[5], put_grad[5]]
    )
)

# bt_american(call::Bool, underlying,strike,ttm,rf,divAmts::Vector{Float64},divTimes::Vector{Int64},ivol,N)
divDate = Date(2022,04,11)
divDays = (divDate - Date(2022,3,13)).value
ttmDays = ttm*365
NPoints = convert(Int64,ttmDays*3)
divPoint = divDays*3
divAmt = 0.88

#Values
am_call = bt_american(true, s,x,ttm,rf,[divAmt],[divPoint],.2,NPoints)
am_put = bt_american(false, s,x,ttm,rf,[divAmt],[divPoint],.2,NPoints)

_x = [s,x,ttm,rf,.2]
function f(_x)
    _in = collect(_x)
    bt_american(true, _in[1],_in[2],_in[3],_in[4],[divAmt],[divPoint],_in[5],NPoints)
end
call_grad = FiniteDiff.finite_difference_gradient(f,_x)
δ = 1 #Need to play with the offset value to get a good derivative.  EXTRA 0.5 point if they do this
call_gamma = (bt_american(true, s+δ,x,ttm,rf,[divAmt],[divPoint],.2,NPoints)+bt_american(true, s-δ,x,ttm,rf,[divAmt],[divPoint],.2,NPoints)-2*am_call)/(δ^2)
δ = 1e-6
call_div = (bt_american(true, s,x,ttm,rf,[divAmt+δ],[divPoint],.2,NPoints)-am_call)/(δ)


function f(_x)
    _in = collect(_x)
    bt_american(false, _in[1],_in[2],_in[3],_in[4],[divAmt],[divPoint],_in[5],NPoints)
end
put_grad = FiniteDiff.finite_difference_gradient(f,_x)
δ = 11
put_gamma = (bt_american(false, s+δ,x,ttm,rf,[divAmt],[divPoint],.2,NPoints)+bt_american(false, s-δ,x,ttm,rf,[divAmt],[divPoint],.2,NPoints)-2*am_call)/(δ^2)
δ = 1e-6
put_div = (bt_american(false, s,x,ttm,rf,[divAmt+δ],[divPoint],.2,NPoints)-am_put)/(δ)

outTable = vcat(outTable,
    DataFrame(
        :Valuation => ["BT","BT"],
        :Type => ["Call", "Put"],
        :Method => ["Numeric","Numeric"],
        :Delta => [call_grad[1], put_grad[1]],
        :Gamma => [call_gamma, put_gamma],
        :Vega => [call_grad[5], put_grad[5]],
        :Theta => [-call_grad[3], -put_grad[3]],
        :Rho => [call_grad[4], put_grad[4]],
        :CarryRho => [missing, missing]
    )
)

sort!(outTable,[:Type, :Valuation, :Method])
println(outTable)
println("Call Derivative wrt Dividend: $call_div")
println("Put  Derivative wrt Dividend: $put_div")

# 6×9 DataFrame
#  Row │ Valuation  Type    Method       Delta       Gamma      Vega     Theta      Rho              CarryRho      
#      │ String     String  String       Float64     Float64    Float64  Float64    Float64?         Float64?      
# ─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────
#    1 │ BT         Call    Numeric       0.0749314  0.0186027  6.32562  -7.43533         0.933496   missing       
#    2 │ GBSM       Call    Closed Form   0.0829713  0.0168229  6.93871  -8.12652   missing                1.13295
#    3 │ GBSM       Call    Numeric       0.0829713  0.0168229  6.93871  -8.12652        -0.0303599        1.13295
#    4 │ BT         Put     Numeric      -0.936203   0.301074   5.6463   -0.391418      -12.4527     missing       
#    5 │ GBSM       Put     Closed Form  -0.91655    0.0168229  6.93871  -1.94099   missing              -12.5153
#    6 │ GBSM       Put     Numeric      -0.91655    0.0168229  6.93871  -1.94099        -1.24273        -12.5153
# Call Derivative wrt Dividend: -0.02162170542607811
# Put  Derivative wrt Dividend: 0.9393956741376996



#Problem #2
portfolio = CSV.read("../Week07/Project/problem2.csv",DataFrame)
currentDate = Date(2023,3,3)
divDate = Date(2023,3,15)
divAmt =1.00
currentS=151.03
rf=0.0425
mult = 5
daysDiv = (divDate - currentDate).value

# portfolio[portfolio.Type .== "Stock",:CurrentPrice] .= currentS


prices = CSV.read("../Week07/Project/DailyPrices.csv",DataFrame)[!,[:Date, :AAPL]]
returns = return_calculate(prices,dateColumn="Date")[!,:AAPL]
returns = returns .- mean(returns)
sd = std(returns)

portfolio[!,:ExpirationDate] = [
    portfolio.Type[i] == "Option" ? Date(portfolio.ExpirationDate[i],dateformat"mm/dd/yyyy") : missing
    for i in 1:size(portfolio,1) ]

#Implied Vols
# bt_american(call::Bool, underlying,strike,ttm,rf,divAmts::Vector{Float64},divTimes::Vector{Int64},ivol,N)

portfolio[!,:ImpVol] .= Vector{Float64}(undef,size(portfolio,1))


for i in 1:size(portfolio,1)
    println(i)
    portfolio.ImpVol[i] = 
        portfolio.Type[i] == "Option" ?
                find_zero(x->bt_american(portfolio.OptionType[i]=="Call",
                        currentS,
                        portfolio.Strike[i],
                        (portfolio.ExpirationDate[i]-currentDate).value/365,
                        rf,
                        [divAmt],[daysDiv*mult],x,convert(Int64,(portfolio.ExpirationDate[i]-currentDate).value*mult))
                        -portfolio.CurrentPrice[i],.2) : 0.0     
end

#Delta function for BT American
function bt_delta(call::Bool, underlying,strike,ttm,rf,divAmts::Vector{Float64},divTimes::Vector{Int64},ivol,N)

    f(_x) = bt_american(call::Bool, _x,strike,ttm,rf,divAmts::Vector{Float64},divTimes::Vector{Int64},ivol,N)
    FiniteDiff.finite_difference_derivative(f, underlying)
end

#Position Level Deltas needed for DN VaR
portfolio[!, :Delta] = [
    portfolio.Type[i] == "Option" ?  (
            bt_delta(portfolio.OptionType[i]=="Call",
                currentS, 
                portfolio.Strike[i], 
                (portfolio.ExpirationDate[i]-currentDate).value/365, 
                rf, 
                [divAmt],[daysDiv*mult],
                portfolio.ImpVol[i],convert(Int64,(portfolio.ExpirationDate[i]-currentDate).value*mult))*portfolio.Holding[i]*currentS    
    ) : portfolio.Holding[i] * currentS    
    for i in 1:size(portfolio,1)
]

#Simulate Returns
nSim = 10000
fwdT = 10
_simReturns = rand(Normal(0,sd),nSim*fwdT)

#collect 10 day returns
simPrices = Vector{Float64}(undef,nSim)
for i in 1:nSim
    r = 1.0
    for j in 1:fwdT
        r *= (1+_simReturns[fwdT*(i-1)+j])
    end
    simPrices[i] = currentS*r
end

iteration = [i for i in 1:nSim]
values = crossjoin(portfolio, DataFrame(:iteration=>iteration))
nVals = size(values,1)

#Precalculate the fwd TTM
values[!,:fwd_ttm] = [
    values.Type[i] == "Option" ? (values.ExpirationDate[i]-currentDate-Day(fwdT)).value/365 : missing
    for i in 1:nVals
]

#Valuation
simulatedValue = Vector{Float64}(undef,nVals)
currentValue = Vector{Float64}(undef,nVals)
pnl = Vector{Float64}(undef,nVals)
Threads.@threads for i in 1:nVals
    simprice = simPrices[values.iteration[i]]
    currentValue[i] = values.Holding[i]*values.CurrentPrice[i]
    if values.Type[i] == "Option"
        simulatedValue[i] = values.Holding[i]*bt_american(values.OptionType[i]=="Call",
                                                simprice,
                                                values.Strike[i],
                                                values.fwd_ttm[i],
                                                rf,
                                                [divAmt],[(daysDiv-fwdT)*mult],
                                                values.ImpVol[i],
                                                convert(Int64,values.fwd_ttm[i]*mult*365)
                                            )
    elseif values.Type[i] == "Stock"
        simulatedValue[i] = values.Holding[i]*simprice
    end
    pnl[i] = simulatedValue[i] - currentValue[i]
end

values[!,:simulatedValue] = simulatedValue
values[!,:pnl] = pnl
values[!,:currentValue] = currentValue



#Calculate Simulated Risk Values
risk = aggRisk(values,[:Portfolio])

#Calculate the Portfolio Deltas
gdf = groupby(portfolio, [:Portfolio])
portfolioDelta = combine(gdf,
    :Delta => sum => :PortfolioDelta    
)

#Delta Normal VaR is just the Portfolio Delta * quantile * current Underlying Price
portfolioDelta[!,:DN_VaR] = abs.(quantile(Normal(0,sd),.05)*sqrt(10)*portfolioDelta.PortfolioDelta)
portfolioDelta[!,:DN_ES] = abs.((sqrt(10)*sd*pdf(Normal(0,1),quantile(Normal(0,1),.05))/.05)*portfolioDelta.PortfolioDelta)

leftjoin!(risk,portfolioDelta[!,[:Portfolio, :PortfolioDelta, :DN_VaR, :DN_ES]],on=:Portfolio)

println(risk[risk.Portfolio .!= "Total",[:Portfolio,:PortfolioDelta,:VaR95,:ES95,:DN_VaR,:DN_ES]])

# 9×6 DataFrame
#  Row │ Portfolio     PortfolioDelta  VaR95     ES95      DN_VaR    DN_ES    
#      │ String15      Float64?        Float64   Float64   Float64?  Float64?
# ─────┼──────────────────────────────────────────────────────────────────────
#    1 │ Straddle             11.8764   1.31492   1.32538   0.87451   1.09667
#    2 │ SynLong             152.055   11.495    14.3285   11.1964   14.0408
#    3 │ CallSpread           42.3013   3.00678   3.42106   3.11482   3.90611
#    4 │ PutSpread           -37.9892   2.25321   2.49448   2.7973    3.50793
#    5 │ Stock               151.03    10.7519   13.3845   11.121    13.9461
#    6 │ Call                 81.9656   4.96777   5.47916   6.03546   7.56871
#    7 │ Put                 -70.0892   3.8366    4.16918   5.16095   6.47204
#    8 │ CoveredCall          91.9165   7.42725   9.81586   6.76819   8.48758
#    9 │ ProtectedPut        103.337    6.05453   6.7393    7.60915   9.54218

###Problem 3 ###
#Read All Data
ff3 = CSV.read("../Week07/Project/F-F_Research_Data_Factors_daily.CSV", DataFrame)
mom = CSV.read("../Week07/Project/F-F_Momentum_Factor_daily.CSV",DataFrame)
prices = CSV.read("../Week07/Project/DailyPrices.csv",DataFrame)
returns = return_calculate(prices,dateColumn="Date")
rf = 0.05

# Join the FF3 data with the Momentum Data
ffData = innerjoin(ff3,mom,on=:Date)
rename!(ffData, names(ffData)[size(ffData,2)] => :Mom)
ffData[!,names(ffData)[2:size(ffData,2)]] = Matrix(ffData[!,names(ffData)[2:size(ffData,2)]]) ./ 100
ffData[!,:Date] = Date.(string.(ffData.Date),dateformat"yyyymmdd")

returns[!,:Date] = [d[1:10] for d in returns.Date]
returns[!,:Date] = Date.(returns.Date,dateformat"yyyy-mm-dd")

# Our 20 stocks
stocks = [:AAPL, :META, :UNH, :MA, :MSFT, :NVDA, :HD, :PFE, :AMZN, Symbol("BRK-B"), :PG, :XOM, :TSLA, :JPM, :V, :DIS, :GOOGL, :JNJ, :BAC, :CSCO]

# Data set of all stock returns and FF3+1 returns
to_reg = innerjoin(returns[!,vcat(:Date,stocks)], ffData, on=:Date)

# println("Max RF value is: $(max(to_reg.RF...))")

xnames = [Symbol("Mkt-RF"), :SMB, :HML, :Mom]

#OLS Regression for all Stocks
X = hcat(fill(1.0,size(to_reg,1)),Matrix(to_reg[!,xnames]))
Y = Matrix(to_reg[!,stocks] .- to_reg.RF)

Betas = (inv(X'*X)*X'*Y)'

#Calculate the means of the last 10 years of factor returns
#adding the 0.0 at the front to 0 out the fitted alpha in the next step
means = vcat(0.0,mean.(eachcol(ffData[ffData.Date .>= Date(2014,9,30),xnames])))

#Discrete Returns, convert to Log Returns and scale to 1 year
stockMeans =log.(1 .+ Betas*means)*255 .+ rf
covar = cov(log.(1.0 .+ Matrix(returns[!,stocks])))*255

#optimize.  Directly find the max SR portfolio.  Can also do this like in the notes and
#   build the Efficient Frontier

function sr(w...)
    _w = collect(w)
    m = _w'*stockMeans - rf
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

w = round.(value.(w),digits=4)

icovar = inv(covar)
w2 = icovar*(stockMeans .- rf)/sum(icovar*(stockMeans .- rf))

OptWeights = DataFrame(:Stock=>String.(stocks), :Weight => w, :Er => stockMeans, :UnconstWeight => w2)
println(OptWeights)
println("Expected Retrun = $(stockMeans'*w)")
println("Expected Vol = $(sqrt(w'*covar*w))")
println("Expected SR = $(sr(w...)) ")

# 20×4 DataFrame
#  Row │ Stock   Weight   Er         UnconstWeight 
#      │ String  Float64  Float64    Float64
# ─────┼───────────────────────────────────────────
#    1 │ AAPL     0.0741  0.179502      0.0743396
#    2 │ META     0.041   0.262582      0.040816
#    3 │ UNH      0.0008  0.0566713     0.00133859
#    4 │ MA       0.0732  0.148799      0.0735515
#    5 │ MSFT     0.1098  0.197232      0.110709
#    6 │ NVDA     0.0805  0.364049      0.0808259
#    7 │ HD      -0.0     0.152052     -0.00523634
#    8 │ PFE     -0.0     0.0833909    -0.00219768
#    9 │ AMZN     0.0317  0.245892      0.032024
#   10 │ BRK-B    0.05    0.120461      0.0515039
#   11 │ PG       0.1479  0.0842814     0.147766
#   12 │ XOM      0.0497  0.0757517     0.0494838
#   13 │ TSLA     0.032   0.326127      0.0320759
#   14 │ JPM      0.0358  0.130938      0.0355969
#   15 │ V        0.0278  0.142891      0.0277223
#   16 │ DIS      0.0122  0.129287      0.0126014
#   17 │ GOOGL    0.0482  0.21926       0.0476962
#   18 │ JNJ      0.0565  0.0721752     0.0584292
#   19 │ BAC      0.093   0.156485      0.0944788
#   20 │ CSCO     0.0356  0.13527       0.0364749
# Expected Retrun = 0.1707024635430251
# Expected Vol = 0.12036204020491463
# Expected SR = 1.0028283280802726