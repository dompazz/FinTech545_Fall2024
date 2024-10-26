using DataFrames
using Plots
using Distributions
using CSV
using Dates
using LoopVectorization
using LinearAlgebra
using StatsBase
using JuMP
using Ipopt
using QuadGK
using Random
using StatsPlots

include("../../library/return_calculate.jl")
include("../../library/fitted_model.jl")
include("../../library/simulate.jl")
include("../../library/RiskStats.jl")
include("../../library/ewCov.jl")


# Problem 2
problem1 = CSV.read("Project/problem1.csv",DataFrame).x
# m = mean(problem1)
# problem1 .-= m 
alpha = 0.05

s = sqrt(ewCovar([problem1 problem1],0.97)[1,1])
m2 = expW(length(problem1), 0.97)
n = Normal(m2'*problem1,s)
var_n = VaR(n;alpha=alpha)
es_n = ES(n;alpha=alpha)

tm = fit_general_t(problem1)
var_t = VaR(tm.errorModel;alpha=alpha)
es_t = ES(tm.errorModel;alpha=alpha)

var_h = VaR(problem1;alpha=alpha)
es_h = ES(problem1;alpha=alpha)

density(problem1,label="Historical",color=:black)
x = [i for i in extrema(problem1)[1]:.001:extrema(problem1)[2]]
plot!(x,pdf.(n,x),label="Normal", color=:red)
plot!(x,pdf.(tm.errorModel,x),label="T Distribution",color=:blue)

vline!([-var_n],color=:red,style=:dash,label="")
vline!([-var_t],color=:blue,style=:dash,label="")
vline!([-var_h],color=:black,style=:dash,label="")

vline!([-es_n],color=:red,style=:dot,label="")
vline!([-es_t],color=:blue,style=:dot,label="")
vline!([-es_h],color=:black,style=:dot,label="")

toPrint = DataFrame(
    :Model=>["Normal","T","Historical"],
    :VaR=>[var_n,var_t,var_h],
    :ES=>[es_n,es_t,es_h]
)
println(toPrint)
println(kurtosis(problem1))
# 3×3 DataFrame
#  Row │ Model       VaR        ES       
#      │ String      Float64    Float64
# ─────┼─────────────────────────────────
#    1 │ Normal      0.0911099  0.114047
#    2 │ T           0.0764758  0.113218
#    3 │ Historical  0.0782451  0.116777

# 2.414872528860026

#Data is non-normal.  The T distribution fits well base on the graph.
# Normal VaR is larger than the T VaR as expected given the 
# excess kurtosis.  ES values are similar, likely an artifact
# of the data.

#problem 3
prices = CSV.read("Project/DailyPrices.csv",DataFrame)
returns = return_calculate(prices,dateColumn="Date")
returns = select!(returns,Not([:Date]))
rnames = names(returns)

currentPrice = prices[size(prices,1),:]

portfolio = CSV.read("Project/portfolio.csv",DataFrame)

stocks = portfolio.Stock

tStocks = filter(r->r.Portfolio in ["A","B"],portfolio)[!,:Stock]
nStocks = filter(r->r.Portfolio in ["C"],portfolio)[!,:Stock]

#remove the mean from all returns:
for nm in stocks
    v = returns[!,nm]
    returns[!,nm] = v .- mean(v)
end

fittedModels = Dict{String,FittedModel}()

for s in tStocks
    fittedModels[s] = fit_general_t(returns[!,s])
end
for s in nStocks
    fittedModels[s] = fit_normal(returns[!,s])
end

U = DataFrame()
for nm in stocks
    U[!,nm] = fittedModels[nm].u
end
R = corspearman(Matrix(U))

#what's the rank of R
evals = eigvals(R)
if min(evals...) > -1e-8
    println("Matrix is PSD")
else
    println("Matrix is not PSD")
end

#simulation
NSim = 50000
simU = DataFrame(
            #convert standard normals to U
            cdf(Normal(),
                simulate_pca(R,NSim)  #simulation the standard normals
            )   
            , stocks
        )

simulatedReturns = DataFrame()
Threads.@threads for stock in stocks
    simulatedReturns[!,stock] = fittedModels[stock].eval(simU[!,stock])
end

#Protfolio Valuation
function calcPortfolioRisk(simulatedReturns,NSim)
    iteration = [i for i in 1:NSim]
    values = crossjoin(portfolio, DataFrame(:iteration=>iteration))

    nVals = size(values,1)
    currentValue = Vector{Float64}(undef,nVals)
    simulatedValue = Vector{Float64}(undef,nVals)
    pnl = Vector{Float64}(undef,nVals)
    Threads.@threads for i in 1:nVals
        price = currentPrice[values.Stock[i]]
        currentValue[i] = values.Holding[i] * price
        simulatedValue[i] = values.Holding[i] * price*(1.0+simulatedReturns[values.iteration[i],values.Stock[i]])
        pnl[i] = simulatedValue[i] - currentValue[i]
    end
    values[!,:currentValue] = currentValue
    values[!,:simulatedValue] = simulatedValue
    values[!,:pnl] = pnl

    values[!,:Portfolio] = String.(values.Portfolio)
    aggRisk(values,[:Portfolio])[:,[:Portfolio,:VaR95, :ES95]]
end
risk = calcPortfolioRisk(simulatedReturns,NSim)
# 4×3 DataFrame
#  Row │ Portfolio  VaR95     ES95     
#      │ String     Float64   Float64
# ─────┼───────────────────────────────
#    1 │ A           5297.24   7170.22
#    2 │ B           4459.26   6044.56
#    3 │ C           3867.17   4827.55
#    4 │ Total      13079.8   17083.6

covar = ewCovar(Matrix(returns),.97)
simulatedReturns = DataFrame(simulate_pca(covar,NSim),rnames)
risk_n  = calcPortfolioRisk(simulatedReturns,NSim)
rename!(risk_n,[:VaR95=>:Normal_VaR, :ES95=>:Normal_ES])

leftjoin!(risk,risk_n,on=:Portfolio)
# 4×5 DataFrame
#  Row │ Portfolio  VaR95     ES95      Normal_VaR  Normal_ES 
#      │ String     Float64   Float64   Float64?    Float64?
# ─────┼──────────────────────────────────────────────────────
#    1 │ A           5297.24   7170.22     5903.62    7384.53
#    2 │ B           4459.26   6044.56     4947.11    6161.29
#    3 │ C           3867.17   4827.55     3920.32    4924.44
#    4 │ Total      13079.8   17083.6     14132.5    17621.8

# Compared to the same method as last week, the VaR and ES of the 
# portfolio signiantly decreases.  Curiously, the normal portfolio C
# is larger.  This could be the lambda or the 
# spearman correlation.  Run the portfolio with an unweighted covar

covar = cov(Matrix(returns))
simulatedReturns = DataFrame(simulate_pca(covar,NSim*2),rnames)
risk_n2  = calcPortfolioRisk(simulatedReturns,NSim*2)
rename!(risk_n2,[:VaR95=>:Normal2_VaR, :ES95=>:Normal2_ES])
leftjoin!(risk,risk_n2,on=:Portfolio)

# 4×7 DataFrame
#  Row │ Portfolio  VaR95     ES95      Normal_VaR  Normal_ES  Normal2_VaR  Normal2_ES 
#      │ String     Float64   Float64   Float64?    Float64?   Float64?     Float64?
# ─────┼───────────────────────────────────────────────────────────────────────────────
#    1 │ A           5297.24   7170.22     5903.62    7384.53      5180.2      6511.93
#    2 │ B           4459.26   6044.56     4947.11    6161.29      4324.3      5423.52
#    3 │ C           3867.17   4827.55     3920.32    4924.44      3555.01     4439.85
#    4 │ Total      13079.8   17083.6     14132.5    17621.8      12361.8     15501.6
# The VaR using the unweighted covariance is roughly the same in portfolio C as expected
# Portfolio VaR is lower than the weighted covariance.  This is likely due to the more 
# volitility in more recent data.  
