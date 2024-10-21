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
using Random

# include("../Week04/return_calculate.jl")
include("fitted_model.jl")
include("simulate.jl")
include("RiskStats.jl")

prices = CSV.read("DailyPrices.csv",DataFrame)
#current Prices
current_prices = prices[size(prices,1),:]

#discrete returns
returns = return_calculate(prices,dateColumn="Date")

nms = names(returns)
nms = nms[nms.!="Date"]
nms = nms[nms.!="PLD"]
#remove date column
returns = returns[!,nms]

#all stock names
stocks = nms[nms.!="SPY"]

#setup how much we hold
Portfolio = DataFrame(:stock=>stocks, :holding => fill(1.0,size(stocks,1)))


#remove the mean from all returns:
for nm in nms
    v = returns[!,nm]
    returns[!,nm] = v .- mean(v)
end

st = time()
#fit model for all stocks
fittedModels = Dict{String,FittedModel}()

fittedModels["SPY"] = fit_normal(returns.SPY)

for stock in stocks
    fittedModels[stock] = fit_regression_t(returns[!,stock],returns.SPY)
end
println("Model Fitting Took $(time()-st)")

st = time()

#construct the copula:
#Start the data frame with the U of the SPY - we are assuming normallity for SPY
U = DataFrame()
for nm in nms
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
NSim = 5000
simU = DataFrame(
            #convert standard normals to U
            cdf(Normal(),
                simulate_pca(R,NSim)  #simulation the standard normals
            )   
            , nms
        )

simulatedReturns = DataFrame(:SPY => fittedModels["SPY"].eval(simU.SPY))
for stock in stocks
    simulatedReturns[!,stock] = fittedModels[stock].eval(simulatedReturns.SPY,simU[!,stock])
end

println("Simulation Took $(time()-st)")
st = time()

#Protfolio Valuation
iteration = [i for i in 1:NSim]
values = crossjoin(Portfolio, DataFrame(:iteration=>iteration))

nVals = size(values,1)
currentValue = Vector{Float64}(undef,nVals)
simulatedValue = Vector{Float64}(undef,nVals)
pnl = Vector{Float64}(undef,nVals)
for i in 1:nVals
    price = current_prices[values.stock[i]]
    currentValue[i] = values.holding[i] * price
    simulatedValue[i] = values.holding[i] * price*(1.0+simulatedReturns[values.iteration[i],values.stock[i]])
    pnl[i] = simulatedValue[i] - currentValue[i]
end
values[!,:currentValue] = currentValue
values[!,:simulatedValue] = simulatedValue
values[!,:pnl] = pnl

println("Valuation Took $(time()-st)")
st = time()

#Calculation of Risk Metrics
#Stock Level Metrics
gdf = groupby(values,:stock)

stockRisk = combine(gdf, 
    :currentValue => (x-> first(x,1)) => :currentValue,
    :pnl => (x -> VaR(x,alpha=0.05)) => :VaR95,
    :pnl => (x -> ES(x,alpha=0.05)) => :ES95,
    :pnl => (x -> VaR(x,alpha=0.01)) => :VaR99,
    :pnl => (x -> ES(x,alpha=0.01)) => :ES99,
    :pnl => std => :Standard_Dev,
    :pnl => (x -> [extrema(x)]) => [:min, :max],
    :pnl => mean => :mean
)

#Total Metrics
gdf = groupby(values,:iteration)
#aggregate to totals per simulation iteration
totalValues = combine(gdf,
    :currentValue => sum => :currentValue,
    :simulatedValue => sum => :simulatedValue,
    :pnl => sum => :pnl
)

#calculate Risk
totalRisk = combine(totalValues,
    :currentValue => (x-> first(x,1)) => :currentValue,
    :pnl => (x -> VaR(x,alpha=0.05)) => :VaR95,
    :pnl => (x -> ES(x,alpha=0.05)) => :ES95,
    :pnl => (x -> VaR(x,alpha=0.01)) => :VaR99,
    :pnl => (x -> ES(x,alpha=0.01)) => :ES99,
    :pnl => std => :Standard_Dev,
    :pnl => (x -> [extrema(x)]) => [:min, :max],
    :pnl => mean => :mean
)

totalRisk[!,:stock] = ["Total"]

#Final Output
riskOut = vcat(stockRisk, totalRisk)

println("Aggregation Took $(time()-st)")

CSV.write("ExampleRisk.csv",riskOut)
