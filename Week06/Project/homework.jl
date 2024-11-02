using CSV
using DataFrames
using Dates
using Roots
using Distributions
using Plots
using Random
using StateSpaceModels
using Distributions
using StatsBase
using LinearAlgebra
using LoopVectorization

include("../../library/gbsm.jl")
include("../../library/RiskStats.jl")
include("../../library/return_calculate.jl")

#Problem #1
currentPrice = 165
currentDate=Date("03/03/2023",dateformat"mm/dd/yyyy")
rf = 0.0525
dy = 0.0053
DaysYear = 365

expirationDate = Date("03/17/2023",dateformat"mm/dd/yyyy")
ttm = (expirationDate - currentDate).value/DaysYear

strike = 165
iv = [i/100 for i in 10:2:80]
#gbsm(call::Bool, underlying, strike, ttm, rf, b, ivol)
call_vals = [v.value for v in gbsm.(true,currentPrice,strike,ttm,rf,rf-dy,iv)]
put_vals  = [v.value for v in gbsm.(false,currentPrice,strike,ttm,rf,rf-dy,iv)]

plot(
    plot(iv,call_vals, label="Call Values"),
    plot(iv,put_vals, label="Put Values",linecolor=:red),
    layout=(1,2)
)

# Both put and calls increase with an increase in implied volatility. 
# This is because the option price is a function of the volatility of the underlying asset. 
# Higher volatility increases the probability of the option being in the money, 
# which increases the option price.

# Implied volatility is not directly observable.  As supply/demand for options changes,
# and the price of the option rises or falls, the implied volatility changes.  An increase
# in demand raises the price of the option, and the implied volatility increases.  A decrease
# in demand lowers the price of the option, and the implied volatility decreases.  This
# same logic applies to the supply of options.

# If market makers feel the risk of the underlying asset has increased, they will raise the
# price of the option, and the implied volatility will increase.  Conversely if options buyers
# feel the probability of the option being in the money has increased (decreased), they will
# raise (lower) the price of the option, and the implied volatility will increase (decrease).

#Problem #2
currentDate=Date("10/30/2023",dateformat"mm/dd/yyyy")
rf = 0.0525
dy = 0.0057
currentPrice = 170.15
options = CSV.read("Project/AAPL_Options.csv",DataFrame)

options[!,:Expiration] = Date.(options.Expiration,dateformat"mm/dd/yyyy")


n = length(options.Expiration)

#list comprehension for TTM
options[!,:ttm] = [(options.Expiration[i] - currentDate).value / DaysYear for i in 1:n]

#gbsm(call::Bool, underlying, strike, ttm, rf, b, ivol)
iv = [find_zero(x->gbsm(options.Type[i]=="Call",currentPrice,options.Strike[i],options.ttm[i],rf,rf-dy,x).value-options[i,"Last Price"],.2) for i in 1:n]
options[!,:ivol] = iv
options[!,:gbsm] = [v.value for v in gbsm.(options.Type.=="Call",currentPrice,options.Strike,options.ttm,rf,rf-dy,options.ivol)]


calls = options.Type .== "Call"
puts = [!calls[i] for i in 1:n]

plot(options.Strike[calls],options.ivol[calls],label="Call Implied Vol",title="Implied Volatilities")
plot!(options.Strike[puts],options.ivol[puts],label="Put Implied Vol",linecolor=:red)
vline!([currentPrice],label="Current Price",linestyle=:dash,linecolor=:purple)


#problem 3

currentS=170.15
returns = return_calculate(CSV.read("Project/DailyPrices.csv",DataFrame)[!,[:Date,:AAPL]],method="LOG",dateColumn="Date")[!,:AAPL]
returns = returns .- mean(returns)
sd = std(returns)
current_dt = Date(2023,10,30)

portfolio = CSV.read("Project/problem3.csv", DataFrame)

#Convert Expiration Date for Options to Date object
portfolio[!,:ExpirationDate] = [
    portfolio.Type[i] == "Option" ? Date(portfolio.ExpirationDate[i],dateformat"mm/dd/yyyy") : missing
    for i in 1:size(portfolio,1) ]

# Calculate implied Vol
portfolio[!, :ImpVol] = [
    portfolio.Type[i] == "Option" ?
    find_zero(x->gbsm(portfolio.OptionType[i]=="Call",
                        currentS,
                        portfolio.Strike[i],
                        (portfolio.ExpirationDate[i]-current_dt).value/365,
                        rf,rf-dy,x).value
                -portfolio.CurrentPrice[i],.2)    : missing     
    for i in 1:size(portfolio,1)
]

#Simulate Returns
nSim = 10000
fwdT = 10

#Fit the AR(1) model
ar1 = SARIMA(returns,order=(1,0,0),include_mean=true)
StateSpaceModels.fit!(ar1)
print_results(ar1)

function ar1_simulation(y,coef_table,innovations; ahead=1)
    m = coef_table.coef[findfirst(r->r == "mean",coef_table.names)]
    a1 = coef_table.coef[findfirst(r->r == "ar_L1",coef_table.names)]
    s = sqrt(coef_table.coef[findfirst(r->r == "sigma2_η",coef_table.names)])

    l = length(y)
    n = convert(Int64,length(innovations)/ahead)

    out = fill(0.0,(ahead,n))

    y_last = y[l] - m
    for i in 1:n
        yl = y_last
        next = 0.0
        for j in 1:ahead
            next = a1*yl + s*innovations[(i-1)*ahead + j]
            yl = next
            out[j,i] = next
        end
    end

    out = out .+ m
    return out
end

#simulate nSim paths fwdT days ahead.
arSim = ar1_simulation(returns,ar1.results.coef_table,randn(fwdT*nSim),ahead=fwdT)

# Sum returns since these are log returns and convert to final prices
simReturns = sum.(eachcol(arSim))
simPrices = currentS .* exp.(simReturns)


iteration = [i for i in 1:nSim]
values = crossjoin(portfolio, DataFrame(:iteration=>iteration))
nVals = size(values,1)

#Set the forward ttm
values[!,:fwd_ttm] = [
    values.Type[i] == "Option" ? (values.ExpirationDate[i]-current_dt-Day(fwdT)).value/365 : missing
    for i in 1:nVals
]

#Calculate values of each position
simulatedValue = Vector{Float64}(undef,nVals)
currentValue = Vector{Float64}(undef,nVals)
pnl = Vector{Float64}(undef,nVals)
for i in 1:nVals
    simprice = simPrices[values.iteration[i]]
    currentValue[i] = values.Holding[i]*values.CurrentPrice[i]
    if values.Type[i] == "Option"
        simulatedValue[i] = values.Holding[i]*gbsm(values.OptionType[i]=="Call",simprice,values.Strike[i],values.fwd_ttm[i],rf,rf-dy,values.ImpVol[i]).value
    elseif values.Type[i] == "Stock"
        simulatedValue[i] = values.Holding[i]*simprice
    end
    pnl[i] = simulatedValue[i] - currentValue[i]
end

values[!,:simulatedValue] = simulatedValue
values[!,:pnl] = pnl
values[!,:currentValue] = currentValue


risk = aggRisk(values,[:Portfolio])
# Row │ Portfolio     currentValue  VaR95     ES95      VaR99     ES99      Standard_Dev  min        max        mean       VaR95_Pct   VaR99_Pct   ES95_Pct    ES99_Pct   
# │ String15      Float64       Float64   Float64   Float64   Float64   Float64       Float64    Float64    Float64    Float64     Float64     Float64     Float64    
# ─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# 1 │ Straddle             13.37   1.59287   1.59974   1.60279   1.6031        3.07899   -1.60326   28.6683    0.715321   0.119138    0.11988     0.119652    0.119903
# 2 │ SynLong               1.05  15.024    18.7003   21.2042   23.9804        9.57415  -32.3764    40.9259    0.085685  14.3085     20.1945     17.8098     22.8384
# 3 │ CallSpread            4.54   3.51345   3.88525   4.13763   4.27922       2.23396   -4.50097    5.34574  -0.111517   0.773888    0.911372    0.855781    0.942559
# 4 │ PutSpread             3.17   2.48937   2.73656   2.89494   2.97702       1.91194   -3.14849    6.38197   0.199446   0.785292    0.913229    0.863269    0.939123
# 5 │ Stock               170.15  14.863    18.5549   21.0697   23.8553        9.57807  -32.2719    41.0705    0.277155   0.0873524   0.12383     0.10905     0.140202
# 6 │ Call                  7.21   6.02145   6.47039   6.77088   6.92939       5.51029   -7.16986   34.7971    0.400503   0.835152    0.939096    0.89742     0.96108
# 7 │ Put                   6.16   5.11342   5.50292   5.75038   5.87433       4.49544   -6.12879   25.2066    0.314818   0.8301      0.933503    0.893331    0.953626
# 8 │ CoveredCall         165.52  10.7263   14.2082   16.5864   19.312         5.57598  -27.65       8.64519  -0.202631   0.0648039   0.100208    0.0858399   0.116675
# 9 │ ProtectedPut        174.47   7.77495   8.58728   9.14112   9.48572       6.45865  -10.0885    36.7674    0.492976   0.0445633   0.0523936   0.0492192   0.0543688
# CSV.write("problem3_risk.csv",risk)