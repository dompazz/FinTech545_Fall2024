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

include("../library/gbsm.jl")


function bt_american(call::Bool, underlying,strike,ttm,rf,b,ivol,N)

    dt = ttm/N
    u = exp(ivol*sqrt(dt))
    d = 1/u
    pu = (exp(b*dt)-d)/(u-d)
    pd = 1.0-pu
    df = exp(-rf*dt)
    z = call ? 1 : -1

    nNodeFunc(n) = convert(Int64,(n+1)*(n+2)/2 )
    idxFunc(i,j) = nNodeFunc(j-1)+i+1
    nNodes = nNodeFunc(N)

    optionValues = Vector{Float64}(undef,nNodes)

    for j in N:-1:0
        for i in j:-1:0
            idx = idxFunc(i,j)
            price = underlying*u^i*d^(j-i)
            optionValues[idx] = max(0,z*(price-strike))
            
            if j < N
                optionValues[idx] = max(optionValues[idx], df*(pu*optionValues[idxFunc(i+1,j+1)] + pd*optionValues[idxFunc(i,j+1)])  )
            end
        end
    end

    return optionValues[1]
end

bt_american(false, 100,100,.5,.08,.08,.3,2)

bt_american(false, 100,100,.5,.08,.08,.3,100)
gbsm(false, 100,100,.5,.08,.08,.3).value



# divAmts and divTimes are vectors
# divTimes is the time of the dividends in relation to the grid of j ∈ 0:N 
function bt_american(call::Bool, underlying,strike,ttm,rf,divAmts::Vector{Float64},divTimes::Vector{Int64},ivol,N)

    # println("Call:  divAmts:$divAmts ")
    # println("      divTimes:$divTimes ")
    # println("             N:$N ")
    # println("           ttm:$ttm ")
    # println("    underlying:$underlying ")

    #if there are no dividends or the first dividend is outside out grid, return the standard bt_american value
    if isempty(divAmts) || isempty(divTimes)
        return bt_american(call, underlying,strike,ttm,rf,rf,ivol,N)
    elseif divTimes[1] > N
        return bt_american(call, underlying,strike,ttm,rf,rf,ivol,N)
    end

    dt = ttm/N
    u = exp(ivol*sqrt(dt))
    d = 1/u
    pu = (exp(rf*dt)-d)/(u-d)
    pd = 1.0-pu
    df = exp(-rf*dt)
    z = call ? 1 : -1

    nNodeFunc(n) = convert(Int64,(n+1)*(n+2)/2 )
    idxFunc(i,j) = nNodeFunc(j-1)+i+1
    nDiv = size(divTimes,1)
    nNodes = nNodeFunc(divTimes[1])

    optionValues = Vector{Float64}(undef,nNodes)

    for j in divTimes[1]:-1:0
        for i in j:-1:0
            idx = idxFunc(i,j)
            price = underlying*u^i*d^(j-i)        
            
            if j < divTimes[1]
                #times before the dividend working backward induction
                optionValues[idx] = max(0,z*(price-strike))
                optionValues[idx] = max(optionValues[idx], df*(pu*optionValues[idxFunc(i+1,j+1)] + pd*optionValues[idxFunc(i,j+1)])  )
            else
                #time of the dividend
               valNoExercise = bt_american(call, price-divAmts[1], strike, ttm-divTimes[1]*dt, rf, divAmts[2:nDiv], divTimes[2:nDiv] .- divTimes[1], ivol, N-divTimes[1])
               valExercise =  max(0,z*(price-strike))
               optionValues[idx] = max(valNoExercise,valExercise)
            end
        end
    end

    return optionValues[1]
end

bt_american(true, 100,100,.5,.08,.04,.3,2)
bt_american(true, 100,100,.5,.08,[1.0],[1],.3,2)

bt_american(true, 100,100,.5,.08,.04,.3,100)
bt_american(true, 100,100,.5,.08,[1.0],[50],.3,100)

bt_american(true, 100,100,.5,.08,[1.0],[1],.3,100)
bt_american(true, 100,100,.5,.08,[1.0,1.0],[1,2],.3,100)

@btime bt_american(true, 100,100,.5,.08,.08,.3,100)
@btime bt_american(true, 100,100,.5,.08,[1.0],[50],.3,100)


#Hedging Example
S = 100.50
X = 100
ttm = 15/255
rf = 0.0025
p = 2.5

f(iv) = gbsm(true,S,X,ttm,rf,rf,iv).value - p
implied_vol = find_zero(f,0.2)

nSim = 5000
pnl = Vector{Float64}(undef,nSim)
include("../library/RiskStats.jl")

#1 Day change, need to update the TTM
ttm_new = 14/255
#Simulate Normal based on the implied volatility scaled to daily.
r = rand(Normal(0,implied_vol/sqrt(255)),nSim)  

#Profit and loss
#100 options short
# Each long option PNL is new option value minus the starting value
pnl = -100*([r.value for r in gbsm.(true,S*(1 .+r),X,ttm_new,rf,rf,implied_vol)] .- 2.5)

    #Calculate the VaR and ES
    var_unhedged = VaR(pnl)
    es_unhedged = ES(pnl)

    println("VaR UnHedged: $var_unhedged")
    println("ES UnHedged : $es_unhedged")


#MM is short 100 calls.  He wants to buy the negative of his delta to hedge
delta = -(-100)* (exp((rf-rf)*ttm)*cdf(Normal(),(log(S/X) + (rf+implied_vol^2/2)*ttm)/(implied_vol*sqrt(ttm))))
delta = round(delta)

#Update the PNL, add the return on the stock times the amount we hold (delta)
pnl2 = pnl .+ (delta * S * r)

#Recalculate the VaR and ES
var_hedged = VaR(pnl2)
es_hedged = ES(pnl2)

println("VaR Hedged: $var_hedged")
println("ES Hedged : $es_hedged")


#Example Efficient Frontier
corr = [1 .5 0
        .5 1 .5
        0 .5 1]
sd = [.2, .1, .05]
er = [.05, .04, .03]

covar = diagm(sd)*corr*diagm(sd)

function optimize_risk(R)
    # R = 0.05

    m = Model(Ipopt.Optimizer)
    set_silent(m)
    # Weights with boundry at 0
    @variable(m, w[i=1:3] >= 0,start=1/3)

    @objective(m,Min, w'*covar*w)
    @constraint(m, sum(w)==1.0)
    @constraint(m, sum(er[i]*w[i] for i in 1:3) == R)
    optimize!(m)
    #return the objective(risk) as well as the portfolio weights
    return Dict(:risk=>objective_value(m), :weights=>value.(w), :R=>R)
end

returns = [i for i in 0.03:.001:.05]
optim_portfolios = DataFrame(optimize_risk.(returns))
plot(sqrt.(optim_portfolios.risk), optim_portfolios.R, legend=:bottomright, label="Efficient Frontier", xlabel="Risk - SD", ylabel="Portfolio Expected Return")

# w = [i for i in 0:.1:1.5]
# returns = .1*w .+ .05*(1 .-w)
# risks = .16*w
# plot(risks, returns, legend=:bottomright, label="", title="Investment A + Rf", xlabel="Risk - SD", ylabel="Portfolio Expected Return")
# scatter!((.16,.1), label="Investment A")

#Sharpe Ratios
optim_portfolios[!,:SR] = (optim_portfolios.R .- 0.03)./sqrt.(optim_portfolios.risk)
maxSR = argmax(optim_portfolios.SR)
maxSR_ret=optim_portfolios.R[maxSR]
maxSR_risk=sqrt(optim_portfolios.risk[maxSR])

println("Portfolio Weights at the Maximum Sharpe Ratio: $(optim_portfolios.weights[maxSR])")
println("Portfolio Return : $maxSR_ret")
println("Portfolio Risk   : $maxSR_risk")
println("Portfolio SR     : $(optim_portfolios.SR[maxSR])")






w = [i for i in 0:.1:2]
returns = maxSR_ret*w .+ .03*(1 .-w)
risks = maxSR_risk*w
plot!(risks,returns,label="",color=:red)
scatter!((maxSR_risk,maxSR_ret),label="Max SR Portfolio")