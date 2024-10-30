using BenchmarkTools
using Distributions
using Random
using StatsBase
using HypothesisTests
using Roots
using QuadGK
using DataFrames
using Plots

function test(N,r)
    1.0/(1.0+r/N)^N
end

test(10000,.01) ≈ exp(-.01)

#Example European Call Option
underlying = 100
strike = 100
days=125
tradingDayYear=255
ttm = days/tradingDayYear
rf = 0.05
ivol = 0.2

### Option valuation as an integral
### European Style.  Assumed LogNormal Prices
function integral_bsm(call::Bool, underlying,strike,days,rf,ivol,tradingDayYear)
    ttm = days/tradingDayYear

    dailyVol = ivol / sqrt(tradingDayYear)

    σ = sqrt(days)*dailyVol
    μ = log(underlying) + ttm*rf - 0.5*σ^2

    d = LogNormal(μ,σ)

    if call 
        f(x) = max(0,x-strike)*pdf(d,x)
        val = quadgk(f,0,strike*100)[1]
    else
        g(x) = max(0,strike-x)*pdf(d,x)
        val = quadgk(g,0,strike*100)[1]
    end
    
    return val * exp(-rf*ttm)
end

integral_val = integral_bsm(true, underlying,strike,days,rf,ivol,tradingDayYear)




###
#Generalize Black Scholes Merton
# rf = b       -- Black Scholes 1973
# b = rf - q   -- Merton 1973 stock model where q is the continous dividend yield
# b = 0        -- Black 1976 futures option model
# b,r = 0      -- Asay 1982 margined futures option model
# b = rf - rff -- Garman and Kohlhagen 1983 currency option model where rff is the risk free
#                 rate of the foreign currency
###
function gbsm(call::Bool, underlying, strike, ttm, rf, b, ivol)
    d1 = (log(underlying/strike) + (b+ivol^2/2)*ttm)/(ivol*sqrt(ttm))
    d2 = d1 - ivol*sqrt(ttm)

    if call
        return underlying * exp((b-rf)*ttm) * cdf(Normal(),d1) - strike*exp(-rf*ttm)*cdf(Normal(),d2)
    else
        return strike*exp(-rf*ttm)*cdf(Normal(),-d2) - underlying*exp((b-rf)*ttm)*cdf(Normal(),-d1)
    end
    return nothing
end


call_val = gbsm(true,underlying,strike,ttm,rf,rf,ivol)



#updated integral example with a cost of carry b
function integral_bsm2(call::Bool, underlying,strike,days,rf,b,ivol,tradingDayYear)
    ttm = days/tradingDayYear
    
    dailyVol = ivol / sqrt(tradingDayYear)

    σ = sqrt(days)*dailyVol
    μ = log(underlying) + ttm*b - 0.5*σ^2

    d = LogNormal(μ,σ)

    if call 
        f(x) = max(0,x-strike)*pdf(d,x)
        val = quadgk(f,0,strike*100)[1]
    else
        g(x) = max(0,strike-x)*pdf(d,x)
        val = quadgk(g,0,strike*100)[1]
    end
    
    return val * exp(-rf*ttm)
end

gbsm(false,underlying,strike,ttm,rf,rf,ivol) ≈ integral_bsm2(false, underlying,strike,days,rf,rf,ivol,tradingDayYear)

gbsm(false,underlying,strike,ttm,rf,0,ivol) ≈ integral_bsm2(false, underlying,strike,days,rf,0,ivol,tradingDayYear)

gbsm(false,underlying,strike,ttm,rf,rf+.1,ivol) ≈ integral_bsm2(false, underlying,strike,days,rf,rf+.1,ivol,tradingDayYear)


function sim_bsm(underlying,strike,days,rf,b,ivol,tradingDayYear,nSim)
    ttm = days/tradingDayYear
    dailyVol = ivol / sqrt(tradingDayYear)

    vals = Vector{Float64}(undef,nSim)

    d = Normal(b/tradingDayYear - 0.5*dailyVol^2,dailyVol)

    Threads.@threads for sim in 1:nSim
        r = Vector{Float64}(undef,days)
        p = underlying
        rand!(d,r)
        for day in 1:days
            p *= exp(r[day])
        end
        vals[sim] = max(0,p-strike)
    end

    mean(vals)*exp(-rf*ttm)
end

sim_val = sim_bsm(underlying,strike,days,rf,rf,ivol,tradingDayYear,10000000)

ave = Vector{Float64}(undef,100)
for i in 1:100
    ave[i] = sim_bsm(underlying,strike,days,rf,rf,ivol,tradingDayYear,1000)
end

OneSampleTTest(ave,call_val)

#Option values greater than payoff:
values = [i for i in 65:135]
ttm_vals = [0, .01, .5, 1]
output = Dict{Float64,DataFrame}()

replace_nan(v) = map(x -> isnan(x) ? zero(x) : x,v)
for ttm in ttm_vals
    df = DataFrame(:S=>values)
    df[!,:call_vals] = gbsm.(true,values,strike,ttm,rf,rf,ivol)
    df[!,:call_vals] = replace_nan(df.call_vals)
    output[ttm] = df[:,:]
end

plot(values,output[ttm_vals[1]].call_vals,label=ttm_vals[1],legend=:topleft)
plot!(values,output[ttm_vals[2]].call_vals,label=ttm_vals[2])
plot!(values,output[ttm_vals[3]].call_vals,label=ttm_vals[3])
plot!(values,output[ttm_vals[4]].call_vals,label=ttm_vals[4])

#put call parity 
put_val = gbsm(false,underlying,strike,ttm,rf,rf,ivol)
call_val + strike*exp(-rf*ttm) ≈ put_val + underlying


#binomial tree European Option
function bt_bsm(call::Bool, underlying,strike,ttm,rf,b,ivol,N)

    dt = ttm/N
    u = exp(ivol*sqrt(dt))
    d = exp(-ivol*sqrt(dt))
    pu = (exp(b*dt)-d)/(u-d)
    pd = 1.0-pu

    ps = Vector{Float64}(undef,N+1)
    paths=Vector{Float64}(undef,N+1)
    prices=Vector{Float64}(undef,N+1)

    nFact = factorial(big(N))
    value = 0.0
    
    Threads.@threads for i in 0:N
        prices[i+1] = underlying * u^i * d^(N-i)
        ps[i+1] = pu^i*pd^(N-i)
        paths[i+1] = nFact/(factorial(big(i))*factorial(big(N-i)))
    end

    if call
        prices .= max.(0,prices.-strike)
    else
        prices .= max.(strike .- prices,0)
    end

    prices = prices .* ps
    val = prices'*paths
    return exp(-rf*ttm)*val
end

bt_val = bt_bsm(true,underlying,strike,ttm,rf,rf,ivol,100)

println("BSM Value         : $call_val")
println("Simulated Value   : $(mean(ave))")
println("Integral Value    : $integral_val")
println("Binary Tree Value : $bt_val")

@btime sim_bsm(underlying,strike,days,rf,rf,ivol,tradingDayYear,10000)
@btime bt_bsm(true,underlying,strike,ttm,rf,rf,ivol,100)
@btime integral_bsm(true, underlying,strike,days,rf,ivol,tradingDayYear)
@btime gbsm(true,underlying,strike,ttm,rf,rf,ivol)

#Implied Vol:  
call_price = 6.0
f(iv) = gbsm(true,underlying,strike,ttm,rf,rf,iv) - call_price
implied_vol = find_zero(f,1)
println("Implied Vol: $implied_vol")

@btime find_zero(f,1)