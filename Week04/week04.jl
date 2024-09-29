using DataFrames
using Plots
using Distributions
using CSV
using MarketData
using Dates
using LoopVectorization
using LinearAlgebra
using KernelDensity
using Interpolations
using Roots
using QuadGK

alpha = 0.05
d = TDist(10)*5+1.0

#The alpha-percentile is VaR
VaR = -quantile(d,alpha)
println(VaR)

x = [i for i in -25:0.1:25]
_pdf = pdf.(d,x)
df = DataFrame(:x=>x,:pdf=>_pdf)

#typical distribution plot with shading
plot(df.x,df.pdf,legend=false,fill=(0, .25,:orange), linecolor=:black)
x2 = x[findall(x->x<=-VaR,x)]
#color the area below in red
plot!(x2, df.pdf[1:size(x2,1)],legend=false, fill=(0,.5,:red), linecolor=:black, title="Profit/Loss")




#Unlikely Events
rets = CSV.read("DailyReturn.csv",DataFrame)

spy = rets.SPY
sd = std(spy)

VaR_05 = -quantile(Normal(0,sd),.05)
VaR_01 = -quantile(Normal(0,sd),.01)

# spy_GFC = DataFrame(yahoo("SPY", YahooOpt(period1 = DateTime(2008,9,1), period2=DateTime(2009,4,1))))
# CSV.write("spy_gfc.csv",spy_GFC)
spy_GFC = CSV.read("spy_gfc.csv",DataFrame)

n = size(spy_GFC,1)
spy_GFC[:,:return] = Vector{Float64}(undef,n)
for i in 2:n
    spy_GFC[i,:return] = log(spy_GFC[i,:AdjClose]/spy_GFC[i-1,:AdjClose])
end

#VaR under Normal Conditions
println("5% VaR: $(round(VaR_05*100,digits=2))%")
println("1% VaR: $(round(VaR_01*100,digits=2))%")

#What happened during the GFC
max_loss = min(spy_GFC.return...)
println("Maximum Loss During the GFC: $(round(max_loss*100,digits=2))%")


zscore = max_loss/sd
ml_cdf = cdf(Normal(),zscore)
println("Loss is $(round(zscore,digits=2))σ -->  CDF(loss)=$ml_cdf")

universe_days = 14_000_000_000*365.25*24*60
p0 = (big"1.0"-convert(BigFloat,ml_cdf))^(universe_days)
println("Probability of not seeing a $(round(zscore,digits=2))σ day for the life of the universe = $(100*p0)%")


#Not subadditive:
# 2 identical bonds from different issuers.  4% chance of default.  1 period remaining.  $100 par value.  Price $90

n=1000
pdef = .04
cost = 90.0
par = 100.0
#Bond 1
PL1 = Vector{Float64}(undef,n)
for i in 1:n
    if rand() <= pdef
        #Value minus Cost
        PL1[i] = 0.0 - cost
    else
        PL1[i] = par - cost
    end
end
sort!(PL1)
# Divide by the portfolio cost to make it a percent
PL1 = PL1 ./ cost

#VaR is the α*n value on the simulation -- the α - percentile.
VaR1 = -PL1[convert(Int64,0.05*n)]


#Portfolio of 2 bonds
PLP = Vector{Float64}(undef,n)
for i in 1:n
    #first bond in the portfolio
    if rand() <= pdef
        #Value minus Cost
        PLP[i] = 0.0 - cost
    else
        PLP[i] = par - cost
    end
    
    #second bond in the portfolio
    if rand() <= pdef
        #Value minus Cost
        PLP[i] += 0.0 - cost
    else
        PLP[i] += par - cost
    end
end
sort!(PLP)
# Divide by the portfolio cost to make it a percent
PLP = PLP ./ (2*cost)

VaRP = -PLP[convert(Int64,0.05*n)]

println(VaR1*2cost)  #VaR is negative (meaning we are making money) as the 5% on the simulation is still fullly getting paid.
               #We lose money 4% of the time.
println(VaRP*2cost)  #VaR on the portfolio of 2 bonds is positive.  

stdP1=std(PL1)
stdP=std(PLP)

println("Standard Deviation is still coherent even with nonlinear payoffs")
println("2q(P1) = $(2cost*stdP1)")
println("q(P1+P2) = $(2cost*stdP)")
println("2q(P1) >= q(P1+P2) -- $(2cost*stdP1 >= 2cost*stdP)")



#Delta Normal VaR Example
A = [1,8,20]
P = [7,10]
Q = [10,20,5]
PV = A'*Q
dP1 = [0.5,1,0]
dP2 = [0,0,1]
covar = [0.01 0.0075
         0.0075 0.0225]

dR1 = P[1]/PV * dP1'*Q
dR2 = P[2]/PV * dP2'*Q
∇ = [dR1, dR2]

σ = sqrt(∇'*covar*∇)
var = -PV*quantile(Normal(),0.05)*σ
varPct = var/PV




#Delta Normal VaR
include("return_calculate.jl")
prices = CSV.read("DailyPrices.csv",DataFrame)
current_prices = prices[size(prices,1),:]
returns = return_calculate(prices,dateColumn="Date")

#Our portfolio
holdings = Dict{String,Float64}()
holdings["GOOGL"] = 10
holdings["NVDA"] = 75
holdings["TSLA"] = 25

#filter prices and returns
nm = intersect(names(prices),Set(keys(holdings)))
current_prices= current_prices[nm]
returns = returns[!,nm]

#calculate the portfolio value.
PV = 0.0
delta = Vector{Float64}(undef,length(nm))
i=1
for s in nm
    global i, PV
    value = holdings[s] * current_prices[s]
    PV += value
    delta[i] = value
    i+=1
end

delta = delta ./PV

Sigma = cov(Matrix(returns))
e = eigvals(Sigma)
p_sig = sqrt(delta'*Sigma*delta)
VaR = -PV * quantile(Normal(),0.05) * p_sig

println("Delta Normal ")
println("Current Portfolio Value: $PV")
println("Current Portfolio VaR: $VaR")
println(" ")

#MC VaR - Same Portfolio
n = 10000
#Simulate the returns.  Matrix is PD, so the internal simulation is probably faster
sim_returns = rand(MvNormal(fill(0,3),Sigma),n)'

sim_prices = (1 .+ sim_returns) .* Array(current_prices[nm])'
vHoldings = [holdings[s] for s in nm]
pVals = sim_prices*vHoldings
sort!(pVals)

a=convert(Int64,.05*n)
VaR = PV - pVals[a]
println("MC Normal ")
println("Current Portfolio Value: $PV")
println("Current Portfolio VaR: $VaR")


#KDE - Get bandwidth and use a Normal Kernel 
#not super efficient but is generally more accurate
bw = KernelDensity.default_bandwidth(pVals) 

d = Normal(0,bw)
g(x) = mean(cdf.(d,x .- pVals)) - 0.05
VaR = PV - find_zero(g,pVals[a])
println("Current Portfolio VaR(KDE): $VaR")
hKDE = InterpKDE(kde(pVals))
h2KDE = InterpKDE(kde_lscv(pVals))

function quantile_kde(kde, alpha; guess=mean(kde.kde.x))
    #PDF function based on the KDE
    pdf_kde(x) = pdf(kde,x)
    #Integration of PDF to get CDF
    cdf_kde(y) = quadgk(pdf_kde,0,y)[1]
    #Root solver to find X
    g(z) = cdf_kde(z)-alpha
    find_zero(g,guess)
end

VaR = PV - quantile_kde(hKDE,0.05)
println("Current Portfolio VaR(KDE2): $VaR")  #This should equal the 1st method
VaR = PV - quantile_kde(h2KDE,0.05)
println("Current Portfolio VaR(KDE3): $VaR")
println(" ")


#Historical VaR
sim_prices = (1 .+ Matrix(returns)) .*  Array(current_prices[nm])'
vHoldings = [holdings[s] for s in nm]
pVals = sim_prices*vHoldings
sort!(pVals)

n = size(returns,1)
a=convert(Int64,floor(.05*n))
VaR = PV - pVals[a]
println("Historical VaR ")
println("Current Portfolio Value: $PV")
println("Current Portfolio VaR: $VaR")

#KDE - Get bandwidth and use a Normal Kernel 
#not super efficient but is generally more accurate
bw = KernelDensity.default_bandwidth(pVals) 

d = Normal(0,bw)
g(x) = mean(cdf.(d,x .- pVals)) - 0.05
VaR = PV - find_zero(g,pVals[a])
println("Current Portfolio VaR(KDE): $VaR")


hKDE = InterpKDE(kde(pVals))
h2KDE = InterpKDE(kde_lscv(pVals))

VaR = PV - quantile_kde(hKDE,0.05)
println("Current Portfolio VaR(KDE2): $VaR")  #This should equal the 1st method
VaR = PV - quantile_kde(h2KDE,0.05)
println("Current Portfolio VaR(KDE3): $VaR")
println(" ")