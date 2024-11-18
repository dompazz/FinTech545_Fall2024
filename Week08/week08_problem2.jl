using BenchmarkTools
using Distributions
using Random
using StatsBase
using DataFrames
using Query
using Plots
using LinearAlgebra
using JuMP
using Ipopt
using Dates
using ForwardDiff
using FiniteDiff
using CSV
using LoopVectorization
# using GLM

include("../library/RiskStats.jl")
include("../library/simulate.jl")
include("../library/return_calculate.jl")

ff3 = CSV.read("F-F_Research_Data_Factors_daily.CSV", DataFrame)
mom = CSV.read("F-F_Momentum_Factor_daily.CSV",DataFrame)
returns = CSV.read("DailyReturn.csv",DataFrame)

# Join the FF3 data with the Momentum Data
ffData = innerjoin(ff3,mom,on=:Date)
rename!(ffData, names(ffData)[size(ffData,2)] => :Mom)
rename!(ffData, Symbol("Mkt-RF")=>:Mkt_RF)
ffData[!,names(ffData)[2:size(ffData,2)]] = Matrix(ffData[!,names(ffData)[2:size(ffData,2)]]) ./ 100
ffData[!,:Date] = Date.(string.(ffData.Date),dateformat"yyyymmdd")

returns[!,:Date] = Date.(returns.Date,dateformat"mm/dd/yyyy")

#join the FF3+1 to Stock data - filter to stocks we want
stocks = [:AAPL, :MSFT, Symbol("BRK-B"), :CSCO, :JNJ]
to_reg = innerjoin(returns[!,vcat(:Date, :SPY, stocks)], ffData, on=:Date)


xnames = [:Mkt_RF, :SMB, :HML, :Mom]

#OLS Regression for all Stocks
X = hcat(fill(1.0,size(to_reg,1)),Matrix(to_reg[!,xnames]))


Y = Matrix(to_reg[!,stocks])
Betas = (inv(X'*X)*X'*Y)'[:,2:size(xnames,1)+1]

max_dt = max(to_reg.Date...)
min_dt = max_dt - Year(10)
to_mean = ffData |>  @filter(_.Date >= min_dt && _.Date <= max_dt) |> DataFrame

#historic daily factor returns
exp_Factor_Return = mean.(eachcol(to_mean[!,xnames]))
expFactorReturns = DataFrame(:Factor=>xnames, :Er=>exp_Factor_Return)


#scale returns and covariance to geometric yearly numbers
stockMeans =log.(1 .+ Betas*exp_Factor_Return)*255 
covar = cov(log.(1.0 .+ Y))*255

function sr(w...)
    _w = collect(w)
    m = _w'*stockMeans - .0025
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

w = value.(w)
w = w / sum(w)

OptWeights = DataFrame(:Stock=>String.(stocks), :Weight => w, :cEr => stockMeans .* w)
println(OptWeights)

#Get Updated Prices 
updated = CSV.read("updated_prices.csv",DataFrame)
updated[!,:Date] = Date.(updated.Date,dateformat"mm/dd/yyyy")
upReturns = return_calculate(updated,dateColumn="Date")

upFf3 = CSV.read("updated_F-F_Research_Data_Factors_daily.CSV", DataFrame)
upMom = CSV.read("updated_F-F_Momentum_Factor_daily.CSV",DataFrame)
upFfData = innerjoin(upFf3,upMom,on=:Date)
rename!(upFfData, names(upFfData)[size(ffData,2)] => :Mom)
rename!(upFfData, Symbol("Mkt-RF")=>:Mkt_RF)
upFfData[!,names(upFfData)[2:size(upFfData,2)]] = Matrix(upFfData[!,names(upFfData)[2:size(upFfData,2)]]) ./ 100
upFfData[!,:Date] = Date.(string.(upFfData.Date),dateformat"yyyymmdd")

upFfData=vcat(ffData,upFfData)
select!(upFfData,Not(:RF))

#filter the FF returns to just the Stock return data
upFfData = innerjoin(upReturns,upFfData, on=:Date)[!,vcat(:Date,xnames)]

#calculate portfolio return and updated weights for each day
n = size(upReturns,1)
m = size(stocks,1)

pReturn = Vector{Float64}(undef,n)
residReturn = Vector{Float64}(undef,n)
weights = Array{Float64,2}(undef,n,length(w))
factorWeights = Array{Float64,2}(undef,n,length(xnames))
lastW = copy(w)
matReturns = Matrix(upReturns[!,stocks])
ffReturns = Matrix(upFfData[!,xnames])

for i in 1:n
    # Save Current Weights in Matrix
    weights[i,:] = lastW

    #Factor Weight
    factorWeights[i,:] = sum.(eachcol(Betas .* lastW))

    # Update Weights by return
    lastW = lastW .* (1.0 .+ matReturns[i,:])
    
    # Portfolio return is the sum of the updated weights
    pR = sum(lastW)
    # Normalize the wieghts back so sum = 1
    lastW = lastW / pR
    # Store the return
    pReturn[i] = pR - 1

    #Residual
    residReturn[i] = (pR-1) - factorWeights[i,:]'*ffReturns[i,:]

end

# Set the portfolio return in the Update Return DataFrame
upFfData[!,:Alpha] = residReturn
upFfData[!,:Portfolio] = pReturn

# Calculate the total return
totalRet = exp(sum(log.(pReturn .+ 1)))-1
# Calculate the Carino K
k = log(totalRet + 1 ) / totalRet

# Carino k_t is the ratio scaled by 1/K 
carinoK = log.(1.0 .+ pReturn) ./ pReturn / k
# Calculate the return attribution
attrib = DataFrame(ffReturns .* factorWeights .* carinoK, xnames)
attrib[!,:Alpha] = residReturn .* carinoK

# Set up a Dataframe for output.
Attribution = DataFrame(:Value => ["TotalReturn", "Return Attribution"])

newFactors = [xnames..., :Alpha]
# Loop over the factors
for s in vcat(newFactors, :Portfolio)
    # Total Stock return over the period
    tr = exp(sum(log.(upFfData[!,s] .+ 1)))-1
    # Attribution Return (total portfolio return if we are updating the portfolio column)
    atr =  s != :Portfolio ?  sum(attrib[:,s]) : tr
    # Set the values
    Attribution[!,s] = [ tr,  atr ]
end

# Check that the attribution sums back to the total Portfolio return
sum([Attribution[2,newFactors]...]) ≈ totalRet
 

# Realized Volatility Attribution

# Y is our stock returns scaled by their weight at each time
Y =  hcat(ffReturns .* factorWeights, residReturn)
# Set up X with the Portfolio Return
X = hcat(fill(1.0, size(pReturn,1)),pReturn)
# Calculate the Beta and discard the intercept
B = (inv(X'*X)*X'*Y)[2,:]
# Component SD is Beta times the standard Deviation of the portfolio
cSD = B * std(pReturn)

#Check that the sum of component SD is equal to the portfolio SD
sum(cSD) ≈ std(pReturn)

# Add the Vol attribution to the output 
Attribution = vcat(Attribution,    
    DataFrame(:Value=>"Vol Attribution", [Symbol(newFactors[i])=>cSD[i] for i in 1:size(newFactors,1)]... , :Portfolio=>std(pReturn))
)

println(Attribution)


# Factor Attribution...


