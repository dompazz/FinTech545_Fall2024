using BenchmarkTools
using Distributions
using Random
using StatsBase
using DataFrames
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
using Roots
using QuadGK
# using GLM

include("../library/RiskStats.jl")
include("../library/simulate.jl")
include("../library/return_calculate.jl")
include("../library/fitted_model.jl")
include("../library/missing_cov.jl")
include("../library/ewCov.jl")
include("../library/expost_factor.jl")
include("../library/optimizers.jl")
include("../library/bt_american.jl")
include("../library/gbsm.jl")

#1 - 5 points
# Explain the difference in thinking between data modeling for risk analysis vs data 
# modeling for forecasting. 

# Data modeling for forecasting is focused on the future, and the goal is to predict future values.
# The model uses the expected future value.  Modeling for risk analysis is focused predicting
# the distribution of future values.  This encompasses not just the expected value, but also the
# variance, skewness, kurtosis, and other moments of the distribution.  The goal is to understand
# the potential outcomes and their likelihoods.3


#2 - 5 points
#a. (2) calculate mean, variance, skewness, kurtosis of the data
#b. (1) given a choice between a normal distribution and a t-distribution, 
# which one would you choose to model the data? Why?
#c. (2) fit both disitributions and prove or disprove your choice in b.

#Data Generation:
Random.seed!(123)
d = TDist(8)*.1 + .05
data = rand(d, 1000)
CSV.write("problem2.csv", DataFrame(:X=>data))

#Solution
data = CSV.read("problem2.csv",DataFrame)[!,:X]
#a
mean_data = mean(data)
var_data = var(data)
skew_data = skewness(data)
kurt_data = kurtosis(data)
println("Mean: ", mean_data)
println("Variance: ", var_data)
println("Skewness: ", skew_data)
println("Kurtosis: ", kurt_data)
# Mean: 0.0455797136893345
# Variance: 0.013038048101259763
# Skewness: 0.1965402908480158
# Kurtosis: 1.0595536757724915

#b Given the Kurtosis is >0, I would choose the t-distribution to model the data.

#c 
normal = fit_normal(data)
t = fit_general_t(data)
n_aicc = AICC(normal.errorModel, data,2)
t_aicc = AICC(t.errorModel, data,3)
println("Normal AICC: ", n_aicc)
println("T AICC: ", t_aicc)
if n_aicc < t_aicc
    println("Normal Distribution is better")
else
    println("T-Distribution is better")
end
# Normal AICC: -1498.9943166561643
# T AICC: -1516.628955621469
# T-Distribution is better

#3 - 5 points
#a. (1) Calculate the pairwise covariance matrix of the data.
#b. (2) Is the matrix as least positive semi-definite?  Why?
#c. (2) If not, find the nearest positive semi-definite matrix using Higham's method.

#Data Generation:
function generate_with_missing(n,m; pmiss=.25)
    x = Array{Union{Missing,Float64},2}(undef,n,m)

    c = fill(0.99,(m,m)) + .01*I(m)
    r = rand(MvNormal(fill(0,m),c),n)'
    for i in 1:n, j in 1:m
        if rand() >= pmiss
            x[i,j] = r[i,j]
        end
    end
    return x
end

Random.seed!(5)
x = generate_with_missing(50,5,pmiss=.25)
CSV.write("problem3.csv", DataFrame(x,:auto))

#a.
data=CSV.read("problem3.csv",DataFrame) |> Matrix
c = missing_cov(x; skipMiss=false, fun=cov)
# 5×5 Matrix{Float64}:
#  1.47048   1.45421   0.877269  1.90323  1.44436
#  1.45421   1.25208   0.539548  1.62192  1.23788
#  0.877269  0.539548  1.27242   1.17196  1.09191
#  1.90323   1.62192   1.17196   1.81447  1.58973
#  1.44436   1.23788   1.09191   1.58973  1.39619

#b
c2 = cov2cor(c)
println(min(eigvals(c2)...))
# -0.09482978874911373
# The matrix is not positive semi-definite as the smallest eigenvalue is negative.

#c
ch = higham_nearestPSD(c)
# 5×5 Matrix{Float64}:
#  1.47048   1.33236   0.884378  1.6276   1.39956
#  1.33236   1.25208   0.619028  1.4506   1.21445
#  0.884378  0.619028  1.27242   1.07685  1.05966
#  1.6276    1.4506    1.07685   1.81447  1.57793
#  1.39956   1.21445   1.05966   1.57793  1.39619


#4 - 5 points
# given the data in the problem4.csv file, and assuming the data are normally distributed
# calculate the exponetially weighted covariance matrix with lambda = 0.94.
#a. (3) What are the risk parity portfolio weights using standard deviation as your
#       risk measure.  
#b. (2) What are the risk parity portfolio weights using expected shortfall as your
#       risk measure.

#data generation
Random.seed!(4)
# Create a covariance matrix with random correlations and standard deviations between 0.02 and 0.1
function random_cov_matrix(n)
    # Generate random standard deviations
    std_devs = rand(Uniform(0.02, 0.1), n)
    
    # Generate random correlation matrix
    corr_matrix = Matrix{Float64}(I, n, n)
    for i in 1:n, j in i+1:n
        corr_matrix[i, j] = corr_matrix[j, i] = rand(Uniform(-1, 1))
    end
    
    # Convert correlation matrix to covariance matrix
    cov_matrix = Diagonal(std_devs) * corr_matrix * Diagonal(std_devs)
    return cov_matrix
end
cout = random_cov_matrix(5)
eigvals(cov2cor(cout))
cout = higham_nearestPSD(cout)
out = simulateNormal(1000, cout)
CSV.write("problem4.csv", DataFrame(out,:auto))

#a.
data = CSV.read("problem4.csv",DataFrame) |> Matrix
covar = ewCovar(data,0.94)
# 5×5 Matrix{Float64}:
#   0.00800778  -0.001406    -0.00417763   0.00181268   0.00055517
#  -0.001406     0.00532315   0.00295607  -0.00223105   0.00270149
#  -0.00417763   0.00295607   0.00583109  -0.00303737   0.00182102
#   0.00181268  -0.00223105  -0.00303737   0.00285228  -0.00215035
#   0.00055517   0.00270149   0.00182102  -0.00215035   0.00244845
w_std,status = riskParity(covar)
println(w_std)

# [0.08325194470526755, 0.08296831943674841, 0.2131321542254191, 0.42411305053917386, 0.1965345310933912]

function ewCovar2(x,λ)
    m,n = size(x)

    #Calculate the weights
    w = expW(m,λ)

    #Remove the weighted mean from the series and add the weights to the covariance calculation
    xm = sqrt.(w) .* (x .- mean(x,dims=1))

    #covariance = (sqrt(w) # x)' * (sqrt(w) # x)  where # is elementwise multiplication.
    return xm' * xm
end
covar = ewCovar2(data,0.94)
# 5×5 Matrix{Float64}:
#   0.00806428   -0.0016624   -0.00427848   0.00188879   0.000444114
#  -0.0016624     0.00648666   0.00341373  -0.00257643   0.00320545
#  -0.00427848    0.00341373   0.00601111  -0.00317323   0.00201925
#   0.00188879   -0.00257643  -0.00317323   0.00295481  -0.00229995
#   0.000444114   0.00320545   0.00201925  -0.00229995   0.00266674
w_std,status = riskParity(covar)
# Alt ewCov method, is OK
# [0.0862812080293293, 0.07741278065890032, 0.21589777171317712, 0.42948572877750035, 0.1909225108210929]

#b.
# the expected shortfall of the 5% quantile of the portfolio returns assuming multivariate normality
# is a linear function of standard deviation.  Therefore the risk parity weights are the same.
# [0.08325194470526755, 0.08296831943674841, 0.2131321542254191, 0.42411305053917386, 0.1965345310933912]

#5 - 10 points
# you own a portfolio with the following weights:
# a=.3, b=.2, c=.5
# the returns of each asset for the last 30 days are in the data.csv file
#a. (5) calculate the ex-post return contribution of each asset.
#b. (5) calculate the ex-post risk contribution of each asset. 
# Generate Data
out = simulateNormal(30, covar)[:, 1:3] .* sqrt(1/10)
CSV.write("problem5.csv", DataFrame(out,:auto))

#a. & b.
data = CSV.read("problem5.csv",DataFrame)
Attribution, weights, factorWeights = expost_factor([0.3,0.2,0.5],data,data,I(3))
println(Attribution)
# 3×6 DataFrame
#  Row │ Value               x1            x2           x3         Alpha         Portfolio 
#      │ String              Float64       Float64      Float64    Float64       Float64
# ─────┼───────────────────────────────────────────────────────────────────────────────────
#    1 │ TotalReturn         -0.221446     -0.0160075   0.301467   -3.33067e-16  0.0810983
#    2 │ Return Attribution  -0.0655125    -0.00221981  0.148831   -2.08152e-16  0.0810983
#    3 │ Vol Attribution     -0.000620223   0.0028271   0.0125798  -1.98335e-17  0.0147867

#6 - 20 points
# There are 2 stock price time series in the data.csv file.  
# Use arithmetic returns for your models
# Assume 0 mean for returns going forward.
# Report VaR as a $ value
# The current risk free rate is 4.75%
# Assume 252 trading days in a year
# Assume implied volatility does not change during the simulation
#You own
#  -- 100 shares of stock A
#  -- 100 American put options on stock A with a strike price of 100.  The time 
#     to maturity is 1 year.  The implied 
#     volatility of the put option is 0.2.  The stock pays a 0.025 dividend 
#     60 and 220 days from now.
#  -- 50 Share of stock B
#  -- you are short 50 European call options on stock B with a strike price of 
#     100.  The current price of the call is $6.5.  The time to maturity is 100
#    days.  The stock pays no dividends.
#a. (5) Using a delta normal approach, calculate the 1 day VaR and ES 
#   of the portfolio at the 5% confidence level.
#b. (5) Using a simulated approach and assuming multivariate normallity,
#   calculate the 1 day VaR and ES of the portfolio at the 5% confidence level.
#c. (5) Using the best fit between a normal and t-distribution for each stock,
#   calculate the 1 day VaR and ES of the portfolio at the 5% confidence level.
#d. (5) Compare the results of a, b, and c.  Which one would you use and why?

#Data Generation
# Generate Data
n = 252  # number of trading days in a year
mu = [0.0, 0.0]  # mean returns
sigma = [0.0125, 0.015]  # volatilities
correlation = 0.5  # correlation between returns

# Covariance matrix
cov_matrix = [sigma[1]^2 correlation * sigma[1] * sigma[2]; correlation * sigma[1] * sigma[2] sigma[2]^2]

# Generate correlated returns
Random.seed!(5)
returns = rand(MvNormal(mu, cov_matrix), n)'

# Convert returns to prices
prices_A = [100.0]
prices_B = [100.0]
for i in 1:n
    push!(prices_A, prices_A[end] * (1 + returns[i, 1]))
    push!(prices_B, prices_B[end] * (1 + returns[i, 2]))
end

prices_A = prices_A .* (100 / prices_A[end] )
prices_B = prices_B .* (100 / prices_B[end] )

# Create DataFrame and write to CSV
data = DataFrame(:Date => 0:n, :A => prices_A, :B => prices_B)
CSV.write("problem6.csv", data)

#set up
prices = CSV.read("problem6.csv",DataFrame)
returns = return_calculate(prices,dateColumn="Date")[!,[:A,:B]]
returns[!,:A] = returns[!,:A] .- mean(returns[!,:A])
returns[!,:B] = returns[!,:B] .- mean(returns[!,:B])
covar = cov(Matrix(returns))
currentPrices = prices[end,:]
rf = 0.0475
assets = ["A","B"]

portfolio = DataFrame(:Asset => ["A","A","B","B"],
                      :Type => ["Stock","Put","Stock","Call"],
                      :Amount => [100,100,50,-50],
                      :Strike => [0,100,0,100],
                      :Maturity => [0,252,0,100],
                      :Vol => [0,0.2,0,0.2],
                      :Dividend => [0,0.025,0,0],
                      :Price => [currentPrices[:A],0,currentPrices[:B],6.5],
                      :Delta => [1.0,.0,1.0,.0])

#fill in the needed values
# A Put - find price
portfolio[2,:Price] = bt_american(false, 100,100,1,rf,[0.025,0.025],[60,220],0.2,250)
A_DivSchedule = hcat([0.025,0.025],[60,220])
portfolio[2,:Delta] = (bt_american(false, 100+1e-6,100,1,rf,[0.025,0.025],[60,220],0.2,250) - portfolio[2,:Price]) / 1e-6
# B Call - find implied volatility
_f(x) = 6.5 - gbsm(true,100,100,100/252,rf,rf,x).value
find_zero(_f,0.2)
portfolio[4,:Vol] = find_zero(_f,0.2)
portfolio[4,:Price] = 6.5
portfolio[4,:Delta] = gbsm(true,100,100,100/252,rf,rf,portfolio[4,:Vol]).delta
portfolio[!,:Value] = portfolio[!,:Amount] .* portfolio[!,:Price]

pvalue = sum(portfolio.Value)
# 15292.72907299572

#a. Calculate the Delta Normal portfolio VaR
#asset deltas
portfolio[!,:Exposure] = portfolio[!,:Amount] .* portfolio[!,:Delta] .* 100
deltas = combine(groupby(portfolio,:Asset),:Exposure => sum)
deltas = deltas.Exposure_sum ./ sum(portfolio.Value)
portfolio_std = sqrt(deltas' * covar * deltas)
dnVaR = VaR(Normal(0,portfolio_std),alpha=0.05) * sum(portfolio.Value)
dnES = ES(Normal(0,portfolio_std),alpha=0.05) * sum(portfolio.Value)
println("Delta Normal VaR: \$", dnVaR)
println("Delta Normal ES: \$", dnES)
# Delta Normal VaR: $159.4518115787715
# Delta Normal ES: $199.9590045653506

#b. Calculate the Monte Carlo portfolio VaR
#simulate the portfolio
nSim = 100
vars = zeros(nSim)
ess = zeros(nSim)
for k in 1:nSim
    print(k, " - ")
    n = 1000
    pValues = zeros(n)
    simPrices = DataFrame((1 .+ simulateNormal(n, covar;seed=2*k)) * 100,assets)

    pnls = Array{Float64}(undef,(n,4))
    Threads.@threads for i in 1:n
        for j in 1:4
            if portfolio[j,:Type] == "Stock"
                pnls[i,j] = (simPrices[i,portfolio.Asset[j]] - currentPrices[portfolio.Asset[j]]) * portfolio[j,:Amount]
            elseif portfolio[j,:Type] == "Put"
                pnls[i,j] = (bt_american(false, simPrices[i,portfolio.Asset[j]],portfolio[j,:Strike],(portfolio[j,:Maturity]-1)/252,rf,A_DivSchedule[:,1],convert.(Int64,A_DivSchedule[:,2]).-1,portfolio[j,:Vol],250) - portfolio[j,:Price]) * portfolio[j,:Amount]
            elseif portfolio[j,:Type] == "Call"
                pnls[i,j] = (gbsm(true,simPrices[i,portfolio.Asset[j]],portfolio[j,:Strike],(portfolio[j,:Maturity]-1)/252,rf,rf,portfolio[j,:Vol]).value - portfolio[j,:Price]) * portfolio[j,:Amount]
            end
        end
    end
    tPnL = vec(sum(pnls,dims=2))
    vars[k] = VaR(tPnL)
    ess[k] = ES(tPnL)
    println(vars[k], " - ", ess[k])
end
println("Monte Carlo VaR: \$", mean(vars))
println("99% VaR Range: ", quantile(vars,[0.005,0.995]))
println("Monte Carlo ES: \$", mean(ess))
println("97.5% ES Range: ", quantile(ess,[0.005,0.995]))
# Monte Carlo VaR: $159.45425007763862
# 99% VaR Range: [148.37268228352156, 171.16433896704172]
# Monte Carlo ES: $198.60860951946458
# 99% ES Range: [181.89505494296662, 215.11656091233533]

#c. Calculate the best fit VaR
#fit the distributions
models = Dict{String,FittedModel}()
for a in assets
    normal = fit_normal(returns[!,a])
    t = fit_general_t(returns[!,a])
    println("Asset: ", a, " Normal AICC: ", AICC(normal.errorModel, returns[!,a],2), " T AICC: ", AICC(t.errorModel, returns[!,a],3))
    if AICC(normal.errorModel, returns[!,a],2) < AICC(t.errorModel, returns[!,a],3)
        models[a] = normal
    else
        models[a] = t
    end
end
#look at the models, if both are normal, then we already know the answer
for a in assets
    println("Asset: ", a, " Model: ", models[a].errorModel)
end
#Both models are normal, so we can use the normal Monte Carlo VaR
# Monte Carlo VaR: $159.45425007763862
# 99% VaR Range: [148.37268228352156, 171.16433896704172]
# Monte Carlo ES: $198.60860951946458
# 99% ES Range: [181.89505494296662, 215.11656091233533]

#d. Compare the results of a, b, and c.  Which one would you use and why? 
# The models are all very close.  This is because the long time to maturity of the options
# makes the gamma of the options small.  The value function for those options is nearly
# linear in the underlying asset.  Therefore the delta normal model is a good approximation.
# The two Monte Carlo models assume normallity, so the Delta Normal model, which assumes
# linear payoff functions and normallity is a good choice.

# If the options had a shorter time to maturity, the gamma would be larger and the delta
# normal model would be less accurate.  In that case, the Monte Carlo models would be
# the better choice.