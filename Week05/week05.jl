using DataFrames
using Plots
using Distributions
using CSV
using Dates
using LoopVectorization
using LinearAlgebra
using KernelDensity
using Interpolations
using Roots
using QuadGK
using JuMP
using Ipopt
using StateSpaceModels
using StatsBase

#Expected Shortfall
function VaR_ES(x,alpha=0.05)
    xs = sort(x)
    n = alpha*size(xs,1)
    iup = convert(Int64,ceil(n))
    idn = convert(Int64,floor(n))
    VaR = (xs[iup] + xs[idn])/2


    ES = mean(xs[1:idn])

    return -VaR, -ES
end

nSim = 10000
nDist = Normal(0,.05)
simNorm = rand(nDist,nSim)

normVaR, normES = VaR_ES(simNorm)

eNormVaR = -quantile(nDist,0.05)

eNormES = 0.05*pdf(Normal(),quantile(Normal(),0.05))/0.05

f(x) = x*pdf(nDist,x)
st = quantile(nDist,1e-12)
eNormES2= -quadgk(f,st,-eNormVaR)[1]/.05

println("VaR ($normVaR) vs Expected VaR ($eNormVaR)")
println("ES ($normES) vs Expected ES ($eNormES) vs Expected ES2 ($eNormES2)")




#Model Simulation
include("../Week04/return_calculate.jl")
prices = CSV.read("DailyPrices.csv",DataFrame)
current_prices = prices[size(prices,1),:]
returns = return_calculate(prices,dateColumn="Date")

#Get a stock and market return
spy = returns.SPY
tsla = returns.TSLA
n = size(spy,1)

#Do OLS
X = hcat(fill(1.0,n),spy)
B = inv(X'*X)*X'*tsla
e = tsla - X*B

#covariance of [X,e]
covar = cov(hcat(X,e))
means = mean.(eachcol(hcat(X,e)))

#can we make a MvNormal out of this?
try 
    d = MvNormal(means, covar)
catch(err) 
    println(err)
end

#Nope, remove the Intercept from X, it has no variance and is always =1
means = means[2:3]
covar = covar[2:3,2:3]

#simulate and apply the model y = XB + e
nsim = 100000
sim = rand(MvNormal(means,covar),nsim)'

x_sim = hcat(fill(1.0,nsim),sim[:,1])
y_sim = x_sim*B + sim[:,2]

#How does the distribution compare:
println("Mean TSLA vs Simulated - $(mean(tsla)) vs $(mean(y_sim))")
println("StDev TSLA vs Simulated - $(std(tsla)) vs $(std(y_sim))")


#Multiple models:
aapl = returns.AAPL

function OLS(X,Y)
    _X = hcat(fill(1.0,n),X)
    B = inv(_X'*_X)*_X'*Y
    e = Y - _X*B
    return B, e
end

#OLS on the 2 Stocks
B_tsla, e_tsla = OLS(spy,tsla)
B_aapl, e_aapl = OLS(spy,aapl)

#look at the correlation
corr = cor(hcat(spy,e_tsla,e_aapl))

#covariance
covar = cov(hcat(spy,e_tsla,e_aapl))
means = vec(hcat(mean(spy),0,0))

#same simulation as before
nsim = 1000000
sim = rand(MvNormal(means,covar),nsim)'

x_sim = hcat(fill(1.0,nsim),sim[:,1])
tsla_sim = x_sim*B_tsla + sim[:,2]
aapl_sim = x_sim*B_aapl + sim[:,3]

#distribution compares...
println("Mean TSLA vs Simulated - $(mean(tsla)) vs $(mean(tsla_sim))")
println("StDev TSLA vs Simulated - $(std(tsla)) vs $(std(tsla_sim))")
println(" ")
println("Mean AAPL vs Simulated - $(mean(aapl)) vs $(mean(aapl_sim))")
println("StDev AAPL vs Simulated - $(std(aapl)) vs $(std(aapl_sim))")


#Show block diagonal does not hold when not using the same X values in the simulation
gm = returns.GM
B_tsla, e_tsla = OLS(gm,tsla)
B_aapl, e_aapl = OLS(spy,aapl)

corr = cor(hcat(gm,spy,e_tsla,e_aapl))

#additional distribution compares...
println("Mean TSLA vs Simulated -- $(mean(tsla)) vs $(mean(tsla_sim))")
println("StDev TSLA vs Simulated -- $(std(tsla)) vs $(std(tsla_sim))")
println("Skewness TSLA vs Simulated -- $(skewness(tsla)) vs $(skewness(tsla_sim))")
println("Kurtosis TSLA vs Simulated -- $(kurtosis(tsla)) vs $(kurtosis(tsla_sim))")
println(" ")
println("Mean AAPL vs Simulated - $(mean(aapl)) vs $(mean(aapl_sim))")
println("StDev AAPL vs Simulated - $(std(aapl)) vs $(std(aapl_sim))")
println("Skewness AAPL vs Simulated -- $(skewness(aapl)) vs $(skewness(aapl_sim))")
println("Kurtosis AAPL vs Simulated -- $(kurtosis(aapl)) vs $(kurtosis(aapl_sim))")
println(" ")
println("Skewness and Kurtosis of ϵ_tesla -- $(skewness(e_tsla)) and $(kurtosis(e_tsla))")
println("Skewness and Kurtosis of ϵ_aapl -- $(skewness(e_aapl)) and $(kurtosis(e_aapl))")


#Generalize T, Sum LL function
function general_t_ll(mu,s,nu,x)
    td = TDist(nu)*s + mu
    sum(log.(pdf.(td,x)))
end

#MLE for a Generalize T
function fit_general_t(x)
    global __x
    __x = x
    mle = Model(Ipopt.Optimizer)
    set_silent(mle)

    #approximate values based on moments
    start_m = mean(x)
    start_nu = 6.0/kurtosis(x) + 4
    start_s = sqrt(var(x)*(start_nu-2)/start_nu)

    @variable(mle, m, start=start_m)
    @variable(mle, s>=1e-6, start=1)
    @variable(mle, nu>=2.0001, start=start_s)

    #Inner function to abstract away the X value
    function _gtl(mu,s,nu)
        general_t_ll(mu,s,nu,__x)
    end

    register(mle,:tLL,3,_gtl;autodiff=true)
    @NLobjective(
        mle,
        Max,
        tLL(m, s, nu)
    )
    optimize!(mle)

    m = value(m)
    s = value(s)
    nu = value(nu)

    #return the parameters as well as the Distribution Object
    return (m, s, nu, TDist(nu)*s+m)
end

#Fit each stock return to a T distribution
m_spy, s_spy, nu_spy, d_spy = fit_general_t(spy)
m_tsla, s_tsla, nu_tsla, d_tsla = fit_general_t(tsla)
m_aapl, s_aapl, nu_aapl, d_aapl = fit_general_t(aapl)

#create the U matrix
U = hcat( 
    cdf(d_spy,spy),
    cdf(d_tsla,tsla),
    cdf(d_aapl,aapl)
)

#Transform U into Z
Z = quantile(Normal(),U)

#pearson correlation
R = cor(Z)

#how does that compare to the correlation of the raw observations?
R_emperical = cor(hcat(spy,tsla,aapl))

#Spearman correlations of Z
R_spearman = corspearman(Z)
#same as Spearman correlations of U
corspearman(U)

#Simulate using the Copula
NSim = 10000
#simulate with Pearson Correlation
copula = MvNormal(fill(0,3),R)
pearson = rand(copula,NSim)'
z = Normal()
pearson[:,1] = quantile(d_spy,cdf(z,pearson[:,1]))
pearson[:,2] = quantile(d_tsla,cdf(z,pearson[:,2]))
pearson[:,3] = quantile(d_aapl,cdf(z,pearson[:,3]))


#simulate with Spearman Correlation
copula = MvNormal(fill(0,3),R_spearman)
spearman = rand(copula,NSim)'
z = Normal()
spearman[:,1] = quantile(d_spy,cdf(z,spearman[:,1]))
spearman[:,2] = quantile(d_tsla,cdf(z,spearman[:,2]))
spearman[:,3] = quantile(d_aapl,cdf(z,spearman[:,3]))

#Modeled as Normals
mvn = MvNormal([mean(spy),mean(tsla)],cov(hcat(spy,tsla)))
normal = rand(mvn,NSim)'


#Compare 
plot(pearson[:,1],pearson[:,2],seriestype=:scatter,label="Copula Pearson Corr", color=:red, legend=:bottomright)
xlabel!("SPY")
ylabel!("TSLA")
plot!(spearman[:,1],spearman[:,2],seriestype=:scatter,label="Copula Spearman Corr",color=:blue)
plot!(normal[:,1],normal[:,2],seriestype=:scatter,label="Normal Model",color=:violet)
plot!(spy,tsla,seriestype=:scatter,label="Empirical",color=:green)





