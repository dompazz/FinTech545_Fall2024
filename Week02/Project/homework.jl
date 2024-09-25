using Distributions
using StatsBase
using DataFrames
using CSV
using Plots
using PlotThemes
using Printf
using JuMP
using Ipopt
using StateSpaceModels
using LinearAlgebra

#Problem 1
function first4Moments(sample)

    n = size(sample,1)

    #mean
    μ_hat = sum(sample)/n

    #remove the mean from the sample
    sim_corrected = sample .- μ_hat
    cm2 = sim_corrected'*sim_corrected/n

    #variance
    σ2_hat = sim_corrected'*sim_corrected/(n)

    #skew
    skew_hat = sum(sim_corrected.^3)/n/sqrt(cm2*cm2*cm2)

    #kurtosis
    kurt_hat = sum(sim_corrected.^4)/n/cm2^2

    excessKurt_hat = kurt_hat - 3

    return μ_hat, σ2_hat, skew_hat, excessKurt_hat
end

prob1 = CSV.read("Project/problem1.csv", DataFrame)

m,v,sk,k =first4Moments(prob1.x)

mj,vj,skj,kj = (mean(prob1.x), var(prob1.x), skewness(prob1.x), kurtosis(prob1.x))

n = length(prob1.x)

v * (n/(n-1))


#Problem 2
include("fitting_functions.jl")

prob2 = CSV.read("Project/problem2.csv",DataFrame)
n = size(prob2,1)
ols = fit_ols(prob2.y,prob2.x)
ols.beta
std(ols.errors)

mle_n = fit_regression_mle(prob2.y,prob2.x)
mle_n.beta

mle_t = fit_regression_t(prob2.y,prob2.x)
mle_t.beta

function aicc(ll,n,k)
    aic = 2*k - 2*ll
    aicc = aic + (2*k^2 + 2*k)/(n-k-1)
    return aicc
end

mle_n_ll = sum(log.(pdf.(Normal(0,mle_n.beta[3]),mle_n.errors)))
mle_n_aicc = aicc(mle_n_ll,n,3)

mle_t_ll = sum(log.(pdf.(TDist(mle_t.beta[4])*mle_t.beta[3],mle_t.errors)))
mle_t_aicc = aicc(mle_t_ll,n,4)

#part c 
prob2c = CSV.read("Project/problem2_x.csv",DataFrame)
means = mean.(eachcol(prob2c))
covar = cov(Matrix(prob2c))

function cexpectation(x,μ,covar)
    b = covar[1,2]/covar[1,1]
    expt = μ[2] .+ b*(x .- μ[1])
    var = covar[2,2] - b*covar[1,2]
    nd = Normal.(expt,sqrt(var))
    hcat(expt, quantile.(nd,.025), quantile.(nd,.975))
end

# prob2c_x = CSV.read("Project/problem2_x1.csv",DataFrame)
x = hcat(prob2c,DataFrame(cexpectation(prob2c.x1,means,covar),["E","LCL","UCL"]))
sort!(x,:x1)
scatter(x.x1,x.x2,label="Data")
plot!(x.x1,x.E,label="Expected Value",linecolor=:black)
plot!(x.x1,x.LCL,label="Lower CL",linecolor=:pink)
plot!(x.x1,x.UCL,label="Upper CL",linecolor=:pink)

#Porblem 3
prob3 = CSV.read("Project/problem3.csv",DataFrame)
x = prob3.x 

function plot_ts(y;imgName="series", length=10,title=nothing)
    n = size(y,1)
    l = [i for i in 1:length]
    acf = autocor(y,l)
    p_acf = pacf(y, l)

    df = DataFrame(:t=>l, :acf=>acf, :pacf=>p_acf)

    
    theme(:dark)

    if title === nothing
        p0 = Plots.plot([i for i in 1:n], y, legend=false)
    else
        p0 = Plots.plot([i for i in 1:n], y, legend=false,title=title)
    end
    

    p1 = Plots.plot(df.t, df.acf, title="AutoCorrelation", seriestype=:bar, legend=false)

    p2 = Plots.plot(df.t, df.pacf, title="Partial AutoCorrelation", seriestype=:bar, legend=false)
    


    p = plot(p0, plot(p1,p2,layout=(1,2)),layout=(2,1))

    Plots.savefig(p,imgName)

    p
end

plot_ts(x)



for i in 1:3
    mdl = SARIMA(x,order=(i,0,0),include_mean=true)
    StateSpaceModels.fit!(mdl)
    print_results(mdl)
end
for i in 1:3
    mdl = SARIMA(x,order=(0,0,i),include_mean=true)
    StateSpaceModels.fit!(mdl)
    print_results(mdl)
end
