using Distributions
using HypothesisTests
using DataFrames
using Plots
using BenchmarkTools
using StatsBase


# PDF Example
d = Normal(0,1)

x = [i for i in -5:0.01:5]
df = DataFrame(:x => x)
df[!,:pdf] = pdf.(d,x)

println(first(df,5))

p = Plots.plot(df.x, df.pdf, label="PDF")
# Plots.savefig(p,"pdf.png")


# CDF
df[!,:cdf] = cdf.(d,x)

p = Plots.plot(df.x, df.cdf, label="CDF")
# Plots.savefig(p,"cdf.png")

# Quick and dirty integration of the PDF
n=501
approxCDF = 0.0
for i in 1:n
    global approxCDF
    approxCDF += df.pdf[i]*0.01
end

println("CDF actual $(df.cdf[n]) vs calculated $approxCDF for F_x($(df.x[n]))")

#calculation of moments

#simulate based on the defined Distribution above, N(0,1)
#Expect  μ = 0, 
#      σ^2 = 1,
#     skew = 0,
#     kurt = 3 (excess = 0)

n = 1000
sim  = rand(d, n)

function first4Moments(sample)

    n = size(sample,1)

    #mean
    μ_hat = sum(sample)/n

    #remove the mean from the sample
    sim_corrected = sample .- μ_hat
    cm2 = sim_corrected'*sim_corrected/n

    #variance
    σ2_hat = sim_corrected'*sim_corrected/(n-1)

    #skew
    skew_hat = sum(sim_corrected.^3)/n/sqrt(cm2*cm2*cm2)

    #kurtosis
    kurt_hat = sum(sim_corrected.^4)/n/cm2^2

    excessKurt_hat = kurt_hat - 3

    return μ_hat, σ2_hat, skew_hat, excessKurt_hat
end

m, s2, sk, k = first4Moments(sim)

println("Mean $m ($(mean(sim)))")
println("Variance $s2 ($(var(sim)))")
println("Skew $sk ($(skewness(sim)))")
println("Kurtosis $k ($(kurtosis(sim)))")

println("mean diff = $(m - mean(sim))")
println("Variance diff = $(s2 - var(sim))")
println("Skewness diff = $(sk - skewness(sim))")
println("Kurtosis diff = $(k - kurtosis(sim))")


# Study the limiting expected values from the estimators
sample_size = 1000
samples = 100


means = Vector{Float64}(undef,samples)
vars = Vector{Float64}(undef,samples)
skews = Vector{Float64}(undef,samples)
kurts = Vector{Float64}(undef,samples)

Threads.@threads for i in 1:samples
    means[i], vars[i], skews[i], kurts[i] = first4Moments( rand(d, sample_size) )
end

println("Mean versus Expected $(mean(means) - mean(d))")
println("Variance versus Expected $(mean(vars) - var(d))")
println("Skewness versus Expected $(mean(skews) - skewness(d))")
println("Kurtosis versus Expected $(mean(kurts) - kurtosis(d))")


#########################################################################################
# Test the kurtosis function for bias in small sample sizes
d = Normal(0,1)
sample_size = 100
samples = 100
kurts = Vector{Float64}(undef,samples)
Threads.@threads for i in 1:samples
    kurts[i] = kurtosis(rand(d,sample_size))
end

#summary statistics
describe(kurts)

# t = mean(kurts)/sqrt(var(kurts)/samples)
# p = 2*(1 - cdf(TDist(samples-1),abs(t)))

# println("p-value - $p")

#using the Included TTest
ttest = OneSampleTTest(kurts,0.0)
p2 = pvalue(ttest)

# println("Match the stats package test?: $(p ≈ p2)") 
