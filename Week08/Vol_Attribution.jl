
using LinearAlgebra
using Distributions

#simple volatility Attribution example
corel = fill(0.9,(3,3))
for i in 1:3
    corel[i,i] = 1.0 
end

sd = [ .01, .02, .03]

#Final Asset Covariance Matrix
covar = diagm(sd)*corel*diagm(sd)

#Portfolio Weights
w = [.4, .35, .25]

#True Portfolio Standard Deviation
pStd = sqrt(w'*covar*w)

#True Portfolio volatility Attribution
csd = w.*(covar*w)./pStd
for i in 1:3
    println("Asset $i Vol Contribution: $(csd[i])")
end
println("Test if attribution sums correctly: $(sum(csd) ≈ pStd)")


#Simulate 1000 Asset Return
n = 1_000
d = MvNormal(fill(0.,3), covar)
sim = rand(d,n)'

#Portfolio Returns
ret = sim*w

println("Portfolio Vol: $(std(ret))")
println("Actual P Vol: $pStd")

#Vol Attribution
X = hcat(fill(1.0,n),ret)
Y = sim .* w'
B_s_p = (inv(X'*X)*X'*Y)[2,:]
cSD2 = B_s_p * std(ret)

for i in 1:3
    println("Asset $i Vol Contribution: $(cSD2[i])")
end

println("Test if attribution sums correctly: $(sum(cSD2) ≈ std(ret))")

