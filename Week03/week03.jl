using LinearAlgebra
using Distributions
using Random
using BenchmarkTools
using Plots
using DataFrames

# Cholesky that assumes PD matrix
function chol_pd!(root,a)
    n = size(a,1)
    #Initialize the root matrix with 0 values
    root .= 0.0

    #loop over columns
    for j in 1:n
        s = 0.0
        #if we are not on the first column, calculate the dot product of the preceeding row values.
        if j>1
            s =  root[j,1:(j-1)]'* root[j,1:(j-1)]
        end
  
        #Diagonal Element
        root[j,j] =  sqrt(a[j,j] .- s);

        ir = 1.0/root[j,j]
        #update off diagonal rows of the column
        for i in (j+1):n
            s = root[i,1:(j-1)]' * root[j,1:(j-1)]
            root[i,j] = (a[i,j] - s) * ir 
        end
    end
end


n=5
sigma = fill(0.9,(n,n))
for i in 1:n
    sigma[i,i]=1.0
end

root = Array{Float64,2}(undef,(n,n))

chol_pd!(root,sigma)

root*root' ≈ sigma

root2 = cholesky(sigma).L
root ≈ root2

#make the matrix PSD
sigma[1,2] = 1.0
sigma[2,1] = 1.0
eigvals(sigma)


chol_pd!(root,sigma)

#Cholesky that assumes PSD
function chol_psd!(root,a)
    n = size(a,1)
    #Initialize the root matrix with 0 values
    root .= 0.0

    #loop over columns
    for j in 1:n
        s = 0.0
        #if we are not on the first column, calculate the dot product of the preceeding row values.
        if j>1
            s =  root[j,1:(j-1)]'* root[j,1:(j-1)]
        end
  
        #Diagonal Element
        temp = a[j,j] .- s
        if 0 >= temp >= -1e-8
            temp = 0.0
        end
        root[j,j] =  sqrt(temp);

        #Check for the 0 eigan value.  The column will already be 0, move to 
        #next column
        if 0.0 != root[j,j]
            #update off diagonal rows of the column
            ir = 1.0/root[j,j]
            for i in (j+1):n
                s = root[i,1:(j-1)]' * root[j,1:(j-1)]
                root[i,j] = (a[i,j] - s) * ir 
            end
        end
    end
end


chol_psd!(root,sigma)

root*root' ≈ sigma

root2 = cholesky(sigma).L

#make the matrix slightly non-definite
sigma[1,2] = 0.7357
sigma[2,1] = 0.7357
eigvals(sigma)

chol_psd!(root,sigma)



#Generate some random numbers with missing values.
function generate_with_missing(n,m; pmiss=.25)
    x = Array{Union{Missing,Float64},2}(undef,n,m)

    for i in 1:n, j in 1:m
        if rand() >= pmiss
            x[i,j] = randn()
        end
    end
    return x
end

Random.seed!(2)
x = generate_with_missing(10,5,pmiss=.2)
cov(x)

#calculate either the covariance or correlation function when there are missing values
function missing_cov(x; skipMiss=true, fun=cov)
    n,m = size(x)
    nMiss = count.(ismissing, eachcol(x))
    #nothing missing, just calculate it.
    if 0==sum(nMiss)
        return fun(x)
    end

    idxMissing = Set.(findall.(ismissing,eachcol(x)))
    
    if skipMiss
        #Skipping Missing, get all the rows which have values and calculate the covariance
        rows = Set([i for i in 1:n])
        for c in 1:m
            for rm in idxMissing[c]
                delete!(rows,rm)
            end
        end
        rows = sort(collect(rows))
        return fun(x[rows,:])
    else
        #Pairwise, for each cell, calculate the covariance.
        out = Array{Float64,2}(undef,m,m)
        for i in 1:m
            for j in 1:i
                rows = Set([i for i in 1:n]) 
                for c in (i,j)
                    for rm in idxMissing[c]
                        delete!(rows,rm)
                    end
                end
                rows = sort(collect(rows))
                out[i,j] = fun(x[rows,[i,j]])[1,2]
                if i!=j
                    out[j,i] = out[i,j]
                end
            end
        end
        return out
    end
end

skipMiss = missing_cov(x)
pairwise = missing_cov(x,skipMiss=false)
eigvals(pairwise)


chol_psd!(root,skipMiss)
chol_psd!(root,pairwise)



#Look at Exponential Weights
weights = DataFrame()
cumulative_weights = DataFrame()
n=100
x = Vector{Float64}(undef,n)
w = Vector{Float64}(undef,n)
cumulative_w = Vector{Float64}(undef,n)

function populateWeights!(x,w,cw, λ)
    n = size(x,1)
    tw = 0.0
    for i in 1:n
        x[i] = i
        w[i] = (1-λ)*λ^i
        tw += w[i]
        cw[i] = tw
    end
    for i in 1:n
        w[i] = w[i]/tw
        cw[i] = cw[i]/tw
    end
end

#calculated weights λ=75%
populateWeights!(x,w,cumulative_w,0.75)
weights[!,:x] = copy(x)
weights[!,Symbol("λ=0.75")] = copy(w)
cumulative_weights[!,:x] = copy(x)
cumulative_weights[!,Symbol("λ=0.75")] = copy(cumulative_w)

#calculated weights λ=90%
populateWeights!(x,w,cumulative_w,0.90)
weights[!,Symbol("λ=0.90")] = copy(w)
cumulative_weights[!,Symbol("λ=0.90")] = copy(cumulative_w)

#calculated weights λ=97%
populateWeights!(x,w,cumulative_w,0.97)
weights[!,Symbol("λ=0.97")] = copy(w)
cumulative_weights[!,Symbol("λ=0.97")] = copy(cumulative_w)

#calculated weights λ=99%
populateWeights!(x,w,cumulative_w,0.99)
weights[!,Symbol("λ=0.99")] = copy(w)
cumulative_weights[!,Symbol("λ=0.99")] = copy(cumulative_w)



cnames = names(weights)
cnames = cnames[findall(x->x!="x",cnames)]

#plot Weights
plot(weights.x,Array(weights[:,cnames]), label=hcat(cnames...),title="Weights")

#plot the cumulative weights
plot(cumulative_weights.x,Array(cumulative_weights[:,cnames]), label=hcat(cnames...), legend=:bottomright, title="Cumulative Weights")







#Near PSD Matrix
function near_psd(a; epsilon=0.0)
    n = size(a,1)

    invSD = nothing
    out = copy(a)

    #calculate the correlation matrix if we got a covariance
    if count(x->x ≈ 1.0,diag(out)) != n
        invSD = diagm(1 ./ sqrt.(diag(out)))
        out = invSD * out * invSD
    end

    #SVD, update the eigen value and scale
    vals, vecs = eigen(out)
    vals = max.(vals,epsilon)
    T = 1 ./ (vecs .* vecs * vals)
    T = diagm(sqrt.(T))
    l = diagm(sqrt.(vals))
    B = T*vecs*l
    out = B*B'

    #Add back the variance
    if invSD !== nothing 
        invSD = diagm(1 ./ diag(invSD))
        out = invSD * out * invSD
    end
    return out
end

near_pairwise = near_psd(pairwise)

chol_psd!(root,near_pairwise)

#PCA
vals, vecs = eigen(near_pairwise)

tv = sum(vals)
#Keep values 2:5
vals = vals[3:5]
vecs = vecs[:,3:5]
B = vecs * diagm(sqrt.(vals))
r = (B * randn(3,100_000_000))'
cov(r)


function simulate_pca(a, nsim; nval=nothing)
    #Eigenvalue decomposition
    vals, vecs = eigen(a)

    #julia returns values lowest to highest, flip them and the vectors
    flip = [i for i in size(vals,1):-1:1]
    vals = vals[flip]
    vecs = vecs[:,flip]
    
    tv = sum(vals)

    posv = findall(x->x>=1e-8,vals)
    if nval !== nothing
        if nval < size(posv,1)
            posv = posv[1:nval]
        end
    end
    vals = vals[posv]

    vecs = vecs[:,posv]

    println("Simulating with $(size(posv,1)) PC Factors: $(sum(vals)/tv*100)% total variance explained")
    B = vecs*diagm(sqrt.(vals))

    m = size(vals,1)
    r = randn(m,nsim)

    (B*r)'
end


n=5
sigma = fill(0.9,(n,n))
for i in 1:n
    sigma[i,i]=1.0
end

sigma[1,2]=1
sigma[2,1]=1

v = diagm(fill(.5,n))
sigma = v*sigma*v

sim = simulate_pca(sigma,10000)
cov(sim)

sim = simulate_pca(sigma,10000; nval=3)
cov(sim)

sim = simulate_pca(sigma,10000; nval=2)
cov(sim)

