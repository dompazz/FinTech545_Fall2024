function return_calculate(prices::DataFrame;
    method="DISCRETE",
    dateColumn="date")

    vars = names(prices)
    nVars = length(vars)
    vars = Symbol.(vars[vars.!=dateColumn])
    if nVars == length(vars)
        throw(ArgumentError(string("dateColumn: ", dateColumn, " not in DataFrame: ",vars)))
    end
    nVars = nVars-1

    p = Matrix(prices[!,vars])
    n = size(p,1)
    m = size(p,2)
    p2 = Array{Float64,2}(undef,n-1,m)
    # p = p[1:(n-1),:]

    @inbounds @avx for i ∈ 1:(n-1), j ∈ 1:m
        p2[i,j] = p[i+1,j] / p[i,j]
    end

    if uppercase(method) == "DISCRETE"
        p2 = p2 .- 1.0
    elseif uppercase(method) == "LOG"
        p2 = log.(p2)
    else
        throw(ArgumentError(string("method: ", method, " must be in (\"LOG\",\"DISCRETE\")")))
    end

    dates = prices[2:n,Symbol(dateColumn)]
    out = DataFrame(Symbol(dateColumn)=>dates)
    for i in 1:nVars
        out[!,vars[i]] = p2[:,i]
    end
    return(out)
end