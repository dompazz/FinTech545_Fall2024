
# nsim = 100000
# prices = CSV.read("DailyPrices.csv",DataFrame)
# returns = return_calculate(prices;dateColumn="Date")
# # l = size(returns,1)
# meta = returns[!,"META"]

# ar1 = SARIMA(meta,order=(1,0,0),include_mean=true)

# StateSpaceModels.fit!(ar1)
# ar_sim = simulate_scenarios(ar1,1,nsim)[1,1,:]
# VaR5 = VaR(ar_sim)

# r = StateSpaceModels.get_innovations(ar1) / std(StateSpaceModels.get_innovations(ar1))
# s = ar1.results.coef_table.coef[3]

function ar1_simulation(y,coef_table,innovations)
    m = coef_table.coef[findfirst(r->r == "mean",coef_table.names)]
    a1 = coef_table.coef[findfirst(r->r == "ar_L1",coef_table.names)]
    s = sqrt(coef_table.coef[findfirst(r->r == "sigma2_η",coef_table.names)])

    l = length(y)
    n = length(innovations)

    out = fill(0.0,n)

    y_last = y[l] - m
    for i in 1:n
        out[i] = a1*y_last + innovations[i]*s + m
    end

    return out
end

# r2 = ar1_simulation(meta,ar1.results.coef_table,randn(nsim))

# VaR(r2)

# abs(VaR(ar_sim) - VaR(r2))