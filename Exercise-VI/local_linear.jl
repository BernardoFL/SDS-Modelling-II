# Fitting Local Linear Estimators with a Gaussian Kernel

using DataFrames, StatsPlots, LinearAlgebra, Random, Distributions, Plots, Loess
import CSV


### Get data


data = CSV.read(download("https://github.com/jgscott/SDS383D/raw/master/data/utilities.csv"), DataFrame)


Y = data.gasbill ./ data.billingdays
X = data.temp
H = X * inv(X' * X) * X'
n = length(X)

###
### Functions
###

function gaussian_kernel(x)
    exp(-x^2 / 2) / sqrt(2 * pi)
end

## s_j. Horrible code
function get_sj(x_new, h, j)
    sum = 0
    for i in 1:n
        int = (x_new - X[i]) / h
        sum += gaussian_kernel(int) * (X[i] - x_new)^j
    end
    return (sum)
end

## Weight Function
function get_weights(x_new, x_old, h)
    x_ = (x_new .- x_old) ./ h
    s1 = get_sj(x_new, h, 1)
    s2 = get_sj(x_new, h, 2)
    return gaussian_kernel(x_) * (s2 - (x_old - x_new) * s1)
end


## e) Fit Local Linear Estimators


H_list = LinRange(0.1,7, 20) #list of hs

LOOCV = zeros(length(H_list)) # initialize average squared error in prediction (LOOCV) matrix

M = 1000 # number of "new" points to fit kernel-estimator
grid = LinRange(minimum(X), maximum(X), M)  # new points
Y_matrix = zeros(M, length(H_list)) # save estimated y for plotting

## For each value of h
X = convert(Vector{Float64}, X)

for k in 1:length(H_list)

    h = H_list[k] # pick h
    smooth_y = repeat([0.0], M)

    ## Fit kernel-regression estimator to training data
    for j in 1:M # for each value in grid
        num = 0
        den = 0 # sums
        for i in 1:n# for each observation
            num += get_weights(grid[j], X[i], h) * Y[i]
            den += get_weights(grid[j], X[i], h)
        end
        smooth_y[j] = num / den
    end

    # Piecewise linear approximation
    interpola = loess(grid, smooth_y)
    # get LOOCV 

    temp_error = 0
    for i in 1:n # for each observation
        temp_num = Y[i] - Loess.predict(interpola, X[i])
        temp_den = 1 - H[i, i]
        temp_error += (temp_num / temp_den)^2
    end
    LOOCV[k] = temp_error / n

    Y_matrix[:, k] = smooth_y
end

## Prepare Data for Plotting

Y_matrix = DataFrame(Y_matrix, :auto)
rename!(Y_matrix, ["h = " * string(round(H_list[i]; sigdigits=3)) for i in 1:length(H_list)])
Y_matrix[!, :grid] = grid

@df Y_matrix plot(:grid, cols(1:20))
savefig("~/Desktop/local_lin.png")


## Fit data using new function
interpola = loess(grid, Y_matrix[!, argmin(LOOCV)])

ŷ = Loess.predict(interpola, X)

## plot
plot_df = DataFrame(X=X, ŷ=ŷ)
plot_df = plot_df[sortperm(plot_df[:, 1]), :]

scatter(X, Y, label="Sample")
plot!(plot_df.X, plot_df.ŷ, linewidth=2, label="Fitted curve")
savefig("~/Desktop/fitted.png")

## f)  Residuals

ε = Y .- ŷ
scatter(X, ε, legend=false)
savefig("~/Desktop/residuals_local.png")


# g) Construct an Approximate Point-wise 95% CI

## Obtain confidence intervals

s2 = sum(ε .^ 2) / (n - 2 * tr(H) + tr(H' * H))
inter = 1.96 * sqrt(s2)


## Plot
plot_df = DataFrame(X=X, ŷ=ŷ, inter=inter)
plot_df = plot_df[sortperm(plot_df[:, 1]), :]
scatter(X, Y, label="Sample", ylim=(-1, 9.3))
plot!(plot_df.X, plot_df.ŷ, linewidth=2, label="Fitted curve"; ribbon=plot_df.inter)
savefig("~/Desktop/fitted_confint.png")

