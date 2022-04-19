using Distributions, Plots, Random, Distances, LinearAlgebra

## First Matern kernel
function matern(x, b, τ₁, τ₂)
    dist_mat = pairwise(Euclidean(), x', dims=2)
    Symmetric(τ₁ .* exp.(-0.5 .* (dist_mat ./ b) .^ 2) + τ₂ * I)
end

## Matern with parameter 5/2
function matern2(x, b, τ₁, τ₂)
    dist_mat = pairwise(Euclidean(), x', dims=2)
    τ₁ .* (1 .+ sqrt(5) .* dist_mat ./ b + 5 .* dist_mat .^ 2 ./ (3 * b^2)) .* exp.(-sqrt(5) .* dist_mat ./ b) + τ₂ * I
end

# Intialize stuff
n = 2000
x_grid = sort(rand(n))

τ₂ = 10^(-8)
b = LinRange(0.0001, 1, 6)
τ₁ = LinRange(0, 2, 10)

Y = zeros(n, length(τ₁), length(b))

for i in 1:length(b) # across different values of b
    for j in 1:length(τ₁) # across different values of tau1^2
        temp_sigma = matern(x_grid, b[i], τ₁[j], τ₂)
        Y[:, j, i] = rand(MultivariateNormal(repeat([0], n), Symmetric(temp_sigma))) #forced it
    end
end


plot(x_grid, Y[:, :, 1], legend=false)
savefig("~/Desktop/tau1.pdf")
plot(x_grid, Y[:, :, 2], legend=false)
savefig("~/Desktop/tau2.pdf")
plot(x_grid, Y[:, :, 3], legend=false)
savefig("~/Desktop/tau3.pdf")
plot(x_grid, Y[:, :, 4], legend=false)
savefig("~/Desktop/tau4.pdf")
plot(x_grid, Y[:, :, 5], legend=false)
savefig("~/Desktop/tau5.pdf")
plot(x_grid, Y[:, :, 6], legend=false)
savefig("~/Desktop/tau6.pdf")

##### Matern 2

for i in 1:length(b) # across different values of b
    for j in 1:length(τ₁) # across different values of tau1^2
        temp_sigma = matern2(x_grid, b[i], τ₁[j], τ₂)
        Y[:, j, i] = rand(MultivariateNormal(repeat([0], n), Symmetric(temp_sigma))) #forced it
    end
end

plot(x_grid, Y[:, :, 1], legend=false)
savefig("~/Desktop/tau1_2.pdf")
plot(x_grid, Y[:, :, 2], legend=false)
savefig("~/Desktop/tau2_2.pdf")
plot(x_grid, Y[:, :, 3], legend=false)
savefig("~/Desktop/tau3_2.pdf")
plot(x_grid, Y[:, :, 4], legend=false)
savefig("~/Desktop/tau4_2.pdf")
plot(x_grid, Y[:, :, 5], legend=false)
savefig("~/Desktop/tau5_2.pdf")
plot(x_grid, Y[:, :, 6], legend=false)
savefig("~/Desktop/tau6_2.pdf")


########### b

###
### Gaussian Processes - An Application to Utilities
###

using CSV, DataFrames
BLAS.set_num_threads(1) #weird mac error

## Get Data
data = CSV.read(download("https://github.com/jgscott/SDS383D/raw/master/data/utilities.csv"), DataFrame)
y = data.gasbill ./ data.billingdays
x = data.temp

n = length(x)

## Hyperparameters
τ₂ = 10^(-6) # suggested by James
b = [3, 10, 15]
τ₁ = [1, 5, 10]

s2 = 0.61 # MAP

###
### Implementation
###

for k in 1:length(τ₁)
    for j in 1:length(b)

        sigma = matern(x, b[j], τ₁[k], τ₂)

        post_mean = inv(I + s2 * inv(sigma)) * y
        diff_sigma = inv(I ./ s2 + inv(sigma))

        plot_df = DataFrame(x=x, ŷ=post_mean, inter=1.96 .* sqrt.(diag(diff_sigma)))
        plot_df = plot_df[sortperm(plot_df[:, 1]), :]
        pl = plot(x, y, seriestype=:scatter, legend=false)
        plot!(pl, plot_df.x, plot_df.ŷ; ribbon=plot_df.inter)

        savefig("~/Desktop/utilities_$k_$j.pdf")

    end
end

for k in 1:length(τ₁)
    for j in 1:length(b)
        plot(graficas[k,j])
         savefig("~/Desktop/utilities_$k$j.pdf")
    end
end         

# e) find optimal hyperparameters
###

###
### Obtain Optimal Hyperparameters
###

M = 500 # number of points in each grid
tau1_grid = LinRange(0.001, 100, M)
b_grid = LinRange(0.0001, 100, M)

function margin_loglike(t1, b)
    C = matern(x, b, t1, τ₂) # get C
    sigma = C + s2 * I
    -0.5 * y' * inv(sigma) * y - 0.5 * log(det(sigma)) - n / 2 * log(2 * pi)
end

valores = zeros(length(tau1_grid), length(b_grid))
for i in 1:length(tau1_grid)
    for j in 1:length(b_grid)
        valores[i, j] = margin_loglike(tau1_grid[i], b_grid[j])
    end
end    

max_coord = findmax(valores)[2]

τ̂₁ = tau1_grid[max_coord[1]]
b̂ = b_grid[max_coord[2]]

## posterior mean for f|y 

sigma = matern(x, b̂, τ̂₁, τ₂)

post_mean = inv(I + s2 * inv(sigma)) * y
diff_sigma = inv(I ./ s2 + inv(sigma))

plot_df = DataFrame(x=x, ŷ=post_mean, inter=1.96 .* sqrt.(diag(diff_sigma)))
plot_df = plot_df[sortperm(plot_df[:, 1]), :]
pl = plot(x, y, seriestype=:scatter, legend=false)
plot!(pl, plot_df.x, plot_df.ŷ; ribbon=plot_df.inter)
savefig("~/Desktop/utilities_optim.pdf")