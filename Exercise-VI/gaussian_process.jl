using Distributions, Plots, Random, Distances, LinearAlgebra

## First Matern kernel
function matern(x, b, τ₁, τ₂)
    dist_mat = pairwise(Euclidean(), x', dims=2)
    τ₁ .* exp.(-0.5 .* (dist_mat ./ b) .^ 2) + τ₂ * I
end

## Matern with parameter 5/2
function matern2(x, b, τ₁, τ₂)
    dist_mat = pairwise(Euclidean(), x', dims=2)
    τ₁ .* (1 .+ sqrt(5) .* dist_mat ./ b + 5 .* dist_mat .^ 2 ./ (3 * b^2)) .* exp.(-sqrt(5) .* dist_mat ./ b) + τ₂ * I
end

# Intialize stuff
n = 5000
x_grid = sort(rand(n))

τ₂ = 10^(-3)
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