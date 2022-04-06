using Plots, Random, Statistics, Distributions, StatsBase, Distances, Loess
Random.seed!(122)

function f_true(x)::Float32
    x + cos(x^2)
end


#kernel

function gaussian_kernel(x)
    exp(-x^2 / 2) / sqrt(2 * pi)
end

# weighting function 
function weights(x_old, x_new, h)
    sumando = (x_old - x_new) / h
    return gaussian_kernel(sumando) / h
end

#kernel smoothing
function kernel_smoothing(red, x, y, h)
    temp_y = zeros(length(red))

    for j in 1:length(y)
        #get the weights
        pesos = zeros(length(y))
        for i in 1:length(y)
            pesos[i] = weights(x[i], red[j], h)
        end
        pesos = pesos ./ sum(pesos) #normalize the weights

        temp_sum = 0
        for i in 1:length(x)
            temp_sum += pesos[i] * y[i]
        end
        temp_y[j] = temp_sum
    end
    return temp_y
end


###### Fitting kernel

# Simulate data
x = collect((-5):0.1:5) .+ rand(Normal(), 101)
y = [f_true(x_) + rand(Normal()) for x_ in x]

## Center data
x_cent = [(x[i] - mean(x)) / std(x) for i in 1:length(x)]
y_cent = [(y[i] - mean(y)) / std(y) for i in 1:length(y)]

#different bandwithds
hs = [0.1, 0.5, 1, 2]

#arrays for storing results
y_hat = zeros(length(x), length(hs) + 1)
y_hat[:, 1] = map(f_true, x)

#perform the smoothing, it doesn't work for the centeered things for some reason
red = collect((-5):0.1:5)
for k in 1:length(hs)
    y_hat[:, k+1] = kernel_smoothing(red, x, y, hs[k])
end

#Plotting the results
plot(f_true, -5, 5, label="Truth", legend=:bottomright)
for k in 1:length(hs)
    h = hs[k]
    plot!(red, y_hat[:, k+1], label="h = $h", linestyle=:dot, linewidth=2)
end
current() #this shows the plots on a foor loop
savefig("~/Desktop/kernel.pdf")

#############
### Cross validation
############


# - train: n x 3 array, the first column is the train x, the second the x used to estimate and the third column y
# - test: array of size n with the new x
# return: dictionary with the optimum h and the estimated ys
function optimize_h_cv(train, test, f_true)
    hs = collect(0.1:0.05:4)
    losses = zeros(length(hs))
    y_hat = zeros(size(train)[1], length(hs))
    for i in 1:length(hs)
        y_hat[:, i] = kernel_smoothing(train[:, 1], train[:, 2], train[:, 3], hs[i])
        #obtain functional expression using loess
        spl = loess(train[:, 2], y_hat[:, i], span=0.5)
        losses[i] = euclidean(map(f_true, test), Loess.predict(spl, test))
    end

    # Stack the h and the predicted values and sort the columns so that the first one is the one with the lowest h
    return vcat(hs', losses', y_hat)[:, sortperm(losses)]
end

#used 30 observations as test [:, sortperm(losses)]
train = hcat(red, x, y)
test = collect(LinRange(-5.1, 4.9, 30))

best_h = optimize_h_cv(train, test, f_true)

plot(f_true, -5, 5, legend=false)
plot!(red, best_h[3:103, 1], linestyle=:dot, linewidth=4)
for k in 2:(size(best_h)[2])
    plot!(red, best_h[3:103, k], linewidth=0.5, alpha=0.4)
end
current() #this shows the plots on a foor loop
savefig("~/Desktop/kernel_cv.pdf")


######
## Fuzzy-stuff

### Get new ys. Repeat same as before, save the best, plot together with points.
y_11 = [f_true(x_) + rand(Normal(0, 0.3)) for x_ in x]
y_12 = [f_true(x_) + rand(Normal(0, 3)) for x_ in x]

y_21 = [cos(5 * x_) + x_ + rand(Normal(0, 0.3)) for x_ in x]
y_22 = [cos(5 * x_) + x_ + rand(Normal(0, 3)) for x_ in x]

## Reuse everything possible :)
#11
train_11 = hcat(red, x, y_11)
best_h_11 = optimize_h_cv(train_11, test, f_true)

#12
train_12 = hcat(red, x, y_12)
best_h_12 = optimize_h_cv(train_12, test, f_true)

#21
train_21 = hcat(red, x, y_21)
best_h_21 = optimize_h_cv(train, test, x -> cos(5 * x) + x)

#22
train_22 = hcat(red, x, y_22)
best_h_22 = optimize_h_cv(train, test, x -> cos(5 * x) + x)
###### Plots

plot(f_true, -5, 5, legend=false, linewidth=2)
plot!(red, y_11, alpha=0.3, seriestype=:scatter)
plot!(red, best_h_11[3:103, 1], linestyle=:dashdot, linewidth=2)
savefig("~/Desktop/f11.pdf")

plot(f_true, -5, 5, legend=false, linewidth=2)
plot!(red, y_12, alpha=0.3, seriestype=:scatter)
plot!(red, best_h_12[3:103, 1], linestyle=:dashdot, linewidth=2)
savefig("~/Desktop/f12.pdf")

plot(x -> cos(5 * x) + x, -5, 5, legend=false, linewidth=2)
plot!(red, y_21, alpha=0.3, seriestype=:scatter)
plot!(red, best_h_21[3:103, 1], linestyle=:dashdot, linewidth=2)
savefig("~/Desktop/f21.pdf")

plot(x -> cos(5 * x) + x, -5, 5, legend=false, linewidth=2)
plot!(red, y_22, alpha=0.3, seriestype=:scatter)
plot!(red, best_h_22[3:103, 1], linestyle=:dashdot, linewidth=2)
savefig("~/Desktop/f22.pdf")
## Leave-one-out

### Put the thing in a for loop with the leave one out.