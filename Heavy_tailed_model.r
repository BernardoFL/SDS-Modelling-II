# Heavy tailed model

library(mvtnorm)

#Read the data and subset it
data <- read.csv("https://github.com/jgscott/SDS383D/raw/master/data/greenbuildings.csv")
data <- data[, c("Rent", "leasing_rate", "green_rating", "City_Market_Rent", "age", "class_a", "class_b")]

y <- data$Rent * data$leasing_rate / 100
X <- data[, -(1:2)]
X <- as.matrix(cbind(rep(1, 7820), X))
n <- dim(X)[1]; p <- dim(X)[2]


# Priors
K <- rep(0.11, p)
K <- diag(K)
m <- rep(1, p)
d <- 1
eta <- 1
Lambda <- diag(n)

# Posteriors
nu.star <- n + d
Lambda.star <- t(X) %*% Lambda %*% X + K
mu.star <- solve(Lambda.star) %*% (t(X) %*% Lambda %*% y + t(K) %*% m)
eta.star <- eta + t(y) %*% Lambda %*% y + t(m) %*% K %*% m - (t(y) %*% Lambda %*% X + t(m) %*% K) %*% solve(t(X) %*% Lambda %*% X + K) %*% t(t(y) %*% Lambda %*% X + t(m) %*% K)
Sigma.star <- drop(eta.star / nu.star) * solve(Lambda.star)


### Betas ###

betas <- rmvt(n = 1000, sigma = Sigma.star, df = nu.star, delta = mu.star)

###
### Obtain 95% Intervals
###

## Using our method
bayestestR::ci(betas[, 2], method = "HDI")
#[0.40, 2.29]

## Using lm() method
fit <- lm(y ~ 0 + X)
confint(fit)[2, ]

###
### Residual Analysis
###

res <- y - X %*% apply(betas, 2, mean)
png("hist.png")
hist(res, main = "Histogram of Model Residuals", breaks = 50, col = "black", border = "white")
dev.off()