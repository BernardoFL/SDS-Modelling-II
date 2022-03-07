library(mvtnorm)


data <-  read.csv("https://github.com/jgscott/SDS383D/raw/master/data/greenbuildings.csv")
data <- data[,c("Rent", "leasing_rate", "green_rating", "City_Market_Rent", "age", "class_a", "class_b")]

y <- data$Rent * data$leasing_rate / 100
X <- data[,-(1:2)]
X <- as.matrix(cbind(rep(1,7820), X))


K <- rep(0.001, 6)
K <- diag(K)
m <- rep(1, 6)
d <- 1
eta <- 1
Lambda <- diag(7820)

## Computations Using (c)
nu.star <- 7820 + d
Lambda.star <- t(X) %*% Lambda %*% X + K
mu.star <- solve(Lambda.star) %*% (t(X) %*% Lambda %*% y + t(K) %*% m)
eta.star <- eta + t(y) %*% Lambda %*% y + t(m) %*% K %*% m - (t(y) %*% Lambda %*% X + t(m) %*% K) %*% solve(t(X) %*% Lambda %*% X + K) %*% t(t(y) %*% Lambda %*% X + t(m) %*% K)
Sigma.star <- drop(eta.star / nu.star) * solve(Lambda.star)

# Monte Carlo bc I am lazy 

betas <- rmvt(10000, sigma=Sigma_star, df=7821, delta=mu_star)
#estimators
colMeans(betas)
hist(betas[,3], main="Green rating", xlab="Beta")
bayestestR::ci(betas[,2], method='HDI')
# [ -3.29e+05, 3.22e+05]
confint(lm(y ~ 0 + X))[2,]
#     2.5 %    97.5 % 
#   0.5447429 2.2996670 
# residuals

### Residuals
pdf('~/Desktop/residuales.pdf')
hist(y- X%*% colMeans(betas), main='Residuals', xlab='Residuals') 
dev.off()
