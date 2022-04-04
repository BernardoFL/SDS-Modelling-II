##### Cheese
library(ggplot2); library(invgamma); library(mvtnorm)
data <- read.csv('https://raw.githubusercontent.com/jgscott/SDS383D/master/data/cheese.csv')

data_p <- reshape2::melt(data, id=c('vol', 'price', 'disp'))
ggplot(data_p, aes(x=log(price), y=log(vol), group=as.factor(disp), colour=as.factor(disp))) +
 geom_point()  + theme_minimal()

ggplot(data_p, aes(x = value, y = log(vol), colour = as.factor(disp))) +
    geom_point() + theme_minimal()
 

## Transform X

names <- unique(data$store) # store names
n <- length(names) # number of stores
max_st <- names(which.max(table(data$store)))
maximo <- sum(data$store == max_st) # largest number of observations per store
Y <- matrix(0, n, maximo) # ij matrix
X <- array(0, dim = c(maximo, 4, n)) # n stores at end so we get subsetable matrix containing ; matches form of X_i in write-up
ni <- rep(0, n) # initialize vector to determine Ni

for (i in 1:n) { #iterate each sore
    tienda <- data[which(data$store == names[i]), ] # data set for 
    ni[i] <- dim(tienda)[1] # number of observations for store i
    Y[i, 1:(ni[i])] <- log(tienda$vol) # stores log(quantity)
    for (j in 1:ni[i]) {
        X[j,,i] <- c(1, log(tienda$price[j]), tienda$disp[j], log(tienda$price[j]) * tienda$disp[j])
    }
}

### Gibbs sampler

niter <- 20000
a <- b <- rep(0, niter)
sigma2 <- matrix(0, n, niter)
s2 <- mu_beta <- matrix(0, 4, niter)
beta <- array(0, dim = c(n, 4, niter))
#initial values 

a[1] <- 2
b[1] <- 2
sigma2[, 1] <- MCMCpack::rinvgamma(n, 0.5, 0.5)
s2[, 1] <- rep(1,4)

mu_beta[, 1] <- 0
for (i in 1:n) {
    beta[i, , 1] <- rep(0,2)
}

temp_var <- diag(s2[, 1])

#actual gibbs sampler
for (k in 2:niter) {

    ###
    ### beta
    ###

    for (i in 1:n) {
        Sigma_star <- solve(t(X[1:(ni[i]), , i]) %*% X[1:(ni[i]), , i] / sigma2[i, k - 1] + solve(temp_var))

        mu_star <- Sigma_star %*% (t(X[1:(ni[i]), , i]) %*% Y[i, 1:(ni[i])] / sigma2[i, k - 1] + solve(temp_var) %*% mu_beta[, k - 1])

        beta[i, , k] <- rmvnorm(1, mu_star, Sigma_star)
    }

    ###
    ### mu_beta
    ###

    mu_vec <- rep(0, 4)
    for (i in 1:4) {
        mu_vec[i] <- mean(beta[, i, k])
    }

    mu_beta[, k] <- rmvnorm(1, mu_vec, temp_var/n)

    ###
    ### s2
    ###

    for (p in 1:4) { #for each variable
        temp <- 0 #horrible code
        for (i in 1:n) {
            temp <- temp + (beta[i, p, k] - mu_beta[p, k])^2
        }
        s2[p, k] <- rinvgamma(1, n / 2 + 0.5, 0.5 * (1 + temp))
    }
    temp_var <- diag(s2[, k])

    ### Same horrible code for updating sigma2

    for (i in 1:n) {
        temp <- matrix(Y[i, 1:ni[i]]) - X[1:ni[i], , i] %*% beta[i, , k]

        sigma2[i, k] <- rinvgamma(1, a / 2 + ni[i] / 2,   (b + t(temp) %*% temp)/2)
    }

}


## plots

estim_sigma <- rowMeans(sigma2[,5001:20000])

pdf('~/Desktop/varianzas_ex5.pdf')
plot(estim_sigma, pch=19, main="Variance of stores", xlab="Store", ylab=expression(sigma^2))
grid()
dev.off()

#betas

pdf('~/Desktop/hist_beta_ex5.pdf')
par(mfrow = c(2, 2))
for(p in 1:4){
    temp_beta <- beta[,p,5001:20000]
    temp_beta <- rowMeans(temp_beta)

    hist(temp_beta, main=paste0('Histogram for ', 'beta_', p), xlab=expression(beta))
}
dev.off()
