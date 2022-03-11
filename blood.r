library(dplyr)

data <- read.csv("https://github.com/jgscott/SDS383D/raw/master/data/bloodpressure.csv")

#dumb t test
first_test <- t.test(data$systolic[data$treatment == 1], data$systolic[data$treatment == 2], 
alternative='greater')
print(first_test)
first_test$stder

#less dumb t test

pooled_df <- data %>% group_by(subject, treatment) %>% summarize(y_bar = mean(systolic))

scnd_test <- with(pooled_df, t.test(y_bar[treatment == 1], y_bar[treatment == 2],
    alternative = "greater"
))
print(scnd_test)
scnd_test$stder

### Gibbs sampler


## Initializations
niter <- 1000
p <- 20

theta <- kappa <- matrix(0, p, niter)
sigma2 <- tau2 <- mu <- beta <- rep(0, niter)

mu[1] <- 0
sigma2[1] <- tau2[1] <- beta <- 1

theta[, 1] <- rep(1, p)

x <- c(rep(0, 10), rep(1, 10))

students <- rep(0, p)
for (i in 1:p) {
    students[i] <- nrow(data[data$subject == i,])
    theta[i, 1] <- mean(data[data$subject == i,2])
}
mu[1] <- mean(theta[, 1])

for (i in 2:niter) {
    # mu
    mu[i] <- rnorm(1, (sum(theta[, i - 1])  - beta[i-1]*sum(x))/ p, sqrt(tau2[i - 1] * sigma2[i - 1] / p))

    # tau^2
    rate <- rep(0, p)
    for (j in 1:p) {
        rate[j] <- (theta[j, i - 1] - mu[i]-beta[i-1]*x[j])^2 / (2 * sigma2[i - 1])
    }
    tau2[i] <- MCMCpack::rinvgamma(1, (p + 1) / 2, sum(rate) + 0.5)
    
    # sigma^2
    # preallocating doesnt work so lets use horrible code
    suma <- 1
    suma2 <- rep(0, p)
    for (j in 1:p) {
        for (k in 1:students[j]) {
            yij <- data$systolic[which(data$subject == j)][k]
            suma <- suma + (yij - theta[j, i - 1])^2 / 2
        }
        suma2[j] <- (theta[j, i - 1] - mu[i]-beta[i-1]*x[j])^2 / (2 * tau2[i])
    }
    scale <- suma + sum(suma2)

    sigma2[i] <- MCMCpack::rinvgamma(1, (sum(students) + p) / 2, scale)

    # theta

    V <- 1 / (students/sigma2[i] + 1 / (tau2[i] * sigma2[i]))
    b <- rep(0, p)
    for (j in 1:p) {
        b[j] <- sum(data[data$subject == j, 2]) / sigma2[i] + (mu[i]+beta[i-1]*x[j]) / (tau2[i] * sigma2[i])
    }
    theta[, i] <- mvtnorm::rmvnorm(1, V * b, diag(V))

    # beta
    V <-  1 / (sum(x^2) / (tau2[i] * sigma2[i]) + 1 )
    temp_sum <- 0
    for (j in 1:p) {
        temp_sum <- temp_sum + (theta[j, i] - mu[i]) * x[j] / (tau2[i] * sigma2[i])
    }

    beta[i] <- rnorm(1, V * temp_sum, sqrt(V))
}

pdf("~/Desktop/mu.pdf")
plot(mu, type = "l", main = expression(mu))
dev.off()

pdf("~/Desktop/sigma.pdf")
plot(sigma2, type = "l", main = expression(sigma^2))
dev.off()


pdf("~/Desktop/tau.pdf")
plot(tau2, type = "l", main = expression(tau))
dev.off()

pdf("~/Desktop/beta.pdf")
plot(beta, type = "l", main = expression(beta))
dev.off()

pdf("~/Desktop/beta_hist_blood.pdf")
hist(beta, xlab= expression(beta))
dev.off()

# d
#Get the acfs
acfs <- acf(data$systolic[data$subject == 1], lag.max = 9)$acf[1:8,,1]
for (i in 2:p) {
    ACF <- acf(data$systolic[data$subject == i], lag.max=13)
    acfs <- cbind(acfs,ACF$acf[1:8,,1])
}
acfs <- reshape2::melt(as.data.frame(acfs))

library(ggplot2)
ggplot(acfs, aes(x=rep(1:8,20), y=value,group=variable, colour=variable)) +
 geom_line() + geom_hline(yintercept = -0.5) +
 geom_hline(yintercept = 0.5) +
 xlab("Lags") + ylab('ACF') +
 theme_minimal() + guides(colour='none')

ggsave('~/Desktop/acfs_blood.pdf')
