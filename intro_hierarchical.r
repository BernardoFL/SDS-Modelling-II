library(ggplot2)

data <- read.csv("https://github.com/jgscott/SDS383D/raw/master/data/mathtest.csv")
data$school <- as.factor(data$school)

df <- data.frame(n = as.numeric(table(data$school)),
        mean_score = tapply(data$mathscore, data$school, mean) )

## number of students vs average score
ggplot(df, aes(x=n, y=mean_score)) + geom_point() +
 geom_smooth()+ theme_minimal()
ggsave("~/Desktop/n_vs_mean.pdf")

#### Gibbs sampler


## Initializations
niter <- 1000
p <- length(levels(data$school))

theta <- kappa <- matrix(0, p, niter)
sigma2 <- tau2 <- mu<- rep(0, niter)

mu[1] <- 0
sigma2[1] <- tau2[1] <- 1

theta[, 1] <- rep(1, p)
#get initial kappa
kappa[, 1] <- tau2[1] * df$n / (tau2[1] * df$n + 1)


for (i in 2:niter) {
        #mu
        mu[i] <- rnorm(1, sum(theta[, i - 1]) / p, sqrt(tau2[i-1] * sigma2[i-1] / p))

        #tau^2
        rate <- rep(0, p)
        for (j in 1:p){
                rate[j] <- (theta[j, i-1] - mu[i])^2 / (2 * sigma2[i-1])
        }
        tau2[i] <- MCMCpack::rinvgamma(1, (p+1)/2, sum(rate) + 0.5)

        #sigma^2
        #preallocating doesnt work so lets use horrible code
        suma <- 1
        suma2 <- rep(0, p)
        for (j in 1:p) {
                for (k in 1:df$n[j]) {
                        yij <- data$mathscore[which(data$school == j)[k]]
                        suma <- suma + (yij-theta[j, i-1])^2 / 2
                }
                suma2[j] <- (theta[j, i-1] - mu[i])^2 / (2 * tau2[i])
        }
        scale <- suma + sum(suma2)

        sigma2[i] <- MCMCpack::rinvgamma(1, (sum(df$n) + p)/2, scale)

        #theta

        V <- 1 / (df$n / sigma2[i] + 1 / (tau2[i] * sigma2[i]))
        b <- rep(0, p)
        for (j in 1:p) {
                b[j] <- sum(data[data$school == i, 2]) / sigma2[i] + mu[i] / (tau2[i] * sigma2[i])
        }
        theta[, i] <- mvtnorm::rmvnorm(1, V*b, diag(V))

        #get kappa

        kappa[, i] <- tau2[i] * df$n / (tau2[i] * df$n + 1)
}

pdf("~/Desktop/mu.pdf")
plot(mu, type = "l", main = expression(mu^2))
dev.off()

pdf("~/Desktop/sigma.pdf")
plot(sigma2, type = "l", main = expression(sigma^2))
dev.off()


pdf("~/Desktop/tau.pdf")
plot(tau2, type = "l", main = expression(tau^2))
dev.off()


##### Kappas

post_kappa <- apply(kappa, 1, mean)


pdf("~/Desktop/kappa.pdf")
plot(df$n, post_kappa, pch = 20, xlab = "n", ylab = expression(kappa))
dev.off()




