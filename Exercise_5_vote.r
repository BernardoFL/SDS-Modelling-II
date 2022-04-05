########## Voting polls


library(LaplacesDemon)
library(dplyr)
library(truncnorm)
library(mvtnorm)
library(progress)
set.seed(312)

data <- read.csv("https://raw.githubusercontent.com/jgscott/SDS383D/master/data/polls.csv")


### Data wrangling


# remove NAs
data <- data %>% select(-c(org, year, survey, weight)) %>% na.omit()

# set values for categorical variables
data <- data %>% mutate(Age = as.integer(factor(data$age))) %>% select(-age)
data <- data %>% mutate(Edu = as.integer(factor(data$edu, levels = c("NoHS", "HS", "IncColl", "BsC")))) %>% select(-edu)


names <- unique(data$state)
n <- length(names)

maximo <- sum(data$state == names(which.max(table(data$state)))) # largest number of observations per store
P <- 4 

Y <- matrix(NA, n, maximo)
X <- array(NA, dim = c(maximo, P+1, n))
ni <- rep(0, n)

for(i in 1:n){
    temp <- data[which(data$state==names[i]),] # data set for current state
    ni[i] <- dim(temp)[1] # number of observations for store i
    Y[i, 1:(ni[i])] <- temp$bush # states voters
    for(j in 1:ni[i]){
        X[j, , i] <- c(1, as.numeric(temp[j, 3:6]))
    }
}

###
### Set Starting Values
###


Z_start <- matrix(NA, n, maximo)
for(i in 1:n){
    for(j in 1:ni[i]){
        Z_start[i, j] <- ifelse(Y[i, j] == 1, rtruncnorm(1, a = 0, mean = 0, sd = 1), rtruncnorm(1, b = 0, mean = 0, sd = 1))
    }
}


n.mcmc <- 20000
beta <- array(NA, dim = c(P+1, n, n.mcmc))
nu <- array(NA, dim = n.mcmc)
Z <- array(NA, dim = c(maximo, n, n.mcmc))
lambdas <- array(NA, dim = c(n, n.mcmc))

beta[, , 1] <- rep(1,P+1)
nu[1] <- 1
Z[, , 1] <- Z_start
lambdas[, 1] <- rep(1, n)

#funciotn that returns the unnormalized density for post_nu
post_nu <- function(nu){
    c_nu <- (gamma(nu/2)*(nu/2)^(nu/2)^(-1) 
    mat <- sapply(1:n, function(i) (c_nu*lambdas[i]^(nu/2-1)*exp(-nu*lambdas[i]/2))^(ni[i]))
    return(prod(mat))
}

###
### MCMC
###
pb <- progress_bar$new(total = n.mcmc)
for(k in 2:n.mcmc){
pb$tick()

    ###
    ### Update beta
    ###

    W <- diag(lambdas[,,k-1])
        for (i in 1:n) {
            beta[i, , k] <- rtruncnorm(1, a = 0, mean = X[1:ni[i], , i] %*% beta[, k-1], sd = lambdas[i, k - 1]^(-0.5))
        }
    beta_hat <- solve(t(X) %*% W %*% X)%*% t(X) %*%W %*% Z[,,k-1]
    beta[, , k] <- rmvnorm(1, beta_hat, W, checkSymmetry = FALSE) 
    

    ###
    ### Update Z
    ###

    # i are the observations per store, j the store and k is the mcmc iteration
    for(i in 1:n){
        for(j in 1:ni[i]){
            if(Y[i, j] == 1){
                Z.save[i, j, k] <- rtruncnorm(1, a = 0, mean = X[j, , i]%*%beta[, , k], sd = lambdas[,i,k-1]^(-0.5))
            }
            if(Y[i, j] == 0){
                Z.save[i, j, k] <- rtruncnorm(1, b = 0, mean = X[j, , i] %*% beta[, , k], sd = lambdas[, i, k - 1]^(-0.5))
            }
        }
    }

    ###
    ### Update lambdas
    ###

    for(i in 1:n){
        rgamma(1, (nu+1)/2, 2/(nu+(Z[])^2))
        A.inv <- solve( solve(B.star) + solve(10^4*diag(P+1))  ) 
        tmp.mean <- A.inv%*%( solve(B.star)%*%alpha.save[, i, k] )
        beta.save[, i, k] <- rmvnorm(1, tmp.mean, A.inv)
    }

    ###
    ### Update nu
    ###

    #sample over a grid

    grid <- seq(1,10,by=0.1)
    nu[k] <- sample(grid, 1, prob = sapply(grid, post_nu))
    

}

###
### Trace Plots for alphas for each state
###

n.burn <- .3*n.mcmc
for(i in 1:n){
    tmp.df <- data.frame(iter = n.burn:n.mcmc, mu = alpha.save[1, i, n.burn:n.mcmc], beta1 = alpha.save[2, i, n.burn:n.mcmc], beta2 = alpha.save[3, i, n.burn:n.mcmc], beta3 = alpha.save[4, i, n.burn:n.mcmc], beta4 = alpha.save[5, i, n.burn:n.mcmc])
    plot.df <- tmp.df %>% gather(key = "variable", value = "value", -1)
    traces <- ggplot(plot.df, aes(x = iter, y = value)) + geom_line(aes(color = variable)) + theme_classic() + ggtitle("Trace Plots") + xlab("Iteration") + ylab("Value")
    assign(paste0("t", i), traces)
}

#Plots
#Asked Michael for help bc i couldn't use ggplot :(, not really sure how everything works in here 


tmp.mean <- rep(0, n)

p <- 1
ind <- rep(0, n)
for (i in 1:n) {
    tmp.mean[i] <- mean(alpha.save[p, i, n.burn:n.mcmc])
    if (abs(tmp.mean[i]) < 10) {
        ind[i] <- 1 
    }
}
tmp.names <- names[which(ind == 1)]
tmp.mean <- tmp.mean[which(ind == 1)]
tmp.df <- data.frame(State = tmp.names, beta = tmp.mean)
pos.points <- data.frame(State = tmp.names[which(tmp.mean > 0)], beta = tmp.mean[which(tmp.mean > 0)])
neg.points <- data.frame(State = tmp.names[which(tmp.mean <= 0)], beta = tmp.mean[which(tmp.mean <= 0)])
tmp.plot <- ggplot(tmp.df, aes(x = State)) +
    geom_point(aes(y = beta)) +
    theme_classic() +
    ylab(bquote(mu)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_segment(pos.points, mapping = aes(x = State, xend = State, y = 0, yend = beta), color = "red") +
    geom_segment(neg.points, mapping = aes(x = State, xend = State, y = 0, yend = beta), color = "blue")
assign(paste0("p", p), tmp.plot) + theme(axis.text.x = element_text(angle = 90))

for (p in 2:(P + 1)) { # for each parameter
    ind <- rep(0, n)
    for (i in 1:n) {
        tmp.mean[i] <- mean(alpha.save[p, i, n.burn:n.mcmc]) # obtain posterior mean for each state
        if (abs(tmp.mean[i]) < 10) {
            ind[i] <- 1 # we will plot this one
        }
    }
    tmp.names <- names[which(ind == 1)]
    tmp.mean <- tmp.mean[which(ind == 1)]
    tmp.df <- data.frame(State = tmp.names, beta = tmp.mean)
    pos.points <- data.frame(State = tmp.names[which(tmp.mean > 0)], beta = tmp.mean[which(tmp.mean > 0)])
    neg.points <- data.frame(State = tmp.names[which(tmp.mean <= 0)], beta = tmp.mean[which(tmp.mean <= 0)])
    tmp.plot <- ggplot(tmp.df, aes(x = State)) +
        geom_point(aes(y = beta)) +
        theme_minimal() +
        ylab(paste0('beta_',p)) +
        geom_hline(yintercept = 0, linetype = "dashed") +
        geom_segment(pos.points, mapping = aes(x = State, xend = State, y = 0, yend = beta), color = "red") +
        geom_segment(neg.points, mapping = aes(x = State, xend = State, y = 0, yend = beta), color = "blue") 
    assign(paste0("p", p), tmp.plot)
}
plots <- ggarrange( p2, p3, p4, p5, p6, p7, p8, p9, nrow = 3, ncol = 3) + ggtitle("Posterior Means of Coefficients Per State")
ggsave("~/Desktop/coefficient_estimates.png", plots)
