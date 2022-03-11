
# Get data 
data <- read.csv("https://github.com/jgscott/SDS383D/raw/master/data/wdbc.csv", header = FALSE)

y <- as.numeric(data[,2] == "M")
X <- as.matrix(data[, 3:12])
X <- apply(X, 2, scale)
X <- cbind(rep(1,nrow(data)), X)
p <- 11

#Loglikelihood
loglikelihood <- function(y, theta){
    sum(y*theta - log(1+exp(theta)))
}

# line search
## Function to find optimal step (via line search)
f_gamma <- function(gamma){
  theta <- X%*%(beta+gamma*mini_g)
  return(-sum(y*theta - log(1+exp(theta))))
}

## Initializations
beta_iter <- beta <- matrix(rep(0.1,11),ncol=1)
ll_iter <- c()

conv <- F
ll_prev <- -10000
tol <- 1e-3
## Gradient Descent Loop
while (!conv){

    #Can't make it work so let's just use what Michael told me to do
  mini_g <- 0
  for (i in 1:nrow(X)){
    mini_g <- mini_g + (y[i] - exp(sum(X[i,]*beta))/(1+exp(sum(X[i,]*beta))))*X[i,]
  }

    #transpose mini_g
  mini_g <- matrix(mini_g,ncol=1)
  gamma <- optimize(f_gamma, c(1e-6, 0.3))$minimum
  #update with linesearch
  beta <- beta + gamma*mini_g
  beta_iter <- cbind(beta_iter,beta)
  theta <- X%*%beta
  ll <- loglikelihood(y,theta)
  ll_iter <- c(ll_iter,ll)
  if (abs(ll_prev- ll) < tol){
    conv <- T
  }
  ll_prev <- ll
}


# plot loglikelihoods


pdf("gradientdescent.pdf")
plot(ll_iter, type = "l", main = "Loglikelihood", ylab = "Loglikelihood", ylim=c(-90,-72))
abline(h = logLik(glm(y ~ X, family = binomial())), col = "red")
legend("bottomright", legend = c("Gradient descent", "glm()"), lty = 1, col = c("black", "red"))
dev.off()


##################### Newton-Raphson

## Reset the vectors
beta_iter <- beta <- matrix(rep(0.1, 11), ncol = 1)
theta <- X %*% beta
ll_iter <- c()

converged <- FALSE
ll_prev <- -10000

###
### Newton-Raphson Method
###

while (!converged) {

    #gradient
    #again, horrible code (sorry Khai)
    g <- 0
    for (i in 1:nrow(X)) {
        g <- g + (y[i] - exp(sum(X[i, ] * beta)) / (1 + exp(sum(X[i, ] * beta)))) * X[i, ]
    }
 
    g <- matrix(g, ncol = 1)

    # Hessian
    W <- exp(theta) / (1 + exp(theta))^2
    W <- diag(as.vector(W)) 
    H <- -t(X) %*% W %*% X 

    # new beta
    beta <- beta - solve(H) %*% g
    beta_iter <- cbind(beta_iter, beta)

    # new theta
    theta <- X %*% beta

    
    ll <- loglikelihood(y, theta)
    ll_iter <- c(ll_iter, ll)

    ## Has it converged yet?
    if (abs(ll_prev - ll) < tol) {
        converged <- TRUE
    }

    ll_prev <- ll
}

###
### Compare with glm()
###
 modelo <- glm(y ~ 0 + X, family = binomial()) 
#get coeffs
xtable::xtable(beta - modelo$coefficients, display=c('E',"E"))

###
### Plot Log-likelihood values
###

pdf("compar_newton.pdf")
plot(ll_iter, type = "l", main = "Newton's method", ylab = "Loglikelihood")
abline(h = logLik(modelo), col = "red")
legend("bottomright", legend = c("Newton's method", "glm()"), lty = 1, col = c("black", "red"))
dev.off()

## standard errors
inv_Hess <- -solve(H)
std_errors <- rep(0, p)
for (i in 1:p) {
    std_errors[i] <- sqrt(inv_Hess[i, i])
}

xtable::xtable(std_errors - summary(modelo)$coefficients[, 2])
