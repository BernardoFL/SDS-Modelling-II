#######
library(matlib); library(microbenchmark)

prueba  <- function(n, p){
  X <- matrix(rnorm(n*p,0,10), n, p)
  Y <- rnorm(n)
  list(X, Y)
}

solve_inv <- function(X, Y){
  inv(t(X) %*% X) %*% t(X) %*% Y
}

solve_lu <- function(X, Y){
  lu <- LU(t(X) %*% X)
  
  backsolve(lu$U, forwardsolve(lu$L, t(X) %*% Y ))
}


# n = 1000, p = 200

test <- prueba(1000, 200)
microbenchmark(solve_inv(test[[1]], test[[2]]), solve_lu(test[[1]], test[[2]]), times=20)
