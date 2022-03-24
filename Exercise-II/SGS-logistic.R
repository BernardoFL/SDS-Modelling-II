#### SGD

data <- read.csv('https://github.com/jgscott/SDS383D/raw/master/data/wdbc.csv', header = F)

X <- as.matrix(cbind(rep(1, 569), data[,3:12]))
Y <- as.numeric(data$V2 == "M")

## Llink funciton. If x is too big the ratio is basically 1.
link <- function(x){
  if(is.infinite(exp(x)))
    return(1)
  else
  return(exp(x)/(1+exp(x)))
}

score <- function(beta){
 sapply(1:569, \(i) c(Y[i] - link( t(X[i,]) %*% beta ) ) * X[i,]) |> rowSums()
}


theta <- rep(1,11)

converged <- F; i <- 1
while(!converged){
    theta <- theta + 1/sqrt(sqrt(sqrt(i)))*score(theta)
  
    i <- i+1
    if(norm(matrix(score(theta))) < 1e-6 || i == 1000000)
      converged <- TRUE
}

