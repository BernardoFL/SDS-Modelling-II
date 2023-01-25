install.packages('aire.zmvm')
 

local_periodic <- function(xi,xj) {
    sigma2 * exp(-2*sin(pi*abs(xi-xj)/p)^2/l2 ) * exp(-(xi-xj)^2/(2*l2)) * a*(xi-c)*(xj-c)
}