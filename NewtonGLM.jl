### Newton-Raphson for GLMs
using LinearAlgebra, DataFrames
import CSV

function getA(β, x)
    Diagonal([(exp(x[i, :] ⋅ β) / (1 + exp(x[i, :] ⋅ β))) * (1 - exp(x[i, :] ⋅ β) / (1 + exp(x[i, :] ⋅ β))) for i in 1:size(x, 1)])
end    


function loglikelihood(β, x)
    A = getA(β, x)
    return -x * A * x'
end

function Hessian(β, x, β0)
    y = x * β0
    A = getA(β, x)
    return (y-x*β)'* A * (y-x*β) / 2
end    

function main()
    x = Matrix(CSV.read(download("https://github.com/jgscott/SDS383D/raw/master/data/wdbc.csv"), DataFrame, header=false))[:,3:12]
    x = convert(Array{BigFloat}, x)

    β0 = ones(size(x, 2))
    betas = ones(size(x,1), 10000)
    for i in 2:10000
        betas[:,i] = betas[:,i-1] - loglikelihood(betas[:, i-1], x) * inv(Hessian(betas[:, i-1], x, β0))
    end  
    
    return betas
end
main()