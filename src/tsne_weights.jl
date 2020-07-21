using Distances, LinearAlgebra, Statistics, MultivariateStats, StatsBase, Random
"""
File info:
Provides various ways to transform the similarity weights.
Both exponential and power transforms are available in parameterized form.
Also includes a numerical solver for the exponential transform.

Variables:
X:= The similarity vector to be used
p:= The paramatrical value to transform with (default: NaN)
t:= The treshold value to transform with (default: NaN)
info:= Boolean indicating whether p and t should be returned (default: false)

"""

#Applies a power transform on the similarity weights
function power_transform(X:: AbstractArray, p::Number = NaN, t::Number = NaN, info::Bool = false)
    f(X,p) = X.^p
    if isnan(p)
        if !isnan(t)
            p = log(t)/log(minimum(X)/maximum(X))
            X = X/maximum(X)
        else
            p = log(0.05)/(log(minimum(X))-log(maximum(X)))
            X = X/maximum(X)
        end
    else
        X = X/maximum(X)
    end

    if info == false
        return f(X,p)
    else
        return f(X,p), p
    end
end

#Applies an exponential transform on the similarity weights
function exp_transform(X:: AbstractArray, p::Number = NaN, t::Number = NaN, info::Bool = false)
    f(X,p) = (p .^X .-1)/(p-1)
    if isnan(p)
        if !isnan(t)
            p = exp_transform_newton(minimum(X)/maximum(X),t)
            X = X/maximum(X)
        else
            p = exp_transform_newton(minimum(X)/maximum(X),0.05)
            X = X/maximum(X)
        end
    else
        X = X/maximum(X)
    end

    if info == false
        return f(X,p)
    else
        return f(X,p), p
    end
end

#Finds the value of p if it is not given in exp_transform
function exp_transform_newton(x::Number, t::Number = 0.05, info::Bool=false, cap::Int=10000, eps::Number=1e-6; iter::Int = 0)
    f(x,p,t) = (p^x -1)/(p-1) - t
    df(x,p) = (p^x*(x*(1-(1/p))-1))/(p-1)^2
    init_p = t^(1/(x-1))

    old = 0
    new = init_p

    while norm(old - new)>eps
        num = f(x, new, t)
        denom = df(x,new)

        old = new
        new = old - (num/denom)
        iter += 1

        if iter >= cap
            if new > 0.95 && new <1.05
                println("Treshold too close to p, this limit equals the treshold value.")
                return 1
                #return 1, t, new
                break
            else
                println("Warning: o root found at $iter iterations, using intitial guess instead.")
                return init_p
                #return init_p, f(x,init_p,t), new
                break
            end
        end
    end

    if info == false
        return new
    else
        return new, f(x,new,t),init_p, iter
    end
end
